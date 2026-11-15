"""Short picking/layout check and GPU attribution; no recording or full regression.

Normal samples preserve playback overlap. A separate diagnostic records CUDA
events inside the simulation Graph and waits after physics to read those events.
GL elapsed queries are read after each batch. Map/unmap/present are CPU wall
intervals, including driver/wait time; they must not be added to GPU durations.
"""
import argparse
import ctypes
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]
from profile_rsp_render import statistics


class Measurements:
    def __init__(self, runner):
        self.runner = runner
        self.viewer = runner.viewer
        self.renderer = self.viewer.renderer
        self.row = None
        self.queries = []

    def hook(self, stack, obj, name, key, gpu_key=None):
        original = getattr(obj, name)
        gl = self.renderer.gl
        def measured(*args, **kwargs):
            if self.row is None:
                return original(*args, **kwargs)
            query = None
            if gpu_key:
                query = gl.GLuint()
                gl.glGenQueries(1, ctypes.byref(query))
                gl.glBeginQuery(gl.GL_TIME_ELAPSED, query)
            start = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                self.row[key] = self.row.get(key, 0.) + (time.perf_counter() - start) * 1000
                if query is not None:
                    gl.glEndQuery(gl.GL_TIME_ELAPSED)
                    self.queries.append((query, self.row, gpu_key))
        stack.enter_context(patch.object(obj, name, measured))

    def drain(self):
        import warp as wp
        wp.synchronize()
        self.renderer.gl.glFinish()

    def resolve_queries(self):
        gl = self.renderer.gl
        for query, row, key in self.queries:
            value = gl.GLuint64()
            gl.glGetQueryObjectui64v(query, gl.GL_QUERY_RESULT, ctypes.byref(value))
            row[key] = value.value / 1e6
            gl.glDeleteQueries(1, ctypes.byref(query))
        self.queries.clear()

    def run(self, warmup, frames, *, cuda_events=None):
        import warp as wp
        runner, viewer, renderer = self.runner, self.viewer, self.renderer
        env, eng = runner.env, runner.env._engine
        rows = []
        with ExitStack() as stack:
            for obj, name, key, gpu in (
                (runner.policy, "act", "policy_ms", None),
                (env, "_pre_physics_step", "action_ms", None),
                (env, "_physics_step", "physics_submit_ms", None),
                (env, "_post_physics_step", "post_ms", None),
                (eng._presentation, "render", "render_ms", None),
                (viewer, "log_state", "state_update_ms", None),
                (viewer, "_log_particles", "particle_update_ms", None),
                (renderer, "render", "scene_cpu_ms", "scene_gpu_ms"),
                (renderer, "present", "present_ms", None),
                (renderer, "update", "events_ms", None),
                (viewer.gui, "prepare_frame", "ui_build_ms", None),
                (viewer.gui, "render_prepared_frame", "ui_cpu_ms", "ui_gpu_ms"),
                (wp.RegisteredGLBuffer, "map", "interop_map_ms", None),
                (wp.RegisteredGLBuffer, "unmap", "interop_unmap_ms", None),
            ):
                self.hook(stack, obj, name, key, gpu)
            if cuda_events is not None:
                original = env._physics_step
                def physics():
                    result = original()
                    start = time.perf_counter()
                    wp.synchronize()
                    if self.row is not None:
                        self.row["physics_wait_ms"] = (time.perf_counter() - start) * 1000
                        for name, pairs in cuda_events.items():
                            self.row[name] = sum(wp.get_event_elapsed_time(a, b, synchronize=False) for a, b in pairs)
                    return result
                stack.enter_context(patch.object(env, "_physics_step", physics))
            for _ in range(warmup):
                runner.tick()
            self.drain()
            start_batch = time.perf_counter()
            for _ in range(frames):
                self.row = {}
                start = time.perf_counter()
                runner.tick()
                self.row["frame_ms"] = (time.perf_counter() - start) * 1000
                self.row["rigid_update_ms"] = self.row["state_update_ms"] - self.row["particle_update_ms"]
                self.row["interop_ms"] = self.row["interop_map_ms"] + self.row["interop_unmap_ms"]
                rows.append(self.row)
                self.row = None
            self.drain()
            completed = (time.perf_counter() - start_batch) * 1000 / frames
        self.resolve_queries()
        for row in rows:
            row["draw_gpu_ms"] = row["scene_gpu_ms"] + row["ui_gpu_ms"]
        return dict(summary={k: statistics([r.get(k, 0.) for r in rows]) for k in rows[0]},
                    completed_batch_mean_ms=completed, samples=rows)


def capture_diagnostic_graph(runner):
    import warp as wp
    runtime = runner.env._engine._runtime
    events = {"rigid_gpu_ms": [], "pbf_gpu_ms": []}
    def instrument(original, pairs):
        def step(*args, **kwargs):
            start = wp.Event("cuda:0", enable_timing=True)
            end = wp.Event("cuda:0", enable_timing=True)
            pairs.append((start, end))
            wp.record_event(start)
            result = original(*args, **kwargs)
            wp.record_event(end)
            return result
        return step
    # Capture on a Warp-owned non-default stream, as in engine initialization.
    # PyTorch's imported legacy/default stream is not a valid capture stream.
    wp.synchronize()
    with ExitStack() as stack, wp.ScopedStream(wp.Stream("cuda:0")):
        for solver, name in ((runtime._solver.rigid, "rigid_gpu_ms"), (runtime._solver.fluid, "pbf_gpu_ms")):
            stack.enter_context(patch.object(solver, "step", instrument(solver.step, events[name])))
        runtime.build_graphs()
    assert len(events["rigid_gpu_ms"]) == 4 and len(events["pbf_gpu_ms"]) == 2
    return events


def picking_check(runner):
    import numpy as np
    import warp as wp
    from pyglet.window import mouse
    from newton._src.viewer.picking import Picking
    eng, viewer = runner.env._engine, runner.viewer
    runner.env.render_only()
    runner.env.render_only()
    raw = eng._state.raw_state
    model = eng.get_sim_model()
    target = eng.get_root_pos(runner.env._get_char_id())[0].cpu().numpy() + viewer.world_offsets.numpy()[0]
    camera = viewer.camera
    delta = target - camera.pos
    depth = np.dot(delta, camera.get_front())
    alpha = np.tan(np.radians(camera.fov) / 2)
    x = (1 + np.dot(delta, camera.get_right()) / (depth * alpha * camera.width / camera.height)) * camera.width / 2
    y = (1 + np.dot(delta, camera.get_up()) / (depth * alpha)) * camera.height / 2
    x, y = float(x), float(y)
    ray = camera.get_world_ray(x, y)
    native = Picking(model, world_offsets=viewer.world_offsets)
    native.pick(raw, *ray)
    old_shape = int(native.min_index.numpy()[0])
    old_body = int(native.pick_body.numpy()[0])
    blocked_by = model.shape_label[old_shape] if old_shape >= 0 else "miss"
    width, height = viewer.renderer.window.get_size()
    wx, wy = x * width / camera.width, y * height / camera.height
    viewer.on_mouse_press(wx, wy, mouse.RIGHT, 0)
    body = int(viewer.picking.pick_body.numpy()[0])
    assert body in eng.get_obj_body_ids(runner.env._get_char_id()), (body, (wx, wy), blocked_by)
    before = viewer.picking.pick_state.numpy()[0]["picking_target_world"].copy()
    viewer.on_mouse_drag(wx + 35, wy + 30, 35, 30, mouse.RIGHT, 0)
    after = viewer.picking.pick_state.numpy()[0]["picking_target_world"].copy()
    assert np.linalg.norm(after - before) > .01
    probe = model.state()
    wp.copy(probe.body_q, raw.body_q)
    wp.copy(probe.body_qd, raw.body_qd)
    probe.clear_forces()
    viewer.apply_forces(probe)
    force = probe.body_f.numpy()[body]
    assert np.linalg.norm(force) > 1e-3
    runner.tick()  # The production CUDA Graph must also contain the pick kernel.
    graph_force = eng._state.raw_state.body_f.numpy()[body]
    assert np.linalg.norm(graph_force) > 1e-3
    viewer.on_mouse_release(wx + 35, wy + 30, mouse.RIGHT, 0)
    assert not viewer.picking.is_picking()
    probe.clear_forces()
    viewer.apply_forces(probe)
    assert not np.any(probe.body_f.numpy())
    return dict(native_body=old_body, native_hit=blocked_by, picked_body=body,
                target_displacement=float(np.linalg.norm(after - before)),
                eager_force_norm=float(np.linalg.norm(force)), graph_force_norm=float(np.linalg.norm(graph_force)),
                released=True)


def inspect_state(runner):
    import numpy as np
    import warp as wp
    from engines.newton_pbf_runtime import sample_sdf
    eng, env = runner.env._engine, runner.env
    state, scene = eng._state.raw_state, eng.get_play_scene()
    positions = state.particle_q.numpy()
    velocities = state.particle_qd.numpy()
    sdf = wp.empty(len(positions), dtype=float, device=state.particle_q.device)
    wp.launch(sample_sdf, len(positions), inputs=[scene.boundaries.assets[0].sdf.id, state.particle_q], outputs=[sdf])
    distances = sdf.numpy()
    arrays = [positions, velocities, state.body_q.numpy(), state.joint_qd.numpy(),
              eng._runtime._solver.boundary_forces.numpy(), env._obs_buf.cpu().numpy()]
    assert all(np.isfinite(a).all() for a in arrays), "Non-finite scene state"
    root = eng.get_root_pos(env._get_char_id())[0].cpu().numpy()
    return dict(time=eng.get_sim_time(), fluid_time=eng._runtime._solver.clock.time,
                root=root.tolist(), min_position=positions.min(0).tolist(), max_position=positions.max(0).tolist(),
                inside_static=int((distances < 0).sum()), penetration_over_radius=int((distances < -.01).sum()),
                outside=int(np.any((positions < 0) | (positions > scene.config.container_size), axis=1).sum()),
                max_speed=float(np.linalg.norm(velocities, axis=1).max()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="stage7")
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stability-frames", type=int, default=600)
    parser.add_argument("--gpu-only", action="store_true", help="Reuse this label's completed scene/normal checks")
    args = parser.parse_args()
    output = ROOT / "output/rsp_render_profile" / args.label
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "output/newton_pbf_refactor_checks"
    tempfile.tempdir = str(cache / "tmp")
    import numpy as np
    import torch
    import warp as wp
    import yaml
    from PIL import Image
    wp.config.kernel_cache_dir = str(cache / "warp_cache")
    from engines import newton_pbf_engine
    from engines.newton_pbf_viewer import PBFViewer
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import resource_util, mp_util
    from util.util import set_rand_seed
    from RSP.runtime.playback import PlaybackRunner

    class Viewer(PBFViewer):
        def __init__(self, **kwargs):
            kwargs.update(width=1280, height=720, vsync=False)
            super().__init__(**kwargs)
            if args.hidden:
                self.renderer.window.set_visible(False)

    set_rand_seed(1234)
    mp_util.init(0, 1, "cuda:0", None)
    env = agent = None
    report = (json.loads((output / "report.json").read_text()) if args.gpu_only
              else {"args": vars(args), "seed": 1234})
    if args.gpu_only:
        report["diagnostic_args"] = vars(args)
    try:
        ec = yaml.safe_load((ROOT / "data/envs/pbf_play_env.yaml").read_text())
        bc = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text())
        with patch.object(newton_pbf_engine, "PBFViewer", Viewer):
            env = resource_util.construct(PBFPlayEnv, ec, bc, 1, "cuda:0", True)
        checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
        agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
        agent.load(str(checkpoint / "model.pt"))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        runner = PlaybackRunner(env, agent)
        eng, viewer = env._engine, runner.viewer
        def reset():
            set_rand_seed(1234)
            viewer.picking.release()
            runner.reset()
        def save_image(name):
            env.render_only()
            viewer.end_frame()
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output / f"{name}.png")
        with torch.no_grad():
            reset()
            schedule = eng._runtime._solver.clock.schedule
            assert schedule.calls == (0, 1, 0, 1) and runner.policy_interval == 5
            assert abs(eng.get_sim_timestep() - 1 / 600) < 1e-12
            assert abs(schedule.fluid_dt - 1 / 300) < 1e-12
            assert eng.get_sim_model().particle_count == 300105
            root_height = eng.get_root_pos(env._get_char_id())[0, 2]
            torch.testing.assert_close(root_height, env._ref_root_pos[0, 2])
            report.update(particles=eng.get_sim_model().particle_count, schedule=asdict(schedule),
                          config=asdict(eng.get_play_scene().config), msaa=viewer.renderer.msaa_samples,
                          shadows=viewer.renderer.draw_shadows, resolution=list(viewer.renderer.window.get_framebuffer_size()))
            measurements = Measurements(runner)
            if not args.gpu_only:
                report["initial"] = inspect_state(runner)
                save_image("initial")
                report["picking"] = picking_check(runner)
                print("PICKING", json.dumps(report["picking"]), flush=True)
                reset()
                viewer._paused = True
                assert not runner.tick() and eng.get_sim_time() == 0.
                viewer._step_requested = True
                assert runner.tick() and abs(eng.get_sim_time() - 1 / 150) < 1e-12
                env.request_reset()
                assert not runner.tick() and eng.get_sim_time() == 0.
                viewer._paused = False
                report["pause_step_reset"] = True
                report["stability"] = []
                for frame in range(1, args.stability_frames + 1):
                    runner.tick()
                    if frame % 150 == 0 or frame == args.stability_frames:
                        report["stability"].append(inspect_state(runner))
                        print("STABILITY", json.dumps(report["stability"][-1]), flush=True)
                save_image("settled")
                reset()
                report["normal"] = measurements.run(30, args.frames)
                print("NORMAL", json.dumps(report["normal"]["summary"]), flush=True)
            reset()
            events = capture_diagnostic_graph(runner)
            reset()
            report["synchronized_diagnostic"] = measurements.run(10, 30, cuda_events=events)
            print("GPU", json.dumps(report["synchronized_diagnostic"]["summary"]), flush=True)
            report["final"] = inspect_state(runner)
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        resource_util.cleanup(env.close if env else None, agent.close if agent else None, mp_util.shutdown)
    print(f"SCENE DIAGNOSTIC PASS: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
