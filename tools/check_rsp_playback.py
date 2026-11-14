"""Real-scene playback checks; all readbacks here are outside performance runs."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", default="stage5_checks")
    parser.add_argument("--frames", type=int, default=900)
    parser.add_argument("--check-every", type=int, default=300, help="Diagnostic interval in control frames")
    parser.add_argument("--reference", default="output/rsp_render_profile/stage5_A/source/mimickit/envs/pbf_play_env.py")
    args = parser.parse_args()
    output = ROOT / "output/rsp_render_profile" / args.label
    output.mkdir(parents=True, exist_ok=True)
    cache = ROOT / "output/newton_pbf_refactor_checks"
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cache / "tmp")
    import numpy as np
    import torch
    import warp as wp
    import yaml
    from PIL import Image
    from engines import newton_pbf_engine
    from engines.newton_pbf_runtime import sample_sdf
    from engines.newton_pbf_viewer import PBFViewer
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import resource_util, mp_util
    from util.util import set_rand_seed
    from RSP.runtime.playback import PlaybackRunner
    wp.config.kernel_cache_dir = str(cache / "warp_cache")

    class CheckViewer(PBFViewer):
        def __init__(self, **kwargs):
            kwargs.update(width=1280, height=720, vsync=False)
            super().__init__(**kwargs)
            self.renderer.window.set_visible(False)

    set_rand_seed(1234)
    mp_util.init(0, 1, "cuda:0", None)
    env = agent = None
    result = {"checks": [], "stability": []}
    try:
        ec = yaml.safe_load((ROOT / "data/envs/pbf_play_env.yaml").read_text())
        bc = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text())
        with patch.object(newton_pbf_engine, "PBFViewer", CheckViewer):
            env = resource_util.construct(PBFPlayEnv, ec, bc, 1, "cuda:0", True)
        checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
        agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
        agent.load(str(checkpoint / "model.pt"))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        runner = PlaybackRunner(env, agent)
        runner.reset()
        eng, viewer = env._engine, runner.viewer
        clock = eng._runtime._solver.clock
        assert eng.get_sim_model().particle_count == 301392
        assert clock.schedule.calls == (0, 1, 0, 1)
        assert abs(clock.schedule.control_dt - 1/30) < 1e-12
        assert abs(clock.schedule.fluid_dt - 1/300) < 1e-12
        rigid_steps = clock.schedule.rigid_steps
        assert "disc_obs" not in env._info
        result["checks"].append("new defaults and training-info contract")

        spec = importlib.util.spec_from_file_location("rsp_pre_trim_environment", ROOT / args.reference)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference = object.__new__(reference_module.PBFPlayEnv)
        reference.__dict__.update(env.__dict__)
        max_obs_error = max_action_error = max_normalized_error = max_target_error = 0.
        with torch.no_grad():
            for _ in range(4):
                for use_sphere in (False, True):
                    env.policy_use_scene_sphere = reference.policy_use_scene_sphere = use_sphere
                    old = reference._compute_obs()
                    env._update_observations()
                    new = env._obs_buf
                    torch.testing.assert_close(new, old, rtol=1e-5, atol=1e-6)
                    old_action = agent._decide_action(old, {})[0]
                    new_action = runner.policy.act(new)
                    torch.testing.assert_close(new_action, old_action, rtol=1e-5, atol=1e-6)
                    old_norm, new_norm = agent._obs_norm.normalize(old), agent._obs_norm.normalize(new)
                    torch.testing.assert_close(new_norm, old_norm, rtol=1e-5, atol=1e-6)
                    old_target, new_target = env._clip_action(old_action), env._clip_action(new_action)
                    torch.testing.assert_close(new_target, old_target, rtol=1e-5, atol=1e-6)
                    max_obs_error = max(max_obs_error, float((new - old).abs().max()))
                    max_action_error = max(max_action_error, float((new_action - old_action).abs().max()))
                    max_normalized_error = max(max_normalized_error, float((new_norm - old_norm).abs().max()))
                    max_target_error = max(max_target_error, float((new_target - old_target).abs().max()))
                env.set_policy_use_scene_sphere(False)
                runner.tick()
            result["equivalence"] = dict(states=4, sphere_modes=2, max_obs_error=max_obs_error,
                                         max_action_error=max_action_error, max_normalized_error=max_normalized_error,
                                         max_target_error=max_target_error)
            reference.policy_use_scene_sphere = False

            # Graph and ordinary execution must update the persistent views.
            # Do not compare two
            # independently diverging physics trajectories for bitwise equality.
            runtime = eng._runtime
            graph = runtime._graph
            for use_graph in (True, False):
                runtime._graph = graph if use_graph else None
                runner.reset()
                for frame in range(1, 3):
                    runner.tick()
                    assert clock.rigid_ticks == frame * rigid_steps and clock.fluid_ticks == frame * 2
                    assert abs(eng.get_sim_time() - frame / 30) < 1e-12
                    assert abs(clock.time - frame / 150) < 1e-12
                    if rigid_steps % 2:
                        for name in ("joint_q", "joint_qd", "body_q", "particle_q", "particle_qd"):
                            torch.testing.assert_close(wp.to_torch(getattr(eng._state.raw_state, name)),
                                wp.to_torch(getattr(eng._state.swap_state, name)), rtol=0, atol=0)
                    old = reference._compute_obs()
                    torch.testing.assert_close(env._obs_buf, old, rtol=1e-5, atol=1e-6)
            runtime._graph = graph
            result["checks"].append(f"{rigid_steps} rigid/two fluid steps: graph/eager clocks and persistent state views")

            # Audit ordinary frames, not loading, resetting, picking or screenshots.
            with patch.object(wp.array, "numpy", side_effect=AssertionError("Warp host readback")), \
                    patch.object(torch.Tensor, "cpu", side_effect=AssertionError("Torch CPU transfer")), \
                    patch.object(torch.Tensor, "item", side_effect=AssertionError("Scalar readback")), \
                    patch.object(env, "_update_disc_hist", side_effect=AssertionError("Training history")), \
                    patch.object(env, "_update_disc_obs", side_effect=AssertionError("Training features")), \
                    patch.object(env, "_compute_obs", side_effect=AssertionError("Legacy observation path")), \
                    patch.object(agent, "_decide_action", side_effect=AssertionError("Legacy policy path")):
                for _ in range(5):
                    runner.tick()
            result["checks"].append("five ordinary frames without explicit state readback or training features")
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                    torch.profiler.ProfilerActivity.CUDA]) as trace:
                for _ in range(2):
                    runner.tick()
            trace.export_chrome_trace(str(output / "playback_trace.json"))
            events = json.loads((output / "playback_trace.json").read_text())["traceEvents"]
            downloads = [e for e in events if "dtoh" in e.get("name", "").lower()
                         or "device to host" in e.get("name", "").lower()]
            result["profile_dtoh_events"] = len(downloads)
            result["profile_stream_syncs_per_frame"] = sum(e.get("name") == "cudaStreamSynchronize" for e in events) / 2
            assert not downloads, "Unexpected D2H during ordinary playback"

            # Exercise the actual UI widgets, including deferred reset semantics.
            imgui = viewer.ui.imgui
            original_checkbox, original_button = imgui.checkbox, imgui.button
            def pause(label, value):
                return (True, True) if label == "Pause" else original_checkbox(label, value)
            with patch.object(imgui, "checkbox", pause):
                viewer.end_frame()
            assert viewer.is_paused()
            ticks = clock.rigid_ticks
            assert not runner.tick() and clock.rigid_ticks == ticks
            with patch.object(imgui, "button", lambda label, *a, **k: True if label == "Step" else original_button(label, *a, **k)):
                viewer.end_frame()
            assert runner.tick() and clock.rigid_ticks == ticks + rigid_steps
            assert not runner.tick() and clock.rigid_ticks == ticks + rigid_steps
            with patch.object(imgui, "button", lambda label, *a, **k: True if label == "Reset" else original_button(label, *a, **k)):
                viewer.end_frame()
            assert clock.rigid_ticks == ticks + rigid_steps and env._pending_reset
            assert not runner.tick()
            assert clock.rigid_ticks == clock.fluid_ticks == 0
            assert not np.any(eng._runtime._solver.boundary_forces.numpy())
            result["checks"].append("UI pause, exact one-step, deferred reset and cleared feedback")

            from pyglet.window import key
            viewer.renderer._on_key_press(key.SPACE, 0)
            assert runner.tick() and clock.rigid_ticks == rigid_steps and clock.fluid_ticks == 2
            assert not runner.tick()
            viewer.renderer._on_key_press(key.R, 0)
            assert not runner.tick() and clock.rigid_ticks == 0
            viewer.renderer._on_key_press(key.ENTER, 0)
            assert not viewer.is_paused() and runner.tick()
            result["checks"].append("Enter/Space/R have one control owner")

            # One uninterrupted trajectory; no unchanged renderer/recording tests.
            set_rand_seed(1234)
            runner.reset()
            raw = eng._state.raw_state
            sdf_distances = wp.empty(eng.get_sim_model().particle_count, dtype=float, device="cuda:0")
            scene = eng.get_play_scene()
            for index in range(args.frames):
                assert runner.tick()
                if (index + 1) % args.check_every == 0 or index == args.frames - 1:
                    wp.synchronize()
                    points, velocities, bodies = raw.particle_q.numpy(), raw.particle_qd.numpy(), raw.body_q.numpy()
                    forces = eng._runtime._solver.boundary_forces.numpy()
                    arrays = dict(particles=points, particle_velocity=velocities, bodies=bodies, feedback=forces,
                                  joints=raw.joint_q.numpy(), joint_velocity=raw.joint_qd.numpy(),
                                  observations=env._obs_buf.cpu().numpy())
                    nonfinite = {name: int((~np.isfinite(a)).sum()) for name, a in arrays.items()}
                    if any(nonfinite.values()):
                        result["failure"] = dict(frame=index + 1, main_time=eng.get_sim_time(),
                                                 fluid_time=clock.time, nonfinite=nonfinite)
                        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                        raise AssertionError(f"Non-finite state: {result['failure']}")
                    assert clock.rigid_ticks == (index + 1) * rigid_steps
                    assert clock.fluid_ticks == (index + 1) * 2
                    wp.launch(sample_sdf, len(points), inputs=[scene.boundaries.assets[0].sdf.id, raw.particle_q],
                              outputs=[sdf_distances], device="cuda:0")
                    distances = sdf_distances.numpy()
                    outside = np.any((points < 0) | (points > np.asarray(scene.config.container_size)), axis=1)
                    row = dict(frame=index + 1, main_time=eng.get_sim_time(), fluid_time=clock.time,
                               outside_count=int(outside.sum()), static_penetration_over_radius=int((distances < -.01).sum()),
                               min_sdf=float(distances.min()), max_speed=float(np.linalg.norm(velocities, axis=1).max()),
                               min_position=points.min(0).tolist(), max_position=points.max(0).tolist(),
                               max_force=float(np.linalg.norm(forces[:, :3], axis=1).max()),
                               max_joint_speed=float(np.abs(arrays["joint_velocity"]).max()),
                               humanoid_height=float(eng.get_root_pos(env._get_char_id())[0, 2]))
                    result["stability"].append(row)
                    if index == args.frames - 1 or index + 1 == 300:
                        viewer.end_frame()
                        Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output / f"frame_{index + 1}.png")
                    (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
                    print("STABILITY " + json.dumps(row), flush=True)
            result["checks"].append("uninterrupted trajectory: finite particles, humanoid, observations and feedback")
        (output / "report.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"PLAYBACK CHECK PASS: {output / 'report.json'}", flush=True)
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None, mp_util.shutdown)


if __name__ == "__main__":
    main()
