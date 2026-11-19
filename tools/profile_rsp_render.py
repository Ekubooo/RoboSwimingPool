"""Measure policy + physics + Newton scene/UI rendering in the PBF play scene.

Run before and after a renderer change with the same arguments, for example:
  python tools/profile_rsp_render.py --label cubes --hidden
  python tools/profile_rsp_render.py --label billboard --hidden

The normal ViewerGL/UI path is used even with --hidden (which hides the window,
not the UI). Only this process bypasses Engine's rate limiter for the uncapped
sample. CPU call timings are not GPU kernel timings. GPU work is drained at the
measurement boundaries, never between individual measured frames.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))


def statistics(values):
    import numpy as np

    return {"mean": float(np.mean(values)), "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95))}


def run(args):
    from contextlib import ExitStack
    playback_stack = ExitStack()
    from RSP.app.playback import bind_playback
    from RSP.integrations.mimickit.session import MimicKitSession
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

    wp.config.kernel_cache_dir = str(cache / "warp_cache")
    import newton
    from engines import engine
    from RSP.integrations.mimickit import recorder as newton_pbf_engine
    from engines.newton_pbf_viewer import PBFViewer
    from envs.base_env import EnvMode
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util
    from util.util import set_rand_seed
    from RSP.physics import fluid_clock

    if args.fluid_profile == "legacy":
        base = fluid_clock.DEFAULT_SETTINGS
        fluid_clock.DEFAULT_SETTINGS = replace(base, time_scale=1., max_dt=1/240,
                                               pbf=replace(base.pbf, substeps=8))
    elif args.fluid_profile == "two":
        base = fluid_clock.DEFAULT_SETTINGS
        fluid_clock.DEFAULT_SETTINGS = replace(base, pbf=replace(base.pbf, substeps=2))
    elif args.fluid_profile == "paused":
        fluid_clock.DEFAULT_SETTINGS = replace(fluid_clock.DEFAULT_SETTINGS, time_scale=0.)
    elif args.fluid_profile == "fast":
        fluid_clock.DEFAULT_SETTINGS = replace(fluid_clock.DEFAULT_SETTINGS, time_scale=1.)

    class ProfileViewer(PBFViewer):
        def __init__(self, **kwargs):
            self.profile_state_ms = 0.0
            self.profile_particles_ms = 0.0
            kwargs.update(width=args.width, height=args.height, vsync=False)
            super().__init__(**kwargs)
            if args.hidden:
                self.renderer.window.set_visible(False)

        def log_state(self, state):
            start = time.perf_counter()
            try:
                return super().log_state(state)
            finally:
                self.profile_state_ms += (time.perf_counter() - start) * 1000

        def _log_particles(self, state):
            start = time.perf_counter()
            try:
                return super()._log_particles(state)
            finally:
                self.profile_particles_ms += (time.perf_counter() - start) * 1000

    with (ROOT / "data/engines/newton_pbf_engine.yaml").open() as f:
        engine_config = yaml.safe_load(f)
    if args.eager:
        engine_config["enable_cuda_graph"] = False
    with (ROOT / "data/envs/pbf_play_env.yaml").open() as f:
        env_config = yaml.safe_load(f)
    checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
    # Archive the actual Viewer implementation for baseline comparisons.
    source = (ROOT / "RSP/rendering/newton_gl.py").read_bytes()
    (output / "viewer_source.py").write_bytes(source)
    set_rand_seed(args.seed)
    mp_util.init(0, 1, "cuda:0", None)
    env = agent = None
    try:
        with patch.object(newton_pbf_engine, "PBFViewer", ProfileViewer):
            env = resource_util.construct(PBFPlayEnv, env_config, engine_config, 1, "cuda:0", True)
        env.set_mode(EnvMode.TEST)
        agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
        agent.load(str(checkpoint / "model.pt"))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        playback_stack.enter_context(bind_playback(MimicKitSession(env, agent)))
        eng = env._engine
        viewer = eng.get_viewer()
        assert viewer.ui is not None and viewer.ui.is_available, "Profile must include the actual UI"
        gl = viewer.renderer.gl
        model = eng.get_sim_model()
        report = {
            "label": args.label, "seed": args.seed, "warmup": args.warmup, "frames": args.frames,
            "window_hidden": args.hidden, "ui_enabled": viewer.show_ui, "vsync": viewer.vsync,
            "framebuffer_size": list(viewer.renderer.window.get_framebuffer_size()),
            "particle_count": model.particle_count, "body_count": model.body_count,
            "engine_config": engine_config, "env_config": env_config,
            "fluid_settings": asdict(eng.get_play_scene().fluid_settings),
            "fluid_schedule": asdict(eng.get_play_scene().fluid_schedule),
            "checkpoint": str(checkpoint / "model.pt"),
            "versions": {"newton": newton.__version__, "warp": wp.__version__, "torch": torch.__version__},
            "device": wp.get_device("cuda:0").name,
            "msaa_samples": viewer.renderer.msaa_samples, "shadows": viewer.renderer.draw_shadows,
            "viewer_source_sha256": hashlib.sha256(source).hexdigest(),
            "timing_note": "CPU wall ms; render includes state upload, scene, UI and present. "
                           "Frame includes policy and simulation. Rate-limit call is separately measured. "
                           "GPU drain occurs only at batch boundaries.",
            "runs": {},
        }
        report["source_hashes"] = {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for directory in ("app", "integrations", "physics", "rendering", "simulation")
            for path in sorted((ROOT / "RSP" / directory).rglob("*.py"))}
        counters = {}
        normal_limit = engine.Engine.render
        normal_render = eng._presentation.render
        normal_poll = viewer.renderer.update
        paced = True

        def rate_limit(instance):
            start = time.perf_counter()
            if paced:
                normal_limit(instance)
            counters["limit_ms"] += (time.perf_counter() - start) * 1000

        def render(*positional, **kwargs):
            start = time.perf_counter()
            result = normal_render(*positional, **kwargs)
            counters["render_ms"] += (time.perf_counter() - start) * 1000
            return result

        def poll_events():
            start = time.perf_counter()
            try:
                return normal_poll()
            finally:
                counters["event_poll_ms"] += (time.perf_counter() - start) * 1000

        def drain():
            viewer.renderer._make_current()
            wp.synchronize()
            gl.glFinish()

        with patch.object(engine.Engine, "render", rate_limit), \
                patch.object(viewer.renderer, "update", poll_events), \
                patch.object(eng._presentation, "render", render), torch.no_grad():
            for mode in ("paced", "uncapped"):
                paced = mode == "paced"
                set_rand_seed(args.seed)
                obs, info = env.reset()
                samples = []
                for index in range(args.warmup + args.frames):
                    if index == args.warmup:
                        drain()
                        batch_start = time.perf_counter()
                    counters.update(render_ms=0.0, limit_ms=0.0, event_poll_ms=0.0)
                    viewer.profile_state_ms = viewer.profile_particles_ms = 0.0
                    start = time.perf_counter()
                    action, _ = agent._decide_action(obs, info)
                    obs, _, _, info = env.step(action)
                    elapsed = (time.perf_counter() - start) * 1000
                    counters["rigid_update_ms"] = viewer.profile_state_ms - viewer.profile_particles_ms
                    if index >= args.warmup:
                        samples.append({"frame_ms": elapsed, **counters})
                drain()
                completed_mean = (time.perf_counter() - batch_start) * 1000 / args.frames
                assert torch.isfinite(obs).all(), "Non-finite observations"
                assert all(np.isfinite(list(sample.values())).all() for sample in samples)
                summary = {key: statistics([sample[key] for sample in samples]) for key in samples[0]}
                summary["frame_without_limit_ms"] = statistics(
                    [sample["frame_ms"] - sample["limit_ms"] for sample in samples])
                summary["completed_batch_mean_ms"] = completed_mean
                report["runs"][mode] = {"summary": summary, "samples": samples}
                # Screenshot/readback is outside the measured interval.
                viewer.end_frame()
                frame = viewer.get_frame(render_ui=True).numpy()
                Image.fromarray(frame).save(output / f"{mode}.png")
                print(json.dumps({"mode": mode, **summary}), flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.stability_frames:
                paced = False
                set_rand_seed(args.seed)
                obs, info = env.reset()
                checkpoints = []
                for index in range(args.stability_frames):
                    # One explicit reset tests both the cached feedback and graph
                    # parity after continuous playback, in addition to startup.
                    if args.reset_at and index == args.reset_at:
                        obs, info = env.reset()
                        assert eng._runtime._solver.clock.time == 0.
                    action, _ = agent._decide_action(obs, info)
                    obs, _, _, info = env.step(action)
                    if index % 300 == 0 or index == args.stability_frames - 1:
                        drain()
                        raw = eng.get_sim_state().raw_state
                        points, velocity, bodies = raw.particle_q.numpy(), raw.particle_qd.numpy(), raw.body_q.numpy()
                        forces = eng._runtime._solver.boundary_forces.numpy()
                        assert all(np.isfinite(x).all() for x in (points, velocity, bodies, forces))
                        assert torch.isfinite(obs).all()
                        clock = eng._runtime._solver.clock
                        expected_steps = round(eng.get_sim_time() / eng.get_timestep())
                        assert clock.rigid_ticks == expected_steps * eng._runtime._sim_steps
                        assert clock.fluid_ticks == expected_steps * clock.schedule.fluid_steps
                        checkpoints.append(dict(frame=index + 1, main_time=eng.get_sim_time(),
                            fluid_time=clock.time, particle_min=points.min(axis=0).tolist(),
                            particle_max=points.max(axis=0).tolist(),
                            max_speed=float(np.linalg.norm(velocity, axis=1).max()),
                            max_force=float(np.linalg.norm(forces[:, :3], axis=1).max())))
                        Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output / f"stability_{index + 1}.png")
                report["stability"] = {"frames": args.stability_frames, "checkpoints": checkpoints}
                raw = eng.get_sim_state().raw_state
                np.savez(output / "final_state.npz", particles=raw.particle_q.numpy(),
                         velocities=raw.particle_qd.numpy(), bodies=raw.body_q.numpy(),
                         joints=raw.joint_q.numpy(), observations=obs.cpu().numpy())
                (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"PROFILE PASS: {output / 'report.json'}", flush=True)
    finally:
        resource_util.cleanup(playback_stack.close, env.close if env is not None else None,
                              agent.close if agent is not None else None, mp_util.shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--hidden", action="store_true")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--fluid-profile", choices=("default", "legacy", "two", "paused", "fast"), default="default")
    parser.add_argument("--eager", action="store_true", help="Disable CUDA Graph for equivalence checks")
    parser.add_argument("--stability-frames", type=int, default=0)
    parser.add_argument("--reset-at", type=int, default=0)
    args = parser.parse_args()
    if args.warmup < 1 or args.frames < 1 or min(args.width, args.height) < 1:
        parser.error("warmup, frames, width and height must be positive")
    run(args)


if __name__ == "__main__":
    main()
