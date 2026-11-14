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
    from engines import engine, newton_pbf_engine
    from engines.newton_pbf_viewer import PBFViewer
    from envs.base_env import EnvMode
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util
    from util.util import set_rand_seed

    class ProfileViewer(PBFViewer):
        def __init__(self, **kwargs):
            kwargs.update(width=args.width, height=args.height, vsync=False)
            super().__init__(**kwargs)
            if args.hidden:
                self.renderer.window.set_visible(False)

    with (ROOT / "data/engines/newton_pbf_engine.yaml").open() as f:
        engine_config = yaml.safe_load(f)
    with (ROOT / "data/envs/pbf_play_env.yaml").open() as f:
        env_config = yaml.safe_load(f)
    checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
    # Archive the small entry point so the baseline's source is unambiguous.
    source = (ROOT / "mimickit/engines/newton_pbf_viewer.py").read_bytes()
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
        counters = {}
        normal_limit = engine.Engine.render
        normal_render = eng._presentation.render
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

        def drain():
            viewer.renderer._make_current()
            wp.synchronize()
            gl.glFinish()

        with patch.object(engine.Engine, "render", rate_limit), \
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
                    counters.update(render_ms=0.0, limit_ms=0.0)
                    start = time.perf_counter()
                    action, _ = agent._decide_action(obs, info)
                    obs, _, _, info = env.step(action)
                    elapsed = (time.perf_counter() - start) * 1000
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
        print(f"PROFILE PASS: {output / 'report.json'}", flush=True)
    finally:
        resource_util.cleanup(env.close if env is not None else None,
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
    args = parser.parse_args()
    if args.warmup < 1 or args.frames < 1 or min(args.width, args.height) < 1:
        parser.error("warmup, frames, width and height must be positive")
    run(args)


if __name__ == "__main__":
    main()
