"""Profile the actual old test loop or RSP playback loop, including scene/UI.

Use separate processes for every repeat. The legacy mode calls MimicKit's
unmodified _rollout_test; a sentinel stops it at a policy-frame boundary.
GPU synchronization and screenshots are outside samples, except in the
explicit --separate-wait attribution run (never use that run for throughput).
"""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]
from profile_rsp_render import statistics


class EndSamples(Exception):
    pass


class FrameProfiler:
    MAIN = ("policy_ms", "action_ms", "physics_submit_ms", "physics_wait_ms",
            "post_ms", "render_ms", "limit_ms", "evaluation_ms")

    def __init__(self, args, drain):
        self.args, self.drain = args, drain
        self.index = 0
        self.samples = []
        self.current = None
        self.started = None
        self.batch_start = None

    def add(self, key, value):
        if self.current is not None:
            self.current[key] = self.current.get(key, 0.) + value

    def hook(self, stack, obj, method, key):
        original = getattr(obj, method)
        def measured(*a, **kw):
            start = time.perf_counter()
            try:
                return original(*a, **kw)
            finally:
                self.add(key, (time.perf_counter() - start) * 1000)
                if key == "policy_ms":
                    self.add("policy_calls", 1)
        stack.enter_context(patch.object(obj, method, measured))

    def next_frame(self):
        now = time.perf_counter()
        if self.current is not None:
            self.current["frame_ms"] = (now - self.started) * 1000
            self.current["other_ms"] = self.current["frame_ms"] - sum(self.current[k] for k in self.MAIN)
            self.current["rigid_update_ms"] = self.current.get("state_update_ms", 0) - self.current.get("particle_update_ms", 0)
            if self.index > self.args.warmup:
                self.samples.append(self.current)
        self.current = None
        if self.index == self.args.warmup + self.args.frames:
            raise EndSamples
        if self.index == self.args.warmup:
            self.drain()
            self.batch_start = time.perf_counter()
        self.current = {key: 0. for key in self.MAIN}
        self.started = time.perf_counter()
        self.index += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True)
    parser.add_argument("--loop", choices=("legacy", "playback"), default="playback")
    parser.add_argument("--fluid", choices=("old", "default"), default="default")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--separate-wait", action="store_true")
    parser.add_argument("--rigid-substeps", type=int, help="Process-local override; production config is unchanged")
    parser.add_argument("--modes", nargs="+", choices=("uncapped",), default=("uncapped",),
                        help="Production playback is uncapped; retained for existing profiler commands")
    parser.add_argument("--screenshot", action="store_true")
    args = parser.parse_args()
    cache = ROOT / "output/newton_pbf_refactor_checks"
    (cache / "tmp").mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(cache / "tmp")
    import numpy as np
    import torch
    import warp as wp
    import yaml
    from PIL import Image
    import newton
    from engines import newton_pbf_engine
    from engines.newton_pbf_viewer import PBFViewer
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import resource_util, mp_util
    from util.util import set_rand_seed
    from RSP.runtime import fluid_clock

    output = ROOT / "output/rsp_render_profile" / args.label
    output.mkdir(parents=True, exist_ok=True)
    wp.config.kernel_cache_dir = str(cache / "warp_cache")
    if args.fluid == "old":
        fluid_clock.DEFAULT_SETTINGS = fluid_clock.FluidSettings()

    class ProfileViewer(PBFViewer):
        def __init__(self, **kwargs):
            kwargs.update(width=1280, height=720, vsync=False)
            super().__init__(**kwargs)
            self.renderer.window.set_visible(False)

    engine_config = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text())
    if args.rigid_substeps is not None:
        if args.rigid_substeps < 1:
            parser.error("--rigid-substeps must be positive")
        engine_config["sim_freq"] = engine_config["control_freq"] * args.rigid_substeps
    env_config = yaml.safe_load((ROOT / "data/envs/pbf_play_env.yaml").read_text())
    checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
    report = {"args": vars(args), "seed": 1234, "runs": {}, "sources": {}}
    paths = [ROOT / p for p in ("mimickit/envs/pbf_play_env.py", "mimickit/run.py",
                               "mimickit/envs/sim_env.py", "mimickit/envs/amp_env.py", "mimickit/envs/smp_env.py",
                               "mimickit/learning/smp_agent.py",
                               "mimickit/engines/newton_pbf_engine.py", "mimickit/learning/base_agent.py",
                               "data/engines/newton_pbf_engine.yaml", "tools/profile_rsp_playback.py")]
    paths += [p for folder in ("RSP/rendering", "RSP/runtime") for p in (ROOT / folder).glob("*.py")]
    for path in paths:
        relative = str(path.relative_to(ROOT))
        report["sources"][relative] = hashlib.sha256(path.read_bytes()).hexdigest()
        destination = output / "source" / path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    set_rand_seed(1234)
    mp_util.init(0, 1, "cuda:0", None)
    env = agent = None
    try:
        with patch.object(newton_pbf_engine, "PBFViewer", ProfileViewer):
            env = resource_util.construct(PBFPlayEnv, env_config, engine_config, 1, "cuda:0", True)
        agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
        agent.load(str(checkpoint / "model.pt"))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        eng, viewer = env._engine, env._engine.get_viewer()
        renderer = viewer.renderer
        def drain():
            renderer._make_current()
            wp.synchronize()
            renderer.gl.glFinish()
        report.update(particles=eng.get_sim_model().particle_count, bodies=eng.get_sim_model().body_count,
                      fluid_settings=asdict(fluid_clock.DEFAULT_SETTINGS),
                      schedule=asdict(eng.get_play_scene().fluid_schedule), device=wp.get_device().name,
                      versions=dict(newton=newton.__version__, warp=wp.__version__, torch=torch.__version__),
                      resolution=list(renderer.window.get_framebuffer_size()),
                      msaa=renderer.msaa_samples, shadows=renderer.draw_shadows,
                      note="CPU wall ms. Nested render/post details must not be added to their parents. "
                           "post may include unfinished physics in normal runs. separate-wait changes scheduling.")
        report["engine_config"] = engine_config
        for mode in args.modes:
            set_rand_seed(1234)
            agent._curr_obs, agent._curr_info = env.reset()
            recorder = FrameProfiler(args, drain)
            with ExitStack() as stack, torch.no_grad():
                if args.loop == "legacy":
                    original_decide = agent._decide_action
                    def decide(*a, **kw):
                        recorder.next_frame()
                        start = time.perf_counter()
                        try:
                            return original_decide(*a, **kw)
                        finally:
                            recorder.add("policy_ms", (time.perf_counter() - start) * 1000)
                    stack.enter_context(patch.object(agent, "_decide_action", decide))
                else:
                    from RSP.runtime.playback import PlaybackRunner
                    runner = PlaybackRunner(env, agent)
                    report["policy_interval_frames"] = runner.policy_interval
                    report["policy_hz"] = 1.0 / (eng.get_timestep() * runner.policy_interval)
                    original_tick = runner.tick
                    def tick():
                        recorder.next_frame()
                        return original_tick()
                    stack.enter_context(patch.object(runner, "tick", tick))
                    recorder.hook(stack, runner.policy, "act", "policy_ms")
                original_physics = env._physics_step
                def physics():
                    start = time.perf_counter()
                    result = original_physics()
                    recorder.add("physics_submit_ms", (time.perf_counter() - start) * 1000)
                    if args.separate_wait:
                        start = time.perf_counter()
                        wp.synchronize()
                        recorder.add("physics_wait_ms", (time.perf_counter() - start) * 1000)
                    return result
                stack.enter_context(patch.object(env, "_physics_step", physics))
                for obj, method, key in (
                    (env, "_pre_physics_step", "action_ms"),
                    (env, "_post_physics_step", "post_ms"),
                    (env, "_update_observations", "observations_ms"),
                    (env, "_update_misc", "misc_ms"),
                    (eng._presentation, "render", "render_ms"),
                    (viewer, "log_state", "state_update_ms"),
                    (viewer, "_log_particles", "particle_update_ms"),
                    (renderer, "render", "scene_draw_ms"),
                    (renderer, "update", "events_ms"),
                    (renderer, "present", "present_ms"),
                    (viewer.gui, "prepare_frame", "ui_prepare_ms"),
                    (viewer.gui, "render_prepared_frame", "ui_draw_ms"),
                    (agent._test_return_tracker, "update", "evaluation_ms"),
                    (agent, "_reset_done_envs", "evaluation_ms"),
                ):
                    recorder.hook(stack, obj, method, key)
                try:
                    if args.loop == "legacy":
                        agent._rollout_test(np.iinfo(np.int64).max)
                    else:
                        runner.run(reset=False)
                except EndSamples:
                    pass
                drain()
                completed = (time.perf_counter() - recorder.batch_start) * 1000 / args.frames
            assert len(recorder.samples) == args.frames
            assert torch.isfinite(env._obs_buf).all()
            keys = sorted(set().union(*(sample.keys() for sample in recorder.samples)))
            summary = {key: statistics([sample.get(key, 0.) for sample in recorder.samples]) for key in keys}
            summary["frame_without_limit_ms"] = statistics([s["frame_ms"] - s["limit_ms"] for s in recorder.samples])
            summary["completed_batch_mean_ms"] = completed
            report["runs"][mode] = dict(summary=summary, samples=recorder.samples)
            (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
            if args.screenshot:
                viewer.end_frame()
                Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(output / f"{mode}.png")
            print(json.dumps(dict(mode=mode, frame=summary["frame_ms"], render=summary["render_ms"])), flush=True)
        # A small, exact-state fixture permits policy equivalence checks after
        # changing the environment glue, independent of trajectory divergence.
        fixtures = []
        with torch.no_grad():
            for use_sphere in (False, True):
                env.set_policy_use_scene_sphere(use_sphere)
                obs = env._compute_obs()
                action, _ = agent._decide_action(obs, env._info)
                methods = ("get_root_pos", "get_root_rot", "get_root_vel", "get_root_ang_vel",
                           "get_dof_pos", "get_dof_vel", "get_body_pos")
                state = {(method, env._get_char_id()): getattr(eng, method)(env._get_char_id()).detach().cpu().clone()
                         for method in methods}
                for method in ("get_root_pos", "get_root_vel"):
                    obj = env._scene_obj_ids[-1]
                    state[(method, obj)] = getattr(eng, method)(obj).detach().cpu().clone()
                fixtures.append(dict(use_sphere=use_sphere, state=state, obs=obs.cpu(), action=action.cpu()))
        torch.save(fixtures, output / "policy_fixtures.pt")
        print(f"PLAYBACK PROFILE PASS: {output / 'report.json'}", flush=True)
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None, mp_util.shutdown)


if __name__ == "__main__":
    main()
