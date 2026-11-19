"""Eleven headless frames and reset; preparation is forbidden during this check."""
from pathlib import Path
import sys
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    from contextlib import ExitStack
    playback_stack = ExitStack()
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", default="pbf_play")
    parser.add_argument("--scene", help="Generated launcher scene YAML, instead of a named preset")
    parser.add_argument("--frames", type=int, default=11)
    args = parser.parse_args()
    import tempfile
    temporary = ROOT / "output/scene_checks/tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(temporary)
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / "output/scene_checks/warp_cache")
    import torch
    import yaml
    from envs.pbf_play_env import PBFPlayEnv
    from learning.agent_builder import build_agent
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util
    from RSP.app.playback import bind_playback
    from RSP.integrations.mimickit.session import MimicKitSession
    from RSP.sim_ready.paths import PUBLISHED_ROOT
    from RSP.sim_ready.catalog import digest
    from RSP.sim_ready.pipelines import articulation, collision, rigid
    from RSP.sim_ready.pipelines import boundary as prepare
    from RSP.sim_ready.pipelines.recipes import pbf_play

    def snapshot():
        return {str(p): (digest(p), p.stat().st_mtime_ns) for p in PUBLISHED_ROOT.rglob("*") if p.is_file()}

    before = snapshot()
    assert before, "Publish the assets first"
    mp_util.init(0, 1, "cuda:0", None)
    torch.manual_seed(1234)
    env = agent = None
    with ExitStack() as stack:
        for module, name in ((articulation, "prepare_articulated_asset"), (collision, "prepare_hulls"),
                             (rigid, "build"), (prepare, "prepare_boundary_asset"),
                             (pbf_play, "prepare_static"), (pbf_play, "primitive_files")):
            stack.enter_context(patch.object(module, name, side_effect=AssertionError("Runtime attempted preparation")))
        try:
            ec = yaml.safe_load((ROOT / "data/envs/pbf_play_env.yaml").read_text())
            ec["scene_preset"] = args.scene or str(ROOT / "data/scenes" / (args.preset + ".yaml"))
            bc = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text())
            env = resource_util.construct(PBFPlayEnv, ec, bc, 1, "cuda:0", False)
            checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
            agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
            agent.load(str(checkpoint / "model.pt"))
            agent.eval()
            agent.set_mode(AgentMode.TEST)
            runner = playback_stack.enter_context(bind_playback(MimicKitSession(env, agent)))
            runner.reset()
            with patch.object(runner.policy, "act", wraps=runner.policy.act) as act:
                for _ in range(args.frames):
                    assert runner.tick()
                assert act.call_count == (args.frames + runner.policy_interval - 1) // runner.policy_interval
            assert torch.isfinite(runner.obs).all()
            runner.reset()
            assert torch.all(env._time_buf == 0)
            assert runner._action is None and runner._policy_phase == 0
            for spec in env._scene_spec.instances:
                obj_id = env._instance_ids[spec.name]
                for getter, value in ((env._engine.get_root_pos, spec.position),
                                      (env._engine.get_root_rot, spec.rotation),
                                      (env._engine.get_root_vel, spec.linear_velocity),
                                      (env._engine.get_root_ang_vel, spec.angular_velocity)):
                    torch.testing.assert_close(getter(obj_id)[0], torch.tensor(value, device="cuda:0"))
            if env._policy_sphere_id is None:
                env.request_policy_use_scene_sphere(True)
                env.apply_playback_commands()
                assert not env.get_policy_use_scene_sphere()
            assert snapshot() == before, "Playback changed published assets"
            print(f"PASS: {args.preset}, {args.frames} frames, policy cadence, reset states, finite observations, read-only assets, no preparation")
        finally:
            resource_util.cleanup(playback_stack.close, env.close if env else None, agent.close if agent else None)


if __name__ == "__main__":
    main()
