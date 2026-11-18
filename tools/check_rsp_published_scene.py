"""Eleven headless frames and reset; preparation is forbidden during this check."""
from pathlib import Path
import sys
from unittest.mock import patch
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
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
    from RSP.runtime.playback import PlaybackRunner
    from RSP.sim_ready.paths import PUBLISHED_ROOT
    from RSP.sim_ready.catalog import digest
    from RSP.sim_ready.pipelines import articulation, collision, rigid
    from RSP.sim_ready.pipelines.boundary import prepare
    from RSP.sim_ready.pipelines.scenes import pbf_play

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
            bc = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text())
            env = resource_util.construct(PBFPlayEnv, ec, bc, 1, "cuda:0", False)
            checkpoint = ROOT / "output/a5000_dodgeball_2048_full_r01"
            agent = build_agent(str(checkpoint / "agent_config.yaml"), env, "cuda:0")
            agent.load(str(checkpoint / "model.pt"))
            agent.eval()
            agent.set_mode(AgentMode.TEST)
            runner = PlaybackRunner(env, agent)
            runner.reset()
            with patch.object(runner.policy, "act", wraps=runner.policy.act) as act:
                for _ in range(11):
                    assert runner.tick()
                assert act.call_count == 3
            assert torch.isfinite(runner.obs).all()
            runner.reset()
            assert torch.all(env._time_buf == 0)
            assert runner._action is None and runner._policy_phase == 0
            for obj_id, spec in zip(env._scene_obj_ids, pbf_play.RIGIDS):
                torch.testing.assert_close(env._engine.get_root_pos(obj_id)[0],
                    torch.tensor(spec.center, device="cuda:0"))
            assert snapshot() == before, "Playback changed published assets"
            print("PASS: 11 frames, 3 policy updates, reset, finite observations, read-only assets, no preparation")
        finally:
            resource_util.cleanup(env.close if env else None, agent.close if agent else None)


if __name__ == "__main__":
    main()
