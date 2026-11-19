"""Explicit short GPU integration check; not collected by unit-test discovery."""
from pathlib import Path
import argparse
import sys
import tempfile
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    output = ROOT / "output/scene_checks/tmp"
    output.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(output)
    import torch
    import warp as wp
    wp.config.kernel_cache_dir = str(ROOT / "output/scene_checks/warp_cache")
    from util import mp_util
    from RSP.integrations.mimickit.profiles import load_profile, read_yaml
    from RSP.app.application import open_playback
    from RSP.app.ui.scene_choices import SceneChoices
    from RSP.app.playback import bind_playback
    from RSP.sim_ready.catalog import AssetRef, resolve_asset, digest

    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', default='g1')
    parser.add_argument('--custom', choices=('empty', 'multi'))
    args = parser.parse_args()
    profile = load_profile(args.profile)
    selection = profile.scene_preset
    template = read_yaml(selection)
    if args.custom:
        choices = SceneChoices(template)
        counts = {} if args.custom == 'empty' else {
            next(iter(versions.values())): 2 for versions in choices.props.values()}
        selection = choices.build(AssetRef(**template['container']), counts, mode='random')
    published = resolve_asset(AssetRef(**template['character']))
    before = {name: digest(published.file(name)) for name in published.record['files']}
    mp_util.init(0, 1, profile.device, None)
    try:
        with wp.ScopedDevice(profile.device), torch.no_grad(), \
                patch('RSP.sim_ready.pipelines.articulation.prepare_articulated_asset', side_effect=AssertionError('Runtime baking')), \
                patch('RSP.sim_ready.pipelines.publication.prepare_sources', side_effect=AssertionError('Runtime publication')):
            with open_playback(profile, selection, None, visualize=False) as session, bind_playback(session) as runner:
                env, agent = session.env, session.agent
                runner.reset()
                torch.testing.assert_close(env._obs_buf, env._compute_obs(), rtol=1e-5, atol=1e-6)
                action, _ = agent._decide_action(runner.obs, runner.info)
                torch.testing.assert_close(runner.policy.act(runner.obs), action, rtol=0, atol=0)
                if args.profile == 'g1':
                    assert runner.obs.shape == (1, 243) and action.shape == (1, 29)
                    scene = env._engine.get_play_scene()
                    assert len(scene.body_meshes) == 21 + len(env._scene_obj_ids)
                assert runner.policy_interval == round(read_yaml(profile.engine_config)["control_freq"] / session.policy_hz)
                frames = 2 if args.custom else 11
                with patch.object(runner.policy, 'act', wraps=runner.policy.act) as act:
                    for _ in range(frames):
                        assert runner.tick()
                    assert act.call_count == (frames + runner.policy_interval - 1) // runner.policy_interval
                assert torch.isfinite(runner.obs).all()
                state = env._play_observations._state
                assert all(torch.isfinite(getattr(state, name)).all() for name in state.__dataclass_fields__)
                env.request_reset()
                runner.reset()
                assert (env._time_buf == 0).all() and runner._action is None
                assert not env.consume_reset_request()
                assert torch.isfinite(runner.obs).all()
                print(f'PASS {args.profile} {args.custom or "preset"}: {frames} frames, reset, observations, policy, binding', flush=True)
        assert before == {name: digest(published.file(name)) for name in before}
    finally:
        mp_util.shutdown()


if __name__ == '__main__':
    main()
