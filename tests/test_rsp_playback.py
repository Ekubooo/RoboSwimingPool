"""Playback contracts that must hold without training or simulator readback."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]

from RSP.runtime.playback import PlaybackRunner
from RSP.runtime.fluid_clock import DEFAULT_SETTINGS, FluidSettings
from solvers.pbf.pbf_helper import init_scene_parameters
from engines.newton_pbf_runtime import runtime_config


class PlaybackChecks(unittest.TestCase):
    def test_pause_step_and_queued_reset_precede_policy(self):
        events = []
        viewer = SimpleNamespace(should_step=Mock(side_effect=[False, True, True]))
        env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/30),
                              _obs_buf="obs", _info={}, _device="cpu")
        env.apply_playback_commands = lambda: events.append("commands")
        env.render_only = lambda: events.append("render")
        def step(action):
            events.append(("step", action))
            return "next", 0, 0, {}
        env.step = step
        def decide(obs):
            events.append(("policy", obs))
            return "action"
        runner = PlaybackRunner(env, None, policy=SimpleNamespace(act=decide))
        self.assertFalse(runner.tick())
        self.assertEqual(events, ["commands", "render"])
        self.assertTrue(runner.tick())
        self.assertEqual(events[-3:], ["commands", ("policy", "obs"), ("step", "action")])
        self.assertTrue(runner.tick())
        self.assertEqual(events[-2], ("policy", "next"))
        self.assertEqual(viewer.should_step.call_count, 3)

    def test_policy_holds_action_for_five_frames_and_resets_while_paused(self):
        viewer = SimpleNamespace(should_step=Mock(return_value=True))
        env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/150),
                              _obs_buf="initial", _info={}, _device="cpu")
        env.apply_playback_commands = Mock(return_value=False)
        env.render_only = Mock()
        applied = []
        def step(action):
            applied.append(action)
            return f"observation-{len(applied)}", 0, 0, {}
        env.step = step
        env.reset = Mock(return_value=("reset", {}))
        actions = [object() for _ in range(5)]
        policy = SimpleNamespace(act=Mock(side_effect=actions))
        runner = PlaybackRunner(env, None, policy=policy)
        self.assertEqual(runner.policy_interval, 5)
        for _ in range(11):
            runner.tick()
        self.assertEqual(applied, [actions[0]] * 5 + [actions[1]] * 5 + [actions[2]])
        self.assertEqual([c.args[0] for c in policy.act.call_args_list],
                         ["initial", "observation-5", "observation-10"])
        viewer.should_step.return_value = False
        self.assertFalse(runner.tick())
        self.assertEqual(runner._policy_phase, 1)
        # A UI reset is applied even while paused; the old held target is discarded.
        env.apply_playback_commands.return_value = True
        self.assertFalse(runner.tick())
        self.assertIsNone(runner._action)
        self.assertEqual(runner._policy_phase, 0)
        self.assertEqual(policy.act.call_count, 3)
        env.apply_playback_commands.return_value = False
        viewer.should_step.return_value = True
        runner.tick()
        self.assertIs(applied[-1], actions[3])
        runner.reset()
        runner.tick()
        self.assertIs(applied[-1], actions[4])
        self.assertEqual(policy.act.call_args.args[0], "reset")

    def test_new_defaults_preserve_reference_config(self):
        reference = FluidSettings()
        self.assertEqual((reference.pbf.particle_radius, reference.pbf.substeps), (.0125, 3))
        actual = DEFAULT_SETTINGS.schedule(1/150, 4)
        self.assertEqual(actual.calls, (0, 1, 0, 1))
        self.assertAlmostEqual(actual.fluid_frame_dt, 1/150)
        self.assertAlmostEqual(actual.fluid_dt, 1/300)
        self.assertAlmostEqual(actual.control_dt / actual.rigid_steps, 1/600)
        self.assertEqual(DEFAULT_SETTINGS.time_scale, 1.)
        self.assertEqual(DEFAULT_SETTINGS.pbf.pressure_iterations, 3)
        config = runtime_config(1/150, 4)
        particles = SimpleNamespace(config=config)
        init_scene_parameters(particles)
        self.assertEqual(particles.n, 299475)
        for actual, expected in zip(config.block_start, (1.56, .75, .50)):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(config.block_end, (2.96, 1.90, 3.55)):
            self.assertAlmostEqual(actual, expected)

    def test_gsi_velocities_keep_the_policy_sample_period(self):
        import torch
        from anim.mjcf_char_model import MJCFCharModel
        from envs.pbf_play_env import PBFPlayEnv
        from envs.sim_env import SimEnv
        kin = MJCFCharModel("cpu")
        kin.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
        env = object.__new__(PBFPlayEnv)
        env._device, env._kin_char_model = "cpu", kin
        env._engine = SimpleNamespace(get_timestep=lambda: 1/150)
        frames = torch.zeros(1, 2, 6 + kin.get_dof_size())
        frames[0, 1, 0] = 1/30  # one metre/second in the original 30 Hz samples
        frames[0, 1, 5] = 1/30  # one radian/second about Z
        frames[0, 1, 6] = .01
        states = env._motion_frames_to_init_states(frames)
        torch.testing.assert_close(states["root_vel"][..., 0], torch.ones(1, 1, 2))
        torch.testing.assert_close(states["root_ang_vel"][..., 2], torch.ones(1, 1, 2))
        torch.testing.assert_close(states["dof_vel"][..., 0], torch.full((1, 1, 2), .3))
        # Environments without playback decimation retain their existing clock.
        self.assertEqual(SimEnv.get_policy_timestep(env), 1/150)
        self.assertEqual(env.get_policy_timestep(), 1/30)


if __name__ == "__main__":
    unittest.main()
