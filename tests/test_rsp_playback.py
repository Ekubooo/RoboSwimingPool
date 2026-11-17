"""Playback contracts that must hold without training or simulator readback."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]

from RSP.runtime.playback import PlaybackRunner
from RSP.rendering.newton_gl import NewtonBillboardViewer
from envs.pbf_play_env import PBFPlayEnv
from RSP.runtime.fluid_clock import DEFAULT_SETTINGS, FluidSettings
from solvers.pbf.pbf_helper import init_scene_parameters
from engines.newton_pbf_runtime import runtime_config


class PlaybackChecks(unittest.TestCase):
    def test_pause_step_and_queued_reset_precede_policy(self):
        events = []
        viewer = SimpleNamespace(should_step=Mock(side_effect=[False, True, True]))
        viewer.bind_playback_controls = Mock()
        viewer.set_paused = lambda paused: NewtonBillboardViewer.set_paused(viewer, paused)
        env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/30),
                              _obs_buf="obs", _info={}, _device="cpu")
        env._pending_reset = False
        env.request_reset = lambda: PBFPlayEnv.request_reset(env)
        env.consume_reset_request = lambda: PBFPlayEnv.consume_reset_request(env)
        env.apply_playback_commands = lambda: events.append("commands")
        env.render_only = lambda: events.append("render")
        def step(action):
            events.append(("step", action))
            return "next", 0, 0, {}
        env.step = step
        def decide(obs):
            events.append(("policy", obs))
            return "action"
        env.get_policy_use_scene_sphere = Mock(return_value=False)
        env.request_policy_use_scene_sphere = Mock()
        if env._engine.get_viewer() is not None:
            env._engine.get_viewer().register_ui_callback = Mock()
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
        viewer.bind_playback_controls = Mock()
        viewer.set_paused = lambda paused: NewtonBillboardViewer.set_paused(viewer, paused)
        env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/150),
                              _obs_buf="initial", _info={}, _device="cpu")
        env._pending_reset = False
        env.request_reset = lambda: PBFPlayEnv.request_reset(env)
        env.consume_reset_request = lambda: PBFPlayEnv.consume_reset_request(env)
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
        env.get_policy_use_scene_sphere = Mock(return_value=False)
        env.request_policy_use_scene_sphere = Mock()
        if env._engine.get_viewer() is not None:
            env._engine.get_viewer().register_ui_callback = Mock()
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
        env.request_reset()
        env.request_reset()
        self.assertFalse(runner.tick())
        self.assertIsNone(runner._action)
        self.assertEqual(runner._policy_phase, 0)
        self.assertEqual(policy.act.call_count, 3)
        env.reset.assert_called_once_with()
        viewer.should_step.return_value = True
        runner.tick()
        self.assertIs(applied[-1], actions[3])
        runner.reset()
        runner.tick()
        self.assertIs(applied[-1], actions[4])
        self.assertEqual(policy.act.call_args.args[0], "reset")

    def test_reset_order_and_explicit_reset_consume_pending_requests(self):
        events = []
        viewer = NewtonBillboardViewer.__new__(NewtonBillboardViewer)
        viewer.gui = None
        viewer.set_paused(False)
        env = SimpleNamespace(_pending_reset=False, _pending_sphere=True,
                              _obs_buf="obs", _info={}, _device="cpu")
        env.request_reset = lambda: PBFPlayEnv.request_reset(env)
        env.consume_reset_request = lambda: PBFPlayEnv.consume_reset_request(env)
        env.set_policy_use_scene_sphere = lambda enabled: events.append(("sphere", enabled))
        env.apply_playback_commands = lambda: PBFPlayEnv.apply_playback_commands(env)
        env.reset = Mock(side_effect=lambda: (events.append("reset") or "fresh", {}))
        env.render_only = Mock()
        env.step = Mock(return_value=("next", 0, 0, {}))
        settings = Mock(side_effect=lambda **kw: events.append("settings"))
        env._engine = SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/150,
                                     set_fluid_controls=settings, prepare_playback_step=Mock())
        policy = SimpleNamespace(act=Mock(return_value="action"))
        env.get_policy_use_scene_sphere = Mock(return_value=False)
        env.request_policy_use_scene_sphere = Mock()
        if env._engine.get_viewer() is not None:
            env._engine.get_viewer().register_ui_callback = Mock()
        runner = PlaybackRunner(env, None, policy=policy)
        self.assertEqual(events, [])
        viewer.request_step()
        viewer._reset_callback()
        viewer._reset_callback()
        self.assertFalse(runner.tick())
        self.assertEqual(events, ["settings", "reset", ("sphere", True)])
        env.reset.assert_called_once_with()
        policy.act.assert_not_called()
        self.assertFalse(viewer.should_step())
        self.assertIsNone(runner._action)
        self.assertEqual(runner._policy_phase, 0)
        self.assertFalse(runner.tick())
        env.reset.assert_called_once_with()
        viewer.set_paused(False)
        env.request_reset()
        runner.reset()
        self.assertFalse(env.consume_reset_request())
        self.assertFalse(viewer.is_paused())
        runner.tick()
        self.assertEqual(env.reset.call_count, 2)
        policy.act.assert_called_once_with("fresh")

    def test_component_binding_is_cpu_only_and_optional(self):
        from RSP.runtime.mimickit_bindings import build_playback_components
        policy, agent = object(), Mock()
        for with_viewer in (False, True):
            with self.subTest(with_viewer=with_viewer):
                viewer = SimpleNamespace(bind_playback_controls=Mock(), register_ui_callback=Mock()) if with_viewer else None
                env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer),
                                      request_reset=Mock(), reset=Mock(), get_policy_use_scene_sphere=Mock(return_value=False),
                                      request_policy_use_scene_sphere=Mock())
                components = build_playback_components(env, agent, policy=policy)
                self.assertIs(components.policy, policy)
                self.assertIs(components.viewer, viewer)
                self.assertIsNone(components.fluid_controls)
                env.reset.assert_not_called()
                env.request_reset.assert_not_called()
                if viewer is not None:
                    viewer.bind_playback_controls.assert_called_once_with(None, env.request_reset)
        agent.get_playback_policy_components.assert_not_called()

    def test_headless_request_resets_then_advances(self):
        env = SimpleNamespace(_pending_reset=True, _obs_buf="old", _info={}, _device="cpu",
                              _engine=SimpleNamespace(get_viewer=lambda: None, get_timestep=lambda: 1/150))
        env.consume_reset_request = lambda: PBFPlayEnv.consume_reset_request(env)
        env.reset = Mock(return_value=("fresh", {}))
        env.apply_playback_commands = Mock()
        env.step = Mock(return_value=("next", 0, 0, {}))
        policy = SimpleNamespace(act=Mock(return_value="action"))
        env.get_policy_use_scene_sphere = Mock(return_value=False)
        env.request_policy_use_scene_sphere = Mock()
        if env._engine.get_viewer() is not None:
            env._engine.get_viewer().register_ui_callback = Mock()
        runner = PlaybackRunner(env, None, policy=policy)
        self.assertTrue(runner.tick())
        env.reset.assert_called_once_with()
        policy.act.assert_called_once_with("fresh")

    def test_policy_panel_queues_edits_until_frame_boundary(self):
        from RSP.runtime.mimickit_bindings import build_playback_components
        viewer = SimpleNamespace(bind_playback_controls=Mock(), register_ui_callback=Mock())
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        env._engine = Mock()
        env._engine.get_viewer.return_value = viewer
        env.policy_use_scene_sphere = False
        env._pending_sphere = None
        env._update_observations = Mock()
        env._setup_gui()
        env._engine.register_ui_callback.assert_not_called()
        env._engine.register_keyboard_callback.assert_not_called()
        build_playback_components(env, None, policy=object())
        viewer.register_ui_callback.assert_called_once()
        self.assertEqual(viewer.register_ui_callback.call_args.kwargs, {"position": "side"})
        draw = viewer.register_ui_callback.call_args.args[0]
        imgui = SimpleNamespace(text=Mock(), checkbox=Mock(return_value=(True, True)))
        draw(imgui)
        imgui.checkbox.assert_called_once_with("Policy: scene sphere", False)
        self.assertFalse(env.policy_use_scene_sphere)
        self.assertTrue(env._pending_sphere)
        env._update_observations.assert_not_called()
        env.apply_playback_commands()
        self.assertTrue(env.policy_use_scene_sphere)
        self.assertIsNone(env._pending_sphere)
        env._update_observations.assert_called_once_with()
        imgui.checkbox.return_value = (False, True)
        draw(imgui)
        self.assertEqual(imgui.checkbox.call_args.args, ("Policy: scene sphere", True))
        self.assertIsNone(env._pending_sphere)

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
