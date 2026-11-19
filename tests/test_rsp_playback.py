"""Playback contracts that must hold without training or simulator readback."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]

from RSP.app.config import boundary_support_radius, get_fluid_settings
from RSP.app.playback import PlaybackRunner, bind_playback, play_mimickit
from RSP.app.ui.interaction import FluidInteraction
from RSP.integrations.mimickit.session import MimicKitSession
from RSP.rendering.newton_gl import NewtonBillboardViewer
from envs.pbf_play_env import PBFPlayEnv
from RSP.physics import fluid_clock


class PlaybackChecks(unittest.TestCase):
    def test_pause_step_and_queued_reset_precede_policy(self):
        events = []
        viewer = SimpleNamespace(should_step=Mock(side_effect=[False, True, True]))
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
        runner = PlaybackRunner(MimicKitSession(env, None, policy=SimpleNamespace(act=decide)))
        self.assertFalse(runner.tick())
        self.assertEqual(events, ["commands", "render"])
        self.assertTrue(runner.tick())
        self.assertEqual(events[-3:], ["commands", ("policy", "obs"), ("step", "action")])
        self.assertTrue(runner.tick())
        self.assertEqual(events[-2], ("policy", "next"))
        self.assertEqual(viewer.should_step.call_count, 3)

    def test_policy_holds_action_for_five_frames_and_resets_while_paused(self):
        viewer = SimpleNamespace(should_step=Mock(return_value=True))
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
        runner = PlaybackRunner(MimicKitSession(env, None, policy=policy))
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
        runner = PlaybackRunner(MimicKitSession(env, None, policy=policy),
                                fluid_controls=FluidInteraction(settings))
        self.assertEqual(events, [])
        viewer.request_step()
        env.request_reset()
        env.request_reset()
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
        policy, agent = object(), Mock()
        for with_viewer in (False, True):
            with self.subTest(with_viewer=with_viewer):
                viewer = Mock() if with_viewer else None
                env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/120),
                                      _obs_buf="obs", _info={}, _device="cpu", reset=Mock(), request_reset=Mock())
                with bind_playback(MimicKitSession(env, agent, policy=policy)) as runner:
                    self.assertIs(runner.policy, policy)
                    self.assertIs(runner.viewer, viewer)
                    self.assertIsNone(runner.fluid_controls)
                    env.reset.assert_not_called()
                    env.request_reset.assert_not_called()
                    if viewer is not None:
                        panel = viewer.install_gui.return_value.set_panel.call_args.args[0]
                        panel.request_reset()
                        env.request_reset.assert_called_once_with()
                if viewer is not None:
                    viewer.install_gui.return_value.set_panel.assert_called_with(None)
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
        stats_sink = Mock()
        runner = PlaybackRunner(MimicKitSession(env, None, policy=policy), stats_sink=stats_sink)
        self.assertTrue(runner.tick())
        self.assertEqual(stats_sink.call_count, 3)
        env.reset.assert_called_once_with()
        policy.act.assert_called_once_with("fresh")

    def test_policy_panel_queues_edits_until_frame_boundary(self):
        viewer = Mock()
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        env._engine = SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1/120,
                                      register_ui_callback=Mock(), register_keyboard_callback=Mock())
        env._device, env._obs_buf, env._info = "cpu", None, {}
        env.policy_use_scene_sphere = False
        env._policy_sphere_id = 1
        env._pending_sphere = None
        env._update_observations = Mock()
        env._setup_gui()
        env._engine.register_ui_callback.assert_not_called()
        env._engine.register_keyboard_callback.assert_not_called()
        with bind_playback(MimicKitSession(env, None, policy=object())):
            panel = viewer.install_gui.return_value.set_panel.call_args.args[0]
            imgui = SimpleNamespace(checkbox=Mock(return_value=(True, True)))
            panel.draw_policy(imgui)
            imgui.checkbox.assert_called_once_with("Use scene sphere", False)
            self.assertFalse(env.policy_use_scene_sphere)
            self.assertTrue(env._pending_sphere)
            env._update_observations.assert_not_called()
            env.apply_playback_commands()
            self.assertTrue(env.policy_use_scene_sphere)
            self.assertIsNone(env._pending_sphere)
            env._update_observations.assert_called_once_with()
            imgui.checkbox.return_value = (False, True)
            panel.draw_policy(imgui)
            self.assertEqual(imgui.checkbox.call_args.args, ("Use scene sphere", True))
            self.assertIsNone(env._pending_sphere)

    def test_new_defaults_preserve_reference_config(self):
        reference = fluid_clock.FluidSettings()
        self.assertEqual((reference.pbf.particle_radius, reference.pbf.substeps), (.0125, 3))
        settings = get_fluid_settings()
        self.assertIs(settings, fluid_clock.DEFAULT_SETTINGS)
        engine = yaml.safe_load((ROOT / "data/engines/newton_pbf_engine.yaml").read_text(encoding="utf-8"))
        actual = settings.schedule(1 / engine["control_freq"], engine["sim_freq"] // engine["control_freq"])
        self.assertEqual(actual.calls, (0, 1, 0, 1))
        self.assertAlmostEqual(actual.fluid_frame_dt, 1 / 120)
        self.assertAlmostEqual(actual.fluid_dt, 1 / 240)
        self.assertAlmostEqual(actual.control_dt / actual.rigid_steps, 1 / 480)
        self.assertEqual(settings.pbf.pressure_iterations, 3)
        self.assertAlmostEqual(boundary_support_radius(), .04)
        override = replace(settings, pbf=replace(settings.pbf, particle_radius=.0125))
        with patch.object(fluid_clock, "DEFAULT_SETTINGS", override):
            self.assertIs(get_fluid_settings(), override)
            self.assertAlmostEqual(boundary_support_radius(), .05)
            self.assertAlmostEqual(boundary_support_radius(settings), .04)

    def test_legacy_play_uses_environment_cadence_and_leaves_viewer_open(self):
        viewer, agent = Mock(), Mock()
        agent.get_playback_policy_components.return_value = {}
        engine = SimpleNamespace(get_viewer=lambda: viewer, get_timestep=lambda: 1 / 120)
        env = SimpleNamespace(_engine=engine, _device="cpu", _obs_buf="obs", _info={},
                              get_policy_timestep=Mock(return_value=1 / 40), close=Mock())
        with patch("RSP.integrations.mimickit.session.PolicyInference"), \
                patch("RSP.integrations.mimickit.session.configure_test_agent") as configure, \
                patch.object(PlaybackRunner, "run", autospec=True) as run:
            play_mimickit(env, agent)

        run.assert_called_once()
        runner = run.call_args.args[0]
        self.assertIs(runner.viewer, viewer)
        self.assertEqual(runner.policy_interval, 3)
        env.get_policy_timestep.assert_called_once_with()
        configure.assert_called_once_with(agent)
        viewer.install_gui.assert_called_once_with()
        installed, removed = viewer.install_gui.return_value.set_panel.call_args_list
        self.assertIs(installed.args[0].viewer, viewer)
        self.assertIsNone(removed.args[0])
        viewer.close.assert_not_called()
        env.close.assert_not_called()
        agent.close.assert_not_called()

    def test_gsi_velocities_keep_the_policy_sample_period(self):
        import torch
        from anim.mjcf_char_model import MJCFCharModel
        from envs.pbf_play_env import PBFPlayEnv
        from envs.sim_env import SimEnv
        kin = MJCFCharModel("cpu")
        kin.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
        env = object.__new__(PBFPlayEnv)
        env._device, env._kin_char_model = "cpu", kin
        env._policy_hz = 30
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
