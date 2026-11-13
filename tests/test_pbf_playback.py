"""Playback contracts and actual CUDA/GL uploads; run in the Newton environment."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import newton
import numpy as np
import torch
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "mimickit"), str(ROOT)]
OUTPUT = ROOT / "output/rsp_playback_checks"
(OUTPUT / "tmp").mkdir(parents=True, exist_ok=True)
tempfile.tempdir = str(OUTPUT / "tmp")
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")

from RSP.playback import run_playback
from engines.newton_pbf_viewer import PBFViewer
from envs.amp_env import AMPEnv
from envs.pbf_play_env import PBFPlayEnv
from envs.sim_env import PlayMode
from envs.smp_env import SMPEnv
from learning.base_agent import AgentMode
from util import arg_parser, torch_util
import run as entry


class PlaybackContracts(unittest.TestCase):
    def test_runner_preserves_actor_inputs_without_episode_reset(self):
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        observations = [torch.tensor([i], dtype=torch.float32) for i in range(4)]
        infos = [{"step": i} for i in range(4)]
        env.reset = Mock(return_value=(observations[0], infos[0]))
        # Even a terminal flag must not trigger an automatic reset in play.
        env.step = Mock(side_effect=[(observations[i], None, torch.ones(1), infos[i])
                                     for i in range(1, 4)])
        agent = Mock()
        def decide(obs, info):
            self.assertFalse(torch.is_grad_enabled())
            self.assertIs(obs, observations[info["step"]])
            return obs + 1, {}
        agent._decide_action.side_effect = decide
        run_playback(agent, env, num_steps=3)
        env.reset.assert_called_once_with()
        self.assertEqual(env.step.call_count, 3)
        agent.eval.assert_called_once_with()
        agent.set_mode.assert_called_once_with(AgentMode.TEST)
        agent.test_model.assert_not_called()
        agent._test_return_tracker.update.assert_not_called()

    def test_runner_zero_steps_and_invalid_environment(self):
        with self.assertRaisesRegex(ValueError, "PBFPlayEnv"):
            run_playback(Mock(), Mock(), num_steps=0)
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        env.reset = Mock(return_value=(None, None))
        env.step = Mock()
        agent = Mock()
        run_playback(agent, env, num_steps=0)
        env.reset.assert_called_once_with()
        agent._decide_action.assert_not_called()
        env.step.assert_not_called()
        with self.assertRaises(ValueError):
            run_playback(agent, env, num_steps=-1)

    def test_training_observations_remain_opt_out(self):
        with patch.object(SMPEnv, "__init__"):
            env = PBFPlayEnv({}, {"engine_name": "newton_pbf"}, 1, "cpu", False)
        self.assertFalse(env._playback_mode)
        torch.testing.assert_close(env._inactive_projectile_pos, torch.tensor([[[10., 10., .1]]]))
        self.assertEqual(torch.count_nonzero(env._inactive_projectile_vel), 0)
        with patch.object(AMPEnv, "_update_disc_hist") as hist, \
                patch.object(AMPEnv, "_update_disc_obs") as obs:
            env._update_disc_hist()
            env._update_disc_obs()
            hist.assert_called_once_with()
            obs.assert_called_once_with(None)
            env.set_playback_mode(True)
            env._update_disc_hist()
            env._update_disc_obs()
            self.assertEqual(hist.call_count, 1)
            self.assertEqual(obs.call_count, 1)
            env.set_playback_mode(False)
            env._update_disc_hist()
            env._update_disc_obs()
            self.assertEqual(hist.call_count, 2)
            self.assertEqual(obs.call_count, 2)

    def test_play_exit_uses_existing_entry_cleanup(self):
        args = arg_parser.ArgParser()
        args.load_args(["--mode", "play", "--out_dir", "", "--rand_seed", "1234"])
        for exit_signal in (SystemExit(0), KeyboardInterrupt()):
            with self.subTest(signal=type(exit_signal).__name__):
                env = PBFPlayEnv.__new__(PBFPlayEnv)
                env.reset = Mock(return_value=(None, None))
                env.step = Mock(side_effect=exit_signal)
                env.close = Mock()
                agent = Mock()
                agent._decide_action.return_value = (None, None)
                with patch.object(entry, "build_env", return_value=env), \
                        patch.object(entry, "build_agent", return_value=agent), \
                        patch.object(entry.mp_util, "init"), \
                        patch.object(entry, "set_rand_seed"), \
                        patch.object(entry, "create_output_dir"), \
                        patch.object(entry.display, "close_virtual_display") as display_close:
                    with self.assertRaises(type(exit_signal)):
                        entry.run(0, 1, "cpu", None, args)
                env.close.assert_called_once_with()
                agent.close.assert_called_once_with()
                display_close.assert_called_once_with()

    def test_pause_step_and_manual_reset_callbacks(self):
        env = PBFPlayEnv.__new__(PBFPlayEnv)
        env._play_mode = PlayMode.PLAY
        env._engine = Mock()
        callbacks = {}
        env._engine.register_keyboard_callback.side_effect = callbacks.__setitem__
        env.reset = Mock()
        env._setup_gui()
        callbacks["R"]()
        env.reset.assert_called_once_with()
        env._engine.get_viewer().set_reset_callback.assert_called_once_with(env.reset)
        callbacks["ENTER"]()
        self.assertIs(env._play_mode, PlayMode.PAUSE)
        env._update_camera = Mock()
        env._render_scene = Mock(side_effect=callbacks["SPACE"])
        env._render()
        self.assertIs(env._play_mode, PlayMode.PAUSE)
        env._render_scene.assert_called_once_with()
        callbacks["ENTER"]()
        self.assertIs(env._play_mode, PlayMode.PLAY)


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA/OpenGL")
class ParticleUploads(unittest.TestCase):
    def setUp(self):
        # Newton closes pyglet's process-wide loop with each viewer. Each test
        # owns a fresh window and needs a fresh event-loop exit flag as well.
        import pyglet
        pyglet.app.event_loop.has_exit = False

    @staticmethod
    def model(count):
        builder = newton.ModelBuilder()
        for i in range(count):
            builder.add_particle((i * .25, 0., 1.), (0., 0., 0.), 1., radius=.1)
        return builder.finalize(device="cuda:0")

    def test_interop_cache_and_model_lifecycle(self):
        viewer = PBFViewer(width=320, height=240, headless=True)
        try:
            flags = viewer.CudaInterop
            self.assertEqual(viewer._enable_cuda_interop,
                             flags.DYNAMIC_MESH | flags.INSTANCES | flags.POINTS | flags.LINES)
            model = self.model(3)
            state = model.state()
            viewer.set_model(model)
            viewer.show_particles = True
            viewer._log_particles(state)
            batch = viewer.objects[viewer._cube_instances]
            self.assertIsNotNone(batch._instance_transform_cuda_buffer)
            # No full-particle downloads after initial appearance upload.
            with patch.object(wp.array, "numpy", side_effect=AssertionError("unexpected host readback")):
                viewer._log_particles(state)
                viewer.show_particles = False
                viewer._log_particles(state)
                self.assertTrue(batch.hidden)
                viewer.show_particles = True
                viewer._log_particles(state)
                self.assertFalse(batch.hidden)
            buffers = viewer._particle_transforms
            model.particle_count = 1
            state.particle_q = wp.array([[0., 0., 1.]], dtype=wp.vec3, device="cuda:0")
            viewer._log_particles(state)
            self.assertEqual(batch.active_instances, 1)
            self.assertIsNot(viewer._particle_transforms, buffers)
            model.particle_count = 0
            viewer._log_particles(state)
            self.assertTrue(batch.hidden)
            model.particle_count = 1
            viewer._log_particles(state)
            self.assertFalse(batch.hidden)
            model.particle_count = 5
            model.particle_radius = wp.full(5, .1, device="cuda:0")
            state.particle_q = wp.zeros(5, dtype=wp.vec3, device="cuda:0")
            viewer._log_particles(state)
            self.assertEqual(viewer.objects[viewer._cube_instances].active_instances, 5)
            self.assertEqual(len(viewer._particle_colors), 5)
            # Replacing a model must also invalidate same-sized appearance data.
            buffers = viewer._particle_transforms
            replacement = self.model(1)
            viewer.set_model(replacement)
            viewer.show_particles = True
            self.assertFalse(hasattr(viewer, "_particle_transforms"))
            viewer._log_particles(replacement.state())
            self.assertIsNot(viewer._particle_transforms, buffers)
            viewer.set_camera(wp.vec3(2., -3., 2.), -15., 120.)
            viewer.begin_frame(0.)
            viewer.log_state(replacement.state())
            viewer.end_frame()
            frame = viewer.get_frame().numpy()
            self.assertEqual(frame.shape, (240, 320, 3))
            self.assertGreater(float(frame.std()), 1.)
            viewer.set_model(None)
            self.assertFalse(hasattr(viewer, "_particle_transforms"))
        finally:
            viewer.close()
            viewer.close()

    def test_explicit_interop_setting(self):
        viewer = PBFViewer(width=320, height=240, headless=True,
                           enable_cuda_interop=PBFViewer.CudaInterop.NONE)
        try:
            model = self.model(1)
            viewer.set_model(model)
            viewer.show_particles = True
            viewer._log_particles(model.state())
            self.assertIsNone(viewer.objects[viewer._cube_instances]._instance_transform_cuda_buffer)
        finally:
            viewer.close()

    def test_interop_matches_host_upload_pixels(self):
        images = []
        for options in ({"enable_cuda_interop": PBFViewer.CudaInterop.NONE}, {}):
            self.setUp()
            viewer = PBFViewer(width=320, height=240, headless=True, **options)
            try:
                model = self.model(3)
                viewer.set_model(model)
                viewer.show_particles = True
                viewer.set_camera(wp.vec3(2., -3., 2.), -15., 120.)
                def frame():
                    viewer.begin_frame(0.)
                    viewer.log_state(model.state())
                    viewer.end_frame()
                    return viewer.get_frame().numpy()
                visible = frame()
                # Cached appearance must render identically on subsequent frames.
                np.testing.assert_array_equal(frame(), visible)
                viewer.show_particles = False
                self.assertTrue(np.any(frame() != visible), "Particles were outside the image")
                images.append(visible)
            finally:
                viewer.close()
        np.testing.assert_array_equal(images[0], images[1])


class RotationHelpers(unittest.TestCase):
    def test_identity_near_zero_and_quarter_turn(self):
        for device in (["cpu", "cuda:0"] if torch.cuda.is_available() else ["cpu"]):
            with self.subTest(device=device):
                x = torch.tensor([[0., 0., 0.], [1e-8, 0., 0.], [0., 0., np.pi / 2]], device=device)
                q = torch_util.exp_map_to_quat(x)
                expected = torch.tensor([[0., 0., 0., 1.], [0., 0., 0., 1.],
                                         [0., 0., 2.**-.5, 2.**-.5]], device=device)
                torch.testing.assert_close(q, expected)
                axis, angle = torch_util.quat_to_axis_angle(q)
                torch.testing.assert_close(axis, torch.tensor([[0., 0., 1.]] * 3, device=device))
                torch.testing.assert_close(angle, torch.tensor([0., 0., np.pi / 2], device=device))
                torch.testing.assert_close(torch_util.calc_heading_quat(q), q)
                torch.testing.assert_close(torch_util.calc_heading_quat_inv(q), torch_util.quat_conjugate(q))
                expected_basis = torch.tensor([[1., 0., 0., 0., 0., 1.]] * 2 +
                                              [[0., 1., 0., 0., 0., 1.]], device=device)
                torch.testing.assert_close(torch_util.quat_to_tan_norm(q), expected_basis)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA sync diagnostics")
    def test_scripted_helpers_do_not_synchronize_scalar_writes(self):
        x = torch.randn(32, 3, device="cuda:0")
        q = torch_util.exp_map_to_quat(x)
        functions = ((torch_util.exp_map_to_axis_angle, x), (torch_util.quat_to_axis_angle, q),
                     (torch_util.quat_to_tan_norm, q), (torch_util.calc_heading_quat, q),
                     (torch_util.calc_heading_quat_inv, q))
        for func, value in functions:
            for _ in range(5):
                func(value)
        previous = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            for func, value in functions:
                func(value)
        finally:
            torch.cuda.set_sync_debug_mode(previous)


if __name__ == "__main__":
    unittest.main()
