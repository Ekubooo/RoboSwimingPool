"""Targeted control, clock and captured-feedback checks; no assets or recording."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]

import numpy as np
import warp as wp

from RSP.rendering.playback_ui import PlaybackGui
from RSP.rendering.newton_gl import NewtonBillboardViewer
from envs.pbf_play_env import PBFPlayEnv
from RSP.runtime.coupling import ClockedPBFMuJoCo, ClockedStepGraph, _apply_feedback
from RSP.runtime.fluid_clock import DEFAULT_SETTINGS, FluidClock
from RSP.runtime.interaction import FluidInteraction


def control_solver(device="cpu"):
    """Exercise the real adapter's control/clock code without a scene builder."""
    solver = object.__new__(ClockedPBFMuJoCo)
    solver.model = SimpleNamespace(device=device, particle_count=1)
    solver.fluid = SimpleNamespace(gravity=wp.vec3(0., 0., -9.81))
    solver._gravity = (0., 0., -9.81)
    solver._base_schedule = DEFAULT_SETTINGS.schedule(1/150, 4)
    solver.clock = FluidClock(solver._base_schedule)
    solver._slow_motion = False
    solver._feedback_value = 1.
    solver.feedback_parameter = wp.full(1, 1., dtype=float, device=device)
    solver.graph_revision = 0
    return solver


class ControlChecks(unittest.TestCase):
    def test_initial_layout_allows_water_and_internal_static_overlap_only(self):
        from dataclasses import replace
        from RSP.sim_ready.pipelines.scenes.spec import load_scene
        from RSP.sim_ready.pipelines.scenes.assembly import resolve_scene
        spec = load_scene()
        # Water may cover props; scene preflight only rejects container/prop overlap.
        body = replace(spec.instances[0], name="probe", position=(1., 1., 1.))
        scene = replace(spec, instances=(body,), fluid_start=(0., 0., 0.), fluid_end=(3., 2., 5.))
        self.assertEqual(resolve_scene(scene).clearances, [])
        with self.assertRaisesRegex(ValueError, "overlaps probe"):
            resolve_scene(replace(scene, instances=(body, replace(body, name="other"))))
        with self.assertRaisesRegex(ValueError, "intersects the container"):
            resolve_scene(replace(scene, instances=(replace(body, position=(.01, 1., 1.)),)))

    def test_runner_reset_pauses_before_policy_and_failed_capture_cannot_advance(self):
        from RSP.runtime.playback import PlaybackRunner
        prepare_step = Mock()
        viewer = SimpleNamespace(gui=SimpleNamespace(), _paused=False, _step_requested=True)
        viewer.set_paused = lambda paused: NewtonBillboardViewer.set_paused(viewer, paused)
        viewer.set_reset_callback = lambda callback: setattr(viewer, "reset_callback", callback)
        viewer.bind_playback_controls = lambda controls, reset: NewtonBillboardViewer.bind_playback_controls(viewer, controls, reset)
        viewer.should_step = lambda: not viewer._paused
        env = SimpleNamespace(_engine=SimpleNamespace(set_fluid_controls=Mock(), prepare_playback_step=prepare_step,
                              get_viewer=lambda: viewer,
                              get_timestep=lambda: 1/150), _obs_buf="obs", _info={}, _device="cpu")
        env._pending_reset = False
        env.request_reset = Mock(side_effect=lambda: PBFPlayEnv.request_reset(env))
        env.consume_reset_request = lambda: PBFPlayEnv.consume_reset_request(env)
        env.reset = Mock(return_value=("reset", {}))
        env.apply_playback_commands = Mock()
        env.render_only, env.step = Mock(), Mock()
        policy = SimpleNamespace(act=Mock())
        env.get_policy_use_scene_sphere = Mock(return_value=False)
        env.request_policy_use_scene_sphere = Mock()
        if env._engine.get_viewer() is not None:
            env._engine.get_viewer().register_ui_callback = Mock()
        runner = PlaybackRunner(env, None, policy=policy)
        runner.fluid_controls.set_slow_motion(True)
        viewer.reset_callback()
        self.assertFalse(runner.tick())
        env.request_reset.assert_called_once()
        env.reset.assert_called_once_with()
        self.assertTrue(viewer._paused)
        self.assertFalse(viewer._step_requested)
        policy.act.assert_not_called()
        env.step.assert_not_called()
        self.assertFalse(runner.fluid_controls.slow_motion)
        env.request_reset.reset_mock()
        viewer._paused = False
        prepare_step.side_effect = RuntimeError("capture rejected")
        with self.assertLogs(level="ERROR"):
            self.assertFalse(runner.tick())
        self.assertTrue(viewer._paused)
        self.assertEqual(runner.fluid_controls.error, "capture rejected")
        policy.act.assert_not_called()
        runner.viewer = None
        with self.assertRaisesRegex(RuntimeError, "capture rejected"):
            runner.tick()
        env.step.assert_not_called()

    def test_keyboard_directions_repeat_capture_and_reset(self):
        from pyglet.window import key
        solver = Mock()
        controls = FluidInteraction(solver.set_fluid_controls)
        viewer = SimpleNamespace(_paused=False, _step_requested=False,
                                 _reset_callback=Mock())
        viewer.set_paused = lambda paused: NewtonBillboardViewer.set_paused(viewer, paused)
        viewer.is_paused = lambda: viewer._paused
        viewer.request_step = lambda: NewtonBillboardViewer.request_step(viewer)
        base = SimpleNamespace(_viewer=viewer, is_keyboard_capturing=Mock(return_value=False),
                               handle_key_press=Mock())
        gui = PlaybackGui(base)
        gui.fluid_controls = controls
        def press(symbol):
            gui.handle_key_press(symbol)
            gui.handle_key_release(symbol)

        gui.handle_key_press(key.Q)
        self.assertEqual(controls.direction, (-1., 0., 0.))
        gui.handle_key_press(key.Q)  # OS repeat must not restore vertical gravity.
        self.assertEqual(controls.direction, (-1., 0., 0.))
        gui.handle_key_release(key.Q)
        press(key.Q)
        self.assertEqual(controls.direction, (0., 0., -1.))
        press(key.E)
        self.assertEqual(controls.direction, (1., 0., 0.))
        press(key.G)
        self.assertEqual(controls.direction, (-1., 0., 0.))
        press(key.E)  # G cleared the previous right-direction selection.
        self.assertEqual(controls.direction, (1., 0., 0.))
        press(key.E)
        press(key.G)
        self.assertEqual(controls.direction, (0., 0., 1.))
        base.is_keyboard_capturing.return_value = True
        press(key.X)
        self.assertFalse(controls.slow_motion)
        base.is_keyboard_capturing.return_value = False
        press(key.X)
        self.assertTrue(controls.slow_motion)
        gui.handle_key_press(key.SPACE)
        gui.clear_pressed_keys()  # Losing focus must not leave a key latched.
        press(key.SPACE)
        self.assertFalse(viewer._paused)
        controls.set_feedback_scale(2.)
        controls.set_gravity_strength(15.)
        press(key.R)
        viewer._reset_callback.assert_called_once_with()
        # UI only requests reset; the runner restores settings at a frame boundary.
        self.assertEqual(controls.gravity_strength, 15.)
        gui.request_reset()
        self.assertEqual(viewer._reset_callback.call_count, 2)

    def test_gravity_commit_and_batched_upload(self):
        solver = Mock()
        controls = FluidInteraction(solver.set_fluid_controls)
        controls.set_gravity_strength(12., commit=False)
        controls.apply_pending()
        solver.set_fluid_controls.assert_not_called()
        controls.commit_gravity()
        controls.set_feedback_scale(.5)
        controls.set_slow_motion(True)
        controls.apply_pending()
        solver.set_fluid_controls.assert_called_once_with(
            gravity=(0., 0., -12.), feedback_scale=.5, slow_motion=True)
        controls.apply_pending()
        self.assertEqual(solver.set_fluid_controls.call_count, 1)

    def test_slow_clock_is_continuous_and_does_not_change_call_counts(self):
        solver = control_solver()
        for _ in range(7):
            solver.clock.advance_frame()
        previous = solver.clock.time
        solver.set_fluid_controls(gravity=(0., 0., -9.81), feedback_scale=1., slow_motion=True)
        self.assertEqual(solver.clock.time, previous)
        self.assertEqual(solver.clock.schedule.calls, (0, 1, 0, 1))
        self.assertAlmostEqual(solver.clock.schedule.fluid_dt, 1/600)
        for _ in range(4):
            solver.clock.advance()
        self.assertAlmostEqual(solver.clock.time, previous + 1/300)
        solver.set_fluid_controls(gravity=(0., 0., -9.81), feedback_scale=1., slow_motion=False)
        solver.clock.advance_frame()
        self.assertAlmostEqual(solver.clock.time, previous + 1/300 + 1/150)
        self.assertEqual((solver.clock.rigid_ticks, solver.clock.fluid_ticks), (36, 18))
        solver.clock.reset()
        self.assertEqual((solver.clock.time, solver.clock.rigid_ticks), (0., 0))

    def test_captured_feedback_can_be_disabled_and_reenabled_without_recapture(self):
        wp.init()
        if not wp.is_cuda_available():
            self.skipTest("CUDA is required for the captured feedback check")
        solver = control_solver("cuda:0")
        source = wp.array([[1., 2., 3., 4., 5., 6.]], dtype=wp.spatial_vector, device="cuda:0")
        target = wp.zeros(1, dtype=wp.spatial_vector, device="cuda:0")
        wp.load_module(module=_apply_feedback.module.name, device="cuda:0")
        with wp.ScopedStream(wp.Stream("cuda:0")):
            with wp.ScopedCapture() as capture:
                target.zero_()
                wp.launch(_apply_feedback, 1, inputs=[source, solver.feedback_parameter], outputs=[target])
            pointer = solver.feedback_parameter.ptr
            for scale in (0., 2., 1.):
                with patch.object(wp.array, "numpy", side_effect=AssertionError("Control readback")):
                    solver.set_fluid_controls(gravity=(0., 0., -9.81), feedback_scale=scale, slow_motion=False)
                    wp.capture_launch(capture.graph)
                np.testing.assert_allclose(target.numpy()[0], np.arange(1, 7) * scale)
                self.assertEqual(solver.feedback_parameter.ptr, pointer)
                self.assertEqual(solver.graph_revision, 0)

    def test_recapture_preserves_clock_buffers_and_failure_does_not_publish_graph(self):
        solver = control_solver()
        a, b = SimpleNamespace(ptr=1), SimpleNamespace(ptr=2)
        solver.fluid.pre_Pos, solver.fluid.temp_pPos = a, b
        solver.fluid.grid = Mock()
        solver.rigid = SimpleNamespace(_step=20)
        solver.clock.advance_frame()
        def step():
            for _ in range(4):
                for _ in range(solver.clock.schedule.calls[solver.clock.phase]):
                    solver.fluid.pre_Pos, solver.fluid.temp_pPos = solver.fluid.temp_pPos, solver.fluid.pre_Pos
                solver.clock.advance()
                solver.rigid._step += 1
        saved = (solver.clock.time, solver.clock.rigid_ticks, solver.rigid._step)
        with patch.object(wp, "Stream"), patch.object(wp, "ScopedStream"), \
             patch.object(wp, "load_module"), patch.object(wp, "ScopedCapture"), patch.object(wp, "capture_launch"):
            graph = ClockedStepGraph(solver, step)
            solver.set_fluid_controls(gravity=(9.81, 0., 0.), feedback_scale=1., slow_motion=True)
            graph.ensure_current()
            self.assertEqual((solver.clock.time, solver.clock.rigid_ticks, solver.rigid._step), saved)
            self.assertIs(solver.fluid.pre_Pos, a)
            self.assertEqual(graph.rebuild_count, 1)
            graph.ensure_current()
            self.assertEqual(graph.rebuild_count, 1)
            previous = graph.graphs
            def fail():
                step()
                raise RuntimeError("capture probe")
            graph._step = fail
            solver.graph_revision += 1
            with self.assertRaisesRegex(RuntimeError, "capture probe"):
                graph.launch()
            self.assertIs(graph.graphs, previous)
            self.assertEqual((solver.clock.time, solver.clock.rigid_ticks, solver.rigid._step), saved)
            self.assertIs(solver.fluid.pre_Pos, a)


if __name__ == "__main__":
    unittest.main()
