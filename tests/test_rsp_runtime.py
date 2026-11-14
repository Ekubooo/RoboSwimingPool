"""Clock arithmetic, force hold and capture bookkeeping, independent of assets."""
from dataclasses import replace
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
wp.config.kernel_cache_dir = str(ROOT / "output/newton_pbf_refactor_checks/warp_cache")

from RSP.runtime.fluid_clock import FluidSettings, FluidClock
from RSP.runtime.coupling import ClockedPBFMuJoCo, ClockedStepGraph


class ClockChecks(unittest.TestCase):
    def test_default_two_steps_and_maximum_dt(self):
        settings = FluidSettings()
        schedule = settings.schedule(1/30, 8)
        self.assertEqual(schedule.calls, (0, 0, 1, 0, 0, 1, 0, 1))
        self.assertAlmostEqual(schedule.fluid_dt, 1/450)
        two = replace(settings, pbf=replace(settings.pbf, substeps=2)).schedule(1/30, 8)
        self.assertEqual(two.calls, (0, 0, 0, 1, 0, 0, 0, 1))
        self.assertAlmostEqual(two.fluid_dt, 1/300)
        fast = replace(settings, time_scale=1).schedule(1/30, 8)
        self.assertEqual(fast.fluid_steps, 10)
        self.assertEqual(sum(fast.calls), 10)
        self.assertLessEqual(fast.fluid_dt, settings.max_dt)

    def test_pause_reset_and_long_clock(self):
        settings = FluidSettings()
        clock = FluidClock(settings.schedule(1/30, 8))
        for _ in range(8 * 1800):
            clock.advance()
        self.assertEqual(clock.rigid_ticks, 14400)
        self.assertAlmostEqual(clock.time, 12.)
        clock.reset()
        self.assertEqual((clock.phase, clock.time), (0, 0.))
        paused = FluidClock(replace(settings, time_scale=0).schedule(1/30, 8))
        paused.advance_frame()
        self.assertEqual((paused.rigid_ticks, paused.fluid_ticks, paused.time), (8, 0, 0.))

    def test_invalid_settings(self):
        for name, value in (("time_scale", -1), ("time_scale", float("nan")),
                            ("feedback_scale", -1), ("max_dt", 0)):
            with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                replace(FluidSettings(), **{name: value})

    def test_capture_parity_and_restore_on_failure(self):
        for count in (0, 2, 3, 8, 10):
            settings = FluidSettings(time_scale=0 if count == 0 else 0.2,
                pbf=replace(FluidSettings().pbf, substeps=max(count, 1)))
            clock = FluidClock(settings.schedule(1/30, 8))
            a, b = SimpleNamespace(ptr=1), SimpleNamespace(ptr=2)
            fluid = SimpleNamespace(pre_Pos=a, temp_pPos=b, grid=Mock())
            solver = SimpleNamespace(fluid=fluid, clock=clock, rigid=SimpleNamespace(_step=4),
                                     model=SimpleNamespace(device="cuda:0", particle_count=1))

            def step():
                for _ in range(8):
                    for _ in range(clock.schedule.calls[clock.phase]):
                        fluid.pre_Pos, fluid.temp_pPos = fluid.temp_pPos, fluid.pre_Pos
                    solver.rigid._step += 1
                    clock.advance()

            with patch.object(wp, "load_module"), patch.object(wp, "ScopedCapture"), patch.object(wp, "capture_launch"):
                graph = ClockedStepGraph(solver, step)
                self.assertEqual((clock.rigid_ticks, clock.fluid_ticks, solver.rigid._step), (0, 0, 4))
                graph.launch()
                self.assertIs(fluid.pre_Pos, (a, b)[count % 2])
                self.assertEqual((clock.rigid_ticks, clock.fluid_ticks, solver.rigid._step), (8, count, 12))
                graph.launch()
                self.assertIs(fluid.pre_Pos, a)
                saved = (clock.rigid_ticks, clock.fluid_ticks, solver.rigid._step)

                def fail():
                    step()
                    raise RuntimeError("capture failed")

                with self.assertRaisesRegex(RuntimeError, "capture failed"):
                    ClockedStepGraph(solver, fail)
                self.assertEqual((clock.rigid_ticks, clock.fluid_ticks, solver.rigid._step), saved)
                self.assertIs(fluid.pre_Pos, a)


class FeedbackChecks(unittest.TestCase):
    def test_skipped_state_force_hold_scale_and_pause(self):
        wp.init()
        for scale, speed in ((1., .2), (2., .2), (0., .2), (1., 0.)):
            settings = FluidSettings(feedback_scale=scale, time_scale=speed)
            states = [SimpleNamespace(particle_q=wp.full(1, wp.vec3(1, 2, 3), dtype=wp.vec3),
                        particle_qd=wp.full(1, wp.vec3(4, 5, 6), dtype=wp.vec3),
                        body_f=wp.zeros(1, dtype=wp.spatial_vector)) for _ in range(2)]
            forces = wp.full(1, wp.spatial_vector(1, 2, 3, 4, 5, 6), dtype=wp.spatial_vector)
            solver = SimpleNamespace(clock=FluidClock(settings.schedule(1/30, 8)), settings=settings,
                boundaries=SimpleNamespace(sync_body_poses=Mock()), boundary_forces=forces,
                model=SimpleNamespace(device=forces.device, body_count=1), rigid=SimpleNamespace(step=Mock()))
            # First two rigid ticks deliberately perform no fluid solve.
            with patch.object(wp.array, "numpy", side_effect=AssertionError("Host readback")):
                ClockedPBFMuJoCo.step(solver, *states, None, None, 1/240)
            np.testing.assert_equal(states[1].particle_q.numpy(), [[1, 2, 3]])
            np.testing.assert_equal(states[1].particle_qd.numpy(), [[4, 5, 6]])
            expected = np.arange(1, 7) * (scale if speed else 0)
            np.testing.assert_equal(states[0].body_f.numpy()[0], expected)
            self.assertEqual(solver.rigid.step.call_args.args[4], 1/240)


if __name__ == "__main__":
    unittest.main()
