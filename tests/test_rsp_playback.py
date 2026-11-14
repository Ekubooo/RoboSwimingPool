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
        env = SimpleNamespace(_engine=SimpleNamespace(get_viewer=lambda: viewer),
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

    def test_new_defaults_preserve_reference_config(self):
        reference = FluidSettings()
        self.assertEqual((reference.pbf.particle_radius, reference.pbf.substeps), (.0125, 3))
        actual = DEFAULT_SETTINGS.schedule(1/30, 4)
        self.assertEqual(actual.calls, (0, 1, 0, 1))
        self.assertAlmostEqual(actual.fluid_frame_dt, 1/150)
        self.assertAlmostEqual(actual.fluid_dt, 1/300)
        self.assertEqual(DEFAULT_SETTINGS.pbf.pressure_iterations, 3)
        config = runtime_config(1/30, 4)
        particles = SimpleNamespace(config=config)
        init_scene_parameters(particles)
        self.assertEqual(particles.n, 301392)
        self.assertEqual(tuple(config.block_start), (.1, .75, .5))
        self.assertEqual(tuple(config.block_end), (1.5, 1.9, 2.08))


if __name__ == "__main__":
    unittest.main()
