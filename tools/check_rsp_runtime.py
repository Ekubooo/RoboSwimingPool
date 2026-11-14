"""Exercise clock presets with real PBF/Newton states in separate processes.

Reuses the backend checker (fixed commands, force input, reset, stable views).
Use --profile default|two|paused|fast and --eager; checks cover both clocks.
"""
import argparse
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import check_newton_pbf_refactor as checks
from RSP.runtime import fluid_clock
from RSP.runtime.coupling import ClockedPBFMuJoCo, ClockedStepGraph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("default", "two", "paused", "fast"), default="default")
    parser.add_argument("--eager", action="store_true")
    parser.add_argument("--name", help="Artifact name for an independent repeat")
    args = parser.parse_args()
    settings = fluid_clock.DEFAULT_SETTINGS
    if args.profile == "two":
        settings = replace(settings, pbf=replace(settings.pbf, substeps=2))
    elif args.profile == "paused":
        settings = replace(settings, time_scale=0.)
    elif args.profile == "fast":
        settings = replace(settings, time_scale=1.)
    fluid_clock.DEFAULT_SETTINGS = settings
    step, launch = ClockedPBFMuJoCo.step, ClockedStepGraph.launch

    def check_clock(solver):
        clock = solver.clock
        frames, phase = divmod(clock.rigid_ticks, clock.schedule.rigid_steps)
        expected = frames * clock.schedule.fluid_steps + sum(clock.schedule.calls[:phase])
        assert clock.fluid_ticks == expected

    def checked_step(self, *positional, **kwargs):
        result = step(self, *positional, **kwargs)
        check_clock(self)
        return result

    def checked_launch(self):
        result = launch(self)
        check_clock(self.solver)
        return result

    name = args.name or f"stage3_{args.profile}_{'eager' if args.eager else 'graph'}"
    with patch.object(ClockedPBFMuJoCo, "step", checked_step), patch.object(ClockedStepGraph, "launch", checked_launch):
        checks.snapshot(SimpleNamespace(revision="after", graph=not args.eager, substeps=8, name=name))
    if args.profile == "paused":
        import numpy as np
        states = np.load(checks.OUTPUT / f"{name}.npz")
        for attr in ("particle_q", "particle_qd"):
            for label in ("step0", "step5", "resumed1"):
                np.testing.assert_array_equal(states[f"initial/raw/{attr}"], states[f"{label}/raw/{attr}"])
    print(f"CLOCK PASS: {name}", flush=True)


if __name__ == "__main__":
    main()
