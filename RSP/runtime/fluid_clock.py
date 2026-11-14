"""Code-configured fluid playback relative to each simulation frame.

Edit DEFAULT_SETTINGS before starting the application. All physical parameters
come from the existing WarpLab-compatible PBFConfig; frame_dt is derived from
time_scale, not a second speed multiplier. No runtime UI or hot reconfiguration.
"""
from dataclasses import dataclass, field, replace
import math

from solvers.pbf.pbf_helper import PBFConfig


@dataclass(frozen=True)
class FluidSettings:
    pbf: PBFConfig = field(default_factory=PBFConfig)
    time_scale: float = 0.2
    feedback_scale: float = 1.0
    max_dt: float = 1.0 / 300.0

    def __post_init__(self):
        for name in ("time_scale", "feedback_scale", "max_dt"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if self.max_dt == 0:
            raise ValueError("max_dt must be positive")

    def schedule(self, control_dt, rigid_steps):
        if not math.isfinite(control_dt) or control_dt <= 0 or rigid_steps < 1:
            raise ValueError("A fluid schedule requires positive main timing")
        advance = control_dt * self.time_scale
        count = max(self.pbf.substeps, math.ceil(advance / self.max_dt - 1e-12)) if advance else 0
        # Integer scheduling avoids accumulated rounding drift at frame boundaries.
        calls = tuple(((i + 1) * count) // rigid_steps - (i * count) // rigid_steps
                      for i in range(rigid_steps))
        return FluidSchedule(control_dt, rigid_steps, advance, count,
                             advance / count if count else 0.0, calls)

    def integrator_config(self, schedule):
        # PBFConfig requires positive frame_dt/substeps even while playback is
        # paused. No PBF call is made in that case; the schedule remains zero.
        return replace(self.pbf, frame_dt=schedule.fluid_frame_dt or self.pbf.frame_dt,
                       substeps=schedule.fluid_steps or self.pbf.substeps)


@dataclass(frozen=True)
class FluidSchedule:
    control_dt: float
    rigid_steps: int
    fluid_frame_dt: float
    fluid_steps: int
    fluid_dt: float
    calls: tuple


class FluidClock:
    def __init__(self, schedule):
        self.schedule = schedule
        self.reset()

    def reset(self):
        self.rigid_ticks = 0
        self.fluid_ticks = 0

    @property
    def phase(self):
        return self.rigid_ticks % self.schedule.rigid_steps

    @property
    def time(self):
        return self.fluid_ticks * self.schedule.fluid_dt

    def advance(self):
        self.fluid_ticks += self.schedule.calls[self.phase]
        self.rigid_ticks += 1

    def advance_frame(self):
        self.rigid_ticks += self.schedule.rigid_steps
        self.fluid_ticks += self.schedule.fluid_steps


# PBFConfig exposes every fluid parameter, including radius, density, viscosity,
# vorticity, pressure regularization, initial velocity and block bounds. Supply
# overrides here, e.g. pbf=PBFConfig(substeps=2, viscosity=0.25).
DEFAULT_SETTINGS = FluidSettings(
    pbf=PBFConfig(particle_radius=0.01, substeps=2, pressure_iterations=3,
                  block_end=(1.5, 2.08, 1.25)),
    time_scale=1.0,
)
