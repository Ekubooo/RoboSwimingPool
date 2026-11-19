"""Multirate playback adapter; numerical PBF and MuJoCo solvers are unchanged."""
from dataclasses import replace
import math
import time
from types import SimpleNamespace

import newton
import warp as wp

from .pbf.coupling import SolverPBFMuJoCo
from .fluid_clock import FluidClock


@wp.kernel
def _apply_feedback(source: wp.array(dtype=wp.spatial_vector), scale: wp.array(dtype=float),
                    target: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    target[i] += source[i] * scale[0]


class ClockedPBFMuJoCo(SolverPBFMuJoCo):
    """Keep the rigid SolverBase contract, replacing only coupled scheduling.

The existing Newton State pair remains authoritative. Skipped fluid updates
copy particles on-device into the outgoing state. One persistent scratch pair
supports multiple fluid steps within a rigid step without modifying the input.
The PBF force buffer itself holds the most recently computed feedback.
"""
    def __init__(self, model, *, config, boundaries, rigid, settings, schedule, verbose=False):
        super().__init__(model, config=config, boundaries=boundaries, rigid=rigid, verbose=verbose)
        self.settings = settings
        self.clock = FluidClock(schedule)
        self._base_schedule = schedule
        self._gravity = tuple(config.gravity)
        self._slow_motion = False
        self._feedback_value = settings.feedback_scale
        self.feedback_parameter = wp.full(1, settings.feedback_scale, dtype=float, device=model.device)
        self.graph_revision = 0
        self.fluid.gravity = wp.vec3(*self._gravity)
        self._scratch = None
        if max(schedule.calls) > 1:
            self._scratch = SimpleNamespace(
                particle_q=wp.empty(model.particle_count, dtype=wp.vec3, device=model.device),
                particle_qd=wp.empty(model.particle_count, dtype=wp.vec3, device=model.device))

    def set_fluid_controls(self, *, gravity, feedback_scale, slow_motion):
        """Apply a frame-boundary edit; never change the shared model's gravity.

        The scale buffer stays at the same address across captures. Gravity and
        dt are kernel values, so their edits invalidate the recorded graph.
        """
        gravity = tuple(float(value) for value in gravity)
        if len(gravity) != 3 or not all(math.isfinite(value) for value in gravity):
            raise ValueError("Fluid gravity must contain three finite components")
        if not math.isfinite(feedback_scale) or feedback_scale < 0:
            raise ValueError("Fluid feedback scale must be finite and nonnegative")
        if self.clock.phase:
            raise RuntimeError("Fluid controls must be applied between complete frames")
        changed = gravity != self._gravity or bool(slow_motion) != self._slow_motion
        if changed:
            self._gravity, self._slow_motion = gravity, bool(slow_motion)
            speed = .5 if self._slow_motion else 1.0
            self.clock.schedule = replace(self._base_schedule,
                fluid_dt=self._base_schedule.fluid_dt * speed,
                fluid_frame_dt=self._base_schedule.fluid_frame_dt * speed)
            self.fluid.gravity = wp.vec3(*gravity)
            self.graph_revision += 1
        if feedback_scale != self._feedback_value:
            self.feedback_parameter.fill_(float(feedback_scale))
            self._feedback_value = feedback_scale

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        schedule = self.clock.schedule
        if abs(dt - schedule.control_dt / schedule.rigid_steps) > 1e-10:
            raise ValueError("Rigid timestep changed without rebuilding the fluid schedule")
        count = schedule.calls[self.clock.phase]
        self.boundaries.sync_body_poses(state_in)
        source = state_in
        if count:
            for index in range(count):
                # Alternation always ends in the engine's outgoing particle state.
                target = state_out if (count - index) % 2 else self._scratch
                self.boundary_forces.zero_()
                self.fluid.step(source, target, None, None, schedule.fluid_dt)
                source = target
        else:
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        if schedule.fluid_steps:
            wp.launch(_apply_feedback, self.model.body_count,
                      inputs=[self.boundary_forces, self.feedback_parameter],
                      outputs=[state_in.body_f], device=self.model.device)
        self.rigid.step(state_in, state_out, control, contacts, dt, observables=observables)
        self.boundaries.sync_body_poses(state_out)
        self.clock.advance()

    def reset(self, state, world_mask=None, flags=None):
        super().reset(state, world_mask, flags)
        self.clock.reset()

    def notify_model_changed(self, flags):
        super().notify_model_changed(flags)
        # Fluid-only configuration must never change the shared rigid gravity.
        self.fluid.gravity = wp.vec3(*self._gravity)
        self.graph_revision += 1

    def capture_step_graph(self, step, substeps):
        if substeps != self.clock.schedule.rigid_steps or self.clock.phase:
            raise ValueError("Capture must start at a complete control-frame boundary")
        return ClockedStepGraph(self, step)


class ClockedStepGraph:
    """Capture particle parity independently of MuJoCo's rigid-step counter."""
    def __init__(self, solver, step):
        self.solver = solver
        self._step = step
        # Playback may run on PyTorch's default CUDA stream, which cannot be
        # captured. A Warp-owned stream also isolates lazy setup from capture.
        self._capture_stream = wp.Stream(solver.model.device)
        self.graphs = {}
        self._capture()
        self.rebuild_count = 0
        self.last_rebuild_ms = 0.0

    def _capture(self):
        solver = self.solver
        fluid, clock = solver.fluid, solver.clock
        if clock.phase:
            raise RuntimeError("Graph capture requires a complete frame boundary")
        for module in ("RSP.physics.pbf.pbf_helper", "RSP.physics.pbf.PBF",
                       "RSP.physics.pbf.boundary_model", __name__):
            wp.load_module(module=module, device=solver.model.device)
        fluid.grid.reserve(solver.model.particle_count)
        prediction = (fluid.pre_Pos, fluid.temp_pPos)
        rigid_step = solver.rigid._step
        ticks = (clock.rigid_ticks, clock.fluid_ticks, clock.elapsed)
        starts = [prediction] if clock.schedule.fluid_steps % 2 == 0 else [prediction, prediction[::-1]]
        graphs = {}
        try:
            for buffers in starts:
                fluid.pre_Pos, fluid.temp_pPos = buffers
                clock.rigid_ticks, clock.fluid_ticks, clock.elapsed = ticks
                solver.rigid._step = rigid_step
                with wp.ScopedStream(self._capture_stream):
                    with wp.ScopedCapture(device=solver.model.device) as capture:
                        self._step()
                graphs[buffers[0].ptr] = (capture.graph, (fluid.pre_Pos, fluid.temp_pPos))
        finally:
            fluid.pre_Pos, fluid.temp_pPos = prediction
            solver.rigid._step = rigid_step
            clock.rigid_ticks, clock.fluid_ticks, clock.elapsed = ticks
        self.graphs = graphs
        self._revision = solver.graph_revision

    def ensure_current(self):
        """Recapture once per edit batch, leaving the previous graph on failure."""
        if self._revision != self.solver.graph_revision:
            start = time.perf_counter()
            self._capture()
            self.last_rebuild_ms = (time.perf_counter() - start) * 1000
            self.rebuild_count += 1

    def launch(self):
        self.ensure_current()
        fluid = self.solver.fluid
        graph, prediction = self.graphs[fluid.pre_Pos.ptr]
        wp.capture_launch(graph)
        fluid.pre_Pos, fluid.temp_pPos = prediction
        self.solver.rigid._step += self.solver.clock.schedule.rigid_steps
        self.solver.clock.advance_frame()
