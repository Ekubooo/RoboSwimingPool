"""Multirate playback adapter; numerical PBF and MuJoCo solvers are unchanged."""
from types import SimpleNamespace

import newton
import warp as wp

from solvers.pbf.coupling import SolverPBFMuJoCo
from .fluid_clock import FluidClock


@wp.kernel
def _apply_feedback(source: wp.array(dtype=wp.spatial_vector), scale: float,
                    target: wp.array(dtype=wp.spatial_vector)):
    i = wp.tid()
    target[i] += source[i] * scale


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
        self.fluid.gravity = wp.vec3(*config.gravity)
        self._scratch = None
        if max(schedule.calls) > 1:
            self._scratch = SimpleNamespace(
                particle_q=wp.empty(model.particle_count, dtype=wp.vec3, device=model.device),
                particle_qd=wp.empty(model.particle_count, dtype=wp.vec3, device=model.device))

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
        if schedule.fluid_steps and self.settings.feedback_scale:
            wp.launch(_apply_feedback, self.model.body_count,
                      inputs=[self.boundary_forces, self.settings.feedback_scale],
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
        self.fluid.gravity = wp.vec3(*self.fluid.config.gravity)

    def capture_step_graph(self, step, substeps):
        if substeps != self.clock.schedule.rigid_steps or self.clock.phase:
            raise ValueError("Capture must start at a complete control-frame boundary")
        return ClockedStepGraph(self, step)


class ClockedStepGraph:
    """Capture particle parity independently of MuJoCo's rigid-step counter."""
    def __init__(self, solver, step):
        self.solver = solver
        self.graphs = {}
        fluid, clock = solver.fluid, solver.clock
        for module in ("solvers.pbf.pbf_helper", "solvers.pbf.PBF",
                       "solvers.pbf.boundary_model", __name__):
            wp.load_module(module=module, device=solver.model.device)
        fluid.grid.reserve(solver.model.particle_count)
        prediction = (fluid.pre_Pos, fluid.temp_pPos)
        rigid_step = solver.rigid._step
        ticks = (clock.rigid_ticks, clock.fluid_ticks)
        starts = [prediction] if clock.schedule.fluid_steps % 2 == 0 else [prediction, prediction[::-1]]
        try:
            for buffers in starts:
                fluid.pre_Pos, fluid.temp_pPos = buffers
                clock.rigid_ticks, clock.fluid_ticks = ticks
                solver.rigid._step = rigid_step
                with wp.ScopedCapture(device=solver.model.device) as capture:
                    step()
                self.graphs[buffers[0].ptr] = (capture.graph, (fluid.pre_Pos, fluid.temp_pPos))
        finally:
            fluid.pre_Pos, fluid.temp_pPos = prediction
            solver.rigid._step = rigid_step
            clock.rigid_ticks, clock.fluid_ticks = ticks

    def launch(self):
        fluid = self.solver.fluid
        graph, prediction = self.graphs[fluid.pre_Pos.ptr]
        wp.capture_launch(graph)
        fluid.pre_Pos, fluid.temp_pPos = prediction
        self.solver.rigid._step += self.solver.clock.schedule.rigid_steps
        self.solver.clock.advance_frame()
