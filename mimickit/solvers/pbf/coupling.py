"""Pressure coupling on shared Newton states; PBF retains its particle order."""
import newton
import warp as wp

from .PBF import PBF


@wp.kernel
def add_boundary_forces(forces: wp.array(dtype=wp.spatial_vector),
                        body_f: wp.array(dtype=wp.spatial_vector)):
    index = wp.tid()
    body_f[index] += forces[index]


class CoupledStepGraph:
    """Captured full-engine steps with this solver pair's host-side bookkeeping.

    The runtime owns this handle and supplies its complete step as a capture-only
    callback. The callback is not retained. Buffer addresses, PBF's prediction
    swaps and MuJoCo's uncaptured Python counter stay inside the solver adapter.
    Capturing restores host state even on failure; replay commits it once.
    """

    def __init__(self, solver, step, substeps, *, fluid_substeps=None):
        self._solver = solver
        self._substeps = substeps
        fluid_substeps = substeps if fluid_substeps is None else fluid_substeps
        self._graphs = {}
        fluid = solver.fluid
        for module in ("solvers.pbf.pbf_helper", "solvers.pbf.PBF",
                       "solvers.pbf.boundary_model", "solvers.pbf.coupling"):
            wp.load_module(module=module, device=solver.model.device)
        fluid.grid.reserve(solver.model.particle_count)
        prediction = (fluid.pre_Pos, fluid.temp_pPos)
        rigid_step = solver.rigid._step
        # Odd numbers of PBF substeps alternate the starting buffer address.
        starts = [prediction] if fluid_substeps % 2 == 0 else [prediction, prediction[::-1]]
        try:
            for buffers in starts:
                fluid.pre_Pos, fluid.temp_pPos = buffers
                with wp.ScopedCapture(device=solver.model.device) as capture:
                    step()
                self._graphs[buffers[0].ptr] = (capture.graph, (fluid.pre_Pos, fluid.temp_pPos))
        finally:
            fluid.pre_Pos, fluid.temp_pPos = prediction
            solver.rigid._step = rigid_step

    def launch(self):
        fluid = self._solver.fluid
        graph, prediction = self._graphs[fluid.pre_Pos.ptr]
        wp.capture_launch(graph)
        fluid.pre_Pos, fluid.temp_pPos = prediction
        self._solver.rigid._step += self._substeps


class SolverPBFMuJoCo(newton.solvers.SolverBase):
    """Compose the existing solvers, forwarding only rigid-body observables."""
    ObservableKind = newton.solvers.SolverMuJoCo.ObservableKind

    def __init__(self, model, *, config, boundaries, rigid, verbose=False):
        super().__init__(model)
        self.boundaries = boundaries
        self.rigid = rigid
        self.fluid = PBF(model, config=config, boundaries=boundaries, verbose=verbose)
        self._prediction_buffers = (self.fluid.pre_Pos, self.fluid.temp_pPos)

    @property
    def boundary_forces(self):
        return self.fluid.boundary_forces

    def get_max_contact_count(self):
        return self.rigid.get_max_contact_count()

    def observables(self, kinds, *, contacts=None, requires_grad=None):
        return self.rigid.observables(kinds, contacts=contacts, requires_grad=requires_grad)

    def capture_step_graph(self, step, substeps, *, fluid_substeps=None):
        """Capture the caller's full step without exposing solver-private state."""
        return CoupledStepGraph(self, step, substeps, fluid_substeps=fluid_substeps)

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None,
             advance_fluid=True, fluid_dt=None):
        # The engine has already cleared forces and applied user inputs.
        self.boundaries.sync_body_poses(state_in)
        if advance_fluid:
            self.boundary_forces.zero_()
            self.fluid.step(state_in, state_out, None, None, dt if fluid_dt is None else fluid_dt)
        else:
            # MuJoCo advances rigid state only; carry the latest particles across
            # its ping-pong States without changing PBF's own prediction buffers.
            wp.copy(state_out.particle_q, state_in.particle_q)
            wp.copy(state_out.particle_qd, state_in.particle_qd)
        # Hold the pressure force (not an impulse) until the next fluid update.
        # Summing F * rigid_dt over the group then gives F * fluid_dt.
        wp.launch(add_boundary_forces, self.model.body_count,
                  inputs=[self.boundary_forces], outputs=[state_in.body_f], device=self.model.device)
        self.rigid.step(state_in, state_out, control, contacts, dt, observables=observables)
        self.boundaries.sync_body_poses(state_out)

    def reset(self, state, world_mask=None, flags=None):
        if world_mask is not None or flags is not None:
            raise NotImplementedError("The PBF play scene resets one complete world")
        # GSI and the environment own the joint reset. Clear MuJoCo caches
        # without replacing their freshly sampled positions and velocities.
        self.rigid.reset(state, flags=0)
        self.fluid.pre_Pos, self.fluid.temp_pPos = self._prediction_buffers
        self.fluid.reset(state)
        state.clear_forces()
        newton.eval_fk(self.model, state.joint_q, state.joint_qd, state)
        self.boundaries.sync_body_poses(state)

    def notify_model_changed(self, flags):
        self.fluid.notify_model_changed(flags)
        self.rigid.notify_model_changed(flags)
