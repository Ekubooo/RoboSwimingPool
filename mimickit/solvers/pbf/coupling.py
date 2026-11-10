"""Pressure coupling on shared Newton states; PBF retains its particle order."""
import newton
import warp as wp

from .PBF import PBF


@wp.kernel
def add_boundary_forces(forces: wp.array(dtype=wp.spatial_vector),
                        body_f: wp.array(dtype=wp.spatial_vector)):
    index = wp.tid()
    body_f[index] += forces[index]


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

    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        # The engine has already cleared forces and applied user inputs.
        self.boundaries.sync_body_poses(state_in)
        self.boundary_forces.zero_()
        self.fluid.step(state_in, state_out, None, None, dt)
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
