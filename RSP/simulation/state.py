"""Own the permanent Newton states shared by the solver and its consumers."""

import newton
import warp as wp


class NewtonStatePair:
    """Keep the primary state alive across every step and reset."""

    def __init__(self, model):
        self.model = model
        self.raw_state = model.state()
        self.eval_fk()
        self.swap_state = model.state()

    def eval_fk(self):
        newton.eval_fk(self.model, self.raw_state.joint_q, self.raw_state.joint_qd, self.raw_state)
        return

    def copy(self, raw_state):
        wp.copy(self.raw_state.joint_q, raw_state.joint_q)
        wp.copy(self.raw_state.joint_qd, raw_state.joint_qd)
        wp.copy(self.raw_state.body_q, raw_state.body_q)
        wp.copy(self.raw_state.body_qd, raw_state.body_qd)
        wp.copy(self.raw_state.body_f, raw_state.body_f)
        wp.copy(self.raw_state.particle_q, raw_state.particle_q)
        wp.copy(self.raw_state.particle_qd, raw_state.particle_qd)
        return

    def synchronize_buffers(self):
        self.swap_state.assign(self.raw_state)
