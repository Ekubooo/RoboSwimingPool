"""Convert MimicKit commands to Newton control and force buffers."""

import newton
import warp as wp

import engines.engine as engine
from .state_views import (
    ArticulationLayout,
    dof_positions_to_joint_q,
    joint_q_to_dof_positions,
    rotate_dof_vectors,
)


class CommandBuffers:
    """Own Newton control and the MimicKit command arrays; perform no actuation."""

    def __init__(self, sim_model, num_envs, layout=None):
        if not sim_model.use_coord_layout_targets:
            raise ValueError("NewtonEngine requires coordinate-layout joint targets.")
        self.control = sim_model.control()

        # Keep MimicKit commands in DOF/exp-map space; Newton 1.7 uses q space.
        self.target_dof_pos = wp.zeros_like(self.control.joint_target_qd)
        self.target_dof_vel = wp.zeros_like(self.control.joint_target_qd)
        self.dof_force = wp.zeros_like(self.control.joint_f)
        self._torch_target_pos = wp.to_torch(self.target_dof_pos)
        self._torch_target_vel = wp.to_torch(self.target_dof_vel)
        self._torch_joint_force = wp.to_torch(self.dof_force)

        target_pos = self._torch_target_pos.view(num_envs, -1)
        target_vel = self._torch_target_vel.view(num_envs, -1)
        joint_force = self._torch_joint_force.view(num_envs, -1)

        self.layout = layout if layout is not None else ArticulationLayout(sim_model, num_envs)

        self.target_pos = []
        self.target_vel = []
        self.joint_force = []

        for obj_id in range(self.layout.objs_per_env):
            dofs = self.layout.dof_slices[obj_id]
            obj_target_pos = target_pos[:, dofs]
            obj_target_vel = target_vel[:, dofs]
            obj_joint_force = joint_force[:, dofs]

            self.target_pos.append(obj_target_pos)
            self.target_vel.append(obj_target_vel)
            self.joint_force.append(obj_joint_force)

        return


class Actuation:
    """Convert commands and apply forces using the existing control/state buffers."""

    def __init__(self, model, state, mode, num_envs):
        self.model = model
        self.state = state
        self.layout = state.layout
        self.mode = mode
        self._num_envs = num_envs
        self._build_controls()

    def initialize_force_buffer(self):
        self.has_body_forces = wp.array([0], dtype=wp.int32, device=self.model.device)

    def _build_controls(self):
        kp = self.model.mujoco.dof_passive_stiffness
        kd = self.model.joint_damping

        if (self.mode == engine.ControlMode.none):
            self.model.joint_target_ke.fill_(0.0)
            self.model.joint_target_kd.fill_(0.0)
            self.model.joint_target_mode.fill_(int(newton.JointTargetMode.NONE))

        elif (self.mode == engine.ControlMode.pos):
            wp.copy(self.model.joint_target_ke, kp)
            wp.copy(self.model.joint_target_kd, kd)
            self.model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION))

        elif (self.mode == engine.ControlMode.vel):
            self.model.joint_target_ke.fill_(0.0)
            wp.copy(self.model.joint_target_kd, kd)
            self.model.joint_target_mode.fill_(int(newton.JointTargetMode.VELOCITY))

        elif (self.mode == engine.ControlMode.torque):
            self.model.joint_target_ke.fill_(0.0)
            self.model.joint_target_kd.fill_(0.0)
            self.model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        elif (self.mode == engine.ControlMode.pd_explicit):
            self.model.joint_target_ke.fill_(0.0)
            self.model.joint_target_kd.fill_(0.0)
            self._kp_raw = wp.clone(kp)
            self._kd_raw = wp.clone(kd)
            self._torque_lim_raw = wp.clone(self.model.joint_effort_limit)
            self._pd_dof_pos = wp.zeros_like(self.state.raw_state.joint_qd)
            self._pd_dof_vel = wp.zeros_like(self.state.raw_state.joint_qd)
            self.model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        else:
            assert(False), "Unsupported control mode: {}".format(self.mode)

        self.model.mujoco.dof_passive_stiffness.fill_(0.0)
        self.model.joint_damping.fill_(0.0)

        # The root belongs to the state, not the object's command vector.
        modes = self.model.joint_target_mode.numpy()
        for root_dofs in self.layout.root_dof_slices:
            modes[root_dofs] = int(newton.JointTargetMode.NONE)
        self.model.joint_target_mode.assign(modes)

        num_envs = self._num_envs
        self.controls = CommandBuffers(self.model, num_envs, layout=self.layout)
        return

    def set_cmd(self, obj_id, cmd):
        if (self.mode == engine.ControlMode.none):
            pass
        elif (self.mode == engine.ControlMode.pos):
            self.controls.target_pos[obj_id][:] = cmd
        elif (self.mode == engine.ControlMode.vel):
            self.controls.target_vel[obj_id][:] = cmd
        elif (self.mode == engine.ControlMode.torque):
            self.controls.joint_force[obj_id][:] = cmd
        elif (self.mode == engine.ControlMode.pd_explicit):
            self.controls.target_pos[obj_id][:] = cmd
        else:
            assert(False), "Unsupported control mode: {}".format(self.mode)
        return

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None or len(env_id) > 0):
            assert(len(forces.shape) == 2)
            self.has_body_forces.fill_(1)

            if (env_id is None):
                self.state.body_force[obj_id][:, body_id, :3] = forces
            else:
                self.state.body_force[obj_id][env_id, body_id, :3] = forces
        return

    def update_targets(self):
        if self.mode != engine.ControlMode.pos:
            return
        dof_positions_to_joint_q(self.layout, self.controls.target_dof_pos,
                                 self.controls.control.joint_target_q)

    def _apply_pd_explicit_torque(self, sim_state, control):
        wp.copy(self._pd_dof_vel, sim_state.joint_qd)
        rotate_dof_vectors(self.layout, sim_state.joint_q, sim_state.joint_qd,
                           self._pd_dof_vel, to_newton=False)
        joint_q_to_dof_positions(self.layout, sim_state.joint_q, self._pd_dof_pos)
        wp.launch(apply_pd_torque, dim=self.model.joint_dof_count,
                  inputs=[self.controls.target_dof_pos, self._pd_dof_pos, self._pd_dof_vel,
                          self._kp_raw, self._kd_raw, self._torque_lim_raw,
                          self.model.joint_target_mode, self.controls.dof_force],
                  device=self.model.device)
        wp.copy(control.joint_f, self.controls.dof_force)
        rotate_dof_vectors(self.layout, sim_state.joint_q, self.controls.dof_force,
                           control.joint_f, to_newton=True)
        return

    def apply_control(self, raw_state, control):
        if (self.mode == engine.ControlMode.pd_explicit):
            self._apply_pd_explicit_torque(raw_state, control)
        elif self.mode == engine.ControlMode.vel:
            wp.copy(control.joint_target_qd, self.controls.target_dof_vel)
            rotate_dof_vectors(self.layout, raw_state.joint_q, self.controls.target_dof_vel,
                               control.joint_target_qd, to_newton=True)
        elif self.mode == engine.ControlMode.torque:
            wp.copy(control.joint_f, self.controls.dof_force)
            rotate_dof_vectors(self.layout, raw_state.joint_q, self.controls.dof_force,
                               control.joint_f, to_newton=True)
        return

    def apply_external_forces(self, raw_state):
        def apply_body_force():
            self.state.copy_external_forces_to(raw_state)
            return
        wp.capture_if(self.has_body_forces, on_true=apply_body_force)

    def clear_external_forces(self):
        self.state.clear_forces()
        self.has_body_forces.zero_()

    def finish_step(self):
        wp.capture_if(self.has_body_forces, on_true=self.clear_external_forces)


@wp.kernel
def apply_pd_torque(target: wp.array(dtype=float), pos: wp.array(dtype=float),
                    vel: wp.array(dtype=float), kp: wp.array(dtype=float),
                    kd: wp.array(dtype=float), limit: wp.array(dtype=float),
                    mode: wp.array(dtype=int), force: wp.array(dtype=float)):
    i = wp.tid()
    force[i] = 0.0
    if mode[i] != 0:
        torque = kp[i] * (target[i] - pos[i]) - kd[i] * vel[i]
        force[i] = wp.clamp(torque, -limit[i], limit[i])
