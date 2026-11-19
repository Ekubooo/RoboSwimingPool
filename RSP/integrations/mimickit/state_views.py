"""MimicKit's object views and conversions for borrowed Newton buffers."""

import os

import newton
import torch
import warp as wp

from RSP.simulation.state import NewtonStatePair


class ArticulationLayout:
    """Shared, construction-time topology for MimicKit's object/batch views.

    This backend imports one joint per body and repeats the same object layout
    in every world. Slices cover all worlds; batched Torch views use the first
    world's slices. Only topology is cached here, never mutable gains or masses.
    The Warp indices belong to this layout and are borrowed by state/actuation.
    """

    def __init__(self, model, num_envs):
        self.device = model.device
        self.num_envs = num_envs
        self.objs_per_env = int(model.articulation_count / num_envs)
        self.articulation_start = model.articulation_start.numpy()
        self.joint_q_start = model.joint_q_start.numpy()
        self.joint_qd_start = model.joint_qd_start.numpy()
        joint_types = model.joint_type.numpy()
        joint_parent = model.joint_parent.numpy()

        starts = self.articulation_start
        q_start = self.joint_q_start
        qd_start = self.joint_qd_start
        self.body_slices = tuple(slice(int(a), int(b)) for a, b in zip(starts[:-1], starts[1:]))
        self.q_slices = tuple(slice(int(q_start[a + 1]), int(q_start[b]))
                              for a, b in zip(starts[:-1], starts[1:]))
        self.dof_slices = tuple(slice(int(qd_start[a + 1]), int(qd_start[b]))
                                for a, b in zip(starts[:-1], starts[1:]))
        self.root_q_starts = tuple(int(q_start[a]) for a in starts[:-1])
        self.root_dof_slices = tuple(slice(int(qd_start[a]), int(qd_start[a + 1])) for a in starts[:-1])
        self.fixed_roots = tuple(joint_types[a] == newton.JointType.FIXED for a in starts[:-1])

        sphere_q_start, sphere_dof_start = [], []
        nonsphere_q_idx, nonsphere_dof_idx = [], []
        for j in range(model.joint_count):
            if joint_parent[j] == -1:
                continue
            if joint_types[j] == newton.JointType.BALL:
                assert q_start[j + 1] - q_start[j] == 4
                assert qd_start[j + 1] - qd_start[j] == 3
                sphere_q_start.append(q_start[j])
                sphere_dof_start.append(qd_start[j])
            else:
                nonsphere_q_idx.extend(range(q_start[j], q_start[j + 1]))
                nonsphere_dof_idx.extend(range(qd_start[j], qd_start[j + 1]))
        self.sphere_q_start = wp.array(sphere_q_start, device=self.device, dtype=int)
        self.sphere_dof_start = wp.array(sphere_dof_start, device=self.device, dtype=int)
        self.nonsphere_q_idx = wp.array(nonsphere_q_idx, device=self.device, dtype=int)
        self.nonsphere_dof_idx = wp.array(nonsphere_dof_idx, device=self.device, dtype=int)
        self.has_spherical_joints = bool(sphere_q_start)

    def object_index(self, env_id, obj_id):
        return env_id * self.objs_per_env + obj_id


def dof_positions_to_joint_q(layout, source, target):
    """Write non-root MimicKit scalar/exp-map positions to Newton coordinates."""
    wp.launch(copy_indexed, dim=layout.nonsphere_dof_idx.shape[0],
              inputs=[source, layout.nonsphere_dof_idx, layout.nonsphere_q_idx, target],
              device=layout.device)
    if layout.has_spherical_joints:
        wp.launch(exp_map_to_quat_indexed, dim=layout.sphere_dof_start.shape[0],
                  inputs=[source, layout.sphere_dof_start, layout.sphere_q_start, target],
                  device=layout.device)


def joint_q_to_dof_positions(layout, source, target):
    """Read non-root Newton scalar/quaternion coordinates into MimicKit space."""
    wp.launch(copy_indexed, dim=layout.nonsphere_q_idx.shape[0],
              inputs=[source, layout.nonsphere_q_idx, layout.nonsphere_dof_idx, target],
              device=layout.device)
    if layout.has_spherical_joints:
        wp.launch(quat_to_exp_map_indexed, dim=layout.sphere_q_start.shape[0],
                  inputs=[source, layout.sphere_q_start, layout.sphere_dof_start, target],
                  device=layout.device)


def rotate_dof_vectors(layout, joint_q, source, target, to_newton):
    """Rotate BALL velocities/efforts; leave scalar and root DOFs untouched."""
    if layout.has_spherical_joints:
        wp.launch(rotate_spherical_vectors, dim=layout.sphere_q_start.shape[0],
                  inputs=[joint_q, layout.sphere_q_start, layout.sphere_dof_start,
                          source, to_newton, target], device=layout.device)


class SimState:
    """Borrow Newton states and own their MimicKit views; borrow feedback arrays.

    Object access and representation conversion share these same buffers.
    Scene, control and runtime code never allocate a second set of Torch views.
    """

    def __init__(self, sim_model, num_envs, *, states=None):
        self.model = sim_model
        self._sim_model = sim_model
        self.states = NewtonStatePair(sim_model) if states is None else states
        self.raw_state = self.states.raw_state
        self.swap_state = self.states.swap_state

        self._torch_joint_q = wp.to_torch(self.raw_state.joint_q)
        self._torch_joint_qd = wp.to_torch(self.raw_state.joint_qd)
        self._torch_body_q = wp.to_torch(self.raw_state.body_q)
        self._torch_body_qd = wp.to_torch(self.raw_state.body_qd)

        self._wp_body_force = wp.clone(self.raw_state.body_f)
        self._torch_body_force = wp.to_torch(self._wp_body_force)

        # the joint parent transform is only used for objects with a fixed root
        self._torch_joint_X_p = wp.to_torch(sim_model.joint_X_p)

        self.root_pos = []
        self.root_rot = []
        self.root_vel = []
        self.root_ang_vel = []
        self.dof_pos = []
        self.dof_vel = []

        self.body_pos = []
        self.body_rot = []
        self.body_vel = []
        self.body_ang_vel = []
        self.body_force = []

        self.layout = ArticulationLayout(sim_model, num_envs)
        layout = self.layout

        joint_q = self._torch_joint_q.view(num_envs, -1)
        joint_qd = self._torch_joint_qd.view(num_envs, -1)
        body_q = self._torch_body_q.view(num_envs, -1, 7)
        body_qd = self._torch_body_qd.view(num_envs, -1, 6)
        body_force = self._torch_body_force.view(num_envs, -1, 6)
        joint_X_p = self._torch_joint_X_p.view(num_envs, -1, 7)

        has_sphere_joints = self._has_spherical_joints()

        if (has_sphere_joints):
            self._wp_dof_pos = wp.clone(self.raw_state.joint_qd)
            self._torch_dof_pos = wp.to_torch(self._wp_dof_pos)
            dof_pos = self._torch_dof_pos.view(num_envs, -1)
            # MimicKit velocities are local to the rotating joint; Newton's
            # BALL velocities are expressed in the parent joint anchor frame.
            self._wp_dof_vel = wp.clone(self.raw_state.joint_qd)
            dof_vel = wp.to_torch(self._wp_dof_vel).view(num_envs, -1)
        else:
            dof_vel = joint_qd

        for obj_id in range(layout.objs_per_env):
            bodies = layout.body_slices[obj_id]
            body_start = bodies.start
            q_root_start = layout.root_q_starts[obj_id]
            qd_root_start = layout.root_dof_slices[obj_id].start

            obj_body_q = body_q[:, bodies, :]
            obj_body_qd = body_qd[:, bodies, :]
            obj_body_force = body_force[:, bodies, :]

            fix_root = layout.fixed_roots[obj_id]

            if (fix_root):
                obj_root_pos = joint_X_p[..., body_start, :3]
                obj_root_rot = joint_X_p[..., body_start, 3:7]
                obj_root_vel = obj_body_qd[:, 0, :3]
                obj_root_ang_vel = obj_body_qd[:, 0, 3:6]
            else:
                obj_root_pos = joint_q[:, q_root_start:q_root_start + 3]
                obj_root_rot = joint_q[:, q_root_start + 3:q_root_start + 7]
                obj_root_vel = joint_qd[:, qd_root_start:qd_root_start + 3]
                obj_root_ang_vel = joint_qd[:, qd_root_start + 3:qd_root_start + 6]

            if (has_sphere_joints):
                obj_dof_pos = dof_pos[:, layout.dof_slices[obj_id]]
            else:
                obj_dof_pos = joint_q[:, layout.q_slices[obj_id]]

            obj_dof_vel = dof_vel[:, layout.dof_slices[obj_id]]

            obj_body_pos = obj_body_q[..., :3]
            obj_body_rot = obj_body_q[..., 3:]
            obj_body_vel = obj_body_qd[..., :3]
            obj_body_ang_vel = obj_body_qd[..., 3:]

            self.root_pos.append(obj_root_pos)
            self.root_rot.append(obj_root_rot)
            self.dof_pos.append(obj_dof_pos)

            self.dof_vel.append(obj_dof_vel)
            self.root_vel.append(obj_root_vel)
            self.root_ang_vel.append(obj_root_ang_vel)

            self.body_pos.append(obj_body_pos)
            self.body_rot.append(obj_body_rot)
            self.body_vel.append(obj_body_vel)
            self.body_ang_vel.append(obj_body_ang_vel)
            self.body_force.append(obj_body_force)

        self.post_step_update()
        self.gravity = sim_model.gravity.numpy()[0]
        self._num_envs = num_envs
        return

    def clear_forces(self):
        self._wp_body_force.zero_()
        return

    def copy_external_forces_to(self, raw_state):
        """Copy the buffered command without exposing its storage to actuation."""
        wp.copy(raw_state.body_f, self._wp_body_force)

    def copy(self, raw_state):
        self.states.copy(raw_state)

    def pre_step_update(self):
        if (self._has_spherical_joints()):
            layout = self.layout
            dof_positions_to_joint_q(layout, self._wp_dof_pos, self.raw_state.joint_q)
            # Copy only non-root DOFs: root views already reference raw_state.
            wp.launch(copy_indexed, dim=layout.nonsphere_dof_idx.shape[0],
                      inputs=[self._wp_dof_vel, layout.nonsphere_dof_idx,
                              layout.nonsphere_dof_idx, self.raw_state.joint_qd],
                      device=self._sim_model.device)
            self.rotate_dof_vectors(self.raw_state.joint_q, self._wp_dof_vel,
                                    self.raw_state.joint_qd, to_newton=True)
        return

    def post_step_update(self):
        if (self._has_spherical_joints()):
            layout = self.layout
            joint_q_to_dof_positions(layout, self.raw_state.joint_q, self._wp_dof_pos)
            wp.launch(copy_indexed, dim=layout.nonsphere_dof_idx.shape[0],
                      inputs=[self.raw_state.joint_qd, layout.nonsphere_dof_idx,
                              layout.nonsphere_dof_idx, self._wp_dof_vel],
                      device=self._sim_model.device)
            self.rotate_dof_vectors(self.raw_state.joint_q, self.raw_state.joint_qd,
                                    self._wp_dof_vel, to_newton=False)
        return

    def rotate_dof_vectors(self, joint_q, source, target, to_newton):
        rotate_dof_vectors(self.layout, joint_q, source, target, to_newton)

    def eval_fk(self):
        self.states.eval_fk()

    def _has_spherical_joints(self):
        return self.layout.has_spherical_joints


    def synchronize_buffers(self):
        self.states.synchronize_buffers()

    def initialize_auxiliary_views(self):
        self._raw_dof_pos = torch.zeros_like(self._torch_joint_q)
        self._obj_dof_pos = []

        num_envs = self._num_envs
        dof_pos = self._raw_dof_pos.view(num_envs, -1)

        for obj_id in range(self.layout.objs_per_env):
            obj_dof_pos = dof_pos[:, self.layout.q_slices[obj_id]]
            self._obj_dof_pos.append(obj_dof_pos)
        return


    # Public object/state access. Feedback storage belongs to the runtime.
    def bind_feedback(self, dof_forces, contact_forces, ground_contact_forces):
        self._dof_forces = dof_forces
        self._contact_forces = contact_forces
        self._ground_contact_forces = ground_contact_forces


    def get_root_pos(self, obj_id):
        root_pos = self.root_pos[obj_id]
        return root_pos


    def get_root_rot(self, obj_id):
        root_rot = self.root_rot[obj_id]
        return root_rot


    def get_root_vel(self, obj_id):
        root_vel = self.root_vel[obj_id]
        return root_vel


    def get_root_ang_vel(self, obj_id):
        root_ang_vel = self.root_ang_vel[obj_id]
        return root_ang_vel


    def get_dof_pos(self, obj_id):
        dof_pos = self.dof_pos[obj_id]
        return dof_pos


    def get_dof_vel(self, obj_id):
        dof_vel = self.dof_vel[obj_id]
        return dof_vel


    def get_dof_forces(self, obj_id):
        return self._dof_forces[obj_id]


    def get_body_pos(self, obj_id):
        body_pos = self.body_pos[obj_id]
        return body_pos


    def get_body_rot(self, obj_id):
        body_rot = self.body_rot[obj_id]
        return body_rot


    def get_body_vel(self, obj_id):
        body_vel = self.body_vel[obj_id]
        return body_vel


    def get_body_ang_vel(self, obj_id):
        body_ang_vel = self.body_ang_vel[obj_id]
        return body_ang_vel


    def set_root_pos(self, env_id, obj_id, root_pos):
        if (env_id is None):
            self.root_pos[obj_id][:, :] = root_pos
        else:
            self.root_pos[obj_id][env_id, :] = root_pos
        return


    def set_root_rot(self, env_id, obj_id, root_rot):
        if (env_id is None):
            self.root_rot[obj_id][:, :] = root_rot
        else:
            self.root_rot[obj_id][env_id, :] = root_rot
        return


    def set_root_vel(self, env_id, obj_id, root_vel):
        if (env_id is None):
            self.root_vel[obj_id][:, :] = root_vel
        else:
            self.root_vel[obj_id][env_id, :] = root_vel
        return


    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if (env_id is None):
            self.root_ang_vel[obj_id][:, :] = root_ang_vel
        else:
            self.root_ang_vel[obj_id][env_id, :] = root_ang_vel
        return


    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if (env_id is None):
            self.dof_pos[obj_id][:, :] = dof_pos
        else:
            self.dof_pos[obj_id][env_id, :] = dof_pos
        return


    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if (env_id is None):
            self.dof_vel[obj_id][:, :] = dof_vel
        else:
            self.dof_vel[obj_id][env_id, :] = dof_vel
        return


    def set_body_pos(self, env_id, obj_id, body_pos):
        if (env_id is None):
            self.body_pos[obj_id][:, :, :] = body_pos
        else:
            self.body_pos[obj_id][env_id, :, :] = body_pos
        return


    def set_body_rot(self, env_id, obj_id, body_rot):
        if (env_id is None):
            self.body_rot[obj_id][:, :, :] = body_rot
        else:
            self.body_rot[obj_id][env_id, :, :] = body_rot
        return


    def set_body_vel(self, env_id, obj_id, body_vel):
        if (env_id is None):
            self.body_vel[obj_id][:, :, :] = body_vel
        else:
            self.body_vel[obj_id][env_id, :, :] = body_vel
        return


    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        if (env_id is None):
            self.body_ang_vel[obj_id][:, :, :] = body_ang_vel
        else:
            self.body_ang_vel[obj_id][env_id, :, :] = body_ang_vel
        return


    def get_objs_per_env(self):
        return len(self.root_pos)


    def get_contact_forces(self, obj_id):
        forces = self._contact_forces[obj_id]
        return forces


    def get_ground_contact_forces(self, obj_id):
        forces = self._ground_contact_forces[obj_id]
        return forces


    def get_obj_torque_limits(self, env_id, obj_id):
        obj_idx = self.layout.object_index(env_id, obj_id)
        effort_lim = self._sim_model.joint_effort_limit.numpy()
        torque_lim = effort_lim[self.layout.dof_slices[obj_idx]]
        return torque_lim


    def get_obj_dof_limits(self, env_id, obj_id):
        obj_idx = self.layout.object_index(env_id, obj_id)
        lim_low = self._sim_model.joint_limit_lower.numpy()
        lim_upper = self._sim_model.joint_limit_upper.numpy()

        dofs = self.layout.dof_slices[obj_idx]
        dof_low = lim_low[dofs]
        dof_high = lim_upper[dofs]

        return dof_low, dof_high


    def get_obj_pd_gains(self, env_id, obj_id):
        obj_idx = self.layout.object_index(env_id, obj_id)
        dofs = self.layout.dof_slices[obj_idx]
        kp = self._sim_model.joint_target_ke.numpy()[dofs]
        kd = self._sim_model.joint_target_kd.numpy()[dofs]

        return kp, kd


    def find_obj_body_id(self, obj_id, body_name):
        body_names = self.get_obj_body_names(obj_id)
        body_id = body_names.index(body_name)
        return  body_id


    def get_obj_num_bodies(self, obj_id):
        num_bodies = self.body_pos[obj_id].shape[-2]
        return num_bodies


    def get_obj_num_dofs(self, obj_id):
        num_dofs = self.dof_pos[obj_id].shape[-1]
        return num_dofs


    def get_obj_body_names(self, obj_id):
        body_labels = self._sim_model.body_label[self.layout.body_slices[obj_id]]
        body_names = [os.path.basename(l) for l in body_labels]
        return body_names


    def calc_obj_mass(self, env_id, obj_id):
        obj_idx = self.layout.object_index(env_id, obj_id)
        body_masses = self._sim_model.body_mass.numpy()
        masses = body_masses[self.layout.body_slices[obj_idx]]
        total_mass = float(masses.sum())
        return total_mass


    def get_obj_body_ids(self, obj_id):
        bodies = self.layout.body_slices[obj_id]
        return list(range(bodies.start, bodies.stop))


@wp.kernel
def rotate_spherical_vectors(joint_q: wp.array(dtype=float),
                             q_start: wp.array(dtype=int), dof_start: wp.array(dtype=int),
                             source: wp.array(dtype=float), to_newton: bool,
                             target: wp.array(dtype=float)):
    i = wp.tid()
    q_i = q_start[i]
    d_i = dof_start[i]
    q = wp.quat(joint_q[q_i], joint_q[q_i + 1], joint_q[q_i + 2], joint_q[q_i + 3])
    v = wp.vec3(source[d_i], source[d_i + 1], source[d_i + 2])
    if to_newton:
        v = wp.quat_rotate(q, v)
    else:
        v = wp.quat_rotate_inv(q, v)
    target[d_i] = v[0]
    target[d_i + 1] = v[1]
    target[d_i + 2] = v[2]


@wp.kernel
def copy_indexed(input: wp.array(dtype=float),
                 idx_in: wp.array(dtype=int),
                 idx_out: wp.array(dtype=int),
                 output: wp.array(dtype=float)):
    tid = wp.tid()
    value = input[idx_in[tid]]
    output[idx_out[tid]] = value
    return


@wp.kernel
def quat_to_exp_map_indexed(in_q: wp.array(dtype=float),
                 q_start: wp.array(dtype=int),
                 dof_start: wp.array(dtype=int),
                 out_dof: wp.array(dtype=float)):
    tid = wp.tid()
    q_idx = q_start[tid]
    dof_idx = dof_start[tid]

    q = wp.quaternion(in_q[q_idx], in_q[q_idx + 1], in_q[q_idx + 2], in_q[q_idx + 3])
    axis, angle = wp.quat_to_axis_angle(q)
    exp_map = axis * angle

    out_dof[dof_idx] = exp_map[0]
    out_dof[dof_idx + 1] = exp_map[1]
    out_dof[dof_idx + 2] = exp_map[2]
    return


@wp.kernel
def exp_map_to_quat_indexed(in_dof: wp.array(dtype=float),
                 dof_start: wp.array(dtype=int),
                 q_start: wp.array(dtype=int),
                 out_q: wp.array(dtype=float)):
    min_theta = 1e-5

    tid = wp.tid()
    dof_idx = dof_start[tid]
    q_idx = q_start[tid]

    exp_map = wp.vec3f(in_dof[dof_idx], in_dof[dof_idx + 1], in_dof[dof_idx + 2])

    angle = wp.length(exp_map)
    axis = exp_map / wp.max(angle, min_theta)
    angle = wp.atan2(wp.sin(angle), wp.cos(angle))

    mask = wp.abs(angle) > min_theta
    angle = angle if mask else 0.0
    axis[0] = axis[0] if mask else 0.0
    axis[1] = axis[1] if mask else 0.0
    axis[2] = axis[2] if mask else 1.0

    q = wp.quat_from_axis_angle(axis, angle)

    out_q[q_idx] = q[0]
    out_q[q_idx + 1] = q[1]
    out_q[q_idx + 2] = q[2]
    out_q[q_idx + 3] = q[3]
    return
