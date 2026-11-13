import warp as wp
wp.config.enable_backward = False

import newton
import numpy as np
import os
import torch
import pyglet

import engines.engine as engine
import util.resource_util as resource_util
from engines.newton_pbf_api import NewtonEngineAPI
from engines.newton_pbf_runtime import PBFScene
from RSP.sim_ready.paths import PBF_PLAY_ROOT
from engines.newton_pbf_viewer import PBFViewer, PBFVideoRecorder
from solvers.pbf.coupling import SolverPBFMuJoCo
from util.logger import Logger

def str_to_key_code(key_str):
    key_name = key_str.upper()

    if (len(key_str) == 1 and key_str.isdigit()):
        key_name = "_" + key_str
    elif (key_str == "ESC"):
        key_name = "ESCAPE"
    elif (key_str == "RETURN"):
        key_name = "ENTER"
    elif (key_str == "DELETE"):
        key_name = "DEL"
    elif (key_str == "LEFT_SHIFT"):
        key_name = "LSHIFT"
    elif (key_str == "LEFT_CONTROL"):
        key_name = "LCTRL"
    elif (key_str == "LEFT_ALT"):
        key_name = "LALT"
    elif (key_str == "RIGHT_SHIFT"):
        key_name = "RSHIFT"
    elif (key_str == "RIGHT_CONTROL"):
        key_name = "RCTRL"
    elif (key_str == "RIGHT_ALT"):
        key_name = "RALT"

    key_code = getattr(pyglet.window.key, key_name)
    return key_code

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
    """Own both Newton states and their MimicKit views; borrow feedback arrays.

    Object access and representation conversion share these same buffers.
    Scene, control and runtime code never allocate a second set of Torch views.
    """

    def __init__(self, sim_model, num_envs):
        self.model = sim_model
        self._sim_model = sim_model
        self.raw_state = self._sim_model.state()
        self.eval_fk()

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
        self.swap_state = sim_model.state()
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
        wp.copy(self.raw_state.joint_q, raw_state.joint_q)
        wp.copy(self.raw_state.joint_qd, raw_state.joint_qd)
        wp.copy(self.raw_state.body_q, raw_state.body_q)
        wp.copy(self.raw_state.body_qd, raw_state.body_qd)
        wp.copy(self.raw_state.body_f, raw_state.body_f)
        wp.copy(self.raw_state.particle_q, raw_state.particle_q)
        wp.copy(self.raw_state.particle_qd, raw_state.particle_qd)
        return

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
        newton.eval_fk(self._sim_model, self.raw_state.joint_q, self.raw_state.joint_qd, self.raw_state)
        return

    def _has_spherical_joints(self):
        return self.layout.has_spherical_joints


    def synchronize_buffers(self):
        self.swap_state.assign(self.raw_state)

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


class ObjCfg():
    def __init__(self, asset_file, fix_root, is_visual, enable_self_collisions):
        self.asset_file = asset_file
        self.fix_root = fix_root
        self.is_visual = is_visual
        self.enable_self_collisions = enable_self_collisions
        return

    def __hash__(self):
        values = (self.asset_file, self.fix_root, self.is_visual, self.enable_self_collisions)
        obj_hash = hash(values)
        return obj_hash

    def __eq__(self, other):
        same = self.asset_file == other.asset_file
        same &= self.fix_root == other.fix_root
        same &= self.is_visual == other.is_visual
        same &= self.enable_self_collisions == other.enable_self_collisions
        return same


class SceneBuilder:
    """Declare and assemble objects; delegate asset/boundary work to PBFScene."""

    def __init__(self, config, num_envs, device, timestep, substeps):
        self.device = device
        self.play_scene = PBFScene(config.get("pbf_asset_dir", PBF_PLAY_ROOT),
                                   device, timestep, substeps,
                                   fluid_substeps=config.get("pbf_substeps"),
                                   pressure_iterations=config.get("pbf_pressure_iterations"))
        self.num_envs = num_envs
        self.builder = self._create_model_builder()
        self._builder_cache = {}
        self.object_types = []
        self._obj_builders = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        self.colors = []
        self.visual = []


    def finalize(self):
        self.validate_envs()
        self.play_scene.add_static(self.builder)
        self.build_objects(self.play_scene.add_particles)
        # Fluid contacts and pressure feedback belong exclusively to PBF.
        for i in range(len(self.builder.shape_flags)):
            self.builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_PARTICLES
        Logger.print("Initializing simulation...")
        from RSP.sim_ready.adapters.newton import finalize_model
        model = finalize_model(self.builder, device=self.device, report=self.play_scene.validation)
        self.play_scene.validate_model(model)
        return model

    def bind_boundaries(self, model, state):
        body_ids = [body_id for obj_id, visual in enumerate(self.visual[0])
                    if not visual for body_id in state.get_obj_body_ids(obj_id)]
        return self.play_scene.bind_model(model, body_ids)

    def apply_start_transforms(self, state):
        num_envs = self.num_envs
        objs_per_env = state.get_objs_per_env()
        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                start_pos = self._obj_start_pos[env_id][obj_id]
                start_rot = self._obj_start_rot[env_id][obj_id]
                start_pos = torch.tensor(start_pos, device=self.device)
                start_rot = torch.tensor(start_rot, device=self.device)

                state.set_root_pos(env_id, obj_id, start_pos)
                state.set_root_rot(env_id, obj_id, start_rot)

        state.eval_fk()
        return


    def create_env(self):
        env_id = len(self._obj_builders)
        assert(env_id < self.num_envs)

        self._obj_builders.append([])
        self.object_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
        self.colors.append([])
        self.visual.append([])

        return env_id


    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True,
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])

        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                                               enable_self_collisions=enable_self_collisions)

        obj_id = len(self._obj_builders[env_id])
        self.object_types[env_id].append(obj_type)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)
        self.colors[env_id].append(color)
        self.visual[env_id].append(is_visual)

        return obj_id


    def get_obj_type(self, obj_id):
        obj_type = self.object_types[0][obj_id]
        return obj_type


    def validate_envs(self):
        num_envs = self.num_envs
        objs_per_env = len(self._obj_builders[0])

        for i in range(0, num_envs):
            curr_objs = self._obj_builders[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return


    def build_objects(self, add_particles):
        num_envs = self.num_envs
        objs_per_env = len(self._obj_builders[0])
        total_objs = num_envs * objs_per_env

        for env_id in range(num_envs):
            self.builder.begin_world()

            for obj_id in range(objs_per_env):
                obj_builder = self._obj_builders[env_id][obj_id]
                self.builder.add_builder(obj_builder)

                num_created = env_id * objs_per_env + obj_id + 1
                Logger.print("Building {:d}/{:d} objs".format(num_created, total_objs), end='\r')

            add_particles(self.builder)
            self.builder.end_world()

        Logger.print("")
        return


    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                         enable_self_collisions=enable_self_collisions)

        if (obj_cfg in self._builder_cache):
            obj_builder = self._builder_cache[obj_cfg]
        else:
            obj_builder = self._create_model_builder()

            from RSP.sim_ready.adapters.newton import import_physical_asset
            import_physical_asset(obj_builder, asset_file, fix_root=fix_root,
                                  enable_self_collisions=enable_self_collisions,
                                  report=self.play_scene.validation)
            if obj_builder.articulation_count != 1:
                self.play_scene.validation.add("unsupported", "consumer.articulation_count", asset_file,
                    f"MimicKit requires one articulation per object; imported {obj_builder.articulation_count}")
                self.play_scene.validation.require_valid()

            if (is_visual):
                for i in range(len(obj_builder.shape_flags)):
                    obj_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_SHAPES

            self._builder_cache[obj_cfg] = obj_builder

        return obj_builder


    def _create_model_builder(self):
        builder = newton.ModelBuilder()
        newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
        builder.default_shape_cfg.mu = 1.0
        return builder


    def build_ground(self):
        shape_cfg = self.builder.ShapeConfig(mu=1.0, restitution=0)
        self.builder.add_ground_plane(cfg=shape_cfg)
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


# Compatibility names for callers importing the original internal components.
Controls = CommandBuffers
Control = Actuation


class SceneReset:
    """Coordinate runtime placement/reset; borrow components, own no buffers.

    Placement and reset remain separate operations: the environment first writes
    its sampled joints and scene objects, then places the character, then resets
    solver history. Keep that order and the existing FK synchronization points.
    """

    def __init__(self, state, actuation, runtime, play_scene):
        self._state = state
        self._actuation = actuation
        self._runtime = runtime
        self._play_scene = play_scene

    def place_character(self, obj_id):
        self._state.pre_step_update()
        self._state.eval_fk()
        root = self._state.get_root_pos(obj_id)
        shift = self._play_scene.character_translation(
            self._state.raw_state, self._state.get_obj_body_ids(obj_id), root[0].cpu().numpy())
        root.add_(torch.as_tensor(shift, device=root.device))
        return shift

    def reset(self):
        self._state.pre_step_update()
        self._runtime.reset_solver()
        self._state.synchronize_buffers()
        self._actuation.clear_external_forces()
        self._runtime.reset_clock()
        self._state.post_step_update()
        self._runtime.clear_feedback()


class SimulationRuntime:
    """Own solver execution, contact feedback, Graphs and the physics clock.

    SimState and CommandBuffers own their buffers. The runtime borrows them and
    receives an interaction-force callback; it has no reference back to NewtonEngine.
    """

    def __init__(self, config, num_envs, device):
        if num_envs != 1 or not wp.get_device(device).is_cuda:
            raise ValueError("newton_pbf currently supports one CUDA environment")
        self.device = device
        self.num_envs = num_envs
        self.env_spacing = config["env_spacing"]
        # Preserve contact-rich GSI initialization capacity.
        self._solver_njmax = int(config.get("newton_njmax", 450))
        self._solver_nconmax = int(config.get("newton_nconmax", 150))
        if self._solver_njmax <= 0 or self._solver_nconmax <= 0:
            raise ValueError("Newton constraint/contact capacities must be positive.")
        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"
        self.timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self.sim_timestep = 1.0 / sim_freq
        self._fluid_steps = config.get("pbf_substeps", self._sim_steps)
        if (isinstance(self._fluid_steps, bool) or not isinstance(self._fluid_steps, int)
                or self._fluid_steps < 1 or self._sim_steps % self._fluid_steps != 0):
            raise ValueError("pbf_substeps must be a positive integer divisor of the rigid step count")
        self.fluid_timestep = self.timestep / self._fluid_steps
        self._rigid_steps_per_fluid_step = self._sim_steps // self._fluid_steps
        self._sim_step_count = 0
        self._enable_cuda_graph = bool(config.get("enable_cuda_graph", True))

    def initialize(self, state, control, scene, interaction_forces):
        self._state = state
        self._sim_model = state.model
        self._control = control
        self._interaction_forces = interaction_forces
        self._build_solver(scene)
        scene.play_scene.initialize_particles(state.raw_state, state.swap_state)
        self._solver.boundaries.sync_body_poses(state.raw_state)
        self._build_contact_feedback()
        self._build_observables()
        state.initialize_auxiliary_views()
        control.initialize_force_buffer()
        self._build_dof_force_tensors()
        state.bind_feedback(self._dof_forces, self._contact_forces, self._ground_contact_forces)

    def advance(self):
        if self._graph is not None:
            self._graph.launch()
        else:
            self._simulate()

    def finish_step(self):
        self._ground_contact_sensor.update(self._state.raw_state, self._contacts,
                                           observables=self._observables)
        self._sim_step_count += 1

    def get_sim_time(self):
        return self.timestep * self._sim_step_count

    def reset_solver(self):
        # The composed solver preserves the freshly sampled GSI joint state.
        self._solver.reset(self._state.raw_state)

    def reset_clock(self):
        self._sim_step_count = 0

    def clear_feedback(self):
        if self._control.mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            self._wp_dof_forces.zero_()
        else:
            self._control.controls.dof_force.zero_()
        self._ground_contact_sensor.total_force.zero_()
        self._ground_contact_sensor.force_matrix.zero_()

    # Initialization: rigid/fluid solvers, contact observations and readout views.
    def _build_solver(self, scene):
        rigid = newton.solvers.SolverMuJoCo(
            self._sim_model,
            solver="newton",
            njmax=self._solver_njmax,
            nconmax=self._solver_nconmax,
            impratio=10,
            iterations=100,
            ls_iterations=50
        )

        boundaries = scene.bind_boundaries(self._sim_model, self._state)

        self._solver = SolverPBFMuJoCo(
            self._sim_model, 
            config=scene.play_scene.config,
            boundaries=boundaries, rigid=rigid
        )
        return


    def _build_contact_feedback(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(self._sim_model,
                                                                   sensing_bodies=list(range(self._sim_model.body_count)),
                                                                   counterpart_shapes="ground*",
                                                                   measure_total=True,
                                                                   request_contact_attributes=False,
                                                                   verbose=True)
        num_envs = self.num_envs
        num_objs = self._state.get_objs_per_env()
        total_forces = wp.to_torch(self._ground_contact_sensor.total_force).view(num_envs, -1, 3)
        ground_forces = wp.to_torch(self._ground_contact_sensor.force_matrix)
        # This engine creates exactly one global ground plane.
        assert ground_forces.shape[1] == 1
        ground_forces = ground_forces.view(num_envs, -1, 3)

        self._contact_forces = []
        self._ground_contact_forces = []

        for obj_id in range(num_objs):
            bodies = self._state.layout.body_slices[obj_id]
            obj_total_forces = total_forces[:, bodies, :]
            obj_ground_forces = ground_forces[:, bodies, :]

            self._contact_forces.append(obj_total_forces)
            self._ground_contact_forces.append(obj_ground_forces)
        return


    def _build_observables(self):
        self._collision_pipeline = newton.CollisionPipeline(
            self._sim_model, rigid_contact_max=self._solver.get_max_contact_count(),
            soft_contact_max=0)
        self._contacts = self._collision_pipeline.contacts()
        kinds = set(self._ground_contact_sensor.solver_observable_kinds)
        if self._control.mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            kinds.add(self._solver.ObservableKind.QFRC_ACTUATOR)
        self._observables = self._solver.observables(kinds, contacts=self._contacts)

    def _build_dof_force_tensors(self):
        if self._control.mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            self._wp_dof_forces = wp.clone(self._observables.qfrc_actuator)
            forces = wp.to_torch(self._wp_dof_forces).view(self.num_envs, -1)
            self._dof_forces = [
                forces[:, self._state.layout.dof_slices[obj]]
                for obj in range(self._state.get_objs_per_env())]
        else:
            self._dof_forces = self._control.controls.joint_force
        return


    # Execution: capture the same substep sequence that eager mode runs.
    def build_graphs(self):
        self._graph = None
        if not self._enable_cuda_graph:
            return
        # Compile adapter kernels before capture, including scalar-only scenes.
        wp.load_module(module=__name__, device=self._sim_model.device)
        self._graph = self._solver.capture_step_graph(
            self._simulate, self._sim_steps, fluid_substeps=self._fluid_steps)
        return


    def _simulate(self):
        state0 = self._state.raw_state
        state1 = self._state.swap_state
        control = self._control.controls.control

        self._prepare_step()

        for i in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._solver.step(state0, state1, control, self._contacts, self.sim_timestep,
                              observables=self._observables if i == self._sim_steps - 1 else None,
                              advance_fluid=i % self._rigid_steps_per_fluid_step == 0,
                              fluid_dt=self.fluid_timestep)
            state0, state1 = state1, state0

        self._finish_simulation(state0)


    def _prepare_step(self):
        self._state.pre_step_update()
        self._control.update_targets()

    def _pre_sim_step(self, raw_state, control):
        raw_state.clear_forces()
        self._control.apply_external_forces(raw_state)
        self._interaction_forces(raw_state)
        self._control.apply_control(raw_state, control)
        return


    def _finish_simulation(self, state0):
        if (self._sim_steps % 2 != 0):
            self._state.copy(self._state.swap_state)

        self._state.post_step_update()
        if self._control.mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            wp.copy(self._wp_dof_forces, self._observables.qfrc_actuator)
            self._state.rotate_dof_vectors(
                state0.joint_q, 
                self._wp_dof_forces,
                self._wp_dof_forces, 
                to_newton=False
            )

        self._control.finish_step()

        # The composed MuJoCo solver already writes joint coordinates.
        # Running the generic IK fallback here would overwrite that result.
        return


class Presentation:
    """Own display, input callbacks and recording; borrow simulation data."""

    def __init__(self, num_envs, env_spacing, visualize, record_video):
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._viewer = None
        self._video_recorder = None
        self._record_video = record_video and not visualize
        self._recording = False
        self._closed = False
        self._keyboard_callbacks = {}
        if visualize:
            self._build_viewer()

    def initialize(self, state, colors):
        self._sim_model = state.model
        self._layout = state.layout
        self._obj_colors = colors
        self._body_counts = [state.get_obj_num_bodies(i) for i in range(state.get_objs_per_env())]
        if self._viewer is not None:
            self.init_viewer(self._viewer)
            self._camera_offsets = self._viewer.world_offsets.numpy()

    def initialize_recording(self, engine_api):
        if self._record_video:
            Logger.print("Video recording enabled")
            # The existing recorder consumes Engine's public API only.
            self._video_recorder = resource_util.construct(PBFVideoRecorder, engine_api)

    def capture_before_step(self):
        if self._record_video and self._recording:
            self._video_recorder.capture_frame()

    def apply_forces(self, raw_state):
        if self._viewer is not None:
            self._viewer.apply_forces(raw_state)

    def render(self, sim_time, raw_state, draw_scene):
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        self._viewer.end_frame()
        # Window events can destroy the GL context during end_frame().
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(raw_state)
        draw_scene(self._viewer)

        self._draw_line_count = 0

        return

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        self._recording = False
        # Keep viewers and their event-handler owners alive until GL windows close.
        viewer = getattr(self, "_viewer", None)
        recorder = getattr(self, "_video_recorder", None)
        resource_util.cleanup(
            recorder.close if recorder is not None else None,
            lambda: resource_util.close_viewer(viewer),
        )
        self._viewer = None
        self._video_recorder = None
        return

    def get_viewer(self):
        """Expose the viewer to opt-in tool adapters without changing its owner."""
        return self._viewer

    def set_camera_pose(self, pos, look_at):
        dx = look_at[0] - pos[0]
        dy = look_at[1] - pos[1]
        dz = look_at[2] - pos[2]

        pitch = np.arctan2(dz, np.sqrt(dx * dx + dy * dy))
        yaw = np.arctan2(dy, dx)
        cam_pos = pyglet.math.Vec3(pos[0] + self._camera_offsets[0, 0],
                                   pos[1] + self._camera_offsets[0, 1],
                                   pos[2] + self._camera_offsets[0, 2])
        self._viewer.set_camera(cam_pos, np.rad2deg(pitch), np.rad2deg(yaw))
        return

    def get_camera_pos(self):
        cam_state_pos = self._viewer.camera.pos
        cam_pos = np.array([float(cam_state_pos[0] - self._camera_offsets[0, 0]),
                            float(cam_state_pos[1] - self._camera_offsets[0, 1]),
                            float(cam_state_pos[2] - self._camera_offsets[0, 2])])
        return cam_pos

    def get_camera_dir(self):
        pitch = float(np.deg2rad(self._viewer.camera.pitch))
        yaw = float(np.deg2rad(self._viewer.camera.yaw))

        dx = np.cos(pitch) * np.cos(yaw)
        dy = np.cos(pitch) * np.sin(yaw)
        dz = np.sin(pitch)

        cam_dir = np.array([dx, dy, dz])
        cam_dir /= np.linalg.norm(cam_dir)
        return cam_dir

    def set_viewer_world_offsets(self, offsets):
        """Change display placement only; preserve the picking array's identity."""
        offsets = np.asarray(offsets, dtype=np.float32)
        if offsets.shape != (self._num_envs, 3) or not np.isfinite(offsets).all():
            raise ValueError("Viewer offsets must be finite with shape [num_envs, 3].")
        if self._viewer is not None:
            self._viewer.world_offsets.assign(offsets)
            self._camera_offsets = offsets.copy()
        return

    def set_body_color(self, env_id, obj_id, body_id, color, viewer=None):
        obj_idx = self._layout.object_index(env_id, obj_id)
        body_start = self._layout.body_slices[obj_idx].start
        body_idx = body_start + body_id

        body_shape_idx = self._sim_model.body_shapes[body_idx]
        # Newton 1.7 viewers synchronize display colors from the model in log_state.
        colors = wp.to_torch(self._sim_model.shape_color)
        colors[body_shape_idx] = torch.as_tensor(color, dtype=colors.dtype, device=colors.device)
        return

    def init_viewer(self, viewer):
        env_spacing = self._env_spacing
        viewer.set_model(self._sim_model)
        viewer.show_particles = True
        viewer.set_world_offsets([env_spacing, env_spacing, 0.0])
        self._apply_obj_colors(viewer)
        # Set the default for the whole Newton UI, including tool callbacks.
        # Keep this in MimicKit so the installed Newton package stays untouched.
        ui = viewer.ui
        if ui is not None and ui.is_available:
            font_config = ui.imgui.ImFontConfig()
            font_config.size_pixels = 18.0
            ui.io.font_default = ui.io.fonts.add_font_default(font_config)
        return

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        cam_offset = self._camera_offsets[env_id]
        start_pts = start_verts.copy()
        end_pts = end_verts.copy()

        start_pts[:, :2] += cam_offset[:2]
        end_pts[:, :2] += cam_offset[:2]
        line_name = "lines{:d}".format(self._draw_line_count)
        self._viewer.log_lines(name=line_name,
                               starts=wp.array(start_pts, dtype=wp.vec3),
                               ends=wp.array(end_pts, dtype=wp.vec3),
                               colors=wp.array(cols[:, :3], dtype=wp.vec3),
                               width=line_width)
        self._draw_line_count += 1
        return

    def register_keyboard_callback(self, key_str, callback_func):
        key_code = str_to_key_code(key_str)
        assert(key_code not in self._keyboard_callbacks)
        self._keyboard_callbacks[key_code] = callback_func
        return

    def register_ui_callback(self, callback_func, position="side"):
        """Expose Newton's optional UI extension without exposing its viewer."""
        viewer = self._viewer
        if viewer is None or viewer.ui is None or not viewer.ui.is_available:
            raise RuntimeError("PBF play controls require Newton's imgui-bundle viewer UI.")
        viewer.register_ui_callback(callback_func, position=position)
        return

    def get_viewer_paused(self):
        """Read the viewer's pause control for an opt-in playback controller."""
        return self._viewer is not None and self._viewer.is_paused()

    def set_viewer_paused(self, paused):
        """Keep opt-in playback controls synchronized with Newton's pause UI."""
        if self._viewer is not None:
            # Current Newton exposes is_paused(), but no public setter.
            setter = getattr(self._viewer, "set_paused", None)
            if callable(setter):
                setter(bool(paused))
            else:
                self._viewer._paused = bool(paused)
        return

    def enabled_record_video(self):
        return self._record_video

    def get_video_recording(self):
        return self._video_recorder.get_video()

    def start_video_recording(self):
        self._video_recorder.clear()
        self._recording = True
        return

    def stop_video_recording(self):
        self._recording = False
        return

    def _build_viewer(self):
        self._viewer = resource_util.construct(PBFViewer, headless=False)
        self._draw_line_count = 0

        def on_keyboard_event(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)
            return

        self._viewer.renderer.register_key_press(on_keyboard_event)
        return

    def _apply_obj_colors(self, viewer):
        num_envs = self._num_envs
        objs_per_env = len(self._obj_colors[0])

        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                obj_col = self._obj_colors[env_id][obj_id]

                if (obj_col is not None):
                    num_bodies = self._body_counts[obj_id]
                    for body_id in range(num_bodies):
                        self.set_body_color(env_id, obj_id, body_id, obj_col, viewer)
        return

    def _on_keyboard_event(self, symbol, modifiers):
        if (symbol in self._keyboard_callbacks):
            callback = self._keyboard_callbacks[symbol]
            callback()
        return


class NewtonEngine(NewtonEngineAPI):
    """Assemble components and run workflows behind NewtonEngineAPI.

    SceneBuilder declares the model. SimState owns state and shared topology;
    Actuation owns command processing. SimulationRuntime drives the coupled
    solver and feedback. SceneReset coordinates runtime reset, and Presentation
    owns display/recording. Components never call back into this facade, except
    for the existing video recorder's public Engine API adapter.
    """

    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)
        self._realtime = config.get("realtime", True)
        self._runtime = SimulationRuntime(config, num_envs, device)
        self._scene = SceneBuilder(config, num_envs, device,
                                   self._runtime.timestep, self._runtime._sim_steps)
        self._control_mode = engine.ControlMode[config.get("control_mode", "none")]
        self._scene.build_ground()
        self._presentation = resource_util.construct(
            Presentation, num_envs, self._runtime.env_spacing, visualize, record_video)

    def get_name(self):
        return "newton_pbf"

    def initialize_sim(self):
        model = self._scene.finalize()
        self._state = SimState(model, self.get_num_envs())
        self._scene.apply_start_transforms(self._state)
        self._control = Actuation(model, self._state, self._control_mode, self.get_num_envs())
        self._runtime.initialize(self._state, self._control, self._scene,
                                 self._presentation.apply_forces)
        self._scene_reset = SceneReset(self._state, self._control, self._runtime,
                                       self._scene.play_scene)
        self._presentation.initialize(self._state, self._scene.colors)
        self._presentation.initialize_recording(self)
        self._runtime.build_graphs()

    def step(self):
        self._presentation.capture_before_step()
        self._runtime.advance()
        self._runtime.finish_step()

    # Placement does not reset the world; the environment calls these in order.

    def place_character(self, obj_id):
        return self._scene_reset.place_character(obj_id)

    def reset_play_scene(self):
        self._scene_reset.reset()

    def render(self):
        self._presentation.render(self.get_sim_time(), self._state.raw_state,
                                  self._scene.play_scene.draw_pool)
        if self._realtime:
            super().render()

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        presentation = getattr(self, "_presentation", None)
        resource_util.cleanup(presentation.close if presentation is not None else None)
        return


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
def apply_pd_torque(target: wp.array(dtype=float), pos: wp.array(dtype=float),
                    vel: wp.array(dtype=float), kp: wp.array(dtype=float),
                    kd: wp.array(dtype=float), limit: wp.array(dtype=float),
                    mode: wp.array(dtype=int), force: wp.array(dtype=float)):
    i = wp.tid()
    force[i] = 0.0
    if mode[i] != 0:
        torque = kp[i] * (target[i] - pos[i]) - kd[i] * vel[i]
        force[i] = wp.clamp(torque, -limit[i], limit[i])

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
