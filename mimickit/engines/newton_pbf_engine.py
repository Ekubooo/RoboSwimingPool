import warp as wp
wp.config.enable_backward = False

import newton
import numpy as np
import os
import torch
import pyglet

import engines.engine as engine
import util.resource_util as resource_util
from engines.newton_pbf_runtime import PBFScene
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

class SimState:
    def __init__(self, sim_model, num_envs):
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

        articulation_start = sim_model.articulation_start.numpy()
        joint_q_start = sim_model.joint_q_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        joint_q = self._torch_joint_q.view(num_envs, -1)
        joint_qd = self._torch_joint_qd.view(num_envs, -1)
        body_q = self._torch_body_q.view(num_envs, -1, 7)
        body_qd = self._torch_body_qd.view(num_envs, -1, 6)
        body_force = self._torch_body_force.view(num_envs, -1, 6)
        joint_X_p = self._torch_joint_X_p.view(num_envs, -1, 7)

        self._record_spherical_joints()
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

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]

            q_root_start = joint_q_start[body_start]
            q_dof_start = joint_q_start[body_start + 1]
            q_dof_end = joint_q_start[body_end]

            qd_root_start = joint_qd_start[body_start]
            qd_dof_start = joint_qd_start[body_start + 1]
            qd_dof_end = joint_qd_start[body_end]

            obj_body_q = body_q[:, body_start:body_end, :]
            obj_body_qd = body_qd[:, body_start:body_end, :]
            obj_body_force = body_force[:, body_start:body_end, :]

            root_type = joint_types[body_start]
            fix_root = root_type == newton.JointType.FIXED

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
                obj_dof_pos = dof_pos[:, qd_dof_start:qd_dof_end]
            else:
                obj_dof_pos = joint_q[:, q_dof_start:q_dof_end]

            obj_dof_vel = dof_vel[:, qd_dof_start:qd_dof_end]

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
        return

    def clear_forces(self):
        self._wp_body_force.zero_()
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

    def pre_step_update(self):
        if (self._has_spherical_joints()):
            wp.launch(
                kernel=copy_indexed,
                dim=self._wp_nonsphere_joint_dof_idx.shape[0],
                inputs=[self._wp_dof_pos, self._wp_nonsphere_joint_dof_idx,
                        self._wp_nonsphere_joint_q_idx, self.raw_state.joint_q]
            )
            wp.launch(
                kernel=exp_map_to_quat_indexed,
                dim=self._wp_sphere_dof_start.shape[0],
                inputs=[self._wp_dof_pos, self._wp_sphere_dof_start,
                        self._wp_sphere_q_start, self.raw_state.joint_q]
            )
            # Copy only non-root DOFs: root views already reference raw_state.
            wp.launch(copy_indexed, dim=self._wp_nonsphere_joint_dof_idx.shape[0],
                      inputs=[self._wp_dof_vel, self._wp_nonsphere_joint_dof_idx,
                              self._wp_nonsphere_joint_dof_idx, self.raw_state.joint_qd],
                      device=self._sim_model.device)
            self.rotate_dof_vectors(self.raw_state.joint_q, self._wp_dof_vel,
                                    self.raw_state.joint_qd, to_newton=True)
        return

    def post_step_update(self):
        if (self._has_spherical_joints()):
            wp.launch(
                kernel=copy_indexed,
                dim=self._wp_nonsphere_joint_q_idx.shape[0],
                inputs=[self.raw_state.joint_q, self._wp_nonsphere_joint_q_idx,
                        self._wp_nonsphere_joint_dof_idx, self._wp_dof_pos]
            )
            wp.launch(
                kernel=quat_to_exp_map_indexed,
                dim=self._wp_sphere_q_start.shape[0],
                inputs=[self.raw_state.joint_q, self._wp_sphere_q_start,
                        self._wp_sphere_dof_start, self._wp_dof_pos]
            )
            wp.launch(copy_indexed, dim=self._wp_nonsphere_joint_dof_idx.shape[0],
                      inputs=[self.raw_state.joint_qd, self._wp_nonsphere_joint_dof_idx,
                              self._wp_nonsphere_joint_dof_idx, self._wp_dof_vel],
                      device=self._sim_model.device)
            self.rotate_dof_vectors(self.raw_state.joint_q, self.raw_state.joint_qd,
                                    self._wp_dof_vel, to_newton=False)
        return

    def rotate_dof_vectors(self, joint_q, source, target, to_newton):
        if self._has_spherical_joints():
            wp.launch(rotate_spherical_vectors, dim=self._wp_sphere_q_start.shape[0],
                      inputs=[joint_q, self._wp_sphere_q_start, self._wp_sphere_dof_start,
                              source, to_newton, target], device=self._sim_model.device)

    def eval_fk(self):
        newton.eval_fk(self._sim_model, self.raw_state.joint_q, self.raw_state.joint_qd, self.raw_state)
        return

    def _record_spherical_joints(self):
        sphere_type = newton.JointType.BALL

        joint_q_start = self._sim_model.joint_q_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        joint_types = self._sim_model.joint_type.numpy()
        joint_parent = self._sim_model.joint_parent.numpy()
        num_joints = self._sim_model.joint_count

        sphere_q_start = []
        sphere_dof_start = []
        nonsphere_q_idx = []
        nonsphere_dof_idx = []

        for j in range(num_joints):
            curr_joint_type = joint_types[j]
            curr_joint_parent = joint_parent[j]
            is_root = curr_joint_parent == -1

            q_start = joint_q_start[j]
            q_end = joint_q_start[j + 1]
            dof_start = joint_qd_start[j]
            dof_end = joint_qd_start[j + 1]

            if (is_root):
                pass
            elif (curr_joint_type == sphere_type):
                assert(q_end - q_start == 4)
                assert(dof_end - dof_start == 3)
                sphere_q_start.append(q_start)
                sphere_dof_start.append(dof_start)
            else:
                curr_q_idx = [i for i in range(q_start, q_end)]
                curr_dof_idx = [i for i in range(dof_start, dof_end)]
                nonsphere_q_idx += curr_q_idx
                nonsphere_dof_idx += curr_dof_idx

        self._wp_sphere_q_start = wp.array(sphere_q_start, device=self._sim_model.device, dtype=int)
        self._wp_sphere_dof_start = wp.array(sphere_dof_start, device=self._sim_model.device, dtype=int)
        self._wp_nonsphere_joint_q_idx = wp.array(nonsphere_q_idx, device=self._sim_model.device, dtype=int)
        self._wp_nonsphere_joint_dof_idx = wp.array(nonsphere_dof_idx, device=self._sim_model.device, dtype=int)
        return

    def _has_spherical_joints(self):
        return self._wp_sphere_q_start.shape[0] > 0


class Controls:
    def __init__(self, sim_model, num_envs):
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

        articulation_start = sim_model.articulation_start.numpy()
        joint_qd_start = sim_model.joint_qd_start.numpy()

        num_objs = sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)

        self.target_pos = []
        self.target_vel = []
        self.joint_force = []

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]
            qd_start = joint_qd_start[body_start + 1]
            qd_end = joint_qd_start[body_end]

            obj_target_pos = target_pos[:, qd_start:qd_end]
            obj_target_vel = target_vel[:, qd_start:qd_end]
            obj_joint_force = joint_force[:, qd_start:qd_end]

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


class NewtonEngine(engine.Engine):
    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize=visualize)

        if num_envs != 1 or not wp.get_device(device).is_cuda:
            raise ValueError("newton_pbf currently supports one CUDA environment")

        self._device = device
        self._num_envs = num_envs
        self._env_spacing = config["env_spacing"]
        # GSI can create contact-rich initial poses; keep historical defaults for existing tasks.
        self._solver_njmax = int(config.get("newton_njmax", 450))
        self._solver_nconmax = int(config.get("newton_nconmax", 150))
        if self._solver_njmax <= 0 or self._solver_nconmax <= 0:
            raise ValueError("Newton constraint/contact capacities must be positive.")

        sim_freq = config.get("sim_freq", 60)
        control_freq = config.get("control_freq", 10)
        assert(sim_freq >= control_freq and sim_freq % control_freq == 0), \
            "Simulation frequency must be a multiple of the control frequency"

        self._timestep = 1.0 / control_freq
        self._sim_steps = int(sim_freq / control_freq)
        self._sim_timestep = 1.0 / sim_freq
        self._sim_step_count = 0
        self._enable_cuda_graph = bool(config.get("enable_cuda_graph", True))
        self._fluid_scene = PBFScene(config.get("pbf_asset_dir", "output/sim_ready/pbf_play"),
                                    device, self._timestep, self._sim_steps)

        self._scene_builder = self._create_model_builder()
        self._builder_cache = dict()

        self._obj_types = []
        self._obj_builders = []
        self._obj_start_pos = []
        self._obj_start_rot = []
        self._obj_colors = []
        self._obj_visual = []

        if ("control_mode" in config):
            self._control_mode = engine.ControlMode[config["control_mode"]]
        else:
            self._control_mode = engine.ControlMode.none

        self._build_ground()

        if (visualize):
            self._build_viewer()
            self._keyboard_callbacks = dict()
            self._record_video = False
        else:
            self._viewer = None
            self._record_video = record_video

        return

    def get_name(self):
        return "newton_pbf"

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

    def create_env(self):
        env_id = len(self._obj_builders)
        assert(env_id < self.get_num_envs())

        self._obj_builders.append([])
        self._obj_types.append([])
        self._obj_start_pos.append([])
        self._obj_start_rot.append([])
        self._obj_colors.append([])
        self._obj_visual.append([])

        return env_id

    def initialize_sim(self):
        self._validate_envs()
        self._fluid_scene.add_static(self._scene_builder)
        self._build_objs()
        # Fluid contacts and pressure feedback belong exclusively to PBF.
        for i in range(len(self._scene_builder.shape_flags)):
            self._scene_builder.shape_flags[i] &= ~newton.ShapeFlags.COLLIDE_PARTICLES

        Logger.print("Initializing simulation...")
        self._sim_model = self._scene_builder.finalize(device=self._device, requires_grad=False)

        num_envs = self.get_num_envs()
        self._sim_state = SimState(self._sim_model, num_envs)
        self._state_swap_buffer = self._sim_model.state()
        self._gravity = self._sim_model.gravity.numpy()[0]

        self._apply_start_xform()

        self._build_controls()
        self._build_solver()
        self._fluid_scene.initialize_particles(self._sim_state.raw_state, self._state_swap_buffer)
        self._solver.boundaries.sync_body_poses(self._sim_state.raw_state)
        self._build_contact_sensors()
        self._collision_pipeline = newton.CollisionPipeline(
            self._sim_model, rigid_contact_max=self._solver.get_max_contact_count(),
            soft_contact_max=0)
        self._contacts = self._collision_pipeline.contacts()
        kinds = set(self._ground_contact_sensor.solver_observable_kinds)
        if self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            kinds.add(self._solver.ObservableKind.QFRC_ACTUATOR)
        self._observables = self._solver.observables(kinds, contacts=self._contacts)

        self._build_sim_tensors()
        self._build_dof_force_tensors()

        if (self._visualize()):
            self.init_viewer(self._viewer)
            self._camera_offsets = self._viewer.world_offsets.numpy()

        if self.enabled_record_video():
            self._video_recorder = self._build_video_recorder()
            self._recording = False

        self._build_graphs()
        return

    def step(self):
        if (self.enabled_record_video() and self._recording):
            self._video_recorder.capture_frame()

        if (self._graph):
            fluid = self._solver.fluid
            graph, prediction = self._fluid_graphs[fluid.pre_Pos.ptr]
            wp.capture_launch(graph)
            fluid.pre_Pos, fluid.temp_pPos = prediction
            self._solver.rigid._step += self._sim_steps
        else:
            self._simulate()

        self._update_contact_sensors()
        self._sim_step_count += 1
        return

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True,
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        if (start_rot is None):
            start_rot = np.array([0.0, 0.0, 0.0, 1.0])

        if (start_pos is None):
            start_pos = np.array([0.0, 0.0, 0.0])

        obj_builder = self._create_obj_builder(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                                               enable_self_collisions=enable_self_collisions)

        obj_id = len(self._obj_builders[env_id])
        self._obj_types[env_id].append(obj_type)
        self._obj_builders[env_id].append(obj_builder)
        self._obj_start_pos[env_id].append(start_pos)
        self._obj_start_rot[env_id].append(start_rot)
        self._obj_colors[env_id].append(color)
        self._obj_visual[env_id].append(is_visual)

        return obj_id

    def set_cmd(self, obj_id, cmd):
        if (self._control_mode == engine.ControlMode.none):
            pass
        elif (self._control_mode == engine.ControlMode.pos):
            self._controls.target_pos[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.vel):
            self._controls.target_vel[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.torque):
            self._controls.joint_force[obj_id][:] = cmd
        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._controls.target_pos[obj_id][:] = cmd
        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)
        return

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

    def render(self):
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        sim_time = self.get_sim_time()

        self._viewer.end_frame()
        # Window events can destroy the GL context during end_frame().
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(self._sim_state.raw_state)
        self._fluid_scene.draw_pool(self._viewer)

        self._draw_line_count = 0

        super().render()
        return

    def get_timestep(self):
        return self._timestep

    def get_sim_timestep(self):
        return self._sim_timestep

    def get_sim_time(self):
        time_step = self.get_timestep()
        return time_step * self._sim_step_count

    def get_num_envs(self):
        return self._num_envs

    def get_gravity(self):
        return self._gravity

    def get_objs_per_env(self):
        return len(self._sim_state.root_pos)

    def get_root_pos(self, obj_id):
        root_pos = self._sim_state.root_pos[obj_id]
        return root_pos

    def get_root_rot(self, obj_id):
        root_rot = self._sim_state.root_rot[obj_id]
        return root_rot

    def get_root_vel(self, obj_id):
        root_vel = self._sim_state.root_vel[obj_id]
        return root_vel

    def get_root_ang_vel(self, obj_id):
        root_ang_vel = self._sim_state.root_ang_vel[obj_id]
        return root_ang_vel

    def get_dof_pos(self, obj_id):
        dof_pos = self._sim_state.dof_pos[obj_id]
        return dof_pos

    def get_dof_vel(self, obj_id):
        dof_vel = self._sim_state.dof_vel[obj_id]
        return dof_vel

    def get_dof_forces(self, obj_id):
        return self._dof_forces[obj_id]

    def get_body_pos(self, obj_id):
        body_pos = self._sim_state.body_pos[obj_id]
        return body_pos

    def get_body_rot(self, obj_id):
        body_rot = self._sim_state.body_rot[obj_id]
        return body_rot

    def get_body_vel(self, obj_id):
        body_vel = self._sim_state.body_vel[obj_id]
        return body_vel

    def get_body_ang_vel(self, obj_id):
        body_ang_vel = self._sim_state.body_ang_vel[obj_id]
        return body_ang_vel

    def get_contact_forces(self, obj_id):
        forces = self._contact_forces[obj_id]
        return forces

    def get_ground_contact_forces(self, obj_id):
        forces = self._ground_contact_forces[obj_id]
        return forces

    def set_root_pos(self, env_id, obj_id, root_pos):
        if (env_id is None):
            self._sim_state.root_pos[obj_id][:, :] = root_pos
        else:
            self._sim_state.root_pos[obj_id][env_id, :] = root_pos
        return

    def set_root_rot(self, env_id, obj_id, root_rot):
        if (env_id is None):
            self._sim_state.root_rot[obj_id][:, :] = root_rot
        else:
            self._sim_state.root_rot[obj_id][env_id, :] = root_rot
        return

    def set_root_vel(self, env_id, obj_id, root_vel):
        if (env_id is None):
            self._sim_state.root_vel[obj_id][:, :] = root_vel
        else:
            self._sim_state.root_vel[obj_id][env_id, :] = root_vel
        return

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        if (env_id is None):
            self._sim_state.root_ang_vel[obj_id][:, :] = root_ang_vel
        else:
            self._sim_state.root_ang_vel[obj_id][env_id, :] = root_ang_vel
        return

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        if (env_id is None):
            self._sim_state.dof_pos[obj_id][:, :] = dof_pos
        else:
            self._sim_state.dof_pos[obj_id][env_id, :] = dof_pos
        return

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        if (env_id is None):
            self._sim_state.dof_vel[obj_id][:, :] = dof_vel
        else:
            self._sim_state.dof_vel[obj_id][env_id, :] = dof_vel
        return

    def set_body_pos(self, env_id, obj_id, body_pos):
        if (env_id is None):
            self._sim_state.body_pos[obj_id][:, :, :] = body_pos
        else:
            self._sim_state.body_pos[obj_id][env_id, :, :] = body_pos
        return

    def set_body_rot(self, env_id, obj_id, body_rot):
        if (env_id is None):
            self._sim_state.body_rot[obj_id][:, :, :] = body_rot
        else:
            self._sim_state.body_rot[obj_id][env_id, :, :] = body_rot
        return

    def set_body_vel(self, env_id, obj_id, body_vel):
        if (env_id is None):
            self._sim_state.body_vel[obj_id][:, :, :] = body_vel
        else:
            self._sim_state.body_vel[obj_id][env_id, :, :] = body_vel
        return

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        if (env_id is None):
            self._sim_state.body_ang_vel[obj_id][:, :, :] = body_ang_vel
        else:
            self._sim_state.body_ang_vel[obj_id][env_id, :, :] = body_ang_vel
        return

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        if (env_id is None or len(env_id) > 0):
            assert(len(forces.shape) == 2)
            self._has_body_forces.fill_(1)

            if (env_id is None):
                self._sim_state.body_force[obj_id][:, body_id, :3] = forces
            else:
                self._sim_state.body_force[obj_id][env_id, body_id, :3] = forces
        return

    def get_obj_torque_limits(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        effort_lim = self._sim_model.joint_effort_limit.numpy()

        body_start = articulation_start[obj_idx]
        body_end = articulation_start[obj_idx + 1]
        qd_start = joint_qd_start[body_start + 1]
        qd_end = joint_qd_start[body_end]

        torque_lim = effort_lim[qd_start:qd_end]
        return torque_lim

    def get_obj_dof_limits(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()
        lim_low = self._sim_model.joint_limit_lower.numpy()
        lim_upper = self._sim_model.joint_limit_upper.numpy()

        body_start = articulation_start[obj_idx]
        body_end = articulation_start[obj_idx + 1]
        qd_start = joint_qd_start[body_start + 1]
        qd_end = joint_qd_start[body_end]

        dof_low = lim_low[qd_start:qd_end]
        dof_high = lim_upper[qd_start:qd_end]

        return dof_low, dof_high

    def get_obj_pd_gains(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_qd_start = self._sim_model.joint_qd_start.numpy()

        body_start = articulation_start[obj_idx]
        body_end = articulation_start[obj_idx + 1]
        qd_start = joint_qd_start[body_start + 1]
        qd_end = joint_qd_start[body_end]

        kp = self._sim_model.joint_target_ke.numpy()[qd_start:qd_end]
        kd = self._sim_model.joint_target_kd.numpy()[qd_start:qd_end]

        return kp, kd

    def find_obj_body_id(self, obj_id, body_name):
        body_names = self.get_obj_body_names(obj_id)
        body_id = body_names.index(body_name)
        return  body_id

    def get_obj_type(self, obj_id):
        obj_type = self._obj_types[0][obj_id]
        return obj_type

    def get_obj_num_bodies(self, obj_id):
        num_bodies = self._sim_state.body_pos[obj_id].shape[-2]
        return num_bodies

    def get_obj_num_dofs(self, obj_id):
        num_dofs = self._sim_state.dof_pos[obj_id].shape[-1]
        return num_dofs

    def get_obj_body_names(self, obj_id):
        articulation_start = self._sim_model.articulation_start.numpy()
        body_start = articulation_start[obj_id]
        body_end = articulation_start[obj_id + 1]
        body_labels = self._sim_model.body_label[body_start:body_end]
        body_names = [os.path.basename(l) for l in body_labels]
        return body_names

    def calc_obj_mass(self, env_id, obj_id):
        objs_per_env = self.get_objs_per_env()
        obj_idx = env_id * objs_per_env + obj_id

        articulation_start = self._sim_model.articulation_start.numpy()
        body_masses = self._sim_model.body_mass.numpy()

        body_start = int(articulation_start[obj_idx])
        body_end = int(articulation_start[obj_idx + 1])
        masses = body_masses[body_start:body_end]
        total_mass = float(masses.sum())
        return total_mass

    def get_control_mode(self):
        return self._control_mode

    def get_sim_model(self):
        return self._sim_model

    def get_sim_state(self):
        return self._sim_state

    def get_viewer(self):
        """Expose the viewer to opt-in tool adapters without changing its owner."""
        return self._viewer

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
        objs_per_env = self.get_objs_per_env()
        articulation_start = self._sim_model.articulation_start.numpy()

        obj_idx = env_id * objs_per_env + obj_id
        body_start = articulation_start[obj_idx]
        body_idx = body_start + body_id

        body_shape_idx = self._sim_model.body_shapes[body_idx]
        # Newton 1.7 viewers synchronize display colors from the model in log_state.
        colors = wp.to_torch(self._sim_model.shape_color)
        colors[body_shape_idx] = torch.as_tensor(color, dtype=colors.dtype, device=colors.device)
        return

    def init_viewer(self, viewer):
        env_spacing = self._get_env_spacing()
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

    def _validate_envs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])

        for i in range(0, num_envs):
            curr_objs = self._obj_builders[i]
            num_objs = len(curr_objs)
            assert(num_objs == objs_per_env), "All envs must have the same number of objects."
        return

    def _build_objs(self):
        num_envs = self.get_num_envs()
        objs_per_env = len(self._obj_builders[0])
        total_objs = num_envs * objs_per_env

        for env_id in range(num_envs):
            self._scene_builder.begin_world()

            for obj_id in range(objs_per_env):
                obj_builder = self._obj_builders[env_id][obj_id]
                self._scene_builder.add_builder(obj_builder)

                num_created = env_id * objs_per_env + obj_id + 1
                Logger.print("Building {:d}/{:d} objs".format(num_created, total_objs), end='\r')

            self._fluid_scene.add_particles(self._scene_builder)
            self._scene_builder.end_world()

        Logger.print("")
        return

    def _apply_start_xform(self):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()
        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                start_pos = self._obj_start_pos[env_id][obj_id]
                start_rot = self._obj_start_rot[env_id][obj_id]
                start_pos = torch.tensor(start_pos, device=self._device)
                start_rot = torch.tensor(start_rot, device=self._device)

                self.set_root_pos(env_id, obj_id, start_pos)
                self.set_root_rot(env_id, obj_id, start_rot)

        self._sim_state.eval_fk()
        return

    def _build_controls(self):
        kp = self._sim_model.mujoco.dof_passive_stiffness
        kd = self._sim_model.joint_damping

        if (self._control_mode == engine.ControlMode.none):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.NONE))

        elif (self._control_mode == engine.ControlMode.pos):
            wp.copy(self._sim_model.joint_target_ke, kp)
            wp.copy(self._sim_model.joint_target_kd, kd)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.POSITION))

        elif (self._control_mode == engine.ControlMode.vel):
            self._sim_model.joint_target_ke.fill_(0.0)
            wp.copy(self._sim_model.joint_target_kd, kd)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.VELOCITY))

        elif (self._control_mode == engine.ControlMode.torque):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        elif (self._control_mode == engine.ControlMode.pd_explicit):
            self._sim_model.joint_target_ke.fill_(0.0)
            self._sim_model.joint_target_kd.fill_(0.0)
            self._kp_raw = wp.clone(kp)
            self._kd_raw = wp.clone(kd)
            self._torque_lim_raw = wp.clone(self._sim_model.joint_effort_limit)
            self._pd_dof_pos = wp.zeros_like(self._sim_state.raw_state.joint_qd)
            self._pd_dof_vel = wp.zeros_like(self._sim_state.raw_state.joint_qd)
            self._sim_model.joint_target_mode.fill_(int(newton.JointTargetMode.EFFORT))

        else:
            assert(False), "Unsupported control mode: {}".format(self._control_mode)

        self._sim_model.mujoco.dof_passive_stiffness.fill_(0.0)
        self._sim_model.joint_damping.fill_(0.0)

        # The root belongs to the state, not the object's command vector.
        qd_start = self._sim_model.joint_qd_start.numpy()
        root_joints = self._sim_model.articulation_start.numpy()[:-1]
        modes = self._sim_model.joint_target_mode.numpy()
        for root in root_joints:
            modes[qd_start[root]:qd_start[root + 1]] = int(newton.JointTargetMode.NONE)
        self._sim_model.joint_target_mode.assign(modes)

        num_envs = self.get_num_envs()
        self._controls = Controls(self._sim_model, num_envs)
        return

    def _build_solver(self):
        rigid = newton.solvers.SolverMuJoCo(
            self._sim_model,
            solver="newton",
            njmax=self._solver_njmax,
            nconmax=self._solver_nconmax,
            impratio=10,
            iterations=100,
            ls_iterations=50
        )
        body_ids = [body_id for obj_id, visual in enumerate(self._obj_visual[0])
                    if not visual for body_id in self.get_obj_body_ids(obj_id)]
        boundaries = self._fluid_scene.bind_model(self._sim_model, body_ids)
        self._solver = SolverPBFMuJoCo(self._sim_model, config=self._fluid_scene.config,
                                       boundaries=boundaries, rigid=rigid)
        return

    def _build_graphs(self):
        self._graph = None
        self._fluid_graphs = {}
        if not self._enable_cuda_graph:
            return

        fluid = self._solver.fluid
        for module in ("solvers.pbf.pbf_helper", "solvers.pbf.PBF",
                       "solvers.pbf.boundary_model", "solvers.pbf.coupling"):
            wp.load_module(module=module, device=self._sim_model.device)
        fluid.grid.reserve(self._sim_model.particle_count)
        prediction = (fluid.pre_Pos, fluid.temp_pPos)
        rigid_step = self._solver.rigid._step
        # PBF swaps prediction buffers as it spatially reorders particles.
        # Odd substep counts need a graph for each starting buffer address.
        starts = [prediction] if self._sim_steps % 2 == 0 else [prediction, prediction[::-1]]
        try:
            for buffers in starts:
                fluid.pre_Pos, fluid.temp_pPos = buffers
                with wp.ScopedCapture(device=self._sim_model.device) as capture:
                    self._simulate()
                self._fluid_graphs[buffers[0].ptr] = (capture.graph, (fluid.pre_Pos, fluid.temp_pPos))
        finally:
            fluid.pre_Pos, fluid.temp_pPos = prediction
            self._solver.rigid._step = rigid_step
        self._graph = self._fluid_graphs[prediction[0].ptr][0]
        return

    def get_play_scene(self):
        return self._fluid_scene

    def get_obj_body_ids(self, obj_id):
        starts = self._sim_model.articulation_start.numpy()
        # As in SimState, this backend imports one joint per body.
        return list(range(int(starts[obj_id]), int(starts[obj_id + 1])))

    def place_character(self, obj_id):
        self._sim_state.pre_step_update()
        self._sim_state.eval_fk()
        root = self.get_root_pos(obj_id)
        shift = self._fluid_scene.character_translation(
            self._sim_state.raw_state, self.get_obj_body_ids(obj_id), root[0].cpu().numpy())
        root.add_(torch.as_tensor(shift, device=self._device))
        return shift

    def reset_play_scene(self):
        self._sim_state.pre_step_update()
        self._solver.reset(self._sim_state.raw_state)
        self._state_swap_buffer.assign(self._sim_state.raw_state)
        self._sim_state.clear_forces()
        self._has_body_forces.zero_()
        self._sim_step_count = 0
        self._sim_state.post_step_update()
        self._wp_dof_forces.zero_()
        self._ground_contact_sensor.total_force.zero_()
        self._ground_contact_sensor.force_matrix.zero_()
        return

    def _build_sim_tensors(self):
        self._raw_dof_pos = torch.zeros_like(self._sim_state._torch_joint_q)
        self._obj_dof_pos = []
        self._has_body_forces = wp.array([0], dtype=wp.int32, device=self._sim_model.device)

        articulation_start = self._sim_model.articulation_start.numpy()
        joint_q_start = self._sim_model.joint_q_start.numpy()

        num_envs = self.get_num_envs()
        num_objs = self._sim_model.articulation_count
        objs_per_env = int(num_objs / num_envs)
        dof_pos = self._raw_dof_pos.view(num_envs, -1)

        for obj_id in range(objs_per_env):
            body_start = articulation_start[obj_id]
            body_end = articulation_start[obj_id + 1]
            q_start = joint_q_start[body_start + 1]
            q_end = joint_q_start[body_end]

            obj_dof_pos = dof_pos[:, q_start:q_end]
            self._obj_dof_pos.append(obj_dof_pos)
        return

    def _build_dof_force_tensors(self):
        if self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            self._wp_dof_forces = wp.clone(self._observables.qfrc_actuator)
            forces = wp.to_torch(self._wp_dof_forces).view(self._num_envs, -1)
            starts = self._sim_model.articulation_start.numpy()
            qd_start = self._sim_model.joint_qd_start.numpy()
            self._dof_forces = [
                forces[:, qd_start[starts[obj] + 1]:qd_start[starts[obj + 1]]]
                for obj in range(self.get_objs_per_env())]
        else:
            self._dof_forces = self._controls.joint_force
        return

    def _create_obj_builder(self, asset_file, fix_root, is_visual, enable_self_collisions):
        obj_cfg = ObjCfg(asset_file=asset_file, fix_root=fix_root, is_visual=is_visual,
                         enable_self_collisions=enable_self_collisions)

        if (obj_cfg in self._builder_cache):
            obj_builder = self._builder_cache[obj_cfg]
        else:
            obj_builder = self._create_model_builder()

            _, asset_ext = os.path.splitext(asset_file)
            if (asset_ext == ".xml"):
                obj_builder.add_mjcf(
                    asset_file,
                    floating=not fix_root,
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    convert_3d_hinge_to_ball_joints=True
                )
            elif (asset_ext == ".urdf"):
                obj_builder.add_urdf(
                    asset_file,
                    floating=not fix_root,
                    ignore_inertial_definitions=False,
                    collapse_fixed_joints=False,
                    enable_self_collisions=enable_self_collisions,
                    joint_ordering="dfs"
                )
            else:
                assert(False), "Unsupported asset format: {:s}".format(asset_ext)

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

    def _build_viewer(self):
        self._viewer = resource_util.construct(PBFViewer, headless=False)
        self._draw_line_count = 0

        def on_keyboard_event(symbol, modifiers):
            self._on_keyboard_event(symbol, modifiers)
            return

        self._viewer.renderer.register_key_press(on_keyboard_event)
        return

    def _build_ground(self):
        shape_cfg = self._scene_builder.ShapeConfig(mu=1.0, restitution=0)
        self._scene_builder.add_ground_plane(cfg=shape_cfg)
        return

    def _build_contact_sensors(self):
        self._build_ground_contact_sensor()
        return

    def _build_ground_contact_sensor(self):
        self._ground_contact_sensor = newton.sensors.SensorContact(self._sim_model,
                                                                   sensing_bodies=list(range(self._sim_model.body_count)),
                                                                   counterpart_shapes="ground*",
                                                                   measure_total=True,
                                                                   request_contact_attributes=False,
                                                                   verbose=True)
        num_envs = self.get_num_envs()
        num_objs = self.get_objs_per_env()
        total_forces = wp.to_torch(self._ground_contact_sensor.total_force).view(num_envs, -1, 3)
        ground_forces = wp.to_torch(self._ground_contact_sensor.force_matrix)
        # This engine creates exactly one global ground plane.
        assert ground_forces.shape[1] == 1
        ground_forces = ground_forces.view(num_envs, -1, 3)

        offset_idx = 0
        self._contact_forces = []
        self._ground_contact_forces = []

        for obj_id in range(num_objs):
            num_bodies = self.get_obj_num_bodies(obj_id)
            obj_total_forces = total_forces[:, offset_idx:offset_idx + num_bodies, :]
            obj_ground_forces = ground_forces[:, offset_idx:offset_idx + num_bodies, :]

            self._contact_forces.append(obj_total_forces)
            self._ground_contact_forces.append(obj_ground_forces)
            offset_idx += num_bodies
        return

    def _get_env_spacing(self):
        return self._env_spacing

    def _simulate(self):
        state0 = self._sim_state.raw_state
        state1 = self._state_swap_buffer
        control = self._controls.control

        self._sim_state.pre_step_update()
        self._update_control_targets()

        for i in range(self._sim_steps):
            self._pre_sim_step(state0, control)
            self._solver.step(state0, state1, control, self._contacts, self._sim_timestep,
                              observables=self._observables if i == self._sim_steps - 1 else None)
            state0, state1 = state1, state0

        if (self._sim_steps % 2 != 0):
            self._sim_state.copy(self._state_swap_buffer)

        self._sim_state.post_step_update()
        if self._control_mode in (engine.ControlMode.pos, engine.ControlMode.vel):
            wp.copy(self._wp_dof_forces, self._observables.qfrc_actuator)
            self._sim_state.rotate_dof_vectors(state0.joint_q, self._wp_dof_forces,
                                               self._wp_dof_forces, to_newton=False)

        def clear_body_forces():
            self._sim_state.clear_forces()
            self._has_body_forces.zero_()
            return
        wp.capture_if(self._has_body_forces, on_true=clear_body_forces)

        # The composed MuJoCo solver already writes joint coordinates.
        # Running the generic IK fallback here would overwrite that result.
        return

    def _pre_sim_step(self, raw_state, control):
        raw_state.clear_forces()

        def apply_body_force():
            wp.copy(raw_state.body_f, self._sim_state._wp_body_force)
            return
        wp.capture_if(self._has_body_forces, on_true=apply_body_force)

        if (self._visualize()):
            self._viewer.apply_forces(raw_state)

        if (self._control_mode == engine.ControlMode.pd_explicit):
            self._apply_pd_explicit_torque(raw_state, control)
        elif self._control_mode == engine.ControlMode.vel:
            wp.copy(control.joint_target_qd, self._controls.target_dof_vel)
            self._sim_state.rotate_dof_vectors(raw_state.joint_q, self._controls.target_dof_vel,
                                               control.joint_target_qd, to_newton=True)
        elif self._control_mode == engine.ControlMode.torque:
            wp.copy(control.joint_f, self._controls.dof_force)
            self._sim_state.rotate_dof_vectors(raw_state.joint_q, self._controls.dof_force,
                                               control.joint_f, to_newton=True)
        return

    def _update_contact_sensors(self):
        self._ground_contact_sensor.update(self._sim_state.raw_state, self._contacts,
                                           observables=self._observables)
        return

    def _update_control_targets(self):
        if self._control_mode != engine.ControlMode.pos:
            return
        state = self._sim_state
        wp.launch(copy_indexed, dim=state._wp_nonsphere_joint_dof_idx.shape[0],
                  inputs=[self._controls.target_dof_pos, state._wp_nonsphere_joint_dof_idx,
                          state._wp_nonsphere_joint_q_idx, self._controls.control.joint_target_q],
                  device=self._sim_model.device)
        if state._has_spherical_joints():
            wp.launch(exp_map_to_quat_indexed, dim=state._wp_sphere_dof_start.shape[0],
                      inputs=[self._controls.target_dof_pos, state._wp_sphere_dof_start,
                              state._wp_sphere_q_start, self._controls.control.joint_target_q],
                      device=self._sim_model.device)

    def _visualize(self):
        return self._viewer is not None

    def _apply_obj_colors(self, viewer):
        num_envs = self.get_num_envs()
        objs_per_env = self.get_objs_per_env()

        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                obj_col = self._obj_colors[env_id][obj_id]

                if (obj_col is not None):
                    num_bodies = self.get_obj_num_bodies(obj_id)
                    for body_id in range(num_bodies):
                        self.set_body_color(env_id, obj_id, body_id, obj_col, viewer)
        return

    def _apply_pd_explicit_torque(self, sim_state, control):
        state = self._sim_state
        wp.copy(self._pd_dof_vel, sim_state.joint_qd)
        state.rotate_dof_vectors(sim_state.joint_q, sim_state.joint_qd,
                                 self._pd_dof_vel, to_newton=False)
        wp.launch(copy_indexed, dim=state._wp_nonsphere_joint_q_idx.shape[0],
                  inputs=[sim_state.joint_q, state._wp_nonsphere_joint_q_idx,
                          state._wp_nonsphere_joint_dof_idx, self._pd_dof_pos],
                  device=self._sim_model.device)
        if state._has_spherical_joints():
            wp.launch(quat_to_exp_map_indexed, dim=state._wp_sphere_q_start.shape[0],
                      inputs=[sim_state.joint_q, state._wp_sphere_q_start,
                              state._wp_sphere_dof_start, self._pd_dof_pos],
                      device=self._sim_model.device)
        wp.launch(apply_pd_torque, dim=self._sim_model.joint_dof_count,
                  inputs=[self._controls.target_dof_pos, self._pd_dof_pos, self._pd_dof_vel,
                          self._kp_raw, self._kd_raw, self._torque_lim_raw,
                          self._sim_model.joint_target_mode, self._controls.dof_force],
                  device=self._sim_model.device)
        wp.copy(control.joint_f, self._controls.dof_force)
        state.rotate_dof_vectors(sim_state.joint_q, self._controls.dof_force,
                                 control.joint_f, to_newton=True)
        return

    def _on_keyboard_event(self, symbol, modifiers):
        if (symbol in self._keyboard_callbacks):
            callback = self._keyboard_callbacks[symbol]
            callback()
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

    def _build_video_recorder(self) -> PBFVideoRecorder:
        Logger.print("Video recording enabled")
        return resource_util.construct(PBFVideoRecorder, self)


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
