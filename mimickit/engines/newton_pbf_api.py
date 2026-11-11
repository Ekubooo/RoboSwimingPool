"""Hand-maintained, explicit MimicKit API for the Newton PBF backend."""

from engines.engine import Engine


class NewtonEngineAPI(Engine):
    """Forward public access to components assembled by the concrete engine.

    NewtonEngine supplies _scene, _runtime, _presentation and _control_mode
    during construction, then _state and _control during initialize_sim().
    This layer has no constructor or resources of its own. Initialization,
    stepping, reset, rendering and cleanup remain concrete-engine workflows.
    """

    # Scene declaration and access.

    def create_env(self):
        return self._scene.create_env()

    def create_obj(self, env_id, obj_type, asset_file, name, is_visual=False, enable_self_collisions=True,
                   fix_root=False, start_pos=None, start_rot=None, color=None, disable_motors=False):
        return self._scene.create_obj(env_id, obj_type, asset_file, name, is_visual, enable_self_collisions,
                                      fix_root, start_pos, start_rot, color, disable_motors)

    def get_obj_type(self, obj_id):
        return self._scene.get_obj_type(obj_id)

    def get_play_scene(self):
        return self._scene.play_scene

    # State, object metadata and borrowed solver feedback.

    def get_sim_model(self):
        return self._state.model

    def get_sim_state(self):
        return self._state

    def get_gravity(self):
        return self._state.gravity

    def get_num_envs(self):
        return self._runtime.num_envs

    def get_root_pos(self, obj_id):
        return self._state.get_root_pos(obj_id)

    def get_root_rot(self, obj_id):
        return self._state.get_root_rot(obj_id)

    def get_root_vel(self, obj_id):
        return self._state.get_root_vel(obj_id)

    def get_root_ang_vel(self, obj_id):
        return self._state.get_root_ang_vel(obj_id)

    def get_dof_pos(self, obj_id):
        return self._state.get_dof_pos(obj_id)

    def get_dof_vel(self, obj_id):
        return self._state.get_dof_vel(obj_id)

    def get_dof_forces(self, obj_id):
        return self._state.get_dof_forces(obj_id)

    def get_body_pos(self, obj_id):
        return self._state.get_body_pos(obj_id)

    def get_body_rot(self, obj_id):
        return self._state.get_body_rot(obj_id)

    def get_body_vel(self, obj_id):
        return self._state.get_body_vel(obj_id)

    def get_body_ang_vel(self, obj_id):
        return self._state.get_body_ang_vel(obj_id)

    def set_root_pos(self, env_id, obj_id, root_pos):
        return self._state.set_root_pos(env_id, obj_id, root_pos)

    def set_root_rot(self, env_id, obj_id, root_rot):
        return self._state.set_root_rot(env_id, obj_id, root_rot)

    def set_root_vel(self, env_id, obj_id, root_vel):
        return self._state.set_root_vel(env_id, obj_id, root_vel)

    def set_root_ang_vel(self, env_id, obj_id, root_ang_vel):
        return self._state.set_root_ang_vel(env_id, obj_id, root_ang_vel)

    def set_dof_pos(self, env_id, obj_id, dof_pos):
        return self._state.set_dof_pos(env_id, obj_id, dof_pos)

    def set_dof_vel(self, env_id, obj_id, dof_vel):
        return self._state.set_dof_vel(env_id, obj_id, dof_vel)

    def set_body_pos(self, env_id, obj_id, body_pos):
        return self._state.set_body_pos(env_id, obj_id, body_pos)

    def set_body_rot(self, env_id, obj_id, body_rot):
        return self._state.set_body_rot(env_id, obj_id, body_rot)

    def set_body_vel(self, env_id, obj_id, body_vel):
        return self._state.set_body_vel(env_id, obj_id, body_vel)

    def set_body_ang_vel(self, env_id, obj_id, body_ang_vel):
        return self._state.set_body_ang_vel(env_id, obj_id, body_ang_vel)

    def get_obj_torque_limits(self, env_id, obj_id):
        return self._state.get_obj_torque_limits(env_id, obj_id)

    def get_obj_dof_limits(self, env_id, obj_id):
        return self._state.get_obj_dof_limits(env_id, obj_id)

    def get_obj_pd_gains(self, env_id, obj_id):
        return self._state.get_obj_pd_gains(env_id, obj_id)

    def get_obj_num_bodies(self, obj_id):
        return self._state.get_obj_num_bodies(obj_id)

    def get_obj_num_dofs(self, obj_id):
        return self._state.get_obj_num_dofs(obj_id)

    def get_obj_body_names(self, obj_id):
        return self._state.get_obj_body_names(obj_id)

    def get_obj_body_ids(self, obj_id):
        return self._state.get_obj_body_ids(obj_id)

    def get_objs_per_env(self):
        return self._state.get_objs_per_env()

    def get_contact_forces(self, obj_id):
        return self._state.get_contact_forces(obj_id)

    def get_ground_contact_forces(self, obj_id):
        return self._state.get_ground_contact_forces(obj_id)

    def find_obj_body_id(self, obj_id, body_name):
        return self._state.find_obj_body_id(obj_id, body_name)

    def calc_obj_mass(self, env_id, obj_id):
        return self._state.calc_obj_mass(env_id, obj_id)

    # Command submission and simulation settings/time.

    def get_control_mode(self):
        return self._control_mode

    def set_cmd(self, obj_id, cmd):
        return self._control.set_cmd(obj_id, cmd)

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        return self._control.set_body_forces(env_id, obj_id, body_id, forces)

    def get_timestep(self):
        return self._runtime.timestep

    def get_sim_timestep(self):
        return self._runtime.sim_timestep

    def get_sim_time(self):
        return self._runtime.get_sim_time()

    # Display, interaction callbacks and recording.

    def get_viewer(self):
        """Expose the viewer to opt-in tool adapters without changing its owner."""
        return self._presentation.get_viewer()

    def set_camera_pose(self, pos, look_at):
        return self._presentation.set_camera_pose(pos, look_at)

    def get_camera_pos(self):
        return self._presentation.get_camera_pos()

    def get_camera_dir(self):
        return self._presentation.get_camera_dir()

    def set_viewer_world_offsets(self, offsets):
        """Change display placement only; preserve the picking array's identity."""
        return self._presentation.set_viewer_world_offsets(offsets)

    def set_body_color(self, env_id, obj_id, body_id, color, viewer=None):
        return self._presentation.set_body_color(env_id, obj_id, body_id, color, viewer)

    def init_viewer(self, viewer):
        return self._presentation.init_viewer(viewer)

    def draw_lines(self, env_id, start_verts, end_verts, cols, line_width):
        return self._presentation.draw_lines(env_id, start_verts, end_verts, cols, line_width)

    def register_keyboard_callback(self, key_str, callback_func):
        return self._presentation.register_keyboard_callback(key_str, callback_func)

    def register_ui_callback(self, callback_func, position="side"):
        """Expose Newton's optional UI extension without exposing its viewer."""
        return self._presentation.register_ui_callback(callback_func, position)

    def get_viewer_paused(self):
        """Read the viewer's pause control for an opt-in playback controller."""
        return self._presentation.get_viewer_paused()

    def set_viewer_paused(self, paused):
        """Keep opt-in playback controls synchronized with Newton's pause UI."""
        return self._presentation.set_viewer_paused(paused)

    def enabled_record_video(self):
        return self._presentation.enabled_record_video()

    def get_video_recording(self):
        return self._presentation.get_video_recording()

    def start_video_recording(self):
        return self._presentation.start_video_recording()

    def stop_video_recording(self):
        return self._presentation.stop_video_recording()
