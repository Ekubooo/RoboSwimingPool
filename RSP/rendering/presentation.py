"""Display a Newton model and its state through one Viewer."""

import numpy as np
import pyglet
import torch
import warp as wp


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


class Presentation:
    """Own display and input callbacks, borrowing the simulation model and state."""

    def __init__(self, num_envs, env_spacing, visualize, *, viewer=None, create_viewer=None):
        self._num_envs = num_envs
        self._env_spacing = env_spacing
        self._viewer = viewer
        self._owns_viewer = viewer is None
        self._keyboard_binding = None
        if viewer is not None and not visualize:
            raise ValueError("A borrowed viewer requires visualization")
        self._closed = False
        self._keyboard_callbacks = {}
        if visualize:
            self._build_viewer(create_viewer)

    def initialize(self, model, object_body_ids, colors):
        self._sim_model = model
        self._object_body_ids = object_body_ids
        self._obj_colors = colors
        if self._viewer is not None:
            self.init_viewer(self._viewer)
            self._camera_offsets = self._viewer.world_offsets.numpy()

    def apply_forces(self, raw_state):
        if self._viewer is not None:
            self._viewer.apply_forces(raw_state)

    def render(self, sim_time, raw_state, draw_scene=None):
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        self._viewer.end_frame()
        # Window events can destroy the GL context during end_frame().
        if (not self._viewer.is_running()):
            raise SystemExit(0)

        self._viewer.begin_frame(sim_time)
        self._viewer.log_state(raw_state)
        if draw_scene is not None:
            draw_scene(self._viewer)

        self._draw_line_count = 0

        return

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        viewer = getattr(self, "_viewer", None)
        try:
            if viewer is not None:
                if self._keyboard_binding is not None:
                    viewer.unbind_keyboard_callback(self._keyboard_binding)
                if self._owns_viewer:
                    viewer.close()
        finally:
            self._keyboard_binding = None
            self._viewer = None

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
        body_idx = self._object_body_ids[env_id][obj_id][body_id]

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
        # Configure the embedded UI without changing the installed Newton package.
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
            self._viewer.set_paused(paused)

    def _build_viewer(self, create_viewer):
        if self._viewer is None:
            self._viewer = create_viewer()
        self._draw_line_count = 0
        self._keyboard_binding = self._viewer.bind_keyboard_callback(self._on_keyboard_event)

    def _apply_obj_colors(self, viewer):
        num_envs = self._num_envs
        objs_per_env = len(self._obj_colors[0])

        for env_id in range(num_envs):
            for obj_id in range(objs_per_env):
                obj_col = self._obj_colors[env_id][obj_id]

                if (obj_col is not None):
                    num_bodies = len(self._object_body_ids[env_id][obj_id])
                    for body_id in range(num_bodies):
                        self.set_body_color(env_id, obj_id, body_id, obj_col, viewer)
        return

    def _on_keyboard_event(self, symbol, modifiers):
        if (symbol in self._keyboard_callbacks):
            callback = self._keyboard_callbacks[symbol]
            callback()
        return
