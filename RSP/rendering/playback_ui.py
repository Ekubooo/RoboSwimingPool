"""Small playback panel composed with Newton's existing GUI and GL context.

Newton still owns input capture, camera gestures, picking, fonts and the UI
renderer. Only panel construction and playback keys are replaced. No model
arrays are inspected by this panel; callbacks should edit CPU settings or
enqueue commands, rather than resetting the simulation during a draw call.
"""


class PlaybackGui:
    def __init__(self, gui):
        self._base = gui
        self._prepared = False

    def __getattr__(self, name):
        # Keep the version-specific ViewerGui interface at this adapter boundary.
        return getattr(self._base, name)

    @property
    def show_ui(self):
        return self._base.show_ui

    @show_ui.setter
    def show_ui(self, value):
        self._base.show_ui = value

    def handle_key_press(self, symbol, close_fn=None):
        if self.is_keyboard_capturing():
            return
        from pyglet.window import key

        viewer = self._base._viewer
        if symbol == key.ENTER:
            viewer._paused = not viewer._paused
        elif symbol == key.SPACE:
            viewer._paused = True
            viewer._step_requested = True
        elif symbol == key.R:
            callback = viewer._reset_callback
            if callback is not None:
                callback()
        else:
            self._base.handle_key_press(symbol, close_fn=close_fn)

    def prepare_frame(self, update_fps=True):
        self._prepared = False
        if not self.is_available:
            return
        loading = self._base._loading_splash_active
        if not self.show_ui and not loading:
            return
        self.ui.begin_frame()
        if self.show_ui:
            self._render_panel()
            for callback in self._base._ui_callbacks["free"]:
                callback(self.ui.imgui)
        if loading:
            self._base._render_loading_splash()
        self.ui.end_frame()
        self._prepared = True

    def render_prepared_frame(self):
        if self._prepared:
            self.ui.render()
            self._prepared = False

    def render_frame(self, update_fps=True):
        self.prepare_frame(update_fps=update_fps)
        self.render_prepared_frame()

    def _render_panel(self):
        viewer, imgui = self._base._viewer, self.ui.imgui
        scale = self.ui.dpi_scale
        imgui.set_next_window_pos(imgui.ImVec2(10 * scale, 10 * scale), imgui.Cond_.first_use_ever)
        if imgui.begin("RSP Playback", flags=imgui.WindowFlags_.always_auto_resize):
            _, viewer._paused = imgui.checkbox("Pause", viewer._paused)
            imgui.same_line()
            imgui.begin_disabled(not viewer._paused)
            if imgui.button("Step"):
                viewer._step_requested = True
            imgui.end_disabled()
            if viewer._reset_callback is not None:
                imgui.same_line()
                if imgui.button("Reset"):
                    viewer._reset_callback()
            imgui.separator()
            _, viewer.show_visual = imgui.checkbox("Rigid geometry", viewer.show_visual)
            _, viewer.show_particles = imgui.checkbox("Fluid particles", viewer.show_particles)
            imgui.push_item_width(140 * scale)
            changed, fov = imgui.slider_float("Camera FOV", viewer.camera.fov, 15., 90.)
            if changed:
                viewer.camera.fov = fov
                viewer._camera_dirty = True
            _, viewer.camera_speed = imgui.slider_float("Camera speed", viewer.camera_speed, .1, 10.)
            imgui.pop_item_width()
            imgui.separator()
            for callback in self._base._ui_callbacks["side"]:
                callback(imgui)
            imgui.text_disabled("Enter: pause  Space: step  R: reset")
            imgui.text_disabled("Mouse / WASD: camera  H: hide UI")
            imgui.text_disabled("Right-drag: pull a body")
        imgui.end()
