"""Small playback panel composed with Newton's existing GUI and GL context.

Newton still owns input capture, camera gestures, picking, fonts and the UI
renderer. Only panel construction and playback keys are replaced. No model
arrays are inspected by this panel; callbacks should edit CPU settings or
enqueue commands, rather than resetting the simulation during a draw call.
"""


def draw_policy_controls(imgui, *, get_scene_sphere, request_scene_sphere):
    """Display CPU policy options and queue edits without updating simulation."""
    changed, enabled = imgui.checkbox("Use scene sphere", get_scene_sphere())
    if changed:
        request_scene_sphere(enabled)


class PlaybackGui:
    def __init__(self, gui):
        self._base = gui
        self._prepared = False
        self.fluid_controls = None
        self._playback_fps = None
        self._pressed_keys = set()
        self.selection_callback = None
        self.selection_cancel = None

    def set_selection_callback(self, callback, *, cancel_callback=None):
        """Select the startup panel without registering persistent model callbacks."""
        self.selection_callback = callback
        self.selection_cancel = cancel_callback
        if callback is not None:
            self.show_ui = True
        self.clear_pressed_keys()

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
        from pyglet.window import key

        if self.selection_callback is not None:
            if symbol == key.ESCAPE and self.selection_cancel is not None:
                self.selection_cancel()
            return

        toggles = (key.ENTER, key.SPACE, key.R, key.G, key.Q, key.E, key.X)
        if symbol in toggles:
            if symbol in self._pressed_keys:
                return
            self._pressed_keys.add(symbol)
        if self.is_keyboard_capturing():
            return
        viewer = self._base._viewer
        if symbol == key.SPACE:
            viewer.set_paused(not viewer.is_paused())
        elif symbol == key.ENTER:
            viewer.request_step()
        elif symbol == key.R:
            self.request_reset()
        elif self.fluid_controls is not None and symbol == key.G:
            self.fluid_controls.reverse_gravity()
        elif self.fluid_controls is not None and symbol in (key.Q, key.E):
            self.fluid_controls.select_horizontal(-1 if symbol == key.Q else 1)
        elif self.fluid_controls is not None and symbol == key.X:
            self.fluid_controls.set_slow_motion(not self.fluid_controls.slow_motion)
        else:
            self._base.handle_key_press(symbol, close_fn=close_fn)

    def update_camera_from_keys(self, dt, is_key_down):
        if self.selection_callback is not None:
            return
        from pyglet.window import key

        # Newton polls held keys separately from the press/release callbacks.
        remap = {key.Q: key.O, key.E: key.P}
        self._base.update_camera_from_keys(dt, lambda symbol: is_key_down(remap.get(symbol, symbol)))

    def handle_key_release(self, symbol, modifiers=0):
        self._pressed_keys.discard(symbol)

    def clear_pressed_keys(self):
        self._pressed_keys.clear()

    def request_reset(self):
        callback = self._base._viewer._reset_callback
        if callback is not None:
            callback()

    def prepare_frame(self, update_fps=True):
        self._prepared = False
        if not self.is_available:
            return
        loading = self._base._loading_splash_active
        if not self.show_ui and not loading and self.selection_callback is None:
            return
        self.ui.begin_frame()
        if self.selection_callback is not None:
            self.selection_callback(self.ui.imgui)
        elif self.show_ui and not loading:
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

    def set_playback_fps(self, fps):
        """Accept a CPU scalar from the runner; never query simulation arrays."""
        self._playback_fps = fps

    @staticmethod
    def _field(imgui, label):
        """Stack a label over a full-width control with a stable hidden ID."""
        imgui.text(label)
        imgui.set_next_item_width(-1.)
        return "##" + label

    def _render_panel(self):
        viewer, imgui = self._base._viewer, self.ui.imgui
        scale = self.ui.dpi_scale
        viewport = imgui.get_main_viewport()
        max_width = max(1., viewport.work_size.x - 20 * scale)
        max_height = max(1., viewport.work_size.y - 20 * scale)
        imgui.set_next_window_pos(imgui.ImVec2(viewport.work_pos.x + 10 * scale,
                                             viewport.work_pos.y + 10 * scale), imgui.Cond_.first_use_ever)
        imgui.set_next_window_size(imgui.ImVec2(min(280 * scale, max_width),
                                              min(640 * scale, max_height)), imgui.Cond_.once)
        imgui.set_next_window_size_constraints(
            imgui.ImVec2(min(240 * scale, max_width), min(160 * scale, max_height)),
            imgui.ImVec2(max_width, max_height))
        visible, _ = imgui.begin("RSP Playback")
        try:
            if not visible:
                return
            paused = viewer.is_paused()
            if paused:
                imgui.text("FPS: — (paused)")
            else:
                imgui.text("FPS: —" if self._playback_fps is None else f"FPS: {self._playback_fps:.1f}")
            changed, paused = imgui.checkbox("Pause", paused)
            if changed:
                viewer.set_paused(paused)
            button_size = imgui.ImVec2(-1., imgui.get_frame_height())
            imgui.begin_disabled(not viewer.is_paused())
            if imgui.button("Step", button_size):
                viewer.request_step()
            imgui.end_disabled()
            if viewer._reset_callback is not None:
                if imgui.button("Reset", button_size):
                    self.request_reset()

            if self.fluid_controls is not None:
                if imgui.collapsing_header("Fluid controls", flags=imgui.TreeNodeFlags_.default_open):
                    self._render_fluid_controls(imgui)
                # Keep failures visible even when the controls are collapsed.
                if self.fluid_controls.error:
                    imgui.text_wrapped(f"Fluid update failed: {self.fluid_controls.error}")
            pipeline = getattr(viewer, "fluid_pipeline", None)
            if pipeline is not None:
                if imgui.collapsing_header("Rendering"):
                    self._render_rendering_controls(imgui, pipeline)
                if pipeline.error:
                    imgui.text_wrapped(pipeline.error)
            if self._base._ui_callbacks["side"] and imgui.collapsing_header("Policy observations"):
                for callback in self._base._ui_callbacks["side"]:
                    callback(imgui)
            if imgui.collapsing_header("Keyboard shortcuts"):
                for text in ("Space: pause / play", "Enter: single step", "R: reset",
                             "G: reverse gravity", "Q / E: gravity left / right", "X: fluid slow motion"):
                    imgui.text_wrapped(text)
        finally:
            imgui.end()

    def _render_fluid_controls(self, imgui):
        controls = self.fluid_controls
        changed, strength = imgui.slider_float(self._field(imgui, "Gravity (m/s²)"),
            controls.gravity_strength, 0., 20., flags=imgui.SliderFlags_.always_clamp)
        if changed:
            controls.set_gravity_strength(strength, commit=False)
        if imgui.is_item_deactivated_after_edit():
            controls.commit_gravity()
        direction = next((label for value, label in zip(controls.direction, ("X", "Y", "Z")) if value), "Z")
        sign = "+" if sum(controls.direction) > 0 else "-"
        imgui.text(f"Direction: {sign}{direction}")
        changed, feedback = imgui.slider_float(self._field(imgui, "Fluid feedback"),
            controls.feedback_scale, 0., 2., format="%.1f", flags=imgui.SliderFlags_.always_clamp)
        if changed:
            controls.set_feedback_scale(round(feedback * 10) / 10)
        changed, slow = imgui.checkbox("Slow motion x0.5", controls.slow_motion)
        if changed:
            controls.set_slow_motion(slow)

    def _render_rendering_controls(self, imgui, pipeline):
        from .fluid.config import MODES, SSFR_PRESETS, WATER_COLORS

        changed, selection = imgui.combo(self._field(imgui, "Fluid rendering"),
                                         MODES.index(pipeline.mode), list(MODES))
        if changed:
            pipeline.request(MODES[selection])
        if pipeline.mode == "Transparent":
            config = pipeline.config
            changed, value = imgui.checkbox("Layer blending", config.transparent_layer_blending)
            if changed:
                pipeline.request_transparent(layer_blending=value)
            changed, value = imgui.checkbox("Depth shading", config.transparent_depth_enabled)
            if imgui.is_item_hovered():
                imgui.set_tooltip("Depth range is determined automatically.")
            if changed:
                pipeline.request_transparent(depth_enabled=value)
            imgui.begin_disabled(not config.transparent_depth_enabled)
            changed, value = imgui.slider_float(self._field(imgui, "Depth strength"),
                                                config.transparent_depth_strength, 0., 1.)
            if changed:
                pipeline.request_transparent(strength=value)
            imgui.end_disabled()
            changed, value = imgui.slider_int(self._field(imgui, "Minimum opacity (%)"),
                round(config.transparent_alpha_low*100), 60, 80, flags=imgui.SliderFlags_.always_clamp)
            if changed:
                pipeline.request_transparent(minimum=value)
            changed, value = imgui.slider_int(self._field(imgui, "Maximum opacity (%)"),
                round(config.transparent_alpha_high*100), 80, 100, flags=imgui.SliderFlags_.always_clamp)
            if changed:
                pipeline.request_transparent(maximum=value)
        if pipeline.mode == "SSFR":
            changed, selection = imgui.combo(self._field(imgui, "SSFR material"),
                                             SSFR_PRESETS.index(pipeline.ssfr_preset), list(SSFR_PRESETS))
            if changed:
                pipeline.request_ssfr_preset(SSFR_PRESETS[selection])
            if pipeline.ssfr_preset == "Water":
                changed, selection = imgui.combo(self._field(imgui, "Water color"),
                                                 WATER_COLORS.index(pipeline.water_color), list(WATER_COLORS))
                if changed:
                    pipeline.request_water_color(WATER_COLORS[selection])
