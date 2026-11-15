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
        self.fluid_controls = None
        self._pressed_keys = set()

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

        toggles = (key.ENTER, key.SPACE, key.R, key.G, key.Q, key.E, key.X)
        if symbol in toggles:
            if symbol in self._pressed_keys:
                return
            self._pressed_keys.add(symbol)
        if self.is_keyboard_capturing():
            return
        viewer = self._base._viewer
        if symbol == key.SPACE:
            viewer._paused = not viewer._paused
        elif symbol == key.ENTER:
            viewer._paused = True
            viewer._step_requested = True
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
                    self.request_reset()
            imgui.separator()
            pipeline = getattr(viewer, "fluid_pipeline", None)
            if pipeline is not None:
                from .fluid.config import MODES, SSFR_PRESETS, WATER_COLORS
                changed, selection = imgui.combo("Fluid rendering", MODES.index(pipeline.mode), list(MODES))
                if changed:
                    pipeline.request(MODES[selection])
                if pipeline.mode == "Transparent":
                    config = pipeline.config
                    changed, value = imgui.checkbox("Depth shading (automatic range)", config.transparent_depth_enabled)
                    if changed:
                        pipeline.request_transparent(depth_enabled=value)
                    imgui.begin_disabled(not config.transparent_depth_enabled)
                    changed, value = imgui.slider_float("Depth strength", config.transparent_depth_strength, 0., 1.)
                    if changed:
                        pipeline.request_transparent(strength=value)
                    imgui.end_disabled()
                    changed, value = imgui.slider_int("Minimum opacity (%)", round(config.transparent_alpha_low*100), 60, 80,
                                                      flags=imgui.SliderFlags_.always_clamp)
                    if changed:
                        pipeline.request_transparent(minimum=value)
                    changed, value = imgui.slider_int("Maximum opacity (%)", round(config.transparent_alpha_high*100), 80, 100,
                                                      flags=imgui.SliderFlags_.always_clamp)
                    if changed:
                        pipeline.request_transparent(maximum=value)
                if pipeline.mode == "SSFR":
                    changed, selection = imgui.combo("SSFR material", SSFR_PRESETS.index(pipeline.ssfr_preset),
                                                     list(SSFR_PRESETS))
                    if changed:
                        pipeline.request_ssfr_preset(SSFR_PRESETS[selection])
                    if pipeline.ssfr_preset == "Water":
                        changed, selection = imgui.combo("Water color", WATER_COLORS.index(pipeline.water_color),
                                                         list(WATER_COLORS))
                        if changed:
                            pipeline.request_water_color(WATER_COLORS[selection])
                if pipeline.error:
                    imgui.text_wrapped(pipeline.error)
            for callback in self._base._ui_callbacks["side"]:
                callback(imgui)
            controls = self.fluid_controls
            if controls is not None:
                imgui.separator()
                imgui.push_item_width(160 * scale)
                changed, strength = imgui.slider_float("Fluid gravity (m/s^2)", controls.gravity_strength, 0., 20.,
                                                       flags=imgui.SliderFlags_.always_clamp)
                if changed:
                    controls.set_gravity_strength(strength, commit=False)
                if imgui.is_item_deactivated_after_edit():
                    controls.commit_gravity()
                direction = next((label for value, label in zip(controls.direction, ("X", "Y", "Z")) if value), "Z")
                sign = "+" if sum(controls.direction) > 0 else "-"
                imgui.text_disabled(f"Direction: {sign}{direction}   G: reverse   Q/E: left/right")
                changed, feedback = imgui.slider_float("Fluid feedback", controls.feedback_scale, 0., 2.,
                                                       format="%.1f",
                                                       flags=imgui.SliderFlags_.always_clamp)
                if changed:
                    # Keep both mouse dragging and typed values on 0.1 steps.
                    controls.set_feedback_scale(round(feedback * 10) / 10)
                changed, slow = imgui.checkbox("Fluid slow motion x0.5 (X)", controls.slow_motion)
                if changed:
                    controls.set_slow_motion(slow)
                imgui.pop_item_width()
                if controls.error:
                    imgui.text_wrapped(f"Fluid update failed: {controls.error}")
            imgui.text_disabled("Space: pause/play  Enter: step  R: reset")
        imgui.end()
