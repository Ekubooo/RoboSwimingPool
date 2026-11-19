"""Host optional panels on Newton's existing ImGui frame and input hooks."""


class NewtonGuiHost:
    def __init__(self, gui, close):
        self._base = gui
        self._close = close
        self._prepared = False
        self._uses_native_frame = False
        self._panel = None
        self.selection_callback = None
        self.selection_cancel = None

    def __getattr__(self, name):
        return getattr(self._base, name)

    @property
    def show_ui(self):
        return self._base.show_ui

    @show_ui.setter
    def show_ui(self, value):
        self._base.show_ui = value

    def set_selection_callback(self, callback, *, cancel_callback=None):
        self.selection_callback = callback
        self.selection_cancel = cancel_callback
        if callback is not None:
            self.show_ui = True
        self.clear_pressed_keys()

    def set_panel(self, panel):
        self.clear_pressed_keys()
        self._panel = panel

    def handle_key_press(self, symbol, close_fn=None):
        if self.selection_callback is not None:
            from pyglet.window import key

            if symbol == key.ESCAPE and self.selection_cancel is not None:
                self.selection_cancel()
            return
        if self._panel is not None and self._panel.handle_key_press(
                symbol, captured=self.is_keyboard_capturing()):
            return
        self._base.handle_key_press(symbol, close_fn=self._close)

    def update_camera_from_keys(self, dt, is_key_down):
        if self.selection_callback is None:
            if self._panel is not None:
                key_map = self._panel.camera_key
                self._base.update_camera_from_keys(dt, lambda symbol: is_key_down(key_map(symbol)))
            else:
                self._base.update_camera_from_keys(dt, is_key_down)

    def handle_key_release(self, symbol):
        if self._panel is not None:
            self._panel.handle_key_release(symbol)

    def clear_pressed_keys(self):
        if self._panel is not None:
            self._panel.clear_pressed_keys()

    def prepare_frame(self, update_fps=True):
        self._prepared = False
        self._uses_native_frame = self.selection_callback is None and self._panel is None
        if self._uses_native_frame:
            self._base.prepare_frame(update_fps=update_fps)
            return
        if not self.is_available:
            return
        loading = self._base._loading_splash_active
        if not self.show_ui and not loading and self.selection_callback is None:
            return
        self.ui.begin_frame()
        try:
            imgui = self.ui.imgui
            if self.selection_callback is not None:
                self.selection_callback(imgui)
            elif self.show_ui and not loading:
                self._panel.draw(imgui, self.ui.dpi_scale, self._base._ui_callbacks["side"])
                for callback in self._base._ui_callbacks["free"]:
                    callback(imgui)
            if loading:
                self._base._render_loading_splash()
        finally:
            self.ui.end_frame()
        self._prepared = True

    def render_prepared_frame(self):
        if self._uses_native_frame:
            self._base.render_prepared_frame()
        elif self._prepared:
            self.ui.render()
            self._prepared = False

    def render_frame(self, update_fps=True):
        self.prepare_frame(update_fps=update_fps)
        self.render_prepared_frame()
