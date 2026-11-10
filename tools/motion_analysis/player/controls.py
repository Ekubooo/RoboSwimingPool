"""PCA playback controls, independent of the simulator and OpenGL lifecycle."""

from collections import deque
from typing import Any

from .view import VIEW_LABELS, VIEW_MODES, cumulative_selections


class PlaybackControls:
    """Hold UI edits and queue requests for the playback loop to apply safely."""

    MODES = ("cumulative", "single", "subset")
    MODE_LABELS = ("First K components", "Single component", "Selected components")

    def __init__(self, session: Any, initial_indices: tuple[int, ...], mode: str,
                 *, view: str = "comparison"):
        if mode not in self.MODES:
            raise ValueError(f"Unknown selection mode: {mode}")
        if view not in VIEW_MODES:
            raise ValueError(f"Unknown playback view: {view}")
        self.session = session
        self.view = view
        self.mode = mode
        self.active_indices = tuple(initial_indices)
        self.k = (initial_indices[0] + 1 if mode == "single" and initial_indices
                  else len(initial_indices))
        self.component_text = ",".join(str(index + 1) for index in initial_indices)
        self.paused = False
        self.time = 0.0
        self.status = "Ready. Selection changes are kept in memory until Save."
        self.error = ""
        self._imgui = None
        self._pending_indices: tuple[int, ...] | None = None
        self._actions: deque[str] = deque()
        self._save_requested = False
        self._pending_view: str | None = None
        self._camera_reset = False

    def request_view(self, view: str) -> None:
        if view not in VIEW_MODES:
            raise ValueError(f"Unknown playback view: {view}")
        if view != self.view:
            self._pending_view = view

    def take_view_request(self) -> str | None:
        view, self._pending_view = self._pending_view, None
        return view

    def request_camera_reset(self) -> None:
        self._camera_reset = True

    def take_camera_reset(self) -> bool:
        requested, self._camera_reset = self._camera_reset, False
        return requested

    @property
    def captures_keyboard(self) -> bool:
        """Prevent playback shortcuts from consuming text-field input."""
        return bool(self._imgui is not None and
                    self._imgui.get_io().want_capture_keyboard)

    @property
    def selected_variance(self) -> float:
        return float(sum(self.session.variance_ratio[index]
                         for index in self.active_indices))

    def request_selection(self) -> bool:
        """Validate the edited selection without doing PCA or disk I/O in ImGui."""
        from ..core.selection import parse_components, resolve_selection

        try:
            components = parse_components(self.component_text) if self.mode == "subset" else ()
            indices = resolve_selection(
                self.mode, self.k, components, self.session.max_components)
        except (TypeError, ValueError) as exc:
            self.error = str(exc)
            return False
        self._pending_indices = tuple(indices)
        self.error = ""
        return True

    def set_mode(self, mode: str) -> None:
        """Enter a mode without discarding text while already editing a subset."""
        if mode not in self.MODES:
            raise ValueError(f"Unknown selection mode: {mode}")
        if mode == self.mode:
            return
        self.mode = mode
        if mode == "subset":
            self.component_text = ",".join(str(index + 1) for index in self.active_indices)
            return
        if mode == "single":
            self.k = max(1, self.k) if self.session.max_components else 0
        self.request_selection()

    def take_selection(self) -> tuple[int, ...] | None:
        indices, self._pending_indices = self._pending_indices, None
        return indices

    def selection_applied(self, indices: tuple[int, ...]) -> None:
        self.active_indices = tuple(indices)
        self.error = ""
        self.status = "Selection updated in memory. Save exports this reconstruction."

    def request_action(self, action: str) -> None:
        if action not in {"toggle", "play", "pause", "step", "restart"}:
            raise ValueError(f"Unknown playback action: {action}")
        self._actions.append(action)

    def take_actions(self) -> tuple[str, ...]:
        actions = tuple(self._actions)
        self._actions.clear()
        return actions

    def request_save(self) -> None:
        self._save_requested = True

    def take_save_request(self) -> bool:
        requested, self._save_requested = self._save_requested, False
        return requested

    def draw(self, imgui: Any) -> None:
        """Newton's public UI callback supplies its existing ImGui context."""
        if self._imgui is None:
            # This tool has no persistent layout preference; do not create an
            # imgui.ini in the repository's working directory on viewer exit.
            imgui.get_io().set_ini_filename(None)
        self._imgui = imgui
        self._draw_contents(imgui)

    def _draw_contents(self, imgui: Any) -> None:
        """Draw the PCA section using the engine's shared default font."""
        imgui.separator()
        imgui.text("PCA motion comparison")
        changed, view_index = imgui.combo("View", VIEW_MODES.index(self.view), list(VIEW_LABELS))
        if changed:
            self.request_view(VIEW_MODES[view_index])
        layout = getattr(self.session, "layout", None)
        if layout is not None:
            imgui.text(f"Analysis: {layout.mode} PCA")
        imgui.text_wrapped(f"Source: {self.session.source.path.name}")
        imgui.text(f"Time: {self.time:.3f} s")
        if self.view != "comparison":
            imgui.text_wrapped("Fixed root position, height and rotation. Camera locked.")
            if imgui.button("Restore standard camera"):
                self.request_camera_reset()

        if imgui.button("Play" if self.paused else "Pause"):
            self.request_action("play" if self.paused else "pause")
        imgui.same_line()
        if imgui.button("Step"):
            self.request_action("step")
        imgui.same_line()
        if imgui.button("Restart"):
            self.request_action("restart")

        if self.view == "classroom":
            imgui.text_wrapped("Top / Blue: full local joint motion")
            for position, color, indices in zip(
                    ("Bottom left", "Bottom center", "Bottom right"),
                    ("Orange", "Green", "Purple"), cumulative_selections(self.session.max_components)):
                variance = float(sum(self.session.variance_ratio[i] for i in indices))
                imgui.text_wrapped(f"{position} / {color}: K={len(indices)}, variance {variance:.2%}")
            imgui.text_wrapped("Each reconstruction includes the mean. Custom selection is preserved.")
        else:
            imgui.text_wrapped("Blue: original | Orange: reconstruction")
            self._draw_selection(imgui)

        cache_state = "loaded from cache" if self.session.fit_cache_hit else "fitted this run"
        imgui.text(f"PCA model: {cache_state}")
        imgui.text_wrapped(self.status)
        if self.error:
            imgui.text_wrapped(f"Error: {self.error}")
        imgui.text_wrapped("Enter: play/pause | Space: one step")

    def _draw_selection(self, imgui: Any) -> None:
        changed, mode_index = imgui.combo(
            "Selection mode", self.MODES.index(self.mode), list(self.MODE_LABELS))
        if changed:
            self.set_mode(self.MODES[mode_index])

        maximum = self.session.max_components
        if self.mode == "subset":
            _, self.component_text = imgui.input_text("Components", self.component_text)
            imgui.text_wrapped("Examples: 1,3,5-8. Empty selection means mean reconstruction.")
            if imgui.button("Apply selection"):
                self.request_selection()
        elif maximum > 0:
            minimum = 1 if self.mode == "single" else 0
            self.k = min(max(self.k, minimum), maximum)
            slider_width = imgui.calc_item_width()
            changed, self.k = imgui.slider_int("K", self.k, minimum, maximum)
            gap = imgui.get_style().item_spacing.x
            button_size = imgui.ImVec2(max(1.0, (slider_width - gap) / 2.0), 0.0)
            if imgui.button("-1", button_size):
                self.k = max(minimum, self.k - 1)
                changed = True
            imgui.same_line()
            if imgui.button("+1", button_size):
                self.k = min(maximum, self.k + 1)
                changed = True
            if changed:
                self.request_selection()
        else:
            imgui.text("Constant data: mean features only.")

        if imgui.button("Mean features only"):
            self.mode, self.k = "cumulative", 0
            self.request_selection()
        if imgui.button("Save reconstruction"):
            self.request_save()

        label = ", ".join(str(index + 1) for index in self.active_indices) or "mean only"
        imgui.text_wrapped(f"Active components: {label}")
        imgui.text(f"Selected variance: {self.selected_variance:.2%}")
