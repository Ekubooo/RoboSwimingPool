"""Opt-in Newton GL compatibility adapter for PCA display visibility.

Newton currently has no public per-world visibility setter. All access to its
shape batches and camera-input hook is isolated here. Shapes are hidden with
zero *render* scales; model scales, motion frames and simulation state never
change. Normal ViewerGL uploads remain responsible for visible geometry.
"""

from typing import Any, Callable

import numpy as np

from .view import ViewLayout


class PlaybackRenderer:
    def __init__(self, engine: Any, enforce_camera: Callable[[], None]):
        import warp as wp

        self.engine = engine
        self.viewer = engine.get_viewer()
        if self.viewer is None or not all(hasattr(self.viewer, name) for name in
                                           ("_shape_instances", "_update_camera", "world_offsets")):
            raise RuntimeError("PCA display modes require a Newton ViewerGL with shape-batch support.")
        self.enforce_camera = enforce_camera
        self.visible = (True, True, False, False)
        self.local = False
        self._masked_batches: list[tuple[Any, Any]] = []
        self._line_visibility: dict[str, bool] = {}
        self._log_state = self.viewer.log_state
        self._update_camera = self.viewer._update_camera
        self._draw_lines = engine.draw_lines
        self._wp = wp
        self.viewer.log_state = self.log_state
        self.viewer._update_camera = self.update_camera
        engine.draw_lines = self.draw_lines

    def configure(self, layout: ViewLayout, view: str) -> None:
        self.visible = layout.visible
        self.local = view != "comparison"
        self.engine.set_viewer_world_offsets(layout.offsets)
        for name, lines in self.viewer.lines.items():
            self._line_visibility.setdefault(name, lines.hidden)
            lines.hidden = True if self.local else self._line_visibility[name]
        self._masked_batches.clear()
        for batch in self.viewer._shape_instances.values():
            worlds = batch.worlds.numpy()
            enabled = np.array([self.visible[int(w)] if w >= 0 else view != "classroom"
                                for w in worlds])
            if not enabled.all():
                scales = batch.scales.numpy().copy()
                scales[~enabled] = 0.0
                masked = self._wp.array(scales, dtype=self._wp.vec3, device=self.viewer.device)
                self._masked_batches.append((batch, masked))

    def log_state(self, state: Any) -> None:
        self._log_state(state)
        # The standard upload also restores previously masked instances when
        # switching back to four characters. Re-upload only affected batches.
        import newton

        for batch, scales in self._masked_batches:
            batch.update(state, self.viewer.world_offsets)
            hidden = not self.viewer._should_show_shape(batch.flags, batch.static)
            draw = (self.viewer.log_capsules if batch.geo_type == newton.GeoType.CAPSULE
                    else self.viewer.log_instances)
            draw(batch.name, batch.mesh, batch.world_xforms, scales, None, None, hidden=hidden)

    def update_camera(self, dt: float) -> None:
        # ViewerGL processes mouse events before this hook and draws afterwards.
        # Reasserting the fixed pose here also discards mouse orbit/zoom changes.
        if self.local:
            self.enforce_camera()
        else:
            self._update_camera(dt)

    def draw_lines(self, env_id, *args, **kwargs):
        if not self.local and (env_id is None or self.visible[env_id]):
            return self._draw_lines(env_id, *args, **kwargs)

    def close(self) -> None:
        self.viewer.log_state = self._log_state
        self.viewer._update_camera = self._update_camera
        self.engine.draw_lines = self._draw_lines
        for name, hidden in self._line_visibility.items():
            if name in self.viewer.lines:
                self.viewer.lines[name].hidden = hidden
        self._masked_batches.clear()
