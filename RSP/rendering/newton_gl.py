"""Newton 1.7 ViewerGL adapter for the independent billboard batch.

Newton has no public custom-pass API. Its scene object protocol (render/destroy,
cast_shadow/draw_edge/has_transparency) is the only integration point used here.
Keep that version-specific knowledge here instead of patching RendererGL or
changing Presentation's frame order. The opaque pass shares rigid-body depth
and precedes transparent compositing, MSAA resolution, and the existing UI.
"""

import newton


class _BillboardObject:
    cast_shadow = False
    draw_edge = False
    has_transparency = False

    def __init__(self, renderer, device, **settings):
        # Non-visual backends may import this adapter without GL dependencies.
        from .billboard import BillboardBatch

        self.renderer = renderer
        self.hidden = False
        self.batch = BillboardBatch(device, **settings)

    def render(self):
        if not self.hidden:
            # RendererGL updates these matrices from the camera before drawing.
            self.batch.draw(self.renderer._view_matrix, self.renderer._projection_matrix)

    def destroy(self):
        self.renderer._make_current()
        self.batch.close()


class NewtonBillboardViewer(newton.viewer.ViewerGL):
    """Native Newton rigid/UI rendering with CUDA billboard particles."""

    def __init__(self, *args, speed_color_mid=2.0, speed_color_max=6.5, **kwargs):
        self._billboard_closed = False
        self._billboard_settings = {"speed_color_mid": speed_color_mid, "speed_color_max": speed_color_max}
        super().__init__(*args, **kwargs)
        # The OS close event can destroy the context before the engine's finally
        # block runs. Release interop resources while that context still exists.
        self.renderer.window.push_handlers(on_close=self._close_window)

    def _log_particles(self, state):
        name = self._qualify("/model/particles")
        drawable = self.objects.get(name)
        hidden = not self.show_particles or self._layer_force_hidden() or not self.model.particle_count
        if hidden:
            if drawable is not None:
                drawable.hidden = True
            return
        self.renderer._make_current()
        if drawable is None:
            drawable = _BillboardObject(self.renderer, self.device, **self._billboard_settings)
            self.objects[name] = drawable
        points = self._apply_layer_transform_to_points(state.particle_q)
        drawable.batch.update(points, state.particle_qd, self.model.particle_radius)
        drawable.hidden = False

    def clear_model(self):
        renderer = getattr(self, "renderer", None)
        if renderer is not None:
            renderer._make_current()
        # Newton calls destroy() on layer-owned scene objects, including ours.
        super().clear_model()

    def _close_window(self):
        from pyglet.event import EVENT_HANDLED

        self.close()
        return EVENT_HANDLED

    def close(self):
        if getattr(self, "_billboard_closed", False):
            return
        self._billboard_closed = True
        renderer = getattr(self, "renderer", None)
        window = getattr(renderer, "window", None)
        if window is not None and window.context is not None:
            renderer._make_current()
            # ViewerGL destroys geometry (and our CUDA registration) first,
            # then closes its window. No parallel renderer/window is created.
            super().close()
