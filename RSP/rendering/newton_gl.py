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
        self._rigid_batches = {}
        self._billboard_closed = False
        self._billboard_settings = {"speed_color_mid": speed_color_mid, "speed_color_max": speed_color_max}
        super().__init__(*args, **kwargs)
        if self.gui is not None:
            from .playback_ui import PlaybackGui
            self.gui = PlaybackGui(self.gui)
        # RSP's application already owns pacing. Newton's requested 1 ms wait
        # can become a Windows timer tick, masking faster GPU uploads. Adapt
        # this renderer instance only; retain all event and camera processing.
        self.renderer.update = self._poll_events
        # The OS close event can destroy the context before the engine's finally
        # block runs. Release interop resources while that context still exists.
        self.renderer.window.push_handlers(on_close=self._close_window)

    def _poll_events(self):
        import ctypes
        import pyglet

        self.renderer._make_current()
        if not self.renderer.headless:
            pyglet.clock.tick()
            self.renderer.app.platform_event_loop.step(0.0)
            try:
                self.renderer.window.dispatch_events()
            except (ctypes.ArgumentError, TypeError):
                # Same xlib compatibility exception as Newton's event pump.
                pass

    def set_model(self, model):
        super().set_model(model)
        if model is not None and self.picking is not None:
            from .picking import VisiblePickModel
            self.picking.model = VisiblePickModel(model)

    def _drop_rigid_batch(self):
        entry = self._rigid_batches.pop(id(getattr(self, "layer", None)), None)
        if entry is not None:
            batch, groups = entry
            batch.close()
            for name, _shapes in groups:
                self.objects.pop(name, None)

    def _build_packed_vbo_arrays(self):
        """Replace Newton's pinned-host upload route with one shared GPU buffer."""
        import numpy as np
        from .rigid import RigidBatch

        self.renderer._make_current()
        self._drop_rigid_batch()
        if self.model is None:
            return
        ids, extras, scales, materials, groups = [], [], [], [], []
        identity = (0., 0., 0., 0., 0., 0., 1.)
        for shapes in self._shape_instances.values():
            shape_ids = list(shapes.model_shapes)
            if not shape_ids:
                continue
            group_scales = shapes.scales.numpy()
            group_materials = shapes.materials.numpy()
            parts = [(shapes.name, shapes.mesh, shape_ids, [identity] * len(shape_ids), group_scales, group_materials)]
            if shapes.geo_type == newton.GeoType.CAPSULE:
                sphere = self._qualify("/geometry/_capsule_instancer/sphere")
                cylinder = self._qualify("/geometry/_capsule_instancer/cylinder")
                if sphere not in self.objects:
                    self.log_geo(sphere, newton.GeoType.SPHERE, (1.,), 0., True, hidden=True)
                if cylinder not in self.objects:
                    self.log_geo(cylinder, newton.GeoType.CYLINDER, (1., 1.), 0., True, hidden=True)
                cap_extras = [(0., 0., sign * float(s[2]), 0., 0., 0., 1.)
                              for s in group_scales for sign in (-1, 1)]
                cap_scales = [(float(s[0]),) * 3 for s in group_scales for _ in range(2)]
                parts = [(shapes.name + "/capsule_cylinder", cylinder, shape_ids,
                          [identity] * len(shape_ids), group_scales, group_materials),
                         (shapes.name + "/capsule_caps", sphere, np.repeat(shape_ids, 2),
                          cap_extras, cap_scales, np.repeat(group_materials, 2, axis=0))]
            for name, mesh, part_ids, part_extra, part_scale, part_material in parts:
                offset = len(ids)
                ids.extend(part_ids)
                extras.extend(part_extra)
                scales.extend(part_scale)
                materials.extend(part_material)
                groups.append((name, shapes, mesh, offset, len(part_ids)))
        batch = RigidBatch(self.model, ids, extras, scales, materials)
        try:
            for name, shapes, mesh, offset, count in groups:
                self.objects[name] = batch.add_drawable(self.objects[mesh], offset, count,
                    transparent=shapes.transparent, draw_edge=shapes.geo_type != newton.GeoType.PLANE)
            self._rigid_batches[id(self.layer)] = (batch, [(name, shapes) for name, shapes, *_ in groups])
        except BaseException:
            batch.close()
            for name, *_ in groups:
                self.objects.pop(name, None)
            raise

    def invalidate_rigid_appearance(self):
        """Call after changing opacity across render passes or mesh topology.

Colors and opacity values themselves are read on GPU every frame. CPU-side
pass classification is refreshed only on this explicit edit notification.
"""
        self.renderer._make_current()
        self._drop_rigid_batch()
        self._rebuild_shape_batches_for_opacity_groups()
        if self.picking is not None:
            from .picking import VisiblePickModel
            self.picking.model = VisiblePickModel(self.model)

    def log_state(self, state):
        self._last_state = state
        if self.model is None:
            return
        self.renderer._make_current()
        batch, groups = self._rigid_batches[id(self.layer)]
        try:
            batch.update(state, self.world_offsets, self.layer.xform)
        finally:
            hidden_layer = self._layer_force_hidden()
            for name, shapes in groups:
                self.objects[name].hidden = (not batch.valid or hidden_layer or not self._should_show_shape(
                    shapes.flags, shapes.static, shapes.geo_type))
        # Keep Newton's optional diagnostics and our billboard hook. The normal
        # rigid path never enters log_capsules/update_from_pinned/color.numpy.
        self._log_gaussian_shapes(state)
        self._log_non_shape_state(state)
        self.model_changed = False
        self._render_picking_line(state)

    def _log_com(self, state):
        # Newton updates and uploads hidden COM markers as well. Keep the UI
        # feature, but incur that diagnostic work only when it is displayed.
        if self.show_com and not self._layer_force_hidden():
            return super()._log_com(state)
        marker = self.objects.get(self._qualify("/model/com"))
        if marker is not None:
            marker.hidden = True

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
        self._drop_rigid_batch()
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
            for batch, _groups in self._rigid_batches.values():
                batch.close()
            self._rigid_batches.clear()
            # ViewerGL destroys geometry (and our CUDA registration) first,
            # then closes its window. No parallel renderer/window is created.
            super().close()
