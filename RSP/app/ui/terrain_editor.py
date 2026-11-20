"""Edit a terrain recipe and preview it in the scene selection window."""

import logging

from RSP.sim_ready.terrain import TerrainRecipe


_TYPES = {
    "pyramid_stairs": "Stairs",
    "random_grid": "Random grid",
    "gap": "Gap",
}
_PARAMETER_LABELS = {
    "step_height": "Step height (m)",
    "max_height": "Maximum height (m)",
    "seed": "Seed",
    "gap_width": "Gap width (m)",
    "depth": "Gap depth (m)",
}


class TerrainEditor:
    """Hold one draft and one preview texture until the modal closes."""

    _TITLE = "Generate terrain"

    def __init__(self):
        self.active = False
        self._open_requested = False
        self._close_requested = False
        self._save_requested = False
        self._texture = None
        self._name = ""
        self._kind = "pyramid_stairs"
        self._parameters = {}
        self._dirty = False
        self._preview_recipe = None
        self._error = None

    def open(self):
        self.active = self._open_requested = True
        self._name = ""
        self._select_kind("pyramid_stairs")

    def request_cancel(self):
        self._save_requested = False
        self._close_requested = True

    def _select_kind(self, kind):
        self._kind = kind
        self._parameters = dict(TerrainRecipe.default(kind).parameters)
        self._invalidate()

    def _invalidate(self):
        self._dirty = True
        self._preview_recipe = None
        self._error = None

    def _release_preview(self, previews):
        if self._texture is not None:
            previews.release_texture(self._texture)
            self._texture = None
        self._preview_recipe = None

    def detach(self, previews):
        self._release_preview(previews)
        self.active = self._open_requested = self._close_requested = self._save_requested = False

    def process(self, previews):
        """Generate, render or save outside the ImGui draw callback."""
        if not self.active or self._close_requested:
            self._release_preview(previews)
            return None
        if self._save_requested:
            self._save_requested = False
            if self._preview_recipe is None:
                return None
            from RSP.sim_ready.pipelines.terrain_authoring import save_terrain

            try:
                asset = save_terrain(self._name.strip(), self._preview_recipe)
            except (OSError, ValueError) as error:
                self._error = str(error)
            else:
                self._close_requested = True
                self._release_preview(previews)
                return asset
        if not self._dirty:
            return None
        self._dirty = False
        self._release_preview(previews)
        from RSP.rendering.preview.geometry import container_geometry
        from RSP.sim_ready.container_geometry import describe_terrain, validate_terrain

        try:
            recipe = TerrainRecipe(self._kind, dict(self._parameters))
            terrain = describe_terrain(recipe=recipe)
            validate_terrain(terrain)
            geometry = container_geometry(terrain.visible_mesh, terrain.layout["size"])
            self._texture = previews.render_geometry(geometry)
            self._preview_recipe = recipe
        except (OSError, ValueError) as error:
            self._error = str(error)
        except Exception as error:
            self._error = str(error)
            logging.exception("Terrain preview failed")
        return None

    def draw(self, imgui, scale):
        if not self.active:
            return
        if self._open_requested:
            imgui.open_popup(self._TITLE)
            self._open_requested = False
        viewport = imgui.get_main_viewport()
        margin = 20. * scale
        width = max(1., min(480. * scale, viewport.work_size.x - margin))
        height = max(1., min(550. * scale, viewport.work_size.y - margin))
        center = imgui.ImVec2(viewport.work_pos.x + viewport.work_size.x * .5,
                             viewport.work_pos.y + viewport.work_size.y * .5)
        imgui.set_next_window_pos(center, pivot=imgui.ImVec2(.5, .5))
        imgui.set_next_window_size(imgui.ImVec2(width, height))
        flags = imgui.WindowFlags_.no_saved_settings | imgui.WindowFlags_.no_move | imgui.WindowFlags_.no_resize
        visible, _ = imgui.begin_popup_modal(self._TITLE, flags=flags)
        if not visible:
            if not imgui.is_popup_open(self._TITLE):
                self.active = self._close_requested = self._save_requested = False
            return
        try:
            if self._close_requested:
                self._close_popup(imgui)
                return
            footer = imgui.get_frame_height_with_spacing() + imgui.get_style().item_spacing.y
            contents = imgui.begin_child("terrain draft", imgui.ImVec2(0, -footer))
            try:
                if contents:
                    self._draw_draft(imgui, scale)
            finally:
                imgui.end_child()
            can_save = self._preview_recipe is not None and bool(self._name.strip())
            imgui.begin_disabled(not can_save)
            if imgui.button("Save terrain") and can_save:
                self._save_requested = True
            imgui.end_disabled()
            imgui.same_line()
            if imgui.button("Cancel"):
                self._close_popup(imgui)
        finally:
            imgui.end_popup()

    def _draw_draft(self, imgui, scale):
        kinds = list(_TYPES)
        changed, index = imgui.combo("Type", kinds.index(self._kind), list(_TYPES.values()))
        if changed:
            self._select_kind(kinds[index])
        for field, (lower, upper) in TerrainRecipe.parameter_bounds(self._kind).items():
            label, value = _PARAMETER_LABELS[field], self._parameters[field]
            if isinstance(value, int):
                changed, value = imgui.input_int(label, value)
            else:
                changed, value = imgui.slider_float(label, value, lower, upper, "%.3f",
                                                   flags=imgui.SliderFlags_.always_clamp)
                value = min(upper, max(lower, round(round(value / .025) * .025, 6)))
            if changed and value != self._parameters[field]:
                self._parameters[field] = value
                self._invalidate()
        _, self._name = imgui.input_text("Name", self._name)
        if self._error:
            imgui.text_wrapped(self._error)
        available = imgui.get_content_region_avail()
        height = max(90. * scale, available.y - imgui.get_style().item_spacing.y)
        width = max(1., min(400. * scale, available.x, height / .75))
        size = imgui.ImVec2(width, width * .75)
        if self._texture is None or self._dirty:
            imgui.dummy(size)
        else:
            imgui.image(imgui.ImTextureRef(self._texture), size, imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))

    def _close_popup(self, imgui):
        imgui.close_current_popup()
        self.active = self._close_requested = self._save_requested = False
