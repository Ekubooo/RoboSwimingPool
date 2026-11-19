"""Card drawing only: selection events and cached texture requests."""
from contextlib import contextmanager


@contextmanager
def card_grid(imgui, name, scale):
    columns = max(1, int(imgui.get_content_region_avail().x / (230*scale)))
    opened = imgui.begin_table(name, columns)
    try:
        yield opened
    finally:
        if opened:
            imgui.end_table()


@contextmanager
def card(imgui, key, height, scale):
    imgui.table_next_column()
    imgui.push_id(key)
    visible = imgui.begin_child('card', imgui.ImVec2(0, height*scale), imgui.ChildFlags_.borders)
    try:
        yield visible
    finally:
        imgui.end_child()
        imgui.pop_id()


def picture(imgui, previews, ref, color, width):
    size = imgui.ImVec2(width, width*.75)
    if not imgui.is_rect_visible(size):
        imgui.dummy(size)
        return
    texture, error = previews.request(ref, color)
    if texture is not None:
        imgui.image(imgui.ImTextureRef(texture), size, imgui.ImVec2(0, 1), imgui.ImVec2(1, 0))
    else:
        imgui.dummy(size)
        if imgui.is_item_hovered():
            imgui.set_tooltip(error or 'Generating preview...')
    if error:
        imgui.text_disabled('Preview unavailable (still selectable)')


def asset_card(imgui, previews, ref, color, label, selected, scale, *, disabled=False):
    picture(imgui, previews, ref, color, min(200*scale, imgui.get_content_region_avail().x))
    imgui.begin_disabled(disabled)
    changed, value = imgui.checkbox(label, selected)
    imgui.end_disabled()
    imgui.text_disabled(f'v{ref.version}')
    return changed, value
