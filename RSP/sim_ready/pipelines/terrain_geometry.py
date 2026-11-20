"""Generate closed terrain solids in the fixed pool's Z-up coordinates."""

import numpy as np
import trimesh

from ..reference import POOL_SIZE


PLATFORM_WIDTH = .7
_BASE_Z = -.1
_STEP_WIDTH = .325
_CELL_SIZE = .5


def _box(x0, x1, y0, y1, top):
    return trimesh.creation.box(bounds=((x0, y0, _BASE_Z), (x1, y1, top)))


def _platform(width, depth, top):
    half = PLATFORM_WIDTH / 2
    return _box(width / 2 - half, width / 2 + half, depth / 2 - half, depth / 2 + half, top)


def _stairs(width, depth, height):
    w = _STEP_WIDTH
    platform = _platform(width, depth, 3 * height)
    # Nested rectangles fill the otherwise low strips flanking the square platform.
    return (_box(0., width, 0., depth, height),
            _box(w, width - w, w, depth - w, 2 * height), platform), platform


def _random_grid(width, depth, max_height, seed):
    from newton import Mesh

    size = _CELL_SIZE
    nx, ny = int(width / size), int(depth / size)
    if (nx * size, ny * size) != (width, depth):
        raise ValueError("Terrain cell size must divide the pool floor exactly")
    generated = Mesh.create_terrain(
        grid_size=(1, 1), block_size=(width, depth), terrain_types="random_grid",
        terrain_params={"random_grid": dict(grid_width=size, grid_height_range=(0., max_height), seed=seed)},
        compute_inertia=False,
    )
    vertices = np.asarray(generated.vertices)
    if vertices.shape != (nx * ny * 24, 3):
        raise ValueError("Unexpected Newton random_grid vertex layout")
    boxes = []
    for index, cell in enumerate(vertices.reshape(-1, 24, 3)):
        i, j = divmod(index, ny)
        x0, y0 = i * size, j * size
        # Newton duplicates vertices per face. Only indices 4–7 receive the
        # randomized height, leaving disconnected side faces in the output.
        top = cell[4:8]
        if (not np.array_equal(top[:, :2].min(axis=0), (x0, y0))
                or not np.array_equal(top[:, :2].max(axis=0), (x0 + size, y0 + size))
                or not np.all(top[:, 2] == top[0, 2])
                or not 0. <= top[0, 2] <= np.float32(max_height)):
            raise ValueError("Unexpected Newton random_grid cell geometry")
        boxes.append(_box(x0, x0 + size, y0, y0 + size, float(top[0, 2])))
    return tuple(boxes), boxes[0]


def _gap(width, depth, gap_width, height):
    # Match Newton's border and central island layout with the pool floor
    # as the trench bottom. All five raised boxes overlap the existing slab.
    span = PLATFORM_WIDTH + 2 * gap_width
    x, y = (width - span) / 2, (depth - span) / 2
    if min(x, y) <= 0.:
        raise ValueError("Gap and platform must leave a positive border")
    platform = _platform(width, depth, height)
    boxes = (_box(0., width, 0., y, height),
             _box(0., width, depth - y, depth, height),
             _box(0., x, y, depth - y, height),
             _box(width - x, width, y, depth - y, height), platform)
    return boxes, platform


def terrain_boxes(recipe):
    """Return the visible base, raised solids, and one fixed placement surface."""
    width, depth = POOL_SIZE[:2]
    params = recipe.parameters
    if recipe.kind == "pyramid_stairs":
        boxes, platform = _stairs(width, depth, params["step_height"])
    elif recipe.kind == "random_grid":
        boxes, platform = _random_grid(width, depth, params["max_height"], params["seed"])
    else:
        boxes, platform = _gap(width, depth, params["gap_width"], params["depth"])
    return _box(0., width, 0., depth, 0.), boxes, platform
