"""Extract solid geometry inputs from the installed Newton terrain generator."""

import numpy as np
import trimesh


def legacy_stair_boxes(size, center, *, step_width, step_height, platform_width):
    """Convert the generated stair boxes once for both rigid and fluid geometry."""
    from newton import Mesh

    terrain = Mesh.create_terrain(
        grid_size=(1, 1), block_size=size, terrain_types="pyramid_stairs",
        terrain_params={"pyramid_stairs": dict(step_width=step_width,
            step_height=step_height, platform_width=platform_width)},
        compute_inertia=False,
    )
    points = np.asarray(terrain.vertices, dtype=np.float64).copy()
    # Align generator coordinates that differ by a float32 ULP at shared faces.
    for axis in range(3):
        coordinates, inverse = np.unique(points[:, axis], return_inverse=True)
        epsilon = float(np.spacing(np.float32(max(1., np.abs(coordinates).max()))))
        anchors = coordinates.copy()
        start = 0
        for end in range(1, len(coordinates) + 1):
            if end == len(coordinates) or coordinates[end] - coordinates[start] > epsilon:
                anchors[start:end] = coordinates[start]
                start = end
        points[:, axis] = anchors[inverse]
    points[:, :2] += np.asarray(center) - np.asarray(size) / 2
    faces = np.asarray(terrain.indices).reshape(-1, 3)
    if not len(faces) or len(faces) % 12:
        raise ValueError("Expected twelve triangles per Newton stair box")
    boxes = []
    for block in faces.reshape(-1, 12, 3):
        ids, local = np.unique(block, return_inverse=True)
        boxes.append(trimesh.Trimesh(points[ids], local.reshape(-1, 3), process=False))
    return boxes


def random_grid_cells(width, depth, size, max_height, seed):
    """Extract cell coordinates and heights from Newton's face-wise vertices."""
    from newton import Mesh

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
        yield x0, x0 + size, y0, y0 + size, float(top[0, 2])
