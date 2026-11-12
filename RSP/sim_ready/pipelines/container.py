"""Terrain/container generation and placement recipes, followed by asset preparation."""

from dataclasses import dataclass

import numpy as np
import trimesh
from newton import Mesh

from ..config import GeometryOptions, BakeOptions
from ..lib.mesh import geometry, validation
from ..lib.volume import sdf
from ..lib.storage.mesh_io import load_mesh_obj, save_mesh_obj
from ..lib.storage.nvdb_io import save_volume
from ..lib.storage.store import AssetStore
from . import collision


@dataclass
class Scene:
    terrain: trimesh.Trimesh
    model: trimesh.Trimesh
    scale: float


def _check_options(options):
    positive = (*options.size, options.step_width, options.step_height,
                options.platform_width, options.wall_thickness, options.ceiling,
                options.simplify_tolerance)
    if not all(np.isfinite(v) and v > 0 for v in positive):
        raise ValueError("Geometry dimensions and tolerances must be finite and positive")
    if not 0 < options.model_fraction <= 1:
        raise ValueError("Model fraction must lie in (0, 1]")
    if not np.isfinite(options.model_sink) or options.model_sink < 0:
        raise ValueError("Model sink must be finite and nonnegative")
    if not np.isfinite(options.model_yaw):
        raise ValueError("Model yaw must be finite")
    if not 0 < options.overlap < options.wall_thickness:
        raise ValueError("Overlap must lie between zero and wall thickness")
    if min(options.size) <= max(options.platform_width, 2 * options.overlap):
        raise ValueError("Terrain must accommodate both the platform and the cavity")


def terrain_boxes(options):
    """Return generator boxes with shared float32 boundaries aligned once.

    Both the solid terrain and native rigid collisions use these coordinates.
    """
    _check_options(options)
    terrain = Mesh.create_terrain(
        grid_size=(1, 1), block_size=options.size, terrain_types="pyramid_stairs",
        compute_inertia=False,
        terrain_params={"pyramid_stairs": dict(step_width=options.step_width,
            step_height=options.step_height, platform_width=options.platform_width)},
    )
    source, indices = terrain.vertices, terrain.indices
    width = float(source[:, 1].max())
    points = np.column_stack((source[:, 0], source[:, 2], width - source[:, 1])).astype(np.float64)
    # Adjacent boxes can differ by one float32 ULP. Align only the generated
    # terrain, before union; applying this to an imported model would alter it.
    for axis in range(3):
        coordinates, inverse = np.unique(points[:, axis], return_inverse=True)
        epsilon = float(np.spacing(np.float32(max(1, np.abs(coordinates).max()))))
        anchors = coordinates.copy()
        start = 0
        for end in range(1, len(coordinates) + 1):
            if end == len(coordinates) or coordinates[end] - coordinates[start] > epsilon:
                anchors[start:end] = coordinates[start]
                start = end
        points[:, axis] = anchors[inverse]
    faces = indices.reshape(-1, 3)
    if not len(faces) or len(faces) % 12:
        raise ValueError("Expected twelve triangles per generated stair box")
    boxes = []
    for block in faces.reshape(-1, 12, 3):
        ids, local = np.unique(block, return_inverse=True)
        boxes.append(trimesh.Trimesh(points[ids], local.reshape(-1, 3), process=False))
    return boxes


def terrain_mesh(options):
    """Union the same aligned boxes used by the rigid collision scene."""
    return geometry.union(terrain_boxes(options))


def platform_bounds(terrain):
    top = terrain.bounds[1, 1]
    selected = (np.all(np.isclose(terrain.triangles[:, :, 1], top, atol=1e-7, rtol=0), axis=1)
                & (terrain.face_normals[:, 1] > 1 - 1e-7) & (terrain.area_faces > 0))
    if not selected.any():
        raise ValueError("Terrain has no upward-facing platform at its maximum height")
    points = terrain.triangles[selected].reshape(-1, 3)
    return np.array([points.min(axis=0), points.max(axis=0)])


def prepare_scene(obj_path, *, options=GeometryOptions(), obj_up_axis="Y"):
    terrain = terrain_mesh(options)
    model = load_mesh_obj(obj_path)
    if not isinstance(model, trimesh.Trimesh) or model.is_empty or not np.isfinite(model.vertices).all():
        raise ValueError("Input OBJ must contain a finite, nonempty triangle mesh")
    if obj_up_axis not in ("Y", "Z"):
        raise ValueError("OBJ up axis must be Y or Z")
    model = geometry.weld(model)
    geometry.to_solid(model)  # Reject open inputs before attempting placement or boolean operations.
    if obj_up_axis == "Z":
        model.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    model.apply_transform(trimesh.transformations.rotation_matrix(np.deg2rad(options.model_yaw), [0, 1, 0]))
    platform = platform_bounds(terrain)
    span = model.extents[[0, 2]].max()
    if span <= 0:
        raise ValueError("Model must have a nonzero horizontal span")
    scale = options.model_fraction * np.min((platform[1] - platform[0])[[0, 2]]) / span
    model.apply_scale(scale)
    target = platform.mean(axis=0)
    target[1] -= options.model_sink
    model.apply_translation(target - geometry.support_center(model, options.model_sink))
    if model.bounds[1, 1] >= options.ceiling:
        raise ValueError("Placed model reaches or exceeds the container ceiling")
    return Scene(terrain, model, float(scale))


def container_parts(terrain, options):
    """Four overlapping walls, lid and outward base extension; one outer outline."""
    _check_options(options)
    low, high = terrain.bounds
    if options.ceiling <= high[1]:
        raise ValueError("Container ceiling must be above the terrain")
    w, delta = options.wall_thickness, options.overlap
    outer_low = low[[0, 2]] - (w - delta)
    outer_high = high[[0, 2]] + (w - delta)
    length, width = outer_high - outer_low
    cx, cz = (outer_low + outer_high) / 2
    wall_top = options.ceiling + delta
    height, cy = wall_top - low[1], (wall_top + low[1]) / 2

    def box(size, center):
        result = trimesh.creation.box(extents=size)
        result.apply_translation(center)
        return result

    short = min(high[2] - low[2] + delta, width)
    parts = [
        box((length, height, w), (cx, cy, outer_low[1] + w / 2)),
        box((length, height, w), (cx, cy, outer_high[1] - w / 2)),
        box((w, height, short), (outer_low[0] + w / 2, cy, cz)),
        box((w, height, short), (outer_high[0] - w / 2, cy, cz)),
        box((length, w, width), (cx, options.ceiling + w / 2, cz)),
    ]
    growth = max(w - 0.15, 0.0)
    parts.append(box((length, growth + delta, width), (cx, low[1] + (delta - growth) / 2, cz)))
    bounds = np.array([[outer_low[0], low[1] - growth, outer_low[1]],
                       [outer_high[0], options.ceiling + w, outer_high[1]]])
    return parts, bounds


def landscape(scene, options):
    return geometry.simplify(geometry.union([scene.terrain, scene.model]), options.simplify_tolerance)


def container(scene, options):
    parts, bounds = container_parts(scene.terrain, options)
    solid = geometry.union([scene.terrain, scene.model, *parts])
    return geometry.simplify(solid, options.simplify_tolerance), bounds


def prepare(source, *, options=GeometryOptions()):
    """Return the placed scene, visible mesh and aligned native collision boxes."""
    scene = prepare_scene(source, options=options)
    walls, _ = container_parts(scene.terrain, options)
    boxes = terrain_boxes(options) + walls
    static_boxes = [(m.bounds.mean(axis=0).tolist(), (m.extents / 2.).tolist()) for m in boxes]
    visible = landscape(scene, options)
    return scene, visible, static_boxes


def build(scene, visible, stem, *, geometry_options=GeometryOptions(),
          bake_options=BakeOptions(), store: AssetStore, collision_name="model", device="cuda:0"):
    """Validate the cavity, prepare model collision and stage an OBJ/SDF pair.

    The caller selects the store and owns scene metadata and final publication.
    Boundary artifacts add volume maps through their separate preparation API.
    """
    stem = store.path(stem)
    relative = stem.relative_to(store.root)
    mesh_path = store.path(relative.with_suffix(".obj"))
    sdf_path = store.path(relative.with_suffix(".nvdb"))
    solid, bounds = container(scene, geometry_options)
    checked = validation.validate_container(solid, bounds, tolerance=geometry_options.simplify_tolerance)
    hulls = collision.prepare_hulls(scene.model, collision_name, store)
    volume = sdf.bake(checked, options=bake_options, device=device)
    save_mesh_obj(visible, mesh_path, digits=8)
    save_volume(volume, sdf_path)
    return mesh_path, sdf_path, hulls
