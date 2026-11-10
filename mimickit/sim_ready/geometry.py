"""Prepare container scenes and independent rigid geometry in memory.

The container's two boolean branches are independent; independent rigid assets
use explicit transforms without placement or simplification. No function in
this module writes files.
"""

from dataclasses import dataclass

import manifold3d as manifold
import numpy as np
import trimesh

from .config import GeometryOptions
from newton import Mesh


@dataclass
class Scene:
    terrain: trimesh.Trimesh
    model: trimesh.Trimesh
    scale: float


def _weld(mesh):
    """Join exact duplicate positions, without repairing faces or normals."""
    vertices, remap = np.unique(mesh.vertices, axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices, remap[mesh.faces], process=False)


def _solid(mesh):
    mesh = _weld(mesh)
    if not np.isfinite(mesh.vertices).all() or not mesh.is_volume:
        raise ValueError("Boolean inputs must be finite, closed, consistently oriented solids")
    result = manifold.Manifold(manifold.Mesh64(
        vert_properties=np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        tri_verts=np.ascontiguousarray(mesh.faces, dtype=np.uint64),
    ))
    if result.status() != manifold.Error.NoError:
        raise ValueError(f"Manifold rejected input: {result.status()}")
    return result


def _mesh(solid):
    if solid.status() != manifold.Error.NoError or solid.is_empty():
        raise ValueError(f"Manifold operation failed: {solid.status()}")
    data = solid.to_mesh64()
    # Own the arrays instead of retaining views backed by a native Mesh64 owner.
    result = trimesh.Trimesh(data.vert_properties[:, :3].copy(), data.tri_verts.copy(), process=False)
    if not np.isfinite(result.vertices).all() or not result.is_volume:
        raise ValueError("Boolean/simplification result is not a finite closed solid")
    return result


def union(meshes):
    if not meshes:
        raise ValueError("A boolean union needs at least one solid")
    return _mesh(manifold.Manifold.batch_boolean([_solid(m) for m in meshes], manifold.OpType.Add))


def simplify(mesh, tolerance):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Simplification tolerance must be finite and positive")
    solid = _solid(mesh)
    if tolerance < solid.get_tolerance():
        raise ValueError("Requested simplification tolerance is below the backend minimum")
    return _mesh(solid.simplify(tolerance))


def prepare_rigid(obj_path, *, scale=1.0, obj_up_axis=None):
    """Read independent geometry, applying only explicitly requested transforms.

    Input units are meters unless changed by ``scale``. The default preserves
    the model origin and axes. Explicit Z-up conversion rotates around that
    origin into Y-up, matching the container input convention.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Rigid model scale must be finite and positive")
    if obj_up_axis not in (None, "Y", "Z"):
        raise ValueError("OBJ up axis must be Y, Z or omitted to preserve its axes")
    model = trimesh.load(obj_path, force="mesh", process=False)
    if not isinstance(model, trimesh.Trimesh) or model.is_empty or not np.isfinite(model.vertices).all():
        raise ValueError("Input OBJ must contain a finite, nonempty triangle mesh")
    # Keep separate parts unwelded until input validation and boolean union.
    if scale != 1.0:
        model.apply_scale(scale)
    if obj_up_axis == "Z":
        model.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return model


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
    return union(terrain_boxes(options))


def platform_bounds(terrain):
    top = terrain.bounds[1, 1]
    selected = (np.all(np.isclose(terrain.triangles[:, :, 1], top, atol=1e-7, rtol=0), axis=1)
                & (terrain.face_normals[:, 1] > 1 - 1e-7) & (terrain.area_faces > 0))
    if not selected.any():
        raise ValueError("Terrain has no upward-facing platform at its maximum height")
    points = terrain.triangles[selected].reshape(-1, 3)
    return np.array([points.min(axis=0), points.max(axis=0)])


def support_center(mesh, depth=0.0):
    """Area centroid of the base or a horizontal support section, in Y-up."""
    bottom = mesh.bounds[0, 1]
    flat = (np.all(np.isclose(mesh.triangles[:, :, 1], bottom, atol=1e-9, rtol=0), axis=1)
            & (mesh.area_faces > 0))
    if flat.any():
        center = np.average(mesh.triangles_center[flat], weights=mesh.area_faces[flat], axis=0)
        return np.array([center[0], bottom, center[2]])
    if depth > 0:
        segments, face_ids = trimesh.intersections.mesh_plane(
            mesh, [0, 1, 0], [0, bottom + depth, 0], return_faces=True)
        if len(segments):
            tangent = np.cross([0, 1, 0], mesh.face_normals[face_ids])
            reverse = np.einsum("ij,ij->i", segments[:, 1] - segments[:, 0], tangent) < 0
            segments[reverse] = segments[reverse, ::-1]
            # (Z,X) is right-handed around +Y. Signed areas preserve holes.
            coordinates = segments[:, :, [2, 0]]
            origin = coordinates.mean(axis=(0, 1))
            a, b = coordinates[:, 0] - origin, coordinates[:, 1] - origin
            cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
            area = cross.sum()
            epsilon = np.finfo(float).eps * max(mesh.extents[[0, 2]].max() ** 2, 1e-20) * 32
            if area > epsilon:
                z, x = origin + ((a + b) * cross[:, None]).sum(axis=0) / (3 * area)
                return np.array([x, bottom, z])
    points = mesh.vertices[np.isclose(mesh.vertices[:, 1], bottom, atol=1e-9, rtol=0)]
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    return np.array([center[0], bottom, center[2]])


def prepare_scene(obj_path, *, options=GeometryOptions(), obj_up_axis="Y"):
    terrain = terrain_mesh(options)
    model = trimesh.load(obj_path, force="mesh", process=False)
    if not isinstance(model, trimesh.Trimesh) or model.is_empty or not np.isfinite(model.vertices).all():
        raise ValueError("Input OBJ must contain a finite, nonempty triangle mesh")
    if obj_up_axis not in ("Y", "Z"):
        raise ValueError("OBJ up axis must be Y or Z")
    model = _weld(model)
    _solid(model)  # Reject open inputs before attempting placement or boolean operations.
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
    model.apply_translation(target - support_center(model, options.model_sink))
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
    return simplify(union([scene.terrain, scene.model]), options.simplify_tolerance)


def container(scene, options):
    parts, bounds = container_parts(scene.terrain, options)
    solid = union([scene.terrain, scene.model, *parts])
    return simplify(solid, options.simplify_tolerance), bounds
