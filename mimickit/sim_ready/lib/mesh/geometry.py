"""Generic geometry operations with explicit transforms; no cache or scene layout."""

import manifold3d as manifold
import numpy as np
import trimesh

from ..storage.mesh_io import load_mesh_obj


def weld(mesh):
    """Join exact duplicate positions, without repairing faces or normals."""
    vertices, remap = np.unique(mesh.vertices, axis=0, return_inverse=True)
    return trimesh.Trimesh(vertices, remap[mesh.faces], process=False)


def to_solid(mesh):
    mesh = weld(mesh)
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
    return _mesh(manifold.Manifold.batch_boolean([to_solid(m) for m in meshes], manifold.OpType.Add))


def simplify(mesh, tolerance):
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("Simplification tolerance must be finite and positive")
    solid = to_solid(mesh)
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
    model = load_mesh_obj(obj_path)
    if not isinstance(model, trimesh.Trimesh) or model.is_empty or not np.isfinite(model.vertices).all():
        raise ValueError("Input OBJ must contain a finite, nonempty triangle mesh")
    # Keep separate parts unwelded until input validation and boolean union.
    if scale != 1.0:
        model.apply_scale(scale)
    if obj_up_axis == "Z":
        model.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    return model


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
