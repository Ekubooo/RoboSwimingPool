"""Validate in-memory collision geometry before uploading float32 data to CUDA.

Shared checks establish closed manifold topology and orientation. Container
checks additionally require a cuboid exterior/single cavity. These checks do
not constitute a general self-intersection proof.
No repair or per-shell normal correction is performed.
"""

from dataclasses import dataclass
import warnings

import numpy as np
import trimesh


@dataclass(frozen=True)
class ValidatedMesh:
    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray
    cavity_bounds: np.ndarray


@dataclass(frozen=True)
class ValidatedSolidMesh:
    """GPU-ready independent solid; its coordinates retain the asset origin."""

    vertices: np.ndarray
    faces: np.ndarray
    bounds: np.ndarray


def _triangles(vertices, faces, label, *, weld=True):
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices):
        raise ValueError(f"{label}: expected nonempty 3D vertices")
    if not np.isfinite(vertices).all():
        raise ValueError(f"{label}: nonfinite coordinates")
    if faces.ndim != 2 or faces.shape[1] != 3 or not len(faces):
        raise ValueError(f"{label}: expected nonempty triangle indices")
    if not np.issubdtype(faces.dtype, np.integer) or faces.min() < 0 or faces.max() >= len(vertices):
        raise ValueError(f"{label}: invalid vertex indices")
    if weld:
        vertices, mapping = np.unique(vertices, axis=0, return_inverse=True)
        faces = mapping[faces]
    canonical = np.sort(faces, axis=1)
    if (np.any(np.diff(canonical, axis=1) == 0)
            or len(np.unique(canonical, axis=0)) != len(faces)):
        raise ValueError(f"{label}: collapsed or duplicate triangles")
    edges = vertices[faces].astype(np.float64)
    area = np.cross(edges[:, 1] - edges[:, 0], edges[:, 2] - edges[:, 0])
    if not np.isfinite(area).all() or np.any(np.all(area == 0, axis=1)):
        raise ValueError(f"{label}: zero-area triangles")
    used, mapping = np.unique(faces, return_inverse=True)
    return trimesh.Trimesh(vertices[used], mapping.reshape(-1, 3), process=False)


def _closed_shells(mesh, label):
    """Check topology before splitting, without repairing or reorienting faces."""
    edges = mesh.edges_sorted
    if np.any(np.unique(edges, axis=0, return_counts=True)[1] != 2):
        raise ValueError(f"{label}: open or nonmanifold edges")
    if not mesh.is_winding_consistent:
        raise ValueError(f"{label}: inconsistent triangle winding")
    # In a closed 2-manifold the triangles incident on each vertex form one
    # connected fan. Edge incidence alone would accept two solids touching at a point.
    neighbors = [[] for _ in mesh.vertices]
    for face in mesh.faces:
        for corner in range(3):
            neighbors[face[corner]].append((face[(corner + 1) % 3], face[(corner + 2) % 3]))
    for links in neighbors:
        graph = {}
        for a, b in links:
            graph.setdefault(a, set()).add(b)
            graph.setdefault(b, set()).add(a)
        if not graph or any(len(adjacent) != 2 for adjacent in graph.values()):
            raise ValueError(f"{label}: nonmanifold vertex")
        reached, pending = set(), [next(iter(graph))]
        while pending:
            vertex = pending.pop()
            if vertex not in reached:
                reached.add(vertex)
                pending.extend(graph[vertex] - reached)
        if len(reached) != len(graph):
            raise ValueError(f"{label}: disconnected vertex fans")
    return mesh.split(only_watertight=False, repair=False)


def _closed_surfaces(mesh, label):
    shells = _closed_shells(mesh, label)
    outer = [s for s in shells if s.volume > 0]
    inner = [s for s in shells if s.volume < 0]
    if (len(shells) != 2 or len(outer) != 1 or len(inner) != 1
            or not np.isfinite(mesh.volume) or mesh.volume <= 0):
        raise ValueError(f"{label}: expected one positive exterior and one negative fluid cavity")
    return outer[0], inner[0]


def _positive_shells(mesh, label):
    shells = _closed_shells(mesh, label)
    if not len(shells) or any(not np.isfinite(s.volume) or s.volume <= 0 for s in shells):
        raise ValueError(f"{label}: expected positive, outward-oriented closed solids; cavities are unsupported")
    return shells


def validate_rigid_parts(mesh):
    """Validate each source solid before union can remove internal surfaces.

    Preserve index-connected parts first: touching solids can have coincident
    vertices/faces that must not be welded together before their boolean union.
    If OBJ seams split a surface into open pieces, join exact duplicate vertices
    and validate the resulting shells. No holes or normals are repaired.
    """
    checked = _triangles(np.asarray(mesh.vertices), np.asarray(mesh.faces), "rigid input", weld=False)
    parts = checked.split(only_watertight=False, repair=False)
    if any(not part.is_watertight for part in parts):
        checked = _triangles(np.asarray(mesh.vertices), np.asarray(mesh.faces), "rigid input")
        parts = _closed_shells(checked, "rigid input")
    solids = []
    for index, part in enumerate(parts):
        label = f"rigid input part {index}"
        part = _triangles(np.asarray(part.vertices), np.asarray(part.faces), label)
        solids.extend(_positive_shells(part, label))
    if not solids:
        raise ValueError("Rigid input must contain at least one closed solid")
    return solids


def _warn_thin_triangles(mesh):
    minimum = np.rad2deg(mesh.face_angles.min())
    if minimum < 1:
        warnings.warn(f"Thin triangles: minimum angle {minimum:.6g} degrees; baking remains enabled",
                      RuntimeWarning, stacklevel=3)


def validate_solid(mesh):
    """Check a union of positive closed solids at source and GPU precision."""
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("Too many vertices for int32 GPU indices")
    for label, dtype in (("source solid", np.float64), ("float32 GPU solid", np.float32)):
        with np.errstate(over="ignore", invalid="ignore"):
            checked = _triangles(vertices.astype(dtype), faces, label)
        _positive_shells(checked, label)
    _warn_thin_triangles(checked)
    return ValidatedSolidMesh(
        np.ascontiguousarray(checked.vertices, dtype=np.float32),
        np.ascontiguousarray(checked.faces, dtype=np.int32),
        checked.bounds.copy(),
    )


def _outline(outer, inner, bounds, tolerance, label):
    bounds = np.asarray(bounds)
    if not np.allclose(outer.bounds, bounds, atol=tolerance, rtol=0):
        raise ValueError(f"{label}: exterior bounds do not match the container outline")
    # Each exterior triangle must lie wholly on one of the six cuboid planes.
    on_plane = np.isclose(outer.triangles[:, :, :, None], bounds.T[None, None, :, :],
                          atol=tolerance, rtol=0).all(axis=1).any(axis=(1, 2))
    if not on_plane.all():
        raise ValueError(f"{label}: exterior has protruding or recessed faces")
    if np.any(inner.bounds[0] <= bounds[0]) or np.any(inner.bounds[1] >= bounds[1]):
        raise ValueError(f"{label}: cavity must lie strictly inside the exterior")


def validate_container(mesh, outer_bounds, *, tolerance=0.001):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if len(vertices) > np.iinfo(np.int32).max:
        raise ValueError("Too many vertices for int32 GPU indices")
    for label, dtype in (("source mesh", np.float64), ("float32 GPU mesh", np.float32)):
        with np.errstate(over="ignore", invalid="ignore"):
            checked = _triangles(vertices.astype(dtype), faces, label)
        outer, inner = _closed_surfaces(checked, label)
        _outline(outer, inner, outer_bounds, max(1e-6, tolerance), label)
    _warn_thin_triangles(checked)
    return ValidatedMesh(
        np.ascontiguousarray(checked.vertices, dtype=np.float32),
        np.ascontiguousarray(checked.faces, dtype=np.int32),
        checked.bounds.copy(), inner.bounds.copy(),
    )
