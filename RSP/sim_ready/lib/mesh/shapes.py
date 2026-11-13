"""Collision shapes to body-local solids, independent of Newton and storage."""
import numpy as np
import trimesh

from . import geometry, validation
from .convex import validate_convex_mesh
from ..asset_validation import ValidationReport, diagnostic_scope


def shape_mesh(shape):
    size = np.asarray(shape.scale)
    if shape.kind == "SPHERE":
        mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size[0]))
    elif shape.kind == "BOX":
        mesh = trimesh.creation.box(extents=2 * size)
    elif shape.kind == "CAPSULE":
        mesh = trimesh.creation.capsule(radius=float(size[0]), height=float(2 * size[1]))
    elif shape.kind == "CYLINDER":
        mesh = trimesh.creation.cylinder(radius=float(size[0]), height=float(2 * size[1]), sections=32)
    elif shape.kind in {"MESH", "CONVEX_MESH"}:
        # The source and the simulator's native collision shape remain untouched.
        mesh = trimesh.Trimesh(np.array(shape.vertices, copy=True), np.array(shape.faces, copy=True), process=False)
        # Trimesh reverses winding for a reflection; do not flip it a second time.
        mesh.apply_scale(size.copy())
        parts = validation.validate_rigid_parts(mesh)
        mesh = geometry.union(parts) if len(parts) > 1 else parts[0]
        if shape.kind == "CONVEX_MESH":
            validate_convex_mesh(mesh)
    else:
        raise ValueError(f"Unsupported collision shape: {shape.kind}")
    pose = shape.transform
    matrix = trimesh.transformations.quaternion_matrix([pose[6], pose[3], pose[4], pose[5]])
    matrix[:3, 3] = pose[:3]
    mesh.apply_transform(matrix)
    return mesh


def body_mesh(body, report=None):
    report = ValidationReport() if report is None else report
    if not body.shapes:
        report.add("error", "geometry.missing", body.name, "No native collision shapes; explicitly exclude auxiliary links")
        report.require_valid()
    parts = []
    for shape in body.shapes:
        with diagnostic_scope(report, "geometry.shape", f"{body.name}/{shape.name}"):
            parts.append(shape_mesh(shape))
    with diagnostic_scope(report, "geometry.body", body.name):
        mesh = geometry.union(parts) if len(parts) > 1 else parts[0]
        checked = validation.validate_solid(mesh)
    return mesh, checked
