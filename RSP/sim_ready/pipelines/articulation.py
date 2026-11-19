"""Validate an entire articulated asset before preparing any of its boundaries."""
from dataclasses import dataclass

import trimesh

from ..lib.asset_validation import (AssetValidationError, ValidationReport, diagnostic_scope,
                                  validate_physical_asset, validate_coverage)
from ..lib.mesh.shapes import body_mesh
from ..lib.storage.fingerprints import mesh_digest
from ..lib.storage.store import AssetStore
from ..boundary.schema import BoundaryArtifact
from .boundary import prepare_boundary_asset


@dataclass
class PreparedArticulation:
    boundaries: dict[int, BoundaryArtifact]
    meshes: dict[int, trimesh.Trimesh]
    report: ValidationReport


def prepare_articulated_asset(asset, *, store: AssetStore, bake_options, support_radius,
                              quadrature_points=16, device="cuda:0", excluded_links=(), previous=None):
    report = validate_physical_asset(asset, excluded_links)
    selected = [i for i, body in enumerate(asset.bodies) if body.name not in excluded_links]
    report.extend(validate_coverage(asset, selected, excluded_links))
    report.require_valid()
    meshes, checked = {}, {}
    first_error = None
    for index in selected:
        try:
            meshes[index], checked[index] = body_mesh(asset.bodies[index], report)
        except AssetValidationError as exc:
            # Finish checking the other links, but never bake a partial invalid asset.
            first_error = first_error or exc
    if first_error is not None:
        raise AssetValidationError(report) from first_error
    report.require_valid()
    boundaries = {}
    by_geometry = {}
    for index in selected:
        geometry = mesh_digest(meshes[index])
        if geometry in by_geometry:
            boundaries[index] = by_geometry[geometry]
            continue
        name = asset.bodies[index].name
        old_relative = None if previous is None else previous.record["links"].get(name)
        old = None if old_relative is None else BoundaryArtifact(previous.file(old_relative))
        with diagnostic_scope(report, "boundary.prepare", f"{asset.name}/{asset.bodies[index].name}"):
            boundaries[index] = prepare_boundary_asset(meshes[index], checked[index], store=store,
                bake_options=bake_options, support_radius=support_radius,
                quadrature_points=quadrature_points, device=device,
                provenance={"source_kind": "newton_collision_geometry"},
                destination=f"links/{index}", previous=old)
        by_geometry[geometry] = boundaries[index]
    report.extend(validate_coverage(asset, boundaries, excluded_links))
    report.require_valid()
    return PreparedArticulation(boundaries, meshes, report)
