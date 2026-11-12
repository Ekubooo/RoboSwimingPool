"""Prepare SDF support variants and volume maps, then publish complete artifacts."""

from dataclasses import asdict, replace
from importlib.metadata import version
import math
import os
from pathlib import Path

import numpy as np

from ...config import BakeOptions, VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from ...lib.storage import fingerprints, nvdb_io
from ...lib.storage.mesh_io import save_mesh_obj
from ...lib.storage.store import AssetStore
from ...lib.volume import sdf
from ...lib.volume.pbf_volume_map import precompute_volume_map
from .schema import (BoundaryArtifact, SCHEMA_VERSION, SDF_ALGORITHM_VERSION, CACHE_DIRECTORY,
                     _write_record, _read_record, _file, _device, _grid_layout,
                     check_volume_map_layout, _map_name)
from .load import _read_grids, _load_sdf, load_boundary_asset


def _bounds(mesh):
    # Match the runtime's existing float32 center, half-extents and radius.
    vertices = np.asarray(mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = np.asarray((low + high) * .5, dtype=np.float32)
    half = np.asarray(np.maximum(high - center, center - low), dtype=np.float32)
    radius = float(np.float32(np.linalg.norm(vertices - center, axis=1).max()))
    return dict(center=center.tolist(), half=half.tolist(), radius=radius)


def _build_sdf(mesh, checked, manifest, geometry, options, device, provenance, store):
    with store.staging(manifest.parent, prefix=".sdf-") as stage:
        volume = (sdf.bake if geometry["container"] else sdf.bake_rigid)(checked, options=options, device=device)
        nvdb_io.save_volume(volume, stage / "sdf.nvdb")
        volume = None
        save_mesh_obj(mesh, stage / "mesh.obj", digits=17)
        persisted, = _read_grids(stage / "sdf.nvdb", 1, device)
        record = dict(schema=SCHEMA_VERSION, kind="sdf", geometry=geometry, bake_options=asdict(options),
                      provenance=provenance, files=dict(nvdb="sdf.nvdb", mesh="mesh.obj"), bounds=_bounds(mesh),
                      sdf_sha256=fingerprints.sdf_digest(persisted), grid=_grid_layout(persisted),
                      nvdb_sha256=fingerprints.file_digest(stage / "sdf.nvdb"),
                      mesh_file_sha256=fingerprints.file_digest(stage / "mesh.obj"))
        _write_record(stage / "sdf.json", record)
        persisted = None
        record, persisted = _load_sdf(stage / "sdf.json", geometry, device)
        store.publish([(stage / name, manifest.parent / name) for name in ("sdf.nvdb", "mesh.obj", "sdf.json")])
        return record, persisted


def prepare_boundary_asset(mesh, checked, *, store: AssetStore, bake_options=BakeOptions(), support_radius,
                           quadrature_points=16, container=False, device="cuda:0", provenance=None):
    """Reuse or publish a boundary artifact, selecting the smallest adequate SDF.

    The SDF family excludes h_max; its individual variants include h_max. Volume
    maps depend on the persisted SDF contents and their own integration settings.
    Provenance describes the source, never runtime bodies, mass or placement.
    """
    check_volume_options(support_radius, quadrature_points)
    sdf.check_options(bake_options)
    selected = _device(device)
    options = replace(bake_options, max_support_radius=max(float(support_radius), bake_options.max_support_radius))
    base_options = asdict(options)
    del base_options["max_support_radius"]
    geometry = dict(mesh_sha256=fingerprints.mesh_digest(mesh), container=bool(container),
                    bake_options=base_options, sdf_algorithm_version=SDF_ALGORITHM_VERSION,
                    dependencies={p: version(p) for p in ("newton", "warp-lang", "trimesh")})
    directory = store.recipe_directory(CACHE_DIRECTORY, geometry)
    candidates = []
    for path in directory.glob("sdfs/*/sdf.json"):
        try:
            record = _read_record(path, "sdf")
            maximum = float(record["bake_options"]["max_support_radius"])
            if math.isfinite(maximum) and maximum >= options.max_support_radius:
                candidates.append((maximum, path))
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    record = distance = None
    for _, sdf_manifest in sorted(candidates):
        try:
            record, distance = _load_sdf(sdf_manifest, geometry, selected)
            break
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
            continue
    if distance is None:
        sdf_manifest = directory / "sdfs" / store.recipe_key(asdict(options)) / "sdf.json"
        print(f"Baking PBF boundary SDF: {sdf_manifest}", flush=True)
        record, distance = _build_sdf(mesh, checked, sdf_manifest, geometry, options, selected, provenance or {}, store)
    map_recipe = dict(sdf_sha256=record["sdf_sha256"], support_radius=float(support_radius),
                      quadrature_points=int(quadrature_points), algorithm_version=VOLUME_MAP_ALGORITHM_VERSION)
    manifest = directory / "maps" / store.recipe_key(map_recipe) / "boundary.json"
    artifact = BoundaryArtifact(manifest)
    if manifest.is_file():
        try:
            load_boundary_asset(artifact, support_radius=support_radius, quadrature_points=quadrature_points,
                                device=selected)
            print(f"Boundary cache hit: {manifest}", flush=True)
            return artifact
        except ValueError:
            pass
    print(f"Building PBF volume map: {manifest}", flush=True)
    volume_map = precompute_volume_map(distance, support_radius, quadrature_points=quadrature_points)
    check_volume_map_layout(distance, volume_map)
    with store.staging(manifest.parent, prefix=".map-") as stage:
        nvdb_io.save_volumes((distance, volume_map), stage / "boundary.nvdb",
                            grid_names=(distance.get_grid_info().name, _map_name(support_radius, quadrature_points)))
        # Temporary and final map directories are siblings, so relative visual references agree.
        mesh_path = _file(sdf_manifest, record["files"]["mesh"])
        boundary_record = dict(record, kind="boundary", volume_map=map_recipe,
                               files=dict(nvdb="boundary.nvdb", mesh=Path(os.path.relpath(mesh_path, manifest.parent)).as_posix()),
                               nvdb_sha256=fingerprints.file_digest(stage / "boundary.nvdb"))
        _write_record(stage / "boundary.json", boundary_record)
        load_boundary_asset(stage / "boundary.json", support_radius=support_radius,
                            quadrature_points=quadrature_points, device=selected)
        store.publish([(stage / name, manifest.parent / name) for name in ("boundary.nvdb", "boundary.json")])
    return artifact
