"""Prepare a boundary SDF and add a volume map when requested."""

from dataclasses import asdict, replace
from importlib.metadata import version
import math

import numpy as np

from ..boundary.load import _read_grids, _read_sdf, load_boundary_asset, load_prepared_sdf
from ..boundary.schema import (
    BoundaryArtifact, SCHEMA_VERSION, SDF_ALGORITHM_VERSION, _device,
    _grid_layout, _map_name, _read_record, _write_record, check_volume_map_layout,
)
from ..config import BakeOptions, VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from ..lib.storage import fingerprints, nvdb_io
from ..lib.storage.mesh_io import save_mesh_obj
from ..lib.storage.store import AssetStore
from ..lib.volume import sdf
from ..lib.volume.pbf_volume_map import precompute_volume_map


def _bounds(mesh):
    # Match the runtime's float32 center, half-extents and radius.
    vertices = np.asarray(mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = np.asarray((low + high) * .5, dtype=np.float32)
    half = np.asarray(np.maximum(high - center, center - low), dtype=np.float32)
    radius = float(np.float32(np.linalg.norm(vertices - center, axis=1).max()))
    return dict(center=center.tolist(), half=half.tolist(), radius=radius)


def _reuse_current_sdf(artifact, geometry, options, device):
    try:
        manifest = artifact.manifest_path
        record = _read_record(manifest, "boundary")
        maximum = float(record["bake_options"]["max_support_radius"])
        if (record["geometry"] != geometry or not math.isfinite(maximum)
                or maximum < options.max_support_radius):
            return None
        _, grids = _read_sdf(manifest, record, options.max_support_radius, device)
        return grids[0], record["bake_options"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError):
        return None


def prepare_boundary_asset(mesh, checked, *, store: AssetStore, bake_options=BakeOptions(), support_radius,
                           quadrature_points=16, container=False, device="cuda:0", provenance=None,
                           destination="boundary", previous=None, sdf_only=False):
    """Write the current SDF, with a matching volume map for playback by default."""
    check_volume_options(support_radius, quadrature_points)
    sdf.check_options(bake_options)
    selected = _device(device)
    options = replace(bake_options, max_support_radius=max(float(support_radius),
                                                          bake_options.max_support_radius))
    base_options = asdict(options)
    del base_options["max_support_radius"]
    geometry = dict(mesh_sha256=fingerprints.mesh_digest(mesh), container=bool(container),
                    bake_options=base_options, sdf_algorithm_version=SDF_ALGORITHM_VERSION,
                    dependencies={p: version(p) for p in ("newton", "warp-lang", "trimesh")})
    directory = store.path(destination)
    artifact = BoundaryArtifact(directory / "boundary.json")
    candidate = previous if previous is not None else artifact
    reuse = _reuse_current_sdf(candidate, geometry, options, selected)
    with store.staging(directory, prefix=".boundary-") as stage:
        volume_path = stage / "boundary.nvdb"
        if reuse is None:
            print(f"Baking PBF boundary SDF: {artifact}", flush=True)
            distance = (sdf.bake if container else sdf.bake_rigid)(checked, options=options, device=device)
            # Persist and reload before hashing, as for the original SDF cache.
            scratch = volume_path if sdf_only else stage / "sdf.nvdb"
            nvdb_io.save_volume(distance, scratch)
            distance = None
            distance, = _read_grids(scratch, 1, selected)
            if not sdf_only:
                scratch.unlink()
            persisted_options = asdict(options)
        else:
            distance, persisted_options = reuse
        sdf_hash = fingerprints.sdf_digest(distance)
        map_recipe = None
        if sdf_only:
            if reuse is not None:
                nvdb_io.save_volumes((distance,), volume_path)
        else:
            map_recipe = dict(sdf_sha256=sdf_hash, support_radius=float(support_radius),
                              quadrature_points=int(quadrature_points), algorithm_version=VOLUME_MAP_ALGORITHM_VERSION)
            print(f"Building PBF volume map: {artifact}", flush=True)
            volume_map = precompute_volume_map(distance, support_radius, quadrature_points=quadrature_points)
            check_volume_map_layout(distance, volume_map)
            nvdb_io.save_volumes((distance, volume_map), volume_path,
                                grid_names=(distance.get_grid_info().name, _map_name(support_radius, quadrature_points)))
        save_mesh_obj(mesh, stage / "mesh.obj", digits=17)
        record = dict(schema=SCHEMA_VERSION, kind="boundary", geometry=geometry,
                      state="sdf_ready" if sdf_only else "ready",
                      bake_options=persisted_options, provenance={} if provenance is None else provenance,
                      files=dict(nvdb="boundary.nvdb", mesh="mesh.obj"), bounds=_bounds(mesh),
                      sdf_sha256=sdf_hash, grid=_grid_layout(distance), volume_map=map_recipe,
                      nvdb_sha256=fingerprints.file_digest(volume_path),
                      mesh_file_sha256=fingerprints.file_digest(stage / "mesh.obj"))
        manifest = stage / "boundary.json"
        _write_record(manifest, record)
        if sdf_only:
            load_prepared_sdf(manifest, support_radius=options.max_support_radius, device=selected)
        else:
            load_boundary_asset(manifest, support_radius=support_radius,
                                quadrature_points=quadrature_points, device=selected)
        store.publish([(stage / name, directory / name)
                       for name in ("mesh.obj", "boundary.nvdb", "boundary.json")])
    return artifact
