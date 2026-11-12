"""Strictly read-only boundary loading; imports no baking or decomposition code."""

import warp as wp

from ...config import VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from ...lib.storage import fingerprints, nvdb_io
from .schema import (BoundaryArtifact, LoadedBoundaryAsset, _read_record, _file, _device,
                     _grid_layout, check_volume_map_layout, _check_support, _map_name, _validate_bounds)


def _read_grids(path, count, device):
    if nvdb_io.grid_count(path) != count:
        raise ValueError(f"Expected {count} NanoVDB grids")
    with path.open("rb") as stream:
        current = wp.Volume.load_from_nvdb(stream, device=device)
    grids = []
    for index in range(count):
        if current is None or current.dtype != wp.float32 or current.is_index:
            raise ValueError("Expected float32 scalar boundary grids")
        info = current.get_grid_info()
        if info.grid_index != index or info.grid_count != count:
            raise ValueError("NanoVDB grid headers disagree with file metadata")
        grids.append(current)
        current = current.load_next_grid()
    if current is not None:
        raise ValueError("Unexpected additional NanoVDB grid")
    wp.synchronize_device(device)
    return grids


def _verify_file(manifest, record):
    path = _file(manifest, record["files"]["nvdb"])
    if fingerprints.file_digest(path) != record["nvdb_sha256"]:
        raise ValueError("Boundary NVDB contents do not match the published manifest")
    return path


def _load_sdf(manifest, geometry, device):
    record = _read_record(manifest, "sdf")
    if record["geometry"] != geometry:
        raise ValueError("Cached SDF geometry recipe differs from the request")
    mesh_path = _file(manifest, record["files"]["mesh"])
    if fingerprints.file_digest(mesh_path) != record["mesh_file_sha256"]:
        raise ValueError("Cached visual geometry differs from the published mesh")
    distance, = _read_grids(_verify_file(manifest, record), 1, device)
    maximum = record["bake_options"]["max_support_radius"]
    if _check_support(distance, maximum) != maximum:
        raise ValueError("SDF support range differs from its manifest")
    if fingerprints.sdf_digest(distance) != record["sdf_sha256"] or _grid_layout(distance) != record["grid"]:
        raise ValueError("Cached SDF values or layout differ from the manifest")
    _validate_bounds(record["bounds"])
    return record, distance


def load_boundary_asset(artifact, *, support_radius, quadrature_points=16, device=None):
    """Read a complete artifact without creating, rebuilding or modifying files.

    The visual OBJ reference is returned even if the OBJ is absent. Physics uses
    published local bounds and never needs to open that visual resource.
    """
    check_volume_options(support_radius, quadrature_points)
    artifact = artifact if isinstance(artifact, BoundaryArtifact) else BoundaryArtifact(artifact)
    selected = _device(device)
    path = artifact.manifest_path
    try:
        record = _read_record(path, "boundary")
        expected = dict(sdf_sha256=record["sdf_sha256"], support_radius=float(support_radius),
                        quadrature_points=int(quadrature_points), algorithm_version=VOLUME_MAP_ALGORITHM_VERSION)
        if record["volume_map"] != expected:
            raise ValueError("Boundary volume map does not match the requested h, quadrature or algorithm")
        volume_path = _verify_file(path, record)
        distance, volume_map = _read_grids(volume_path, 2, selected)
        maximum = record["bake_options"]["max_support_radius"]
        if _check_support(distance, support_radius) != maximum:
            raise ValueError("SDF support range differs from its manifest")
        if fingerprints.sdf_digest(distance) != record["sdf_sha256"]:
            raise ValueError("Boundary SDF fingerprint differs from the manifest")
        if volume_map.get_grid_info().name != _map_name(support_radius, quadrature_points):
            raise ValueError("Volume map grid metadata differs from the request")
        if _grid_layout(distance) != record["grid"]:
            raise ValueError("Boundary SDF grid differs from the manifest")
        check_volume_map_layout(distance, volume_map)
        center, half, radius = _validate_bounds(record["bounds"])
        return LoadedBoundaryAsset(artifact, volume_path, _file(path, record["files"]["mesh"]),
                                   distance, volume_map, center, half, radius)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError(f"Cannot load boundary artifact {path}: {error}. Prepare the asset before loading.") from error
