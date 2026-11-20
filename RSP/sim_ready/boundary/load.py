"""Strictly read-only boundary loading; imports no baking or decomposition code."""

import warp as wp

from ..config import VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from ..lib.storage import fingerprints, nvdb_io
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
        raise ValueError("Boundary NVDB contents do not match its manifest")
    return path


def _boundary_grid_count(record):
    state = record.get("state", "ready")
    settings = record["volume_map"]
    if state == "ready" and isinstance(settings, dict):
        return 2
    if state == "sdf_ready" and settings is None:
        return 1
    raise ValueError("Invalid boundary state or volume map metadata")


def _read_sdf(manifest, record, support_radius, device):
    """Verify the distance field in either preparation state."""
    count = _boundary_grid_count(record)
    path = _verify_file(manifest, record)
    grids = _read_grids(path, count, device)
    distance = grids[0]
    if _check_support(distance, support_radius) != record["bake_options"]["max_support_radius"]:
        raise ValueError("SDF support range differs from its manifest")
    if fingerprints.sdf_digest(distance) != record["sdf_sha256"]:
        raise ValueError("Boundary SDF fingerprint differs from the manifest")
    if _grid_layout(distance) != record["grid"]:
        raise ValueError("Boundary SDF grid differs from the manifest")
    return path, grids


def load_prepared_sdf(artifact, *, support_radius, device=None):
    """Read the validated distance field from either a single-grid or complete artifact."""
    check_volume_options(support_radius, 16)
    artifact = artifact if isinstance(artifact, BoundaryArtifact) else BoundaryArtifact(artifact)
    path = artifact.manifest_path
    try:
        record = _read_record(path, "boundary")
        _, grids = _read_sdf(path, record, support_radius, _device(device))
        _validate_bounds(record["bounds"])
        return grids[0]
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError(f"Cannot load prepared SDF {path}: {error}. Prepare the asset before loading.") from error


def load_boundary_asset(artifact, *, support_radius, quadrature_points=16, device=None):
    """Read a complete artifact without creating, rebuilding or modifying files.

    The visual OBJ reference is returned even if the OBJ is absent. Physics uses
    generated local bounds and never needs to open that visual resource.
    """
    check_volume_options(support_radius, quadrature_points)
    artifact = artifact if isinstance(artifact, BoundaryArtifact) else BoundaryArtifact(artifact)
    selected = _device(device)
    path = artifact.manifest_path
    try:
        record = _read_record(path, "boundary")
        if _boundary_grid_count(record) != 2:
            raise ValueError("Boundary SDF is prepared but its volume map is missing")
        expected = dict(sdf_sha256=record["sdf_sha256"], support_radius=float(support_radius),
                        quadrature_points=int(quadrature_points), algorithm_version=VOLUME_MAP_ALGORITHM_VERSION)
        if record["volume_map"] != expected:
            raise ValueError("Boundary volume map does not match the requested h, quadrature or algorithm")
        volume_path, (distance, volume_map) = _read_sdf(path, record, support_radius, selected)
        if volume_map.get_grid_info().name != _map_name(support_radius, quadrature_points):
            raise ValueError("Volume map grid metadata differs from the request")
        check_volume_map_layout(distance, volume_map)
        center, half, radius = _validate_bounds(record["bounds"])
        return LoadedBoundaryAsset(artifact, volume_path, _file(path, record["files"]["mesh"]),
                                   distance, volume_map, center, half, radius)
    except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
        raise ValueError(f"Cannot load boundary artifact {path}: {error}. Prepare the asset before loading.") from error
