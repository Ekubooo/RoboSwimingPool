"""Boundary types, manifest contract and grid metadata; no build algorithms."""

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import warp as wp

from ...config import VOLUME_MAP_ALGORITHM_VERSION
from ...lib.storage.fingerprints import recipe_digest


SCHEMA_VERSION = 3
SDF_ALGORITHM_VERSION = 1
CACHE_DIRECTORY = "boundaries-v3"


@dataclass(frozen=True)
class BoundaryArtifact:
    manifest_path: Path

    def __post_init__(self):
        object.__setattr__(self, "manifest_path", Path(self.manifest_path).expanduser().resolve())

    def __str__(self):
        return str(self.manifest_path)


@dataclass(frozen=True)
class LoadedBoundaryAsset:
    artifact: BoundaryArtifact
    sdf_path: Path
    mesh_path: Path
    sdf: wp.Volume
    volume_map: wp.Volume
    bounds_center: tuple
    bounds_half: tuple
    bounds_radius: float


def _write_record(path, record):
    record = dict(record, record_sha256=recipe_digest(record))
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_record(path, kind):
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("record_sha256")
    if signature != recipe_digest(record) or record["schema"] != SCHEMA_VERSION or record["kind"] != kind:
        raise ValueError("Invalid boundary artifact manifest or unsupported version")
    return record


def _file(manifest, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Artifact file references must be relative paths")
    return (manifest.parent / relative).resolve()


def _device(device):
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("PBF boundary assets require a CUDA device")
    return selected


def _grid_layout(volume):
    tiles = wp.empty(volume.get_tile_count(), dtype=wp.vec3i, device=volume.device)
    volume.get_tiles(out=tiles)
    coordinates = tiles.numpy()
    order = np.lexsort((coordinates[:, 2], coordinates[:, 1], coordinates[:, 0]))
    canonical = np.ascontiguousarray(coordinates[order], dtype="<i4")
    info = volume.get_grid_info()
    return dict(tile_count=len(coordinates), tiles_sha256=hashlib.sha256(canonical.tobytes()).hexdigest(),
                transform=np.asarray(info.transform_matrix).tolist(),
                translation=np.asarray(info.translation).tolist(), voxel_size=list(volume.get_voxel_size()))


def check_volume_map_layout(distance, volume_map):
    """Compare actual tile coordinates, not just the number of allocated tiles."""
    if _grid_layout(distance) != _grid_layout(volume_map):
        raise ValueError("Volume map grid coverage or coordinate transform differs from SDF")


def _check_support(distance, support_radius):
    name = distance.get_grid_info().name
    if not name.startswith("PBF_SDF|"):
        raise ValueError("Missing PBF SDF support metadata")
    fields = dict(item.split("=", 1) for item in name.split("|")[1:])
    maximum = float.fromhex(fields["hmax"])
    if fields["v"] != "1" or not math.isfinite(maximum) or maximum <= 0:
        raise ValueError("Invalid SDF support metadata")
    if support_radius > maximum:
        raise ValueError(f"SDF supports h <= {maximum:g}, requested {support_radius:g}; prepare a larger SDF")
    return maximum


def _map_name(support_radius, quadrature_points):
    return (f"PBF_VMAP|v={VOLUME_MAP_ALGORITHM_VERSION}"
            f"|h={float(support_radius).hex()}|q={int(quadrature_points)}")


def _validate_bounds(bounds):
    center = np.asarray(bounds["center"], dtype=np.float64)
    half = np.asarray(bounds["half"], dtype=np.float64)
    radius = float(bounds["radius"])
    if (center.shape != (3,) or half.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(half).all() or np.any(half <= 0)
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("Invalid local boundary bounds")
    return tuple(center), tuple(half), radius
