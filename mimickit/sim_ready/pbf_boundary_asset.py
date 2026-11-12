"""Prepare versioned PBF boundary artifacts, or load them without writing.

SDFs are shared across volume-map variants. Only preparation creates files;
runtime loading needs the NVDB and its manifest, not the optional visual OBJ.
All coordinates and bounds remain in the input mesh's local frame.
"""

from dataclasses import asdict, dataclass, replace
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np
import warp as wp

from . import asset_io, collision, nvdb_io, sdf
from .config import BakeOptions
from .pbf_volume_map import VOLUME_MAP_ALGORITHM_VERSION, check_volume_options, precompute_volume_map
from .publish import publish_files, save_mesh_obj


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


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _key(value):
    # Short directory components also leave room for Windows staging filenames.
    return _digest(value)[:20]


def _write_record(path, record):
    record = dict(record, record_sha256=_digest(record))
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _read_record(path, kind):
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("record_sha256")
    if signature != _digest(record) or record["schema"] != SCHEMA_VERSION or record["kind"] != kind:
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


def _bounds(mesh):
    # Match the runtime's existing float32 center, half-extents and radius.
    vertices = np.asarray(mesh.vertices)
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    center = np.asarray((low + high) * .5, dtype=np.float32)
    half = np.asarray(np.maximum(high - center, center - low), dtype=np.float32)
    radius = float(np.float32(np.linalg.norm(vertices - center, axis=1).max()))
    return dict(center=center.tolist(), half=half.tolist(), radius=radius)


def _validate_bounds(bounds):
    center = np.asarray(bounds["center"], dtype=np.float64)
    half = np.asarray(bounds["half"], dtype=np.float64)
    radius = float(bounds["radius"])
    if (center.shape != (3,) or half.shape != (3,) or not np.isfinite(center).all()
            or not np.isfinite(half).all() or np.any(half <= 0)
            or not math.isfinite(radius) or radius <= 0):
        raise ValueError("Invalid local boundary bounds")
    return tuple(center), tuple(half), radius


def _verify_file(manifest, record):
    path = _file(manifest, record["files"]["nvdb"])
    if asset_io.file_digest(path) != record["nvdb_sha256"]:
        raise ValueError("Boundary NVDB contents do not match the published manifest")
    return path


def _load_sdf(manifest, geometry, device):
    record = _read_record(manifest, "sdf")
    if record["geometry"] != geometry:
        raise ValueError("Cached SDF geometry recipe differs from the request")
    mesh_path = _file(manifest, record["files"]["mesh"])
    if asset_io.file_digest(mesh_path) != record["mesh_file_sha256"]:
        raise ValueError("Cached visual geometry differs from the published mesh")
    distance, = _read_grids(_verify_file(manifest, record), 1, device)
    maximum = record["bake_options"]["max_support_radius"]
    if _check_support(distance, maximum) != maximum:
        raise ValueError("SDF support range differs from its manifest")
    if asset_io.sdf_digest(distance) != record["sdf_sha256"] or _grid_layout(distance) != record["grid"]:
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
        if asset_io.sdf_digest(distance) != record["sdf_sha256"]:
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


def _build_sdf(mesh, checked, manifest, geometry, options, device, provenance):
    manifest.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sdf-", dir=manifest.parent.parent) as temporary:
        stage = Path(temporary)
        volume = (sdf.bake if geometry["container"] else sdf.bake_rigid)(checked, options=options, device=device)
        sdf.save_volume(volume, stage / "sdf.nvdb")
        volume = None
        save_mesh_obj(mesh, stage / "mesh.obj", digits=17)
        persisted, = _read_grids(stage / "sdf.nvdb", 1, device)
        record = dict(schema=SCHEMA_VERSION, kind="sdf", geometry=geometry, bake_options=asdict(options),
                      provenance=provenance, files=dict(nvdb="sdf.nvdb", mesh="mesh.obj"), bounds=_bounds(mesh),
                      sdf_sha256=asset_io.sdf_digest(persisted), grid=_grid_layout(persisted),
                      nvdb_sha256=asset_io.file_digest(stage / "sdf.nvdb"),
                      mesh_file_sha256=asset_io.file_digest(stage / "mesh.obj"))
        _write_record(stage / "sdf.json", record)
        persisted = None
        record, persisted = _load_sdf(stage / "sdf.json", geometry, device)
        publish_files([(stage / name, manifest.parent / name) for name in ("sdf.nvdb", "mesh.obj", "sdf.json")])
        return record, persisted


def prepare_boundary_asset(mesh, checked, *, root, bake_options=BakeOptions(), support_radius,
                           quadrature_points=16, container=False, device="cuda:0", provenance=None):
    """Reuse or publish a boundary artifact, selecting the smallest adequate SDF.

    The SDF family excludes h_max; its individual variants include h_max. Volume
    maps depend on the persisted SDF contents and their own integration settings.
    Provenance describes the source, never runtime bodies, mass or placement.
    """
    check_volume_options(support_radius, quadrature_points)
    sdf.check_options(bake_options)
    selected = _device(device)
    root = asset_io.generated_root(root)
    options = replace(bake_options, max_support_radius=max(float(support_radius), bake_options.max_support_radius))
    base_options = asdict(options)
    del base_options["max_support_radius"]
    geometry = dict(mesh_sha256=collision.mesh_digest(mesh), container=bool(container),
                    bake_options=base_options, sdf_algorithm_version=SDF_ALGORITHM_VERSION,
                    dependencies={p: version(p) for p in ("newton", "warp-lang", "trimesh")})
    directory = root / CACHE_DIRECTORY / _key(geometry)
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
        sdf_manifest = directory / "sdfs" / _key(asdict(options)) / "sdf.json"
        print(f"Baking PBF boundary SDF: {sdf_manifest}", flush=True)
        record, distance = _build_sdf(mesh, checked, sdf_manifest, geometry, options, selected, provenance or {})
    map_recipe = dict(sdf_sha256=record["sdf_sha256"], support_radius=float(support_radius),
                      quadrature_points=int(quadrature_points), algorithm_version=VOLUME_MAP_ALGORITHM_VERSION)
    manifest = directory / "maps" / _key(map_recipe) / "boundary.json"
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
    manifest.parent.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".map-", dir=manifest.parent.parent) as temporary:
        stage = Path(temporary)
        nvdb_io.save_volumes((distance, volume_map), stage / "boundary.nvdb",
                            grid_names=(distance.get_grid_info().name, _map_name(support_radius, quadrature_points)))
        # Temporary and final map directories are siblings, so relative visual references agree.
        mesh_path = _file(sdf_manifest, record["files"]["mesh"])
        boundary_record = dict(record, kind="boundary", volume_map=map_recipe,
                               files=dict(nvdb="boundary.nvdb", mesh=Path(os.path.relpath(mesh_path, manifest.parent)).as_posix()),
                               nvdb_sha256=asset_io.file_digest(stage / "boundary.nvdb"))
        _write_record(stage / "boundary.json", boundary_record)
        load_boundary_asset(stage / "boundary.json", support_radius=support_radius,
                            quadrature_points=quadrature_points, device=selected)
        publish_files([(stage / name, manifest.parent / name) for name in ("boundary.nvdb", "boundary.json")])
    return artifact
