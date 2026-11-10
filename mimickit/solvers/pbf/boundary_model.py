"""Load shared boundary assets and manage static/rigid instances.

Volume-map generation and file caching are independent of instance counts and
poses. This module owns boundary resources, not rigid-body dynamics or rendering.
"""

from dataclasses import dataclass
import math
from numbers import Integral
from pathlib import Path

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})


# =============================================================================
# 1. Default scene configuration
# Configure asset paths and the static container. The volume-map
# algorithm version determines cache validity.
# =============================================================================

ASSET_DIR = Path(__file__).resolve().parent / "Assets"
SDF_FILENAME = "stairs_scene.nvdb"
VOLUME_MAP_ALGORITHM_VERSION = 1
# Compile-time choice for all dynamic assets: "sphere" or "box".
BOUNDARY_CULLING = "box"


# =============================================================================
# 2. Host configuration and asset records
# Describe scene selection, loading options, shared Volumes and instance
# relationships without loading resources or solving dynamics.
# =============================================================================

@dataclass(frozen=True)
class BoundaryLoadOptions:
    support_radius: float
    device: str | wp.Device | None = None
    quadrature_points: int = 16


@dataclass(frozen=True)
class BoundaryAsset:
    sdf_path: Path
    mesh_path: Path
    sdf: wp.Volume
    volume_map: wp.Volume


@dataclass(frozen=True)
class BoundaryInstance:
    asset_id: int
    # Internal rigid index, not a Newton body_id. Static instances use -1.
    rigid_id: int = -1

    @property
    def is_dynamic(self):
        return self.rigid_id >= 0


# =============================================================================
# 3. GPU data and boundary sampling
# Define kernel data layouts and use instance poses to transform world-space
# query points into asset-local coordinates before sampling.
# =============================================================================

@wp.struct
class SDFDeviceData:
    volume_id: wp.uint64
    inverse_voxel_size: wp.vec3


@wp.struct
class BoundaryDeviceData:
    """Stable device buffers; resource ownership remains in BoundarySystem."""

    sdf_ids: wp.array(dtype=wp.uint64)
    volume_map_ids: wp.array(dtype=wp.uint64)
    inverse_voxel_sizes: wp.array(dtype=wp.vec3)
    bounds_center: wp.array(dtype=wp.vec3)
    bounds_half: wp.array(dtype=wp.vec3)
    bounds_radius: wp.array(dtype=float)
    bounds_padding: wp.array(dtype=float)
    asset_ids: wp.array(dtype=int)
    rigid_ids: wp.array(dtype=int)
    transforms: wp.array(dtype=wp.transform)
    rigid_instance_ids: wp.array(dtype=int)
    body_ids: wp.array(dtype=int)
    local_transforms: wp.array(dtype=wp.transform)
    world_com: wp.array(dtype=wp.vec3)
    count: int
    rigid_count: int


@wp.struct
class BoundaryCouplingData:
    """Boundary queries and solver-owned body-indexed pressure wrenches."""

    instances: BoundaryDeviceData
    forces: wp.array(dtype=wp.spatial_vector)


@wp.func
def sample_boundary_sdf(boundary: SDFDeviceData, position: wp.vec3):
    """Return distance and world-space gradient within the baked grid domain.

    The baker uses axis-aligned, uniformly spaced voxels. Outside allocated
    voxels NanoVDB returns its background value, not a mesh distance query.
    """
    index = wp.volume_world_to_index(boundary.volume_id, position)
    gradient = wp.vec3(0.0)
    distance = wp.volume_sample_grad_f(boundary.volume_id, index, wp.Volume.LINEAR, gradient)
    world_gradient = wp.cw_mul(gradient, boundary.inverse_voxel_size)
    return distance, world_gradient


@wp.func
def sample_local_sdf(data: BoundaryDeviceData, asset_id: int, local: wp.vec3):
    boundary = SDFDeviceData()
    boundary.volume_id = data.sdf_ids[asset_id]
    boundary.inverse_voxel_size = data.inverse_voxel_sizes[asset_id]
    return sample_boundary_sdf(boundary, local)


@wp.func
def sample_instance_sdf(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Unfiltered sampling, independent of the finite mesh bounds."""
    transform = data.transforms[instance_id]
    local = wp.transform_point(wp.transform_inverse(transform), position)
    distance, gradient = sample_local_sdf(data, data.asset_ids[instance_id], local)
    return distance, wp.transform_vector(transform, gradient)


@wp.func
def sample_instance_volume(data: BoundaryDeviceData, instance_id: int, position: wp.vec3):
    """Return the cached boundary volume [m^3] at a world-space position."""
    local = wp.transform_point(wp.transform_inverse(data.transforms[instance_id]), position)
    volume_id = data.volume_map_ids[data.asset_ids[instance_id]]
    index = wp.volume_world_to_index(volume_id, local)
    return wp.volume_sample_f(volume_id, index, wp.Volume.LINEAR)


@wp.func
def inside_boundary_bounds(data: BoundaryDeviceData, asset_id: int, local: wp.vec3, margin: float):
    offset = local - data.bounds_center[asset_id]
    padding = margin + data.bounds_padding[asset_id]
    if wp.static(BOUNDARY_CULLING == "sphere"):
        radius = data.bounds_radius[asset_id] + padding
        return wp.dot(offset, offset) <= radius * radius
    else:
        half = data.bounds_half[asset_id] + wp.vec3(padding)
        return (wp.abs(offset[0]) <= half[0] and wp.abs(offset[1]) <= half[1]
                and wp.abs(offset[2]) <= half[2])


@wp.func
def query_instance_normal(data: BoundaryDeviceData, instance_id: int, position: wp.vec3, margin: float):
    transform = data.transforms[instance_id]
    local = wp.transform_point(wp.transform_inverse(transform), position)
    asset_id = data.asset_ids[instance_id]
    if not inside_boundary_bounds(data, asset_id, local, margin):
        return False, 0.0, wp.vec3(0.0)
    distance, local_gradient = sample_local_sdf(data, asset_id, local)
    gradient = wp.transform_vector(transform, local_gradient)
    length_squared = wp.dot(gradient, gradient)
    if not wp.isfinite(distance) or not wp.isfinite(length_squared) or length_squared <= 1.0e-12:
        return False, 0.0, wp.vec3(0.0)
    return True, distance, gradient / wp.sqrt(length_squared)


@wp.func
def query_instance_boundary(data: BoundaryDeviceData, instance_id: int,
                            position: wp.vec3, support_radius: float):
    valid, distance, normal = query_instance_normal(data, instance_id, position, support_radius)
    if not valid or distance <= 0.0 or distance >= support_radius:
        return False, 0.0, wp.vec3(0.0)
    pressure_distance = wp.max(distance + 0.125 * support_radius, 0.5 * support_radius)
    return True, pressure_distance, -normal


# =============================================================================
# 4. Instance management and pose updates
# BoundarySystem owns resources and fixed-size device buffers, updates rigid
# poses and resets transforms in place.
# =============================================================================

@wp.kernel
def _set_rigid_poses(poses: wp.array(dtype=wp.transform), instance_ids: wp.array(dtype=int),
                     transforms: wp.array(dtype=wp.transform)):
    i = wp.tid()
    transforms[instance_ids[i]] = poses[i]


@wp.kernel
def _sync_body_poses(data: BoundaryDeviceData, body_q: wp.array(dtype=wp.transform),
                     body_com: wp.array(dtype=wp.vec3)):
    rigid = wp.tid()
    body = data.body_ids[rigid]
    transform = body_q[body]
    data.transforms[data.rigid_instance_ids[rigid]] = transform * data.local_transforms[rigid]
    data.world_com[rigid] = wp.transform_point(transform, body_com[body])


class BoundarySystem:
    """Shared Volumes plus a fixed-size instance table on one CUDA device.

    Dynamic meshes are read once for bounds and reused by rendering.
    Transforms map asset-local coordinates
    to world coordinates and initially equal identity. GPU pose inputs must
    contain finite translations and normalized XYZW quaternions.
    """

    def __init__(self, assets, instances, device):
        self.assets = tuple(assets)
        self.instances = tuple(instances)
        self.device = wp.get_device(device)
        rigid_indices = [i for i, instance in enumerate(self.instances) if instance.is_dynamic]
        self.rigid_count = len(rigid_indices)
        self.static_count = len(self.instances) - self.rigid_count
        self._meshes = {}
        data = BoundaryDeviceData()
        data.sdf_ids = wp.array([a.sdf.id for a in self.assets], dtype=wp.uint64, device=self.device)
        data.volume_map_ids = wp.array([a.volume_map.id for a in self.assets], dtype=wp.uint64, device=self.device)
        spacing = np.array([a.sdf.get_voxel_size() for a in self.assets], dtype=np.float32).reshape(-1, 3)
        data.inverse_voxel_sizes = wp.array(1.0 / spacing, dtype=wp.vec3, device=self.device)
        centers = np.zeros((len(self.assets), 3), dtype=np.float32)
        halves = np.zeros_like(centers)
        radii = np.zeros(len(self.assets), dtype=np.float32)
        for asset_id in sorted({self.instances[i].asset_id for i in rigid_indices}):
            vertices, _ = self.mesh_data(asset_id)
            low, high = vertices.min(axis=0), vertices.max(axis=0)
            centers[asset_id] = (low + high) * .5
            halves[asset_id] = np.maximum(high - centers[asset_id], centers[asset_id] - low)
            radii[asset_id] = np.linalg.norm(vertices - centers[asset_id], axis=1).max()
        data.bounds_center = wp.array(centers, dtype=wp.vec3, device=self.device)
        data.bounds_half = wp.array(halves, dtype=wp.vec3, device=self.device)
        data.bounds_radius = wp.array(radii, dtype=float, device=self.device)
        # One interpolation-cell diagonal, in addition to each query's reach.
        data.bounds_padding = wp.array(np.linalg.norm(spacing, axis=1), dtype=float, device=self.device)
        data.asset_ids = wp.array([i.asset_id for i in self.instances], dtype=int, device=self.device)
        data.rigid_ids = wp.array([i.rigid_id for i in self.instances], dtype=int, device=self.device)
        data.rigid_instance_ids = wp.array(rigid_indices, dtype=int, device=self.device)
        data.body_ids = wp.full(self.rigid_count, -1, dtype=int, device=self.device)
        data.local_transforms = wp.empty(self.rigid_count, dtype=wp.transform, device=self.device)
        data.local_transforms.fill_(wp.transform_identity())
        data.world_com = wp.zeros(self.rigid_count, dtype=wp.vec3, device=self.device)
        data.transforms = wp.empty(len(self.instances), dtype=wp.transform, device=self.device)
        data.count = len(self.instances)
        data.rigid_count = self.rigid_count
        self.device_data = data
        self.bound_model = None
        self.reset()

    def mesh_data(self, asset_id):
        """Share one host mesh between bounds and all rendered instances."""
        if asset_id not in self._meshes:
            import trimesh
            mesh = trimesh.load(self.assets[asset_id].mesh_path, force="mesh", process=False)
            self._meshes[asset_id] = (np.asarray(mesh.vertices), np.asarray(mesh.faces))
        return self._meshes[asset_id]

    def bind_bodies(self, model, body_ids, local_transforms=None):
        """Bind asset instances to Newton bodies once during scene assembly.

        Inputs follow internal rigid order; transforms map asset to body local
        coordinates. COM may differ from both origins. Per-step sync trusts this
        mapping and never scans or reallocates the device arrays.
        """
        ids = np.asarray(body_ids)
        if (ids.shape != (self.rigid_count,) or not np.issubdtype(ids.dtype, np.integer)
                or np.any(ids < 0) or np.any(ids >= model.body_count)):
            raise ValueError(f"Expected {self.rigid_count} valid integer Newton body IDs")
        if model.device != self.device:
            raise ValueError("Boundary and Newton model devices must match")
        if self.bound_model is not None:
            raise ValueError("Boundary bodies are already bound")
        if local_transforms is not None:
            values = local_transforms.numpy() if isinstance(local_transforms, wp.array) else local_transforms
            values = np.array(values, dtype=np.float32, copy=True)
            if values.shape != (self.rigid_count, 7) or not np.isfinite(values).all():
                raise ValueError(f"Expected finite asset-to-body transforms with shape ({self.rigid_count}, 7)")
            norms = np.linalg.norm(values[:, 3:].astype(np.float64), axis=1)
            if np.any(norms == 0):
                raise ValueError("Asset-to-body quaternions must be nonzero")
            values[:, 3:] /= norms[:, None]
            self.device_data.local_transforms.assign(values)
        self.device_data.body_ids.assign(ids.astype(np.int32))
        self.bound_model = model

    def sync_body_poses(self, state):
        """Use the assembly-time mapping to update boundary poses and world COM."""
        wp.launch(_sync_body_poses, self.rigid_count,
            inputs=[self.device_data, state.body_q, self.bound_model.body_com], device=self.device)

    def set_rigid_poses(self, poses):
        """Update by internal rigid order, preserving every static transform.

        CPU inputs are N x 7 arrays (XYZ, XYZW); quaternions are normalized on
        a copy. Warp inputs are one-dimensional transform arrays and are copied
        on device, without downloading data or reallocating instance buffers.
        """
        if isinstance(poses, wp.array):
            if poses.dtype != wp.transform or poses.shape != (self.rigid_count,):
                raise ValueError(f"Expected {self.rigid_count} Warp transforms")
            source = poses.to(self.device)
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                values = np.array(poses, dtype=np.float32, copy=True)
            if values.shape != (self.rigid_count, 7) or not np.isfinite(values).all():
                raise ValueError(f"Expected finite rigid poses with shape ({self.rigid_count}, 7)")
            norms = np.linalg.norm(values[:, 3:].astype(np.float64), axis=1)
            if np.any(norms == 0):
                raise ValueError("Rigid pose quaternions must be nonzero")
            values[:, 3:] /= norms[:, None]
            source = wp.array(values, dtype=wp.transform, device=self.device)
        if self.rigid_count:
            wp.launch(_set_rigid_poses, self.rigid_count,
                inputs=[source, self.device_data.rigid_instance_ids],
                outputs=[self.device_data.transforms], device=self.device)

    def reset(self):
        """Restore identity poses in place, retaining all loaded resources."""
        if self.device_data.count:
            self.device_data.transforms.fill_(wp.transform_identity())


# =============================================================================
# 5. Input validation and path resolution
# Validate loading and quadrature inputs and explicit OBJ/NVDB pairs.
# =============================================================================

def _check_volume_options(support_radius, quadrature_points):
    if not math.isfinite(support_radius) or support_radius <= 0:
        raise ValueError("support_radius must be finite and positive")
    if isinstance(quadrature_points, bool) or not isinstance(quadrature_points, Integral) or quadrature_points < 1:
        raise ValueError("quadrature_points must be a positive integer")


def resolve_sdf_path(filename):
    if (not isinstance(filename, str) or not filename.strip()
            or any(character in filename for character in "/\\:\x00")
            or Path(filename).suffix.lower() != ".nvdb"):
        raise ValueError("SDF must be a bare .nvdb filename without a directory or drive")
    return (Path(ASSET_DIR) / "boundary" / filename).resolve()


def resolve_scene_path(filename):
    """The renderer consumes the OBJ paired with the selected NanoVDB stem."""
    return resolve_sdf_path(filename).with_suffix(".obj")


def _resolve_boundary_paths(static_asset, dynamic_assets):
    """Check all selected attachments before loading or rewriting any VM cache."""
    if isinstance(dynamic_assets, (str, Path)):
        raise ValueError("dynamic_assets must be a sequence of NVDB paths")
    paths = tuple(Path(path).expanduser().resolve() for path in (static_asset, *dynamic_assets))
    if len(paths) > np.iinfo(np.int32).max:
        raise ValueError("Boundary instance count exceeds int32 indexing")
    for path in dict.fromkeys(paths):
        if path.suffix.lower() != ".nvdb":
            raise ValueError(f"Boundary asset must be an NVDB path: {path}")
        for attachment in (path, path.with_suffix(".obj")):
            if not attachment.is_file():
                raise FileNotFoundError(f"Missing boundary asset: {attachment}. "
                                        "Prepare assets offline with Sim_Ready; see Sim_Ready/README.md.")
    return paths


# =============================================================================
# 6. Volume-map quadrature and GPU precomputation
# Build support-ball quadrature points and compute the volume map on the SDF's
# existing sparse grid. File caching is handled separately below.
# =============================================================================

def _build_volume_quadrature(support_radius, quadrature_points):
    """Return ball-filtered Gauss offsets and physical volume weights.

    ``quadrature_points`` is the number of one-dimensional Gauss points,
    not the polynomial order. The cube-to-ball cutoff is deliberately not
    renormalized, matching SPlisHSPlasH's volume-map quadrature.
    """
    nodes, weights = np.polynomial.legendre.leggauss(int(quadrature_points))
    points = np.stack(np.meshgrid(nodes, nodes, nodes, indexing="ij"), axis=-1).reshape(-1, 3)
    products = (weights[:, None, None] * weights[None, :, None] * weights[None, None, :]).reshape(-1)
    inside = np.sum(points * points, axis=1) <= 1.0
    offsets = np.asarray(points[inside] * support_radius, dtype=np.float32)
    volume_weights = np.asarray(products[inside] * support_radius**3, dtype=np.float32)
    if (not np.isfinite(offsets).all() or not np.isfinite(volume_weights).all()
            or np.any(volume_weights <= 0)):
        raise ValueError("Quadrature offsets and weights must be representable as positive finite float32 weights")
    return offsets, volume_weights


@wp.func
def _volume_occupancy(distance: float, support_radius: float):
    """Cubic spline extension gamma*: one inside, zero beyond the radius."""
    if distance <= 0.0:
        return 1.0
    q = distance / support_radius
    if q <= 0.5:
        return 1.0 - 6.0 * q * q + 6.0 * q * q * q
    if q < 1.0:
        t = 1.0 - q
        return 2.0 * t * t * t
    return 0.0


@wp.kernel
def precompute_VM(
    sdf_id: wp.uint64,
    volume_map_id: wp.uint64,
    tiles: wp.array(dtype=wp.vec3i),
    offsets: wp.array(dtype=wp.vec3),
    volume_weights: wp.array(dtype=float),
    support_radius: float,
):
    tid = wp.tid()
    local = tid % 512
    cell = tiles[tid // 512] + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    cell_f = wp.vec3(float(cell[0]), float(cell[1]), float(cell[2]))
    x = wp.volume_index_to_world(volume_map_id, cell_f)
    volume = float(0.0)
    for sample in range(offsets.shape[0]):
        index = wp.volume_world_to_index(sdf_id, x + offsets[sample])
        distance = wp.volume_sample_f(sdf_id, index, wp.Volume.LINEAR)
        volume += volume_weights[sample] * _volume_occupancy(distance, support_radius)
    wp.volume_store_f(volume_map_id, cell[0], cell[1], cell[2], volume)


def precompute_volumeMap(
    sdf: wp.Volume,
    support_radius: float,
    *,
    quadrature_points: int = 16,
) -> wp.Volume:
    """Integrate gamma*(SDF) over each node's support ball, in cubic meters.

    Return a completed, independent float32 CUDA volume with the input's
    allocated tiles and coordinate transform, but a zero background. The input
    is unchanged. Changing the support radius requires recomputing the map.

    Sampling outside allocated SDF voxels uses the input's NanoVDB background.
    Accurate geometric integrals require the SDF to cover the entire support
    ball plus the trilinear interpolation stencil at each queried map node.
    No mesh reconstruction, padding, file IO, or solver integration is done.
    """
    _check_volume_options(support_radius, quadrature_points)
    if not isinstance(sdf, wp.Volume):
        raise TypeError("sdf must be a wp.Volume")
    if not sdf.device.is_cuda:
        raise ValueError("Volume-map precomputation requires a CUDA SDF")
    if sdf.dtype != wp.float32 or sdf.is_index:
        raise ValueError("SDF must be a float32 scalar distance grid")
    return _precompute_volume_map(sdf, support_radius, quadrature_points)


def _precompute_volume_map(sdf, support_radius, quadrature_points):
    # Loading and the standalone entry point have already checked the inputs.
    tile_count = sdf.get_tile_count()
    if tile_count == 0:
        raise ValueError("SDF must have at least one allocated tile")
    offsets, volume_weights = _build_volume_quadrature(support_radius, quadrature_points)

    device = sdf.device
    tiles = wp.empty(tile_count, dtype=wp.vec3i, device=device)
    sdf.get_tiles(out=tiles)
    info = sdf.get_grid_info()
    volume_map = wp.Volume.allocate_by_tiles(
        tiles, transform=info.transform_matrix, translation=info.translation,
        bg_value=0.0, device=device,
    )
    gpu_offsets = wp.array(offsets, dtype=wp.vec3, device=device)
    gpu_weights = wp.array(volume_weights, dtype=float, device=device)
    wp.launch(
        precompute_VM,
        dim=tile_count * 512,
        inputs=[sdf.id, volume_map.id, tiles, gpu_offsets, gpu_weights, float(support_radius)],
        device=device,
    )
    wp.synchronize_device(device)
    return volume_map


# =============================================================================
# 7. NanoVDB loading and in-file caching
# Read and validate SDF/volume-map pairs. Rebuild stale caches, atomically save
# them to the same NVDB file and reload the persisted grids.
# =============================================================================

def _nvdb_io():
    # Direct PBF.py execution puts only its own directory on sys.path.
    if not __package__:
        import sys
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
    from sim_ready import nvdb_io
    return nvdb_io


def load_static_sdf(filename, *, device=None) -> wp.Volume:
    """Read an offline SDF from Assets/boundary; the caller must retain the Volume.

    This startup operation neither reads an OBJ nor invokes the offline baker.
    """
    path = resolve_sdf_path(filename)
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("Static NanoVDB SDF loading requires a CUDA device")
    with path.open("rb") as stream:
        volume = wp.Volume.load_from_nvdb(stream, device=selected)
    if volume.dtype != wp.float32 or volume.is_index:
        raise ValueError("Static SDF must be a float32 scalar distance grid")
    wp.synchronize_device(selected)
    print(f"Static SDF loaded: {path} (voxel size {volume.get_voxel_size()[0]:g} m)")

    return volume


def _volume_map_name(support_radius, quadrature_points):
    return (f"PBF_VMAP|v={VOLUME_MAP_ALGORITHM_VERSION}"
            f"|h={float(support_radius).hex()}|q={int(quadrature_points)}")


def _check_sdf_support(sdf, support_radius):
    name = sdf.get_grid_info().name
    if not name.startswith("PBF_SDF|"):
        return  # Legacy full-domain assets have no positive-side truncation.
    try:
        fields = dict(item.split("=", 1) for item in name.split("|")[1:])
        maximum = float.fromhex(fields["hmax"])
        if fields["v"] != "1" or not math.isfinite(maximum) or maximum <= 0:
            raise ValueError("Unsupported SDF metadata")
    except (KeyError, ValueError) as error:
        raise ValueError(f"Invalid sparse SDF metadata: {name}") from error
    if support_radius > maximum:
        raise ValueError(f"SDF supports h <= {maximum:g} m, requested {support_radius:g} m; "
                         "regenerate the asset with a larger max support radius")


def _read_boundary_volumes(path, device):
    count = _nvdb_io().grid_count(path)
    with path.open("rb") as stream:
        sdf = wp.Volume.load_from_nvdb(stream, device=device)
    if sdf.dtype != wp.float32 or sdf.is_index:
        raise ValueError("Static SDF must be a float32 scalar distance grid")
    info = sdf.get_grid_info()
    if info.grid_index != 0 or info.grid_count != count:
        raise ValueError("NanoVDB file and first grid headers disagree on grid count or index")
    if info.name.startswith("PBF_VMAP|"):
        raise ValueError("The first NanoVDB grid must be the SDF, not a volume map")
    volume_map = sdf.load_next_grid()
    if (count == 2) != (volume_map is not None):
        raise ValueError("NanoVDB file and grid headers disagree on grid count")
    if volume_map is not None:
        info = volume_map.get_grid_info()
        if info.grid_index != 1 or info.grid_count != count:
            raise ValueError("NanoVDB second grid header has an invalid count or index")
        if volume_map.dtype != wp.float32 or volume_map.is_index:
            raise ValueError("Volume map must be a float32 scalar grid")
        if volume_map.load_next_grid() is not None:
            raise ValueError("Unexpected third NanoVDB grid")
    wp.synchronize_device(device)
    return sdf, volume_map


def _check_volume_map_layout(sdf, volume_map):
    a, b = sdf.get_grid_info(), volume_map.get_grid_info()
    if (not np.array_equal(a.transform_matrix, b.transform_matrix)
            or not np.array_equal(a.translation, b.translation)
            or sdf.get_tile_count() != volume_map.get_tile_count()):
        raise ValueError("Volume map coordinate transform or tile count differs from SDF")


def load_boundary_volumes(path, support_radius, *, quadrature_points=16, device=None):
    """Load a matched grid pair, or compute, atomically cache and reload it.

    Only the selected NVDB is updated. Geometry baking and asset publication
    remain the responsibility of Sim_Ready's standalone pipeline.
    """
    _check_volume_options(support_radius, quadrature_points)
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("Static NanoVDB boundary loading requires a CUDA device")
    return _load_boundary_volumes(Path(path).expanduser().resolve(), support_radius, quadrature_points, selected)


def _load_boundary_volumes(path, support_radius, quadrature_points, selected):
    expected = _volume_map_name(support_radius, quadrature_points)
    try:
        sdf, volume_map = _read_boundary_volumes(path, selected)
        _check_sdf_support(sdf, support_radius)
        if volume_map is not None and volume_map.get_grid_info().name == expected:
            _check_volume_map_layout(sdf, volume_map)
            print(f"Boundary cache hit: {path} (h={support_radius:g} m, q={quadrature_points})")
            return sdf, volume_map
        print(f"Boundary cache rebuild: {path} (h={support_radius:g} m, q={quadrature_points})")
        volume_map = None
        volume_map = _precompute_volume_map(sdf, support_radius, quadrature_points)
        _nvdb_io().save_volumes(
            (sdf, volume_map), path, grid_names=(sdf.get_grid_info().name, expected),
        )
        # Reload the persisted pair through the same reader, without retaining
        # the old GPU allocations (including any stale map aliased by sdf).
        sdf = volume_map = None
        sdf, volume_map = _read_boundary_volumes(path, selected)
        if volume_map is None or volume_map.get_grid_info().name != expected:
            raise ValueError("Saved boundary cache metadata does not match requested parameters")
        _check_volume_map_layout(sdf, volume_map)
        return sdf, volume_map
    except Exception as error:
        raise RuntimeError(f"Cannot load boundary volumes {path}: {error}. "
                           "Prepare assets offline with Sim_Ready; see Sim_Ready/README.md.") from error


# =============================================================================
# 8. Unified scene loading entry points
# Load each selected asset once and preserve the explicit instance order.
# =============================================================================

def load_boundaries(static_asset, dynamic_assets, options: BoundaryLoadOptions) -> BoundarySystem:
    """Load one static boundary and an ordered sequence of dynamic instances.

    Repeated NVDB paths share one asset. An empty dynamic sequence loads only
    the static boundary; no directory discovery or geometry baking is done.
    """
    paths = _resolve_boundary_paths(static_asset, dynamic_assets)
    _check_volume_options(options.support_radius, options.quadrature_points)
    device = wp.get_device(options.device)
    if not device.is_cuda:
        raise ValueError("Boundary Volume loading requires a CUDA device")
    assets, instances, asset_ids = [], [], {}
    for index, path in enumerate(paths):
        if path not in asset_ids:
            sdf, volume_map = _load_boundary_volumes(
                path, options.support_radius, options.quadrature_points, device,
            )
            asset_ids[path] = len(assets)
            assets.append(BoundaryAsset(path, path.with_suffix(".obj"), sdf, volume_map))
        instances.append(BoundaryInstance(asset_ids[path], index - 1))
    return BoundarySystem(assets, instances, device)
