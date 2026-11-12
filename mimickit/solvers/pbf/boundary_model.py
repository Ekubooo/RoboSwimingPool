"""Load shared boundary assets and manage static/rigid instances.

Artifacts are prepared by RSP.sim_ready before this read-only runtime is created.
This module owns boundary resources and instances, not construction or rendering.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

from RSP.sim_ready.pipelines.boundary.schema import BoundaryArtifact
from RSP.sim_ready.pipelines.boundary.load import load_boundary_asset
from RSP.sim_ready.config import check_volume_options

wp.set_module_options({"enable_backward": False})


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

    Bounds are read from published metadata; visual meshes are optional.
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
        data = BoundaryDeviceData()
        data.sdf_ids = wp.array([a.sdf.id for a in self.assets], dtype=wp.uint64, device=self.device)
        data.volume_map_ids = wp.array([a.volume_map.id for a in self.assets], dtype=wp.uint64, device=self.device)
        spacing = np.array([a.sdf.get_voxel_size() for a in self.assets], dtype=np.float32).reshape(-1, 3)
        data.inverse_voxel_sizes = wp.array(1.0 / spacing, dtype=wp.vec3, device=self.device)
        centers = np.zeros((len(self.assets), 3), dtype=np.float32)
        halves = np.zeros_like(centers)
        radii = np.zeros(len(self.assets), dtype=np.float32)
        for asset_id in sorted({self.instances[i].asset_id for i in rigid_indices}):
            asset = self.assets[asset_id]
            centers[asset_id] = asset.bounds_center
            halves[asset_id] = asset.bounds_half
            radii[asset_id] = asset.bounds_radius
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


def load_boundaries(static_asset, dynamic_assets, options: BoundaryLoadOptions) -> BoundarySystem:
    """Load prepared artifacts once, preserving the requested instance order.

    No geometry generation, directory discovery, visual-mesh reads or writes
    occur here. Repeated artifact references share the same GPU volumes.
    """
    if isinstance(dynamic_assets, (str, Path, BoundaryArtifact)):
        raise ValueError("dynamic_assets must be a sequence of boundary artifact references")
    check_volume_options(options.support_radius, options.quadrature_points)
    references = tuple(item if isinstance(item, BoundaryArtifact) else BoundaryArtifact(item)
                       for item in (static_asset, *dynamic_assets))
    if len(references) > np.iinfo(np.int32).max:
        raise ValueError("Boundary instance count exceeds int32 indexing")
    device = wp.get_device(options.device)
    if not device.is_cuda:
        raise ValueError("Boundary Volume loading requires a CUDA device")
    assets, instances, asset_ids = [], [], {}
    for index, reference in enumerate(references):
        if reference not in asset_ids:
            loaded = load_boundary_asset(reference, support_radius=options.support_radius,
                                         quadrature_points=options.quadrature_points, device=device)
            asset_ids[reference] = len(assets)
            assets.append(loaded)
        instances.append(BoundaryInstance(asset_ids[reference], index - 1))
    return BoundarySystem(assets, instances, device)
