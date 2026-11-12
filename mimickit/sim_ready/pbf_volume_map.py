"""PBF volume-map quadrature on an existing SDF; no files or runtime instances."""

import math
from numbers import Integral

import numpy as np
import warp as wp

wp.set_module_options({"enable_backward": False})

VOLUME_MAP_ALGORITHM_VERSION = 1


def check_volume_options(support_radius, quadrature_points):
    if not math.isfinite(support_radius) or support_radius <= 0:
        raise ValueError("support_radius must be finite and positive")
    if isinstance(quadrature_points, bool) or not isinstance(quadrature_points, Integral) or quadrature_points < 1:
        raise ValueError("quadrature_points must be a positive integer")



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


def precompute_volume_map(
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
    check_volume_options(support_radius, quadrature_points)
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

