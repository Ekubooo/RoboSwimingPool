"""CUDA SDF baking with complete solid distances and a sparse positive band."""

import math
import os
from dataclasses import replace
from pathlib import Path
import tempfile

import numpy as np
import warp as wp

from .config import BakeOptions

wp.set_module_options({"enable_backward": False})


@wp.kernel
def _select_tiles(mesh: wp.uint64, tiles: wp.array(dtype=wp.vec3i), origin: wp.vec3,
                  spacing: float, threshold: float, search_radius: float,
                  selected: wp.array(dtype=wp.int32), errors: wp.array(dtype=wp.int32)):
    thread = wp.tid()
    tile = tiles[thread]
    center = origin + spacing * (wp.vec3(float(tile[0]), float(tile[1]), float(tile[2])) + wp.vec3(3.5))
    hit = wp.mesh_query_point_sign_winding_number(mesh, center, search_radius)
    if hit.result:
        nearest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        distance = hit.sign * wp.length(center - nearest)
        if wp.isfinite(distance):
            selected[thread] = wp.int32(distance <= threshold)
        else:
            wp.atomic_add(errors, 0, 1)
    else:
        wp.atomic_add(errors, 0, 1)


@wp.kernel
def _fill(mesh: wp.uint64, grid: wp.uint64, tiles: wp.array(dtype=wp.vec3i),
          search_radius: float, background: float, errors: wp.array(dtype=wp.int32)):
    thread = wp.tid()
    tile = tiles[thread // 512]
    local = thread % 512
    cell = tile + wp.vec3i(local // 64, (local // 8) % 8, local % 8)
    position = wp.volume_index_to_world(grid, wp.vec3(float(cell[0]), float(cell[1]), float(cell[2])))
    hit = wp.mesh_query_point_sign_winding_number(mesh, position, search_radius)
    if hit.result:
        nearest = wp.mesh_eval_position(mesh, hit.face, hit.u, hit.v)
        distance = hit.sign * wp.length(position - nearest)
        if wp.isfinite(distance):
            wp.volume_store_f(grid, cell[0], cell[1], cell[2], wp.min(distance, background))
        else:
            wp.atomic_add(errors, 0, 1)
    else:
        wp.atomic_add(errors, 0, 1)


def check_options(options):
    if not math.isfinite(options.voxel_size) or options.voxel_size <= 0:
        raise ValueError("Voxel size must be finite and positive")
    if (type(options.padding_voxels) is not int or options.padding_voxels < 1):
        raise ValueError("Padding must be a positive integer number of voxels")
    if not math.isfinite(options.max_support_radius) or options.max_support_radius <= 0:
        raise ValueError("Maximum support radius must be finite and positive")


def bake_rigid(mesh, *, options=BakeOptions(), device="cuda:0"):
    """Cover an independent solid's exterior for PBF volume-map integration.

    The existing baker still controls sparse tiles and distance values. Only
    the sampling domain changes; container padding and caller options remain
    untouched. The occupancy extension and integration each reach h_max.
    """
    check_options(options)
    spacing = options.voxel_size
    margin = 2.0 * options.max_support_radius + 2.0 * math.sqrt(3.0) * spacing
    if not math.isfinite(margin / spacing):
        raise ValueError("Rigid SDF padding exceeds finite voxel coordinates")
    padding = max(options.padding_voxels, math.ceil(margin / spacing))
    return bake(mesh, options=replace(options, padding_voxels=padding), device=device)


def bake(mesh, *, options=BakeOptions(), device="cuda:0"):
    check_options(options)
    selected = wp.get_device(device)
    if not selected.is_cuda:
        raise ValueError("NanoVDB baking requires a CUDA device")
    spacing = options.voxel_size
    extent = mesh.bounds[1] - mesh.bounds[0]
    # The occupancy extension and its integration ball each reach h. Two
    # voxel diagonals protect the SDF and volume-map interpolation stencils.
    background = 2.0 * options.max_support_radius + 2.0 * math.sqrt(3.0) * spacing
    # Keep the original outer sampling domain. The positive band controls
    # sparsity inside it, rather than extending the container's exterior.
    padding = options.padding_voxels
    origin = mesh.bounds[0] - padding * spacing
    dimensions = np.ceil(extent / spacing) + 2 * padding
    if not np.isfinite(dimensions).all() or np.any(dimensions > np.iinfo(np.int32).max - 8):
        raise ValueError("Grid exceeds NanoVDB int32 coordinate limits")
    upper = dimensions.astype(np.int64)
    tile_axes = [np.arange(0, int(stop) + 1, 8, dtype=np.int32) for stop in upper]
    candidates = np.stack(np.meshgrid(*tile_axes, indexing="ij"), axis=-1).reshape(-1, 3)
    candidate_tiles = wp.array(candidates, dtype=wp.vec3i, device=selected)
    gpu_mesh = wp.Mesh(
        points=wp.array(mesh.vertices, dtype=wp.vec3, device=selected),
        indices=wp.array(mesh.faces.reshape(-1), dtype=wp.int32, device=selected),
        support_winding_number=True,
    )
    failures = wp.zeros(1, dtype=wp.int32, device=selected)
    radius = float(np.linalg.norm(((upper // 8 + 1) * 8 - 1) * spacing) + spacing)
    flags = wp.zeros(len(candidates), dtype=wp.int32, device=selected)
    # Signed distance is 1-Lipschitz: a tile containing any sample <= B
    # must have center distance <= B + its sample-center bounding radius.
    tile_radius = 3.5 * math.sqrt(3.0) * spacing
    tolerance = 32.0 * np.finfo(np.float32).eps * max(radius, background, 1.0)
    wp.launch(_select_tiles, dim=len(candidates),
              inputs=[gpu_mesh.id, candidate_tiles, wp.vec3(*origin), spacing,
                      background + tile_radius + tolerance, radius, flags, failures], device=selected)
    wp.synchronize_device(selected)
    if int(failures.numpy()[0]):
        raise RuntimeError(f"Tile distance query failed at {int(failures.numpy()[0])} tiles")
    retained = candidates[flags.numpy().astype(bool)]
    if not len(retained):
        raise RuntimeError("SDF band contains no allocated tiles")
    tiles = wp.array(retained, dtype=wp.vec3i, device=selected)
    volume = wp.Volume.allocate_by_tiles(tiles, spacing, bg_value=background,
        translation=origin.tolist(), device=selected)
    wp.launch(_fill, dim=512 * volume.get_tile_count(),
              inputs=[gpu_mesh.id, volume.id, tiles, radius, background, failures], device=selected)
    wp.synchronize_device(selected)
    if int(failures.numpy()[0]):
        raise RuntimeError(f"Distance query failed at {int(failures.numpy()[0])} voxels")
    volume._pbf_sdf_grid_name = f"PBF_SDF|v=1|hmax={float(options.max_support_radius).hex()}"
    return volume


def save_volume(volume, path):
    """Write beside the destination, replacing an existing file only on success."""
    grid_name = getattr(volume, "_pbf_sdf_grid_name", None)
    if grid_name is not None:
        from .nvdb_io import save_volumes
        return save_volumes([volume], path, grid_names=[grid_name])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".nvdb.tmp", dir=path.parent)
    os.close(descriptor)
    try:
        volume.save_to_nvdb(temporary, codec="none")
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
