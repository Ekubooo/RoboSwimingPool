"""CPU layout helpers for published scenes."""

from itertools import product
import json

import numpy as np
import trimesh


def asset_bounds(asset):
    path = asset.file(next(iter(asset.record["links"].values())))
    bounds = json.loads(path.read_text(encoding="utf-8"))["bounds"]
    return np.asarray(bounds["center"]), np.asarray(bounds["half"])


def container_obstacle_bounds(asset):
    mesh = trimesh.load(asset.file(asset.record["bunny"]), force="mesh", process=False)
    return np.asarray(mesh.bounds)


def transformed_bounds(center, half, position, rotation):
    """Conservative world-space AABB of local bounds and an XYZW transform."""
    q = rotation
    matrix = trimesh.transformations.quaternion_matrix([q[3], *q[:3]])[:3, :3]
    center = np.asarray(position) + matrix @ center
    half = np.abs(matrix) @ half
    return np.array([center - half, center + half])


def bounds_gap(first, second):
    """Largest signed separation along any axis."""
    return max(float(np.max(first[0] - second[1])), float(np.max(second[0] - first[1])))


def place_props(requests, *, lower, upper, obstacles, mode, seed, gap, labels=None):
    """Return instance positions in order, or fail if the whole layout cannot fit."""
    if not requests:
        return []
    if mode not in ("regular", "random"):
        raise ValueError("Select an available placement mode")
    max_half = np.max([half for _, half in requests], axis=0)
    axes = [np.arange(lo + h, hi - h + 1e-9, 2 * h + gap)
            for lo, hi, h in zip(lower, upper, max_half)]
    grid = iter(product(*axes))
    occupied = list(obstacles)
    rng = np.random.default_rng(seed)
    positions = []
    for index, (local_center, half) in enumerate(requests):
        if np.any(upper - lower < 2 * half):
            label = labels[index] if labels is not None else f"Prop {index + 1}"
            raise ValueError(f"{label} does not fit in the current region")
        candidates = (rng.uniform(lower + half, upper - half) for _ in range(200)) if mode == "random" else grid
        for center in candidates:
            center = np.asarray(center)
            bounds = np.array([center - half, center + half])
            if any(not np.any((bounds[1] + gap <= other[0]) | (other[1] + gap <= bounds[0]))
                   for other in occupied):
                continue
            occupied.append(bounds)
            positions.append(center - local_center)
            break
        else:
            raise ValueError("Not enough space. Reduce quantities, choose another container, or reshuffle.")
    return positions
