"""CPU layout helpers for prepared scenes."""

from itertools import product
import json

import numpy as np
import trimesh


class ScenePlacementError(ValueError):
    """The sampled pose cannot fit at its scene spawn."""


def asset_bounds(asset):
    bounds = asset.record.get("layout_bounds")
    if bounds is None:
        path = asset.file(next(iter(asset.record["links"].values())))
        bounds = json.loads(path.read_text(encoding="utf-8"))["bounds"]
    return np.asarray(bounds["center"]), np.asarray(bounds["half"])


def container_obstacle_bounds(layout):
    """Return the prepared interior obstacle AABBs in scene coordinates."""
    return tuple(np.asarray(bounds, dtype=float) for bounds in layout.get("obstacles", ()))


def terrain_fluid_region(start, end, layout, *, gap=.02):
    """Translate the initial water block above any overlapping terrain."""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    obstacles = container_obstacle_bounds(layout)
    if "fluid_model_bounds" in layout:
        obstacles = (*obstacles[:-1], np.asarray(layout["fluid_model_bounds"], dtype=float))
    tops = [bounds[1, 2] for bounds in obstacles
            if np.all(start[:2] < bounds[1, :2]) and np.all(bounds[0, :2] < end[:2])]
    lift = max(0., max(tops) + gap - start[2]) if tops else 0.
    if end[2] + lift > layout["size"][2]:
        raise ValueError("The initial water block does not fit above the terrain")
    shift = np.array([0., 0., lift])
    return start + shift, end + shift


def terrain_character_lift(body_bounds, layout, *, gap=.02):
    """Conservative reset-only vertical clearance for a sampled character pose."""
    bounds = np.asarray(body_bounds, dtype=float)
    if bounds.ndim != 3 or bounds.shape[1:] != (2, 3) or not len(bounds) or not np.isfinite(bounds).all():
        raise ValueError("Expected finite character body bounds")
    size = np.asarray(layout["size"], dtype=float)
    if np.any(bounds[:, 0, :2] < gap) or np.any(bounds[:, 1, :2] > size[:2] - gap):
        raise ScenePlacementError("Sampled humanoid pose intersects the container walls at this spawn")
    lift = max(0., gap - float(bounds[:, 0, 2].min()))
    for obstacle in container_obstacle_bounds(layout):
        overlap = (np.all(bounds[:, 0, :2] < obstacle[1, :2] + gap, axis=1)
                   & np.all(obstacle[0, :2] - gap < bounds[:, 1, :2], axis=1))
        if overlap.any():
            lift = max(lift, float(obstacle[1, 2] + gap - bounds[overlap, 0, 2].min()))
    if float(bounds[:, 1, 2].max()) + lift > size[2] - gap:
        raise ScenePlacementError("Sampled humanoid pose cannot fit above the terrain")
    return lift


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
