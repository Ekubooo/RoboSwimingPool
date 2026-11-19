"""Read-only CPU preflight before environment or GPU state construction."""
from dataclasses import dataclass
import json
import numpy as np

from RSP.sim_ready.catalog import AssetCatalog, PublishedAsset
from RSP.sim_ready.config import VOLUME_MAP_ALGORITHM_VERSION, check_volume_options
from RSP.sim_ready.paths import PUBLISHED_ROOT
from .layout import asset_bounds, bounds_gap, transformed_bounds


@dataclass(frozen=True)
class ResolvedScene:
    static: PublishedAsset
    dynamic: dict
    layout: dict
    clearances: list


def resolve_scene(spec, root=PUBLISHED_ROOT, *, support_radius):
    check_volume_options(support_radius, 16)
    catalog = AssetCatalog(root)
    static = catalog.resolve(spec.container)
    layout = static.record.get("container")
    if layout is None:
        raise ValueError("Container package lacks assembly metadata; publish and select a new version")
    size = np.asarray(layout["size"], dtype=float)
    if layout["up_axis"] != "Z" or layout["units"] != "m" or size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
        raise ValueError("Expected a positive, origin-based Z-up container in meters")
    if np.any(np.asarray(spec.fluid_end) > size):
        raise ValueError("Initial fluid region exceeds the container")
    if np.any(np.asarray(spec.spawn_xy) <= 0) or np.any(np.asarray(spec.spawn_xy) >= size[:2]):
        raise ValueError("Humanoid spawn lies outside the container")
    dynamic = {ref: catalog.resolve(ref) for ref in dict.fromkeys((spec.character, *spec.props))}
    # All resources are checked before Newton construction, including boundary parameters.
    boundaries = [(static, static.record["boundary"])]
    for asset in dynamic.values():
        asset.file(asset.record["model"])
        boundaries.extend((asset, relative) for relative in asset.record["links"].values())
    for asset, relative in boundaries:
        record = json.loads(asset.file(relative).read_text(encoding="utf-8"))
        settings = record["volume_map"]
        if (settings["support_radius"] != support_radius or settings["quadrature_points"] != 16
                or settings["algorithm_version"] != VOLUME_MAP_ALGORITHM_VERSION):
            raise ValueError("Published boundary parameters do not match playback")
    obstacles, clearances = [], []
    for instance in spec.instances:
        asset = dynamic[instance.asset]
        if len(asset.record["links"]) != 1:
            raise ValueError("Scene props must contain one rigid body")
        center, half = asset_bounds(asset)
        bounds = transformed_bounds(center, half, instance.position, instance.rotation)
        if np.any(bounds[0] <= 0) or np.any(bounds[1] >= size):
            raise ValueError(f"{instance.name} intersects the container")
        for name, other in obstacles:
            gap = bounds_gap(bounds, other)
            if gap <= 0:
                raise ValueError(f"Initial {instance.name} overlaps {name}")
            clearances.append(dict(body=instance.name, obstacle=name, aabb_gap=gap))
        obstacles.append((instance.name, bounds))
    return ResolvedScene(static, dynamic, layout, clearances)
