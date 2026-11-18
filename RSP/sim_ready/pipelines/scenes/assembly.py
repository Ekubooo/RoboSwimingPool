"""Read-only CPU preflight before environment or GPU state construction."""
from dataclasses import dataclass
import json
import numpy as np
import trimesh

from ...catalog import AssetCatalog
from ...paths import PUBLISHED_ROOT


@dataclass(frozen=True)
class ResolvedScene:
    static: object
    dynamic: dict
    layout: dict
    clearances: list


def resolve_scene(spec, root=PUBLISHED_ROOT):
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
    from RSP.runtime.fluid_clock import DEFAULT_SETTINGS
    from ...config import VOLUME_MAP_ALGORITHM_VERSION
    support = 4.0 * DEFAULT_SETTINGS.pbf.particle_radius
    boundaries = [(static, static.record["boundary"])]
    for asset in dynamic.values():
        asset.file(asset.record["model"])
        boundaries.extend((asset, relative) for relative in asset.record["links"].values())
    for asset, relative in boundaries:
        record = json.loads(asset.file(relative).read_text(encoding="utf-8"))
        settings = record["volume_map"]
        if settings["support_radius"] != support or settings["quadrature_points"] != 16 or settings["algorithm_version"] != VOLUME_MAP_ALGORITHM_VERSION:
            raise ValueError("Published boundary parameters do not match playback")
    obstacles, clearances = [], []
    for instance in spec.instances:
        asset = dynamic[instance.asset]
        if len(asset.record["links"]) != 1:
            raise ValueError("Scene props must contain one rigid body")
        path = asset.file(next(iter(asset.record["links"].values())))
        local = json.loads(path.read_text(encoding="utf-8"))["bounds"]
        q = instance.rotation
        rotation = trimesh.transformations.quaternion_matrix([q[3], *q[:3]])[:3, :3]
        center = np.asarray(instance.position) + rotation @ np.asarray(local["center"])
        half = np.abs(rotation) @ np.asarray(local["half"])
        bounds = np.array([center-half, center+half])
        if np.any(bounds[0] <= 0) or np.any(bounds[1] >= size):
            raise ValueError(f"{instance.name} intersects the container")
        for name, other in obstacles:
            gap = max(float(np.max(bounds[0]-other[1])), float(np.max(other[0]-bounds[1])))
            if gap <= 0:
                raise ValueError(f"Initial {instance.name} overlaps {name}")
            clearances.append(dict(body=instance.name, obstacle=name, aabb_gap=gap))
        obstacles.append((instance.name, bounds))
    return ResolvedScene(static, dynamic, layout, clearances)
