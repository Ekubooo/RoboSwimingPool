"""Published-asset choices and bounded, approximate placement; no baking or GPU work."""
from copy import deepcopy
from dataclasses import asdict
import numpy as np

from RSP.sim_ready.catalog import scan_published
from RSP.simulation.scenes.layout import asset_bounds, container_obstacle_bounds, place_props
from RSP.simulation.scenes.spec import parse_scene

GAP = .02


def asset_label(ref):
    return f"{ref.name}  /  v{ref.version}"


class SceneChoices:
    def __init__(self, template):
        self.template = deepcopy(template)
        self.assets = scan_published()
        self.containers = {asset_label(ref): ref for ref, asset in self.assets.items()
                           if ref.category == "containers" and "container" in asset.record}
        self.props = {}
        for ref, asset in self.assets.items():
            if ref.category == "props" and len(asset.record.get("links", {})) == 1:
                self.props.setdefault(ref.name, {})[asset_label(ref)] = ref
        self._bounds = {}
        self._obstacles = {}

    def bounds(self, ref):
        if ref not in self._bounds:
            self._bounds[ref] = asset_bounds(self.assets[ref])
        return self._bounds[ref]

    def obstacles(self, container):
        if container not in self._obstacles:
            self._obstacles[container] = container_obstacle_bounds(self.assets[container])
        return self._obstacles[container]

    def build(self, container, counts, *, mode="regular", seed=1234, use_sphere=False, character=None):
        """Generate an entire layout or fail without dropping requested instances."""
        if container not in self.containers.values() or mode not in ("regular", "random"):
            raise ValueError("Select an available container and placement mode")
        names = set()
        expanded = []
        for ref, count in counts.items():
            if ref.name in names or ref not in self.props.get(ref.name, {}).values():
                raise ValueError("Select only one published version of each prop")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 3:
                raise ValueError("Enabled props require a quantity of 1–3; disabled props use 0")
            names.add(ref.name)
            expanded.extend((ref, index + 1) for index in range(count))
        record = deepcopy(self.template)
        if character is not None:
            if character.category != "articulations" or character not in self.assets:
                raise ValueError("Select a published humanoid asset")
            record["character"] = asdict(character)
        record["container"] = asdict(container)
        record["instances"] = []
        record["policy_use_scene_sphere"] = use_sphere
        if not expanded:
            parse_scene(record)
            return record
        size = np.array(self.assets[container].record["container"]["size"])
        lower = np.maximum(record["fluid"]["start"], np.zeros(3)) + GAP
        upper = np.minimum(record["fluid"]["end"], size) - GAP
        positions = place_props(
            [self.bounds(ref) for ref, _ in expanded], lower=lower, upper=upper,
            obstacles=[self.obstacles(container)], mode=mode, seed=seed, gap=GAP,
            labels=[ref.name for ref, _ in expanded])
        colors = {item["asset"]["name"]: item["color"] for item in self.template["instances"]}
        for (ref, index), position in zip(expanded, positions, strict=True):
            item = dict(name=f"{ref.name}_{index}", asset=asdict(ref), position=np.asarray(position).tolist(),
                        color=colors.get(ref.name, [.7, .7, .7]), rotation=[0., 0., 0., 1.],
                        linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
            if ref.name == "scene_sphere" and index == 1:
                item["role"] = "policy_sphere"
            record["instances"].append(item)
        parse_scene(record)
        return record
