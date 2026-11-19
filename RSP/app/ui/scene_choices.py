"""Browse source assets and generate layouts without preparing runtime data."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from RSP.sim_ready.catalog import scan_sources
from RSP.simulation.scenes.layout import place_props
from RSP.simulation.scenes.spec import parse_scene

GAP = .02


def asset_label(ref):
    return ref.name


class SceneChoices:
    def __init__(self, template, *, sources=None):
        self.template = deepcopy(template)
        self.template_spec = parse_scene(template)
        self.assets = scan_sources() if sources is None else sources
        self.containers = {asset_label(ref): ref for ref, asset in self.assets.items()
                           if ref.category == "containers" and asset.container_size is not None}
        self.props = {ref.name: ref for ref, asset in self.assets.items()
                      if ref.category == "props" and asset.prop_kind is not None}
        self._bounds = {}
        self._obstacles = {}

    def bounds(self, ref):
        if ref not in self._bounds:
            center, half = self.assets[ref].prop_bounds
            self._bounds[ref] = (np.asarray(center, dtype=float), np.asarray(half, dtype=float))
        return self._bounds[ref]

    def obstacles(self, container):
        if container not in self._obstacles:
            self._obstacles[container] = np.asarray(self.assets[container].bunny_mesh().bounds)
        return self._obstacles[container]

    def build(self, container, counts, *, mode="regular", seed=1234, use_sphere=False, character=None):
        """Generate an entire layout or fail without dropping requested instances."""
        if container not in self.containers.values() or mode not in ("regular", "random"):
            raise ValueError("Select an available container and placement mode")
        expanded = []
        for ref, count in counts.items():
            if self.props.get(ref.name) != ref:
                raise ValueError(f"Unknown prop source: {ref.relative}")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 3:
                raise ValueError("Enabled props require a quantity of 1–3; disabled props use 0")
            expanded.extend((ref, index + 1) for index in range(count))
        record = deepcopy(self.template)
        if character is not None:
            if character.category != "articulations" or character not in self.assets:
                raise ValueError("Select an available humanoid source")
            record["character"] = asdict(character)
        else:
            record["character"] = asdict(self.template_spec.character)
        record["container"] = asdict(container)
        record["instances"] = []
        record["policy_use_scene_sphere"] = use_sphere
        if not expanded:
            parse_scene(record)
            return record
        size = np.asarray(self.assets[container].container_size)
        lower = np.maximum(self.template_spec.fluid_start, np.zeros(3)) + GAP
        upper = np.minimum(self.template_spec.fluid_end, size) - GAP
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
