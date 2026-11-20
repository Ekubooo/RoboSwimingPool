"""Browse source assets and generate layouts without preparing runtime data."""
from copy import deepcopy
from dataclasses import asdict

import numpy as np

from RSP.sim_ready.catalog import scan_sources
from RSP.simulation.scenes.layout import place_props, terrain_fluid_region
from RSP.simulation.scenes.spec import parse_scene

GAP = .02


def asset_label(ref):
    return ref.name


class SceneChoices:
    def __init__(self, template, *, sources=None, prepared_assets=None):
        self.template = deepcopy(template)
        self.template_spec = parse_scene(template)
        self.assets = scan_sources() if sources is None else sources
        self.prepared_assets = {} if prepared_assets is None else prepared_assets
        self.errors = tuple(getattr(self.assets, "errors", ()))
        self.containers = {asset_label(ref): ref for ref, asset in self.assets.items()
                           if ref.category == "containers" and asset.container_size is not None}
        self.props = {ref.name: ref for ref, asset in self.assets.items()
                      if ref.category == "props" and asset.prop_kind is not None}
        self._bounds = {}
        self._obstacles = {}
        self._terrain = {}
        self.terrain_props = {ref.name: ref for ref, asset in self.assets.items() if asset.recipe == "obj"}

    def bounds(self, ref):
        if ref not in self._bounds:
            prepared = self.prepared_assets.get(ref)
            if prepared is not None and "layout_bounds" in prepared.record:
                bounds = prepared.record["layout_bounds"]
                center, half = bounds["center"], bounds["half"]
            else:
                center, half = self.assets[ref].prop_bounds
            self._bounds[ref] = (np.asarray(center, dtype=float), np.asarray(half, dtype=float))
        return self._bounds[ref]

    def add_terrain(self, asset):
        """Register a freshly saved terrain in the already filtered catalog."""
        self.assets[asset.ref] = asset
        self.containers[asset_label(asset.ref)] = asset.ref

    def terrain(self, container, terrain_prop=None):
        key = (container, terrain_prop)
        if key not in self._terrain:
            from RSP.sim_ready.pipelines.container import describe_terrain
            if terrain_prop is not None and terrain_prop not in self.terrain_props.values():
                raise ValueError("Select an available terrain OBJ")
            source = None if terrain_prop is None else self.assets[terrain_prop]
            self._terrain[key] = describe_terrain(source, recipe=self.assets[container].terrain_recipe,
                                                  terrain_asset=self.prepared_assets.get(terrain_prop))
        return self._terrain[key]

    def obstacles(self, container, terrain_prop=None):
        if self.assets[container].is_terrain:
            return self.terrain(container, terrain_prop).obstacle_bounds
        if container not in self._obstacles:
            self._obstacles[container] = np.asarray(self.assets[container].bunny_mesh().bounds)
        return [self._obstacles[container]]

    def build(self, container, counts, *, mode="regular", seed=1234, use_sphere=False, character=None,
              terrain_prop=None):
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
        record.pop("terrain_prop", None)
        start, end = np.array(self.template_spec.fluid_start), np.array(self.template_spec.fluid_end)
        if self.assets[container].is_terrain:
            if terrain_prop is not None:
                record["terrain_prop"] = asdict(terrain_prop)
            start, end = terrain_fluid_region(start, end, self.terrain(container, terrain_prop).layout, gap=GAP)
        elif terrain_prop is not None:
            raise ValueError("A terrain model requires a procedural terrain container")
        record["fluid"] = dict(start=start.tolist(), end=end.tolist())
        if not expanded:
            parse_scene(record)
            return record
        size = np.asarray(self.assets[container].container_size)
        lower = np.maximum(start, np.zeros(3)) + GAP
        upper = np.minimum(end, size) - GAP
        positions = place_props(
            [self.bounds(ref) for ref, _ in expanded], lower=lower, upper=upper,
            obstacles=self.obstacles(container, terrain_prop), mode=mode, seed=seed, gap=GAP,
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
