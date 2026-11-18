"""Published-asset choices and bounded, approximate placement; no baking or GPU work."""
from copy import deepcopy
from dataclasses import asdict
from itertools import product
import json
import numpy as np
import trimesh

from RSP.sim_ready.catalog import scan_published
from RSP.sim_ready.pipelines.scenes.spec import parse_scene

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
            asset = self.assets[ref]
            path = asset.file(next(iter(asset.record["links"].values())))
            bounds = json.loads(path.read_text(encoding="utf-8"))["bounds"]
            self._bounds[ref] = (np.array(bounds["center"]), np.array(bounds["half"]))
        return self._bounds[ref]

    def obstacles(self, container):
        if container not in self._obstacles:
            asset = self.assets[container]
            mesh = trimesh.load(asset.file(asset.record["bunny"]), force="mesh", process=False)
            self._obstacles[container] = np.asarray(mesh.bounds)
        return self._obstacles[container]

    def build(self, container, counts, *, mode="regular", seed=1234, use_sphere=False, character=None):
        """Generate an entire layout or fail without dropping requested instances."""
        if container not in self.containers.values() or mode not in ("regular", "random"):
            raise ValueError("请选择可用容器及摆放方式")
        names = set()
        expanded = []
        for ref, count in counts.items():
            if ref.name in names or ref not in self.props.get(ref.name, {}).values():
                raise ValueError("每种刚体只能选择一个已发布版本")
            if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= 3:
                raise ValueError("选中种类的数量须为 1–3，未选中为 0")
            names.add(ref.name)
            expanded.extend((ref, index + 1) for index in range(count))
        record = deepcopy(self.template)
        if character is not None:
            if character.category != "articulations" or character not in self.assets:
                raise ValueError("请选择已发布的人形资产")
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
        max_half = np.max([self.bounds(ref)[1] for ref, _ in expanded], axis=0)
        axes = [np.arange(lo + h, hi - h + 1e-9, 2*h + GAP)
                for lo, hi, h in zip(lower, upper, max_half)]
        grid = iter(product(*axes))
        occupied = [self.obstacles(container)]
        rng = np.random.default_rng(seed)
        colors = {item["asset"]["name"]: item["color"] for item in self.template["instances"]}
        for ref, index in expanded:
            local_center, half = self.bounds(ref)
            if np.any(upper - lower < 2*half):
                raise ValueError(f"{ref.name} 无法放入当前区域")
            candidates = (rng.uniform(lower + half, upper - half) for _ in range(200)) if mode == "random" else grid
            for center in candidates:
                center = np.asarray(center)
                bounds = np.array([center-half, center+half])
                if any(not np.any((bounds[1] + GAP <= other[0]) | (other[1] + GAP <= bounds[0]))
                       for other in occupied):
                    continue
                occupied.append(bounds)
                item = dict(name=f"{ref.name}_{index}", asset=asdict(ref),
                            position=(center-local_center).tolist(), color=colors.get(ref.name, [.7, .7, .7]),
                            rotation=[0., 0., 0., 1.], linear_velocity=[0., 0., 0.], angular_velocity=[0., 0., 0.])
                if ref.name == "scene_sphere" and index == 1:
                    item["role"] = "policy_sphere"
                record["instances"].append(item)
                break
            else:
                raise ValueError("当前摆放空间不足，请减少数量、更换容器或重新随机")
        parse_scene(record)
        return record
