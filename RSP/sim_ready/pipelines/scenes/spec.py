"""Authored scene selections; v1 deliberately accepts the reference topology only."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml

from ...catalog import AssetRef
from ...paths import PROJECT_ROOT, SCENE_ROOT


@dataclass(frozen=True)
class SceneSpec:
    container: AssetRef
    character: AssetRef
    props: tuple[AssetRef, ...]
    spawn_xy: tuple[float, float]
    policy_use_scene_sphere: bool


def load_scene(path=SCENE_ROOT / "pbf_play.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    record = yaml.safe_load(path.read_text(encoding="utf-8"))
    if set(record) != {"schema", "container", "character", "instances", "spawn_xy", "policy_use_scene_sphere"} or record["schema"] != 1:
        raise ValueError("Invalid scene fields or schema")
    from .pbf_play import RIGIDS
    instances = record["instances"]
    names = [entry["name"] for entry in instances]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate scene instance names")
    if names != [spec.name for spec in RIGIDS]:
        raise ValueError("Scene v1 requires the current three rigid instances in their original order")
    props = []
    for entry, spec in zip(instances, RIGIDS):
        if set(entry) != {"name", "asset", "position", "color"}:
            raise ValueError("Invalid instance fields")
        for field, expected in (("position", spec.center), ("color", spec.color)):
            values = entry[field]
            if len(values) != 3 or not all(math.isfinite(v) for v in values) or tuple(values) != expected:
                raise ValueError(f"Scene v1 preserves reference {field}: {spec.name}")
        ref = AssetRef(**entry["asset"])
        if ref.category != "props" or ref.name != spec.name:
            raise ValueError("Unsupported rigid asset combination")
        props.append(ref)
    container, character = AssetRef(**record["container"]), AssetRef(**record["character"])
    if (container.category, container.name, character.category, character.name) != ("containers", "pbf_pool", "articulations", "humanoid"):
        raise ValueError("Scene v1 supports the reference pool and humanoid")
    spawn = tuple(record["spawn_xy"])
    if len(spawn) != 2 or not all(math.isfinite(v) for v in spawn):
        raise ValueError("spawn_xy requires two finite coordinates")
    if not isinstance(record["policy_use_scene_sphere"], bool):
        raise ValueError("policy_use_scene_sphere must be boolean")
    return SceneSpec(container, character, tuple(props), spawn, record["policy_use_scene_sphere"])
