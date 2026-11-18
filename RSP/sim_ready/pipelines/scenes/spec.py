"""CPU scene descriptions: immutable asset references and per-instance state."""
from dataclasses import dataclass
from pathlib import Path
import math
import yaml
from ...catalog import AssetRef
from ...paths import PROJECT_ROOT, SCENE_ROOT


def vector(value, count, name):
    if not isinstance(value, (list, tuple)) or len(value) != count or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        raise ValueError(f"{name} requires {count} finite numbers")
    return tuple(float(v) for v in value)


@dataclass(frozen=True)
class InstanceSpec:
    name: str
    asset: AssetRef
    position: tuple
    color: tuple
    rotation: tuple = (0., 0., 0., 1.)
    linear_velocity: tuple = (0., 0., 0.)
    angular_velocity: tuple = (0., 0., 0.)
    role: str | None = None


@dataclass(frozen=True)
class SceneSpec:
    container: AssetRef
    character: AssetRef
    instances: tuple[InstanceSpec, ...]
    spawn_xy: tuple
    policy_use_scene_sphere: bool
    fluid_start: tuple = (1.56, .75, .5)
    fluid_end: tuple = (2.96, 1.9, 3.1)

    @property
    def props(self):
        return tuple(instance.asset for instance in self.instances)


def load_scene(path=SCENE_ROOT / "pbf_play.yaml"):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return parse_scene(yaml.safe_load(path.read_text(encoding="utf-8")))


def parse_scene(record):
    """Parse the same scene schema from a preset or a launcher-generated mapping."""
    required = {"schema", "container", "character", "instances", "spawn_xy", "policy_use_scene_sphere"}
    if not isinstance(record, dict) or not required <= record.keys() or record.keys() - required - {"fluid"} or record["schema"] != 1:
        raise ValueError("Invalid scene fields or schema")
    instances, names, roles = [], {"character"}, set()
    for entry in record["instances"]:
        required_instance = {"name", "asset", "position", "color"}
        if not required_instance <= entry.keys() or entry.keys() - required_instance - {"rotation", "linear_velocity", "angular_velocity", "role"}:
            raise ValueError("Invalid instance fields")
        name = entry["name"]
        if not isinstance(name, str) or not name or name in names or any(c in name for c in "/\\:"):
            raise ValueError("Duplicate, reserved or invalid instance name")
        names.add(name)
        ref = AssetRef(**entry["asset"])
        if ref.category != "props":
            raise ValueError("Scene instances must reference props")
        rotation = vector(entry.get("rotation", [0, 0, 0, 1]), 4, "rotation")
        if not math.isclose(sum(v*v for v in rotation), 1., abs_tol=1e-6):
            raise ValueError("rotation must be a unit XYZW quaternion")
        color = vector(entry["color"], 3, "color")
        if any(v < 0 or v > 1 for v in color):
            raise ValueError("color must be in [0, 1]")
        role = entry.get("role")
        if role is not None:
            if role != "policy_sphere" or role in roles:
                raise ValueError("Unknown or duplicate instance role")
            roles.add(role)
        instances.append(InstanceSpec(name, ref, vector(entry["position"], 3, "position"), color, rotation,
            vector(entry.get("linear_velocity", [0, 0, 0]), 3, "linear_velocity"),
            vector(entry.get("angular_velocity", [0, 0, 0]), 3, "angular_velocity"), role))
    container, character = AssetRef(**record["container"]), AssetRef(**record["character"])
    if container.category != "containers" or (character.category, character.name) != ("articulations", "humanoid"):
        raise ValueError("Expected a container and the supported humanoid")
    enabled = record["policy_use_scene_sphere"]
    if not isinstance(enabled, bool) or (enabled and "policy_sphere" not in roles):
        raise ValueError("Scene sphere observations require a policy_sphere instance")
    fluid = record.get("fluid", {"start": [1.56, .75, .5], "end": [2.96, 1.9, 3.1]})
    if set(fluid) != {"start", "end"}:
        raise ValueError("fluid requires start and end")
    start, end = vector(fluid["start"], 3, "fluid.start"), vector(fluid["end"], 3, "fluid.end")
    if any(a < 0 or b <= a for a, b in zip(start, end)):
        raise ValueError("Invalid initial fluid region")
    return SceneSpec(container, character, tuple(instances), vector(record["spawn_xy"], 2, "spawn_xy"), enabled, start, end)
