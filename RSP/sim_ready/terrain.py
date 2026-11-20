"""User-authored parameters for the fixed-size pool terrain generators."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


_PARAMETERS = {
    "pyramid_stairs": {"step_height": (.10, .02, .25)},
    "random_grid": {"max_height": (.15, .02, .50), "seed": (0, 0, 2**31 - 1)},
    "gap": {"gap_width": (.25, .05, .60), "depth": (.15, .02, .50)},
}


@dataclass(frozen=True)
class TerrainRecipe:
    kind: str
    parameters: Mapping[str, float | int]

    def __post_init__(self):
        rules = self._rules(self.kind)
        if not isinstance(self.parameters, Mapping) or self.parameters.keys() != rules.keys():
            raise ValueError(f"{self.kind} requires parameters: {', '.join(rules)}")
        parameters = {}
        for name, (default, low, high) in rules.items():
            value = self.parameters[name]
            types = int if isinstance(default, int) else (int, float)
            if isinstance(value, bool) or not isinstance(value, types) or not low <= value <= high:
                raise ValueError(f"{name} must be {'an integer' if types is int else 'a number'} from {low} to {high}")
            parameters[name] = type(default)(value)
        object.__setattr__(self, "parameters", MappingProxyType(parameters))

    @staticmethod
    def _rules(kind):
        if not isinstance(kind, str) or kind not in _PARAMETERS:
            raise ValueError(f"Unsupported terrain type: {kind}")
        return _PARAMETERS[kind]

    @classmethod
    def default(cls, kind="pyramid_stairs"):
        return cls(kind, {name: rule[0] for name, rule in cls._rules(kind).items()})

    @classmethod
    def parameter_bounds(cls, kind):
        return {name: (rule[1], rule[2]) for name, rule in cls._rules(kind).items()}

    @classmethod
    def from_dict(cls, record):
        if not isinstance(record, Mapping) or record.keys() != {"kind", "parameters"}:
            raise ValueError("A terrain recipe requires kind and parameters")
        return cls(record["kind"], record["parameters"])

    def to_dict(self):
        return {"kind": self.kind, "parameters": dict(self.parameters)}
