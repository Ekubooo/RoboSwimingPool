"""Discover source declarations and read verified, current runtime assets."""
from dataclasses import dataclass, asdict
import hashlib
import json
import os
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .paths import ORIGIN_ROOT, RUNTIME_ROOT

CATEGORIES = ("articulations", "props", "containers", "environments")


def physical_signature(asset):
    """Compare local geometry/topology, independent of scene label prefixes."""
    def plain(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {k: (v.split("/")[-1] if k == "name" else plain(v)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value
    record = plain(asdict(asset))
    record.pop("name")
    return hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contained(root, relative):
    root = Path(root).resolve()
    path = Path(relative)
    if path.is_absolute() or path.drive:
        raise ValueError(f"Expected relative asset path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"Asset path escapes its directory: {relative}")
    return result


def _name_component(value):
    return (isinstance(value, str) and bool(value) and value not in {".", ".."}
            and not value.endswith((".", " "))
            and not any(c in '/\\:*?"<>|' or ord(c) < 32 for c in value))


@dataclass(frozen=True)
class AssetRef:
    category: str
    name: str

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown asset category: {self.category}")
        if not _name_component(self.name):
            raise ValueError("Asset names must be single path components")

    @classmethod
    def from_mapping(cls, record):
        """Read the current reference or an old selection of the same asset."""
        if not isinstance(record, dict) or set(record) not in ({"category", "name"}, {"category", "name", "version"}):
            raise ValueError("Expected an asset category and name")
        if "version" in record:
            version = record["version"]
            if not _name_component(version):
                raise ValueError("Invalid legacy asset version")
        return cls(record["category"], record["name"])

    @property
    def relative(self):
        return Path(self.category) / self.name


@dataclass(frozen=True)
class SourceAsset:
    ref: AssetRef
    record: dict
    origin: Path

    @property
    def recipe(self):
        return self.record["recipe"]

    def input_files(self):
        files = {name: contained(self.origin, name) for name in self.record["inputs"]}
        for path in files.values():
            if not path.is_file():
                raise ValueError(f"Missing asset input: {path}")
        if self.recipe == "mimickit_g1":
            from .pipelines.recipes.g1 import source_files
            return source_files()
        return files

    @property
    def model_path(self):
        files = self.input_files()
        if self.recipe == "mimickit_g1":
            return files["g1.xml"]
        if self.recipe == "mjcf":
            path, = files.values()
            if any("file" in element.attrib for element in ET.parse(path).iter()):
                raise ValueError("External MJCF dependencies require an explicit source recipe")
            return path
        return None

    @property
    def prop_kind(self):
        from .source_geometry import PROPS
        return PROPS[self.ref.name].kind if self.recipe == "reference_prop" else None

    @property
    def prop_size(self):
        from .source_geometry import PROPS
        return PROPS[self.ref.name].size if self.recipe == "reference_prop" else None

    @property
    def prop_bounds(self):
        import numpy as np
        extent = self.prop_size / 2 if self.prop_kind == "box" else self.prop_size
        return np.zeros(3, dtype=np.float32), np.full(3, extent, dtype=np.float32)

    @property
    def container_size(self):
        from .source_geometry import CONTAINER_SIZES
        return CONTAINER_SIZES.get(self.recipe)

    def bunny_mesh(self):
        if self.container_size is None:
            raise ValueError(f"Not a reference container: {self.ref.relative}")
        from .source_geometry import load_bunny
        path, = self.input_files().values()
        return load_bunny(path)


def scan_sources(origin=ORIGIN_ROOT):
    """Read declarations on CPU. Resolve each asset's inputs when it is used."""
    origin = Path(origin).resolve()
    sources = {}
    paths = set()
    for path in sorted(origin.rglob("source.yaml")):
        record = yaml.safe_load(path.read_text(encoding="utf-8"))
        if (not isinstance(record, dict) or set(record) != {"schema", "identity", "recipe", "inputs"}
                or record["schema"] != 1 or not isinstance(record["inputs"], list)
                or any(not isinstance(name, str) or not name for name in record["inputs"])
                or not isinstance(record["recipe"], str)):
            raise ValueError(f"Invalid source manifest: {path}")
        ref = AssetRef.from_mapping(record["identity"])
        key = os.path.normcase(str(ref.relative))
        if key in paths:
            raise ValueError(f"Duplicate source identity: {ref.relative}")
        paths.add(key)
        for name in record["inputs"]:
            contained(origin, name)
        sources[ref] = SourceAsset(ref, dict(record, identity=asdict(ref)), origin)
    return sources


@dataclass(frozen=True)
class RuntimeAsset:
    root: Path
    record: dict
    source: SourceAsset

    def file(self, relative):
        path = contained(self.root, relative)
        if relative not in self.record["files"] or not path.is_file():
            raise ValueError(f"Unknown or missing runtime asset file: {path}")
        return path

    @property
    def model_path(self):
        if self.source.recipe == "reference_prop":
            return self.file(self.record["model"])
        path = self.source.model_path
        if path is None:
            raise ValueError(f"Asset has no model: {self.source.ref.relative}")
        return path


def read_runtime_asset(path, source, *, verify=True):
    from .lib.storage.fingerprints import recipe_digest

    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("record_sha256", None)
    if (record["schema"] != 1 or record["identity"] != asdict(source.ref)
            or not isinstance(record["build_digest"], str) or signature != recipe_digest(record)):
        raise ValueError(f"Invalid runtime asset manifest: {path}")
    asset = RuntimeAsset(path.parent, record, source)
    if not verify:
        return asset
    for relative, expected in record["files"].items():
        if digest(asset.file(relative)) != expected:
            raise ValueError(f"Runtime asset checksum mismatch: {relative}")
        if relative.endswith("/boundary.json"):
            nested = json.loads(asset.file(relative).read_text(encoding="utf-8"))
            if nested.get("kind") != "boundary":
                raise ValueError(f"Invalid boundary metadata: {relative}")
            for dependency in nested["files"].values():
                target = contained(asset.root, Path(relative).parent / dependency)
                asset.file(target.relative_to(asset.root).as_posix())
    if source.container_size is not None:
        for role in ("boundary", "bunny", "hulls"):
            asset.file(record[role])
        if not isinstance(record["container"], dict):
            raise ValueError(f"Missing container layout: {path}")
    else:
        asset.model_path
        if not isinstance(record["physical_signature"], str):
            raise ValueError(f"Missing physical signature: {path}")
        for relative in record["links"].values():
            asset.file(relative)
    return asset


def resolve_runtime_asset(ref, root=RUNTIME_ROOT, *, sources=None):
    sources = scan_sources() if sources is None else sources
    if ref not in sources:
        raise ValueError(f"Unknown asset: {ref.relative}")
    path = contained(root, ref.relative / "asset.json")
    if not path.is_file():
        raise ValueError(f"Runtime asset is missing: {ref.relative}. Prepare it before opening the scene.")
    return read_runtime_asset(path, sources[ref])


class RuntimeAssetCatalog:
    """Verify each asset once per scene assembly."""
    def __init__(self, root=RUNTIME_ROOT, origin=ORIGIN_ROOT):
        self.root = Path(root).resolve()
        self._sources = scan_sources(origin)
        self._assets = {}

    def resolve(self, ref):
        if ref not in self._assets:
            self._assets[ref] = resolve_runtime_asset(ref, self.root, sources=self._sources)
        return self._assets[ref]
