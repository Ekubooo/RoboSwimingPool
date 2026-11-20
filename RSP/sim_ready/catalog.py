"""Discover source assets and read verified generated assets."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import yaml

from .paths import G1_SOURCE_ROOT, ORIGIN_ROOT, RUNTIME_ROOT, contained

CATEGORIES = ("articulations", "props", "containers", "environments")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


STAIRS_REF = AssetRef("containers", "pbf_stairs")
FULL_STAIRS_REF = AssetRef("containers", "pbf_stairs_full")
_REFERENCE_POOLS = {"pbf_pool": "reference_pool", "pbf_pool_wide": "wide_pool"}
_BUILTIN_CONTAINER_NAMES = {STAIRS_REF.name, FULL_STAIRS_REF.name, *_REFERENCE_POOLS}


def terrain_reference(name):
    """Validate an authored terrain name in the container namespace."""
    ref = AssetRef("containers", name)
    stem = name.split(".", 1)[0].rstrip(" ").upper()
    if (stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
            or (len(stem) == 4 and stem[:3] in {"COM", "LPT"} and stem[3] in "123456789¹²³")):
        raise ValueError(f"Reserved device name: {name}")
    if name.casefold() in _BUILTIN_CONTAINER_NAMES:
        raise ValueError(f"Reserved built-in container name: {name}")
    return ref


def _g1_source_files():
    """Resolve only dependencies of the repository's unchanged g1.xml."""
    root = ET.parse(G1_SOURCE_ROOT / "g1.xml").getroot()
    names = {"g1.xml", "LICENSE.txt"}
    names.update(element.attrib["file"] for element in root.iter() if "file" in element.attrib)
    files = {name: contained(G1_SOURCE_ROOT, name) for name in sorted(names)}
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing G1 source dependency: {path}")
    return files


@dataclass(frozen=True)
class SourceAsset:
    ref: AssetRef
    record: dict
    origin: Path

    @property
    def recipe(self):
        return self.record["recipe"]

    @property
    def is_terrain(self):
        return self.ref.category == "containers" and self.recipe in {"stairs", "procedural_terrain"}

    @property
    def terrain_recipe(self):
        """Procedural settings, or None for the legacy local stairs."""
        if self.recipe != "procedural_terrain":
            return None
        from .terrain import TerrainRecipe

        return TerrainRecipe.from_dict(self.record["terrain_recipe"])

    def input_files(self):
        files = {name: contained(self.origin, name) for name in self.record["inputs"]}
        for path in files.values():
            if not path.is_file():
                raise ValueError(f"Missing asset input: {path}")
        if self.recipe == "mimickit_g1":
            return _g1_source_files()
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
        if self.recipe == "obj":
            return "mesh"
        if self.recipe == "reference_prop":
            return PROPS[self.ref.name].kind
        return None

    @property
    def prop_size(self):
        from .source_geometry import OBJ_SIZE, PROPS
        if self.recipe == "obj":
            return OBJ_SIZE
        if self.recipe == "reference_prop":
            return PROPS[self.ref.name].size
        return None

    def prop_mesh(self):
        from .source_geometry import normalized_obj, prop_mesh
        if self.recipe == "obj":
            return normalized_obj(self)
        return prop_mesh(self.prop_kind, self.prop_size)

    @property
    def prop_bounds(self):
        import numpy as np
        kind = self.prop_kind
        if kind == "mesh":
            low, high = self.prop_mesh().bounds
            return (low + high) / 2, (high - low) / 2
        if kind not in {"box", "sphere"}:
            raise ValueError(f"Not a prop: {self.ref.relative}")
        extent = self.prop_size / 2 if kind == "box" else self.prop_size
        return np.zeros(3, dtype=np.float32), np.full(3, extent, dtype=np.float32)

    @property
    def container_size(self):
        if self.is_terrain:
            from .reference import POOL_SIZE

            return POOL_SIZE
        from .source_geometry import CONTAINER_SIZES

        return CONTAINER_SIZES.get(self.recipe)

    def bunny_mesh(self):
        if self.recipe not in {"reference_pool", "wide_pool"}:
            raise ValueError(f"Not a reference container: {self.ref.relative}")
        from .source_geometry import load_bunny
        path, = self.input_files().values()
        return load_bunny(path)


@dataclass(frozen=True)
class SourceIssue:
    path: Path | str
    message: str
    ref: AssetRef | None = None

    def __str__(self):
        return f"{self.path}: {self.message}"


class SourceCatalog(dict[AssetRef, SourceAsset]):
    """Discovered assets alongside per-input problems for the caller to report."""

    def __init__(self):
        super().__init__()
        self.errors: list[SourceIssue] = []


def _source(origin, ref, recipe, inputs=(), **metadata):
    return SourceAsset(ref, dict(schema=1, identity=asdict(ref), recipe=recipe, inputs=list(inputs),
                                 **metadata), origin)


def _articulation_source(path, origin):
    record = yaml.safe_load(contained(origin, path.relative_to(origin)).read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or set(record) != {"schema", "identity", "recipe", "inputs"}
            or record["schema"] != 1 or not isinstance(record["inputs"], list)
            or any(not isinstance(name, str) or not name for name in record["inputs"])
            or record["recipe"] not in ("mjcf", "mimickit_g1")):
        raise ValueError("Invalid articulation source manifest")
    ref = AssetRef.from_mapping(record["identity"])
    if ref.category != "articulations":
        raise ValueError("Articulation declaration must name an articulation")
    for name in record["inputs"]:
        contained(origin, name)
    return _source(origin, ref, record["recipe"], record["inputs"])


def read_terrain_source(path, origin):
    """Read a named terrain definition into the existing container catalog."""
    from .terrain import TerrainRecipe

    ref = terrain_reference(path.parent.name)
    relative = path.relative_to(origin)
    resolved = contained(origin, relative)
    if relative.parts != ("terrains", ref.name, "terrain.yaml") or resolved != path:
        raise ValueError("Terrain definitions must use physical paths inside origin/terrains")
    data = path.read_bytes()
    record = yaml.safe_load(data)
    if (not isinstance(record, dict) or record.keys() != {"schema", "kind", "parameters"}
            or type(record["schema"]) is not int or record["schema"] != 1):
        raise ValueError("Invalid terrain definition schema")
    recipe = TerrainRecipe.from_dict({key: record[key] for key in ("kind", "parameters")})
    return _source(origin, ref, "procedural_terrain", (relative.as_posix(),), terrain_recipe=recipe.to_dict(),
                   terrain_input_digest=hashlib.sha256(data).hexdigest())


def scan_sources(origin=ORIGIN_ROOT):
    """Discover source assets without preparing runtime data."""
    from .source_geometry import PROPS
    from .terrain import TerrainRecipe

    origin = Path(origin).resolve()
    sources = SourceCatalog()
    for name in PROPS:
        ref = AssetRef("props", name)
        sources[ref] = _source(origin, ref, "reference_prop")
    sources[STAIRS_REF] = _source(origin, STAIRS_REF, "stairs")
    sources[FULL_STAIRS_REF] = _source(origin, FULL_STAIRS_REF, "procedural_terrain",
                                      terrain_recipe=TerrainRecipe.default().to_dict())
    bunny = "props/bunny_flat_bottom.obj"
    for name, recipe in _REFERENCE_POOLS.items():
        ref = AssetRef("containers", name)
        try:
            if not contained(origin, bunny).is_file():
                raise ValueError("Missing reference pool OBJ")
            sources[ref] = _source(origin, ref, recipe, (bunny,))
        except (OSError, ValueError) as exc:
            sources.errors.append(SourceIssue(origin / bunny, str(exc), ref))

    reserved = {ref.relative.as_posix().casefold() for ref in sources}
    reserved.update(f"containers/{name}" for name in _BUILTIN_CONTAINER_NAMES)
    discovered = {}

    def collect(path, source):
        key = source.ref.relative.as_posix().casefold()
        discovered.setdefault(key, []).append((path, source))

    props = origin / "props"
    for path in sorted(props.rglob("*"), key=lambda item: (item.as_posix().casefold(), item)):
        if path.suffix.casefold() != ".obj":
            continue
        ref = None
        try:
            ref = AssetRef("props", path.stem)
            relative = path.relative_to(origin)
            resolved = contained(origin, relative)
            if not resolved.is_relative_to(props) or not resolved.is_file():
                raise ValueError("Expected an OBJ file inside the props directory")
            collect(path, _source(origin, ref, "obj", (relative.as_posix(),)))
        except (OSError, ValueError) as exc:
            sources.errors.append(SourceIssue(path, str(exc), ref))

    for path in sorted((origin / "articulations").rglob("source.yaml")):
        try:
            collect(path, _articulation_source(path, origin))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            sources.errors.append(SourceIssue(path, str(exc)))

    for path in sorted((origin / "terrains").glob("*/terrain.yaml")):
        try:
            collect(path, read_terrain_source(path, origin))
        except (OSError, ValueError, yaml.YAMLError) as exc:
            sources.errors.append(SourceIssue(path, str(exc)))

    for key, group in discovered.items():
        if key in reserved or len(group) > 1:
            reason = "reserved built-in name" if key in reserved else "duplicate asset name"
            for path, source in group:
                sources.errors.append(SourceIssue(path, f"{source.ref.relative}: {reason}", source.ref))
        else:
            _, source = group[0]
            sources[source.ref] = source
    return sources


@dataclass(frozen=True)
class PreparedAsset:
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
        if self.source.recipe in {"reference_prop", "obj"}:
            return self.file(self.record["model"])
        path = self.source.model_path
        if path is None:
            raise ValueError(f"Asset has no model: {self.source.ref.relative}")
        return path


class RuntimeAsset(PreparedAsset):
    """A prepared asset with complete, simulation-ready boundary data."""


def read_prepared_asset(path, source, *, verify=True):
    """Read a full asset slot, which may contain only its prepared SDF."""
    from .lib.storage.fingerprints import recipe_digest

    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    signature = record.pop("record_sha256", None)
    state = record.get("state", "ready")
    if (record["schema"] != 1 or state not in ("sdf_ready", "ready")
            or record["identity"] != asdict(source.ref)
            or not isinstance(record["build_digest"], str) or signature != recipe_digest(record)):
        raise ValueError(f"Invalid runtime asset manifest: {path}")
    asset_type = RuntimeAsset if state == "ready" else PreparedAsset
    asset = asset_type(path.parent, record, source)
    if not verify:
        return asset
    for relative, expected in record["files"].items():
        if digest(asset.file(relative)) != expected:
            raise ValueError(f"Runtime asset checksum mismatch: {relative}")
        if relative.endswith("/boundary.json"):
            nested = json.loads(asset.file(relative).read_text(encoding="utf-8"))
            if nested.get("kind") != "boundary" or nested.get("state", "ready") != state:
                raise ValueError(f"Invalid boundary metadata: {relative}")
            for dependency in nested["files"].values():
                target = contained(asset.root, Path(relative).parent / dependency)
                asset.file(target.relative_to(asset.root).as_posix())
            if state == "sdf_ready" and nested["volume_map"] is not None:
                raise ValueError(f"SDF preparation must not include a volume map: {relative}")
            if state == "ready" and not isinstance(nested.get("volume_map"), dict):
                raise ValueError(f"Prepared boundary lacks a volume map: {relative}")
    if source.container_size is not None:
        asset.file(record["boundary"])
        asset.file(record.get("visible_mesh") or record["bunny"])
        if record.get("hulls") is not None:
            asset.file(record["hulls"])
        layout = record["container"]
        if not isinstance(layout, dict):
            raise ValueError(f"Missing container layout: {path}")
        if source.is_terrain:
            from .terrain import TerrainRecipe

            kind = "stairs" if source.terrain_recipe is None else TerrainRecipe.from_dict(layout.get("terrain_recipe")).kind
            if layout.get("terrain") != kind or "terrain_prop" not in layout:
                raise ValueError(f"Invalid prepared terrain layout: {path}")
            prop = layout["terrain_prop"]
            if prop is not None and (AssetRef.from_mapping(prop).category != "props" or not record.get("hulls")):
                raise ValueError(f"Missing prepared terrain OBJ collision geometry: {path}")
    else:
        asset.model_path
        if source.recipe == "obj":
            asset.file(record["mesh"])
            if record.get("hulls") is not None:
                asset.file(record["hulls"])
        if not isinstance(record["physical_signature"], str):
            raise ValueError(f"Missing physical signature: {path}")
        for relative in record["links"].values():
            asset.file(relative)
    return asset


def read_runtime_asset(path, source, *, verify=True):
    asset = read_prepared_asset(path, source, verify=verify)
    if not isinstance(asset, RuntimeAsset):
        raise ValueError(f"Runtime asset still needs a volume map: {path}")
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
