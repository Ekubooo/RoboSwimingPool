"""CPU-only discovery of complete, versioned asset packages."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path

from .paths import PUBLISHED_ROOT

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
        raise ValueError(f"Asset path escapes package: {relative}")
    return result


@dataclass(frozen=True)
class AssetRef:
    category: str
    name: str
    version: str

    def __post_init__(self):
        if self.category not in CATEGORIES:
            raise ValueError(f"Unknown asset category: {self.category}")
        for value in (self.name, self.version):
            if not isinstance(value, str) or not value or value in (".", "..") or any(c in value for c in '/\\:'):
                raise ValueError("Asset names and versions must be single path components")

    @property
    def relative(self):
        return Path(self.category) / self.name / self.version


@dataclass(frozen=True)
class PublishedAsset:
    root: Path
    record: dict

    def file(self, relative):
        path = contained(self.root, relative)
        if relative not in self.record["files"] or not path.is_file():
            raise ValueError(f"Unpublished or missing asset file: {path}")
        return path


def read_asset(path, *, verify=True):
    path = Path(path).resolve()
    record = json.loads(path.read_text(encoding="utf-8"))
    if record["schema"] != 1:
        raise ValueError(f"Unsupported asset schema: {path}")
    AssetRef(**record["identity"])
    asset = PublishedAsset(path.parent, record)
    if not verify:
        return asset
    for relative, expected in record["files"].items():
        if digest(asset.file(relative)) != expected:
            raise ValueError(f"Published asset checksum mismatch: {relative}")
        if relative.endswith(".json"):
            nested = json.loads(asset.file(relative).read_text(encoding="utf-8"))
            if nested.get("kind") in ("sdf", "boundary"):
                for dependency in nested["files"].values():
                    target = contained(asset.root, Path(relative).parent / dependency)
                    asset.file(target.relative_to(asset.root).as_posix())
    return asset


def scan_published(root=PUBLISHED_ROOT):
    """Browse manifests only; resolve_asset performs full verification before use."""
    root = Path(root).resolve()
    result = {}
    for category in CATEGORIES:
        for path in sorted((root / category).glob("*/*/asset.json")):
            asset = read_asset(path, verify=False)
            ref = AssetRef(**asset.record["identity"])
            if path.parent != root / ref.relative or ref in result:
                raise ValueError(f"Duplicate or misplaced asset identity: {path}")
            result[ref] = asset
    return result


def resolve_asset(ref, root=PUBLISHED_ROOT):
    path = contained(root, ref.relative) / "asset.json"
    if not path.is_file():
        raise ValueError(f"Asset not published: {ref.relative}. Run tools/prepare_rsp_assets.py first.")
    asset = read_asset(path)
    if AssetRef(**asset.record["identity"]) != ref:
        raise ValueError(f"Asset identity mismatch: {path}")
    return asset


class AssetCatalog:
    """One assembly's verified resources; no process-wide or per-frame cache."""
    def __init__(self, root=PUBLISHED_ROOT):
        self.root = Path(root).resolve()
        self._assets = {}

    def resolve(self, ref):
        if ref not in self._assets:
            self._assets[ref] = resolve_asset(ref, self.root)
        return self._assets[ref]
