"""JSON presets for PBFConfig, independent of simulation initialization."""

from dataclasses import asdict, fields
import json
from pathlib import Path

if __package__:
    from .pbf_helper import PBFConfig
else:
    from pbf_helper import PBFConfig


CONFIG_DIR = Path(__file__).resolve().parent / "config"
DEFAULT_CONFIG_NAME = "default_para.json"


def load_config(path):
    """Load JSON overrides of PBFConfig defaults and validate their values."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot load config {path}: {error}") from error
    if not isinstance(data, dict):
        raise ValueError("Config JSON must contain an object")
    config_fields = fields(PBFConfig)
    unknown = data.keys() - {field.name for field in config_fields}
    if unknown:
        raise ValueError(f"Unknown config fields: {', '.join(sorted(unknown))}")
    for field in config_fields:
        if field.name not in data:
            continue
        if isinstance(field.default, tuple) and isinstance(data[field.name], list):
            data[field.name] = tuple(data[field.name])
    return PBFConfig(**data)


def save_config(config, name=DEFAULT_CONFIG_NAME):
    """Export all fields to a filename in the demo's fixed config directory."""
    if (
        not isinstance(name, str) or name in ("", ".", "..")
        or any(character in name for character in "/\\:")
    ):
        raise ValueError("Export name must be a filename without a directory")
    content = json.dumps(asdict(config), indent=2, allow_nan=False) + "\n"
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = CONFIG_DIR / name
    path.write_text(content, encoding="utf-8")
    return path
