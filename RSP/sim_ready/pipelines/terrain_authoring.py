"""Save confirmed terrain definitions in the authored source catalog."""

from contextlib import suppress
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import yaml

from ..catalog import read_terrain_source, terrain_reference
from ..paths import ORIGIN_ROOT
from ..terrain import TerrainRecipe
from .container import describe_terrain, validate_terrain


def save_terrain(name, recipe, *, origin=ORIGIN_ROOT):
    """Publish a validated recipe under a new name without writing runtime assets."""
    ref = terrain_reference(name)
    source_dir = Path(origin).resolve()
    terrains = source_dir / "terrains"
    if terrains.resolve() != terrains:
        raise ValueError("Terrain definitions require a physical origin/terrains directory")
    directory = terrains / ref.name

    def name_conflicts():
        return (path for path in terrains.iterdir() if path.name.casefold() == ref.name.casefold())

    if terrains.exists() and any(name_conflicts()):
        raise ValueError(f"Terrain name is already in use: {name}")
    recipe = TerrainRecipe.from_dict(recipe.to_dict())
    validate_terrain(describe_terrain(recipe=recipe))
    content = yaml.safe_dump({"schema": 1, **recipe.to_dict()}, sort_keys=False, allow_unicode=True)
    terrains.mkdir(parents=True, exist_ok=True)
    try:
        directory.mkdir()
    except FileExistsError as error:
        raise ValueError(f"Terrain name is already in use: {name}") from error

    path = directory / "terrain.yaml"
    temporary_path = None
    try:
        if any(other != directory for other in name_conflicts()):
            raise ValueError(f"Terrain name is already in use: {name}")
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                prefix=".terrain-", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(content)
        os.replace(temporary_path, path)
    except BaseException:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        with suppress(OSError):
            directory.rmdir()
        raise
    return read_terrain_source(path, source_dir)
