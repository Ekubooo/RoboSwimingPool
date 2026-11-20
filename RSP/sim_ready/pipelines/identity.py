"""Content identities for the recipes and numerical stages that produce assets."""

from hashlib import sha256
from functools import cache
from importlib.metadata import version
from pathlib import Path

from ..lib.storage.fingerprints import recipe_digest


_ROOT = Path(__file__).resolve().parents[1]
_MODEL = ("adapters/newton.py", "lib/asset_validation.py")
_ARTICULATION = (*_MODEL, "pipelines/articulation.py", "lib/mesh/shapes.py")
_SOURCE_GEOMETRY = ("source_geometry.py", "reference.py", "lib/mesh/geometry.py")
_WARP = ("warp-lang", "numpy")
_GROUPS = {
    # This shared path coordinates all recipes. Discovery and authoring do
    # not produce runtime files; their resulting source records are hashed.
    "shared": (("pipelines/runtime.py", "pipelines/boundary.py", "config.py",
                "boundary/schema.py", "lib/storage/fingerprints.py", "lib/storage/mesh_io.py",
                "lib/storage/nvdb_io.py", "lib/mesh/validation.py"), ("newton", "trimesh")),
    "sdf": (("lib/volume/sdf.py",), _WARP),
    "volume_map": (("lib/volume/pbf_volume_map.py",), _WARP),
    "hulls": (("lib/mesh/convex.py", "pipelines/collision.py", "lib/storage/hull_io.py"),
              ("coacd", "trimesh", "numpy")),
    "obj": ((*_SOURCE_GEOMETRY, *_MODEL, "pipelines/rigid.py"), ("manifold3d",)),
    "mjcf": ((*_ARTICULATION, "lib/mesh/geometry.py", "lib/mesh/convex.py"), ("manifold3d",)),
    "mimickit_g1": ((*_ARTICULATION, "lib/mesh/geometry.py", "pipelines/recipes/g1.py"), ("manifold3d",)),
    "reference_prop": ((*_ARTICULATION, "source_geometry.py", "pipelines/recipes/pbf_play.py"), ()),
    "reference_pool": ((*_SOURCE_GEOMETRY, "pipelines/recipes/container.py", "pipelines/recipes/pbf_play.py",
                        "lib/asset_validation.py"), ("manifold3d",)),
    "terrain": ((*_SOURCE_GEOMETRY, "container_geometry.py", "terrain.py", "adapters/terrain.py",
                  "pipelines/container.py", "lib/mesh/convex.py", "lib/storage/hull_io.py"), ("manifold3d",)),
}


def _code_digest(relative):
    source = (_ROOT / relative).read_text(encoding="utf-8")
    return sha256(source.encode("utf-8")).hexdigest()


def _group_identity(group, code, dependency):
    files, packages = _GROUPS[group]
    return recipe_digest(dict(code={path: code(path) for path in files},
                              dependencies={name: dependency(name) for name in packages}))


def stage_identity(group):
    """Identify one stage before considering its cached output."""
    return _group_identity(group, _code_digest, version)


def producer_snapshot():
    """Read each shared producer and dependency once for a preparation pass."""
    code = cache(_code_digest)
    dependency = cache(version)
    return {group: _group_identity(group, code, dependency) for group in _GROUPS}


def asset_producer(source, snapshot):
    recipe = source.recipe
    if source.is_terrain:
        recipe = "terrain"
    elif recipe == "wide_pool":
        recipe = "reference_pool"
    elif recipe not in _GROUPS:
        raise ValueError(f"Unsupported asset recipe: {recipe}")
    groups = ("shared", "sdf", recipe)
    if recipe in ("obj", "reference_pool"):
        groups += ("hulls",)
    return {group: snapshot[group] for group in groups}
