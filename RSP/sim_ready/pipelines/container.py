"""Bake and stage the generated resources for a procedural pool."""

import json

from ..config import BakeOptions
from ..container_geometry import PLATFORM_MODEL_FRACTION, describe_terrain, validate_terrain
from ..lib.mesh.convex import measure_hull_error
from ..lib.storage.fingerprints import recipe_digest
from ..lib.storage.hull_io import save_hulls
from ..lib.storage.mesh_io import save_mesh_obj


def prepare_terrain(store, device, support_radius, *, recipe=None, terrain_asset=None, previous=None):
    """Stage the visual, native collision and dual-grid fluid boundary data."""
    from ..boundary.schema import BoundaryArtifact
    from .boundary import prepare_boundary_asset

    terrain = describe_terrain(recipe=recipe, terrain_asset=terrain_asset)
    solid, checked = validate_terrain(terrain)
    old = None if previous is None else BoundaryArtifact(previous.file(previous.record["boundary"]))
    provenance = dict(terrain=terrain.layout["terrain"], terrain_prop=terrain.layout["terrain_prop"],
                      model_fraction=PLATFORM_MODEL_FRACTION)
    if recipe is not None:
        provenance["terrain_recipe"] = recipe.to_dict()
    artifact = prepare_boundary_asset(
        solid, checked, store=store, device=device, support_radius=support_radius, container=True,
        bake_options=BakeOptions(voxel_size=.0075, max_support_radius=support_radius),
        provenance=provenance,
        previous=old,
    )
    save_mesh_obj(terrain.visible_mesh, store.path("landscape.obj"), digits=17)
    metadata = dict(boundary=artifact.manifest_path.relative_to(store.root).as_posix(),
                    visible_mesh="landscape.obj", container=terrain.layout)
    if terrain.hulls:
        relative = "collision/hulls.npz"
        stored = terrain_asset.record
        files = stored["files"]
        source = dict(mesh=files[stored["mesh"]], hulls=files[stored["hulls"]] if "hulls" in stored else None)
        identity = dict(source=source, transform=terrain.model_transform.tolist())
        fingerprint = recipe_digest(identity)
        metrics = dict(identity, fingerprint=fingerprint, parts=len(terrain.hulls),
                       **measure_hull_error(terrain.model, terrain.hulls))
        store.directory("collision")
        save_hulls(terrain.hulls, store.path(relative))
        store.path("collision", "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        metadata.update(hulls=relative, hulls_fingerprint=fingerprint)
    return metadata
