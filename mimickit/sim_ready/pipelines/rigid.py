"""Prepare and bake local-frame rigid solids without selecting a consumer."""

from pathlib import Path

from ..config import BakeOptions


def prepare(description):
    """Create the primitive mesh described by a caller's geometry recipe."""
    import trimesh

    if description["kind"] == "box":
        return trimesh.creation.box(extents=description["size"])
    if description["kind"] == "sphere":
        return trimesh.creation.icosphere(radius=description["radius"],
                                          subdivisions=description["subdivisions"])
    raise ValueError(f"Unsupported primitive geometry: {description['kind']}")


def build(mesh, stem, *, bake_options=BakeOptions(), collision_cache=None, device="cuda:0"):
    """Validate a complete local solid and stage its mesh/SDF and optional hulls.

    Native primitives do not need convex decomposition. Imported meshes can
    also be supplied by callers; no origin shift, scaling or mass integration
    is implicit in this step.
    """
    from .. import validation, sdf, collision
    from ..publish import save_mesh_obj

    stem = Path(stem)
    checked = validation.validate_solid(mesh)
    hulls = () if collision_cache is None else collision.prepare_hulls(mesh, stem.name, collision_cache)
    volume = sdf.bake_rigid(checked, options=bake_options, device=device)
    mesh_path, sdf_path = stem.with_suffix(".obj"), stem.with_suffix(".nvdb")
    save_mesh_obj(mesh, mesh_path, digits=17)
    sdf.save_volume(volume, sdf_path)
    return mesh_path, sdf_path, hulls
