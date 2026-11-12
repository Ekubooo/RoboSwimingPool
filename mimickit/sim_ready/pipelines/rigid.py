"""Prepare and bake local-frame rigid solids without selecting a consumer."""

from ..lib.storage.store import AssetStore

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


def build(mesh, stem, *, bake_options=BakeOptions(), store: AssetStore, collision=False, device="cuda:0"):
    """Validate a complete local solid and stage its mesh/SDF and optional hulls.

    Native primitives do not need convex decomposition. Imported meshes can
    also be supplied by callers; no origin shift, scaling or mass integration
    is implicit in this step.
    """
    from ..lib.mesh import validation
    from ..lib.volume import sdf
    from .collision import prepare_hulls
    from ..lib.storage.nvdb_io import save_volume
    from ..lib.storage.mesh_io import save_mesh_obj

    stem = store.path(stem)
    relative = stem.relative_to(store.root)
    mesh_path = store.path(relative.with_suffix(".obj"))
    sdf_path = store.path(relative.with_suffix(".nvdb"))
    checked = validation.validate_solid(mesh)
    hulls = prepare_hulls(mesh, stem.name, store) if collision else ()
    volume = sdf.bake_rigid(checked, options=bake_options, device=device)
    save_mesh_obj(mesh, mesh_path, digits=17)
    save_volume(volume, sdf_path)
    return mesh_path, sdf_path, hulls
