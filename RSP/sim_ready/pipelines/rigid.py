"""Prepare rigid bodies from source geometry and preserve their local frame."""

import xml.etree.ElementTree as ET

from ..lib.storage.store import AssetStore

from ..config import BakeOptions


def _write_obj_model(path, mesh, hull_files, name):
    """Use source inertia and place each rigid collision proxy on the same body."""
    import numpy as np
    from ..source_geometry import OBJ_DENSITY

    properties = mesh.mass_properties
    mass = properties.mass * OBJ_DENSITY
    center = properties.center_mass
    inertia = properties.inertia * OBJ_DENSITY
    if not (np.isfinite(mass) and mass > 0 and np.isfinite(center).all()
            and np.isfinite(inertia).all() and np.all(np.linalg.eigvalsh(inertia) > 0)):
        raise ValueError("OBJ must have a positive mass and valid inertia")

    def vector(values):
        return " ".join(map(str, values))

    model = ET.Element("mujoco", model=name)
    ET.SubElement(model, "compiler", inertiafromgeom="false")
    meshes = ET.SubElement(model, "asset")
    ET.SubElement(meshes, "mesh", name="source", file="mesh.obj")
    world = ET.SubElement(model, "worldbody")
    body = ET.SubElement(world, "body", name="rigid")
    ET.SubElement(body, "freejoint", name="root")
    full_inertia = (*inertia.diagonal(), inertia[0, 1], inertia[0, 2], inertia[1, 2])
    ET.SubElement(body, "inertial", mass=str(mass), pos=vector(center), fullinertia=vector(full_inertia))
    geom = dict(type="mesh", friction="1.0 0.05 0.05")
    source_flags = dict(contype="0", conaffinity="0") if hull_files else {}
    ET.SubElement(body, "geom", name="source", mesh="source", **geom, **source_flags)
    for index, relative in enumerate(hull_files):
        name = f"hull_{index}"
        ET.SubElement(meshes, "mesh", name=name, file=relative)
        ET.SubElement(body, "geom", name=name, mesh=name, **geom)
    path.write_text(ET.tostring(model, encoding="unicode") + "\n", encoding="utf-8")


def prepare_obj(source, *, store, device, support_radius, previous=None, sdf_only=False):
    """Prepare the render mesh, rigid collision model and one source-mesh boundary."""
    import newton
    import numpy as np

    from ..adapters.newton import describe_articulations, finalize_model, import_physical_asset
    from ..boundary.schema import BoundaryArtifact
    from ..lib.asset_validation import ValidationReport, check_source, diagnostic_scope, validate_physical_asset
    from ..lib.mesh import validation
    from ..lib.storage.fingerprints import physical_signature
    from ..lib.storage.mesh_io import save_mesh_obj
    from ..source_geometry import normalized_obj
    from .boundary import prepare_boundary_asset
    from .collision import hulls_fingerprint, prepare_hulls

    path, = source.input_files().values()
    report = ValidationReport()
    check_source(path, {".obj"}, report)
    with diagnostic_scope(report, "geometry.rigid", path):
        mesh = normalized_obj(source)
        checked = validation.validate_solid(mesh)
    save_mesh_obj(mesh, store.path("mesh.obj"), digits=17)

    hulls = ()
    metadata = {}
    if not mesh.is_convex:
        with diagnostic_scope(report, "geometry.convex", path):
            hulls = prepare_hulls(mesh, source.ref.name, store, destination="collision", previous=previous)
        metadata.update(hulls="collision/hulls.npz", hulls_fingerprint=hulls_fingerprint(mesh))
    hull_files = []
    for index, hull in enumerate(hulls):
        relative = f"collision/hull_{index}.obj"
        save_mesh_obj(hull, store.path(relative), digits=17)
        hull_files.append(relative)

    model_path = store.path("model.xml")
    with diagnostic_scope(report, "geometry.inertia", path):
        _write_obj_model(model_path, mesh, hull_files, source.ref.name)
    builder = newton.ModelBuilder()
    newton.solvers.SolverMuJoCo.register_custom_attributes(builder)
    builder.default_shape_cfg.mu = 1.0
    import_physical_asset(builder, model_path, fix_root=False, enable_self_collisions=True, report=report)
    model = finalize_model(builder, device=device, report=report, require_dfs=True)
    binding, = describe_articulations(model)
    report.extend(validate_physical_asset(binding.asset))
    report.require_valid()
    body, = binding.asset.bodies
    if not body.shapes:
        raise ValueError(f"No rigid collision geometry in prepared OBJ: {path}")

    old = None
    if previous is not None:
        relative = previous.record["links"].get(body.name)
        if relative is not None:
            old = BoundaryArtifact(previous.file(relative))
    with diagnostic_scope(report, "boundary.prepare", path):
        artifact = prepare_boundary_asset(
            mesh, checked, store=store, device=device, support_radius=support_radius,
            bake_options=BakeOptions(voxel_size=.005, max_support_radius=support_radius),
            destination="links/0", previous=old, sdf_only=sdf_only,
            provenance={"source_kind": "obj", "input_up_axis": "Y", "world_up_axis": "Z"},
        )
    bounds = np.asarray([item.bounds for item in (mesh, *hulls)])
    low, high = bounds[:, 0].min(axis=0), bounds[:, 1].max(axis=0)
    return dict(metadata, state="sdf_ready" if sdf_only else "ready", mesh="mesh.obj", model="model.xml",
                links={body.name: artifact.manifest_path.relative_to(store.root).as_posix()},
                physical_signature=physical_signature(binding.asset),
                layout_bounds=dict(center=((low + high) / 2).tolist(), half=((high - low) / 2).tolist()))


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
