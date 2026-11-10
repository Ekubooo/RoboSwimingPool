"""Temporary assets for the single-world PBF play scene.

The existing producers retain their conventions. Only this scene adapter knows
the pool layout, the Y-up to Z-up conversion, and the publication directory.
"""
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import warp as wp

from .. import asset_io, collision, geometry, sdf, validation
from ..config import BakeOptions
from ..publish import save_mesh_obj


POOL_SIZE = np.array([3.0, 2.0, 5.0])
WALL_THICKNESS = 0.1
Y_TO_Z = np.array([[1., 0., 0.], [0., 0., -1.], [0., 1., 0.]])
SCENE_TRANSFORM = np.eye(4)
SCENE_TRANSFORM[:3, :3] = Y_TO_Z
SCENE_TRANSFORM[:3, 3] = [0., 2., 0.]


@dataclass(frozen=True)
class RigidSpec:
    name: str
    kind: str
    size: float
    density: float
    center: tuple
    color: tuple


RIGIDS = (
    RigidSpec("light_box", "box", .3, 500., (2.55, .45, 1.2), (.90, .55, .15)),
    RigidSpec("heavy_box", "box", .3, 1200., (2.55, 1.55, 1.3), (.70, .35, .15)),
    RigidSpec("scene_sphere", "sphere", .15, 750., (1.75, 1.55, 1.1), (.90, .25, .25)),
)


def primitive_files(root):
    """Use ordinary MJCF objects so MimicKit retains its object bookkeeping."""
    directory = Path(root) / "primitives"
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in RIGIDS:
        document = ET.Element("mujoco", model=spec.name)
        world = ET.SubElement(document, "worldbody")
        body = ET.SubElement(world, "body", name=spec.name)
        ET.SubElement(body, "freejoint", name="root")
        size = " ".join([str(spec.size / 2)] * 3) if spec.kind == "box" else str(spec.size)
        ET.SubElement(body, "geom", name=spec.name, type=spec.kind, size=size,
                      density=str(spec.density), friction="1.0 0.05 0.05")
        path = directory / f"{spec.name}.xml"
        content = ET.tostring(document, encoding="unicode") + "\n"
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            path.write_text(content, encoding="utf-8")
        paths.append(str(path.resolve()))
    return paths


def _box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def static_geometry():
    """One exterior, one cavity and a disjoint positive bunny shell; no union."""
    size_y = np.array([3., 5., 2.])
    outer = _box(size_y + 2 * WALL_THICKNESS, size_y / 2)
    inner = _box(size_y, size_y / 2)
    inner.invert()
    container_y = trimesh.util.concatenate([outer, inner])
    validation.validate_container(container_y, outer.bounds)

    source = Path(__file__).resolve().parents[1] / "Assets/bunny_flat_bottom.obj"
    bunny = geometry.prepare_rigid(source, obj_up_axis="Y")
    parts = validation.validate_rigid_parts(bunny)
    if len(parts) != 1:
        raise ValueError("The play scene expects the original single-shell bunny")
    bunny = parts[0]
    bunny.apply_scale(.6 / bunny.extents[1])
    bunny.apply_translation(np.array([2.15, .5, 1.]) - bunny.bounds.mean(axis=0))
    if np.any(bunny.bounds[0] <= 0) or np.any(bunny.bounds[1] >= size_y):
        raise ValueError("Suspended bunny must be strictly inside the fluid cavity")

    container = container_y.copy()
    container.apply_transform(SCENE_TRANSFORM)
    bunny.apply_transform(SCENE_TRANSFORM)
    container_data = validation.validate_container(container, container.bounds)
    bunny_data = validation.validate_solid(bunny)
    # Independent closed shells are disjoint, so concatenating their oriented
    # triangles preserves the solid/cavity winding without a Boolean operation.
    vertices = np.concatenate([container_data.vertices, bunny_data.vertices])
    faces = np.concatenate([container_data.faces, bunny_data.faces + len(container_data.vertices)])
    combined = trimesh.Trimesh(vertices, faces, process=False)
    checked = validation.ValidatedMesh(vertices, faces, container_data.bounds, container_data.cavity_bounds)
    provenance = dict(source=str(source), source_sha256=asset_io.file_digest(source),
                      input_up_axis="Y", world_up_axis="Z", transform=SCENE_TRANSFORM.tolist(),
                      bunny_height=.6, bunny_center=[2.15, 1., .5],
                      pool_size=POOL_SIZE.tolist(), wall_thickness=WALL_THICKNESS,
                      static_composition="disjoint oriented shells; no Boolean union")
    return combined, checked, bunny, provenance


def publish_boundary(mesh, checked, name, root, device, *, support_radius, container=False,
                     provenance=None):
    """Content-addressed SDF cache. VM contents may later be appended by PBF."""
    options = BakeOptions(voxel_size=.0075 if container else .005,
                          max_support_radius=support_radius)
    recipe = dict(layout=2, mesh_sha256=collision.mesh_digest(mesh),
                  options=asdict(options), container=container,
                  dependencies={p: version(p) for p in ("newton", "warp-lang", "trimesh")})
    directory = collision.cache_directory(Path(root) / "boundaries", name, recipe)
    stem = directory / name
    metadata_path = stem.with_suffix(".json")
    mesh_path, volume_path = stem.with_suffix(".obj"), stem.with_suffix(".nvdb")
    if metadata_path.is_file() and mesh_path.is_file() and volume_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata["recipe"] != recipe or metadata["mesh_file_sha256"] != asset_io.file_digest(mesh_path):
            raise ValueError(f"Published boundary does not match its recipe: {stem}")
        with volume_path.open("rb") as stream:
            published = wp.Volume.load_from_nvdb(stream, device=device)
        if asset_io.sdf_digest(published) != metadata["sdf_sha256"]:
            raise ValueError(f"Published SDF content has changed: {volume_path}")
        return volume_path
    print(f"Baking PBF boundary: {name}", flush=True)
    volume = sdf.bake(checked, options=options, device=device) if container else sdf.bake_rigid(
        checked, options=options, device=device)
    save_mesh_obj(mesh, mesh_path, digits=17)
    sdf.save_volume(volume, volume_path)
    # Publishing assigns the grid's support-radius name. Hash the persisted
    # SDF, not the unnamed intermediate volume returned by the baker.
    with volume_path.open("rb") as stream:
        published = wp.Volume.load_from_nvdb(stream, device=device)
    metadata = dict(recipe=recipe, provenance=provenance or {},
                    mesh_file_sha256=asset_io.file_digest(mesh_path),
                    sdf_sha256=asset_io.sdf_digest(published),
                    bounds=mesh.bounds.tolist(), voxel_size=volume.get_voxel_size(),
                    tiles=volume.get_tile_count())
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return volume_path


def prepare_static(root, device, support_radius):
    mesh, checked, bunny, provenance = static_geometry()
    path = publish_boundary(mesh, checked, "pool", root, device,
                            support_radius=support_radius, container=True, provenance=provenance)
    hulls = collision.prepare_hulls(bunny, "suspended_bunny", Path(root) / "collision")
    return path, bunny, hulls


def add_static_collisions(builder, bunny, hulls):
    """The existing ground plane supplies the bottom; remaining shapes are static."""
    import newton
    cfg = builder.ShapeConfig(density=0., mu=1.)
    cfg.flags = newton.ShapeFlags.COLLIDE_SHAPES
    width, depth, height = POOL_SIZE
    wall = WALL_THICKNESS
    boxes = [
        ((-wall / 2, depth / 2, height / 2), (wall / 2, depth / 2 + wall, height / 2)),
        ((width + wall / 2, depth / 2, height / 2), (wall / 2, depth / 2 + wall, height / 2)),
        ((width / 2, -wall / 2, height / 2), (width / 2, wall / 2, height / 2)),
        ((width / 2, depth + wall / 2, height / 2), (width / 2, wall / 2, height / 2)),
        ((width / 2, depth / 2, height + wall / 2), (width / 2 + wall, depth / 2 + wall, wall / 2)),
    ]
    for index, (center, half) in enumerate(boxes):
        builder.add_shape_box(-1, xform=wp.transform(center, wp.quat_identity()),
                              hx=half[0], hy=half[1], hz=half[2], cfg=cfg, label=f"pool_wall_{index}")
    for index, hull in enumerate(hulls):
        shape = newton.Mesh(np.asarray(hull.vertices, dtype=np.float32),
                            np.asarray(hull.faces, dtype=np.int32).reshape(-1), compute_inertia=False)
        builder.add_shape_convex_hull(-1, mesh=shape, cfg=cfg, label=f"pool_bunny_collision_{index}")
    visible = builder.ShapeConfig(density=0.)
    visible.flags = newton.ShapeFlags.VISIBLE
    shape = newton.Mesh(np.asarray(bunny.vertices, dtype=np.float32),
                        np.asarray(bunny.faces, dtype=np.int32).reshape(-1), compute_inertia=False)
    builder.add_shape_mesh(-1, mesh=shape, cfg=visible, label="pool_bunny")


def body_meshes(model, body_ids):
    """Bake actual native collision shapes in each Newton body's local frame."""
    import newton
    body = model.shape_body.numpy()
    kind = model.shape_type.numpy()
    size = model.shape_scale.numpy()
    transforms = model.shape_transform.numpy()
    flags = model.shape_flags.numpy()
    for body_id in body_ids:
        parts = []
        for index in np.flatnonzero(body == body_id):
            if not flags[index] & int(newton.ShapeFlags.COLLIDE_SHAPES):
                continue
            shape_type = int(kind[index])
            if shape_type == int(newton.GeoType.SPHERE):
                mesh = trimesh.creation.icosphere(subdivisions=3, radius=float(size[index, 0]))
            elif shape_type == int(newton.GeoType.BOX):
                mesh = trimesh.creation.box(extents=2 * size[index])
            elif shape_type == int(newton.GeoType.CAPSULE):
                mesh = trimesh.creation.capsule(radius=float(size[index, 0]),
                                                height=float(2 * size[index, 1]))
            else:
                raise ValueError(f"Unsupported play-scene body geometry: {model.shape_label[index]} ({shape_type})")
            pose = transforms[index]
            matrix = trimesh.transformations.quaternion_matrix([pose[6], pose[3], pose[4], pose[5]])
            matrix[:3, 3] = pose[:3]
            mesh.apply_transform(matrix)
            parts.append(mesh)
        if not parts:
            continue
        mesh = geometry.union(parts) if len(parts) > 1 else parts[0]
        yield int(body_id), mesh, validation.validate_solid(mesh)
