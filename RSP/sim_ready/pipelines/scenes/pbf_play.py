"""Temporary assets for the single-world PBF play scene.

The existing producers retain their conventions. This scene adapter chooses
the pool layout, the Y-up to Z-up conversion and asset recipes. An explicit
AssetStore supplies output locations.
"""
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import warp as wp

from ...lib.mesh import geometry, validation
from ...lib.storage.fingerprints import file_digest
from ...lib.storage.store import AssetStore
from ...lib.asset_validation import ValidationReport, check_source
from ...paths import ORIGIN_ROOT
from .. import collision
from ...config import BakeOptions
from ..boundary.prepare import prepare_boundary_asset


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


def primitive_files(store: AssetStore):
    """Use ordinary MJCF objects so MimicKit retains its object bookkeeping."""
    directory = store.directory("primitives")
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

    source = ORIGIN_ROOT / "bunny_flat_bottom.obj"
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
    provenance = dict(source=str(source), source_sha256=file_digest(source),
                      input_up_axis="Y", world_up_axis="Z", transform=SCENE_TRANSFORM.tolist(),
                      bunny_height=.6, bunny_center=[2.15, 1., .5],
                      pool_size=POOL_SIZE.tolist(), wall_thickness=WALL_THICKNESS,
                      static_composition="disjoint oriented shells; no Boolean union")
    return combined, checked, bunny, provenance


def prepare_boundary(mesh, checked, store, device, *, support_radius, container=False, provenance=None):
    """Select scene-specific bake resolution and prepare a complete artifact."""
    options = BakeOptions(voxel_size=.0075 if container else .005,
                          max_support_radius=support_radius)
    return prepare_boundary_asset(mesh, checked, store=store, device=device, bake_options=options,
                                  support_radius=support_radius, container=container, provenance=provenance)


def prepare_static(store: AssetStore, device, support_radius, *, report=None):
    report = ValidationReport() if report is None else report
    check_source(ORIGIN_ROOT / "bunny_flat_bottom.obj", {".obj"}, report)
    mesh, checked, bunny, provenance = static_geometry()
    report.add("pass", "static.conventions", "pool and bunny", provenance)
    path = prepare_boundary(mesh, checked, store, device,
                            support_radius=support_radius, container=True, provenance=provenance)
    hulls = collision.prepare_hulls(bunny, "suspended_bunny", store)
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
