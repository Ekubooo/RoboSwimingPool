"""Temporary assets for the single-world PBF play scene.

The existing producers retain their conventions. This scene adapter chooses
the pool layout, the Y-up to Z-up conversion and asset recipes. An explicit
AssetStore supplies output locations.
"""
import xml.etree.ElementTree as ET

import numpy as np
import trimesh

from ...lib.mesh import validation
from ...lib.storage.fingerprints import file_digest
from ...paths import ORIGIN_ROOT
from ...config import BakeOptions
from ..boundary import prepare_boundary_asset
from ...reference import POOL_SIZE, SCENE_TRANSFORM
from ...source_geometry import PROPS, WALL_THICKNESS, load_bunny


def write_prop_model(name, path):
    """Generate only the selected primitive as an ordinary MJCF object."""
    spec = PROPS[name]
    document = ET.Element("mujoco", model=spec.name)
    world = ET.SubElement(document, "worldbody")
    body = ET.SubElement(world, "body", name=spec.name)
    ET.SubElement(body, "freejoint", name="root")
    size = " ".join([str(spec.size / 2)] * 3) if spec.kind == "box" else str(spec.size)
    ET.SubElement(body, "geom", name=spec.name, type=spec.kind, size=size,
                  density=str(spec.density), friction="1.0 0.05 0.05")
    path.write_text(ET.tostring(document, encoding="unicode") + "\n", encoding="utf-8")
    return path


def _box(extents, center):
    mesh = trimesh.creation.box(extents=extents)
    mesh.apply_translation(center)
    return mesh


def static_geometry(source=ORIGIN_ROOT / "props" / "bunny_flat_bottom.obj"):
    """One exterior, one cavity and a disjoint positive bunny shell; no union."""
    size_y = np.array([3., 5., 2.])
    outer = _box(size_y + 2 * WALL_THICKNESS, size_y / 2)
    inner = _box(size_y, size_y / 2)
    inner.invert()
    container_y = trimesh.util.concatenate([outer, inner])
    validation.validate_container(container_y, outer.bounds)

    container = container_y.copy()
    scene_transform = np.asarray(SCENE_TRANSFORM)
    container.apply_transform(scene_transform)
    bunny = load_bunny(source)
    container_data = validation.validate_container(container, container.bounds)
    bunny_data = validation.validate_solid(bunny)
    # Independent closed shells are disjoint, so concatenating their oriented
    # triangles preserves the solid/cavity winding without a Boolean operation.
    vertices = np.concatenate([container_data.vertices, bunny_data.vertices])
    faces = np.concatenate([container_data.faces, bunny_data.faces + len(container_data.vertices)])
    combined = trimesh.Trimesh(vertices, faces, process=False)
    checked = validation.ValidatedMesh(vertices, faces, container_data.bounds, container_data.cavity_bounds)
    provenance = dict(source=str(source), source_sha256=file_digest(source),
                      input_up_axis="Y", world_up_axis="Z", transform=scene_transform.tolist(),
                      bunny_height=.6, bunny_center=[2.15, 1., .5],
                      pool_size=list(POOL_SIZE), wall_thickness=WALL_THICKNESS,
                      static_composition="disjoint oriented shells; no Boolean union")
    return combined, checked, bunny, provenance


def prepare_boundary(mesh, checked, store, device, *, support_radius, container=False,
                     provenance=None, previous=None):
    """Select scene-specific bake resolution and prepare a complete artifact."""
    options = BakeOptions(voxel_size=.0075 if container else .005,
                          max_support_radius=support_radius)
    return prepare_boundary_asset(mesh, checked, store=store, device=device, bake_options=options,
                                  support_radius=support_radius, container=container,
                                  provenance=provenance, previous=previous)


