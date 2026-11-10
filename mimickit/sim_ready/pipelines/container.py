"""Prepare and bake a terrain/container boundary in caller-owned staging paths."""

from pathlib import Path

from ..config import GeometryOptions, BakeOptions


def prepare(source, *, options=GeometryOptions()):
    """Return the placed scene, visible mesh and aligned native collision boxes."""
    from .. import geometry

    scene = geometry.prepare_scene(source, options=options)
    walls, _ = geometry.container_parts(scene.terrain, options)
    boxes = geometry.terrain_boxes(options) + walls
    static_boxes = [(m.bounds.mean(axis=0).tolist(), (m.extents / 2.).tolist()) for m in boxes]
    visible = geometry.landscape(scene, options)
    return scene, visible, static_boxes


def build(scene, visible, stem, *, geometry_options=GeometryOptions(),
          bake_options=BakeOptions(), collision_cache, collision_name="model", device="cuda:0"):
    """Validate the cavity, prepare model collision and stage an OBJ/SDF pair.

    The caller owns cache policy, metadata, volume maps and final publication.
    """
    from .. import geometry, validation, sdf, collision
    from ..publish import save_mesh_obj

    stem = Path(stem)
    solid, bounds = geometry.container(scene, geometry_options)
    checked = validation.validate_container(solid, bounds, tolerance=geometry_options.simplify_tolerance)
    hulls = collision.prepare_hulls(scene.model, collision_name, collision_cache)
    volume = sdf.bake(checked, options=bake_options, device=device)
    mesh_path, sdf_path = stem.with_suffix(".obj"), stem.with_suffix(".nvdb")
    save_mesh_obj(visible, mesh_path, digits=8)
    sdf.save_volume(volume, sdf_path)
    return mesh_path, sdf_path, hulls
