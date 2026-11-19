"""Rectangular container preparation recipes and published layout metadata."""
import numpy as np
import trimesh

from . import pbf_play as reference
from ...lib.mesh import validation


def description(size, bunny):
    width, depth, height = size
    wall = reference.WALL_THICKNESS
    boxes = [
        ([-wall/2, depth/2, height/2], [wall/2, depth/2+wall, height/2]),
        ([width+wall/2, depth/2, height/2], [wall/2, depth/2+wall, height/2]),
        ([width/2, -wall/2, height/2], [width/2, wall/2, height/2]),
        ([width/2, depth+wall/2, height/2], [width/2, wall/2, height/2]),
        ([width/2, depth/2, height+wall/2], [width/2+wall, depth/2+wall, wall/2]),
    ]
    probes = {"cavity": [.5, .5, 1.], "bottom": [1., 1., -.05],
              "left_wall": [-.05, 1., 1.], "right_wall": [width+.05, 1., 1.],
              "front_wall": [1., -.05, 1.], "back_wall": [1., depth+.05, 1.],
              "ceiling": [1., 1., height+.05], "bunny": bunny.center_mass.tolist()}
    return dict(units="m", up_axis="Z", size=list(size), wall_thickness=wall,
                boxes=[dict(center=c, half=h) for c, h in boxes], probes=probes)


def prepare_container(store, device, support_radius, size, report):
    """Keep the reference recipe exact; only widen the second cavity."""
    if tuple(size) == (3., 2., 5.):
        artifact, bunny, hulls = reference.prepare_static(store, device, support_radius, report=report)
    elif tuple(size) == (4., 2., 5.):
        _, _, bunny, provenance = reference.static_geometry()
        size = np.asarray(size, dtype=float)
        outer = trimesh.creation.box(extents=size + 2*reference.WALL_THICKNESS)
        outer.apply_translation(size/2)
        inner = trimesh.creation.box(extents=size)
        inner.apply_translation(size/2)
        inner.invert()
        container = trimesh.util.concatenate([outer, inner])
        checked_container = validation.validate_container(container, container.bounds)
        solid = validation.validate_solid(bunny)
        vertices = np.concatenate([checked_container.vertices, solid.vertices])
        faces = np.concatenate([checked_container.faces, solid.faces + len(checked_container.vertices)])
        mesh = trimesh.Trimesh(vertices, faces, process=False)
        checked = validation.ValidatedMesh(vertices, faces, checked_container.bounds, checked_container.cavity_bounds)
        provenance = dict(provenance, pool_size=size.tolist(), recipe="wide_pool")
        artifact = reference.prepare_boundary(mesh, checked, store, device, support_radius=support_radius,
                                               container=True, provenance=provenance)
        hulls = reference.collision.prepare_hulls(bunny, "suspended_bunny", store)
    else:
        raise ValueError("Supported container sizes are 3x2x5 and 4x2x5")
    return artifact, bunny, hulls, description(size, bunny)


