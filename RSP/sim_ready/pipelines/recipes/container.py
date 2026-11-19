"""Rectangular container preparation recipes and published layout metadata."""
import numpy as np
import trimesh

from . import pbf_play as reference
from .. import collision
from ...boundary.schema import BoundaryArtifact
from ...lib.asset_validation import check_source
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


def prepare_container(store, device, support_radius, size, report, *, bunny_source, previous=None):
    """Keep the reference recipe exact; only widen the second cavity."""
    check_source(bunny_source, {".obj"}, report)
    mesh, checked, bunny, provenance = reference.static_geometry(bunny_source)
    if tuple(size) == (4., 2., 5.):
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
    elif tuple(size) != (3., 2., 5.):
        raise ValueError("Supported container sizes are 3x2x5 and 4x2x5")
    report.add("pass", "static.conventions", "pool and bunny", provenance)
    old = BoundaryArtifact(previous.file(previous.record["boundary"])) if previous is not None else None
    artifact = reference.prepare_boundary(
        mesh, checked, store, device, support_radius=support_radius, container=True,
        provenance=provenance, previous=old,
    )
    collision.prepare_hulls(bunny, "suspended_bunny", store, destination="collision", previous=previous)
    return artifact, bunny, description(size, bunny)


