"""Rectangular container recipes and declarative native collision assembly."""
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


def add_collisions(builder, layout, bunny, hulls):
    """Assemble published shapes using Newton's public builder interface."""
    import newton
    import warp as wp
    cfg = builder.ShapeConfig(density=0., mu=1.)
    cfg.flags = newton.ShapeFlags.COLLIDE_SHAPES
    for index, box in enumerate(layout["boxes"]):
        hx, hy, hz = box["half"]
        builder.add_shape_box(-1, xform=wp.transform(box["center"], wp.quat_identity()),
                             hx=hx, hy=hy, hz=hz, cfg=cfg, label=f"pool_wall_{index}")
    for index, hull in enumerate(hulls):
        mesh = newton.Mesh(np.asarray(hull.vertices, dtype=np.float32),
                           np.asarray(hull.faces, dtype=np.int32).reshape(-1), compute_inertia=False)
        builder.add_shape_convex_hull(-1, mesh=mesh, cfg=cfg, label=f"pool_bunny_collision_{index}")
    visible = builder.ShapeConfig(density=0.)
    visible.flags = newton.ShapeFlags.VISIBLE
    mesh = newton.Mesh(np.asarray(bunny.vertices, dtype=np.float32),
                       np.asarray(bunny.faces, dtype=np.int32).reshape(-1), compute_inertia=False)
    builder.add_shape_mesh(-1, mesh=mesh, cfg=visible, label="pool_bunny")
