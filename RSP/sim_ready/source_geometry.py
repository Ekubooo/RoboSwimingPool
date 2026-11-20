"""Source geometry and import conventions shared by preparation and previews."""

from dataclasses import dataclass

from .reference import POOL_SIZE, SCENE_TRANSFORM


WALL_THICKNESS = 0.3
CONTAINER_SIZES = {"reference_pool": POOL_SIZE, "wide_pool": (4.0, 2.0, 5.0), "stairs": POOL_SIZE}
OBJ_SIZE = .3
OBJ_DENSITY = 1000.


@dataclass(frozen=True)
class RigidSpec:
    name: str
    kind: str
    size: float
    density: float


PROPS = {
    spec.name: spec for spec in (
        RigidSpec("light_box", "box", .3, 500.),
        RigidSpec("heavy_box", "box", .3, 1200.),
        RigidSpec("scene_sphere", "sphere", .15, 750.),
    )
}


def normalized_obj(source):
    """Read a validated OBJ in body-local, Z-up coordinates in meters."""
    import numpy as np
    from .lib.mesh import geometry, validation

    path, = source.input_files().values()
    parts = validation.validate_rigid_parts(geometry.prepare_rigid(path))
    mesh = parts[0] if len(parts) == 1 else geometry.union(parts)
    # Convert (x, y, z) to (x, -z, y) before centering. This rotation does
    # not include the translation used to position the reference container.
    mesh.apply_transform(np.array(((1., 0., 0., 0.), (0., 0., -1., 0.),
                                   (0., 1., 0., 0.), (0., 0., 0., 1.))))
    span = mesh.extents.max()
    if not np.isfinite(span) or span <= 0:
        raise ValueError("OBJ must have a finite, positive extent")
    mesh.apply_translation(-mesh.bounds.mean(axis=0))
    mesh.apply_scale(OBJ_SIZE / span)
    return mesh


def prop_mesh(kind, size):
    """Produce display geometry for the standard box or sphere."""
    import trimesh

    if kind == "box":
        return trimesh.creation.box(extents=(size,) * 3)
    if kind == "sphere":
        return trimesh.creation.icosphere(subdivisions=3, radius=size)
    raise ValueError(f"Unsupported primitive geometry: {kind}")


def load_bunny(source):
    """Read and position the original single-shell bunny in the Z-up scene."""
    import numpy as np
    from .lib.mesh import geometry, validation

    mesh = geometry.prepare_rigid(source, obj_up_axis="Y")
    parts = validation.validate_rigid_parts(mesh)
    if len(parts) != 1:
        raise ValueError("The play scene expects the original single-shell bunny")
    bunny = parts[0]
    bunny.apply_scale(.6 / bunny.extents[1])
    bunny.apply_translation(np.array([2.15, .5, 1.]) - bunny.bounds.mean(axis=0))
    if np.any(bunny.bounds[0] <= 0) or np.any(bunny.bounds[1] >= (3., 5., 2.)):
        raise ValueError("Suspended bunny must be strictly inside the fluid cavity")
    bunny.apply_transform(np.asarray(SCENE_TRANSFORM))
    return bunny
