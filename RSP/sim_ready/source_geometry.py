"""Reference asset geometry shared by preparation and the CPU selector."""

from dataclasses import dataclass

from .reference import POOL_SIZE, SCENE_TRANSFORM


WALL_THICKNESS = 0.3
CONTAINER_SIZES = {"reference_pool": POOL_SIZE, "wide_pool": (4.0, 2.0, 5.0)}


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
