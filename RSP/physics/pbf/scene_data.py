"""Geometry dimensions shared by offline production and runtime assembly."""

BOX_SIZE = .30
SPHERE_RADIUS = .15
SPHERE_SUBDIVISIONS = 3

# Asset geometry is shared by offline production and runtime assembly.
# Instance placement and density belong to scene.RIGIDS, not these recipes.
RIGID_ASSETS = {
    "box": dict(kind="box", size=[BOX_SIZE] * 3),
    "sphere": dict(kind="sphere", radius=SPHERE_RADIUS, subdivisions=SPHERE_SUBDIVISIONS),
    "spot": dict(kind="mesh", source="spot_triangulated_good.obj", scale=.25, obj_up_axis="Y"),
}
