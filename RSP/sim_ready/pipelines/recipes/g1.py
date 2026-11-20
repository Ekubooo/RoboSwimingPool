"""Repository G1 recipe: authored visuals, native collision-only boundaries."""
EXCLUDED_LINKS = tuple(f"{side}_{part}_link" for side in ("left", "right")
                       for part in ("hip_pitch", "ankle_pitch", "shoulder_pitch", "wrist_roll")) + (
    "waist_yaw_link", "waist_roll_link")


def check_collision_scope(asset):
    """Fail explicitly if the original model no longer matches this recipe."""
    excluded = {body.name for body in asset.bodies if not body.shapes}
    shapes = [shape for body in asset.bodies for shape in body.shapes]
    if (len(asset.bodies) != 31 or excluded != set(EXCLUDED_LINKS)
            or len(shapes) != 27 or any(shape.kind not in {"CAPSULE", "SPHERE"} for shape in shapes)):
        raise ValueError("G1 recipe requires 31 bodies, 27 native primitives and the explicit 10 exclusions")
