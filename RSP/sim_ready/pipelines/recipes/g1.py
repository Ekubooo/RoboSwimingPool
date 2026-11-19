"""Repository G1 recipe: authored visuals, native collision-only boundaries."""
import xml.etree.ElementTree as ET

from ...catalog import contained
from ...paths import G1_SOURCE_ROOT

SOURCE_ROOT = G1_SOURCE_ROOT
EXCLUDED_LINKS = tuple(f"{side}_{part}_link" for side in ("left", "right")
                       for part in ("hip_pitch", "ankle_pitch", "shoulder_pitch", "wrist_roll")) + (
    "waist_yaw_link", "waist_roll_link")


def source_files():
    """Resolve only dependencies of the repository's unchanged g1.xml."""
    root = ET.parse(SOURCE_ROOT / "g1.xml").getroot()
    names = {"g1.xml", "LICENSE.txt"}
    names.update(element.attrib["file"] for element in root.iter() if "file" in element.attrib)
    files = {name: contained(SOURCE_ROOT, name) for name in sorted(names)}
    for path in files.values():
        if not path.is_file():
            raise ValueError(f"Missing G1 source dependency: {path}")
    return files


def check_collision_scope(asset):
    """Fail explicitly if the original model no longer matches this recipe."""
    excluded = {body.name for body in asset.bodies if not body.shapes}
    shapes = [shape for body in asset.bodies for shape in body.shapes]
    if (len(asset.bodies) != 31 or excluded != set(EXCLUDED_LINKS)
            or len(shapes) != 27 or any(shape.kind not in {"CAPSULE", "SPHERE"} for shape in shapes)):
        raise ValueError("G1 recipe requires 31 bodies, 27 native primitives and the explicit 10 exclusions")
