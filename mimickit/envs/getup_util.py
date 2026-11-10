"""Humanoid MJCF collision supports used only to place Getup reset poses."""
import itertools
import xml.etree.ElementTree as ET

import torch
import util.torch_util as torch_util


class GroundSupports:
    """Exact lower supports of the spheres/capsules/boxes in this Humanoid asset."""
    def __init__(self, char_file, kin_model, device):
        points, radii, bodies = [], [], []
        tree = ET.parse(char_file)
        for body in tree.findall(".//worldbody//body"):
            body_id = kin_model.get_body_id(body.attrib["name"])
            for geom in body.findall("geom"):
                kind = geom.get("type", "sphere")
                size = [float(x) for x in geom.get("size", "").split()]
                pos = torch.tensor([float(x) for x in geom.get("pos", "0 0 0").split()])
                if any(key in geom.attrib for key in ("euler", "axisangle", "xyaxes", "zaxis")):
                    raise ValueError("Getup support parser expects pos/quat/fromto geoms.")
                if kind == "sphere":
                    local, radius = pos[None], size[0]
                elif kind == "capsule" and "fromto" in geom.attrib:
                    local = torch.tensor([float(x) for x in geom.get("fromto").split()]).view(2, 3)
                    radius = size[0]
                elif kind == "box":
                    local = torch.tensor(list(itertools.product(*[(-v, v) for v in size])))
                    if "quat" in geom.attrib:
                        q = [float(x) for x in geom.get("quat").split()]
                        q = torch.tensor([q[1], q[2], q[3], q[0]]).expand(len(local), -1)
                        local = torch_util.quat_rotate(q, local)
                    local, radius = local + pos, 0.0
                else:
                    raise ValueError(f"Unsupported Getup collision geometry: {kind}")
                points.extend(local.tolist())
                radii.extend([radius] * len(local))
                bodies.extend([body_id] * len(local))
        if not points:
            raise ValueError("No collision supports found.")
        self.points = torch.tensor(points, device=device, dtype=torch.float32)
        self.radii = torch.tensor(radii, device=device, dtype=torch.float32)
        self.bodies = torch.tensor(bodies, device=device, dtype=torch.long)

    def minimum_height(self, body_pos, body_rot):
        rotations = body_rot[:, self.bodies]
        points = self.points.unsqueeze(0).expand(len(body_pos), -1, -1)
        world = torch_util.quat_rotate(rotations, points) + body_pos[:, self.bodies]
        return (world[..., 2] - self.radii).amin(dim=1)
