"""G1 kinematics <-> the released Humanoid LaFAN prior's representation.

No network weights or normalizers are transformed. Local rotations are obtained
from complete physical chains, with explicit rest-frame changes. Unavailable
neck/wrist degrees of freedom are projected, not invented as learned channels.
"""
import math
import xml.etree.ElementTree as ET

import torch

from anim.mjcf_char_model import MJCFCharModel
from envs.amp_env import compute_disc_obs
from util import torch_util as tu


def identity(shape, device):
    q = torch.zeros((*shape, 4), device=device)
    q[..., 3] = 1
    return q


def rotation(axis, angle, device):
    v = torch.zeros(3, device=device)
    v[axis] = angle
    return tu.exp_map_to_quat(v)


class G1LafanAdapter:
    KEY_BODIES = ("head", "right_hand", "left_hand", "right_foot", "left_foot")

    def __init__(self, g1, device, char_file="data/assets/g1/g1.xml"):
        self.g1, self.device = g1, device
        self.human = MJCFCharModel(device)
        self.human.load("data/assets/humanoid/humanoid.xml")
        if g1.get_dof_size() != 29 or g1.get_num_joints() != 31:
            raise ValueError("LaFAN adapter requires the repository G1 29DoF skeleton")
        self.key_ids = [self.human.get_body_id(n) for n in self.KEY_BODIES]
        self.chains = []
        self.source_ids = [g1.get_body_id("pelvis")]
        self.corrections = [identity((), device)]
        # C maps canonical body axes into the physical link's axes: Rh = Rg C.
        mapping = {"torso": ("torso_link", identity((), device)),
                   "head": ("head_link", identity((), device))}
        for side, sign in (("right", -1), ("left", 1)):
            upper = rotation(0, -sign * math.pi / 2, device)
            lower = tu.quat_mul(upper, rotation(2, -sign * math.pi / 2, device))
            hip_id = g1.get_body_id(side + "_hip_roll_link")
            mapping.update({
                side + "_upper_arm": (side + "_shoulder_yaw_link", upper),
                side + "_lower_arm": (side + "_elbow_link", lower),
                side + "_hand": (side + "_elbow_link", lower),
                side + "_thigh": (side + "_hip_yaw_link", tu.quat_conjugate(g1._local_rotation[hip_id])),
                side + "_shin": (side + "_knee_link", identity((), device)),
                side + "_foot": (side + "_ankle_roll_link", identity((), device)),
            })
        for name in self.human.get_body_names()[1:]:
            link, corr = mapping[name]
            self.source_ids.append(g1.get_body_id(link))
            self.corrections.append(corr)
        self.corrections = torch.stack(self.corrections)
        ranges = {j.attrib["name"]: [float(x) for x in j.attrib["range"].split()]
                  for j in ET.parse(char_file).iter("joint") if "range" in j.attrib}
        self.low = torch.empty(29, device=device)
        self.high = torch.empty_like(self.low)
        for j in range(1, g1.get_num_joints()):
            joint = g1.get_joint(j)
            if joint.get_dof_dim():
                lo, hi = ranges[joint.name]
                self.low[joint.dof_idx] = max(lo + .03, .03) if "knee" in joint.name else lo + .03
                self.high[joint.dof_idx] = hi - .03
        # Topological order; inverse solve uses actual solved parent orientations.
        groups = [("torso", ["waist_yaw_link", "waist_roll_link", "torso_link"])]
        for side in ("right", "left"):
            groups += [(side + "_thigh", [side + "_hip_" + a + "_link" for a in ("pitch", "roll", "yaw")]),
                       (side + "_shin", [side + "_knee_link"]),
                       (side + "_foot", [side + "_ankle_" + a + "_link" for a in ("pitch", "roll")]),
                       (side + "_upper_arm", [side + "_shoulder_" + a + "_link" for a in ("pitch", "roll", "yaw")]),
                       (side + "_lower_arm", [side + "_elbow_link"])]
        for name, links in groups:
            self.chains.append((self.human.get_body_id(name), [g1.get_body_id(x) for x in links]))
        # Ground-to-root ratio from actual neutral FK and collision sole geometry.
        heights = []
        for model, kind in ((g1, "g1"), (self.human, "human")):
            p, r = model.forward_kinematics(torch.zeros(1, 3, device=device), identity((1,), device),
                                            model.dof_to_rot(torch.zeros(1, model.get_dof_size(), device=device)))
            heights.append(-self.sole_height(p, r, kind).item())
        self.scale = heights[1] / heights[0]

    def sole_height(self, pos, rot, kind):
        model = self.g1 if kind == "g1" else self.human
        suffix = "_ankle_roll_link" if kind == "g1" else "_foot"
        ids = [model.get_body_id(s + suffix) for s in ("right", "left")]
        if kind == "g1":
            # Capsule endpoints; subtract radius along world Z (not link Z).
            points = [[x, y, -.015] for x in (-.035, .115) for y in (-.015, .015)]
            points += [[-.047, 0, -.015], [.126, 0, -.015]]
            radius = .02
        else:
            points = [[x, y, z] for x in (.045-.0885, .045+.0885)
                      for y in (-.045, .045) for z in (-.05, .005)]
            radius = 0
        points = torch.tensor(points, device=pos.device, dtype=pos.dtype)
        q = rot[..., ids, :].unsqueeze(-2).expand(*rot.shape[:-2], 2, len(points), 4)
        pts = points.expand(*q.shape[:-1], 3)
        z = tu.quat_rotate(q, pts)[..., 2] + pos[..., ids, 2].unsqueeze(-1)
        return z.amin(dim=(-2, -1)) - radius

    def to_human(self, root_pos, root_rot, joint_rot):
        gp, gr = self.g1.forward_kinematics(root_pos, root_rot, joint_rot)
        world = tu.quat_mul(gr[..., self.source_ids, :], self.corrections.expand(*gr.shape[:-2], -1, -1))
        parents = self.human._parent_indices[1:]
        local = tu.quat_mul(tu.quat_conjugate(world[..., parents, :]), world[..., 1:, :])
        # Respect canonical hinge/fixed joints. Neck is fixed on G1; hands on Humanoid.
        local = self.human.dof_to_rot(self.human.rot_to_dof(local))
        hp = root_pos * self.scale
        bp, br = self.human.forward_kinematics(hp, root_rot, local)
        # Preserve actual foot clearance, rather than reward an artificial body size.
        shift = self.sole_height(gp, gr, "g1") * self.scale - self.sole_height(bp, br, "human")
        hp = hp.clone()
        hp[..., 2] += shift
        bp = bp.clone()
        bp[..., 2] += shift.unsqueeze(-1)
        return hp, local, bp

    def observations(self, root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dt):
        hp, local, bp = self.to_human(root_pos, root_rot, joint_rot)
        velocity = root_vel * self.scale
        # The height correction depends on joint pose. Differentiate that correction
        # over the actual history; XY retains measured engine velocity.
        correction = hp[..., 2] - root_pos[..., 2] * self.scale
        velocity = velocity.clone()
        dz = (correction[:, 1:] - correction[:, :-1]) / dt
        velocity[:, :-1, 2] += dz
        velocity[:, -1, 2] += dz[:, -1]
        return compute_disc_obs(hp[:, -1], root_rot[:, -1], hp, root_rot, velocity,
                                root_ang_vel, local, velocity[..., :0], bp[..., self.key_ids, :],
                                False, True, False)

    def _solve_chain(self, target, ids):
        dofs = [self.g1.get_joint(i).dof_idx for i in ids]
        low, high = self.low[dofs], self.high[dofs]
        if len(ids) == 1:
            joint = self.g1.get_joint(ids[0])
            local = tu.quat_mul(tu.quat_conjugate(self.g1._local_rotation[ids[0]]).expand_as(target), target)
            q = joint.rot_to_dof(local).clamp(low, high)
        else:
            q = torch.zeros((*target.shape[:-1], len(ids)), device=self.device)
            eye = torch.eye(3, device=self.device)
            for _ in range(20):
                current = identity(target.shape[:-1], self.device)
                axes = []
                for k, i in enumerate(ids):
                    current = tu.quat_mul(current, self.g1._local_rotation[i].expand_as(current))
                    joint = self.g1.get_joint(i)
                    axes.append(tu.quat_rotate(current, joint.axis.expand(*current.shape[:-1], 3)))
                    current = tu.quat_mul(current, joint.dof_to_rot(q[..., k:k+1]))
                error = tu.quat_to_exp_map(tu.quat_mul(target, tu.quat_conjugate(current)))
                jac = torch.stack(axes, dim=-1)
                delta = (jac.transpose(-1, -2) @ torch.linalg.solve(
                    jac @ jac.transpose(-1, -2) + .01 * eye, error.unsqueeze(-1))).squeeze(-1)
                q = (q + delta.clamp(-.35, .35)).clamp(low, high)
        current = identity(target.shape[:-1], self.device)
        for k, i in enumerate(ids):
            current = tu.quat_mul(current, self.g1._local_rotation[i].expand_as(current))
            current = tu.quat_mul(current, self.g1.get_joint(i).dof_to_rot(q[..., k:k+1]))
        return q, current

    @torch.no_grad()
    def from_samples(self, samples):
        if samples.ndim != 3 or samples.shape[1:] != (10, 114) or not torch.isfinite(samples).all():
            raise ValueError("Expected finite LaFAN windows [batch,10,114]")
        hp = samples[..., :3]
        rr = tu.tan_norm_to_quat(samples[..., 3:9])
        hj = tu.tan_norm_to_quat(samples[..., 9:93].reshape(*samples.shape[:2], 14, 6))
        hj = self.human.dof_to_rot(self.human.rot_to_dof(hj))
        hb, hr = self.human.forward_kinematics(hp, rr, hj)
        q = torch.zeros((*samples.shape[:2], 29), device=self.device)
        solved = {0: rr}
        for h_id, ids in self.chains:
            parent = self.g1.get_parent_id(ids[0])
            parent_rot = solved[parent]
            desired = tu.quat_mul(hr[..., h_id, :], tu.quat_conjugate(self.corrections[h_id]).expand_as(rr))
            target = tu.quat_mul(tu.quat_conjugate(parent_rot), desired)
            angles, relative = self._solve_chain(target, ids)
            for k, i in enumerate(ids):
                q[..., self.g1.get_joint(i).dof_idx] = angles[..., k]
            solved[ids[-1]] = tu.quat_mul(parent_rot, relative)
        # No canonical wrist orientation exists; neutral wrists do not add a new prior.
        gp = hp / self.scale
        gb, gr = self.g1.forward_kinematics(gp, rr, self.g1.dof_to_rot(q))
        clearance = (self.sole_height(hb, hr, "human") / self.scale).clamp_min(.002)
        gp = gp.clone()
        gp[..., 2] += clearance - self.sole_height(gb, gr, "g1")
        frames = torch.cat((gp, tu.quat_to_exp_map(rr), q), dim=-1)
        if not torch.isfinite(frames).all():
            raise ValueError("Nonfinite LaFAN inverse retargeting")
        return frames
