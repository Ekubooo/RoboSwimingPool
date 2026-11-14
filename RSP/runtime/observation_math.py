"""MimicKit observation math with graph-safe constant fills.

These formulas mirror util.torch_util and envs.char_env/task_dodgeball_env.
TorchScript's ``tensor[..., i] = 1`` creates a CPU scalar then copies it to
CUDA. ``fill_`` preserves the value while removing that transfer/synchronization.
Keep the scripted expression order: the trained normalizer amplifies tiny
heading components, so mathematically equivalent quaternion formulas can differ.
"""
import torch
from util import torch_util as tu


@torch.jit.script
def exp_map_to_quat(exp_map):
    angle = torch.norm(exp_map, dim=-1)
    axis = exp_map / angle.unsqueeze(-1)
    angle = tu.normalize_angle(angle)
    default_axis = torch.zeros_like(exp_map)
    default_axis[..., -1].fill_(1.)
    mask = torch.abs(angle) > 1e-5
    angle = torch.where(mask, angle, torch.zeros_like(angle))
    axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
    return tu.axis_angle_to_quat(axis, angle)


@torch.jit.script
def heading_inverse(q):
    ref_dir = torch.zeros_like(q[..., 0:3])
    ref_dir[..., 0].fill_(1.)
    rot_dir = tu.quat_rotate(q, ref_dir)
    heading = torch.atan2(rot_dir[..., 1], rot_dir[..., 0])
    axis = torch.zeros_like(q[..., 0:3])
    axis[..., 2].fill_(1.)
    return tu.axis_angle_to_quat(axis, -heading)


@torch.jit.script
def tangent_normal(q):
    tangent = torch.zeros_like(q[..., 0:3])
    tangent[..., 0].fill_(1.)
    normal = torch.zeros_like(q[..., 0:3])
    normal[..., -1].fill_(1.)
    return torch.cat((tu.quat_rotate(q, tangent), tu.quat_rotate(q, normal)), dim=-1)


@torch.jit.script
def character_observation(root_pos, root_rot, root_vel, root_ang_vel, joint_rot,
                          dof_vel, key_pos, global_obs: bool, root_height_obs: bool):
    heading = heading_inverse(root_rot)
    if global_obs:
        rotation = tangent_normal(root_rot)
        velocity, angular = root_vel, root_ang_vel
    else:
        rotation = tangent_normal(tu.quat_mul(heading, root_rot))
        velocity = tu.quat_rotate(heading, root_vel)
        angular = tu.quat_rotate(heading, root_ang_vel)
    joint_flat = joint_rot.reshape(-1, 4)
    joints = tangent_normal(joint_flat).reshape(joint_rot.shape[0], -1)
    parts = [rotation, velocity, angular, joints, dof_vel]
    if len(key_pos) > 0:
        keys = key_pos - root_pos.unsqueeze(-2)
        if not global_obs:
            key_heading = heading_inverse(root_rot).unsqueeze(-2).repeat((1, keys.shape[1], 1))
            keys = tu.quat_rotate(key_heading.reshape(-1, 4), keys.reshape(-1, 3)).reshape(keys.shape)
        parts = parts + [keys.reshape(keys.shape[0], -1)]
    if root_height_obs:
        parts = [root_pos[:, 2:3]] + parts
    return torch.cat(parts, dim=-1)


@torch.jit.script
def projectile_observation(root_pos, root_rot, proj_pos, proj_vel):
    heading = heading_inverse(root_rot)
    heading = heading.unsqueeze(-2).repeat((1, proj_pos.shape[1], 1))
    local_pos = proj_pos - root_pos.unsqueeze(-2)
    local_pos = tu.quat_rotate(heading, local_pos)
    local_vel = tu.quat_rotate(heading, proj_vel)
    return torch.cat((local_pos, local_vel), dim=-1).reshape(root_pos.shape[0], -1)
