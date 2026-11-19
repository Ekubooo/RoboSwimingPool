"""Replay the original policy observations into a persistent GPU buffer.

Small heading components are amplified by the trained normalizer. Keeping the
original Torch operations also preserves their rounding, while CUDA Graphs
remove the per-frame Python joint loop, allocations and kernel submissions.
"""
from dataclasses import dataclass, fields

import torch
import warp as wp


@dataclass(frozen=True, eq=False)
class CharacterStateViews:
    """Borrowed GPU views; their storage must remain stable across graph replays."""

    root_pos: torch.Tensor
    root_rot: torch.Tensor
    root_vel: torch.Tensor
    root_ang_vel: torch.Tensor
    dof_pos: torch.Tensor
    dof_vel: torch.Tensor
    body_pos: torch.Tensor


class ObservationBuilder:
    """Capture once for each projectile mode; replay without reading GPU state."""

    def __init__(self, *, state_views: CharacterStateViews, joints, key_body_ids,
                 inactive_sphere_state, scene_sphere_state, output,
                 global_obs, root_height_obs, use_scene_sphere=False):
        from util import torch_util
        from .observation_math import character_observation, projectile_observation, exp_map_to_quat

        self._state = state_views
        self.output = output
        if not self.output.is_cuda or any(t.dtype != torch.float32 or t.device != self.output.device
                                         for t in (getattr(state_views, f.name) for f in fields(state_views))):
            raise ValueError("Playback observations require FP32 CUDA state on the observation device")
        root_pos, root_rot = state_views.root_pos, state_views.root_rot
        root_vel, root_ang_vel = state_views.root_vel, state_views.root_ang_vel
        dof_pos, dof_vel, body_pos = state_views.dof_pos, state_views.dof_vel, state_views.body_pos
        self._keys = key_body_ids
        self._spheres = (inactive_sphere_state, scene_sphere_state)
        # Topology and axis tensors are fixed. Only their GPU values change on
        # replay; no per-frame traversal of the kinematic model is needed.
        joints = tuple(joints)
        self._joint_rot = dof_pos.new_empty((dof_pos.shape[0], len(joints), 4))
        joint_rot = self._joint_rot

        def compute(sphere_pos, sphere_vel):
            for index, joint in enumerate(joints):
                dim = joint.get_dof_dim()
                dof = joint.get_joint_dof(dof_pos)
                if dim == 1:
                    axis = torch.broadcast_to(joint.axis, dof.shape[:-1] + (3,))
                    rotation = torch_util.axis_angle_to_quat(axis, dof.squeeze(-1))
                elif dim == 3:
                    rotation = exp_map_to_quat(dof)
                elif dim == 0:
                    rotation = torch.zeros_like(joint_rot[:, index])
                    rotation[..., -1].fill_(1.)
                else:
                    raise ValueError(f"Unsupported playback joint dimension: {dim}")
                joint_rot[:, index].copy_(rotation)
            keys = body_pos[:, self._keys] if len(self._keys) else body_pos.new_zeros((0,))
            character = character_observation(root_pos, root_rot, root_vel, root_ang_vel,
                joint_rot, dof_vel, keys, global_obs, root_height_obs)
            task = projectile_observation(root_pos, root_rot, sphere_pos, sphere_vel)
            self.output.copy_(torch.cat((character, task), dim=-1))

        self._graphs = []
        caller_stream = torch.cuda.current_stream(self.output.device)
        stream = torch.cuda.Stream(device=self.output.device)
        stream.wait_stream(caller_stream)
        # Construction is allowed to wait. Playback only replays GPU work on
        # the runner's stream, including the output copy to stable env views.
        with torch.no_grad(), torch.cuda.stream(stream), \
                wp.ScopedStream(wp.stream_from_torch(stream), sync_exit=True):
            for sphere_pos, sphere_vel in self._spheres:
                for _ in range(3):
                    compute(sphere_pos, sphere_vel)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    compute(sphere_pos, sphere_vel)
                self._graphs.append(graph)
        caller_stream.wait_stream(stream)
        self.update(use_scene_sphere=use_scene_sphere)

    def update(self, env_ids=None, *, use_scene_sphere=False):
        # PBFPlayEnv has one world; empty reset selections must be a no-op.
        if env_ids is None or len(env_ids):
            self._graphs[int(use_scene_sphere)].replay()
