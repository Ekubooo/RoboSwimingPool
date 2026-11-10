"""Independent SMP Getup baseline. Success is measured, never an early reset."""
import math
import torch

import envs.base_env as base_env
import envs.smp_env as smp_env
from envs.getup_util import GroundSupports
import util.torch_util as torch_util


@torch.jit.script
def compute_getup_reward(head_height: torch.Tensor, root_vertical_velocity: torch.Tensor,
                         target_height: float, low_height: float, target_speed: float,
                         height_scale: float, speed_scale: float, literal: bool):
    # Paper D.5 prints min(deficit, 0), opposite to its written objective.
    # Default uses positive deficits; the literal form remains an explicit option.
    h = target_height - head_height
    v = target_speed - root_vertical_velocity
    if literal:
        h, v = h.clamp_max(0.0), v.clamp_max(0.0)
    else:
        h, v = h.clamp_min(0.0), v.clamp_min(0.0)
    height_reward = torch.exp(-height_scale * h.square())
    velocity_reward = torch.where(head_height < low_height,
                                   torch.exp(-speed_scale * v.square()),
                                   torch.ones_like(head_height))
    return 0.3 * height_reward + 0.7 * velocity_reward


class TaskGetupEnv(smp_env.SMPEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        if env_config["enable_early_termination"] or env_config.get("pose_termination", False):
            raise ValueError("Getup must allow body/hand/knee ground contacts and fallen poses.")
        if env_config.get("enable_tar_obs", False) or env_config.get("enable_phase_obs", False):
            raise ValueError("Getup baseline does not track a target motion or phase.")
        self._aug_probability = float(env_config.get("augmented_reset_probability", 0.1))
        self._test_init = env_config.get("getup_test_init", "starts")
        self._reward_mode = env_config.get("getup_reward_mode", "deficit")
        if not 0 <= self._aug_probability <= 1:
            raise ValueError("Invalid augmented_reset_probability.")
        if self._test_init not in ("starts", "augmented", "mixed"):
            raise ValueError("getup_test_init must be starts, augmented or mixed.")
        if self._reward_mode not in ("deficit", "paper_literal"):
            raise ValueError("getup_reward_mode must be deficit or paper_literal.")
        self._target_height = float(env_config.get("getup_target_head_height", 1.2))
        self._low_height = float(env_config.get("getup_low_head_height", 0.6))
        self._target_speed = float(env_config.get("getup_target_up_speed", 0.25))
        self._height_scale = float(env_config.get("getup_height_scale", 6.0))
        self._speed_scale = float(env_config.get("getup_speed_scale", 100.0))
        self._hold_target = float(env_config.get("success_hold_seconds", 5.0))
        self._success_root_height = float(env_config.get("success_root_height", 0.75))
        self._success_upright = float(env_config.get("success_torso_up_dot", 0.8))
        self._success_linear_speed = float(env_config.get("success_max_root_speed", 0.5))
        self._success_angular_speed = float(env_config.get("success_max_root_ang_speed", 1.0))
        self._clearance = float(env_config.get("reset_ground_clearance", 0.005))
        self._joint_noise = float(env_config.get("augmented_joint_noise", 0.10))
        self._early_time = float(env_config.get("augmented_motion_time_max", 0.4))
        self._max_distance = float(env_config.get("max_root_distance", 20.0))
        self._episode_length_cfg = float(env_config["episode_length"])
        if self._hold_target <= 0 or self._episode_length_cfg < self._hold_target:
            raise ValueError("Episode must allow the full standing hold interval.")
        super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)

    def init_gsi_buffer(self, gsi_samples):
        raise ValueError("Getup uses reference-frame/augmented resets; set enable_gsi: False.")

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        n = self.get_num_envs()
        self._head_id = self._kin_char_model.get_body_id("head")
        self._torso_id = self._kin_char_model.get_body_id("torso")
        self._foot_ids = [self._kin_char_model.get_body_id(name)
                          for name in ("right_foot", "left_foot")]
        self._supports = GroundSupports(config["char_file"], self._kin_char_model, self._device)
        self._dof_low, self._dof_high = [
            torch.as_tensor(v, device=self._device, dtype=torch.float32)
            for v in self._engine.get_obj_dof_limits(0, self._get_char_id())]
        self._init_augmented = torch.zeros(n, device=self._device, dtype=torch.bool)
        self._height_shift = torch.zeros(n, device=self._device)
        self._hold = torch.zeros(n, device=self._device)
        self._best_hold = torch.zeros(n, device=self._device)
        self._ever_success = torch.zeros(n, device=self._device, dtype=torch.bool)
        self._first_success = torch.full((n,), -1.0, device=self._device)
        self._standing = torch.zeros(n, device=self._device, dtype=torch.bool)
        self._initial_head = torch.zeros(n, device=self._device)
        # count, ever-success, end-stable, max-hold sum, time-success sum,
        # augmented count/success, numerical failures, initial-head sum.
        self._totals = torch.zeros(2, 9, device=self._device)

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info.update({
            "getup_success": self._ever_success,
            "getup_hold_seconds": self._hold,
            "getup_best_hold_seconds": self._best_hold,
            "getup_first_success_seconds": self._first_success,
            "getup_init_augmented": self._init_augmented,
            "getup_init_head_height": self._initial_head,
            "getup_motion_id": self._motion_ids,
        })

    def set_mode(self, mode):
        if mode != self._mode and mode == base_env.EnvMode.TEST:
            self._totals[mode.value].zero_()
        super().set_mode(mode)

    def _sample_motion_times(self, n):
        # Balance the two known start poses during evaluation. Training keeps
        # the weighted random reference sampler used by the prior dataset.
        if self._mode == base_env.EnvMode.TEST:
            ids = torch.arange(n, device=self._device) % self._motion_lib.get_num_motions()
        else:
            ids = self._motion_lib.sample_motions(n)
        if self._mode == base_env.EnvMode.TEST and self._test_init != "mixed":
            times = torch.zeros(n, device=self._device)
        else:
            times = self._motion_lib.sample_time(ids)
        return ids, times

    def _reset_ref_motion(self, env_ids):
        super()._reset_ref_motion(env_ids)
        n = len(env_ids)
        if self._mode == base_env.EnvMode.TEST and self._test_init == "starts":
            augmented = torch.zeros(n, device=self._device, dtype=torch.bool)
        elif self._mode == base_env.EnvMode.TEST and self._test_init == "augmented":
            augmented = torch.ones(n, device=self._device, dtype=torch.bool)
        else:
            augmented = torch.rand(n, device=self._device) < self._aug_probability
        self._init_augmented[env_ids] = augmented
        if augmented.any():
            self._augment_fallen(env_ids[augmented])
        # Minimal positive lift for references; augmented poses have already
        # been grounded using collision shapes. Never teleport below the floor.
        pos, rot = self._kin_char_model.forward_kinematics(
            self._ref_root_pos[env_ids], self._ref_root_rot[env_ids],
            self._ref_joint_rot[env_ids])
        lift = (self._clearance - self._supports.minimum_height(pos, rot)).clamp_min(0)
        self._height_shift[env_ids] = lift
        self._ref_root_pos[env_ids, 2] += lift
        pos[..., 2] += lift[:, None]
        self._ref_body_pos[env_ids], self._ref_body_rot[env_ids] = pos, rot

    def _augment_fallen(self, env_ids):
        # Paper specifies augmented fallen states but not their distribution.
        # This implementation randomizes early reference poses and orientation,
        # then rejects high (non-fallen) placements. This is an explicit adaptation.
        pending = env_ids
        for _ in range(12):
            if len(pending) == 0:
                break
            n = len(pending)
            motion_ids = self._motion_ids[pending]
            times = torch.rand(n, device=self._device) * torch.minimum(
                self._motion_lib.get_motion_length(motion_ids),
                torch.full((n,), self._early_time, device=self._device))
            pos, rot, _, _, joints, _ = self._motion_lib.calc_motion_frame(motion_ids, times)
            dofs = self._kin_char_model.rot_to_dof(joints)
            dofs = (dofs + self._joint_noise * torch.randn_like(dofs)).clamp(
                self._dof_low, self._dof_high)
            joints = self._kin_char_model.dof_to_rot(dofs)
            euler = (torch.rand(n, 3, device=self._device) * 2 - 1) * math.pi
            rotation = torch_util.euler_xyz_to_quat(euler[:, 0], euler[:, 1], euler[:, 2])
            rot = torch_util.quat_normalize(torch_util.quat_mul(rotation, rot))
            pos = torch.zeros_like(pos)
            bodies, body_rot = self._kin_char_model.forward_kinematics(pos, rot, joints)
            lift = self._clearance - self._supports.minimum_height(bodies, body_rot)
            pos[:, 2] += lift
            bodies[..., 2] += lift[:, None]
            fallen = (bodies[:, self._head_id, 2] < self._low_height) & (pos[:, 2] < 0.5)
            accepted = pending[fallen]
            self._ref_root_pos[accepted] = pos[fallen]
            self._ref_root_rot[accepted] = rot[fallen]
            self._ref_joint_rot[accepted] = joints[fallen]
            self._ref_dof_pos[accepted] = dofs[fallen]
            pending = pending[~fallen]
        # Rejected slots retain their original reference pose: deterministic
        # early-frame fallback, not an unbounded rejection loop.
        if len(pending):
            ids = self._motion_ids[pending]
            pos, rot, _, _, joints, _ = self._motion_lib.calc_motion_frame(
                ids, torch.zeros(len(pending), device=self._device))
            self._ref_root_pos[pending], self._ref_root_rot[pending] = pos, rot
            self._ref_joint_rot[pending] = joints
            self._ref_dof_pos[pending] = self._kin_char_model.rot_to_dof(joints)
        self._ref_root_vel[env_ids] = 0
        self._ref_root_ang_vel[env_ids] = 0
        self._ref_dof_vel[env_ids] = 0
        self._motion_time_offsets[env_ids] = 0

    def _reset_disc_hist(self, env_ids):
        if len(env_ids) == 0:
            return
        data = list(self._fetch_disc_demo_data(
            self._motion_ids[env_ids], self._get_motion_times(env_ids)))
        data[0][..., 2] += self._height_shift[env_ids, None]
        data[6][..., 2] += self._height_shift[env_ids, None, None]
        mask = self._init_augmented[env_ids]
        refs = (self._ref_root_pos, self._ref_root_rot, self._ref_root_vel,
                self._ref_root_ang_vel, self._ref_joint_rot,
                self._ref_dof_vel, self._ref_body_pos)
        names = ("root_pos", "root_rot", "root_vel", "root_ang_vel",
                 "joint_rot", "dof_vel", "body_pos")
        for history, ref, name in zip(data, refs, names):
            history[mask] = ref[env_ids[mask], None]
            getattr(self, "_disc_hist_" + name).fill(env_ids, history)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids):
            self._hold[env_ids] = 0
            self._best_hold[env_ids] = 0
            self._ever_success[env_ids] = False
            self._first_success[env_ids] = -1
            self._standing[env_ids] = False
            self._initial_head[env_ids] = self._engine.get_body_pos(
                self._get_char_id())[env_ids, self._head_id, 2]

    def _compute_obs(self, env_ids=None):
        # Invalid simulator states terminate as FAIL; don't feed NaNs to PPO.
        return torch.nan_to_num(super()._compute_obs(env_ids), nan=0.0, posinf=0.0, neginf=0.0)

    def _update_disc_obs(self, env_ids=None):
        super()._update_disc_obs(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        self._disc_obs_buf[ids] = torch.nan_to_num(
            self._disc_obs_buf[ids], nan=0.0, posinf=0.0, neginf=0.0)

    def _update_reward(self):
        char = self._get_char_id()
        pos = self._engine.get_root_pos(char)
        velocity = self._engine.get_root_vel(char)
        angular = self._engine.get_root_ang_vel(char)
        bodies = self._engine.get_body_pos(char)
        rotations = self._engine.get_body_rot(char)
        head = bodies[:, self._head_id, 2]
        reward = compute_getup_reward(
            head, velocity[:, 2], self._target_height, self._low_height,
            self._target_speed, self._height_scale, self._speed_scale,
            self._reward_mode == "paper_literal")
        self._reward_buf[:] = torch.nan_to_num(reward, nan=0.0)
        up = torch.zeros_like(pos)
        up[:, 2] = 1
        torso_up = torch_util.quat_rotate(rotations[:, self._torso_id], up)[:, 2]
        force = self._engine.get_ground_contact_forces(char)[:, self._foot_ids]
        foot_contact = (force.norm(dim=-1) > 1.0).any(dim=1)
        self._standing[:] = (
            (head >= self._target_height) & (pos[:, 2] >= self._success_root_height)
            & (torso_up >= self._success_upright) & foot_contact
            & (velocity.norm(dim=-1) <= self._success_linear_speed)
            & (angular.norm(dim=-1) <= self._success_angular_speed))
        self._hold[:] = torch.where(self._standing, self._hold + self._engine.get_timestep(), 0)
        self._best_hold[:] = torch.maximum(self._best_hold, self._hold)
        newly = (self._hold >= self._hold_target - 1e-5) & ~self._ever_success
        self._first_success[newly] = self._time_buf[newly]
        self._ever_success |= newly

    def _update_done(self):
        char = self._get_char_id()
        values = (self._engine.get_root_pos(char), self._engine.get_root_rot(char),
                  self._engine.get_root_vel(char), self._engine.get_root_ang_vel(char),
                  self._engine.get_dof_pos(char), self._engine.get_dof_vel(char),
                  self._engine.get_body_pos(char), self._engine.get_body_rot(char))
        invalid = torch.zeros(self.get_num_envs(), device=self._device, dtype=torch.bool)
        for value in values:
            invalid |= ~torch.isfinite(value).flatten(1).all(dim=1)
        pos = values[0]
        escaped = (pos[:, :2].norm(dim=-1) > self._max_distance) | (pos[:, 2] < -1) | (pos[:, 2] > 5)
        self._done_buf[:] = base_env.DoneFlags.NULL.value
        self._done_buf[self._time_buf >= self._episode_length] = base_env.DoneFlags.TIME.value
        self._done_buf[invalid | escaped] = base_env.DoneFlags.FAIL.value
        terminal = self._done_buf != base_env.DoneFlags.NULL.value
        if terminal.any():
            success = self._ever_success[terminal].float()
            aug = self._init_augmented[terminal].float()
            self._totals[self._mode.value] += torch.stack((
                terminal.sum().float(), success.sum(),
                (self._hold[terminal] >= self._hold_target - 1e-5).float().sum(),
                self._best_hold[terminal].sum(),
                (self._first_success[terminal].clamp_min(0) * success).sum(),
                aug.sum(), (aug * success).sum(), invalid[terminal].float().sum(),
                self._initial_head[terminal].sum()))

    def record_diagnostics(self):
        for mode in base_env.EnvMode:
            t = self._totals[mode.value]
            n, successes = t[0].clamp_min(1), t[1].clamp_min(1)
            prefix = "getup_" + mode.name.lower() + "_"
            self._diagnostics.update({
                prefix + "episodes": t[0].clone(),
                prefix + "success_rate": (t[1] / n).clone(),
                prefix + "end_rate": (t[2] / n).clone(),
                prefix + "hold_s": (t[3] / n).clone(),
                prefix + "success_s": (t[4] / successes).clone(),
                prefix + "aug_episodes": t[5].clone(),
                prefix + "aug_success": (t[6] / t[5].clamp_min(1)).clone(),
                prefix + "nan_failures": t[7].clone(),
            })
        return super().record_diagnostics()

    def clear_train_diagnostics(self):
        # Called after a saved log/checkpoint, not on every console iteration.
        # Otherwise sparse log.txt rows often miss all completed episodes.
        self._totals[base_env.EnvMode.TRAIN.value].zero_()
