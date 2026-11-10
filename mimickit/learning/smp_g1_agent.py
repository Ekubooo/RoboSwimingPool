"""SMP with G1 position-target protection; task rewards remain configurable."""
import torch
from learning.smp_agent import SMPAgent


def physical_target_loss(target, lower, upper, scale):
    excess = (lower-target).clamp_min(0) + (target-upper).clamp_min(0)
    return (excess / scale).square().mean()


def joint_limit_pressure(q, lower, upper, margin):
    # Saturated per-joint cost: constant holding at a stop keeps paying each control step.
    proximity = torch.maximum(lower + margin - q, q - (upper - margin)).clamp_min(0)
    return (proximity / margin).square().clamp_max(1).mean(dim=-1)


class SMPG1Agent(SMPAgent):
    def __init__(self, config, env, device):
        super().__init__(config, env, device)
        if not getattr(env, "_clip_position_targets_to_physical_limits", False):
            raise ValueError("Use a G1 position-target-limit environment.")
        low, high = env._engine.get_obj_dof_limits(0, env._get_char_id())
        self.register_buffer("_physical_lower", torch.as_tensor(low, device=device).clone())
        self.register_buffer("_physical_upper", torch.as_tensor(high, device=device).clone())
        model = env._kin_char_model
        leg_ids = [model.get_joint(i).dof_idx for i in range(1, model.get_num_joints())
                   if model.get_joint(i).get_dof_dim() == 1
                   and any(part in model.get_joint(i).name for part in ("hip", "knee", "ankle"))]
        if model.get_dof_size() != 29 or len(leg_ids) != 12:
            raise ValueError("Expected G1 29DoF with 12 leg joints.")
        self.register_buffer("_leg_dof_ids", torch.tensor(leg_ids, device=device))

    def _load_params(self, config):
        super()._load_params(config)
        self._physical_target_weight = float(config["physical_target_weight"])
        self._physical_target_scale = float(config["physical_target_scale_rad"])
        self._limit_pressure_weight = float(config["limit_pressure_weight"])
        self._limit_margin = float(config["limit_margin_rad"])
        if self._physical_target_weight < 0 or self._limit_pressure_weight < 0:
            raise ValueError("Repair weights must be nonnegative.")
        if self._physical_target_scale <= 0 or self._limit_margin <= 0:
            raise ValueError("Repair angular scales must be positive.")

    def _compute_actor_loss(self, batch):
        info = super()._compute_actor_loss(batch)
        obs = self._obs_norm.normalize(batch["obs"][(batch["rand_action_mask"] == 1.0)])
        distribution = self._model.eval_actor(obs)
        target = self._a_norm.unnormalize(distribution.mode)
        penalty = physical_target_loss(target, self._env._physical_target_low,
                                       self._env._physical_target_high, self._physical_target_scale)
        info["actor_loss"] = info["actor_loss"] + self._physical_target_weight * penalty
        info["physical_target_loss"] = penalty.detach()
        return info

    def _record_data_post_step(self, next_obs, reward, done, next_info):
        super()._record_data_post_step(next_obs, reward, done, next_info)
        self._exp_buffer.record("g1_dof_pos", self._env._engine.get_dof_pos(self._env._get_char_id()))

    def _compute_rewards(self):
        info = super()._compute_rewards()
        q = self._exp_buffer.get_data_flat("g1_dof_pos")[:, self._leg_dof_ids]
        cost = joint_limit_pressure(q, self._physical_lower[self._leg_dof_ids],
                                   self._physical_upper[self._leg_dof_ids], self._limit_margin)
        reward = self._exp_buffer.get_data_flat("reward")
        self._exp_buffer.set_data_flat("reward", reward - self._limit_pressure_weight * cost)
        info["g1_limit_pressure_mean"] = cost.mean()
        # Keep below the legacy text logger's 25-character column width.
        info["g1_pressure_cost"] = self._limit_pressure_weight * cost.mean()
        return info
