"""Read-only 10-Hz joint/geometry traces; thresholds are diagnostics, not fall labels."""
import csv
import json
from pathlib import Path
import torch


def joint_statistics(q, velocity, target, lower, upper, requested=None):
    """Rows are active first-episode samples, columns follow the model's DoF order."""
    outside = (target < lower - 1e-4) | (target > upper + 1e-4)
    near = (q < lower + 0.03) | (q > upper - 0.03)
    requested = target if requested is None else requested
    requested_outside = (requested < lower - 1e-4) | (requested > upper + 1e-4)
    return {"q_min": q.amin(0), "q_max": q.amax(0), "q_std": q.std(0, unbiased=False),
            "target_std": target.std(0, unbiased=False), "mean_abs_velocity": velocity.abs().mean(0),
            "mean_abs_tracking_error": (target-q).abs().mean(0),
            "command_outside_physical_limits_fraction": outside.float().mean(0),
            "requested_outside_physical_limits_fraction": requested_outside.float().mean(0),
            "near_physical_limit_fraction": near.float().mean(0)}


def temporal_target_rate(state, target):
    """Within-slot motion rate; different initial poses must not masquerade as target motion."""
    order = torch.argsort(state[:, 0] * (state[:, 1].max() + 1) + state[:, 1])
    s, t = state[order], target[order]
    dt = s[1:, 1] - s[:-1, 1]
    same = (s[1:, 0] == s[:-1, 0]) & (dt > 0)
    if not bool(same.any()):
        return None
    return ((t[1:] - t[:-1]).abs()[same] / dt[same, None]).mean(0)


class JointDiagnostics:
    def __init__(self, env):
        from util import torch_util
        self.quat_rotate = torch_util.quat_rotate
        self.char = env._get_char_id()
        self.names = [""] * env._kin_char_model.get_dof_size()
        for i in range(1, env._kin_char_model.get_num_joints()):
            joint = env._kin_char_model.get_joint(i)
            if joint.get_dof_dim() == 1:
                self.names[joint.dof_idx] = joint.name
            elif joint.get_dof_dim() != 0:
                raise ValueError("G1 diagnostics require one-dimensional actuated joints.")
        if any(not name for name in self.names):
            raise ValueError("Incomplete G1 joint-name mapping.")
        low, high = env._engine.get_obj_dof_limits(0, self.char)
        self.low, self.high = torch.as_tensor(low).cpu(), torch.as_tensor(high).cpu()
        self.head = env._kin_char_model.get_body_id("head_link")
        self.batches = []

    @torch.no_grad()
    def record(self, env, active, action, launches):
        ids = active.nonzero(as_tuple=False).flatten()
        q = env._engine.get_dof_pos(self.char)[ids]
        velocity = env._engine.get_dof_vel(self.char)[ids]
        # CharEnv sends these expanded action bounds directly as position-PD targets.
        requested = action.clamp(env._action_bound_low, env._action_bound_high)[ids]
        target = env._clip_action(action)[ids]
        rotation = env._engine.get_root_rot(self.char)[ids]
        up = torch.zeros(len(ids), 3, device=q.device)
        up[:, 2] = 1
        up_z = self.quat_rotate(rotation, up)[:, 2]
        ground = env._engine.get_ground_contact_forces(self.char)[ids].clone()
        ground[:, env._contact_body_ids, :] = 0
        state = torch.stack([ids.float(), env.get_env_time()[ids],
            env._engine.get_root_pos(self.char)[ids, 2],
            env._engine.get_body_pos(self.char)[ids, self.head, 2], up_z,
            ground.abs().flatten(1).amax(1), launches[ids].float()], dim=1)
        for tensor in (state, q, velocity, target, requested):
            if not bool(torch.isfinite(tensor).all()):
                raise RuntimeError("Non-finite joint/geometry trace; inspect console.txt.")
        self.batches.append(tuple(t.detach().cpu().clone() for t in (state, q, velocity, target, requested)))

    def save(self, out):
        out = Path(out)
        state, q, velocity, target, requested = [torch.cat([b[i] for b in self.batches]) for i in range(5)]
        partitions = {"all": torch.ones(len(state), dtype=torch.bool),
                      "before_first_launch": state[:, 6] == 0,
                      "after_first_launch": state[:, 6] > 0}
        result = {"sample_hz": 10, "includes_extra_terminal_samples": True, "threshold_notes":
            "Root z<0.45m and root up.z<0.5 are low/tilted flags, not verified falls. "
            "Near-limit margin=0.03rad. Action bounds intentionally exceed physical joint limits. "
            "Regular 10-Hz samples plus terminal steps; fractions weight samples, not episodes. "
            "Partition is by first projectile LAUNCH, not the unknown first dodge/contact time.",
            "partitions": {}}
        text = [result["threshold_notes"]]
        for name, mask in partitions.items():
            if not bool(mask.any()):
                continue
            s = state[mask]
            stats = joint_statistics(q[mask], velocity[mask], target[mask], self.low, self.high, requested[mask])
            target_rate = temporal_target_rate(s, target[mask])
            joints = [{"joint": joint, "physical_lower": float(self.low[i]), "physical_upper": float(self.high[i]),
                       "mean_abs_target_rate_rad_s": float(target_rate[i]) if target_rate is not None else None,
                       **{k: float(v[i]) for k, v in stats.items()}} for i, joint in enumerate(self.names)]
            geometry = {"sample_count": len(s), "root_height_min": float(s[:, 2].min()),
                "root_height_median": float(s[:, 2].median()), "head_height_min": float(s[:, 3].min()),
                "root_up_z_min": float(s[:, 4].min()),
                "low_root_fraction": float((s[:, 2] < 0.45).float().mean()),
                "tilted_root_fraction": float((s[:, 4] < 0.5).float().mean()),
                "nonfoot_contact_fraction": float((s[:, 5] > 0.1).float().mean())}
            result["partitions"][name] = {"geometry": geometry, "joints": joints}
            text.append("\n" + name + ": " + json.dumps(geometry))
            text.append("leg_joint | actual_min/max_rad | target_rate_rad/s | velocity_rad/s | tracking_error_rad | target_outside_% | near_limit_% | requested_outside_%")
            for row in joints:
                if any(key in row["joint"] for key in ("hip", "knee", "ankle")):
                    rate = row["mean_abs_target_rate_rad_s"]
                    rate_text = f"{rate:.3f}" if rate is not None else "no_time_pairs"
                    text.append(f"{row['joint']} | {row['q_min']:.3f}/{row['q_max']:.3f} | "
                        f"{rate_text} | {row['mean_abs_velocity']:.3f} | "
                        f"{row['mean_abs_tracking_error']:.3f} | "
                        f"{100*row['command_outside_physical_limits_fraction']:.1f} | "
                        f"{100*row['near_physical_limit_fraction']:.1f} | "
                        f"{100*row['requested_outside_physical_limits_fraction']:.1f}")
        with (out / "state_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "time_seconds", "root_height", "head_height", "root_up_z",
                             "nonfoot_ground_force_max_component", "projectile_launches"])
            writer.writerows(state.tolist())
        with (out / "joint_trace.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["env_id", "time_seconds", "launches", "joint", "actual_rad", "target_rad",
                             "velocity_rad_s", "physical_lower", "physical_upper", "requested_rad"])
            for s, qs, vs, ts, rs in zip(state.tolist(), q.tolist(), velocity.tolist(), target.tolist(), requested.tolist()):
                writer.writerows([int(s[0]), s[1], int(s[6]), joint, qs[i], ts[i], vs[i],
                                  float(self.low[i]), float(self.high[i]), rs[i]] for i, joint in enumerate(self.names))
        (out / "diagnostics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        (out / "diagnostics.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
        return {name: data["geometry"] for name, data in result["partitions"].items()}
