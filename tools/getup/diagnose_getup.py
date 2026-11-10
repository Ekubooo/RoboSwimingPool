"""Read-only Getup checkpoint comparison: motion, reward terms and frozen SDS.

Each policy runs one first episode per reference start (2 environments). SDS is
scored AFTER physics, with saved checkpoint normalizers and repeated noise.
No optimizer, reward configuration, normalizer or checkpoint is updated.
"""
import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import torch
import yaml
from tools.getup.common import AGENT_CONFIG, ENGINE_CONFIG, ENV_CONFIG, preflight, provenance, sha256, write_json
from tools.getup.evaluate_getup import Tee


def reward_components(head, vz, env):
    h, v = env._target_height - head, env._target_speed - vz
    clamp = torch.clamp_max if env._reward_mode == "paper_literal" else torch.clamp_min
    height = torch.exp(-env._height_scale * clamp(h, 0).square())
    velocity = torch.where(head < env._low_height,
                           torch.exp(-env._speed_scale * clamp(v, 0).square()),
                           torch.ones_like(head))
    return height, velocity


@torch.no_grad()
def score_windows(agent, windows, seed, repeats=4, batch_size=256):
    """Mean of per-noise exp rewards, NOT exp of the averaged SDS error."""
    device = torch.device(agent._device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    raw_sum = norm_sum = reward_sum = None
    count_before = agent._sds_normalizer.get_count().clone()
    with torch.random.fork_rng(devices=devices):
        for repeat in range(repeats):
            torch.random.default_generator.manual_seed(seed + repeat)
            if device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed + repeat)
            raw_parts = []
            for chunk in windows.split(batch_size):
                x = chunk.to(device).reshape(len(chunk), agent._env._num_disc_obs_steps, -1)
                x = agent._prior_model.normalize(x).reshape(len(chunk), -1)
                raw_parts.append(agent._prior_model.ESM_SDS_loss(x, t_lst=agent._diffusion_steps))
            raw = torch.cat(raw_parts)
            normalized = agent._sds_normalizer.normalize(raw)
            reward = torch.exp(-normalized.mean(dim=1) * agent._sds_loss_scale) * agent._smp_reward_scale
            raw_sum = raw if raw_sum is None else raw_sum + raw
            norm_sum = normalized if norm_sum is None else norm_sum + normalized
            reward_sum = reward if reward_sum is None else reward_sum + reward
    if not torch.equal(count_before, agent._sds_normalizer.get_count()):
        raise RuntimeError("Diagnostic unexpectedly changed SDS normalizer count.")
    return (raw_sum / repeats).cpu(), (norm_sum / repeats).cpu(), (reward_sum / repeats).cpu()


def state_fingerprint(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode())
        digest.update(str((value.dtype, tuple(value.shape))).encode())
        digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def controls(env):
    """Pure reference windows and static candidates, not verified balance states."""
    from envs.amp_env import compute_disc_obs
    n, device = 32, env._device
    ids = torch.arange(2, device=device).repeat_interleave(n)
    phase = torch.linspace(0, 1, n, device=device).repeat(2)
    references = env._compute_disc_obs_demo(ids, phase * env._motion_lib.get_motion_length(ids))
    result = {"reference_windows_unlifted": references}
    def static(pos, rot, joints):
        bodies, body_rot = env._kin_char_model.forward_kinematics(pos, rot, joints)
        lift = (env._clearance - env._supports.minimum_height(bodies, body_rot)).clamp_min(0)
        pos, bodies = pos.clone(), bodies.clone()
        pos[:, 2] += lift
        bodies[..., 2] += lift[:, None]
        h = env._num_disc_obs_steps
        expand = lambda x: x[:, None].expand(-1, h, *x.shape[1:])
        velocity = torch.zeros(len(pos), h, 3, device=device)
        dof_velocity = torch.zeros(len(pos), h, env._kin_char_model.get_dof_size(), device=device)
        return compute_disc_obs(pos, rot, expand(pos), expand(rot), velocity, velocity,
                                expand(joints), dof_velocity, expand(bodies[:, env._key_body_ids]),
                                env._global_obs, env._root_height_obs, env._disc_dof_vel_obs)
    for mid in range(2):
        pos, rot, _, _, joints, _ = env._motion_lib.calc_motion_frame(
            torch.tensor([mid], device=device), torch.zeros(1, device=device))
        name = Path(env._motion_lib.get_motion_file(mid)).stem.split("_")[-1]
        result["static_start_" + name] = static(pos, rot, joints).repeat(32, 1)
    result["static_standing_init_candidate"] = static(
        env._init_root_pos[None], env._init_root_rot[None],
        env._kin_char_model.dof_to_rot(env._init_dof_pos[None])).repeat(32, 1)
    return result


def summarize_rows(rows):
    if not rows:
        return {"steps": 0}
    def mean(key): return round(sum(row[key] for row in rows) / len(rows), 6)
    keys = ("head_z", "root_z", "root_vz", "root_vz_fd", "torso_up", "right_foot_force",
            "left_foot_force", "dof_speed", "action_change", "dof_change", "height_reward",
            "velocity_reward", "task_reward", "smp_reward", "weighted_task", "weighted_smp",
            "combined_reward", "vz_fd_abs_error", "standing")
    return {"steps": len(rows), **{key + "_mean": mean(key) for key in keys},
            "max_head_z": round(max(r["head_z"] for r in rows), 6),
            "max_root_z": round(max(r["root_z"] for r in rows), 6),
            "best_hold_seconds": round(max(r["hold_seconds"] for r in rows), 6),
            "success": any(r["success"] for r in rows),
            "net_root_z_change": round(sum(r["root_vz_fd"] * r["dt"] for r in rows), 6),
            "integrated_reported_vz": round(sum(r["root_vz"] * r["dt"] for r in rows), 6)}


@torch.no_grad()
def diagnose_policy(args, model_file, evidence, out):
    from envs.base_env import DoneFlags
    from envs.task_getup_env import TaskGetupEnv
    from learning import agent_builder
    from learning.base_agent import AgentMode
    from util import resource_util, util
    import util.torch_util as torch_util
    cfg, _, engine, _ = evidence
    cfg = {**cfg, "getup_test_init": "starts"}
    env = agent = None
    util.set_rand_seed(args.seed)
    try:
        env = resource_util.construct(TaskGetupEnv, env_config=cfg, engine_config=engine,
                                      num_envs=2, device=args.device, visualize=False)
        agent = agent_builder.build_agent(args.agent_config, env, args.device)
        agent.load(model_file)
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        obs, info = env.reset()
        char, dt = env._get_char_id(), env._engine.get_timestep()
        previous_z = env._engine.get_root_pos(char)[:, 2].clone()
        previous_dof = env._engine.get_dof_pos(char).clone()
        previous_action = None
        seen = torch.zeros(2, dtype=torch.bool, device=args.device)
        rows, windows = [], []
        names = [Path(env._motion_lib.get_motion_file(i)).stem.split("_")[-1] for i in range(2)]
        for step in range(math.ceil(cfg["episode_length"] / dt) + 3):
            active = ~seen
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            windows.append(info["disc_obs"].detach().cpu().clone())
            pos, vel = env._engine.get_root_pos(char), env._engine.get_root_vel(char)
            body, rot = env._engine.get_body_pos(char), env._engine.get_body_rot(char)
            dof = env._engine.get_dof_pos(char)
            force = env._engine.get_ground_contact_forces(char)[:, env._foot_ids].norm(dim=-1)
            up = torch.zeros_like(pos)
            up[:, 2] = 1
            torso = torch_util.quat_rotate(rot[:, env._torso_id], up)[:, 2]
            height, speed = reward_components(body[:, env._head_id, 2], vel[:, 2], env)
            if not torch.allclose(reward, .3 * height + .7 * speed, atol=1e-5):
                raise RuntimeError("Task reward decomposition mismatch or non-finite physical state.")
            fd = (pos[:, 2] - previous_z) / dt
            action_change = (action - previous_action).abs().mean(dim=1) if previous_action is not None else torch.zeros_like(fd)
            values = torch.stack((body[:, env._head_id, 2], pos[:, 2], vel[:, 2], fd, torso,
                                  force[:, 0], force[:, 1], env._engine.get_dof_vel(char).abs().mean(dim=1),
                                  action_change, (dof - previous_dof).abs().mean(dim=1), height, speed, reward,
                                  (vel[:, 2] - fd).abs(), env._standing.float(), env._hold,
                                  env._ever_success.float(), env._time_buf), dim=1).cpu().tolist()
            fields = ("head_z", "root_z", "root_vz", "root_vz_fd", "torso_up", "right_foot_force",
                      "left_foot_force", "dof_speed", "action_change", "dof_change", "height_reward",
                      "velocity_reward", "task_reward", "vz_fd_abs_error", "standing", "hold_seconds", "success", "time")
            for index, values_row in enumerate(values):
                rows.append({"step": step, "env_id": index, "motion": names[index],
                             "valid": int(active[index]), "done": int(done[index]), "dt": dt,
                             **dict(zip(fields, values_row))})
            seen |= done != DoneFlags.NULL.value
            if bool(seen.all()):
                break
            terminal_ids = (done != DoneFlags.NULL.value).nonzero(as_tuple=False).flatten()
            obs, info = env.reset(terminal_ids)
            previous_z = env._engine.get_root_pos(char)[:, 2].clone()
            previous_dof = env._engine.get_dof_pos(char).clone()
            previous_action = action.clone()
        if not bool(seen.all()):
            raise RuntimeError("Both first episodes must terminate before diagnostic completion.")
        print("Scoring captured windows using frozen checkpoint normalizers...", flush=True)
        raw, norm, smp = score_windows(agent, torch.cat(windows), args.seed + 1000, args.repeats)
        for index, row in enumerate(rows):
            row.update(smp_reward=float(smp[index]),
                       weighted_task=agent._task_reward_weight * row["task_reward"],
                       weighted_smp=agent._smp_reward_weight * float(smp[index]))
            row["combined_reward"] = row["weighted_task"] + row["weighted_smp"]
            for ti, timestep in enumerate(agent._diffusion_steps):
                row[f"raw_sds_t{timestep}"] = float(raw[index, ti])
                row[f"normalized_sds_t{timestep}"] = float(norm[index, ti])
        valid_rows = [row for row in rows if row["valid"]]
        if any(not math.isfinite(value) for row in valid_rows for value in row.values() if isinstance(value, (float, int))):
            raise RuntimeError("Non-finite diagnostic values; no completed summary emitted.")
        control_scores = {}
        for name, control in controls(env).items():
            cr, cn, cs = score_windows(agent, control, args.seed + 2000, args.repeats)
            control_scores[name] = {"raw_sds_mean": float(cr.mean()), "normalized_sds_mean": float(cn.mean()),
                                    "smp_reward_mean": float(cs.mean())}
        with (out / "trajectory.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return {"checkpoint": model_file, "policy_sha256": sha256(model_file),
                "loaded_prior_state_sha256": state_fingerprint(agent._prior_model),
                "sds_normalizer_count": agent._sds_normalizer.get_count().cpu().tolist(),
                "sds_normalizer_mean_abs": agent._sds_normalizer.get_abs_mean().cpu().tolist(),
                "controls": control_scores,
                "motions": {name: {"whole_episode": summarize_rows([r for r in valid_rows if r["motion"] == name]),
                                    "last_5_seconds": summarize_rows([r for r in valid_rows if r["motion"] == name
                                                                      and r["time"] > cfg["episode_length"] - 5])}
                            for name in names}}
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_files", nargs="+", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--agent_config", default=AGENT_CONFIG)
    parser.add_argument("--env_config", default=ENV_CONFIG)
    parser.add_argument("--engine_config", default=ENGINE_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=4)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.repeats < 1:
        parser.error("--repeats must be positive.")
    evidence = preflight(args.env_config, args.agent_config, args.engine_config)
    for path in args.model_files:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    from util import mp_util
    from datetime import datetime
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            mp_util.init(0, 1, args.device, 0)
            try:
                start = time.monotonic()
                policies = []
                for index, model_file in enumerate(args.model_files):
                    folder = out / f"policy_{index:02d}"
                    folder.mkdir()
                    print(f"Diagnosing {model_file}", flush=True)
                    policies.append(diagnose_policy(args, model_file, evidence, folder))
                result = provenance({**evidence[3], "timestamp": datetime.now().astimezone().isoformat(),
                                     "seed": args.seed, "noise_repeats": args.repeats,
                                     "initialization": "starts", "parallel_envs": 2,
                                     "reward_mode": evidence[0]["getup_reward_mode"],
                                     "elapsed_seconds": time.monotonic() - start,
                                     "notes": ["No training or checkpoint/normalizer updates.",
                                               "SMP is a repeated-noise estimate using each checkpoint's saved normalization.",
                                               "Finite-difference velocity is a step average; reported velocity is instantaneous.",
                                               "Static controls are kinematic candidates, not verified physical balance."],
                                     "policies": policies})
                for name, cfg in zip(("env_config.yaml", "agent_config.yaml", "engine_config.yaml"), evidence[:3]):
                    if name == "env_config.yaml":
                        cfg = {**cfg, "getup_test_init": "starts"}
                    (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
                write_json(out / "summary.json", result)
                (out / "summary.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                print("Completed. Send: " + str(out / "summary.txt"), flush=True)
            except KeyboardInterrupt:
                print("Diagnostic interrupted; no complete comparison is claimed.")
                raise SystemExit(130)
            except Exception:
                import traceback
                traceback.print_exc()
                raise SystemExit(1)
            finally:
                mp_util.shutdown()


if __name__ == "__main__":
    main()
