"""Exact parallel first-episode evaluation for pure-SMP G1 spinkick."""
import argparse
import contextlib
import csv
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
from tools.g1_spinkick.workflow import ENV, ENGINE, AGENT, PRIOR_DIR, audit_assets, check_prior
from tools.getup.common import read_yaml, provenance, sha256, write_json
from tools.getup.evaluate_getup import Tee
from tools.getup.diagnose_getup import score_windows


def standing_candidate(env, count=64):
    from envs.amp_env import compute_disc_obs
    n, h = count, env._num_disc_obs_steps
    pos, rot = env._init_root_pos.repeat(n, 1), env._init_root_rot.repeat(n, 1)
    joints = env._kin_char_model.dof_to_rot(env._init_dof_pos.repeat(n, 1))
    bodies, _ = env._kin_char_model.forward_kinematics(pos, rot, joints)
    expand = lambda x: x[:, None].expand(-1, h, *x.shape[1:])
    vel = torch.zeros(n, h, 3, device=env._device)
    dv = torch.zeros(n, h, env._kin_char_model.get_dof_size(), device=env._device)
    return compute_disc_obs(pos, rot, expand(pos), expand(rot), vel, vel, expand(joints), dv,
                            expand(bodies[:, env._key_body_ids]), env._global_obs,
                            env._root_height_obs, env._disc_dof_vel_obs)


@torch.no_grad()
def run(args, out):
    from envs.smp_env import SMPEnv
    from envs.base_env import DoneFlags
    from learning import agent_builder
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util, util
    audit = audit_assets(args.device)
    check_prior(args.prior_dir)
    env_cfg, agent_cfg = read_yaml(ENV), read_yaml(AGENT)
    env_cfg["rand_reset"] = args.init == "random"
    agent_cfg.update(smp_prior_cfg=str(Path(args.prior_dir) / "diffusion_config.yaml"),
                     smp_prior_model=str(Path(args.prior_dir) / "model.pt"))
    for name, cfg in (("env_config.yaml", env_cfg), ("engine_config.yaml", read_yaml(ENGINE)),
                      ("agent_config.yaml", agent_cfg)):
        (out / name).write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    util.set_rand_seed(args.seed)
    env = agent = None
    started = time.monotonic()
    mp_util.init(0, 1, args.device, 0)
    try:
        env = resource_util.construct(SMPEnv, env_config=env_cfg, engine_config=read_yaml(ENGINE),
                                      num_envs=args.episodes, device=args.device, visualize=False)
        agent = agent_builder.build_agent(str(out / "agent_config.yaml"), env, args.device)
        agent.load(args.model_file)
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        obs, info = env.reset()
        n, char, dt = args.episodes, env._get_char_id(), env._engine.get_timestep()
        seen = torch.zeros(n, dtype=torch.bool, device=args.device)
        max_foot = torch.zeros(n, device=args.device)
        angle_speed = torch.zeros(n, device=args.device)
        dof_speed = torch.zeros(n, device=args.device)
        steps = torch.zeros(n, device=args.device)
        windows, slot_ids, rows = [], [], []
        for step in range(math.ceil(env_cfg["episode_length"] / dt) + 3):
            active = ~seen
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            if not bool(torch.isfinite(obs).all() & torch.isfinite(info["disc_obs"]).all()):
                raise RuntimeError("Non-finite policy/prior observations; stop and report console.txt.")
            foot = env._engine.get_body_pos(char)[:, env._key_body_ids[:2], 2].amax(dim=1)
            max_foot = torch.where(active, torch.maximum(max_foot, foot), max_foot)
            angle_speed += env._engine.get_root_ang_vel(char)[:, 2].abs() * active
            dof_speed += env._engine.get_dof_vel(char).abs().mean(dim=1) * active
            steps += active
            if step % 3 == 0:
                ids = active.nonzero(as_tuple=False).flatten()
                windows.append(info["disc_obs"][ids].cpu().clone())
                slot_ids.append(ids.cpu())
            terminal = done != DoneFlags.NULL.value
            for i in (terminal & active).nonzero(as_tuple=False).flatten().tolist():
                rows.append({"env_id": i, "episode_seconds": float(env.get_env_time()[i]),
                             "done": DoneFlags(int(done[i])).name,
                             "max_foot_body_height_m": float(max_foot[i]),
                             "mean_abs_root_yaw_speed": float(angle_speed[i] / steps[i]),
                             "mean_abs_joint_speed": float(dof_speed[i] / steps[i])})
            seen |= terminal
            if bool(seen.all()):
                break
            obs, info = env.reset(terminal.nonzero(as_tuple=False).flatten())
        if len(rows) != n:
            raise RuntimeError("Incomplete first-episode evaluation; no finished summary emitted.")
        print("Scoring 10-Hz trajectory windows with the saved checkpoint normalizer...", flush=True)
        raw, normalized, reward = score_windows(agent, torch.cat(windows), args.seed + 1000, repeats=2)
        if not bool(torch.isfinite(raw).all() & torch.isfinite(reward).all()):
            raise RuntimeError("Non-finite SDS/reward diagnostic.")
        ids = torch.cat(slot_ids)
        for row in rows:
            mask = ids == row["env_id"]
            row["smp_reward_mean"] = float(reward[mask].mean())
            row["raw_sds_mean"] = float(raw[mask].mean())
        control_data = {
            "reference_windows": env._compute_disc_obs_demo(
                torch.zeros(64, dtype=torch.long, device=args.device),
                torch.linspace(0, float(env._motion_lib.get_motion_length(torch.tensor([0], device=args.device))[0]),
                               64, device=args.device)),
            "static_init_candidate_not_verified_balance": standing_candidate(env),
        }
        controls = {}
        for name, window in control_data.items():
            r, norm, smp = score_windows(agent, window, args.seed + 2000, repeats=4)
            controls[name] = {"raw_sds_mean": float(r.mean()), "normalized_sds_mean": float(norm.mean()),
                              "smp_reward_mean": float(smp.mean())}
        rows.sort(key=lambda r: r["env_id"])
        with (out / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = provenance({"policy": args.model_file, "policy_sha256": sha256(args.model_file),
            "seed": args.seed, "initialization": args.init, "episodes": n, "parallel_envs": n,
            "mean_episode_seconds": sum(r["episode_seconds"] for r in rows) / n,
            "timeout_rate": sum(r["done"] == "TIME" for r in rows) / n,
            "failures": sum(r["done"] == "FAIL" for r in rows),
            "mean_smp_reward": float(reward.mean()), "raw_sds_mean": float(raw.mean()),
            "mean_episode_smp_reward": sum(r["smp_reward_mean"] for r in rows) / n,
            "mean_max_foot_body_height_m": sum(r["max_foot_body_height_m"] for r in rows) / n,
            "mean_abs_root_yaw_speed": sum(r["mean_abs_root_yaw_speed"] for r in rows) / n,
            "mean_abs_joint_speed": sum(r["mean_abs_joint_speed"] for r in rows) / n,
            "sds_normalizer_count": agent._sds_normalizer.get_count().cpu().tolist(),
            "sds_normalizer_mean_abs": agent._sds_normalizer.get_abs_mean().cpu().tolist(),
            "controls": controls, "motion_audit": audit, "elapsed_seconds": time.monotonic() - started,
            "notes": ["Timeout rate is survival, not kick success rate.",
                      "mean_smp_reward is window-weighted; mean_episode_smp_reward weights each episode equally.",
                      "SMP scores are repeated-noise estimates on windows sampled at 10 Hz.",
                      "Starts uses the first reference frame and original velocities, not a static standing initialization.",
                      "Visual inspection is required to verify an actual spinkick; repeated fixed starts are not independent cases."]})
        write_json(out / "evaluation.json", result)
        (out / "summary.txt").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Completed. Send: " + str(out / "summary.txt"), flush=True)
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None, mp_util.shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prior_dir", default=None)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--init", choices=("starts", "random"), default="starts")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.episodes < 1:
        parser.error("--episodes must be positive.")
    if not Path(args.model_file).is_file():
        raise FileNotFoundError(args.model_file)
    if args.prior_dir is None:
        model_dir = Path(args.model_file).parent
        for candidate in (model_dir / "agent_config.yaml", model_dir.parent / "agent_config.yaml"):
            if candidate.is_file():
                args.prior_dir = str(Path(read_yaml(candidate)["smp_prior_cfg"]).parent)
                break
        else:
            args.prior_dir = PRIOR_DIR
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, out)
            except KeyboardInterrupt:
                print("Evaluation interrupted; no completed evaluation claimed.")
                raise SystemExit(130)
            except Exception:
                import traceback
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
