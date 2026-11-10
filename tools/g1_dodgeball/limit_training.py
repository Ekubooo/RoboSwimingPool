"""Paper reward + bounded G1 PD targets; explicit user-selected long budget."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import yaml
from tools.g1_dodgeball.workflow import AGENT, ENV, ENGINE, PRIOR_DIR, audit_assets, check_prior
from tools.g1_dodgeball.style_balance import validate_log
from tools.g1_spinkick.workflow import capture
from tools.getup.common import read_yaml, provenance, sha256, write_json

RECIPE = "paper_targets_v1"
BUDGETS = {"smoke": (8, 3), "expanded": (2048, 8001), "full": (2048, 20402)}


def build_configs():
    env, agent = read_yaml(ENV), read_yaml(AGENT)
    env.update(dodge_reward_mode="paper_v3", clip_position_targets_to_physical_limits=True,
               position_target_margin_rad=0.03,
               position_target_lower_overrides={"left_knee_joint": 0.03, "right_knee_joint": 0.03})
    agent.update(agent_name="SMPG1Dodgeball", task_reward_weight=0.6,
                 smp_reward_weight=1.2, sds_loss_scale=4,
                 physical_target_weight=1.0, physical_target_scale_rad=0.25,
                 limit_pressure_weight=0.0, limit_margin_rad=0.03)
    return env, agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=BUDGETS, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    envs, updates = BUDGETS[args.stage]
    env, agent = build_configs()
    if args.stage == "smoke":
        agent["test_episodes"] = 0
    out = Path(args.out_dir)
    samples = envs * agent["steps_per_iter"] * updates
    command = [sys.executable, "-u", "-B", "mimickit/run.py", "--arg_file", "args/smp_dodgeball_g1_args.txt",
               "--mode", "train", "--engine_config", ENGINE, "--env_config", str(out / "env_runtime.yaml"),
               "--agent_config", str(out / "agent_runtime.yaml"), "--num_envs", str(envs),
               "--max_samples", str(samples), "--out_dir", str(out), "--rand_seed", "123", "--devices", "cuda:0"]
    plan = {"recipe": RECIPE, "stage": args.stage, "num_envs": envs, "updates": updates,
            "max_samples": samples, "last_iteration": updates - 1, "seed": 123,
            "initialization": "fresh RL weights/normalizers; frozen existing 200k prior",
            "paper": "https://arxiv.org/html/2512.03028v3#A4.SS4",
            "deviations": ["G1 morphology/prior", "12-15 m/s projectiles; existing 1-4 s relaunch interval",
                           "bounded PD targets and actor target regularizer; original G1 gains/effort limits",
                           "2048 environments; retained repository optimizer/network/GSI"],
            "env_config": env, "agent_config": agent, "command": command}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required; long stages run on remote A5000.")
    audit = audit_assets("cuda:0")
    prior = check_prior(PRIOR_DIR, audit)
    out.mkdir(parents=True, exist_ok=True)
    for name, config in (("env_runtime.yaml", env), ("agent_runtime.yaml", agent)):
        (out / name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    write_json(out / "motion_audit.json", audit)
    manifest = provenance({**plan, "status": "running", "gpu": torch.cuda.get_device_name(0),
                           "prior_ema_step": prior["verified_ema_step"],
                           "prior_sha256": sha256(Path(PRIOR_DIR) / "model.pt")})
    write_json(out / "run_manifest.json", manifest)
    code = 1
    try:
        code = capture(command, out)
        if code:
            raise RuntimeError(f"Training exit={code}")
        code = 2
        validation = validate_log(out / "log.txt", samples, updates)
        console = (out / "console.txt").read_text(encoding="utf-8", errors="replace")
        if any("overflow" in line.lower() and any(k in line.lower() for k in ("nefc", "ncon", "njmax"))
               for line in console.splitlines()):
            raise ValueError("Constraint/contact overflow in training console")
        weights = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        tensors = [v for v in weights.values() if torch.is_tensor(v)]
        if not tensors or any(not bool(torch.isfinite(v).all()) for v in tensors):
            raise ValueError("Empty/non-finite saved model")
        manifest.update(status="budget_completed", validation=validation)
        write_json(out / "validation.json", {"passed": True, **validation})
        code = 0
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        write_json(out / "validation.json", {"passed": False, "error": str(exc)})
        raise
    finally:
        manifest["workflow_exit_code"] = code
        write_json(out / "run_manifest.json", manifest)
        write_json(out / "feedback.json", manifest)
        with (out / "runtime.txt").open("a", encoding="utf-8") as stream:
            stream.write(f"WORKFLOW_EXIT_CODE={code}\n")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
