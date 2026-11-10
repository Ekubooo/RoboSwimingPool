"""Short, matched-budget G1 dodgeball limit experiments. No automatic full training."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import yaml
from tools.g1_dodgeball.workflow import (ENGINE, PRIOR_DIR, REFERENCE, audit_assets, check_prior,
    capture, validate_training_output, collect_feedback, provenance, sha256, write_json, read_yaml)

ENV = "data/envs/smp_dodgeball_g1_target_limits_env.yaml"
AGENTS = {"targets": "data/agents/smp_dodgeball_g1_target_limits_agent.yaml",
          "pressure": "data/agents/smp_dodgeball_g1_limit_pressure_agent.yaml"}
BUDGETS = {"smoke": (8, 256), "pilot": (2048, 26_214_400)}


def output_dir(variant, stage):
    suffix = "8_smoke" if stage == "smoke" else "2048_pilot"
    return f"output/a5000_g1_dodgeball_{variant}_{suffix}_r01"


def feedback(folder):
    folder = Path(folder)
    collect_feedback(folder)
    additions = []
    for name in ("repair_spec.json", "validation.json"):
        if (folder / name).is_file():
            additions.append(f"=== {name} ===\n" + (folder / name).read_text(encoding="utf-8"))
    p = folder / "log.txt"
    if p.is_file():
        lines = [line.split() for line in p.read_text().splitlines() if line.strip()]
        headers = lines[0]
        keys = [key for key in headers if key.startswith(("Physical_", "G1_"))]
        rows = [dict(zip(headers, row)) for row in lines[1:] if len(row) == len(headers)]
        additions.append("=== repair metrics ===\n" + "\n".join(
            " | ".join(f"{key}={row[key]}" for key in ("Iteration", *keys)) for row in rows))
    with (folder / "feedback.txt").open("a", encoding="utf-8") as stream:
        stream.write("\n\n" + "\n\n".join(additions) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("check", *BUDGETS, "feedback"))
    parser.add_argument("--variant", choices=tuple(AGENTS), default="targets")
    parser.add_argument("--prior_dir", default=PRIOR_DIR)
    parser.add_argument("--out_dir")
    parser.add_argument("--run_dir")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == "feedback":
        feedback(args.run_dir or output_dir(args.variant, "pilot"))
        return 0
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Use the CUDA-enabled mimickit-newton environment.")
    torch.manual_seed(args.seed)
    audit = audit_assets(args.device)
    prior = check_prior(args.prior_dir, audit)
    if prior["verified_ema_step"] < 200000:
        raise ValueError("Use the completed 200k-or-longer G1 locomotion prior, not a local diagnostic prior.")
    if args.stage == "check":
        print("G1 REPAIR PREFLIGHT OK; prior completed; targets/pressure are fresh 400-update RL runs.")
        return 0
    out = Path(args.out_dir or output_dir(args.variant, args.stage))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}; use a fresh --out_dir.")
    out.mkdir(parents=True, exist_ok=True)
    cfg = read_yaml(AGENTS[args.variant])
    cfg.update(smp_prior_cfg=str(Path(args.prior_dir) / "diffusion_config.yaml"),
               smp_prior_model=str(Path(args.prior_dir) / "model.pt"))
    if args.stage == "smoke":
        cfg["test_episodes"] = 0
    runtime = out / "agent_runtime.yaml"
    runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    n, samples = BUDGETS[args.stage]
    command = [sys.executable, "-u", "mimickit/run.py", "--arg_file", "args/smp_dodgeball_g1_args.txt",
               "--env_config", ENV, "--engine_config", ENGINE, "--agent_config", str(runtime),
               "--num_envs", str(n), "--max_samples", str(samples), "--out_dir", str(out),
               "--rand_seed", str(args.seed), "--devices", args.device]
    spec = {"variant": args.variant, "initialization": "fresh RL, no old policy/optimizer/normalizer loaded",
            "physical_target_clip": True, "physical_target_weight": cfg["physical_target_weight"],
            "physical_target_scale_rad": cfg["physical_target_scale_rad"],
            "limit_pressure_weight": cfg["limit_pressure_weight"], "limit_margin_rad": cfg["limit_margin_rad"],
            "task_reward_weight": cfg["task_reward_weight"], "smp_reward_weight": cfg["smp_reward_weight"],
            "prior_model_sha256": sha256(Path(args.prior_dir) / "model.pt"), "prior_ema_step": prior["verified_ema_step"],
            "note": "Expanded PPO action representation is preserved; only sent PD targets are clipped. "
                    "Physical joints can still transiently exceed soft solver limits. Pressure is a per-step leg cost."}
    write_json(out / "repair_spec.json", spec)
    write_json(out / "motion_audit.json", audit)
    write_json(out / "reference_budget.json", json.loads(Path(REFERENCE).read_text()))
    write_json(out / "run_manifest.json", provenance({"stage": "repair-" + args.stage, "variant": args.variant,
        "seed": args.seed, "num_envs": n, "max_samples": samples, "updates": samples // (n*32),
        "command": command, **spec}))
    code = capture(command, out)
    if code == 0 and not validate_training_output(out):
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
