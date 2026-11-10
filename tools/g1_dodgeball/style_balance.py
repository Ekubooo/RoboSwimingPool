"""G1 SMP objective-balance trial; unchanged physics/actions, fresh RL weights."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import yaml
from tools.getup.common import provenance, read_yaml, sha256, write_json
from tools.g1_dodgeball.workflow import AGENT, ENGINE, ENV, PRIOR_DIR, audit_assets, check_prior
from tools.g1_spinkick.workflow import capture

OVERRIDES = {"smp_reward_weight": 1.5, "sds_loss_scale": 4}
BUDGETS = {"smoke": (8, 3), "trial": (2048, 2001)}
BASELINE = "output/a5000_g1_dodgeball_2048_full_r01/int_models/model_0000002000.pt"


def validate_log(path, samples, updates):
    """Reject malformed rows before checking all numbers, including the final column."""
    lines = [line.split() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("No training rows")
    header = lines[0]
    if len(set(header)) != len(header):
        raise ValueError("Duplicate log columns")
    for required in ("Iteration", "Samples", "Smp_Reward_Mean", "Sds_Loss_Mean"):
        if required not in header:
            raise ValueError(f"Missing log column: {required}")
    rows = []
    for line in lines[1:]:
        if len(line) != len(header):
            raise ValueError(f"Log column count mismatch: {len(line)} != {len(header)}")
        values = [float(v) for v in line]
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite training metric")
        rows.append(dict(zip(header, values, strict=True)))
    if rows[-1]["Samples"] != samples or rows[-1]["Iteration"] != updates - 1:
        raise ValueError("Training did not complete the exact sample/update budget")
    if any(b["Samples"] <= a["Samples"] or b["Iteration"] <= a["Iteration"]
           for a, b in zip(rows, rows[1:])):
        raise ValueError("Nonmonotonic training progress")
    return {"rows": len(rows), "last": rows[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=BUDGETS, required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--dry_run", action="store_true", help="Show exact configuration and budget without writes/GPU.")
    args = parser.parse_args()
    os.chdir(ROOT)
    envs, updates = BUDGETS[args.stage]
    cfg = read_yaml(AGENT)
    cfg.update(OVERRIDES)
    if args.stage == "smoke":
        cfg["test_episodes"] = 0
    samples = envs * cfg["steps_per_iter"] * updates
    out = Path(args.out_dir)
    runtime = out / "agent_runtime.yaml"
    command = [sys.executable, "-u", "-B", "mimickit/run.py", "--arg_file", "args/smp_dodgeball_g1_args.txt",
               "--mode", "train", "--engine_config", ENGINE, "--env_config", ENV,
               "--agent_config", str(runtime), "--num_envs", str(envs), "--max_samples", str(samples),
               "--out_dir", str(out), "--rand_seed", "123", "--devices", "cuda:0"]
    plan = {"stage": args.stage, "seed": 123, "num_envs": envs, "updates": updates,
            "max_samples": samples, "last_iteration": updates - 1, "overrides": OVERRIDES,
            "initialization": "fresh RL weights and normalizers; existing 200k prior",
            "matched_budget_baseline": BASELINE, "command": command}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory: {out}")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Requires the remote CUDA environment")
    if args.stage == "trial" and not Path(BASELINE).is_file():
        raise FileNotFoundError(BASELINE)
    audit = audit_assets("cuda:0")
    prior = check_prior(PRIOR_DIR, audit)
    out.mkdir(parents=True, exist_ok=True)
    runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_json(out / "motion_audit.json", audit)
    manifest = provenance({**plan, "status": "running", "gpu": torch.cuda.get_device_name(0),
                           "prior_ema_step": prior["verified_ema_step"],
                           "matched_baseline_sha256": sha256(BASELINE) if Path(BASELINE).is_file() else None,
                           "prior_sha256": sha256(Path(PRIOR_DIR) / "model.pt")})
    write_json(out / "run_manifest.json", manifest)
    code = 1
    try:
        code = capture(command, out)
        if code:
            raise RuntimeError(f"Training process failed: exit={code}")
        code = 2  # Training may exit 0 while output validation fails.
        validation = validate_log(out / "log.txt", samples, updates)
        console = (out / "console.txt").read_text(encoding="utf-8", errors="replace")
        if any("overflow" in line.lower() and any(k in line.lower() for k in ("nefc", "ncon", "njmax"))
               for line in console.splitlines()):
            raise ValueError("Constraint overflow in training console")
        weights = torch.load(out / "model.pt", map_location="cpu", weights_only=True)
        if not weights or any(not bool(torch.isfinite(v).all()) for v in weights.values() if torch.is_tensor(v)):
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
