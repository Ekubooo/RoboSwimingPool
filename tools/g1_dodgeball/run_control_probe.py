"""Run bounded, sequential frozen-policy control experiments on the training host."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.getup.common import provenance, write_json


def stamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    episodes = 2 if args.smoke else 32
    seeds = [123] if args.smoke else [123, 456]
    variants = ["release"] if args.smoke else ["baseline", "release", "clip"]
    manifest = provenance({"started": stamp(), "pid": os.getpid(), "status": "running",
                           "episodes_per_case": episodes, "seeds": seeds, "variants": variants,
                           "cases": [], "training_updates": 0})
    write_json(out / "run_manifest.json", manifest)
    started, code = time.monotonic(), 1
    try:
        for seed in seeds:
            for variant in variants:
                case_dir = out / f"{variant}_seed{seed}"
                command = [sys.executable, "-B", "tools/g1_dodgeball/evaluate.py",
                           "--model_file", "output/a5000_g1_dodgeball_2048_full_r01/model.pt",
                           "--init", "reference", "--episodes", str(episodes), "--seed", str(seed),
                           "--projectiles", "on", "--joint_diagnostics", "--control_probe", variant,
                           "--out_dir", str(case_dir)]
                case = {"variant": variant, "seed": seed, "command": command, "started": stamp()}
                manifest["cases"].append(case)
                write_json(out / "run_manifest.json", manifest)
                print(f"START {variant} seed={seed} {case['started']}", flush=True)
                with (out / f"{variant}_seed{seed}.launcher.txt").open("w", encoding="utf-8") as log:
                    try:
                        child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                               timeout=1200, env={**os.environ, "PYTHONUTF8": "1",
                                               "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg"})
                        case["exit_code"] = child.returncode
                    except subprocess.TimeoutExpired:
                        case["exit_code"] = 124
                case["ended"] = stamp()
                write_json(out / "run_manifest.json", manifest)
                if case["exit_code"]:
                    raise RuntimeError(f"Case failed: {case_dir}, exit={case['exit_code']}")
                console = (case_dir / "console.txt").read_text(encoding="utf-8", errors="replace")
                if any("overflow" in line.lower() and any(k in line.lower() for k in ("nefc", "ncon", "njmax"))
                       for line in console.splitlines()):
                    raise RuntimeError(f"Constraint overflow: {case_dir}")
                result = json.loads((case_dir / "evaluation.json").read_text(encoding="utf-8"))
                case["metrics"] = {key: result[key] for key in (
                    "mean_episode_seconds", "timeout_rate", "failures", "raw_sds_mean",
                    "episodes_with_hit_detector", "episodes_with_nonfoot_ground_contact")}
                case["probe"] = {key: result["control_probe"][key] for key in (
                    "event_count", "responded_events", "complete_windows", "censored_windows")}
                print(f"DONE {variant} seed={seed}: {case['metrics']} {case['probe']}", flush=True)
        # Check common random initialization explicitly; no assertion of later trajectory equality.
        for seed in seeds:
            initial = [json.loads((out / f"{variant}_seed{seed}" / "control_probe.json").read_text(
                encoding="utf-8"))["initial_dof_pos"] for variant in variants]
            if any(q != initial[0] for q in initial[1:]):
                raise RuntimeError(f"Initial joint states differ between cases for seed {seed}")
        code = 0
        manifest["status"] = "completed"
    except Exception as exc:
        manifest.update(status="failed", error=str(exc))
        raise
    finally:
        manifest.update(ended=stamp(), elapsed_seconds=time.monotonic() - started, exit_code=code)
        write_json(out / "run_manifest.json", manifest)
        write_json(out / "summary.json", manifest)
        (out / "runtime.txt").write_text(
            f"START={manifest['started']}\nEND={manifest['ended']}\n"
            f"ELAPSED_SECONDS={manifest['elapsed_seconds']:.2f}\nEXIT_CODE={code}\n", encoding="utf-8")
        print(f"FINISHED status={manifest['status']} exit={code} {stamp()}", flush=True)


if __name__ == "__main__":
    main()
