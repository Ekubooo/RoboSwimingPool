"""Run one-update smoke or full Getup RL, with console/runtime capture.

Run from the repository root. The actual training remains in mimickit/run.py.
"""
import argparse
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.getup.common import (
    AGENT_CONFIG, ENGINE_CONFIG, ENV_CONFIG, FULL_OUTPUT, FULL_SAMPLES,
    SMOKE_OUTPUT, preflight, provenance, write_json,
)
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("smoke", "full"))
    parser.add_argument("--out_dir")
    parser.add_argument("--agent_config", default=AGENT_CONFIG)
    parser.add_argument("--env_config", default=ENV_CONFIG)
    parser.add_argument("--engine_config", default=ENGINE_CONFIG)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--check_only", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    env, agent, engine, prior = preflight(args.env_config, args.agent_config, args.engine_config)
    import torch
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This Newton training preset requires a working CUDA PyTorch installation.")
    gpu = torch.cuda.get_device_name(torch.device(args.device))
    full = args.stage == "full"
    count, samples = (2048, FULL_SAMPLES) if full else (8, 256)
    if agent["steps_per_iter"] != 32:
        raise ValueError("Preset budget requires steps_per_iter=32.")
    updates = samples // (count * agent["steps_per_iter"])
    print(f"PREFLIGHT OK | GPU={gpu} | envs={count} | samples={samples} | updates={updates}", flush=True)
    print(f"Prior: {prior['prior_model']} | sha256={prior['prior_model_sha256']}", flush=True)
    if args.check_only:
        return 0
    out = Path(args.out_dir or (FULL_OUTPUT if full else SMOKE_OUTPUT))
    # A new output directory protects previous checkpoints AND partial logs.
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}. Choose a new --out_dir.")
    out.mkdir(parents=True, exist_ok=True)
    if not full:
        agent["test_episodes"] = 0  # Plumbing smoke, no policy-quality claim.
    runtime_agent = out / "agent_runtime.yaml"
    runtime_agent.write_text(yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
    command = [sys.executable, "-u", "mimickit/run.py",
               "--arg_file", "args/smp_getup_humanoid_args.txt",
               "--engine_config", args.engine_config, "--env_config", args.env_config,
               "--agent_config", str(runtime_agent), "--devices", args.device,
               "--num_envs", str(count), "--max_samples", str(samples),
               "--rand_seed", str(args.seed), "--out_dir", str(out)]
    metadata = provenance({**prior, "stage": args.stage, "gpu": gpu,
                           "num_envs": count, "max_samples": samples, "updates": updates,
                           "seed": args.seed, "command": command,
                           "reward_mode": env["getup_reward_mode"]})
    write_json(out / "run_manifest.json", metadata)
    started = time.monotonic()
    start_stamp = datetime.now().astimezone().isoformat(timespec="seconds")
    exit_code = 1
    process = None
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        print("START=" + start_stamp, file=log, flush=True)
        try:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace",
                                       env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = process.wait()
        except KeyboardInterrupt:
            exit_code = 130
            if process is not None and process.poll() is None:
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    process.wait()
        finally:
            end_stamp = datetime.now().astimezone().isoformat(timespec="seconds")
            runtime = (f"START={start_stamp}\nEND={end_stamp}\n"
                       f"ELAPSED_SECONDS={time.monotonic() - started:.2f}\nEXIT_CODE={exit_code}\n")
            (out / "runtime.txt").write_text(runtime, encoding="utf-8")
            print(runtime, end="", flush=True)
            print(runtime, end="", file=log, flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
