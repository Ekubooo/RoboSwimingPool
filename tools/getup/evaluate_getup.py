"""Evaluate exactly N first episodes in N parallel environments (default 100).

Starts are balanced across the two reference clips; augmented starts vary by
seed. Each slot contributes exactly once, including early numerical failures.
"""
import argparse
import contextlib
import csv
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

from tools.getup.common import (
    AGENT_CONFIG, ENGINE_CONFIG, ENV_CONFIG, preflight, provenance, sha256, write_json,
)


class Tee:
    def __init__(self, terminal, log):
        self.terminal, self.log = terminal, log

    def write(self, text):
        self.terminal.write(text)
        self.log.write(text)
        self.flush()
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def run(args, out, evidence):
    import torch
    from envs import base_env
    from learning import agent_builder
    from learning.base_agent import AgentMode
    from util import mp_util, resource_util, util
    from envs.task_getup_env import TaskGetupEnv

    env_config, _, engine_config, prior = evidence
    env_config["getup_test_init"] = args.init
    import yaml
    for name, config in (("env_config.yaml", env_config), ("engine_config.yaml", engine_config),
                         ("agent_config.yaml", evidence[1])):
        (out / name).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    device = args.device
    env = agent = None
    started = time.monotonic()
    mp_util.init(0, 1, device, 0)
    util.set_rand_seed(args.seed)
    try:
        env = resource_util.construct(TaskGetupEnv, env_config=env_config,
                                      engine_config=engine_config, num_envs=args.episodes,
                                      device=device, visualize=args.visualize)
        agent = agent_builder.build_agent(args.agent_config, env, device)
        agent.load(args.model_file)
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        obs, info = env.reset()
        seen = torch.zeros(args.episodes, dtype=torch.bool, device=device)
        returns = torch.zeros(args.episodes, device=device)
        rows = []
        dt = env._engine.get_timestep()
        # TIME termination bounds every episode; do not silently omit failures.
        max_steps = math.ceil(env_config["episode_length"] / dt) + 3
        with torch.no_grad():
            for step in range(max_steps):
                action, _ = agent._decide_action(obs, info)
                obs, reward, done, info = env.step(action)
                returns += reward * ~seen
                terminal = done != base_env.DoneFlags.NULL.value
                completed = terminal & ~seen
                for index in completed.nonzero(as_tuple=False).flatten().tolist():
                    mid = int(info["getup_motion_id"][index])
                    rows.append({
                        "env_id": index, "motion_id": mid,
                        "motion": Path(env._motion_lib.get_motion_file(mid)).stem,
                        "augmented": bool(info["getup_init_augmented"][index]),
                        "initial_head_height": float(info["getup_init_head_height"][index]),
                        "success": bool(info["getup_success"][index]),
                        "end_stable": float(info["getup_hold_seconds"][index]) >= env._hold_target - 1e-5,
                        "best_hold_seconds": float(info["getup_best_hold_seconds"][index]),
                        "first_success_seconds": float(info["getup_first_success_seconds"][index]),
                        "episode_seconds": float(env.get_env_time()[index]),
                        "task_return": float(returns[index]),
                        "done": base_env.DoneFlags(int(done[index])).name,
                    })
                seen |= completed
                if bool(seen.all()):
                    break
                # Reset terminal slots to keep the physics well defined while
                # other slots finish. Already counted slots never contribute again.
                obs, info = env.reset(terminal.nonzero(as_tuple=False).flatten())
        if len(rows) != args.episodes:
            raise RuntimeError(f"Only {len(rows)}/{args.episodes} episodes completed; no success summary emitted.")
        rows.sort(key=lambda row: row["env_id"])
        def summarize(subset):
            n = len(subset)
            successful = [r for r in subset if r["success"]]
            return {
                "episodes": n,
                "success_rate": sum(r["success"] for r in subset) / n if n else None,
                "end_stable_rate": sum(r["end_stable"] for r in subset) / n if n else None,
                "mean_best_hold_seconds": sum(r["best_hold_seconds"] for r in subset) / n if n else None,
                "mean_success_time_seconds": (sum(r["first_success_seconds"] for r in successful)
                                               / len(successful) if successful else None),
                "failures": sum(r["done"] == "FAIL" for r in subset),
                "mean_task_return": sum(r["task_return"] for r in subset) / n if n else None,
            }
        summary = provenance({
            **prior, "timestamp": datetime.now().astimezone().isoformat(),
            "policy": args.model_file, "policy_sha256": sha256(args.model_file),
            "seed": args.seed, "initialization": args.init,
            "parallel_envs": args.episodes, "reward_mode": env_config["getup_reward_mode"],
            "success_hold_seconds": env._hold_target,
            "overall": summarize(rows),
            "by_motion": {str(mid): summarize([r for r in rows if r["motion_id"] == mid])
                          for mid in sorted({r["motion_id"] for r in rows})},
            "elapsed_seconds": time.monotonic() - started,
        })
        write_json(out / "evaluation.json", summary)
        with (out / "episodes.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        import json
        report = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        (out / "summary.txt").write_text(report, encoding="utf-8")
        print(report)
    finally:
        resource_util.cleanup(env.close if env else None, agent.close if agent else None,
                              mp_util.shutdown)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--init", choices=("starts", "augmented", "mixed"), default="starts")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env_config", default=ENV_CONFIG)
    parser.add_argument("--agent_config", default=AGENT_CONFIG)
    parser.add_argument("--engine_config", default=ENGINE_CONFIG)
    parser.add_argument("--visualize", action="store_true")
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.episodes < 1:
        parser.error("--episodes must be positive; it is also the parallel environment count.")
    if not Path(args.model_file).is_file():
        raise FileNotFoundError(args.model_file)
    evidence = preflight(args.env_config, args.agent_config, args.engine_config)
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {out}")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            try:
                run(args, out, evidence)
            except KeyboardInterrupt:
                print("Evaluation interrupted; no completed evaluation is claimed.")
                raise SystemExit(130)
            except Exception:
                import traceback
                traceback.print_exc()
                raise SystemExit(1)


if __name__ == "__main__":
    main()
