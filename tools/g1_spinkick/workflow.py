"""Windows-friendly G1 SMP spinkick stages. Run from the repository root."""
import argparse
from datetime import datetime
import hashlib
import json
import math
import os
import pickle
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import yaml
from tools.getup.common import read_yaml, provenance, sha256, write_json

ENV = "data/envs/smp_spinkick_g1_env.yaml"
AGENT = "data/agents/smp_spinkick_g1_agent.yaml"
ENGINE = "data/engines/newton_engine.yaml"
PRIOR_CFG = "tools/diffusion_model/config/tinymdm_spinkick_g1.yaml"
PRIOR_DIR = "output/a5000_g1_spinkick_prior_50k_r01"
RL_DIR = "output/a5000_g1_spinkick_2048_full_r01"
BUDGETS = {"rl-smoke": (8, 256), "pilot": (2048, 26_214_400), "full": (2048, 380_239_872)}
OUTPUTS = {"prior-smoke": "output/a5000_g1_spinkick_prior_smoke_r01", "prior": PRIOR_DIR,
           "sample": PRIOR_DIR + "_samples", "rl-smoke": "output/a5000_g1_spinkick_rl_smoke_r01",
           "pilot": "output/a5000_g1_spinkick_2048_pilot_r01", "full": RL_DIR}


def prepare_motion():
    """Minimal projection to legal joint angles, not a dynamics/IK solution."""
    import numpy as np
    source = Path("data/motions/g1/g1_spinkick.pkl")
    with source.open("rb") as stream:
        motion = pickle.load(stream)
    frames = np.asarray(motion["frames"], dtype=np.float64)
    joints = ET.parse("data/assets/g1/g1.xml").findall(".//worldbody//joint")
    limits = np.asarray([[float(v) for v in j.attrib["range"].split()] for j in joints])
    if frames.shape[1] != 6 + len(joints) or not np.isfinite(frames).all():
        raise ValueError("Unexpected/non-finite G1 motion frame data.")
    corrected = frames.copy()
    corrected[:, 6:] = np.clip(frames[:, 6:], limits[:, 0], limits[:, 1])
    changes = []
    for i, joint in enumerate(joints):
        delta = np.abs(corrected[:, 6+i] - frames[:, 6+i])
        if delta.max() > 1e-6:
            changes.append({"joint": joint.attrib["name"], "changed_frames": int((delta > 1e-6).sum()),
                            "max_change_rad": float(delta.max()), "limits": limits[i].tolist()})
    result = {**motion, "frames": corrected.tolist()}
    folder = Path("output/g1_spinkick_assets")
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / "g1_spinkick_joint_limited.pkl"
    content = pickle.dumps(result, protocol=4)
    if target.exists() and target.read_bytes() != content:
        raise FileExistsError(f"Derived motion differs from expected: {target}. Preserve your version before regenerating.")
    if not target.exists():
        target.write_bytes(content)
    report = {"source": str(source), "source_sha256": sha256(source), "derived": str(target),
              "derived_sha256": sha256(target), "method": "per-frame joint-limit projection; no physical feasibility claim",
              "changes": changes}
    write_json(folder / "preparation.json", report)
    view = read_yaml("data/envs/view_motion_spinkick_g1_env.yaml")
    view["motion_file"] = str(source)
    (folder / "view_original.yaml").write_text(yaml.safe_dump(view, sort_keys=False), encoding="utf-8")
    return report


def audit_assets(device="cpu"):
    import torch
    from tools.diffusion_model.motion_prior_dataset import MotionPriorData
    preparation = prepare_motion()
    env, prior = read_yaml(ENV), read_yaml(PRIOR_CFG)
    root = ET.parse(env["char_file"]).getroot()
    joints = root.findall(".//worldbody//joint")
    compiler = root.find("compiler")
    mesh_dir = Path(env["char_file"]).parent / (compiler.get("meshdir", "") if compiler is not None else "")
    meshes = [mesh_dir / m.attrib["file"] for m in root.findall("./asset/mesh")]
    for path in [Path(env["motion_file"]), *meshes]:
        if not path.is_file():
            raise FileNotFoundError(f"Missing G1 asset: {path}. Copy the original data bundle, preserving directories.")
    if len(joints) != 29:
        raise ValueError("This preset requires the repository's 29-DoF G1 asset.")
    data = MotionPriorData(prior, device)
    obs = data.fetch_obs_demo(8)
    if obs.shape[-1] != 2100 or not bool(torch.isfinite(obs).all()):
        raise ValueError("Expected finite 10x210 G1 prior features.")
    limits = torch.tensor([[float(v) for v in j.attrib["range"].split()] for j in joints], device=device)
    dofs = data._kin_char_model.rot_to_dof(data._motion_lib._frame_joint_rot)
    excess = torch.maximum(limits[:, 0] - dofs, dofs - limits[:, 1]).clamp_min(0)
    digest = hashlib.sha256()
    for path in meshes:
        digest.update(str(path).replace("\\", "/").encode())
        digest.update(sha256(path).encode())
    return {"motion_preparation": preparation, "char_file": env["char_file"], "char_sha256": sha256(env["char_file"]),
            "motion_file": env["motion_file"], "motion_sha256": sha256(env["motion_file"]),
            "dof": data._kin_char_model.get_dof_size(), "bodies": data._kin_char_model.get_num_joints(),
            "motion_seconds": float(data._motion_lib.get_motion_length(torch.tensor([0], device=device))[0]),
            "frames": len(dofs), "prior_feature_size": obs.shape[-1], "mesh_count": len(meshes),
            "meshes_sha256": digest.hexdigest(), "max_joint_limit_excess_rad": float(excess.max()),
            "xml_mass_kg": sum(float(x.get("mass", "0")) for x in root.findall(".//inertial"))}


def check_prior(folder):
    folder = Path(folder)
    for name in ("model.pt", "diffusion_config.yaml", "env_config.yaml"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing G1 prior: {folder / name}. Train the G1 prior first; output/ is Git-ignored.")
    cfg = read_yaml(folder / "diffusion_config.yaml")
    actual, expected = read_yaml(cfg["env_config"]), read_yaml(ENV)
    for key in ("char_file", "global_obs", "root_height_obs", "enable_tar_obs", "disc_dof_vel_obs",
                "num_disc_obs_steps", "key_bodies"):
        if actual[key] != expected[key]:
            raise ValueError(f"Prior/runtime mismatch: {key}")
    if cfg["control_freq"] != read_yaml(ENGINE)["control_freq"] or cfg["motion_file"] != expected["motion_file"]:
        raise ValueError("G1 prior control frequency/motion mismatch.")
    import torch
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    step = state.get("ema_dmodel.step")
    if step is None or int(step.item()) != cfg["num_iterations"]:
        raise ValueError(f"Prior checkpoint has not completed its configured budget: EMA.step={step}, "
                         f"expected {cfg['num_iterations']}. Do not start full RL with a partial prior.")
    cfg["verified_ema_step"] = int(step.item())
    return cfg


def capture(command, out):
    started, stamp = time.monotonic(), datetime.now().astimezone().isoformat(timespec="seconds")
    exit_code, child = 1, None
    with (out / "console.txt").open("w", encoding="utf-8") as log:
        try:
            child = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace",
                                     env={**os.environ, "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8",
                                          "MPLBACKEND": "Agg", "PYTHONUTF8": "1"})
            for line in child.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            exit_code = child.wait()
        except KeyboardInterrupt:
            exit_code = 130
            if child is not None and child.poll() is None:
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.terminate()
                    child.wait()
        finally:
            runtime = (f"START={stamp}\nEND={datetime.now().astimezone().isoformat(timespec='seconds')}\n"
                       f"ELAPSED_SECONDS={time.monotonic()-started:.2f}\nEXIT_CODE={exit_code}\n")
            (out / "runtime.txt").write_text(runtime, encoding="utf-8")
            log.write(runtime)
            print(runtime, end="", flush=True)
    return exit_code


def collect_feedback(folder):
    folder = Path(folder)
    blocks = []
    for name in ("runtime.txt", "run_manifest.json", "reference_budget.json", "motion_audit.json", "engine_config.yaml",
                 "env_config.yaml", "agent_config.yaml", "diffusion_config.yaml"):
        path = folder / name
        if path.is_file():
            blocks.append(f"=== {name} ===\n" + path.read_text(encoding="utf-8"))
    path = folder / "log.txt"
    if path.is_file() and path.stat().st_size:
        lines = [line.split() for line in path.read_text().splitlines() if line.strip()]
        headers, values = lines[0], lines[1:]
        rows = [dict(zip(headers, r)) for r in values if len(r) == len(headers)]
        nonfinite = [(r.get("Iteration"), k, v) for r in rows for k, v in r.items() if not math.isfinite(float(v))]
        blocks.append(f"=== log audit ===\nrows={len(rows)} incomplete_rows={len(values)-len(rows)} nonfinite={nonfinite[:20]}")
        keys = [k for k in headers if k in ("Iteration", "Loss", "Wall_Time", "Samples", "Samples_Per_Second",
                                            "Train_Return", "Test_Return", "Train_Episode_Length", "Test_Episode_Length")
                or "smp" in k.lower() or "sds" in k.lower()]
        indices = sorted(set([*range(min(4, len(rows))), len(rows)//2, *range(max(0, len(rows)-8), len(rows))]))
        blocks.append("=== selected rows ===\n" + "\n".join(" | ".join(f"{k}={rows[i][k]}" for k in keys)
                                                                     for i in indices if i < len(rows)))
    path = folder / "console.txt"
    if path.is_file():
        blocks.append("=== console tail ===\n" + "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]))
    result = folder / "feedback.txt"
    result.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
    print(result.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("check", *OUTPUTS, "feedback"))
    parser.add_argument("--out_dir")
    parser.add_argument("--prior_dir", default=PRIOR_DIR)
    parser.add_argument("--run_dir", default=RL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--prior_iterations", type=int, default=None)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == "feedback":
        collect_feedback(args.run_dir)
        return 0
    if args.prior_iterations is not None:
        if args.stage != "prior" or args.prior_iterations < 2:
            parser.error("--prior_iterations is only for stage prior and must be >=2.")
        if args.prior_iterations != 50000 and args.out_dir is None:
            parser.error("Use an explicit fresh --out_dir when changing the prior budget.")
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Use the CUDA-enabled mimickit-newton environment.")
    torch.manual_seed(args.seed)
    audit = audit_assets(args.device)
    print(json.dumps(audit, indent=2), flush=True)
    if args.stage == "check":
        print("G1 ASSETS OK | GPU=" + torch.cuda.get_device_name(torch.device(args.device)), flush=True)
        return 0
    prior_info = check_prior(args.prior_dir) if args.stage in (*BUDGETS, "sample") else None
    out = Path(args.out_dir or OUTPUTS[args.stage])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}; choose a fresh --out_dir.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "motion_audit.json", audit)
    write_json(out / "reference_budget.json", json.loads(Path("tools/g1_spinkick/reference_budget.json").read_text()))
    common = [sys.executable, "-u"]
    extra = {}
    if args.stage in ("prior", "prior-smoke"):
        cfg = read_yaml(PRIOR_CFG)
        if args.prior_iterations is not None:
            cfg["num_iterations"] = args.prior_iterations
        if args.stage == "prior-smoke":
            cfg.update(num_iterations=2, num_samples_stat=64, output_iter=2)
        runtime = out / "prior_runtime.yaml"
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command = common + ["tools/diffusion_model/train_tinymdm.py", "--mode", "train", "--cfg_path", str(runtime),
                            "--out_dir", str(out), "--device", args.device, "--seed", str(args.seed)]
        extra["prior_iterations"] = cfg["num_iterations"]
    elif args.stage == "sample":
        command = common + ["tools/diffusion_model/train_tinymdm.py", "--mode", "test", "--cfg_path",
                            str(Path(args.prior_dir) / "diffusion_config.yaml"), "--model_file",
                            str(Path(args.prior_dir) / "model.pt"), "--out_dir", str(out), "--device", args.device]
    else:
        agent = read_yaml(AGENT)
        agent.update(smp_prior_cfg=str(Path(args.prior_dir) / "diffusion_config.yaml"),
                     smp_prior_model=str(Path(args.prior_dir) / "model.pt"))
        if args.stage == "rl-smoke":
            agent["test_episodes"] = 0
        runtime = out / "agent_runtime.yaml"
        runtime.write_text(yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
        num_envs, samples = BUDGETS[args.stage]
        command = common + ["mimickit/run.py", "--arg_file", "args/smp_spinkick_g1_args.txt", "--agent_config",
                            str(runtime), "--num_envs", str(num_envs), "--max_samples", str(samples),
                            "--out_dir", str(out), "--rand_seed", str(args.seed), "--devices", args.device]
        extra.update(num_envs=num_envs, max_samples=samples, updates=samples//(num_envs*32),
                     prior_ema_step=prior_info["verified_ema_step"],
                     prior_model_sha256=sha256(Path(args.prior_dir) / "model.pt"))
    write_json(out / "run_manifest.json", provenance({"stage": args.stage, "seed": 0 if args.stage == "sample" else args.seed,
               "gpu": torch.cuda.get_device_name(torch.device(args.device)), "command": command, **extra}))
    return capture(command, out)


if __name__ == "__main__":
    raise SystemExit(main())
