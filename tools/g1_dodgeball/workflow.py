"""G1 walk/run prior -> Newton dodgeball; long stages run on the user's A5000."""
import argparse
import hashlib
import json
import os
import pickle
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import yaml
from tools.getup.common import read_yaml, provenance, sha256, write_json
from tools.g1_spinkick.workflow import capture, collect_feedback

ENV = "data/envs/smp_dodgeball_g1_env.yaml"
PRIOR_ENV = "data/envs/smp_locomotion_g1_prior_env.yaml"
AGENT = "data/agents/smp_dodgeball_g1_agent.yaml"
ENGINE = "data/engines/newton_g1_dodgeball_engine.yaml"
PRIOR_CFG = "tools/diffusion_model/config/tinymdm_locomotion_g1.yaml"
PRIOR_DIR = "output/a5000_g1_locomotion_prior_200k_r01"
RL_DIR = "output/a5000_g1_dodgeball_2048_full_r01"
REFERENCE = "tools/g1_dodgeball/reference_budget.json"
DATASET = "data/datasets/dataset_g1_locomotion.yaml"
BUDGETS = {"rl-smoke": (8, 256), "pilot": (2048, 26_214_400), "full": (2048, 1_337_065_472)}
OUTPUTS = {"prior-smoke": "output/a5000_g1_locomotion_prior_smoke_r01", "prior": PRIOR_DIR,
           "sample": PRIOR_DIR + "_samples", "rl-smoke": "output/a5000_g1_dodgeball_rl_smoke_r01",
           "pilot": "output/a5000_g1_dodgeball_2048_pilot_r01", "full": RL_DIR}
FEATURE_KEYS = ("char_file", "global_obs", "root_height_obs", "enable_tar_obs", "enable_phase_obs",
                "disc_dof_vel_obs", "num_disc_obs_steps", "key_bodies")


def validate_training_output(folder):
    """A zero child exit code alone does not certify a usable policy."""
    import math
    folder = Path(folder)
    lines = (folder / "log.txt").read_text(encoding="utf-8").splitlines()
    headers = lines[0].split() if lines else []
    bad = [(row[0], key, value) for line in lines[1:] if (row := line.split())
           for key, value in zip(headers, row) if not math.isfinite(float(value))]
    console = (folder / "console.txt").read_text(encoding="utf-8", errors="replace")
    overflow = [line for line in console.splitlines() if "overflow" in line.lower()
                and any(s in line.lower() for s in ("nefc", "ncon", "njmax"))]
    ok = len(lines) > 1 and not bad and not overflow
    report = {"passed": ok, "nonfinite": bad[:20], "constraint_overflow": overflow[:8]}
    write_json(folder / "validation.json", report)
    with (folder / "runtime.txt").open("a", encoding="utf-8") as stream:
        stream.write("OUTPUT_VALIDATION=" + ("PASSED" if ok else "FAILED") + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return ok


def prepare_motions():
    """Preserve sources. Project limited ankle samples, without claiming IK/dynamics feasibility."""
    import numpy as np
    joints = ET.parse("data/assets/g1/g1.xml").findall(".//worldbody//joint")
    limits = np.asarray([[float(v) for v in j.attrib["range"].split()] for j in joints])
    if len(joints) != 29:
        raise ValueError("Expected the repository's 29-DoF G1 XML.")
    folder = Path("output/g1_locomotion_assets")
    folder.mkdir(parents=True, exist_ok=True)
    reports = []
    for entry in read_yaml(DATASET)["motions"]:
        target = Path(entry["file"])
        source = Path("data/motions/g1") / target.name.replace("_joint_limited", "")
        with source.open("rb") as stream:
            motion = pickle.load(stream)
        frames = np.asarray(motion["frames"], dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != 35 or not np.isfinite(frames).all():
            raise ValueError(f"Unexpected/non-finite G1 frames: {source}")
        corrected = frames.copy()
        corrected[:, 6:] = np.clip(frames[:, 6:], limits[:, 0], limits[:, 1])
        changes = []
        for i, joint in enumerate(joints):
            delta = np.abs(corrected[:, 6+i] - frames[:, 6+i])
            if delta.max() > 1e-6:
                changes.append({"joint": joint.attrib["name"], "changed_frames": int((delta > 1e-6).sum()),
                                "max_change_rad": float(delta.max()), "limits": limits[i].tolist()})
        content = pickle.dumps({**motion, "frames": corrected.tolist()}, protocol=4)
        if target.exists() and target.read_bytes() != content:
            raise FileExistsError(f"Preserve your edited derivative before regenerating: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(content)
        report = {"source": str(source), "source_sha256": sha256(source), "derived": str(target),
                  "derived_sha256": sha256(target), "fps": motion["fps"], "frames": len(frames),
                  "seconds": (len(frames)-1) / motion["fps"], "loop_mode": motion["loop_mode"],
                  "sampling_weight": entry["weight"], "changes": changes}
        reports.append(report)
    write_json(folder / "preparation.json", {"method": "per-frame joint-limit projection; not IK retargeting",
                                             "motions": reports})
    return reports


def audit_assets(device="cpu"):
    import torch
    from tools.diffusion_model.motion_prior_dataset import MotionPriorData
    preparation = prepare_motions()
    env, pe, cfg = read_yaml(ENV), read_yaml(PRIOR_ENV), read_yaml(PRIOR_CFG)
    for key in FEATURE_KEYS:
        if env[key] != pe[key]:
            raise ValueError(f"Prior/dodgeball features differ: {key}")
    root = ET.parse(env["char_file"]).getroot()
    names = {b.get("name") for b in root.findall(".//body")}
    if env["proj_target_body"] not in names:
        raise ValueError("Projectile target does not exist in G1.")
    mesh_dir = Path(env["char_file"]).parent
    compiler = root.find("compiler")
    if compiler is not None:
        mesh_dir /= compiler.get("meshdir", "")
    meshes = [mesh_dir / m.attrib["file"] for m in root.findall("./asset/mesh")]
    ball = Path("data/assets/objects/dodgeball.xml")
    for p in [*meshes, ball]:
        if not p.is_file():
            raise FileNotFoundError(f"Missing asset: {p}. Git pull does not download ignored assets.")
    data = MotionPriorData(cfg, device)
    ids = torch.arange(len(preparation), device=device).repeat_interleave(16)
    phases = torch.linspace(0, 1, 16, device=device).repeat(len(preparation))
    features = data._compute_smp_obs_demo(ids, phases * data._motion_lib.get_motion_length(ids))
    if features.shape[-1] != 2100 or not bool(torch.isfinite(features).all()):
        raise ValueError("Expected finite G1 10x210 prior features for both walk and run.")
    limits = torch.tensor([[float(v) for v in j.attrib["range"].split()]
                           for j in root.findall(".//worldbody//joint")], device=device)
    dofs = data._kin_char_model.rot_to_dof(data._motion_lib._frame_joint_rot)
    excess = torch.maximum(limits[:, 0] - dofs, dofs - limits[:, 1]).clamp_min(0)
    digest = hashlib.sha256()
    for p in meshes:
        digest.update(str(p).replace("\\", "/").encode())
        digest.update(sha256(p).encode())
    return {"motion_preparation": preparation, "dataset": DATASET, "dataset_sha256": sha256(DATASET),
            "char_file": env["char_file"], "char_sha256": sha256(env["char_file"]),
            "dof": data._kin_char_model.get_dof_size(), "bodies": data._kin_char_model.get_num_joints(),
            "prior_feature_size": features.shape[-1], "global_obs": env["global_obs"],
            "max_joint_limit_excess_rad": float(excess.max()), "mesh_count": len(meshes),
            "meshes_sha256": digest.hexdigest(), "projectile_sha256": sha256(ball),
            "xml_mass_kg": sum(float(x.get("mass", "0")) for x in root.findall(".//inertial")),
            "dataset_note": "G1 walk/run only. Not a G1 retargeting of the full LaFAN1 dataset."}


def check_prior(folder, audit=None):
    import torch
    folder = Path(folder)
    for name in ("model.pt", "diffusion_config.yaml", "env_config.yaml", "motion_audit.json"):
        if not (folder / name).is_file():
            raise FileNotFoundError(f"Missing G1 locomotion prior: {folder / name}")
    cfg = read_yaml(folder / "diffusion_config.yaml")
    actual, expected = read_yaml(cfg["env_config"]), read_yaml(ENV)
    for key in FEATURE_KEYS:
        if actual[key] != expected[key]:
            raise ValueError(f"G1 locomotion prior/runtime mismatch: {key}")
    if cfg["motion_file"] != DATASET or cfg["control_freq"] != read_yaml(ENGINE)["control_freq"]:
        raise ValueError("G1 locomotion prior dataset/control-frequency mismatch.")
    saved = json.loads((folder / "motion_audit.json").read_text(encoding="utf-8"))
    current = audit if audit is not None else audit_assets()
    for key in ("char_sha256", "dataset_sha256", "meshes_sha256"):
        if saved[key] != current[key]:
            raise ValueError(f"Assets changed since prior training: {key}")
    for a, b in zip(saved["motion_preparation"], current["motion_preparation"], strict=True):
        if a["derived_sha256"] != b["derived_sha256"]:
            raise ValueError("Walk/run reference changed since prior training.")
    state = torch.load(folder / "model.pt", map_location="cpu", weights_only=True)
    step = state.get("ema_dmodel.step")
    if step is None or int(step.item()) != cfg["num_iterations"]:
        raise ValueError(f"Incomplete prior: EMA.step={step}; expected {cfg['num_iterations']}")
    cfg["verified_ema_step"] = int(step.item())
    return cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("check", *OUTPUTS, "feedback"))
    parser.add_argument("--out_dir")
    parser.add_argument("--prior_dir", default=PRIOR_DIR)
    parser.add_argument("--run_dir", default=RL_DIR)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.stage == "feedback":
        collect_feedback(args.run_dir)
        validation = Path(args.run_dir) / "validation.json"
        if validation.is_file():
            with (Path(args.run_dir) / "feedback.txt").open("a", encoding="utf-8") as stream:
                stream.write("\n=== validation.json ===\n" + validation.read_text(encoding="utf-8"))
        return 0
    import torch
    if not torch.cuda.is_available() or not args.device.startswith("cuda"):
        raise RuntimeError("Use the CUDA-enabled mimickit-newton Conda environment.")
    torch.manual_seed(args.seed)
    audit = audit_assets(args.device)
    print(json.dumps(audit, indent=2), flush=True)
    if args.stage == "check":
        print("G1 LOCOMOTION / DODGEBALL ASSETS OK | GPU=" + torch.cuda.get_device_name(torch.device(args.device)))
        return 0
    prior = check_prior(args.prior_dir, audit) if args.stage in (*BUDGETS, "sample") else None
    out = Path(args.out_dir or OUTPUTS[args.stage])
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Nonempty output directory: {out}; choose a fresh --out_dir.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "motion_audit.json", audit)
    write_json(out / "reference_budget.json", json.loads(Path(REFERENCE).read_text()))
    command, extra = [sys.executable, "-u"], {}
    if args.stage in ("prior", "prior-smoke"):
        cfg = read_yaml(PRIOR_CFG)
        if args.stage == "prior-smoke":
            cfg.update(num_iterations=2, num_samples_stat=64, output_iter=2)
        runtime = out / "prior_runtime.yaml"
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        command += ["tools/diffusion_model/train_tinymdm.py", "--mode", "train", "--cfg_path", str(runtime),
                    "--out_dir", str(out), "--device", args.device, "--seed", str(args.seed)]
        extra["prior_iterations"] = cfg["num_iterations"]
    elif args.stage == "sample":
        command += ["tools/diffusion_model/train_tinymdm.py", "--mode", "test", "--cfg_path",
                    str(Path(args.prior_dir) / "diffusion_config.yaml"), "--model_file",
                    str(Path(args.prior_dir) / "model.pt"), "--out_dir", str(out), "--device", args.device]
    else:
        cfg = read_yaml(AGENT)
        cfg.update(smp_prior_cfg=str(Path(args.prior_dir) / "diffusion_config.yaml"),
                   smp_prior_model=str(Path(args.prior_dir) / "model.pt"))
        if args.stage == "rl-smoke":
            cfg["test_episodes"] = 0
        runtime = out / "agent_runtime.yaml"
        runtime.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        n, samples = BUDGETS[args.stage]
        command += ["mimickit/run.py", "--arg_file", "args/smp_dodgeball_g1_args.txt", "--engine_config", ENGINE, "--agent_config", str(runtime),
                    "--num_envs", str(n), "--max_samples", str(samples), "--out_dir", str(out),
                    "--rand_seed", str(args.seed), "--devices", args.device]
        extra.update(num_envs=n, max_samples=samples, updates=samples // (n * 32),
                     prior_ema_step=prior["verified_ema_step"],
                     prior_model_sha256=sha256(Path(args.prior_dir) / "model.pt"))
    write_json(out / "run_manifest.json", provenance({"stage": args.stage, "seed": 0 if args.stage == "sample" else args.seed,
               "gpu": torch.cuda.get_device_name(torch.device(args.device)), "command": command, **extra}))
    code = capture(command, out)
    if code == 0 and args.stage != "sample" and not validate_training_output(out):
        print("Invalid training output: stop before expanding the budget; send feedback.txt.", flush=True)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
