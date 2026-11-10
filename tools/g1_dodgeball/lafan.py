"""Released frozen LaFAN prior + G1 adapter; fresh RL, exact published sample budget."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))
import yaml
from tools.getup.common import provenance, read_yaml, sha256, write_json
from tools.g1_spinkick.workflow import capture
from tools.g1_dodgeball.style_balance import validate_log

ENV = "data/envs/smp_dodgeball_g1_lafan_env.yaml"
AGENT = "data/agents/smp_dodgeball_g1_lafan_agent.yaml"
ENGINE = "data/engines/newton_g1_dodgeball_engine.yaml"
ENGINES = {"newton": ENGINE,
           "newton_1_7": "data/engines/newton_1_7_g1_dodgeball_engine.yaml"}
FULL_SAMPLES = 1_337_065_472


def preflight(device, count=1024, env_file=ENV, agent_file=AGENT):
    import torch
    from anim.mjcf_char_model import MJCFCharModel
    from envs.g1_lafan_adapter import G1LafanAdapter
    from learning.tinymdm.tinymdm_model import TinyMDMModel
    from util import torch_util as tu
    from envs.smp_env import SMPEnv
    from types import SimpleNamespace
    torch.manual_seed(123)
    a = read_yaml(agent_file)
    if sha256(a["smp_prior_model"]) != a["smp_prior_sha256"]:
        raise ValueError("Wrong released LaFAN prior")
    cfg = read_yaml(a["smp_prior_cfg"])
    cfg["input_dim"] = 1140
    prior = TinyMDMModel(cfg, device).to(device).eval()
    state = torch.load(a["smp_prior_model"], map_location=device, weights_only=True)
    prior.load_state_dict(state, strict=True)
    prior.requires_grad_(False)
    g = MJCFCharModel(device)
    g.load(read_yaml(env_file)["char_file"])
    adapter = G1LafanAdapter(g, device)
    with torch.no_grad():
        samples = []
        for start in range(0, count, 256):
            x = prior.sample_ema(shape=(1140,), batch_size=min(256, count-start), device=device,
                                 sampler="ddpm", num_inference_steps=10)
            samples.append(prior.unnormalize(x.reshape(-1, 10, 114)))
        samples = torch.cat(samples)
        frames = adapter.from_samples(samples)
        stub = SimpleNamespace(_kin_char_model=g, _device=device,
                               _engine=SimpleNamespace(get_timestep=lambda: 1/30))
        init = {k: v[:, 0] for k, v in SMPEnv._motion_frames_to_init_states(stub, frames).items()}
        obs = adapter.observations(init["root_pos"], init["root_rot"], init["root_vel"],
                                   init["root_ang_vel"], init["joint_rot"], 1/30)
        normalized = prior.normalize(obs.reshape(-1, 10, 114)).reshape(-1, 1140)
        sds = prior.ESM_SDS_loss(norm_x_obs=normalized[:256], t_lst=a["diffusion_steps"])
        sole = adapter.sole_height(init["body_pos"], init["body_rot"], "g1")
        q = frames[..., 6:]
        hp, hj, _ = adapter.to_human(init["root_pos"], init["root_rot"], init["joint_rot"])
        source_j = tu.tan_norm_to_quat(samples[..., 9:93].reshape(-1, 10, 14, 6))
        source_j = adapter.human.dof_to_rot(adapter.human.rot_to_dof(source_j))
        error = torch.linalg.vector_norm(tu.quat_to_exp_map(tu.quat_mul(tu.quat_conjugate(source_j), hj)), dim=-1)
        boundary = ((q-adapter.low).abs() < 1e-4) | ((q-adapter.high).abs() < 1e-4)
        report = {"samples": count, "prior_sha256": a["smp_prior_sha256"], "prior_ema_step": int(state["ema_dmodel.step"]),
                  "canonical_dim": 114, "scale": adapter.scale, "discarded_windows": 0,
                  "minimum_sole_height_m": sole.min().item(),
                  "projected_joint_fraction": boundary.float().mean().item(),
                  "orientation_error_mean_rad": error.mean().item(),
                  "orientation_error_p99_rad": torch.quantile(error.flatten(), .99).item(),
                  "max_dof_velocity_rad_s": init["dof_vel"].abs().max().item(),
                  "dof_velocity_p99_rad_s": torch.quantile(init["dof_vel"].abs().flatten(), .99).item(),
                  "max_root_speed_m_s": init["root_vel"].norm(dim=-1).max().item(),
                  "sds_mean": sds.mean().item(),
                  "finite": bool(torch.isfinite(obs).all() & torch.isfinite(sds).all()),
                  "limits_valid": bool((q >= adapter.low-1e-5).all() & (q <= adapter.high+1e-5).all())}
        if not report["finite"] or not report["limits_valid"] or sole.min() < .0018:
            raise ValueError(f"LaFAN retargeting failed: {report}")
        # Frozen prior AND normalizer must be bit-identical after sampling/scoring.
        if not all(torch.equal(v, prior.state_dict()[k]) for k, v in state.items()):
            raise ValueError("Released prior changed during preflight")
    return report


def main(task="dodgeball"):
    if task == "dodgeball":
        env_file, agent_file, full_samples = ENV, AGENT, FULL_SAMPLES
    elif task == "steering":
        env_file = "data/envs/smp_steering_g1_lafan_env.yaml"
        agent_file = "data/agents/smp_steering_g1_lafan_agent.yaml"
        full_samples = 2_267_676_672  # Published Steering log: 4096 * 32 * 17301.
    else:
        raise ValueError(f"Unknown G1 LaFAN task: {task}")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preflight", "smoke", "full"), required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--engine", choices=tuple(ENGINES),
                        default="newton_1_7" if task == "steering" else "newton",
                        help="Select the backend and run in its matching Conda environment.")
    parser.add_argument("--num_envs", type=int, choices=(2048, 4096), default=4096,
                        help="Full run only; 2048 is the resource fallback, same total samples")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    engine_file = ENGINES[args.engine]
    os.chdir(ROOT)
    out = Path(args.out_dir)
    n = 8 if args.stage == "smoke" else args.num_envs
    samples = n*32*3 if args.stage == "smoke" else full_samples
    agent = read_yaml(agent_file)
    if args.stage == "smoke":
        agent["test_episodes"] = 0
    command = [sys.executable, "-u", "-B", "mimickit/run.py", "--mode", "train", "--devices", "cuda:0",
               "--visualize", "false", "--env_config", env_file, "--engine_config", engine_file,
               "--agent_config", str(out/"agent_runtime.yaml"), "--num_envs", str(n),
               "--max_samples", str(samples), "--rand_seed", "123", "--out_dir", str(out),
               "--save_int_models", "true", "--logger", "txt"]
    plan = {"recipe": "g1_released_lafan_v1", "task": task, "stage": args.stage, "num_envs": n,
            "engine": args.engine,
            "max_samples": samples, "updates": samples//(n*32), "command": command,
            "initialization": "fresh RL; released frozen LaFAN; adapted GSI",
            "control_deviations": ["G1/Newton", "bounded PD targets and actor target penalty"],
            "prior_sha256": agent["smp_prior_sha256"]}
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0
    from engines.newton_version import check_newton_version
    check_newton_version(args.engine)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    if args.stage == "preflight":
        result = preflight("cuda:0", env_file=env_file, agent_file=agent_file)
        write_json(out/"preflight.json", result)
        print(json.dumps(result, indent=2))
        return 0
    # Fail before an expensive run on missing/wrong assets, never fall back to two clips.
    if sha256(agent["smp_prior_model"]) != agent["smp_prior_sha256"]:
        raise ValueError("Wrong released LaFAN prior")
    # DeepMimic's construction-time reset still needs native reference metadata.
    # These clips never train/replace the prior; all RL resets use the LaFAN GSI buffer.
    from tools.g1_dodgeball.workflow import prepare_motions
    bootstrap = prepare_motions()
    (out/"agent_runtime.yaml").write_text(yaml.safe_dump(agent, sort_keys=False), encoding="utf-8")
    manifest = provenance({**plan, "status": "running", "construction_bootstrap": bootstrap,
                           "asset_sha256": {p: sha256(p) for p in (env_file, agent_file, engine_file,
                               "data/assets/g1/g1.xml", "data/assets/humanoid/humanoid.xml")}})
    write_json(out/"run_manifest.json", manifest)
    try:
        code = capture(command, out)
        if code:
            raise RuntimeError(f"Training exit={code}")
        result = validate_log(out/"log.txt", samples, plan["updates"])
        console = (out/"console.txt").read_text(encoding="utf-8", errors="replace").lower()
        if "overflow" in console:
            raise ValueError("Overflow in training console")
        import torch
        saved = torch.load(out/"model.pt", map_location="cpu", weights_only=True)
        if not all(bool(torch.isfinite(v).all()) for v in saved.values() if torch.is_tensor(v)):
            raise ValueError("Nonfinite saved model")
        released = torch.load(agent["smp_prior_model"], map_location="cpu", weights_only=True)
        if not all("_prior_model."+k in saved and torch.equal(saved["_prior_model."+k], v)
                   for k, v in released.items()):
            raise ValueError("Frozen LaFAN prior/normalizer changed during RL")
        manifest.update(status="budget_completed", validation=result, workflow_exit_code=0)
    except Exception as exc:
        manifest.update(status="failed", error=str(exc), workflow_exit_code=1)
        raise
    finally:
        write_json(out/"run_manifest.json", manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
