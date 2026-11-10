"""Offline Getup SDS diagnostics. Run from the repository root; no simulator."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "mimickit"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from tools.diffusion_model.motion_prior_dataset import MotionPriorData
from envs.amp_env import compute_disc_obs
from learning.tinymdm.tinymdm_model import TinyMDMModel
import util.torch_util as torch_util


FIELDS = ("root_pos", "root_rot", "root_vel", "root_ang_vel",
          "joint_rot", "dof_vel", "body_pos")
STATIC_CASES = ("standing_init_pose", "static_end_faceup", "static_end_facedown")


def clone_window(window):
    return {key: value.clone() for key, value in window.items()}


def recompute_body_positions(dataset, window):
    window["body_pos"], _ = dataset._kin_char_model.forward_kinematics(
        window["root_pos"], window["root_rot"], window["joint_rot"])
    return window


def observe(dataset, window):
    """Use precisely the same feature builder and reference frame as training."""
    root_pos, root_rot = window["root_pos"], window["root_rot"]
    ref_pos, ref_rot = root_pos[:, -1], root_rot[:, -1]
    if dataset._track_global_root():
        ref_pos, ref_rot = torch.zeros_like(ref_pos), torch.zeros_like(ref_rot)
        ref_rot[:, -1] = 1
    key_pos = window["body_pos"][..., dataset._key_body_ids, :]
    return compute_disc_obs(
        ref_pos, ref_rot, root_pos, root_rot,
        window["root_vel"], window["root_ang_vel"],
        window["joint_rot"], window["dof_vel"], key_pos,
        dataset._global_obs, dataset._root_height_obs, dataset._disc_dof_vel_obs)


def reference_windows(dataset, samples_per_motion):
    lib, device = dataset._motion_lib, dataset._device
    count = lib.get_num_motions()
    ids = torch.arange(count, device=device).repeat_interleave(samples_per_motion)
    # Stratified coverage of the full clips, retaining the trainer's start padding.
    phases = (torch.arange(samples_per_motion, device=device)[None, :]
              + torch.rand(count, samples_per_motion, device=device)) / samples_per_motion
    times = phases.flatten() * lib.get_motion_length(ids)
    window = dict(zip(FIELDS, dataset._fetch_smp_demo_data(ids, times)))
    return window, ids


def static_pose(dataset, pose, count):
    pose = torch.as_tensor(pose, dtype=torch.float32, device=dataset._device).reshape(1, -1)
    if pose.shape[-1] != dataset._motion_lib.get_motion_frame_size():
        raise ValueError("Standing pose and prior skeleton have different frame sizes.")
    pos, rot, joints = dataset._motion_lib._extract_frame_data(pose.cpu().numpy())
    horizon = dataset._num_disc_obs_steps
    expand = lambda x: x[:, None].expand(count, horizon, *x.shape[1:]).clone()
    window = {
        "root_pos": expand(pos), "root_rot": expand(rot), "joint_rot": expand(joints),
        "root_vel": torch.zeros(count, horizon, 3, device=dataset._device),
        "root_ang_vel": torch.zeros(count, horizon, 3, device=dataset._device),
        "dof_vel": torch.zeros(count, horizon, dataset._kin_char_model.get_dof_size(),
                               device=dataset._device),
    }
    return recompute_body_positions(dataset, window)


def perturb_joints(dataset, reference, amplitude, jitter=False, noise=None):
    """Offsets are constant over time; jitter uses independent offsets per frame."""
    window = clone_window(reference)
    kin = dataset._kin_char_model
    dofs = kin.rot_to_dof(window["joint_rot"])
    shape = dofs.shape if jitter else (dofs.shape[0], 1, dofs.shape[-1])
    if noise is None:
        noise = torch.randn(shape, device=dofs.device)
    window["joint_rot"] = kin.dof_to_rot(dofs + amplitude * noise)
    velocity = kin.compute_dof_vel(
        window["joint_rot"][:, :-1], window["joint_rot"][:, 1:],
        dataset._timestep)
    window["dof_vel"] = torch.cat((velocity, velocity[:, -1:]), dim=1)
    return recompute_body_positions(dataset, window)


def transform_world(dataset, reference):
    """A pure yaw/XY-translation should not change local-heading observations."""
    window = clone_window(reference)
    yaw = torch.zeros_like(window["root_rot"])
    yaw[..., 2:] = np.sqrt(0.5)  # x,y,z,w: a 90-degree world yaw.
    window["root_pos"] = torch_util.quat_rotate(yaw, window["root_pos"])
    window["root_pos"] += window["root_pos"].new_tensor([3.0, -2.0, 0.0])
    window["root_rot"] = torch_util.quat_mul(yaw, window["root_rot"])
    for key in ("root_vel", "root_ang_vel"):
        window[key] = torch_util.quat_rotate(yaw, window[key])
    return recompute_body_positions(dataset, window)


def feature_transform_diagnostics(dataset, reference):
    original = observe(dataset, reference)
    moved = observe(dataset, transform_world(dataset, reference))
    raw_error = float((original - moved).abs().max())
    norm_error = float((reference["root_rot"].norm(dim=-1) - 1).abs().max())
    # Isolate interpolation roundoff without changing any scored input window.
    unit_reference = clone_window(reference)
    unit_reference["root_rot"] = torch_util.quat_normalize(unit_reference["root_rot"])
    recompute_body_positions(dataset, unit_reference)
    unit_error = float((observe(dataset, unit_reference) - observe(
        dataset, transform_world(dataset, unit_reference))).abs().max())
    return {"raw_yaw_xy_max_abs_obs_error": raw_error,
            "reference_root_quaternion_max_norm_error": norm_error,
            "unit_quaternion_diagnostic_yaw_xy_max_abs_obs_error": unit_error,
            "scored_inputs_changed": False}


def build_cases(dataset, samples_per_motion, pose_env, prior_char_file):
    lib = dataset._motion_lib
    suffixes = [Path(lib.get_motion_file(index)).stem.split("_")[-1]
                for index in range(lib.get_num_motions())]
    if sorted(suffixes) != ["facedown", "faceup"]:
        raise ValueError("Expected exactly one Humanoid faceup clip and one facedown clip.")
    reference, motion_ids = reference_windows(dataset, samples_per_motion)
    cases = {"reference_all": reference}
    count = len(motion_ids)
    for motion_id in range(dataset._motion_lib.get_num_motions()):
        path = Path(dataset._motion_lib.get_motion_file(motion_id))
        suffix = "faceup" if path.stem.endswith("_faceup") else "facedown"
        if not path.stem.endswith(("_faceup", "_facedown")):
            raise ValueError("This diagnostic expects the Humanoid faceup/facedown dataset.")
        start = int(dataset._motion_lib._motion_start_idx[motion_id])
        end = start + int(dataset._motion_lib._motion_num_frames[motion_id]) - 1
        for label, frame in (("start", start), ("end", end)):
            lib = dataset._motion_lib
            # Keep a batch axis: the existing hinge conversion expects one.
            pose = torch.cat((
                lib._frame_root_pos[frame],
                torch_util.quat_to_exp_map(lib._frame_root_rot[frame:frame + 1])[0],
                lib.joint_rot_to_dof(lib._frame_joint_rot[frame:frame + 1])[0]))
            cases[f"static_{label}_{suffix}"] = static_pose(dataset, pose, count)
    if Path(pose_env["char_file"]).resolve() != Path(prior_char_file).resolve():
        raise ValueError("Standing candidate uses a different character asset.")
    cases["standing_init_pose"] = static_pose(dataset, pose_env["init_pose"], count)
    kin = dataset._kin_char_model
    offset_noise = torch.randn(count, 1, kin.get_dof_size(), device=dataset._device)
    cases["pose_offset_0p15"] = perturb_joints(dataset, reference, 0.15, noise=offset_noise)
    cases["pose_offset_0p60"] = perturb_joints(dataset, reference, 0.60, noise=offset_noise)
    cases["joint_jitter_0p35"] = perturb_joints(dataset, reference, 0.35, jitter=True)
    raised = clone_window(reference)
    raised["root_pos"][..., 2] += 0.75
    cases["root_raised_0p75"] = recompute_body_positions(dataset, raised)
    return cases, motion_ids


@torch.no_grad()
def score_windows(model, observations, horizon, timesteps, batch_size, repeats, seed):
    if not torch.isfinite(observations).all():
        raise ValueError("Non-finite input observations.")
    norm = model.normalize(observations.reshape(len(observations), horizon, -1))
    norm = norm.reshape(len(observations), -1)
    if not torch.isfinite(norm).all():
        raise ValueError("Prior normalizer produced non-finite values.")
    result = torch.zeros(len(observations), len(timesteps), device=norm.device)
    repeat_means = []
    for repeat in range(repeats):
        # Same shapes, batches and seeds across cases give paired Gaussian noise.
        torch.manual_seed(seed + 1009 * repeat)
        chunks = []
        for start in range(0, len(norm), batch_size):
            loss = model.ESM_SDS_loss(norm[start:start + batch_size], t_lst=timesteps)
            if not torch.isfinite(loss).all() or (loss < 0).any():
                raise ValueError("Non-finite/negative SDS loss.")
            result[start:start + len(loss)] += loss
            chunks.append(loss.mean(dim=1))
        repeat_means.append(float(torch.cat(chunks).mean()))
    return result.cpu() / repeats, repeat_means


def summarize(losses, reference_scale):
    raw = losses.mean(dim=1)
    relative = (losses / reference_scale).mean(dim=1)
    quantiles = torch.quantile(raw, torch.tensor([0.10, 0.50, 0.90]))
    return {
        "num_windows": len(raw), "raw_sds_mean": float(raw.mean()),
        "raw_sds_p10": float(quantiles[0]), "raw_sds_p50": float(quantiles[1]),
        "raw_sds_p90": float(quantiles[2]),
        "raw_sds_mean_per_timestep": losses.mean(dim=0).tolist(),
        "relative_sds_mean": float(relative.mean()),
        "relative_sds_p50": float(relative.median()),
    }


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plot_report(out_dir, all_scores, scale, cases, dataset):
    groups = list(all_scores)
    values = [(all_scores[name] / scale).mean(dim=1).clamp_min(1e-8).numpy()
              for name in groups]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(values, showfliers=False)
    ax.set_xticks(np.arange(1, len(groups) + 1))
    ax.set_xticklabels(groups)
    ax.set_yscale("log")
    ax.axhline(1.0, linestyle="--", color="gray", linewidth=1)
    ax.set_ylabel("Relative SDS (lower = closer to prior)")
    ax.set_title("Reference-calibrated diagnostic; not the RL reward")
    ax.tick_params(axis="x", labelrotation=60, labelsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "score_distributions.png", dpi=140)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 3.7))
    kin = dataset._kin_char_model
    head = kin.get_body_id("head")
    feet = [kin.get_body_id("right_foot"), kin.get_body_id("left_foot")]
    for index, name in enumerate(STATIC_CASES):
        ax = fig.add_subplot(1, 3, index + 1, projection="3d")
        pos = cases[name]["body_pos"][0, -1].cpu().numpy()
        for joint, parent in enumerate(kin._parent_indices):
            if parent >= 0:
                pair = pos[[parent, joint]]
                ax.plot(pair[:, 0], pair[:, 1], pair[:, 2], color="crimson")
        ax.scatter(*pos[head], color="royalblue", s=25)
        ax.scatter(pos[feet, 0], pos[feet, 1], pos[feet, 2], color="forestgreen", s=25)
        root = pos[0]
        ax.set_xlim(root[0] - 1, root[0] + 1)
        ax.set_ylim(root[1] - 1, root[1] + 1)
        ax.set_zlim(-0.1, 1.8)
        ax.set_title(name, fontsize=9)
    fig.suptitle("Static candidates, not verified balance: blue=head; green=feet")
    fig.tight_layout()
    fig.savefig(out_dir / "standing_candidates.png", dpi=140)
    plt.close(fig)


def make_summary(report):
    lines = [
        "GETUP PRIOR OFFLINE SDS CHECK",
        f"Model: {report['model_file']}",
        f"Checkpoint EMA step counter: {report['checkpoint_ema_step_counter']}",
        f"Noise levels: {report['diffusion_steps']}; repeats: {report['noise_repeats']}",
        f"Raw reference means per noise level: {report['reference_scale_per_timestep']}",
        "",
        "Lower SDS = closer to prior. Relative SDS divides by reference means (floor 1e-4).",
        "Relative SDS is NOT the adaptive RL reward or a physical success rate.",
        "",
        f"{'Case':<25} {'N':>5} {'SDS mean':>12} {'SDS p90':>12} {'Relative':>12}",
    ]
    for name, stats in report["groups"].items():
        lines.append(f"{name:<25} {stats['num_windows']:>5} "
                     f"{stats['raw_sds_mean']:>12.6f} {stats['raw_sds_p90']:>12.6f} "
                     f"{stats['relative_sds_mean']:>12.3f}")
    lines += ["", "Paired controls (larger SDS than the SAME reference windows):"]
    for name, stats in report["paired_controls"].items():
        lines.append(f"  {name}: fraction={stats['fraction_worse_than_reference']:.3f}, "
                     f"raw mean delta={stats['raw_sds_mean_delta']:.6f}")
    lines += ["", "Diagnostics (warnings require review, not automatic retraining):"]
    transform = report["feature_transform_diagnostics"]
    lines.append("  Yaw/XY feature error: "
                 f"raw={transform['raw_yaw_xy_max_abs_obs_error']:.6g}, "
                 f"unit-quaternion diagnostic="
                 f"{transform['unit_quaternion_diagnostic_yaw_xy_max_abs_obs_error']:.6g}")
    lines += [f"  - {message}" for message in report["notes"]]
    lines += ["", f"Elapsed seconds: {report['elapsed_seconds']:.2f}"]
    return "\n".join(lines) + "\n"


@torch.no_grad()
def run(args):
    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    if Path.cwd().resolve() != ROOT:
        raise ValueError(f"Run this command from the repository root: {ROOT}")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(Path(args.cfg_path).read_text(encoding="utf-8-sig"))
    env = yaml.safe_load(Path(cfg["env_config"]).read_text(encoding="utf-8-sig"))
    pose_env = yaml.safe_load(Path(args.standing_pose_env).read_text(encoding="utf-8-sig"))
    if cfg["arch_name"] != "DiT":
        raise ValueError("This checker does not supply labels to a conditional prior.")
    if env["global_obs"] or not env.get("root_height_obs", False):
        raise ValueError("Expected local-heading, root-height-enabled Getup features.")
    if not all(0 <= step < cfg["T"] for step in args.diffusion_steps):
        raise ValueError("Diffusion noise level is outside the model schedule.")
    dataset = MotionPriorData(cfg, args.device)
    cases, ids = build_cases(dataset, args.num_samples, pose_env, env["char_file"])
    observations = {name: observe(dataset, window) for name, window in cases.items()}
    reference_obs = observations["reference_all"]
    horizon = dataset._num_disc_obs_steps
    cfg["input_dim"] = reference_obs.shape[-1]
    cfg["input_channel"] = cfg["input_dim"] // horizon
    model = TinyMDMModel(cfg, args.device)
    model.load_state_dict(torch.load(args.model_file, map_location=args.device,
                                     weights_only=True), strict=True)
    model.to(args.device).eval().requires_grad_(False)
    if (model.obs_normalizer.get_std() <= 0).any():
        raise ValueError("Invalid checkpoint normalization standard deviations.")
    scores, repeated = {}, {}
    for index, (name, obs) in enumerate(observations.items(), start=1):
        print(f"Scoring {index}/{len(observations)}: {name}", flush=True)
        scores[name], repeated[name] = score_windows(
            model, obs, horizon, args.diffusion_steps, args.batch_size,
            args.noise_repeats, args.seed + 100000)
    reference = scores["reference_all"]
    scale = reference.mean(dim=0).clamp_min(1e-4)
    # Per-clip scores are slices of the same balanced reference evaluation.
    for motion_id in range(dataset._motion_lib.get_num_motions()):
        suffix = Path(dataset._motion_lib.get_motion_file(motion_id)).stem.split("_")[-1]
        scores[f"reference_{suffix}"] = reference[ids.cpu() == motion_id]
    groups = {name: summarize(value, scale) for name, value in scores.items()}
    controls = {}
    notes = [
        "Reference windows are training-distribution diagnostics, not held-out generalization.",
        "Static cases repeat one pose; their spread measures noise variation, not pose diversity.",
        "Zeroing end-pose velocities does not establish physical standing stability.",
        "Real RL uses a policy-data adaptive SDS normalizer; no RL reward is inferred here.",
    ]
    if (reference.mean(dim=0) < 1e-4).any():
        notes.append("Reference denominator floor is active; relative reference mean can be below 1.")
    for name in ("pose_offset_0p15", "pose_offset_0p60",
                 "joint_jitter_0p35", "root_raised_0p75"):
        delta = (scores[name] - reference).mean(dim=1)
        controls[name] = {"fraction_worse_than_reference": float((delta > 0).float().mean()),
                          "raw_sds_mean_delta": float(delta.mean())}
        if name != "pose_offset_0p15" and delta.mean() <= 0:
            notes.append(f"WARNING: {name} is not worse on average; inspect score sensitivity.")
    ref_p95 = float(torch.quantile((reference / scale).mean(dim=1), 0.95))
    high_static = [name for name in STATIC_CASES
                   if groups[name]["relative_sds_mean"] > ref_p95]
    if high_static:
        notes.append("Standing candidate means above reference p95: " + ", ".join(high_static)
                     + ". This is a mismatch hint, not a validated pass/fail threshold.")
    transform_checks = feature_transform_diagnostics(dataset, cases["reference_all"])
    if transform_checks["raw_yaw_xy_max_abs_obs_error"] > 2e-4:
        notes.append("Raw yaw/XY feature error exceeds 2e-4; slight non-unit interpolated "
                     "quaternions can amplify heading sensitivity in fallen poses. "
                     "See the unit-quaternion diagnostic; original scored features are unchanged.")
    if transform_checks["unit_quaternion_diagnostic_yaw_xy_max_abs_obs_error"] > 2e-4:
        notes.append("WARNING: unit-quaternion diagnostic still exceeds 2e-4; inspect features.")
    ema_step = int(model.ema_dmodel.step) if model.model_ema else None
    if ema_step is not None and ema_step < cfg["model_ema_update_after"]:
        notes.append("WARNING: checkpoint precedes EMA warmup; quality results are not meaningful.")
    static_metadata = {}
    kin = dataset._kin_char_model
    for name in STATIC_CASES:
        window = cases[name]
        pos = window["body_pos"][0, -1]
        static_metadata[name] = {
            "root_height": float(window["root_pos"][0, -1, 2]),
            "head_body_origin_height": float(pos[kin.get_body_id("head"), 2]),
            "foot_body_origin_heights": pos[
                [kin.get_body_id("right_foot"), kin.get_body_id("left_foot")], 2].tolist(),
        }
    plot_report(out_dir, scores, scale, cases, dataset)
    report = {
        "started_utc": started_utc,
        "cfg_path": args.cfg_path, "model_file": args.model_file,
        "model_sha256": sha256(args.model_file), "device": args.device,
        "torch_version": torch.__version__, "seed": args.seed,
        "checkpoint_ema_step_counter": ema_step,
        "normalizer_observation_count": int(model.obs_normalizer.get_count().item()),
        "num_samples_per_motion": args.num_samples, "noise_repeats": args.noise_repeats,
        "batch_size": args.batch_size, "diffusion_steps": args.diffusion_steps,
        "input_dim": cfg["input_dim"], "input_channel": cfg["input_channel"],
        "feature_env": env, "prior_config": cfg,
        "char_asset_sha256": sha256(env["char_file"]),
        "standing_pose_env": args.standing_pose_env, "standing_pose": pose_env["init_pose"],
        "motions": [
            {"file": dataset._motion_lib.get_motion_file(index),
             "sha256": sha256(dataset._motion_lib.get_motion_file(index)),
             "length_seconds": float(dataset._motion_lib.get_motion_lengths()[index]),
             "weight": float(dataset._motion_lib.get_motion_weights()[index])}
            for index in range(dataset._motion_lib.get_num_motions())],
        "reference_scale_per_timestep": scale.tolist(),
        "reference_relative_sds_p95": ref_p95,
        "feature_transform_diagnostics": transform_checks,
        "static_candidate_geometry": static_metadata,
        "groups": groups, "paired_controls": controls,
        "raw_repeat_group_means": repeated, "notes": notes,
        "elapsed_seconds": time.perf_counter() - started,
        "interpretation": "Offline score diagnostic only; no simulator or RL success validation.",
    }
    summary = make_summary(report)
    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, allow_nan=False), encoding="utf-8")
    (out_dir / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("CHECK COMPLETE: " + str(out_dir.resolve()), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg_path", required=True)
    parser.add_argument("--model_file", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_samples", type=int, default=256,
                        help="Stratified reference windows per clip (two clips).")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--noise_repeats", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--diffusion_steps", type=int, nargs="+", default=[22, 15, 8])
    parser.add_argument("--standing_pose_env", default="data/envs/smp_humanoid_env.yaml")
    args = parser.parse_args()
    if min(args.num_samples, args.batch_size, args.noise_repeats) <= 0:
        parser.error("Sample count, batch size and noise repeats must be positive.")
    run(args)


if __name__ == "__main__":
    main()
