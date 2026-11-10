"""Numerical diagnostics for saved motion collections; no fitting or sampling."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..core.collection import CollectionAnalysis, load_collection
from ..data.motion import discontinuity_diagnostics
from ..data.storage import atomic_write
from ..paths import ROOT, ensure_animation_imports


def load_character() -> Any:
    """Load the official Humanoid CPU kinematics, without a simulator."""
    ensure_animation_imports()
    from anim.mjcf_char_model import MJCFCharModel

    character = MJCFCharModel("cpu")
    character.load(str(ROOT / "data/assets/humanoid/humanoid.xml"))
    return character


def body_positions(character: Any, frames: np.ndarray) -> np.ndarray:
    """Return FK body origins in metres, preserving the supplied root pose."""
    import torch
    import util.torch_util as rotations

    tensor = torch.as_tensor(np.array(frames, dtype=np.float32, copy=True))
    with torch.no_grad():
        result, _ = character.forward_kinematics(
            tensor[:, :3], rotations.exp_map_to_quat(tensor[:, 3:6]),
            character.dof_to_rot(tensor[:, 6:]))
    return result.numpy()


def dof_labels(character: Any) -> list[str]:
    """Name hinge angles and spherical exponential-map coordinates explicitly."""
    labels = []
    for index in range(1, character.get_num_joints()):
        joint = character.get_joint(index)
        dim = joint.get_dof_dim()
        labels.extend([joint.name] if dim == 1 else
                      [f"{joint.name}.{axis}" for axis in "xyz"[:dim]])
    return labels


def scan_components(analysis: CollectionAnalysis, character: Any) -> list[dict]:
    """Measure each collection against its own source, never against another group."""
    source = np.concatenate([clip.frames for clip in analysis.clips])
    positions = body_positions(character, source)
    rank = analysis.max_components
    rows = []
    for k in sorted({min(value, rank) for value in (0, 1, 2, 3, 4, 8, rank)}):
        reconstructed = np.concatenate([
            analysis.for_clip(index).reconstruct(tuple(range(k)))
            for index in range(len(analysis.clips))])
        errors = reconstructed[:, 6:] - source[:, 6:]
        distances = np.linalg.norm(body_positions(character, reconstructed) - positions, axis=-1) * 100
        rows.append({"k": k, "variance": float(analysis.model.variance_ratio[:k].sum()),
                     "dof_rmse_rad": float(np.sqrt(np.mean(errors ** 2))),
                     "fk_rmse_cm": float(np.sqrt(np.mean(distances ** 2))),
                     "fk_mean_cm": float(distances.mean()), "fk_max_cm": float(distances.max())})
    return rows


def shared_residual(analysis: CollectionAnalysis, character: Any) -> dict:
    source = np.concatenate([clip.frames for clip in analysis.clips])
    frames = np.concatenate([analysis.for_clip(i).reconstruct(tuple(range(analysis.max_components)))
                             for i in range(len(analysis.clips))])
    distances = np.linalg.norm(body_positions(character, frames) - body_positions(character, source), axis=-1) * 100
    return {"dof_rmse_rad": float(np.sqrt(np.mean((frames[:, 6:] - source[:, 6:]) ** 2))),
            "fk_rmse_cm": float(np.sqrt(np.mean(distances ** 2)))}


def write_metrics(manifest: Path, output: Path) -> dict:
    """Write compact statistics and K scans from verified saved snapshots."""
    character = load_character()
    report = {"dof_labels": dof_labels(character), "independent": {}, "shared_residual": {},
              "metric_definition": "DOF coordinate RMSE (rad); body-origin Euclidean FK RMSE (cm)",
              "fk_dtype": "float32", "weighting": "equal weight per frame and DOF/body"}
    for mode in ("pose", "window"):
        for group in ("reference", "generated"):
            analysis = load_collection(manifest, group=group, mode=mode, basis="independent")
            diagnostics = [discontinuity_diagnostics(clip) for clip in analysis.clips]
            report["independent"][f"{group}/{mode}"] = {
                "rank": analysis.max_components, "sample_count": len(analysis.samples),
                "feature_count": analysis.samples.shape[1], "clip_count": len(analysis.clips),
                "scan": scan_components(analysis, character),
                "possible_branch_jump_count": sum(row["possible_branch_jump_count"] for row in diagnostics),
                "max_dof_step_rad": max(row["max_dof_step_rad"] for row in diagnostics)}
            print(f"Metrics: {group}/{mode}", flush=True)
            shared = load_collection(manifest, group=group, mode=mode, basis="shared")
            report["shared_residual"][f"{group}/{mode}"] = shared_residual(shared, character)
    atomic_write(output, (json.dumps(report, indent=2, allow_nan=False) + "\n").encode())
    return report


def write_decode_diagnostics(data_dir: Path) -> dict:
    """Quantify redundant features without overwriting them with FK derivatives.

    This case uses global_obs=True: key positions are root-relative in world
    axes, and root velocity uses the same axes as decoded root translation.
    """
    import yaml

    case = json.loads((data_dir / "case.json").read_text(encoding="utf-8"))
    prior_manifest = data_dir / case["sampling_manifest"]
    prior = json.loads(prior_manifest.read_text(encoding="utf-8"))
    payload = (prior_manifest.parent / "samples.npz").read_bytes()
    if hashlib.sha256(payload).hexdigest() != prior["sha256"]:
        raise ValueError("Cannot diagnose a corrupt sampling archive.")
    environment = yaml.safe_load((ROOT / "data/envs/smp_humanoid_env.yaml").read_text())
    if not environment["global_obs"]:
        raise ValueError("These consistency metrics require global_obs=True.")
    character = load_character()
    key_ids = [character.get_body_id(name) for name in environment["key_bodies"]]
    rotation_end = 9 + 6 * (character.get_num_joints() - 1)
    result = {"definition": "Redundant feature consistency, not a reference-vs-generation error",
              "velocity_difference": "forward position differences vs midpoint predicted root velocities",
              "key_bodies": environment["key_bodies"], "groups": {}}
    import io
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        for group, prefix in [("generated", ""), *[(f"stage_{t}", f"t{t}_") for t in (49, 39, 29, 19, 9, 0)]]:
            features, frames = archive[f"{prefix}features"], archive[f"{prefix}frames"]
            positions = body_positions(character, frames.reshape(-1, 34)).reshape(*frames.shape[:2], -1, 3)
            keys = positions[:, :, key_ids] - frames[:, :, None, :3]
            predicted_keys = features[:, :, rotation_end:rotation_end + len(key_ids) * 3].reshape(keys.shape)
            key_distance = np.linalg.norm(keys - predicted_keys, axis=-1) * 100
            velocity = np.diff(frames[:, :, :3], axis=1) * 30
            predicted_velocity = .5 * (features[:, 1:, -6:-3] + features[:, :-1, -6:-3])
            six = features[:, :, 3:rotation_end].reshape(-1, 6)
            result["groups"][group] = {
                "key_fk_rmse_cm": float(np.sqrt(np.mean(key_distance ** 2))),
                "root_velocity_coordinate_rmse_mps": float(np.sqrt(np.mean((velocity - predicted_velocity) ** 2))),
                "six_d_unit_length_rmse": float(np.sqrt(np.mean((np.linalg.norm(six.reshape(-1, 2, 3), axis=-1) - 1) ** 2))),
                "six_d_abs_dot_mean": float(np.abs(np.sum(six[:, :3] * six[:, 3:], axis=-1)).mean())}
    atomic_write(data_dir / "decode_diagnostics.json", (json.dumps(result, indent=2) + "\n").encode())
    return result
