"""Sample frozen SMP priors, retain optional traces and cache native motion data.

The trace field named ``noise`` is the normalized input x_t at that diffusion
timestep, not the independently drawn Gaussian epsilon. Clean feature estimates
are kept unnormalized and decoded exactly like the repository's exporter.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import operator
from pathlib import Path
import pickle
import re
from typing import Any
from zipfile import BadZipFile

import numpy as np
import yaml

from ..data.storage import atomic_write
from ..paths import DEFAULT_CASE, ROOT, ensure_animation_imports


DEFAULT_MODEL = ROOT / "data/models/smp_priors/smp_prior_spinkick.pt"
DEFAULT_CONFIG = ROOT / "tools/diffusion_model/config/tinymdm_single_clip.yaml"
DEFAULT_OUTPUT = DEFAULT_CASE / "assets/data/smp"
SAMPLING_SCHEMA = 1


@dataclass(frozen=True)
class SamplingConfig:
    model_path: Path = DEFAULT_MODEL
    config_path: Path = DEFAULT_CONFIG
    output_dir: Path = DEFAULT_OUTPUT
    num_samples: int = 512
    batch_size: int = 64
    seed: int = 0
    device: str = "cuda:0"
    use_ema: bool = True
    trace_steps: tuple[int, ...] = (49, 39, 29, 19, 9, 0)
    force: bool = False
    version: str = "v1"


@dataclass(frozen=True)
class StageSamples:
    noise: np.ndarray
    features: np.ndarray
    frames: np.ndarray


@dataclass(frozen=True)
class SampleBundle:
    manifest_path: Path
    cache_hit: bool
    fps: float
    loop_mode: int
    features: np.ndarray
    frames: np.ndarray
    stages: dict[int, StageSamples]

    @property
    def cache_dir(self) -> Path:
        return self.manifest_path.parent


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _repo_path(path: Path | str) -> Path:
    path = Path(path)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _read_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def _positive_integer(value: int, name: str, minimum: int = 1) -> int:
    try:
        if isinstance(value, bool):
            raise TypeError(name)
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return result


def _sampling_inputs(config: SamplingConfig) -> tuple[dict, dict, Path, Path, dict]:
    model_path, config_path = _repo_path(config.model_path), _repo_path(config.config_path)
    prior = _read_yaml(config_path)
    env_path = _repo_path(prior["env_config"])
    env = _read_yaml(env_path)
    char_path = _repo_path(env["char_file"])
    count = _positive_integer(config.num_samples, "num_samples")
    batch = _positive_integer(config.batch_size, "batch_size")
    seed = _positive_integer(config.seed, "seed", 0)
    if not isinstance(config.version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", config.version):
        raise ValueError("Version must use letters, digits, _, . or - and start with a letter/digit.")
    horizon = _positive_integer(env["num_disc_obs_steps"], "num_disc_obs_steps", 2)
    total_steps = _positive_integer(prior["T"], "T")
    steps = tuple(sorted({_positive_integer(step, "trace timestep", 0)
                          for step in config.trace_steps}, reverse=True))
    if any(step >= total_steps for step in steps):
        raise ValueError(f"Trace timesteps must be in 0..{total_steps - 1}.")
    fps = float(prior["control_freq"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("control_freq must be finite and positive.")
    if not env.get("root_height_obs", True):
        raise ValueError("Native export requires root_height_obs=True; root height is otherwise absent.")
    if config.use_ema and not prior.get("model_ema", False):
        raise ValueError("This prior has no EMA model; select use_ema=False.")
    if prior.get("arch_name") != "DiT":
        raise ValueError("This sampler currently supports the repository's unconditional DiT priors.")
    identity = {"schema": SAMPLING_SCHEMA, "version": config.version,
                "model_sha256": _sha(model_path.read_bytes()),
                "config_sha256": _sha(config_path.read_bytes()),
                "env_sha256": _sha(env_path.read_bytes()),
                "character_sha256": _sha(char_path.read_bytes()),
                "num_samples": count, "batch_size": batch, "seed": seed,
                "device": str(config.device), "use_ema": bool(config.use_ema),
                "sampler": "ddpm", "diffusion_steps": total_steps,
                "trace_steps": list(steps), "horizon": horizon, "fps": fps,
                "decoder": "native_tan_norm_no_projection"}
    prior = dict(prior, env_config=str(env_path))
    return prior, env, model_path, char_path, identity


def _load_prior(prior: dict, env: dict, model_path: Path,
                char_path: Path, device: str) -> tuple[Any, Any, int]:
    """Load kinematics and the frozen model; never load an environment/policy."""
    import torch

    ensure_animation_imports()
    if char_path.suffix.lower() == ".xml":
        from anim.mjcf_char_model import MJCFCharModel
        character = MJCFCharModel(device)
    elif char_path.suffix.lower() == ".urdf":
        from anim.urdf_char_model import URDFCharModel
        character = URDFCharModel(device)
    else:
        raise ValueError("Sampling export supports MJCF .xml or URDF characters.")
    character.load(str(char_path))
    key_bodies = env.get("key_bodies", [])
    for body in key_bodies:
        character.get_body_id(body)
    channels = (3 + 6 + (character.get_num_joints() - 1) * 6
                + len(key_bodies) * 3 + 6
                + (character.get_dof_size() if env.get("disc_dof_vel_obs", False) else 0))
    configuration = dict(prior, input_dim=channels * env["num_disc_obs_steps"], input_channel=channels)
    from learning.tinymdm.tinymdm_model import TinyMDMModel
    model = TinyMDMModel(configuration, device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if (not torch.isfinite(model.obs_normalizer.get_mean()).all()
            or not torch.isfinite(model.obs_normalizer.get_std()).all()
            or (model.obs_normalizer.get_std() <= 0).any()):
        raise ValueError("Checkpoint has invalid feature normalization statistics.")
    return model, character, channels


def decode_features(features: Any, character: Any) -> Any:
    """Decode [B,H,C] using existing rotation semantics on the tensor's device.

    No Gram-Schmidt projection, clamping, key-position correction or velocity
    recomputation is applied to the retained raw features.
    """
    import torch

    ensure_animation_imports()
    import util.torch_util as torch_util
    if not isinstance(features, torch.Tensor) or features.ndim != 3:
        raise ValueError("Features must be a torch tensor of shape [B,H,C].")
    joint_count = character.get_num_joints() - 1
    rotation_end = 9 + joint_count * 6
    if features.shape[-1] < rotation_end or not torch.isfinite(features).all():
        raise ValueError("Generated features are non-finite or have an invalid channel count.")
    frames = torch.zeros((*features.shape[:2], 6 + character.get_dof_size()),
                         device=features.device, dtype=features.dtype)
    frames[..., :3] = features[..., :3]
    frames[..., 3:6] = torch_util.quat_to_exp_map(
        torch_util.tan_norm_to_quat(features[..., 3:9]))
    joints = features[..., 9:rotation_end].reshape(*features.shape[:2], joint_count, 6)
    frames[..., 6:] = character.rot_to_dof(torch_util.tan_norm_to_quat(joints))
    if not torch.isfinite(frames).all():
        raise ValueError("Generated features decode to non-finite native motion frames.")
    return frames


def _readonly(array: np.ndarray) -> np.ndarray:
    array.setflags(write=False)
    return array


def _read_bundle(manifest_path: Path, identity: dict) -> SampleBundle | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        data_path = manifest_path.parent / "samples.npz"
        payload = data_path.read_bytes()
        if (not isinstance(manifest, dict) or manifest.get("complete") is not True
                or manifest.get("identity") != identity or manifest.get("sha256") != _sha(payload)):
            return None
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            features = archive["features"].copy()
            frames = archive["frames"].copy()
            stages = {step: StageSamples(archive[f"t{step}_xt"].copy(),
                                        archive[f"t{step}_features"].copy(),
                                        archive[f"t{step}_frames"].copy())
                      for step in identity["trace_steps"]}
        expected = (identity["num_samples"], identity["horizon"])
        if (features.ndim != 3 or features.shape[:2] != expected
                or features.shape[-1] != manifest["feature_channels"]
                or frames.shape != (*expected, manifest["frame_channels"])):
            return None
        arrays = [features, frames]
        for stage in stages.values():
            if (stage.noise.shape != features.shape or stage.features.shape != features.shape
                    or stage.frames.shape != frames.shape):
                return None
            arrays.extend((stage.noise, stage.features, stage.frames))
        if not all(np.isfinite(array).all() for array in arrays):
            return None
        for array in arrays:
            _readonly(array)
        return SampleBundle(manifest_path, True, identity["fps"], 0, features, frames, stages)
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None


def _generate(config: SamplingConfig, prior: dict, env: dict, model_path: Path,
              char_path: Path, identity: dict, manifest_path: Path) -> SampleBundle:
    import torch

    device = torch.device(config.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Sampling device must be cpu or cuda.")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; choose device='cpu'.")
    horizon = identity["horizon"]
    accumulated: dict[str, list[np.ndarray]] = {"features": [], "frames": []}
    for step in identity["trace_steps"]:
        for suffix in ("xt", "features", "frames"):
            accumulated[f"t{step}_{suffix}"] = []
    cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=cuda_devices), torch.no_grad():
        model, character, channels = _load_prior(prior, env, model_path, char_path, str(device))
        torch.random.default_generator.manual_seed(identity["seed"])
        if device.type == "cuda":
            torch.cuda.default_generators[cuda_devices[0]].manual_seed(identity["seed"])
        sample = model.sample_ema if config.use_ema else model.sample
        selected_steps = set(identity["trace_steps"])
        for start in range(0, identity["num_samples"], identity["batch_size"]):
            batch_size = min(identity["batch_size"], identity["num_samples"] - start)
            trace: dict[int, tuple[Any, Any]] = {}

            def record(step: int, xt: Any, clean: Any) -> None:
                if step in selected_steps:
                    trace[step] = (xt.detach().clone(), clean.detach().clone())

            normalized = sample(shape=(horizon * channels,), batch_size=batch_size,
                                device=str(device), sampler="ddpm",
                                trace_callback=record if selected_steps else None)
            features = model.unnormalize(normalized.reshape(batch_size, horizon, channels))
            frames = decode_features(features, character)
            accumulated["features"].append(features.cpu().numpy().copy())
            accumulated["frames"].append(frames.cpu().numpy().copy())
            if set(trace) != selected_steps:
                raise RuntimeError("Sampler did not visit all requested trace timesteps.")
            for step, (xt, clean) in trace.items():
                clean_features = model.unnormalize(clean.reshape(batch_size, horizon, channels))
                stage_frames = decode_features(clean_features, character)
                accumulated[f"t{step}_xt"].append(xt.reshape(batch_size, horizon, channels).cpu().numpy().copy())
                accumulated[f"t{step}_features"].append(clean_features.cpu().numpy().copy())
                accumulated[f"t{step}_frames"].append(stage_frames.cpu().numpy().copy())
    arrays = {key: np.concatenate(chunks, axis=0) for key, chunks in accumulated.items()}
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Sampler produced non-finite trace values; no completed cache was written.")
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    payload = stream.getvalue()
    manifest = {"complete": True, "identity": identity, "sha256": _sha(payload),
                "model_path": str(model_path), "config_path": str(_repo_path(config.config_path)),
                "character_path": str(char_path), "feature_channels": channels,
                "frame_channels": arrays["frames"].shape[-1], "loop_mode": 0,
                "trace_noise_meaning": "normalized input x_t, not Gaussian epsilon",
                "trace_features_meaning": "unnormalized predicted clean x_0",
                "decoder": "existing tan_norm_to_quat; no rotation projection"}
    atomic_write(manifest_path.parent / "samples.npz", payload)
    atomic_write(manifest_path, _json(manifest))
    result = _read_bundle(manifest_path, identity)
    if result is None:
        raise RuntimeError("Sampling cache did not pass integrity validation after writing.")
    return SampleBundle(result.manifest_path, False, result.fps, result.loop_mode,
                        result.features, result.frames, result.stages)


def sample_prior(config: SamplingConfig) -> SampleBundle:
    """Return a verified cache hit or run the frozen prior's ordinary DDPM sampler."""
    prior, env, model_path, char_path, identity = _sampling_inputs(config)
    key = _sha(_json(identity))[:20]
    manifest_path = _repo_path(config.output_dir) / f"SMP_{config.version}_{key}" / "manifest.json"
    if not config.force:
        cached = _read_bundle(manifest_path, identity)
        if cached is not None:
            return cached
    return _generate(config, prior, env, model_path, char_path, identity, manifest_path)


def export_clip(bundle: SampleBundle, index: int, stage: int | None = None) -> Path:
    """Export one final sample or a traced clean estimate as native CLAMP motion."""
    index = _positive_integer(index, "sample index", 0)
    if index >= len(bundle.frames):
        raise ValueError(f"Sample index must be in 0..{len(bundle.frames) - 1}.")
    if stage is not None:
        stage = _positive_integer(stage, "stage", 0)
        if stage not in bundle.stages:
            raise ValueError(f"No trace exists for timestep {stage}.")
    frames = bundle.frames[index] if stage is None else bundle.stages[stage].frames[index]
    tag = "final" if stage is None else f"t{stage:03d}"
    path = bundle.cache_dir / "motions" / f"sample_{index:04d}_{tag}.pkl"
    payload = pickle.dumps({"fps": bundle.fps, "loop_mode": bundle.loop_mode,
                            "frames": frames.tolist()}, protocol=pickle.HIGHEST_PROTOCOL)
    checksum = _sha(payload)
    manifest_path = path.with_suffix(".manifest.json")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (isinstance(manifest, dict) and manifest.get("complete") is True
                and manifest.get("sha256") == checksum and _sha(path.read_bytes()) == checksum):
            return path
    except (OSError, ValueError, TypeError):
        pass
    atomic_write(path, payload)
    atomic_write(manifest_path, _json({"complete": True, "sha256": checksum,
                                      "sample_index": index, "stage": stage}))
    return path
