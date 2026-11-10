"""Prepare a frozen SMP case and saved PCA collections without opening a viewer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ..paths import DEFAULT_CASE, ROOT

from ..core.collection import fit_collection, load_collection, project_collection, save_collection
from ..data.motion import MotionData, motion_bytes, read_motion
from ..data.resampling import resample_motion
from ..data.storage import atomic_write
from ..smp.sampling import SamplingConfig, sample_prior


CASE_VERSION = 1


def _json(value: dict) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clips(frames: np.ndarray, directory: Path, fps: float, loop_mode: int) -> tuple[MotionData, ...]:
    result = []
    for index, array in enumerate(frames):
        array = np.asarray(array, dtype=np.float64)
        clip = MotionData(directory / f"sample_{index:04d}.pkl", array, fps, loop_mode, "")
        result.append(MotionData(clip.path, array, fps, loop_mode,
                                 hashlib.sha256(motion_bytes(clip, array)).hexdigest()))
    return tuple(result)


def prepare_case(case_dir: Path, *, num_samples: int = 512, batch_size: int = 64,
                 seed: int = 0, device: str = "cuda:0", force: bool = False) -> Path:
    """Reuse verified artifacts, otherwise fit independent and reference bases."""
    case_dir = Path(case_dir).resolve()
    data_dir = case_dir / "assets/data"
    source_path = ROOT / "data/motions/humanoid/humanoid_spinkick.pkl"
    char_path = ROOT / "data/assets/humanoid/humanoid.xml"
    sample_config = SamplingConfig(output_dir=data_dir / "smp", num_samples=num_samples,
                                   batch_size=batch_size, seed=seed, device=device, force=force)
    bundle = sample_prior(sample_config)
    print(f"SMP sampling cache: {'hit' if bundle.cache_hit else 'computed'}", flush=True)
    index_path = data_dir / "analysis.json"
    metadata_path = data_dir / "case.json"
    identity = {"case_version": CASE_VERSION, "reference_sha256": _hash(source_path),
                "character_sha256": _hash(char_path), "sampling_manifest_sha256": _hash(bundle.manifest_path),
                "analysis_fps": 30, "window": 10, "stride": 1,
                "feature": "native_dof", "centering": True, "whitening": False}
    groups = ("reference", "generated", *(f"stage_{step}" for step in bundle.stages))
    expected = [(group, mode, basis) for mode in ("pose", "window") for group in groups
                for basis in (("shared", "independent") if group in ("reference", "generated") else ("shared",))]
    if not force:
        try:
            previous = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(previous, dict) and previous.get("identity") == identity:
                for group, mode, basis in expected:
                    load_collection(index_path, group=group, mode=mode, basis=basis)
                print(f"PCA analysis cache: hit ({len(expected)} collections)", flush=True)
                return index_path
        except (OSError, ValueError, KeyError, TypeError):
            pass
    reference = resample_motion(source_path, char_path, fps=30,
                                output_path=data_dir / "reference_30fps.pkl")
    atomic_write(reference.path, motion_bytes(reference, reference.frames))
    collections = {"reference": (reference,),
                   "generated": _clips(bundle.frames, data_dir / "generated", bundle.fps, bundle.loop_mode)}
    collections.update({f"stage_{step}": _clips(stage.frames, data_dir / f"stage_{step}",
                                               bundle.fps, bundle.loop_mode)
                        for step, stage in bundle.stages.items()})
    # This case owns the complete index. Start a fresh atomic index on a rebuild,
    # including recovery from a corrupt JSON index. Content-addressed arrays and
    # models remain reusable and are verified again by save_collection().
    atomic_write(index_path, _json({"schema_version": 1, "complete": True, "entries": []}))
    for mode in ("pose", "window"):
        fitted = {}
        for group in ("reference", "generated"):
            analysis = fit_collection(collections[group], mode=mode, window=10, stride=1)
            fitted[group] = analysis
            save_collection(analysis, index_path, group=group, basis="independent")
            print(f"Fitted {group}/{mode}: {analysis.samples.shape}, rank={analysis.max_components}", flush=True)
        for group, clips in collections.items():
            analysis = project_collection(clips, fitted["reference"].model,
                                          mode=mode, window=10, stride=1)
            save_collection(analysis, index_path, group=group, basis="shared")
    original = read_motion(source_path)
    metadata = {"identity": identity, "source_path": str(source_path), "character_path": str(char_path),
                "source_frames": len(original.frames), "source_fps": original.fps,
                "source_duration": (len(original.frames) - 1) / original.fps,
                "reference_frames": len(reference.frames), "reference_fps": reference.fps,
                "reference_duration": (len(reference.frames) - 1) / reference.fps,
                "sampling_manifest": bundle.manifest_path.relative_to(data_dir).as_posix(),
                "analysis_manifest": index_path.name, "groups": list(groups),
                "clip_count": num_samples, "seed": seed, "batch_size": batch_size,
                "stage_meaning": "predicted clean x_0 at the recorded input diffusion timestep",
                "policy_rollouts": "not included; no Newton-compatible policy is exercised"}
    atomic_write(metadata_path, _json(metadata))
    print(f"Saved analysis index: {index_path}", flush=True)
    return index_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    try:
        prepare_case(args.case, num_samples=args.num_samples, batch_size=args.batch_size,
                     seed=args.seed, device=args.device, force=args.force)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Case preparation error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
