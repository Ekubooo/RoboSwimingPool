"""Read and write native MimicKit motion data without a simulator."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import pickle

import numpy as np


@dataclass(frozen=True)
class MotionData:
    path: Path
    frames: np.ndarray
    fps: float
    loop_mode: int
    content_hash: str


def read_motion(path: Path) -> MotionData:
    """Read an existing native motion, preserving its source sampling rate."""
    path = Path(path).resolve()
    raw = path.read_bytes()
    data = pickle.loads(raw)
    if not isinstance(data, dict) or not {"frames", "fps", "loop_mode"} <= data.keys():
        raise ValueError("Expected a native motion with frames, fps and loop_mode.")
    source_frames = data["frames"]
    if hasattr(source_frames, "detach"):
        source_frames = source_frames.detach().cpu().numpy()
    frames = np.asarray(source_frames, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] < 7:
        raise ValueError("Motion frames must have shape [T, 6 + D], with T >= 2 and D >= 1.")
    invalid = ~np.isfinite(frames)
    if invalid.any():
        invalid_frames = int(np.count_nonzero(invalid.any(axis=1)))
        columns = np.flatnonzero(invalid.any(axis=0)).tolist()
        examples = np.argwhere(invalid)[:5].tolist()
        raise ValueError(
            f"Motion contains non-finite frame values: {path}. "
            f"NaN={int(np.isnan(frames).sum())}, Inf={int(np.isinf(frames).sum())}; "
            f"affected frames={invalid_frames}/{len(frames)}; "
            f"columns (0-based)={columns}; first [frame, column] indices (0-based)={examples}. "
            "Invalid values are already present in the input, before PCA sample construction. "
            "Use a valid source motion or repair/re-export it from its original data."
        )
    fps = float(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Motion fps must be finite and positive.")
    loop_mode = data["loop_mode"]
    if loop_mode not in (0, 1):
        raise ValueError("Motion loop_mode must be 0 (CLAMP) or 1 (WRAP).")
    return MotionData(path, frames.copy(), fps, int(loop_mode), hashlib.sha256(raw).hexdigest())


def motion_bytes(source: MotionData, frames: np.ndarray) -> bytes:
    """Serialize reconstructed frames using MimicKit's existing pickle schema."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.shape != source.frames.shape or not np.isfinite(frames).all():
        raise ValueError("Reconstruction must be finite and match the source frame shape.")
    return pickle.dumps({"loop_mode": source.loop_mode, "fps": source.fps,
                         "frames": frames.tolist()}, protocol=pickle.HIGHEST_PROTOCOL)


def discontinuity_diagnostics(source: MotionData) -> dict:
    """Flag possible parameter jumps; do not alter angle/exp-map branches."""
    delta = np.abs(np.diff(source.frames[:, 6:], axis=0))
    return {"max_dof_step_rad": float(delta.max()),
            "possible_branch_jump_count": int(np.count_nonzero(delta > np.pi)),
            "branch_jump_threshold_rad": float(np.pi)}
