"""Resample native motions on a regular clock using MimicKit interpolation."""

import hashlib
from pathlib import Path

import numpy as np

from .motion import MotionData, motion_bytes, read_motion
from ..paths import ensure_animation_imports


def resample_motion(source_path: Path, character_path: Path, *, fps: float = 30.0,
                    output_path: Path | None = None) -> MotionData:
    """Sample within the source duration; never append an irregular end frame."""
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Resampling FPS must be finite and positive.")
    source = read_motion(source_path)
    target_path = Path(output_path or source.path).resolve()
    if fps == source.fps:
        frames = source.frames.copy()
    else:
        import torch

        ensure_animation_imports()
        from anim.motion_lib import MotionLib
        from util.torch_util import quat_to_exp_map

        character_path = Path(character_path).resolve()
        if character_path.suffix.lower() == ".xml":
            from anim.mjcf_char_model import MJCFCharModel
            model = MJCFCharModel("cpu")
        elif character_path.suffix.lower() == ".urdf":
            from anim.urdf_char_model import URDFCharModel
            model = URDFCharModel("cpu")
        else:
            raise ValueError("Offline resampling requires an MJCF or URDF character.")
        model.load(str(character_path))
        if model.get_dof_size() != source.frames.shape[1] - 6:
            raise ValueError("Character DOF count does not match the source motion.")
        duration = (len(source.frames) - 1) / source.fps
        count = int(np.floor(duration * fps + 1e-9)) + 1
        if count < 2:
            raise ValueError("The requested FPS leaves fewer than two frames.")
        library = MotionLib(str(source.path), model, "cpu")
        with torch.no_grad():
            times = torch.arange(count, dtype=torch.float32) / fps
            ids = torch.zeros(count, dtype=torch.long)
            pos, rot, _, _, joints, _ = library.calc_motion_frame(ids, times)
            frames = torch.cat((pos, quat_to_exp_map(rot), model.rot_to_dof(joints)), dim=-1).numpy()
    frames = np.asarray(frames, dtype=np.float64)
    if not np.isfinite(frames).all():
        raise ValueError("Resampling produced non-finite motion frames.")
    result = MotionData(target_path, frames, float(fps), source.loop_mode, "")
    return MotionData(target_path, frames, float(fps), source.loop_mode,
                      hashlib.sha256(motion_bytes(result, frames)).hexdigest())
