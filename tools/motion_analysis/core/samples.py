"""Build pose/window sample matrices and restore their original timeline."""

from dataclasses import dataclass
import operator

import numpy as np


@dataclass(frozen=True)
class SampleLayout:
    mode: str
    frame_count: int
    feature_count: int
    window: int
    stride: int
    starts: np.ndarray


def build_samples(features: np.ndarray, mode: str = "pose", window: int = 10,
                  stride: int = 1, *,
                  allow_single_window: bool = False) -> tuple[np.ndarray, SampleLayout]:
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or features.shape[1] < 1:
        raise ValueError("Features require at least two frames and one feature.")
    if not np.isfinite(features).all():
        raise ValueError("Features must be finite.")
    if mode == "pose":
        starts = np.arange(len(features), dtype=np.int64)
        return features.copy(), SampleLayout(mode, *features.shape, 1, 1, starts)
    if mode != "window":
        raise ValueError("PCA mode must be pose or window.")
    try:
        window, stride = operator.index(window), operator.index(stride)
    except TypeError as exc:
        raise ValueError("Window and stride must be integers.") from exc
    if not 1 <= stride <= window <= len(features):
        raise ValueError("Window mode requires 1 <= stride <= window <= frame count.")
    starts = np.arange(0, len(features) - window + 1, stride, dtype=np.int64)
    final_start = len(features) - window
    if starts[-1] != final_start:
        starts = np.append(starts, final_start)
    if len(starts) < 2 and not allow_single_window:
        raise ValueError("PCA requires at least two windows; reduce the window length.")
    samples = np.stack([features[start:start + window].reshape(-1) for start in starts])
    return samples, SampleLayout(mode, *features.shape, window, stride, starts)


def restore_samples(samples: np.ndarray, layout: SampleLayout) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    expected = (len(layout.starts), layout.window * layout.feature_count)
    if samples.shape != expected or not np.isfinite(samples).all():
        raise ValueError(f"Reconstructed samples must be finite with shape {expected}.")
    if layout.mode == "pose":
        return samples.copy()
    result = np.zeros((layout.frame_count, layout.feature_count), dtype=np.float64)
    count = np.zeros(layout.frame_count, dtype=np.int64)
    windows = samples.reshape(-1, layout.window, layout.feature_count)
    for start, window in zip(layout.starts, windows):
        result[start:start + layout.window] += window
        count[start:start + layout.window] += 1
    if np.any(count == 0):
        raise ValueError("Window layout leaves uncovered source frames.")
    return result / count[:, None]
