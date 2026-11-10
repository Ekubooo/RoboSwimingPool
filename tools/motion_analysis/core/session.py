"""A motion PCA session with independent fitting and reconstruction caches."""

from collections import OrderedDict
import hashlib
import io
import json
from pathlib import Path
import pickle
import re
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from ..data.storage import atomic_write
from ..data.motion import MotionData, motion_bytes, read_motion
from .pca import PCAModel, fit_pca
from .samples import SampleLayout, build_samples, restore_samples
from .selection import normalize_indices


PROCESSING_VERSION = 1


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(data: dict) -> bytes:
    return (json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


class AnalysisSession:
    """Own the original timeline, cached fit and on-demand reconstructions."""

    def __init__(self, source: MotionData, layout: SampleLayout, model: PCAModel,
                 scores: np.ndarray, cache_dir: Path, fit_key: str,
                 fit_cache_hit: bool, version: str, force: bool = False):
        self.source = source
        self.layout = layout
        self.model = model
        self.scores = scores
        self.cache_dir = cache_dir
        self.fit_key = fit_key
        self.fit_cache_hit = fit_cache_hit
        self.version = version
        self.force = force
        self._memory: OrderedDict[tuple[int, ...], np.ndarray] = OrderedDict()
        self._forced_selections: set[tuple[int, ...]] = set()

    @property
    def max_components(self) -> int:
        return self.model.max_components

    @property
    def variance_ratio(self) -> np.ndarray:
        return self.model.variance_ratio

    def reconstruct(self, indices_0based: Sequence[int]) -> np.ndarray:
        """Reconstruct a selection in memory; never write a preview to disk."""
        indices = normalize_indices(indices_0based, self.max_components)
        if indices not in self._memory:
            frames = self._read_reconstruction(indices)
            if frames is None:
                samples = self.model.inverse_transform(self.scores, indices)
                frames = self.source.frames.copy()
                frames[:, 6:] = restore_samples(samples, self.layout)
            self._remember(indices, frames)
        self._memory.move_to_end(indices)
        return self._memory[indices].copy()

    def _remember(self, indices: tuple[int, ...], frames: np.ndarray) -> None:
        self._memory[indices] = frames
        self._memory.move_to_end(indices)
        if len(self._memory) > 8:
            self._memory.popitem(last=False)

    def _selection_identity(self, indices: tuple[int, ...]) -> dict:
        return {"processing_version": PROCESSING_VERSION, "fit_key": self.fit_key,
                "indices": list(indices)}

    def _read_reconstruction(self, indices: tuple[int, ...]) -> np.ndarray | None:
        if self.force and indices not in self._forced_selections:
            return None
        path = self.reconstruction_path(indices)
        try:
            manifest = json.loads(path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
            payload = path.read_bytes()
            if (not isinstance(manifest, dict) or manifest.get("complete") is not True
                    or manifest.get("identity") != self._selection_identity(indices)
                    or manifest.get("sha256") != _digest(payload)):
                return None
            data = pickle.loads(payload)
            if not isinstance(data, dict):
                return None
            frames = np.asarray(data["frames"], dtype=np.float64)
            if (frames.shape != self.source.frames.shape or not np.isfinite(frames).all()
                    or data["fps"] != self.source.fps or data["loop_mode"] != self.source.loop_mode
                    or not np.array_equal(frames[:, :6], self.source.frames[:, :6])):
                return None
            return frames
        except (OSError, ValueError, TypeError, KeyError, EOFError, pickle.UnpicklingError):
            return None

    def reconstruction_path(self, indices_0based: Sequence[int]) -> Path:
        indices = normalize_indices(indices_0based, self.max_components)
        key = _digest(_json_bytes({"fit_key": self.fit_key, "indices": list(indices)}))[:16]
        return self.cache_dir / f"{self.source.path.stem}_PCA_{self.version}_{key}.pkl"

    def save_reconstruction(self, indices_0based: Sequence[int]) -> Path:
        """Reuse a valid native motion, or atomically persist this selection."""
        indices = normalize_indices(indices_0based, self.max_components)
        path = self.reconstruction_path(indices)
        manifest_path = path.with_suffix(".manifest.json")
        expected = self._selection_identity(indices)
        cached = self._read_reconstruction(indices)
        if cached is not None:
            self._remember(indices, cached)
            return path
        payload = motion_bytes(self.source, self.reconstruct(indices))
        atomic_write(path, payload)
        atomic_write(manifest_path, _json_bytes(
            {"complete": True, "identity": expected, "sha256": _digest(payload)}))
        self._forced_selections.add(indices)
        return path


def _encode_fit(model: PCAModel, scores: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, mean=model.mean, components=model.components,
                        variance=model.variance, variance_ratio=model.variance_ratio,
                        sample_count=np.array(model.sample_count), scores=scores)
    return buffer.getvalue()


def _read_fit(path: Path, manifest_path: Path, identity: dict,
              sample_shape: tuple[int, int]) -> tuple[PCAModel, np.ndarray] | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload = path.read_bytes()
        if (not isinstance(manifest, dict) or manifest.get("complete") is not True
                or manifest.get("identity") != identity
                or manifest.get("sha256") != _digest(payload)):
            return None
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            model = PCAModel(archive["mean"].copy(), archive["components"].copy(),
                             archive["variance"].copy(), archive["variance_ratio"].copy(),
                             int(archive["sample_count"]))
            scores = archive["scores"].copy()
        rank = model.max_components
        if (model.mean.shape != (sample_shape[1],)
                or model.components.shape != (rank, sample_shape[1])
                or model.variance.shape != (rank,) or model.variance_ratio.shape != (rank,)
                or scores.shape != (sample_shape[0], rank)
                or model.sample_count != sample_shape[0]
                or rank > min(sample_shape[0] - 1, sample_shape[1])):
            return None
        if not all(np.isfinite(array).all() for array in
                   (model.mean, model.components, model.variance, model.variance_ratio, scores)):
            return None
        return model, scores
    except (OSError, ValueError, TypeError, KeyError, EOFError, BadZipFile):
        return None


def load_session(source_path: Path, *, mode: str = "pose", window: int = 10,
                 stride: int = 1, version: str = "v1", cache_root: Path,
                 force: bool = False) -> AnalysisSession:
    """Load or fit one source clip. K and display settings do not affect this cache."""
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", version):
        raise ValueError("Version must start with a letter/digit and use letters, digits, _, . or -.")
    source = read_motion(source_path)
    samples, layout = build_samples(source.frames[:, 6:], mode, window, stride)
    identity = {"processing_version": PROCESSING_VERSION, "version": version,
                "source_sha256": source.content_hash, "feature": "native_dof",
                "mode": mode, "window": layout.window, "stride": layout.stride,
                "dtype": "float64", "centering": True, "whitening": False}
    fit_key = _digest(_json_bytes(identity))
    cache_dir = Path(cache_root).resolve() / source.path.stem / f"PCA_{version}_{fit_key[:16]}"
    fit_path, manifest_path = cache_dir / "pca_model.npz", cache_dir / "manifest.json"
    cached = None if force else _read_fit(fit_path, manifest_path, identity, samples.shape)
    cache_hit = cached is not None
    if cached is None:
        model = fit_pca(samples)
        scores = model.transform(samples)
        payload = _encode_fit(model, scores)
        atomic_write(fit_path, payload)
        atomic_write(manifest_path, _json_bytes({
            "complete": True, "identity": identity, "sha256": _digest(payload),
            "source_path": str(source.path), "source_shape": list(source.frames.shape),
            "fps": source.fps, "loop_mode": source.loop_mode,
            "sample_shape": list(samples.shape), "max_components": model.max_components,
            "window_starts": layout.starts.tolist()}))
    else:
        model, scores = cached
    return AnalysisSession(source, layout, model, scores, cache_dir, fit_key,
                           cache_hit, version, force)
