"""PCA over independent motion clips and portable saved-analysis collections.

Window extraction never crosses clip boundaries. Saved collections embed source
frames and projections so playback does not need source files or another fit.
"""

from dataclasses import dataclass, field
import hashlib
import io
import json
import operator
from pathlib import Path
import re
from typing import Sequence
from zipfile import BadZipFile

import numpy as np

from ..data.motion import MotionData
from .pca import PCAModel, fit_pca
from .samples import SampleLayout, build_samples
from .session import AnalysisSession
from ..data.storage import atomic_write
from ..paths import CACHE_ROOT


SCHEMA_VERSION = 1
DEFAULT_CACHE = CACHE_ROOT / "collections"
_SAFE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def _json_bytes(value: dict) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _array_hash(array: np.ndarray) -> str:
    array = np.asarray(array, dtype="<f8", order="C")
    shape = json.dumps(list(array.shape)).encode("ascii")
    return _sha256(shape + b"\n" + array.tobytes())


def _model_key(model: PCAModel) -> str:
    return _sha256(_json_bytes({
        "sample_count": model.sample_count,
        **{name: _array_hash(getattr(model, name))
           for name in ("mean", "components", "variance", "variance_ratio")},
    }))


def _validate_model(model: PCAModel, width: int) -> None:
    rank = model.max_components
    if (model.mean.shape != (width,) or model.components.shape != (rank, width)
            or model.variance.shape != (rank,) or model.variance_ratio.shape != (rank,)
            or model.sample_count < 2 or rank > min(width, model.sample_count - 1)):
        raise ValueError("Saved PCA model has inconsistent dimensions or sample count.")
    arrays = (model.mean, model.components, model.variance, model.variance_ratio)
    if not all(np.isfinite(array).all() for array in arrays):
        raise ValueError("PCA model arrays must be finite.")
    if (np.any(model.variance < 0) or np.any(model.variance_ratio < 0)
            or np.any(model.variance_ratio > 1 + 1e-10)
            or float(model.variance_ratio.sum()) > 1 + 1e-9):
        raise ValueError("PCA model variances must be nonnegative valid proportions.")
    if not np.allclose(model.components @ model.components.T, np.eye(rank),
                       rtol=1e-7, atol=1e-9):
        raise ValueError("PCA model components must be orthonormal.")


def _validate_clip(clip: MotionData) -> None:
    frames = np.asarray(clip.frames)
    if (frames.ndim != 2 or frames.shape[0] < 2 or frames.shape[1] < 7
            or not np.isfinite(frames).all()):
        raise ValueError(f"Collection clip must contain finite [T, 6+D] frames: {clip.path}")
    if not np.isfinite(clip.fps) or clip.fps <= 0 or clip.loop_mode not in (0, 1):
        raise ValueError(f"Invalid FPS or loop mode in collection clip: {clip.path}")


def _samples_for_clips(clips: Sequence[MotionData], mode: str, window: int,
                       stride: int) -> tuple[tuple[MotionData, ...],
                                              tuple[SampleLayout, ...],
                                              tuple[np.ndarray, ...]]:
    incoming = tuple(clips)
    if not incoming:
        raise ValueError("A motion collection must contain at least one clip.")
    normalized, layouts, samples = [], [], []
    width = None
    for clip in incoming:
        _validate_clip(clip)
        # Session reconstruction assigns floating projections into source frames.
        # Normalize externally constructed MotionData as well as read_motion data.
        clip = MotionData(Path(clip.path), np.array(clip.frames, dtype=np.float64, copy=True),
                          float(clip.fps), int(clip.loop_mode), clip.content_hash)
        matrix, layout = build_samples(clip.frames[:, 6:], mode, window, stride,
                                       allow_single_window=True)
        if width is not None and matrix.shape[1] != width:
            raise ValueError("All collection clips must use the same pose feature dimensions.")
        width = matrix.shape[1]
        normalized.append(clip)
        layouts.append(layout)
        samples.append(matrix)
    return tuple(normalized), tuple(layouts), tuple(samples)


@dataclass
class CollectionAnalysis:
    """One PCA basis, independent clip timelines, and per-clip projections."""

    clips: tuple[MotionData, ...]
    layouts: tuple[SampleLayout, ...]
    scores: tuple[np.ndarray, ...]
    model: PCAModel
    samples: np.ndarray
    fitted_on_collection: bool
    source_ids: tuple[str, ...]
    cache_dir: Path
    fit_cache_hit: bool = False
    group: str = "collection"
    basis: str = "independent"
    _sessions: dict[int, AnalysisSession] = field(default_factory=dict, init=False, repr=False)

    @property
    def mode(self) -> str:
        return self.layouts[0].mode

    @property
    def max_components(self) -> int:
        return self.model.max_components

    def for_clip(self, index: int) -> AnalysisSession:
        """Adapt one saved/projected clip to the existing interactive viewer."""
        if isinstance(index, bool):
            raise ValueError("Clip index must be an integer.")
        try:
            index = operator.index(index)
        except TypeError as exc:
            raise ValueError("Clip index must be an integer.") from exc
        if not 0 <= index < len(self.clips):
            raise ValueError(f"Clip index must be in 0..{len(self.clips) - 1}.")
        if index not in self._sessions:
            source, layout = self.clips[index], self.layouts[index]
            identity = {"model": _model_key(self.model), "source_id": self.source_ids[index],
                        "frames": _array_hash(source.frames), "fps": source.fps,
                        "loop_mode": source.loop_mode, "mode": layout.mode,
                        "window": layout.window, "stride": layout.stride}
            session = AnalysisSession(
                source, layout, self.model, self.scores[index],
                self.cache_dir / f"clip_{index:04d}", _sha256(_json_bytes(identity)),
                self.fit_cache_hit, "collection_v1")
            session.source_id = self.source_ids[index]
            self._sessions[index] = session
        return self._sessions[index]


def _analysis(clips: tuple[MotionData, ...], layouts: tuple[SampleLayout, ...],
              matrices: tuple[np.ndarray, ...], model: PCAModel,
              fitted: bool) -> CollectionAnalysis:
    _validate_model(model, matrices[0].shape[1])
    scores = tuple(model.transform(matrix) for matrix in matrices)
    identity = _sha256(_json_bytes({"model": _model_key(model),
                                    "sources": [_array_hash(clip.frames) for clip in clips]}))
    return CollectionAnalysis(
        clips, layouts, scores, model, np.concatenate(matrices), fitted,
        tuple(f"{index:04d}:{clip.path.name}" for index, clip in enumerate(clips)),
        DEFAULT_CACHE / identity[:16])


def fit_collection(clips: Sequence[MotionData], *, mode: str = "pose",
                   window: int = 10, stride: int = 1) -> CollectionAnalysis:
    """Fit one basis over all clips; a single window per clip is supported."""
    clips, layouts, matrices = _samples_for_clips(clips, mode, window, stride)
    samples = np.concatenate(matrices)
    model = fit_pca(samples)  # The total collection must contain at least two samples.
    return _analysis(clips, layouts, matrices, model, True)


def project_collection(clips: Sequence[MotionData], model: PCAModel, *,
                       mode: str = "pose", window: int = 10,
                       stride: int = 1) -> CollectionAnalysis:
    """Project another collection on an existing basis without refitting it."""
    clips, layouts, matrices = _samples_for_clips(clips, mode, window, stride)
    return _analysis(clips, layouts, matrices, model, False)


def _archive_bytes(**arrays) -> bytes:
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    return buffer.getvalue()


def _offsets(lengths: Sequence[int]) -> np.ndarray:
    return np.concatenate(([0], np.cumsum(lengths, dtype=np.int64)))


def _model_from_bytes(payload: bytes) -> PCAModel:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        count = archive["sample_count"]
        if count.shape != () or count.dtype.kind not in "iu":
            raise ValueError("Saved model sample_count must be a scalar integer.")
        model = PCAModel(
            archive["mean"].copy(), archive["components"].copy(),
            archive["variance"].copy(), archive["variance_ratio"].copy(), int(count))
    _validate_model(model, len(model.mean))
    return model


def _safe_name(value: str, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_NAME.fullmatch(value):
        raise ValueError(f"{label} must use letters, digits, underscores, dots or hyphens.")
    return value


def _read_index(path: Path) -> dict:
    try:
        index = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read saved PCA analysis index: {path}") from exc
    if (not isinstance(index, dict) or index.get("schema_version") != SCHEMA_VERSION
            or index.get("complete") is not True or not isinstance(index.get("entries"), list)
            or any(not isinstance(entry, dict) for entry in index["entries"])):
        raise ValueError("Saved PCA analysis index is incomplete or has an unsupported schema.")
    return index


def save_collection(analysis: CollectionAnalysis, manifest_path: Path, *, group: str,
                    basis: str = "shared", source_ids: Sequence[str] | None = None,
                    env_config: Path | None = None,
                    engine_config: Path | None = None) -> dict:
    """Append/replace one group+mode+basis in an atomic, portable analysis index."""
    group = _safe_name(group, "Group")
    if basis not in ("shared", "independent"):
        raise ValueError("Basis must be shared or independent.")
    ids = tuple(source_ids) if source_ids is not None else analysis.source_ids
    if (len(ids) != len(analysis.clips) or any(not isinstance(item, str) or not item for item in ids)
            or len(set(ids)) != len(ids)):
        raise ValueError("source_ids must contain one unique nonempty string per clip.")
    manifest_path = Path(manifest_path).resolve()
    root = manifest_path.parent
    index = (_read_index(manifest_path) if manifest_path.exists()
             else {"schema_version": SCHEMA_VERSION, "complete": True, "entries": []})
    model_key = _model_key(analysis.model)
    model_path = root / "models" / f"{model_key}.npz"
    model_payload = None
    if model_path.is_file():
        existing = model_path.read_bytes()
        try:
            if _model_key(_model_from_bytes(existing)) == model_key:
                model_payload = existing
        except (ValueError, TypeError, KeyError, EOFError, BadZipFile):
            pass
    if model_payload is None:
        model_payload = _archive_bytes(
            mean=analysis.model.mean, components=analysis.model.components,
            variance=analysis.model.variance, variance_ratio=analysis.model.variance_ratio,
            sample_count=np.array(analysis.model.sample_count, dtype=np.int64))
        atomic_write(model_path, model_payload)

    motion_payload = _archive_bytes(
        frames=np.concatenate([clip.frames for clip in analysis.clips]),
        frame_offsets=_offsets([len(clip.frames) for clip in analysis.clips]))
    motion_hash = _sha256(motion_payload)
    motion_path = root / "motions" / f"{motion_hash}.npz"
    if not motion_path.is_file() or _sha256(motion_path.read_bytes()) != motion_hash:
        atomic_write(motion_path, motion_payload)
    payload = _archive_bytes(
        scores=np.concatenate(analysis.scores),
        score_offsets=_offsets([len(scores) for scores in analysis.scores]),
        starts=np.concatenate([layout.starts for layout in analysis.layouts]),
        start_offsets=_offsets([len(layout.starts) for layout in analysis.layouts]))
    payload_hash = _sha256(payload)
    data_path = root / "collections" / f"{group}_{analysis.mode}_{basis}_{payload_hash[:16]}.npz"
    if not data_path.is_file() or _sha256(data_path.read_bytes()) != payload_hash:
        atomic_write(data_path, payload)

    layout = analysis.layouts[0]
    entry = {"group": group, "mode": analysis.mode, "basis": basis,
             "feature": "native_dof", "window": layout.window, "stride": layout.stride,
             "fitted_on_collection": analysis.fitted_on_collection,
             "model": {"path": model_path.relative_to(root).as_posix(),
                       "sha256": _sha256(model_payload), "identity": model_key},
             "motions": {"path": motion_path.relative_to(root).as_posix(), "sha256": motion_hash},
             "data": {"path": data_path.relative_to(root).as_posix(), "sha256": payload_hash},
             "source_ids": list(ids), "clips": [
                 {"source_id": source_id, "path": str(clip.path),
                  "content_hash": clip.content_hash, "frames_sha256": _array_hash(clip.frames),
                  "shape": list(clip.frames.shape), "fps": clip.fps,
                  "loop_mode": clip.loop_mode}
                 for source_id, clip in zip(ids, analysis.clips)]}
    key = (group, analysis.mode, basis)
    index["entries"] = [item for item in index["entries"]
                        if (item.get("group"), item.get("mode"), item.get("basis")) != key]
    index["entries"].append(entry)
    playback = index.setdefault("playback", {})
    if env_config is not None:
        playback["env_config"] = str(Path(env_config).resolve())
    if engine_config is not None:
        playback["engine_config"] = str(Path(engine_config).resolve())
    atomic_write(manifest_path, _json_bytes(index))
    analysis.group, analysis.basis, analysis.source_ids = group, basis, ids
    analysis.cache_dir = root / "reconstructions" / group / analysis.mode / basis
    analysis._sessions.clear()
    return entry


def _read_artifact(root: Path, description: dict) -> bytes:
    if (not isinstance(description, dict) or not isinstance(description.get("path"), str)
            or not isinstance(description.get("sha256"), str)):
        raise ValueError("Saved analysis artifact reference is invalid.")
    path = (root / description["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Saved analysis artifacts must remain inside the analysis directory.")
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise ValueError(f"Saved analysis artifact is missing: {path}") from exc
    if _sha256(payload) != description["sha256"]:
        raise ValueError(f"Saved analysis artifact hash mismatch: {path}")
    return payload


def _checked_offsets(offsets: np.ndarray, count: int, length: int, label: str) -> None:
    if (offsets.shape != (count + 1,) or offsets.dtype.kind not in "iu"
            or offsets[0] != 0 or offsets[-1] != length or np.any(np.diff(offsets) <= 0)):
        raise ValueError(f"Saved collection {label} offsets are invalid.")


def load_collection(manifest_path: Path, *, group: str, mode: str = "pose",
                    basis: str = "shared") -> CollectionAnalysis:
    """Load a validated snapshot. Missing/corrupt data never triggers a PCA fit."""
    manifest_path = Path(manifest_path).resolve()
    index = _read_index(manifest_path)
    matches = [entry for entry in index["entries"] if isinstance(entry, dict)
               and (entry.get("group"), entry.get("mode"), entry.get("basis"))
               == (group, mode, basis)]
    if len(matches) != 1:
        available = [f"{entry.get('group')}/{entry.get('mode')}/{entry.get('basis')}"
                     for entry in index["entries"] if isinstance(entry, dict)]
        raise ValueError(f"Expected one saved analysis for {group}/{mode}/{basis}; available: {available}")
    entry = matches[0]
    root = manifest_path.parent
    try:
        model = _model_from_bytes(_read_artifact(root, entry["model"]))
        if entry["model"].get("identity") != _model_key(model):
            raise ValueError("Saved PCA model identity mismatch.")
        with np.load(io.BytesIO(_read_artifact(root, entry["motions"])), allow_pickle=False) as archive:
            frames, frame_offsets = [archive[name].copy() for name in ("frames", "frame_offsets")]
        with np.load(io.BytesIO(_read_artifact(root, entry["data"])), allow_pickle=False) as archive:
            scores, starts = [archive[name].copy() for name in ("scores", "starts")]
            score_offsets, start_offsets = [archive[name].copy() for name in
                                            ("score_offsets", "start_offsets")]
        metadata = entry["clips"]
        source_ids = tuple(entry["source_ids"])
        if (not isinstance(metadata, list) or not metadata or len(metadata) != len(source_ids)
                or len(set(source_ids)) != len(source_ids)
                or any(not isinstance(item, str) or not item for item in source_ids)):
            raise ValueError("Saved collection source identities are invalid.")
        count = len(metadata)
        if (frames.ndim != 2 or scores.ndim != 2 or starts.ndim != 1
                or not np.isfinite(frames).all() or not np.isfinite(scores).all()
                or starts.dtype.kind not in "iu" or scores.shape[1] != model.max_components):
            raise ValueError("Saved collection arrays have invalid shapes or non-finite values.")
        _checked_offsets(frame_offsets, count, len(frames), "frame")
        _checked_offsets(score_offsets, count, len(scores), "score")
        _checked_offsets(start_offsets, count, len(starts), "window")
        if entry.get("feature") != "native_dof":
            raise ValueError("Saved collection uses an unsupported feature representation.")
        if mode not in ("pose", "window"):
            raise ValueError("Saved PCA mode must be pose or window.")
        clips, layouts, projections, matrices = [], [], [], []
        for clip_index, info in enumerate(metadata):
            clip_frames = np.array(frames[frame_offsets[clip_index]:frame_offsets[clip_index + 1]],
                                   dtype=np.float64, copy=True)
            if (not isinstance(info, dict) or info["source_id"] != source_ids[clip_index]
                    or list(clip_frames.shape) != info["shape"]
                    or _array_hash(clip_frames) != info["frames_sha256"]):
                raise ValueError("Saved source frame metadata does not match its snapshot.")
            clip = MotionData(Path(info["path"]), clip_frames, float(info["fps"]),
                              info["loop_mode"], info["content_hash"])
            _validate_clip(clip)
            matrix, layout = build_samples(clip.frames[:, 6:], mode,
                                           entry["window"], entry["stride"],
                                           allow_single_window=True)
            clip_starts = starts[start_offsets[clip_index]:start_offsets[clip_index + 1]]
            clip_scores = scores[score_offsets[clip_index]:score_offsets[clip_index + 1]].copy()
            if (not np.array_equal(layout.starts, clip_starts)
                    or clip_scores.shape != (len(matrix), model.max_components)):
                raise ValueError("Saved window layout or projection shape is inconsistent.")
            if model.mean.shape != (matrix.shape[1],):
                raise ValueError("Saved motion features do not match the PCA model width.")
            if not np.allclose(clip_scores, model.transform(matrix), rtol=1e-8, atol=1e-10):
                raise ValueError("Saved projection scores do not match the saved model and motion.")
            clips.append(clip)
            layouts.append(layout)
            projections.append(clip_scores)
            matrices.append(matrix)
        fitted = entry.get("fitted_on_collection")
        if not isinstance(fitted, bool):
            raise ValueError("Saved analysis must identify whether its basis was fitted on this collection.")
        if fitted and model.sample_count != sum(len(matrix) for matrix in matrices):
            raise ValueError("Fitted collection sample count differs from its saved PCA model.")
        return CollectionAnalysis(
            tuple(clips), tuple(layouts), tuple(projections), model,
            np.concatenate(matrices), fitted, source_ids,
            root / "reconstructions" / group / mode / basis,
            True, group, basis)
    except (KeyError, TypeError, IndexError, OverflowError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Invalid saved PCA collection {group}/{mode}/{basis}: {exc}") from exc
