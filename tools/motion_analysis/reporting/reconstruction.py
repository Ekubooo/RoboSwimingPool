"""Offline PCA diagnostics and plots; plotting/FK dependencies are lazy."""

import hashlib
import json
from pathlib import Path
from typing import Sequence, TYPE_CHECKING

import numpy as np

from ..data.motion import discontinuity_diagnostics
from ..core.selection import normalize_indices
from ..core.session import AnalysisSession
from ..data.storage import atomic_write
from ..paths import ensure_animation_imports

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def _save_figure(figure: "Figure", path: Path) -> None:
    import io
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    stream = io.BytesIO()
    figure.savefig(stream, format="png", dpi=130, bbox_inches="tight")
    atomic_write(path, stream.getvalue())
    figure.clear()


def _variance_plot(session: AnalysisSession, path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(7, 4))
    axes = figure.subplots()
    count = session.max_components
    x = np.arange(1, count + 1)
    axes.bar(x, session.variance_ratio, label="Individual")
    axes.plot(x, np.cumsum(session.variance_ratio), color="tab:orange", label="Cumulative")
    axes.set(xlabel="Principal component (1-based)", ylabel="Explained variance ratio",
             ylim=(0, 1.04), title="PCA explained variance")
    if not count:
        axes.text(.5, .5, "Constant features: mean-only reconstruction",
                  transform=axes.transAxes, ha="center")
    axes.legend()
    _save_figure(figure, path)


def _scores_plot(session: AnalysisSession, indices: tuple[int, ...], path: Path) -> None:
    from matplotlib.figure import Figure

    figure = Figure(figsize=(8, 4))
    axes = figure.subplots()
    times = session.layout.starts / session.source.fps
    # A large arbitrary subset is still useful; restrict the plot rather than
    # creating an unreadable legend with hundreds of component labels.
    shown = indices[:8]
    for index in shown:
        axes.plot(times, session.scores[:, index], label=f"PC {index + 1}")
    title = "Selected component scores"
    if len(indices) > len(shown):
        title += f" (first {len(shown)} of {len(indices)})"
    axes.set(xlabel="Window start time (s)" if session.layout.mode == "window" else "Time (s)",
             ylabel="PCA score", title=title)
    if shown:
        axes.legend(ncol=2)
    else:
        axes.text(.5, .5, "Mean-only: no selected components", transform=axes.transAxes, ha="center")
    _save_figure(figure, path)


def _fk_metrics(source: np.ndarray, reconstructed: np.ndarray, char_file: Path) -> dict:
    """Reuse MimicKit's CPU kinematics, without importing environment/engine code."""
    import torch

    ensure_animation_imports()
    import util.torch_util as torch_util

    if char_file.suffix.lower() == ".xml":
        from anim.mjcf_char_model import MJCFCharModel
        model = MJCFCharModel("cpu")
    elif char_file.suffix.lower() == ".urdf":
        from anim.urdf_char_model import URDFCharModel
        model = URDFCharModel("cpu")
    elif char_file.suffix.lower() == ".usd":
        from anim.usd_char_model import USDCharModel
        model = USDCharModel("cpu")
    else:
        raise ValueError("FK report requires an MJCF .xml, .urdf or .usd character.")
    model.load(str(char_file))
    if model.get_dof_size() != source.shape[1] - 6:
        raise ValueError("Character DOF count does not match the motion.")

    def positions(frames: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(frames, dtype=torch.float32)
        joints = model.dof_to_rot(tensor[:, 6:])
        root_rotation = torch_util.exp_map_to_quat(tensor[:, 3:6])
        return model.forward_kinematics(tensor[:, :3], root_rotation, joints)[0]

    with torch.no_grad():
        distances = torch.linalg.vector_norm(positions(source) - positions(reconstructed), dim=-1) * 100
    return {"fk_mean_cm": float(distances.mean()), "fk_max_cm": float(distances.max()),
            "fk_rmse_cm": float(torch.sqrt(torch.mean(distances ** 2)))}


def generate_report(session: AnalysisSession, indices: Sequence[int],
                    reconstruction_path: Path, char_file: Path | None = None) -> dict:
    """Save or restore metrics/plots for one selection, without refitting PCA."""
    selected = normalize_indices(indices, session.max_components)
    reconstruction_path = Path(reconstruction_path).resolve()
    report_path = reconstruction_path.with_suffix(".report.json")
    variance_path = session.cache_dir / "explained_variance.png"
    scores_path = reconstruction_path.with_suffix(".scores.png")
    char_file = Path(char_file).resolve() if char_file is not None else None
    character_hash = hashlib.sha256(char_file.read_bytes()).hexdigest() if char_file else None
    identity = {"fit_key": session.fit_key, "indices": list(selected),
                "character_sha256": character_hash, "report_version": 1}
    if not session.force:
        try:
            previous = json.loads(report_path.read_text(encoding="utf-8"))
            if (isinstance(previous, dict) and previous.get("identity") == identity
                    and variance_path.is_file() and scores_path.is_file()):
                return previous
        except (OSError, ValueError, TypeError):
            pass
    reconstructed = session.reconstruct(selected)
    difference = reconstructed[:, 6:] - session.source.frames[:, 6:]
    report = {"identity": identity, "source_path": str(session.source.path),
              "reconstruction_path": str(reconstruction_path), "report_path": str(report_path),
              "mode": session.layout.mode, "feature": "native_dof", "fps": session.source.fps,
              "frame_count": len(reconstructed), "window": session.layout.window,
              "stride": session.layout.stride, "sample_count": session.model.sample_count,
              "max_components": session.max_components,
              "selected_components": [value + 1 for value in selected],
              "retained_variance_ratio": float(session.variance_ratio[list(selected)].sum()),
              "reconstruction_rmse": float(np.sqrt(np.mean(difference ** 2))),
              "max_absolute_dof_error": float(np.abs(difference).max()),
              "variance_ratio": session.variance_ratio.tolist(),
              "variance_plot": str(variance_path), "scores_plot": str(scores_path),
              "plotted_components": [value + 1 for value in selected[:8]],
              **discontinuity_diagnostics(session.source)}
    if char_file is not None:
        report["character_path"] = str(char_file)
        report.update(_fk_metrics(session.source.frames, reconstructed, char_file))
    if not variance_path.is_file() or session.force:
        _variance_plot(session, variance_path)
    _scores_plot(session, selected, scores_path)
    atomic_write(report_path, (json.dumps(report, indent=2, ensure_ascii=False,
                                         allow_nan=False) + "\n").encode("utf-8"))
    return report
