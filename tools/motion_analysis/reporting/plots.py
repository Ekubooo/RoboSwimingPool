"""Export scientific figures for a frozen Humanoid spinkick analysis case."""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .metrics import write_decode_diagnostics, write_metrics
from ..core.collection import load_collection
from .density import gaussian_density, kernel_density, reference_bandwidth
from ..data.storage import atomic_write
from ..paths import DEFAULT_CASE


COLORS = {"reference": "#2469ad", "generated": "#df741e"}
STEPS = (49, 39, 29, 19, 9, 0)


class CaseFigures:
    """Keep figure scales, source collections and output paths in one place."""

    def __init__(self, case: Path):
        self.case = case.resolve()
        self.data = self.case / "assets/data"
        self.output = self.case / "assets/plots"
        self.manifest = self.data / "analysis.json"
        self.collections = {}
        self.files = []
        plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                             "axes.spines.right": False, "figure.dpi": 110,
                             "savefig.dpi": 150, "axes.grid": True, "grid.alpha": .18})

    def get(self, group: str, mode: str = "pose", basis: str = "shared"):
        key = (group, mode, basis)
        if key not in self.collections:
            self.collections[key] = load_collection(self.manifest, group=group, mode=mode, basis=basis)
        return self.collections[key]

    def scores(self, group: str, mode: str = "pose") -> np.ndarray:
        return np.concatenate(self.get(group, mode).scores)

    def save(self, figure, name: str) -> None:
        self.output.mkdir(parents=True, exist_ok=True)
        path = self.output / f"{name}.png"
        figure.savefig(path, bbox_inches="tight", facecolor="white")
        plt.close(figure)
        self.files.append(path.name)
        print(f"Figure: {path.name}", flush=True)

    def quality(self, metrics: dict) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for row, mode in enumerate(("pose", "window")):
            for group, color in COLORS.items():
                model = self.get(group, mode, "independent").model
                x = np.arange(1, model.max_components + 1)
                axes[row, 0].plot(x, model.variance_ratio, color=color, label=f"{group} (rank {len(x)})")
                axes[row, 1].plot(x, np.cumsum(model.variance_ratio), color=color, label=group)
            for col in range(2):
                axes[row, col].set(xlabel="Number of components K", ylabel="Variance fraction",
                                   title=f"{mode}: {'individual' if col == 0 else 'cumulative'} variance",
                                   ylim=(-.02, 1.03))
                axes[row, col].legend()
                if mode == "window":
                    axes[row, col].set_xscale("log")
        fig.suptitle("G01 | Independent PCA fits (no shared coordinate comparison)")
        self.save(fig, "G01_variance")

        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for row, mode in enumerate(("pose", "window")):
            for group, color in COLORS.items():
                rows = metrics["independent"][f"{group}/{mode}"]["scan"]
                for col, (key, unit) in enumerate((("dof_rmse_rad", "DOF RMSE (rad)"),
                                                   ("fk_rmse_cm", "FK body-origin RMSE (cm)"))):
                    axes[row, col].plot([r["k"] for r in rows], [r[key] for r in rows],
                                        "o-", color=color, label=group)
                    axes[row, col].set(xlabel="Retained K (symlog spacing)", ylabel=unit,
                                       title=f"{mode}: own-motion reconstruction")
                    axes[row, col].set_xscale("symlog", linthresh=4)
                    axes[row, col].legend()
        fig.suptitle("G02 | K scan; generated motions are compared with themselves")
        self.save(fig, "G02_k_error")

        labels = metrics["dof_labels"]
        reference = self.get("reference").clips[0].frames[:, 6:]
        # One leg hinge and the largest-varying spherical coordinate are readable representatives.
        knee = next(i for i, name in enumerate(labels) if "right_knee" in name)
        other = int(np.argmax(reference.var(axis=0)))
        chosen = [knee, other] if knee != other else [knee, int(np.argsort(reference.var(axis=0))[-2])]
        metrics["representative_dofs"] = chosen
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), layout="constrained")
        for row, group in enumerate(COLORS):
            a = self.get(group, basis="independent")
            session = a.for_clip(0)
            t = np.arange(len(session.source.frames)) / 30
            for col, joint in enumerate(chosen):
                ax = axes[row, col]
                ax.plot(t, session.source.frames[:, 6 + joint], "k", lw=2, label="full")
                for k, style in ((1, "--"), (2, "-."), (3, ":")):
                    ax.plot(t, session.reconstruct(tuple(range(k)))[:, 6 + joint], style, label=f"K={k}")
                ax.set(xlabel="Clip time (s)", ylabel="Native DOF coordinate (rad)",
                       title=f"{group} clip 0: {labels[joint]}")
                ax.legend(ncol=2, fontsize=8)
        fig.suptitle("G03 | Independent pose PCA: mean + cumulative contributions")
        self.save(fig, "G03_joint_curves")

        arrays = []
        for mode in ("pose", "window"):
            for group in COLORS:
                session = self.get(group, mode, "independent").for_clip(0)
                arrays.append(np.abs(session.reconstruct((0, 1, 2))[:, 6:] - session.source.frames[:, 6:]))
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
        for ax, error, (mode, group) in zip(axes.flat, arrays,
                                         ((m, g) for m in ("pose", "window") for g in COLORS)):
            image = ax.imshow(error.T, origin="lower", aspect="auto", vmin=0,
                              vmax=max(x.max() for x in arrays), cmap="magma",
                              extent=(-.5 / 30, (len(error) - .5) / 30, -.5, 27.5))
            ax.set(xlabel="Clip time (s)", ylabel="DOF index (0-based)", title=f"{group}, {mode}, clip 0, K=3")
            ax.grid(False)
        fig.colorbar(image, ax=axes, label="Absolute DOF coordinate error (rad)", shrink=.8)
        fig.suptitle("G04 | Own-motion PCA reconstruction errors; common color scale")
        self.save(fig, "G04_error_heatmaps")

    def structure(self, metrics: dict) -> None:
        labels = metrics["dof_labels"]
        fig, axes = plt.subplots(2, 2, figsize=(13, 7), layout="constrained")
        vmax = max(np.abs(self.get(g, m, "independent").model.components[:3]).max()
                   for g in COLORS for m in ("pose", "window"))
        for row, mode in enumerate(("pose", "window")):
            for col, group in enumerate(COLORS):
                a = self.get(group, mode, "independent")
                ax = axes[row, col]
                if mode == "pose":
                    values = a.model.components[:3]
                    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                    ax.set(yticks=[0, 1, 2], yticklabels=["PC1", "PC2", "PC3"],
                           xticks=np.arange(28), xticklabels=labels)
                    ax.tick_params(axis="x", rotation=90, labelsize=6)
                    ax.set_xlabel("Native DOF coordinate")
                else:
                    # Retain every coefficient; do not average away temporal signs.
                    values = a.model.components[:3].reshape(3, 10, 28).transpose(1, 0, 2).reshape(10, 84)
                    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
                    ax.axvline(27.5, color="black", lw=.8)
                    ax.axvline(55.5, color="black", lw=.8)
                    ax.set(xlabel="DOF index within each component (0..27)", ylabel="Window frame (0..9)",
                           xticks=[13.5, 41.5, 69.5], xticklabels=["PC1", "PC2", "PC3"])
                ax.set_title(f"{group}: {mode} loadings")
                ax.grid(False)
        fig.colorbar(image, ax=axes, label="Unit-norm component loading (dimensionless)", shrink=.7)
        fig.suptitle("G05 | Independent bases; signs fixed numerically, axes are not shared")
        self.save(fig, "G05_loadings")

        fig = plt.figure(figsize=(12, 5.5), layout="constrained")
        a = self.get("reference")
        sample_scores = np.concatenate([a.scores[0], *self.get("generated").scores[:4]])[:, :3]
        limits = [(sample_scores[:, i].min() - .1, sample_scores[:, i].max() + .1) for i in range(3)]
        for index, group in enumerate(COLORS):
            ax = fig.add_subplot(1, 2, index + 1, projection="3d")
            scores = self.get(group).scores[:1 if group == "reference" else 4]
            for clip, values in enumerate(scores):
                ax.plot(*values[:, :3].T, ".-", ms=3, label=f"clip {clip}")
                ax.scatter(*values[0, :3], marker="s", s=35, color="black")
            ax.set(xlabel="PC1 score (rad)", ylabel="PC2 score (rad)", zlabel="PC3 score (rad)",
                   xlim=limits[0], ylim=limits[1], zlim=limits[2], title=f"{group}: chronological pose trajectories")
            ax.view_init(24, -55)
            ax.legend(fontsize=8)
        fig.suptitle("G06 | Shared reference pose basis; squares mark clip starts")
        self.save(fig, "G06_pca_trajectories")

        fig = plt.figure(figsize=(12, 5.5), layout="constrained")
        first, second = metrics["representative_dofs"]
        clips = [self.get(g).clips[0] for g in COLORS]
        y = np.concatenate([clip.frames[:, 6 + first] for clip in clips])
        z = np.concatenate([clip.frames[:, 6 + second] for clip in clips])
        for index, (group, clip) in enumerate(zip(COLORS, clips)):
            ax = fig.add_subplot(1, 2, index + 1, projection="3d")
            ax.plot(np.arange(len(clip.frames)) / 30, clip.frames[:, 6 + first],
                    clip.frames[:, 6 + second], ".-", color=COLORS[group])
            ax.set(xlabel="Clip time (s)", ylabel=f"DOF {first} (rad)", zlabel=f"DOF {second} (rad)",
                   ylim=(y.min() - .1, y.max() + .1), zlim=(z.min() - .1, z.max() + .1),
                   title=f"{group}, clip 0 (not time-aligned across groups)")
            ax.view_init(24, -55)
        fig.suptitle(f"G07 | Time curves, not probabilities: {labels[first]} and {labels[second]}")
        self.save(fig, "G07_time_joint_curve")

    def distributions(self) -> dict:
        metadata = {}
        for mode in ("pose", "window"):
            points = {group: self.scores(group, mode)[:, :2] for group in COLORS}
            both = np.concatenate(list(points.values()))
            low, high = both.min(axis=0), both.max(axis=0)
            margin = (high - low) * .12
            x = np.linspace(low[0] - margin[0], high[0] + margin[0], 80)
            y = np.linspace(low[1] - margin[1], high[1] + margin[1], 80)
            xx, yy = np.meshgrid(x, y)
            grid = np.stack([xx, yy], axis=-1)
            bandwidth = reference_bandwidth(points["reference"])
            metadata[mode] = {"bandwidth_covariance": bandwidth.tolist(), "grid_x": [float(x[0]), float(x[-1]), len(x)],
                              "grid_y": [float(y[0]), float(y[-1]), len(y)], "kde": "equal-weight fixed Gaussian kernels"}
            fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")
            for group, color in COLORS.items():
                p = points[group]
                axes[0].scatter(*p.T, s=8 if group == "reference" else 3,
                                alpha=.9 if group == "reference" else .2, color=color, label=f"{group} N={len(p)}",
                                zorder=3 if group == "reference" else 2)
                for dimension, ax in enumerate(axes[1:]):
                    bins = np.linspace(both[:, dimension].min() - margin[dimension],
                                       both[:, dimension].max() + margin[dimension], 25)
                    ax.hist(p[:, dimension], bins=bins, density=True, histtype="step", lw=2, color=color, label=group)
                    ax.set(xlabel=f"PC{dimension + 1} score (rad)", ylabel="Probability density (1/rad)")
                    ax.legend()
            axes[0].set(xlabel="PC1 score (rad)", ylabel="PC2 score (rad)", xlim=(x[0], x[-1]), ylim=(y[0], y[-1]))
            axes[0].legend(fontsize=8)
            fig.suptitle(f"G08 | {mode}: shared reference basis, normalized histograms")
            self.save(fig, f"G08_{mode}_scatter_histograms")

            saved = {"grid_x": xx, "grid_y": yy, "bandwidth": bandwidth}
            for method in ("Gaussian", "KDE"):
                surfaces = {g: (gaussian_density(p, grid) if method == "Gaussian" else kernel_density(p, grid, bandwidth))
                            for g, p in points.items()}
                fig = plt.figure(figsize=(12, 9), layout="constrained")
                maximum = max(z.max() for z in surfaces.values())
                levels = np.linspace(0, maximum, 15)
                for col, (group, surface) in enumerate(surfaces.items()):
                    ax = fig.add_subplot(2, 2, col + 1, projection="3d")
                    ax.plot_surface(xx, yy, surface, cmap="viridis", vmin=0, vmax=maximum,
                                    rcount=60, ccount=60, linewidth=0)
                    ax.set(xlabel="PC1 score (rad)", ylabel="PC2 score (rad)", zlabel="Density (1/rad²)",
                           xlim=(x[0], x[-1]), ylim=(y[0], y[-1]), zlim=(0, maximum * 1.05), title=group)
                    ax.view_init(27, -55)
                    ax = fig.add_subplot(2, 2, col + 3)
                    image = ax.contourf(xx, yy, surface, levels=levels, cmap="viridis")
                    ax.contour(xx, yy, surface, levels=levels[2::3], colors="white", linewidths=.5)
                    ax.set(xlabel="PC1 score (rad)", ylabel="PC2 score (rad)", title=f"{group}: same scale")
                    saved[f"{method.lower()}_{group}"] = surface
                fig.colorbar(image, ax=fig.axes, label="Probability density (1/rad²)", shrink=.5)
                suffix = "single fitted Gaussian (different fitted covariance)" if method == "Gaussian" else "one reference-derived bandwidth for both groups"
                fig.suptitle(f"G09 | {mode} {method}: {suffix}")
                self.save(fig, f"G09_{mode}_{method.lower()}_density")
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **saved)
            atomic_write(self.data / f"density_{mode}.npz", buffer.getvalue())
        return metadata

    def stages(self) -> None:
        for mode in ("pose", "window"):
            ref = self.scores("reference", mode)[:, :2]
            samples = [self.scores(f"stage_{t}", mode)[:, :2] for t in STEPS]
            all_points = np.concatenate([ref, *samples])
            low, high = all_points.min(axis=0), all_points.max(axis=0)
            low -= (high - low) * .05
            high += (high - low) * .05
            xx, yy = np.meshgrid(np.linspace(low[0], high[0], 85), np.linspace(low[1], high[1], 85))
            grid = np.stack([xx, yy], axis=-1)
            bandwidth = reference_bandwidth(ref)
            surfaces = [kernel_density(p, grid, bandwidth) for p in samples]
            maximum = max(z.max() for z in surfaces)
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
            for ax, step, points, z in zip(axes.flat, STEPS, samples, surfaces):
                image = ax.contourf(xx, yy, z, levels=np.linspace(0, maximum, 16), cmap="viridis")
                ax.scatter(*ref.T, s=9, facecolors="none", edgecolors="white", linewidths=.6, label="reference")
                first = self.get(f"stage_{step}", mode).scores[0][:, :2]
                ax.plot(*first.T, "r.-", lw=1, ms=4, label="sample 0")
                ax.set(title=f"t={step}: predicted clean x0", xlabel="PC1 score (rad)", ylabel="PC2 score (rad)",
                       xlim=(low[0], high[0]), ylim=(low[1], high[1]))
                ax.legend(fontsize=7, loc="upper right")
            fig.colorbar(image, ax=axes, label="Probability density (1/rad²)", shrink=.8)
            fig.suptitle(f"G10 | {mode}: same 512 sample IDs, reference basis, kernel and density scale")
            self.save(fig, f"G10_{mode}_stages")
            fig, axes = plt.subplots(2, 3, figsize=(13, 8), sharex=True, sharey=True, layout="constrained")
            for ax, step, points in zip(axes.flat, STEPS, samples):
                ax.scatter(*points.T, s=3, alpha=.18, color=COLORS["generated"])
                ax.scatter(*ref.T, s=10, color=COLORS["reference"])
                ax.set(title=f"t={step}, N={len(points)}", xlabel="PC1 score (rad)", ylabel="PC2 score (rad)",
                       xlim=(low[0], high[0]), ylim=(low[1], high[1]))
            fig.suptitle(f"G11 | {mode}: clean-estimate projection scatter; all points retained")
            self.save(fig, f"G11_{mode}_stage_scatter")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, default=DEFAULT_CASE)
    args = parser.parse_args()
    figures = CaseFigures(args.case)
    metrics = write_metrics(figures.manifest, figures.data / "metrics.json")
    write_decode_diagnostics(figures.data)
    figures.quality(metrics)
    figures.structure(metrics)
    density = figures.distributions()
    figures.stages()
    metadata = {"figures": figures.files, "density": density, "representative_dofs": metrics["representative_dofs"]}
    atomic_write(figures.data / "plots.json", (json.dumps(metadata, indent=2) + "\n").encode())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
