"""Analyze one motion, reuse cached PCA results, and optionally compare playback."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.motion_analysis.paths import CACHE_ROOT, MOTION_ROOT, ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--motion", required=True,
        help="Input .pkl path relative to data/motions, including its character folder.",
    )
    parser.add_argument("--mode", choices=("pose", "window"), default="pose")
    parser.add_argument("--window", type=int, default=10, help="Frames per window (window mode).")
    parser.add_argument("--stride", type=int, default=1, help="Window start spacing (window mode).")
    parser.add_argument(
        "--selection", choices=("cumulative", "single", "subset"), default="cumulative",
    )
    parser.add_argument(
        "--components", type=int, default=None,
        help="Cumulative count or single component number (1-based); default: min(4, rank).",
    )
    parser.add_argument("--pcs", help='Subset of 1-based components, e.g. "1,3,5-8"; "" means mean-only.')
    parser.add_argument("--version", default="v1", help="Human-readable analysis version label.")
    parser.add_argument("--force", action="store_true", help="Recompute the current analysis and selection.")
    parser.add_argument("--play", action="store_true", help="Open the interactive Original/Reconstructed viewer.")
    parser.add_argument("--view", choices=("comparison", "local", "classroom"), default="comparison",
                        help="Playback layout: original motion, fixed-root pair, or full / K=1,2,3.")
    parser.add_argument(
        "--env-config", type=Path,
        default=ROOT / "data/envs/view_motion_humanoid_env.yaml",
        help="Existing view_motion environment template; also supplies the skeleton for FK metrics.",
    )
    parser.add_argument(
        "--engine-config", type=Path,
        default=ROOT / "data/engines/newton_engine.yaml",
        help="Newton engine configuration (only used with --play).",
    )
    parser.add_argument("--device", default="cuda:0", help="Playback device; offline PCA runs on CPU.")
    return parser


def resolve_motion_path(name: str) -> Path:
    """Keep CLI input within the original motion tree; results use a separate tree."""
    relative = Path(name)
    if relative.is_absolute():
        raise ValueError("--motion must be relative to data/motions.")
    path = (MOTION_ROOT / relative).resolve()
    if not path.is_relative_to(MOTION_ROOT.resolve()):
        raise ValueError("--motion must stay inside data/motions.")
    if path.suffix.lower() != ".pkl" or not path.is_file():
        raise ValueError(f"Motion .pkl does not exist: {path}")
    return path


def resolve_repo_path(path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def skeleton_from_env(env_path: Path) -> Path:
    import yaml

    with env_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or config.get("env_name") != "view_motion":
        raise ValueError("--env-config must be a view_motion environment template.")
    if not config.get("char_file"):
        raise ValueError("Environment template must specify char_file.")
    char_file = resolve_repo_path(Path(config["char_file"]))
    if not char_file.is_file():
        raise ValueError(f"Character asset does not exist: {char_file}")
    return char_file


def run(args: argparse.Namespace) -> None:
    # Keep Newton and the environment framework off the offline import path.
    from tools.motion_analysis.reporting.reconstruction import generate_report
    from tools.motion_analysis.core.selection import parse_components, resolve_selection
    from tools.motion_analysis.core.session import load_session

    if args.selection == "subset":
        if args.pcs is None:
            raise ValueError('Subset mode requires --pcs, e.g. --pcs "1,3,5-8".')
        if args.components is not None:
            raise ValueError("Use --pcs instead of --components for a subset.")
    elif args.pcs is not None:
        raise ValueError("--pcs is only used with --selection subset.")

    source_path = resolve_motion_path(args.motion)
    env_path = resolve_repo_path(args.env_config)
    char_file = skeleton_from_env(env_path)
    session = load_session(
        source_path, mode=args.mode, window=args.window, stride=args.stride,
        version=args.version, cache_root=CACHE_ROOT, force=args.force,
    )
    k = args.components
    if k is None:
        k = min(4, session.max_components)
    if session.max_components == 0 and args.components is None and args.selection != "subset":
        indices: tuple[int, ...] = ()
    else:
        indices = resolve_selection(
            args.selection, k, parse_components(args.pcs) if args.pcs is not None else None,
            session.max_components,
        )

    reconstruction_path = session.save_reconstruction(indices)
    report = generate_report(session, indices, reconstruction_path, char_file=char_file)
    component_text = ",".join(str(index + 1) for index in indices) or "mean only"
    print(f"Input: {source_path}")
    print(f"Analysis: {args.mode}; effective components: {session.max_components}")
    print(f"Fit cache: {'hit' if session.fit_cache_hit else 'computed'}")
    print(f"Selected components: {component_text}")
    if "retained_variance_ratio" in report:
        print(f"Selected variance: {report['retained_variance_ratio']:.2%}")
    if "reconstruction_rmse" in report:
        print(f"DOF RMSE: {report['reconstruction_rmse']:.6g} rad")
    if "fk_mean_cm" in report:
        print(f"FK body error: mean {report['fk_mean_cm']:.3f} cm; max {report['fk_max_cm']:.3f} cm")
    print(f"Reconstruction: {reconstruction_path}")
    if report.get("report_path"):
        print(f"Report: {report['report_path']}")

    if args.play:
        try:
            from tools.motion_analysis.player.environment import play_session

            # Asset loaders in the existing simulator resolve paths from the repo.
            previous_cwd = Path.cwd()
            try:
                os.chdir(ROOT)
                play_session(
                    session, indices, selection_mode=args.selection, view=args.view, env_config=env_path,
                    engine_config=resolve_repo_path(args.engine_config), device=args.device,
                )
            finally:
                os.chdir(previous_cwd)
        except (ImportError, RuntimeError, OSError) as exc:
            raise RuntimeError(
                f"Playback failed: {exc}\nOffline PCA results remain at {reconstruction_path}"
            ) from exc


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Interrupted. Completed cache files are preserved.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"PCA error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
