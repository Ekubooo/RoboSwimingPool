"""Play a previously saved PCA collection; never fit or run a sampling model."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.motion_analysis.paths import ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Saved analysis.json index.")
    parser.add_argument("--group", default="reference", help="Saved group, e.g. generated or stage_49.")
    parser.add_argument("--clip", type=int, default=0, help="Zero-based clip index within the group.")
    parser.add_argument("--mode", choices=("pose", "window"), default="pose")
    parser.add_argument("--basis", choices=("shared", "independent"), default="shared")
    parser.add_argument("--selection", choices=("cumulative", "single", "subset"), default="cumulative")
    parser.add_argument("--components", type=int, help="Cumulative K or single 1-based component number.")
    parser.add_argument("--pcs", help='Subset of 1-based components, e.g. "1,3,5-8".')
    parser.add_argument("--view", choices=("comparison", "local", "classroom"), default="comparison")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--env-config", type=Path, help="Override the saved playback environment config.")
    parser.add_argument("--engine-config", type=Path, help="Override the saved Newton engine config.")
    return parser


def _config_path(override: Path | None, saved: str | None, default: str) -> Path:
    path = Path(override or saved or default)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def run(args: argparse.Namespace) -> None:
    from tools.motion_analysis.core.collection import load_collection
    from tools.motion_analysis.core.selection import parse_components, resolve_selection

    if args.selection == "subset":
        if args.pcs is None or args.components is not None:
            raise ValueError("Subset playback requires --pcs and does not accept --components.")
    elif args.pcs is not None:
        raise ValueError("--pcs is only used with --selection subset.")
    manifest_path = args.manifest.resolve()
    analysis = load_collection(manifest_path, group=args.group, mode=args.mode, basis=args.basis)
    session = analysis.for_clip(args.clip)
    k = min(4, session.max_components) if args.components is None else args.components
    indices = resolve_selection(
        args.selection, k, parse_components(args.pcs) if args.pcs is not None else (),
        session.max_components)
    with manifest_path.open(encoding="utf-8") as stream:
        playback = json.load(stream).get("playback", {})
    if not isinstance(playback, dict):
        raise ValueError("Saved playback configuration must be an object.")
    env_config = _config_path(args.env_config, playback.get("env_config"),
                              "data/envs/view_motion_humanoid_env.yaml")
    engine_config = _config_path(args.engine_config, playback.get("engine_config"),
                                 "data/engines/newton_engine.yaml")
    print(f"Saved analysis: {manifest_path}")
    print(f"Group: {args.group}; mode: {args.mode}; basis: {args.basis}; clip: {args.clip}")
    print(f"Source ID: {session.source_id}; available components: {session.max_components}")
    print("Playback uses saved PCA data. No fitting or diffusion sampling is performed.")
    from tools.motion_analysis.player.environment import play_session

    previous_cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        play_session(session, indices, selection_mode=args.selection, view=args.view,
                     env_config=env_config, engine_config=engine_config, device=args.device)
    finally:
        os.chdir(previous_cwd)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except KeyboardInterrupt:
        print("Playback interrupted. Saved analysis files are unchanged.", file=sys.stderr)
        return 130
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Saved PCA playback error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
