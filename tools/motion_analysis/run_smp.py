"""Sample an existing SMP prior and cache its final/trace motion features."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.motion_analysis.smp.sampling import (
    DEFAULT_CONFIG, DEFAULT_MODEL, DEFAULT_OUTPUT, SamplingConfig, export_clip, sample_prior,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-ema", action="store_true", help="Use ordinary weights instead of EMA.")
    parser.add_argument("--trace-steps", type=int, nargs="*", default=[49, 39, 29, 19, 9, 0],
                        help="Record input x_t and predicted clean motion at these timesteps; no values disables traces.")
    parser.add_argument("--version", default="v1")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--export-index", type=int, help="Export this zero-based sample index as a native .pkl.")
    parser.add_argument("--stage", type=int, help="With --export-index, export this traced clean estimate.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.stage is not None and args.export_index is None:
        parser.error("--stage requires --export-index")
    try:
        config = SamplingConfig(model_path=args.model, config_path=args.config, output_dir=args.output,
                                num_samples=args.num_samples, batch_size=args.batch_size, seed=args.seed,
                                device=args.device, use_ema=not args.no_ema,
                                trace_steps=tuple(args.trace_steps), force=args.force, version=args.version)
        bundle = sample_prior(config)
        print(f"Cache: {'hit' if bundle.cache_hit else 'sampled'}")
        print(f"Samples: {bundle.frames.shape}; features: {bundle.features.shape}; FPS: {bundle.fps:g}")
        print(f"Manifest: {bundle.manifest_path}")
        if args.export_index is not None:
            print(f"Motion: {export_clip(bundle, args.export_index, args.stage)}")
    except KeyboardInterrupt:
        print("Interrupted. Completed caches are preserved.", file=sys.stderr)
        return 130
    except (ValueError, OSError, RuntimeError, ImportError, KeyError) as exc:
        print(f"SMP sampling error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
