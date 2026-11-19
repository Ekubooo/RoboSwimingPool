"""Optionally prepare current runtime assets; playback also prepares on demand."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset", help="Prepare only CATEGORY/NAME")
    args = parser.parse_args()
    from RSP.app.config import boundary_support_radius, get_fluid_settings
    from RSP.sim_ready.pipelines.runtime import ensure_runtime_assets
    from RSP.sim_ready.catalog import AssetRef, scan_sources
    if args.asset:
        parts = args.asset.split("/")
        if len(parts) != 2:
            parser.error("--asset must be CATEGORY/NAME (no version)")
        refs = [AssetRef(*parts)]
    else:
        refs = list(scan_sources())
    assets = ensure_runtime_assets(refs, support_radius=boundary_support_radius(get_fluid_settings()),
                                   device=args.device, progress=print)
    for asset in assets.values():
        print(f"Prepared: {asset.root}")


if __name__ == "__main__":
    main()
