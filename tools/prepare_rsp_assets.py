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
    parser.add_argument("--sdf-only", action="store_true", help="Preprocess discovered OBJ files without volume maps")
    args = parser.parse_args()
    from RSP.app.config import boundary_support_radius, get_fluid_settings
    from RSP.sim_ready.config import BakeOptions
    from RSP.sim_ready.pipelines.runtime import ensure_runtime_assets, prepare_sources
    from RSP.sim_ready.catalog import AssetRef, scan_sources
    support_radius = boundary_support_radius(get_fluid_settings())
    if args.sdf_only:
        if args.asset:
            parser.error("--sdf-only scans all OBJ sources; omit --asset")
        prepared = prepare_sources(support_radius=max(BakeOptions().max_support_radius, support_radius),
                                   device=args.device, progress=print)
        for error in prepared.errors:
            print(f"Asset unavailable: {error}")
        for asset in prepared.assets.values():
            print(f"Prepared SDF: {asset.root}")
        return
    if args.asset:
        parts = args.asset.split("/")
        if len(parts) != 2:
            parser.error("--asset must be CATEGORY/NAME (no version)")
        refs = [AssetRef(*parts)]
    else:
        refs = list(scan_sources())
    assets = ensure_runtime_assets(refs, support_radius=support_radius,
                                   device=args.device, progress=print)
    for asset in assets.values():
        print(f"Prepared: {asset.root}")


if __name__ == "__main__":
    main()
