"""Prepare and publish the reference scene assets before playback."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--asset", help="Prepare only CATEGORY/NAME/VERSION")
    args = parser.parse_args()
    from RSP.app.config import boundary_support_radius, get_fluid_settings
    from RSP.sim_ready.pipelines.publication import prepare_sources
    from RSP.sim_ready.catalog import AssetRef
    only = AssetRef(*args.asset.split("/")) if args.asset else None
    for asset in prepare_sources(support_radius=boundary_support_radius(get_fluid_settings()),
                                 device=args.device, only=only):
        print(f"Published: {asset.root}")


if __name__ == "__main__":
    main()
