"""Prepare and publish the reference scene assets before playback."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    from RSP.sim_ready.pipelines.publication import prepare_sources
    for asset in prepare_sources(device=args.device):
        print(f"Published: {asset.root}")


if __name__ == "__main__":
    main()
