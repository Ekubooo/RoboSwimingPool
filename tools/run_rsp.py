"""Select and simulate a scene in the same Newton window."""
from pathlib import Path
import os
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    os.chdir(ROOT)  # Existing asset and motion/prior paths are project-relative.
    from RSP.launcher.application import run
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="humanoid", choices=("humanoid", "g1"))
    parser.add_argument("--direct", action="store_true", help="Skip selection and run the profile's default scene")
    args = parser.parse_args()
    run(profile_id=args.profile, direct=args.direct)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
