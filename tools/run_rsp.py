"""Select and simulate a scene in the same Newton window."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "mimickit")]


def main():
    os.chdir(ROOT)  # Existing asset and motion/prior paths are project-relative.
    from RSP.launcher.application import run
    run()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
