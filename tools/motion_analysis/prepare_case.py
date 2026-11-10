"""Prepare the official spinkick sampling and PCA case."""

from pathlib import Path
import sys


# Support the existing direct-script command from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.motion_analysis.workflows.spinkick import main, prepare_case


if __name__ == "__main__":
    raise SystemExit(main())
