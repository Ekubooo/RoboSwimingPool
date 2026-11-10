"""Export Newton GIFs for a saved motion analysis case."""

from pathlib import Path
import sys


# Support the existing direct-script command from any working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.motion_analysis.reporting.gifs import main, record_case


if __name__ == "__main__":
    raise SystemExit(main())
