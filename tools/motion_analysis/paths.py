"""Stable project paths shared by CLI, offline tools and playback adapters."""

from pathlib import Path
import sys


TOOL_ROOT = Path(__file__).resolve().parent
ROOT = TOOL_ROOT.parents[1]
MOTION_ROOT = ROOT / "data/motions"
CACHE_ROOT = TOOL_ROOT / "PCA_Result"
DEFAULT_CASE = ROOT / "report/humanoid_spinkick"


def ensure_animation_imports() -> None:
    """Expose MimicKit's existing top-level anim/util imports when needed."""
    runtime = str(ROOT / "mimickit")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
