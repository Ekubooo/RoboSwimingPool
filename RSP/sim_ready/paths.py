"""RSP source locations and output rules; MimicKit paths remain caller-owned."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_ROOT = PROJECT_ROOT / "data" / "assets_RSP"
ORIGIN_ROOT = ASSET_ROOT / "origin"
PBF_PLAY_ROOT = ASSET_ROOT / "pbf_play"
TEST_ROOT = ASSET_ROOT / "tests"
RUNTIME_ROOT = ASSET_ROOT / "runtime"
SCENE_ROOT = PROJECT_ROOT / "data" / "scenes"
G1_SOURCE_ROOT = PROJECT_ROOT / "data" / "assets" / "g1"


def output_root(path):
    """Allow explicit external roots and protect repository sources from writes.

    Resolving first also rejects '..' and symlink/junction routes into origin.
    Merely validating a root does not create directories.
    """
    root = Path(path).expanduser().resolve()
    if root.is_relative_to(PROJECT_ROOT):
        if (root == ASSET_ROOT or not root.is_relative_to(ASSET_ROOT)
                or root.is_relative_to(ORIGIN_ROOT)):
            raise ValueError("Repository asset outputs must be in a data/assets_RSP subdirectory outside origin")
    return root


def runtime_root(path, *, origin=ORIGIN_ROOT):
    root = output_root(path)
    for protected in (ORIGIN_ROOT, Path(origin), G1_SOURCE_ROOT, PBF_PLAY_ROOT, ASSET_ROOT / "published"):
        protected = protected.resolve()
        if root.is_relative_to(protected) or protected.is_relative_to(root):
            raise ValueError(f"Runtime output {root} overlaps a protected directory: {protected}")
    return root
