"""Read/write the existing convex-parts NPZ format, without cache decisions."""

import numpy as np
import trimesh


def load_hulls(path):
    with np.load(path, allow_pickle=False) as saved:
        count = int(saved["count"])
        hulls = [trimesh.Trimesh(saved[f"vertices_{i}"], saved[f"faces_{i}"], process=False)
                 for i in range(count)]
    return tuple(hulls)


def save_hulls(hulls, path):
    """Serialize to a caller-owned staging path; publication is separate."""
    arrays = {"count": np.asarray(len(hulls), dtype=np.int32)}
    for i, hull in enumerate(hulls):
        arrays[f"vertices_{i}"] = np.asarray(hull.vertices, dtype=np.float64)
        arrays[f"faces_{i}"] = np.asarray(hull.faces, dtype=np.int32)
    np.savez_compressed(path, **arrays)
