"""Stable content identities shared by asset pipelines."""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np


def recipe_digest(value, *, allow_nan=False):
    """Retain the existing sorted-JSON encoding used by cache recipes."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=allow_nan).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def physical_signature(asset):
    """Compare local geometry/topology, independent of scene label prefixes."""
    def plain(value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, dict):
            return {k: (v.split("/")[-1] if k == "name" else plain(v)) for k, v in value.items()}
        if isinstance(value, (tuple, list)):
            return [plain(v) for v in value]
        return value

    record = plain(asdict(asset))
    record.pop("name")
    return recipe_digest(record)


def sdf_digest(volume):
    """Identify SDF contents independently of the mutable second NVDB grid.

    Saving a new grid pair changes the checksum, flags, index and grid count.
    Exclude these header fields; retain the SDF values, topology and transform.
    """
    import warp as wp
    size = volume.get_grid_info().size_in_bytes
    raw = wp.array(ptr=volume.array().ptr, dtype=wp.uint8, shape=size, device=volume.device).numpy()
    digest = hashlib.sha256()
    # Hash slices rather than copying a full grid or modifying a CPU-backed view.
    for chunk in (raw[:8], bytes(8), raw[16:20], bytes(12), raw[32:]):
        digest.update(chunk)
    return digest.hexdigest()


def mesh_digest(mesh):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int32).tobytes())
    return digest.hexdigest()
