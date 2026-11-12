"""Content identities for published geometry, independent of mutable volume maps."""

import hashlib
from pathlib import Path


def generated_root(path):
    """Keep repository-local generated assets in the ignored output tree."""
    root = Path(path).expanduser().resolve()
    project = Path(__file__).resolve().parents[2]
    if root.is_relative_to(project) and not root.is_relative_to(project / "output"):
        raise ValueError("Generated assets inside this repository must be under output/")
    return root


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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


