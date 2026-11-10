"""Atomic persistence for motion, PCA and SMP artifacts."""

import os
from pathlib import Path
import tempfile


def atomic_write(path: Path, content: bytes) -> None:
    """Replace one completed artifact; a manifest is written only after its data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".pca-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)

