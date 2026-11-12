"""OBJ reads and atomic writes, without geometry or cache policy."""

import os
from pathlib import Path

import trimesh

from .publish import _temporary


def load_mesh_obj(path):
    return trimesh.load(path, force="mesh", process=False)


def save_mesh_obj(mesh, path, *, digits=8):
    """Atomically save geometry; containers retain their original precision."""
    path = Path(path)
    temporary = _temporary(path)
    try:
        temporary.write_text(mesh.export(file_type="obj", digits=digits), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
