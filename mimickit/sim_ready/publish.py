"""Stage complete asset groups and restore previous files on publication failure."""

import os
from pathlib import Path
import shutil
import tempfile


def _temporary(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, filename = tempfile.mkstemp(prefix=path.stem + ".", suffix=".asset.tmp", dir=path.parent)
    os.close(descriptor)
    return Path(filename)


def save_mesh_obj(mesh, path, *, digits=8):
    """Atomically save geometry; containers retain their original precision."""
    temporary = _temporary(path)
    try:
        temporary.write_text(mesh.export(file_type="obj", digits=digits), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def publish_files(destinations):
    """Publish explicit (source, target) pairs as one rollback-protected group.

    Supports OBJ/NVDB plus physical/collision attachments across subdirectories.
    Like publish_pair, consumers must be stopped during publication.
    """
    prepared, replaced, recovery_files, rollback_errors = [], [], set(), []
    try:
        for source, destination in destinations:
            temporary = _temporary(destination)
            entry = [destination, temporary, None]
            prepared.append(entry)
            shutil.copyfile(source, temporary)
            if destination.exists():
                entry[2] = _temporary(destination)
                shutil.copyfile(destination, entry[2])
        for destination, temporary, backup in prepared:
            os.replace(temporary, destination)
            replaced.append((destination, backup))
    except BaseException as error:
        for destination, backup in reversed(replaced):
            try:
                if backup is None:
                    destination.unlink(missing_ok=True)
                else:
                    os.replace(backup, destination)
            except OSError as rollback_error:
                rollback_errors.append(f"{destination}: {rollback_error}")
                if backup is not None:
                    recovery_files.add(backup)
        if rollback_errors:
            raise RuntimeError(f"Publication rollback could not restore all files: {rollback_errors}; "
                               f"recovery copies: {sorted(str(p) for p in recovery_files)}") from error
        raise
    finally:
        for _, temporary, backup in prepared:
            temporary.unlink(missing_ok=True)
            if backup is not None and backup not in recovery_files:
                backup.unlink(missing_ok=True)


