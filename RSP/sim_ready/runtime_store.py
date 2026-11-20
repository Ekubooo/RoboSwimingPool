"""Serialize asset preparation and safely replace one generated asset directory."""

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
import os
import shutil
import stat
import tempfile
from threading import local

from .catalog import read_prepared_asset
from .paths import RUNTIME_ROOT, runtime_root


class AssetPreparationError(RuntimeError):
    """The requested runtime asset could not be prepared."""


_leases = local()


@contextmanager
def runtime_asset_lease(root=RUNTIME_ROOT, *, reuse=False):
    """Hold the generated root through preparation and use."""
    try:
        root = runtime_root(root)
    except (OSError, ValueError) as exc:
        raise AssetPreparationError(f"Cannot use runtime asset directory: {exc}") from exc
    held = getattr(_leases, "roots", None)
    if held is None:
        held = _leases.roots = set()
    if root in held:
        if not reuse:
            raise AssetPreparationError(f"Runtime asset directory is in use: {root}")
        yield root
        return

    try:
        root.mkdir(parents=True, exist_ok=True)
        lock_path = _plain_path(root / ".runtime.lock")
        lock_file = lock_path.open("a+b")
    except (OSError, ValueError) as exc:
        raise AssetPreparationError(f"Cannot open runtime asset directory {root}: {exc}") from exc
    with lock_file as lock:
        if lock.tell() == 0:
            lock.write(b"\0")
            lock.flush()
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            def acquire():
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)

            def release():
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            def acquire():
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            def release():
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        try:
            acquire()
        except OSError as exc:
            raise AssetPreparationError(f"Runtime asset directory is in use: {root}") from exc
        held.add(root)
        try:
            yield root
        finally:
            held.remove(root)
            release()


def _plain_path(path):
    """Refuse links and junctions before examining or removing a generated slot."""
    try:
        info = path.lstat()
    except FileNotFoundError:
        info = None
    if (info is not None and (stat.S_ISLNK(info.st_mode)
            or getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0))):
        raise ValueError(f"Runtime output uses a link or junction: {path}")
    if path.resolve() != path:
        raise ValueError(f"Runtime output is not canonical: {path}")
    return path


def _paths(root, source):
    root = runtime_root(root, origin=source.origin)
    parent = _plain_path(root / source.ref.category)
    current = _plain_path(parent / source.ref.name)
    backup = _plain_path(parent / f".{source.ref.name}.backup")
    return current, backup


def _read_candidate(directory, source):
    _check_tree(directory)
    try:
        return read_prepared_asset(directory / "asset.json", source)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def _check_tree(directory):
    for parent, dirs, files in os.walk(_plain_path(directory), followlinks=False):
        for name in (*dirs, *files):
            _plain_path(Path(parent) / name)


def _discard(path):
    _check_tree(path)
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def recover_runtime_asset(root, source):
    """Recover a previous complete directory if promotion was interrupted."""
    current, backup = _paths(root, source)
    if current.parent.is_dir():
        for stale in current.parent.iterdir():
            if stale.name.startswith(f".{current.name}.stage-"):
                _discard(stale)
    asset = _read_candidate(current, source)
    if backup.exists():
        if asset is None:
            asset = _read_candidate(backup, source)
            if asset is not None:
                if current.exists():
                    _discard(current)
                backup.rename(current)
                return replace(asset, root=current)
        _discard(backup)
    return asset


@contextmanager
def stage_runtime_asset(root, source):
    try:
        current, _ = _paths(root, source)
        current.parent.mkdir(parents=True, exist_ok=True)
        staging = tempfile.TemporaryDirectory(dir=current.parent, prefix=f".{current.name}.stage-")
    except (OSError, ValueError) as exc:
        raise AssetPreparationError(f"Cannot stage {source.ref.relative}: {exc}") from exc
    with staging as temporary:
        yield Path(temporary)


def commit_runtime_asset(root, asset):
    """Promote a verified state while the caller holds the root lease."""
    current, backup = _paths(root, asset.source)
    stage = _plain_path(asset.root)
    if stage.parent != current.parent or not stage.name.startswith(f".{current.name}.stage-"):
        raise ValueError(f"Asset staging path is not a sibling of its slot: {stage}")
    if backup.exists():
        raise ValueError(f"Recover the runtime asset before replacing it: {current}")
    if current.exists():
        _check_tree(current)
        current.rename(backup)
    try:
        stage.rename(current)
    except BaseException:
        if backup.exists():
            backup.rename(current)
        raise
    if backup.exists():
        _discard(backup)
    return replace(asset, root=current)
