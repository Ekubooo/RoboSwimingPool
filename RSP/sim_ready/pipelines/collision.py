"""Prepare convex collision assets using the existing cache recipe and format."""

import json
import time

from ..lib.mesh.convex import COACD_OPTIONS, check_backend, decompose_convex, validate_hulls, measure_hull_error
from ..lib.storage.fingerprints import mesh_digest, recipe_digest
from ..lib.storage.hull_io import load_hulls, save_hulls
from ..lib.storage.store import AssetStore


CACHE_VERSION = 1


def _description(mesh):
    return dict(version=CACHE_VERSION, geometry=mesh_digest(mesh),
                backend=check_backend(), options=COACD_OPTIONS)


def hulls_fingerprint(mesh):
    return recipe_digest(_description(mesh), allow_nan=True)


def prepare_hulls(mesh, name, store: AssetStore, *, destination=None, previous=None):
    """Reuse matching validated parts or replace one set of hulls."""
    description = _description(mesh)
    fingerprint = recipe_digest(description, allow_nan=True)
    directory = store.path("collision", name) if destination is None else store.path(destination)
    path = directory / "hulls.npz"
    record = directory / "metrics.json"
    if path.is_file() and record.is_file():
        if json.loads(record.read_text(encoding="utf-8")).get("fingerprint") == fingerprint:
            hulls = load_hulls(path)
            validate_hulls(hulls)
            return hulls

    started = time.perf_counter()
    if previous is not None and previous.record.get("hulls_fingerprint") == fingerprint:
        hulls = load_hulls(previous.file(previous.record["hulls"]))
        validate_hulls(hulls)
    else:
        print(f"Preparing {name} convex decomposition (CoACD {description['backend']})", flush=True)
        hulls = decompose_convex(mesh)
    metrics = dict(description, fingerprint=fingerprint, parts=len(hulls), seconds=time.perf_counter() - started,
        source_volume=float(mesh.volume), summed_hull_volume=float(sum(h.volume for h in hulls)),
        maximum_vertices=max(len(h.vertices) for h in hulls), **measure_hull_error(mesh, hulls))
    with store.staging(directory, prefix=".hulls-") as stage:
        save_hulls(hulls, stage / "hulls.npz")
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        # Publish hulls last: a cache hit has its completed diagnostics.
        store.publish([(stage / name, directory / name) for name in ("metrics.json", "hulls.npz")])
    print(f"Prepared {name}: {len(hulls)} convex pieces in {metrics['seconds']:.1f} s", flush=True)
    return hulls
