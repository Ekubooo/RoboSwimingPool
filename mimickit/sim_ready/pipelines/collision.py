"""Prepare convex collision assets using the existing cache recipe and format."""

import json
import time

from ..lib.mesh.convex import COACD_OPTIONS, check_backend, decompose_convex, validate_hulls, measure_hull_error
from ..lib.storage.fingerprints import mesh_digest
from ..lib.storage.hull_io import load_hulls, save_hulls
from ..lib.storage.store import AssetStore


CACHE_VERSION = 1


def prepare_hulls(mesh, name, store: AssetStore):
    """Reuse validated parts or generate them; corrupt caches still raise."""
    installed = check_backend()
    description = dict(version=CACHE_VERSION, geometry=mesh_digest(mesh),
        backend=installed, options=COACD_OPTIONS)
    directory = store.recipe_directory("collision", description, name=name, allow_nan=True)
    path = directory / "hulls.npz"
    if path.is_file():
        hulls = load_hulls(path)
        validate_hulls(hulls)
        return hulls

    print(f"Preparing {name} convex decomposition (CoACD {installed})", flush=True)
    started = time.perf_counter()
    hulls = decompose_convex(mesh)
    metrics = dict(description, parts=len(hulls), seconds=time.perf_counter() - started,
        source_volume=float(mesh.volume), summed_hull_volume=float(sum(h.volume for h in hulls)),
        maximum_vertices=max(len(h.vertices) for h in hulls), **measure_hull_error(mesh, hulls))
    with store.staging(directory, prefix=".hulls-") as stage:
        save_hulls(hulls, stage / "hulls.npz")
        (stage / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        # Publish hulls last: a cache hit has its completed diagnostics.
        store.publish([(stage / name, directory / name) for name in ("metrics.json", "hulls.npz")])
    print(f"Prepared {name}: {len(hulls)} convex pieces in {metrics['seconds']:.1f} s", flush=True)
    return hulls
