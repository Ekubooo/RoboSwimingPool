"""Offline convex collision generation, validation and deterministic build cache."""

import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import trimesh


CACHE_VERSION = 1
COACD_VERSION = "1.0.14"
COACD_OPTIONS = dict(threshold=.005, real_metric=True, max_convex_hull=32,
    preprocess_mode="auto", preprocess_resolution=50, resolution=2000,
    mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3, pca=False,
    merge=True, decimate=True, max_ch_vertex=64, extrude=False,
    extrude_margin=.01, apx_mode="ch", seed=0)


def mesh_digest(mesh):
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.vertices, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.faces, dtype=np.int32).tobytes())
    return digest.hexdigest()


def cache_directory(root, name, description):
    key = hashlib.sha256(json.dumps(description, sort_keys=True).encode()).hexdigest()[:20]
    directory = Path(root) / f"{name}-{key}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _validate_hulls(hulls):
    if not hulls or len(hulls) > COACD_OPTIONS["max_convex_hull"]:
        raise ValueError("CoACD must produce between one and 32 convex solids")
    for hull in hulls:
        if (len(hull.vertices) > COACD_OPTIONS["max_ch_vertex"]
                or not np.isfinite(hull.vertices).all() or not hull.is_volume or not hull.is_convex):
            raise ValueError("CoACD produced an invalid convex solid")


def _surface_error(mesh, hulls):
    """Record hull-vertex distance to the source surface, not a Hausdorff bound.

    Convex pieces also have internal cut faces; their vertices can be inside
    the source. The unsigned statistic is descriptive, not a collision margin.
    """
    vertices = np.concatenate([h.vertices for h in hulls])
    # Closest-point distance needs no optional spatial-index dependency.
    outside = []
    for chunk in np.array_split(vertices, max(1, len(vertices) // 32 + 1)):
        distance = trimesh.proximity.closest_point_naive(mesh, chunk)[1]
        outside.extend(distance.tolist())
    from . import geometry
    proxy = geometry._solid(geometry.union(hulls))
    source = geometry._solid(mesh)
    excess, missing = (proxy - source).volume(), (source - proxy).volume()
    return dict(sampled_hull_vertex_surface_distance_max=float(max(outside, default=0.)),
                sampled_hull_vertex_surface_distance_mean=float(np.mean(outside)) if outside else 0.,
                proxy_excess_volume=float(excess), proxy_missing_volume=float(missing),
                relative_symmetric_difference_volume=float((excess + missing) / mesh.volume))


def prepare_hulls(mesh, name, root):
    """Cache deterministic CoACD output; failures never fall back to a hull."""
    installed = version("coacd")
    if installed != COACD_VERSION:
        raise RuntimeError(f"Expected coacd {COACD_VERSION}, found {installed}")
    description = dict(version=CACHE_VERSION, geometry=mesh_digest(mesh),
        backend=installed, options=COACD_OPTIONS)
    directory = cache_directory(root, name, description)
    path = directory / "hulls.npz"
    if path.is_file():
        with np.load(path, allow_pickle=False) as saved:
            count = int(saved["count"])
            hulls = [trimesh.Trimesh(saved[f"vertices_{i}"], saved[f"faces_{i}"], process=False)
                     for i in range(count)]
        _validate_hulls(hulls)
        return tuple(hulls)

    import coacd
    coacd.set_log_level("info")
    print(f"Preparing {name} convex decomposition (CoACD {installed})", flush=True)
    started = time.perf_counter()
    parts = coacd.run_coacd(coacd.Mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32)), **COACD_OPTIONS)
    hulls = [trimesh.Trimesh(vertices, faces, process=False) for vertices, faces in parts]
    _validate_hulls(hulls)
    metrics = dict(description, parts=len(hulls), seconds=time.perf_counter() - started,
        source_volume=float(mesh.volume), summed_hull_volume=float(sum(h.volume for h in hulls)),
        maximum_vertices=max(len(h.vertices) for h in hulls), **_surface_error(mesh, hulls))
    arrays = {"count": np.asarray(len(hulls), dtype=np.int32)}
    for i, hull in enumerate(hulls):
        arrays[f"vertices_{i}"] = np.asarray(hull.vertices, dtype=np.float64)
        arrays[f"faces_{i}"] = np.asarray(hull.faces, dtype=np.int32)
    temporary = directory / "hulls.tmp.npz"
    np.savez_compressed(temporary, **arrays)
    metadata = directory / "metrics.tmp.json"
    metadata.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    metadata.replace(directory / "metrics.json")
    # Publish the hulls last: a cache hit always has its completed diagnostics.
    temporary.replace(path)
    print(f"Prepared {name}: {len(hulls)} convex pieces in {metrics['seconds']:.1f} s", flush=True)
    return tuple(hulls)
