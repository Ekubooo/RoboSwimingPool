"""Deterministic convex decomposition and diagnostics; no file IO or cache."""

from importlib.metadata import version

import numpy as np
import trimesh


COACD_VERSION = "1.0.14"
COACD_OPTIONS = dict(threshold=.005, real_metric=True, max_convex_hull=32,
    preprocess_mode="auto", preprocess_resolution=50, resolution=2000,
    mcts_nodes=20, mcts_iterations=100, mcts_max_depth=3, pca=False,
    merge=True, decimate=True, max_ch_vertex=64, extrude=False,
    extrude_margin=.01, apx_mode="ch", seed=0)


def check_backend():
    installed = version("coacd")
    if installed != COACD_VERSION:
        raise RuntimeError(f"Expected coacd {COACD_VERSION}, found {installed}")
    return installed


def decompose_convex(mesh):
    """Run the fixed CoACD recipe; never silently substitute a single hull."""
    check_backend()
    import coacd
    coacd.set_log_level("info")
    parts = coacd.run_coacd(coacd.Mesh(
        np.ascontiguousarray(mesh.vertices, dtype=np.float64),
        np.ascontiguousarray(mesh.faces, dtype=np.int32)), **COACD_OPTIONS)
    hulls = [trimesh.Trimesh(vertices, faces, process=False) for vertices, faces in parts]
    validate_hulls(hulls)
    return tuple(hulls)


def validate_convex_mesh(mesh):
    """Generic convex solid validity without CoACD output-budget restrictions."""
    if not np.isfinite(mesh.vertices).all() or not mesh.is_volume or not mesh.is_convex:
        raise ValueError("Invalid convex solid")


def validate_hulls(hulls):
    if not hulls or len(hulls) > COACD_OPTIONS["max_convex_hull"]:
        raise ValueError("CoACD must produce between one and 32 convex solids")
    for hull in hulls:
        if len(hull.vertices) > COACD_OPTIONS["max_ch_vertex"]:
            raise ValueError("CoACD produced an invalid convex solid")
        try:
            validate_convex_mesh(hull)
        except ValueError:
            raise ValueError("CoACD produced an invalid convex solid") from None


def measure_hull_error(mesh, hulls):
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
    proxy = geometry.to_solid(geometry.union(hulls))
    source = geometry.to_solid(mesh)
    excess, missing = (proxy - source).volume(), (source - proxy).volume()
    return dict(sampled_hull_vertex_surface_distance_max=float(max(outside, default=0.)),
                sampled_hull_vertex_surface_distance_mean=float(np.mean(outside)) if outside else 0.,
                proxy_excess_volume=float(excess), proxy_missing_volume=float(missing),
                relative_symmetric_difference_volume=float((excess + missing) / mesh.volume))
