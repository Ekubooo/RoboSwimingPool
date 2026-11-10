"""Low-dimensional normalized density estimates with an explicit shared kernel."""

import numpy as np


def reference_bandwidth(points: np.ndarray) -> np.ndarray:
    """Scott bandwidth from reference points, reused verbatim for every group."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3 or not np.isfinite(points).all():
        raise ValueError("Density input must be finite [N,2] points with N >= 3.")
    covariance = np.cov(points, rowvar=False)
    bandwidth = covariance * len(points) ** (-1 / 3)
    if np.linalg.eigvalsh(bandwidth).min() <= 0:
        raise ValueError("Reference density bandwidth must be positive definite.")
    return bandwidth


def gaussian_density(points: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Maximum-likelihood Gaussian; its covariance is a fitted model parameter."""
    points = np.asarray(points, dtype=np.float64)
    covariance = np.cov(points, rowvar=False, bias=True)
    return _normal(grid - points.mean(axis=0), covariance)


def _normal(delta: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    determinant = float(np.linalg.det(covariance))
    if determinant <= 0 or not np.isfinite(determinant):
        raise ValueError("Density covariance must be finite positive definite.")
    inverse = np.linalg.inv(covariance)
    squared = np.einsum("...i,ij,...j->...", delta, inverse, delta)
    return np.exp(-0.5 * squared) / (2 * np.pi * np.sqrt(determinant))


def kernel_density(points: np.ndarray, grid: np.ndarray, bandwidth: np.ndarray) -> np.ndarray:
    """Equal-weight Gaussian mixture; fixed bandwidth permits honest comparisons."""
    points = np.asarray(points, dtype=np.float64)
    flat = np.asarray(grid, dtype=np.float64).reshape(-1, 2)
    values = np.empty(len(flat))
    for start in range(0, len(flat), 256):
        values[start:start + 256] = _normal(
            flat[start:start + 256, None, :] - points[None, :, :], bandwidth).mean(axis=1)
    return values.reshape(grid.shape[:-1])
