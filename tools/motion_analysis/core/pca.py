"""Small deterministic, centered float64 PCA implementation."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAModel:
    mean: np.ndarray
    components: np.ndarray
    variance: np.ndarray
    variance_ratio: np.ndarray
    sample_count: int

    @property
    def max_components(self) -> int:
        return len(self.components)

    def transform(self, samples: np.ndarray) -> np.ndarray:
        return (samples - self.mean) @ self.components.T

    def inverse_transform(self, scores: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
        return self.mean + scores[:, list(indices)] @ self.components[list(indices)]


def fit_pca(samples: np.ndarray) -> PCAModel:
    """Fit all numerically nonzero components, without scaling or whitening."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 2 or samples.shape[1] < 1:
        raise ValueError("PCA requires a 2D matrix with at least two samples and one feature.")
    if not np.isfinite(samples).all():
        raise ValueError("PCA samples must be finite.")
    if np.all(samples == samples[0]):
        return PCAModel(samples[0].copy(), np.empty((0, samples.shape[1])),
                        np.empty(0), np.empty(0), len(samples))
    mean = samples.mean(axis=0)
    centered = samples - mean
    _, singular, basis = np.linalg.svd(centered, full_matrices=False)
    threshold = np.finfo(np.float64).eps * max(samples.shape) * singular[0]
    rank = min(int(np.count_nonzero(singular > threshold)), len(samples) - 1)
    basis = basis[:rank].copy()
    # Remove the arbitrary sign ambiguity for reproducible component displays.
    if rank:
        pivot = np.argmax(np.abs(basis), axis=1)
        signs = np.where(basis[np.arange(rank), pivot] < 0, -1.0, 1.0)
        basis *= signs[:, None]
    variance = singular[:rank] ** 2 / (len(samples) - 1)
    total = float(np.sum(singular ** 2))
    ratio = singular[:rank] ** 2 / total if total > 0 else np.zeros(0)
    return PCAModel(mean, basis, variance, ratio, len(samples))
