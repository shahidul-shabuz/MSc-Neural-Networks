"""K-means clustering with a choice of distance function."""

from __future__ import annotations

from typing import Callable

import numpy as np

Distance = Callable[[np.ndarray, np.ndarray], np.ndarray]


def manhattan(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """City-block distance |x₁ − c₁| + |x₂ − c₂| + … from every row of X to c."""
    return np.abs(X - c).sum(axis=1)


def squared_euclidean(X: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Squared Euclidean distance; gives the same assignments as Euclidean."""
    return ((X - c) ** 2).sum(axis=1)


def assign(X: np.ndarray, centers: np.ndarray, dist: Distance) -> np.ndarray:
    """Index of the nearest center for each point (ties → lowest index)."""
    D = np.stack([dist(X, c) for c in centers], axis=1)
    return D.argmin(axis=1)


def kmeans(
    X: np.ndarray,
    init_centers: np.ndarray,
    dist: Distance = squared_euclidean,
    max_iter: int = 100,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Lloyd's K-means: alternate (1) assign points, (2) recompute means.

    Stops when assignments no longer change. Returns the final centers,
    the labels, and a per-iteration trace.

    Note: the center update uses the *mean*, which minimises squared
    Euclidean error. With Manhattan distance the error-minimising center is
    the median (K-medians); the exam problems nevertheless use the mean, so
    we follow that convention.
    """
    centers = init_centers.astype(float).copy()
    labels = None
    trace = []
    for it in range(1, max_iter + 1):
        new_labels = assign(X, centers, dist)
        if labels is not None and np.array_equal(new_labels, labels):
            break
        labels = new_labels
        centers = np.array(
            [X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k]
             for k in range(len(centers))]
        )
        trace.append({"iteration": it, "labels": labels.copy(), "centers": centers.copy()})
    return centers, labels, trace


def sse(X: np.ndarray, centers: np.ndarray, labels: np.ndarray) -> float:
    """Sum of squared errors, the quantity plotted in the elbow method."""
    return float(((X - centers[labels]) ** 2).sum())
