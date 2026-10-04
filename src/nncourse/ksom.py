"""Kohonen self-organising map with radius R = 0 (winner-take-all)."""

from __future__ import annotations

import numpy as np


def winner(W: np.ndarray, x: np.ndarray) -> tuple[int, np.ndarray]:
    """Index of the cluster unit closest to x, and the squared distances.

    W has one *column* per cluster unit (shape: n_features × n_clusters),
    matching the layout used in the course notes.
    """
    d2 = ((W - x[:, None]) ** 2).sum(axis=0)
    return int(d2.argmin()), d2


def train_epoch(W: np.ndarray, X: np.ndarray, alpha: float) -> tuple[np.ndarray, list[dict]]:
    """One pass over X. Only the winning column is updated (R = 0):

        w_iJ ← w_iJ + α (x_i − w_iJ)
    """
    W = W.astype(float).copy()
    trace = []
    for x in X:
        J, d2 = winner(W, x)
        W[:, J] += alpha * (x - W[:, J])
        trace.append({"x": x, "d2": d2, "winner": J, "W": W.copy()})
    return W, trace


def train(
    W: np.ndarray, X: np.ndarray, alpha0: float, decay: float, epochs: int
) -> np.ndarray:
    """Several epochs with a geometric learning-rate schedule α ← decay·α."""
    alpha = alpha0
    for _ in range(epochs):
        W, _ = train_epoch(W, X, alpha)
        alpha *= decay
    return W


def recall(W: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cluster index for every row of X (nearest weight column)."""
    return np.array([winner(W, x)[0] for x in X])
