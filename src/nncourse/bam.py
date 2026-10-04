"""Bidirectional Associative Memory (Kosko) with bipolar patterns."""

from __future__ import annotations

import numpy as np


def weight_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """W = Σ_k A_kᵀ B_k  (sum of outer products), shape len(A_k) × len(B_k)."""
    return sum(np.outer(a, b) for a, b in zip(A, B))


def threshold(net: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Bipolar hard limiter: +1 if net > 0, −1 if net < 0, unchanged if 0."""
    out = previous.copy()
    out[net > 0] = 1
    out[net < 0] = -1
    return out


def recall_forward(W: np.ndarray, a: np.ndarray) -> np.ndarray:
    """A → B direction: f(a W)."""
    return threshold(a @ W, np.zeros(W.shape[1], dtype=int))


def recall_backward(W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """B → A direction: f(b Wᵀ)."""
    return threshold(b @ W.T, np.zeros(W.shape[0], dtype=int))
