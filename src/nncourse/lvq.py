"""Learning Vector Quantisation (LVQ1)."""

from __future__ import annotations

import numpy as np


def train_epoch(
    W: np.ndarray,
    W_classes: np.ndarray,
    X: np.ndarray,
    T: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, list[dict]]:
    """One LVQ1 epoch. W has one reference vector per *row*.

    For each input x with target class T, find the nearest reference vector J:
        correct class (T = C_J):  w_J ← w_J + α (x − w_J)   (move closer)
        wrong class   (T ≠ C_J):  w_J ← w_J − α (x − w_J)   (move away)
    """
    W = W.astype(float).copy()
    trace = []
    for x, t in zip(X, T):
        d2 = ((W - x) ** 2).sum(axis=1)
        J = int(d2.argmin())
        sign = 1.0 if W_classes[J] == t else -1.0
        W[J] += sign * alpha * (x - W[J])
        trace.append({"x": x, "target": int(t), "d2": d2, "winner": J,
                      "correct": sign > 0, "W": W.copy()})
    return W, trace
