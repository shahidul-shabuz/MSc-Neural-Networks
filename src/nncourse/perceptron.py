"""Single-layer perceptron with a hard threshold."""

from __future__ import annotations

import numpy as np


def predict(w: np.ndarray, x: np.ndarray, theta: float) -> int:
    """Return 1 if the net input w·x reaches the threshold, else 0."""
    return int(np.dot(w, x) >= theta)


def train_epoch(
    w: np.ndarray,
    X: np.ndarray,
    t: np.ndarray,
    theta: float,
    c: float = 1.0,
) -> tuple[np.ndarray, list[dict]]:
    """Run one epoch of the perceptron learning rule.

    Update rule (only when the output is wrong): w ← w + c·(t − z)·x.

    Returns the updated weights and a per-sample trace for inspection.
    """
    w = w.astype(float).copy()
    trace = []
    for x, target in zip(X, t):
        net = float(np.dot(w, x))
        z = int(net >= theta)
        if z != target:
            w = w + c * (target - z) * x
        trace.append({"net": net, "output": z, "target": int(target), "w": w.copy()})
    return w, trace
