"""ART-1 for binary inputs (Fausett's formulation)."""

from __future__ import annotations

import numpy as np


def present(
    s: np.ndarray, B: np.ndarray, T: np.ndarray, L: float, rho: float
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Present one binary pattern s and run steps 3–13 of the ART-1 algorithm.

    B: bottom-up weights, shape (n_inputs, n_clusters)
    T: top-down weights,  shape (n_clusters, n_inputs)
    L: learning parameter (> 1);  rho: vigilance (0 < ρ ≤ 1)

    Returns updated (B, T) and a trace of every reset decision.
    """
    B, T = B.astype(float).copy(), T.astype(float).copy()
    norm_s = s.sum()
    y = s @ B                                   # F2 net inputs
    inhibited = np.zeros(len(y), dtype=bool)
    resets = []
    while True:
        if inhibited.all():
            return B, T, {"winner": None, "resets": resets, "y": y}
        J = int(np.where(inhibited, -np.inf, y).argmax())   # ties → lowest index
        x = s * T[J]                            # top-down match (AND)
        ratio = x.sum() / norm_s
        if ratio < rho:                         # vigilance fails → reset
            resets.append({"J": J, "x": x, "ratio": ratio})
            inhibited[J] = True
            continue
        break
    # Step 13: learning for the winner J (fast learning)
    B[:, J] = L * x / (L - 1 + x.sum())
    T[J] = x
    return B, T, {"winner": J, "resets": resets, "y": y, "x": x, "ratio": ratio}
