"""The 2-2-1 sigmoid network from the backpropagation worked example.

Units: u0 = bias (x0 = 1), u1, u2 = inputs; u3, u4 = hidden; u5 = output.
Weights are named w_ij for the connection from unit i to unit j.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


def sigmoid(a: float) -> float:
    return 1.0 / (1.0 + np.exp(-a))


@dataclass(frozen=True)
class Weights:
    w03: float
    w13: float
    w23: float
    w04: float
    w14: float
    w24: float
    w05: float
    w35: float
    w45: float


def forward(w: Weights, x1: float, x2: float, x0: float = 1.0) -> dict:
    """Forward pass; returns net inputs a_j and outputs y_j."""
    a3 = w.w03 * x0 + w.w13 * x1 + w.w23 * x2
    a4 = w.w04 * x0 + w.w14 * x1 + w.w24 * x2
    y3, y4 = sigmoid(a3), sigmoid(a4)
    a5 = w.w05 * x0 + w.w35 * y3 + w.w45 * y4
    y5 = sigmoid(a5)
    return {"a3": a3, "y3": y3, "a4": a4, "y4": y4, "a5": a5, "y5": y5}


def backprop_step(
    w: Weights, x1: float, x2: float, target: float, eta: float, x0: float = 1.0
) -> tuple[Weights, dict]:
    """One step of the generalised delta rule.

    Deltas (sigmoid derivative y(1 − y)):
        δ5 = y5(1 − y5)(t − y5)
        δj = yj(1 − yj)·δ5·w_j5            for hidden j ∈ {3, 4}
    Updates:
        Δw_ij = η·δj·y_i,   w_ij ← w_ij + Δw_ij

    All deltas are computed with the *old* weights before any update.
    """
    f = forward(w, x1, x2, x0)
    d5 = f["y5"] * (1 - f["y5"]) * (target - f["y5"])
    d4 = f["y4"] * (1 - f["y4"]) * d5 * w.w45
    d3 = f["y3"] * (1 - f["y3"]) * d5 * w.w35

    new = replace(
        w,
        w03=w.w03 + eta * d3 * x0, w13=w.w13 + eta * d3 * x1, w23=w.w23 + eta * d3 * x2,
        w04=w.w04 + eta * d4 * x0, w14=w.w14 + eta * d4 * x1, w24=w.w24 + eta * d4 * x2,
        w05=w.w05 + eta * d5 * x0, w35=w.w35 + eta * d5 * f["y3"], w45=w.w45 + eta * d5 * f["y4"],
    )
    return new, {**f, "delta3": d3, "delta4": d4, "delta5": d5}
