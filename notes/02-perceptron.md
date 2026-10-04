# 02 — Perceptron Learning

> Lecture 01 exercise · Verified by [`test_perceptron_one_epoch`](../tests/test_worked_examples.py) · Code: [`perceptron.py`](../src/nncourse/perceptron.py)

## Problem

A perceptron has initial weights `w1 = 0.4`, `w2 = −0.2`, threshold `θ = 0.1` and learning rate `c = 1`. Train it for one epoch on:

| x1 | x2 | target t |
|---:|---:|---:|
| 0.8 | 0.3 | 1 |
| 0.4 | 0.1 | 0 |

## Learning rule

```text
net = w1·x1 + w2·x2
z   = 1 if net ≥ θ, else 0
if z ≠ t:   Δwi = c·(t − z)·xi,   wi ← wi + Δwi
```

The update only happens on a mistake. It moves the weights **towards** the input when the output was too low (t = 1, z = 0) and **away** from it when the output was too high (t = 0, z = 1).

## Solution

**Sample 1:** `net = 0.8×0.4 + 0.3×(−0.2) = 0.32 − 0.06 = 0.26 ≥ 0.1` → `z = 1 = t` → correct, no update.

**Sample 2:** `net = 0.4×0.4 + 0.1×(−0.2) = 0.16 − 0.02 = 0.14 ≥ 0.1` → `z = 1 ≠ t = 0` → update:

```text
Δw1 = 1·(0 − 1)·0.4 = −0.4   →  w1 = 0.4 − 0.4 =  0.0
Δw2 = 1·(0 − 1)·0.1 = −0.1   →  w2 = −0.2 − 0.1 = −0.3
```

**Weights after one epoch:** `w1 = 0.0`, `w2 = −0.3`.

## Check

With the new weights, sample 1 gives `net = 0.8×0 + 0.3×(−0.3) = −0.09 < 0.1`, so `z = 0`: it is now **wrong**. One epoch is not enough.

The data is linearly separable (e.g., `w = (0.2, 0)` gives 0.16 ≥ 0.1 and 0.08 < 0.1), so by the perceptron convergence theorem training must eventually stop. Running the same rule in code, the weights oscillate before settling: **the last update happens in epoch 15, giving `w = (0.0, 0.4)`, and epoch 16 makes no errors** (sample 1: 0.12 ≥ 0.1; sample 2: 0.04 < 0.1). The slow progress is because `c = 1` makes each step large relative to the inputs, and the threshold is fixed rather than learned.
