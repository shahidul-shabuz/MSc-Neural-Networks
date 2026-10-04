# 06 — Kohonen Self-Organizing Map (KSOM)

> Handwritten: [pp. 28–32](../handwritten-notes/neural-network-handwritten-notes.pdf) · Code: [`ksom.py`](../src/nncourse/ksom.py) · Verified by [`test_ksom_one_epoch_and_recall`](../tests/test_worked_examples.py)

## Idea

A SOM is an **unsupervised, competitive** network. Each cluster unit has a weight vector. For every input:

1. **Compete:** find the winning unit `J` whose weight vector is closest to the input.
2. **Adapt:** move the winner (and, if the radius `R > 0`, its neighbors on the map) toward the input:

```text
w_iJ(new) = w_iJ(old) + α (x_i − w_iJ(old))
```

Over time, each weight vector moves to the center of a group of similar inputs. With `R = 0`, as here, only the winner learns ("winner-take-all").

## Problem

Cluster four vectors into two clusters J1 and J2:

```text
x1 = (1, 1, 0, 0)    x2 = (0, 0, 0, 1)    x3 = (1, 0, 0, 0)    x4 = (0, 0, 1, 1)
```

Learning rate `α = 0.6`, radius `R = 0`. Initial weights, one **column** per cluster unit:

```text
        J1    J2
W  = [ 0.2   0.8 ]
     [ 0.6   0.4 ]
     [ 0.5   0.7 ]
     [ 0.9   0.3 ]
```

The winner is found with the squared Euclidean distance `D(J) = Σ_i (w_iJ − x_i)²`. Taking the square root does not change which unit wins.

## Solution: one epoch

### Vector 1: (1, 1, 0, 0)

```text
D(1) = (0.2−1)² + (0.6−1)² + (0.5−0)² + (0.9−0)² = 0.64 + 0.16 + 0.25 + 0.81 = 1.86
D(2) = (0.8−1)² + (0.4−1)² + (0.7−0)² + (0.3−0)² = 0.04 + 0.36 + 0.49 + 0.09 = 0.98
```

**J2 wins.** Update column 2:

```text
w12 = 0.8 + 0.6(1 − 0.8) = 0.92        w22 = 0.4 + 0.6(1 − 0.4) = 0.76
w32 = 0.7 + 0.6(0 − 0.7) = 0.28        w42 = 0.3 + 0.6(0 − 0.3) = 0.12
```

### Vector 2: (0, 0, 0, 1)

```text
D(1) = 0.04 + 0.36 + 0.25 + 0.01 = 0.66
D(2) = 0.8464 + 0.5776 + 0.0784 + 0.7744 = 2.2768
```

**J1 wins.** Update column 1:

```text
w11 = 0.2 + 0.6(0 − 0.2) = 0.08        w21 = 0.6 + 0.6(0 − 0.6) = 0.24
w31 = 0.5 + 0.6(0 − 0.5) = 0.20        w41 = 0.9 + 0.6(1 − 0.9) = 0.96
```

### Vector 3: (1, 0, 0, 0)

```text
D(1) = 0.8464 + 0.0576 + 0.04 + 0.9216 = 1.8656
D(2) = 0.0064 + 0.5776 + 0.0784 + 0.0144 = 0.6768
```

**J2 wins.** Update column 2:

```text
w12 = 0.92 + 0.6(1 − 0.92) = 0.968     w22 = 0.76 + 0.6(0 − 0.76) = 0.304
w32 = 0.28 + 0.6(0 − 0.28) = 0.112     w42 = 0.12 + 0.6(0 − 0.12) = 0.048
```

### Vector 4: (0, 0, 1, 1)

```text
D(1) = 0.0064 + 0.0576 + 0.64 + 0.0016 = 0.7056
D(2) = 0.937 + 0.0924 + 0.7885 + 0.9063 ≈ 2.7243
```

**J1 wins.** Update column 1:

```text
w11 = 0.08 + 0.6(0 − 0.08) = 0.032     w21 = 0.24 + 0.6(0 − 0.24) = 0.096
w31 = 0.20 + 0.6(1 − 0.20) = 0.680     w41 = 0.96 + 0.6(1 − 0.96) = 0.984
```

### Weights after epoch 1

```text
        J1      J2
W  = [ 0.032   0.968 ]
     [ 0.096   0.304 ]
     [ 0.680   0.112 ]
     [ 0.984   0.048 ]
```

J1 has moved toward the patterns ending in 1 (x2, x4), and J2 toward the patterns starting with 1 (x1, x3).

## Recall mode

Assign each vector to its nearest weight column:

| Vector | D(1) | D(2) | Cluster |
|---|---:|---:|:---:|
| x1 (1,1,0,0) | 3.18 | **0.50** | J2 |
| x2 (0,0,0,1) | **0.47** | 1.95 | J1 |
| x3 (1,0,0,0) | 2.38 | **0.11** | J2 |
| x4 (0,0,1,1) | **0.11** | 2.72 | J1 |

**Result:** cluster J1 = {x2, x4}, cluster J2 = {x1, x3}.

The notes compute recall as a matrix product, which gives the same answer here. Note that the input matrix must have the **vectors as columns**:

```text
Wᵀ Xᵀ = [ 0.032 0.096 0.680 0.984 ]   [ 1 0 1 0 ]   [ 0.128  0.984  0.032  1.664 ]
        [ 0.968 0.304 0.112 0.048 ] × [ 1 0 0 0 ] = [ 1.272  0.048  0.968  0.160 ]
                                      [ 0 0 0 1 ]
                                      [ 0 1 0 1 ]
```

The larger entry in each column (x1 → J2, x2 → J1, x3 → J2, x4 → J1) matches the distance-based result. In general, the **largest dot product equals the smallest distance only when the weight vectors have equal norms**, so the distance form is the safer rule.

## Training schedule

In a full training run, α is reduced after each epoch (e.g., `α(t+1) = 0.5 α(t)`) so that the weights settle. Repeating epochs makes the weights converge toward the cluster means.

## Corrections to the handwritten version

- p. 28: `J1 = 1.36` is a square root (√1.86) but `J2 = 0.98` is a squared distance (√0.98 ≈ 0.99). On pp. 29–31 all values are squared distances. Use one convention throughout; the winners are unchanged.
- p. 30: "J2 wins (0.6768 < 0.6768)" should read **0.6768 < 1.8656**.
- p. 32: the recall step was left unfinished, and the input matrix was written with vectors as rows. It must be transposed (vectors as columns). Completed above.
