# 07 — Learning Vector Quantization (LVQ)

> Handwritten: [pp. 39–44](../handwritten-notes/neural-network-handwritten-notes.pdf) · Exam: DUET 2018–19, Q6(a) · Code: [`lvq.py`](../src/nncourse/lvq.py) · Verified by [`test_lvq_one_epoch`](../tests/test_worked_examples.py)

## Idea

LVQ is the **supervised** counterpart of the Kohonen map. Each output unit holds a **reference (codebook) vector** labelled with a class. For each training input `x` with target class `T`:

1. find the nearest reference vector `J`,
2. if its class is correct (`T = C_J`), **pull** it toward the input; if wrong, **push** it away:

```text
T = C_J:   w_J(new) = w_J(old) + α (x − w_J(old))
T ≠ C_J:   w_J(new) = w_J(old) − α (x − w_J(old))
```

3. reduce α after each epoch, and stop after a fixed number of epochs or when α is small enough.

After training, a new input is classified by the class of its nearest reference vector. The classes are separated by the boundaries between reference vectors (a Voronoi partition).

## Problem

Construct and test an LVQ net with five vectors assigned to two classes, `α = 0.1`.

| Vector | Class |
|---|:---:|
| [0, 0, 1, 1] | 1 |
| [1, 0, 0, 0] | 2 |
| [0, 0, 0, 1] | 2 |
| [1, 1, 0, 0] | 1 |
| [0, 1, 1, 0] | 1 |

The first two vectors initialize the reference vectors; the remaining three are training inputs.

```text
w1 = [0, 0, 1, 1]   (class 1)        w2 = [1, 0, 0, 0]   (class 2)
```

Distance: `D(J) = Σ_i (w_iJ − x_i)²`.

## Solution: first epoch

### Input 1: x = [0, 0, 0, 1], T = 2

```text
D(1) = (0−0)² + (0−0)² + (1−0)² + (1−1)² = 1
D(2) = (1−0)² + (0−0)² + (0−0)² + (0−1)² = 2
```

`J = 1` (class 1) ≠ `T = 2` → **wrong**, push w1 away:

```text
w1 = w1 − 0.1 (x − w1) = [0,0,1,1] − 0.1·[0, 0, −1, 0] = [0, 0, 1.1, 1]
```

### Input 2: x = [1, 1, 0, 0], T = 1

```text
D(1) = 1 + 1 + 1.21 + 1 = 4.21
D(2) = 0 + 1 + 0 + 0 = 1
```

`J = 2` (class 2) ≠ `T = 1` → **wrong**, push w2 away:

```text
w2 = w2 − 0.1 (x − w2) = [1,0,0,0] − 0.1·[0, 1, 0, 0] = [1, −0.1, 0, 0]
```

### Input 3: x = [0, 1, 1, 0], T = 1

```text
D(1) = 0 + 1 + 0.01 + 1 = 2.01
D(2) = 1 + 1.21 + 1 + 0 = 3.21
```

`J = 1` (class 1) = `T = 1` → **correct**, pull w1 closer:

```text
w1 = w1 + 0.1 (x − w1) = [0,0,1.1,1] + 0.1·[0, 1, −0.1, −1] = [0, 0.1, 1.09, 0.9]
```

### After epoch 1

```text
w1 = [0, 0.1, 1.09, 0.9]   (class 1)
w2 = [1, −0.1, 0, 0]       (class 2)
```

The first two inputs were misclassified, so training continues. Reduce the learning rate, e.g., `α(t+1) = 0.5 α(t) = 0.05`, and repeat epochs until all inputs are classified correctly or α is sufficiently small.

## LVQ vs. KSOM

| | KSOM | LVQ |
|---|---|---|
| Learning | Unsupervised | Supervised |
| Winner update | Always toward the input | Toward if correct, away if wrong |
| Neighborhood | Winner and neighbors (radius R) | Winner only |
| Output | Clusters | Class labels |
