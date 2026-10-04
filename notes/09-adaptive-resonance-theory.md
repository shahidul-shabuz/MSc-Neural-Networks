# 09 — Adaptive Resonance Theory (ART-1)

> Handwritten: [pp. 47–52](../handwritten-notes/neural-network-handwritten-notes.pdf) · Exam: DUET 2018–19, Q3(b) · Code: [`art1.py`](../src/nncourse/art1.py) · Verified by [`test_art1_reset_then_learn`](../tests/test_worked_examples.py)

## Idea: the stability–plasticity dilemma

A learning system must stay **plastic** (able to learn new patterns) while remaining **stable** (not overwriting what it already knows). ART (Carpenter & Grossberg) addresses this with a **match-based** check:

1. **Bottom-up competition:** the input activates cluster units through bottom-up weights `B`, and the strongest unit `J` wins.
2. **Top-down expectation:** the winner's stored template (top-down weights `T_J`) is compared with the input.
3. **Vigilance test:** if the match ratio `‖x‖/‖s‖` is at least the vigilance `ρ`, the network **resonates** and learns. Otherwise the winner is **reset** (inhibited), and the next-best unit is tried. If no unit matches, a new cluster can be recruited.

ART-1 handles binary inputs. Higher `ρ` produces more, finer clusters; lower `ρ` produces fewer, coarser ones.

## Problem (DUET 2018–19, Q3b)

An ART-1 net with 5 input units and 3 cluster units has, after some training, these weights:

```text
         Y1   Y2   Y3                  1  2  3  4  5
B5×3 = [ 0.2  0.0  0.2 ]      T3×5 = [ 1  1  1  1  1 ]   Y1
       [ 0.5  0.8  0.2 ]             [ 0  1  1  1  0 ]   Y2
       [ 0.5  0.5  0.2 ]             [ 1  1  1  1  1 ]   Y3
       [ 0.5  0.8  0.2 ]
       [ 0.1  0.0  0.2 ]
```

Show the behavior of the net when presented with `s = [0, 1, 1, 1, 1]`, with `L = 2` (learning parameter) and `ρ = 0.8` (vigilance).

## Solution

**Steps 3–6: present the input.** Set the F1(a) activations to `s`, all F2 activations to 0, and propagate to the interface layer: `x = s = [0, 1, 1, 1, 1]`.

```text
‖s‖ = Σ s_i = 0 + 1 + 1 + 1 + 1 = 4
```

**Step 7: bottom-up net inputs** `y_j = Σ_i b_ij x_i`:

```text
y1 = 0·0.2 + 1·0.5 + 1·0.5 + 1·0.5 + 1·0.1 = 1.6
y2 = 0·0.0 + 1·0.8 + 1·0.5 + 1·0.8 + 1·0.0 = 2.1
y3 = 0·0.2 + 1·0.2 + 1·0.2 + 1·0.2 + 1·0.2 = 0.8
```

### First attempt: Y2

**Step 9: choose the winner.** No unit is inhibited; Y2 has the largest net input (2.1), so `J = 2`.

**Step 10: top-down check.** `x_i = s_i · t_Ji`:

```text
x = s ∘ T2 = [0,1,1,1,1] ∘ [0,1,1,1,0] = [0, 1, 1, 1, 0]
```

**Steps 11–12: vigilance test.**

```text
‖x‖ = 3,     ‖x‖/‖s‖ = 3/4 = 0.75 < ρ = 0.8   →  RESET
```

Y2's template does not match the input closely enough (it lacks the 5th component). Inhibit Y2: `y2 = −1`.

### Second attempt: Y1

**Step 9.** Activations are now `[1.6, −1, 0.8]`; Y1 wins, `J = 1`.

**Step 10.**

```text
x = s ∘ T1 = [0,1,1,1,1] ∘ [1,1,1,1,1] = [0, 1, 1, 1, 1]
```

**Steps 11–12.**

```text
‖x‖ = 4,     ‖x‖/‖s‖ = 4/4 = 1.0 ≥ 0.8   →  no reset; RESONANCE
```

### Step 13: learning (update the winner J = 1)

**Bottom-up weights** (column 1 of B):

```text
b_i1 = L · x_i / (L − 1 + ‖x‖) = 2 · x_i / (2 − 1 + 4) = 0.4 · x_i
b·1 = [0, 0.4, 0.4, 0.4, 0.4]
```

**Top-down weights** (row 1 of T): `t_1i = x_i`

```text
T1 = [0, 1, 1, 1, 1]
```

### Updated weights

```text
         Y1   Y2   Y3                  1  2  3  4  5
B5×3 = [ 0.0  0.0  0.2 ]      T3×5 = [ 0  1  1  1  1 ]   Y1
       [ 0.4  0.8  0.2 ]             [ 0  1  1  1  0 ]   Y2
       [ 0.4  0.5  0.2 ]             [ 1  1  1  1  1 ]   Y3
       [ 0.4  0.8  0.2 ]
       [ 0.4  0.0  0.2 ]
```

**Behavior summary:** Y2 initially wins the bottom-up competition but fails the vigilance test (0.75 < 0.8) and is reset. Y1 then wins, passes (1.0 ≥ 0.8), and **learns the pattern**: its template shrinks from `[1,1,1,1,1]` to exactly `s = [0,1,1,1,1]`, and its bottom-up weights are renormalized. Because ART-1 learning takes the AND of the template and the input, templates can only lose 1s, never gain them, which is what keeps learned categories stable.

## Advantages and limitations of ART

**Advantages**

- Stable: learning new inputs does not destroy existing categories (solves the stability–plasticity dilemma).
- Learns online, one pattern at a time, and creates new categories when needed.
- Can be combined with other techniques, e.g., ARTMAP for supervised learning.
- Applications: mobile robot control, face recognition, target recognition, medical diagnosis, signature verification.
- Advantages over plain competitive learning: the vigilance test prevents a winner from absorbing patterns that do not really match it.

**Limitations**

- Results depend on the **order** in which training data is presented, and on the vigilance and learning parameters (some variants, e.g., Fuzzy ART and ART-1, are inconsistent in this sense).
- ART-1 handles binary inputs only (Fuzzy ART / ART-2 extend it).
- Sensitive to noise: templates only shrink, so noisy inputs can erode them.

## Corrections to the handwritten version

- p. 47: row 5 of B is written as `1 0 0.2`; the exam paper and the calculation on p. 48 use **0.1**.
- p. 49: "inhibit the Jth cluster unit by setting y3 = −1" should read **y_J = −1**.
- p. 51: step 13 updates only T. The **bottom-up weights B must also be updated**: `b·1 = [0, 0.4, 0.4, 0.4, 0.4]`.
