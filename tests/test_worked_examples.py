"""Each test reproduces a worked example from the handwritten course notes.

Expected values are the hand-calculated results (rounded as in the notes).
Run with:  pytest -q
"""

import numpy as np
import pytest

from nncourse import art1, bam, kmeans, ksom, lvq, mlp, perceptron


# --- Perceptron (Lecture 01) -------------------------------------------------
def test_perceptron_one_epoch():
    w, trace = perceptron.train_epoch(
        np.array([0.4, -0.2]), np.array([[0.8, 0.3], [0.4, 0.1]]), np.array([1, 0]),
        theta=0.1, c=1.0,
    )
    assert [round(t["net"], 2) for t in trace] == [0.26, 0.14]
    assert [t["output"] for t in trace] == [1, 1]          # 2nd sample misclassified
    np.testing.assert_allclose(w, [0.0, -0.3])


# --- K-means, 8 points, Manhattan distance -----------------------------------
POINTS = np.array([[2, 10], [2, 5], [8, 4], [5, 8], [7, 5], [6, 4], [1, 2], [4, 9]], float)


def test_kmeans_manhattan_two_iterations():
    _, _, trace = kmeans.kmeans(POINTS, POINTS[[0, 3, 6]], kmeans.manhattan)
    it1, it2 = trace[0], trace[1]
    # Iteration 1: C1 = {A1}, C2 = {A3, A4, A5, A6, A8}, C3 = {A2, A7}
    assert list(it1["labels"] + 1) == [1, 3, 2, 2, 2, 2, 3, 2]
    np.testing.assert_allclose(it1["centers"], [[2, 10], [6, 6], [1.5, 3.5]])
    # Iteration 2: A8 moves from C2 to C1
    assert list(it2["labels"] + 1) == [1, 3, 2, 2, 2, 2, 3, 1]
    np.testing.assert_allclose(it2["centers"], [[3, 9.5], [6.5, 5.25], [1.5, 3.5]])


def test_kmeans_manhattan_converges_after_three_iterations():
    centers, labels, trace = kmeans.kmeans(POINTS, POINTS[[0, 3, 6]], kmeans.manhattan)
    assert len(trace) == 3                                   # A4 moves to C1 in iteration 3
    assert list(labels + 1) == [1, 3, 2, 1, 2, 2, 3, 1]
    np.testing.assert_allclose(centers, [[11 / 3, 9], [7, 13 / 3], [1.5, 3.5]])


# --- K-means, medicines, squared Euclidean (DUET 2018-19, Q4b) ----------------
def test_kmeans_medicines():
    X = np.array([[1, 1], [2, 3], [4, 5], [3, 1]], float)    # a, b, c, d
    centers, labels, trace = kmeans.kmeans(X, X[[0, 1]], kmeans.squared_euclidean)
    assert list(labels + 1) == [1, 2, 2, 1]                  # {a, d}, {b, c}
    np.testing.assert_allclose(centers, [[2, 1], [3, 4]])
    assert len(trace) == 1                                   # converged: no change in iteration 2


# --- Backpropagation, 2-2-1 sigmoid network (Lecture 05) ---------------------
W0 = mlp.Weights(w03=1, w13=3, w23=4, w04=-6, w14=6, w24=5, w05=-3.92, w35=2, w45=4)


def test_backprop_forward_pass():
    f = mlp.forward(W0, x1=1, x2=0)
    assert f["a3"] == pytest.approx(4)
    assert f["y3"] == pytest.approx(0.982, abs=1e-3)
    assert f["a4"] == pytest.approx(0)
    assert f["y4"] == pytest.approx(0.5)
    assert f["a5"] == pytest.approx(0.044, abs=1e-3)
    assert f["y5"] == pytest.approx(0.511, abs=1e-3)


def test_backprop_deltas_and_updates():
    new, info = mlp.backprop_step(W0, x1=1, x2=0, target=1, eta=0.1)
    # The notes round y5 to 0.51, giving δ5 = δ4 ≈ 0.1225; exact values are ≈ 0.1222.
    assert info["delta5"] == pytest.approx(0.1225, abs=5e-4)
    assert info["delta4"] == pytest.approx(0.1225, abs=5e-4)
    assert info["delta3"] == pytest.approx(0.0043, abs=1e-4)
    assert new.w03 == pytest.approx(1.0004, abs=1e-4)
    assert new.w13 == pytest.approx(3.0004, abs=1e-4)
    assert new.w23 == pytest.approx(4.0)                     # x2 = 0 → no change
    assert new.w04 == pytest.approx(-5.9878, abs=1e-3)
    assert new.w14 == pytest.approx(6.0123, abs=1e-3)
    assert new.w24 == pytest.approx(5.0)
    assert new.w05 == pytest.approx(-3.9078, abs=1e-3)
    assert new.w35 == pytest.approx(2.0120, abs=1e-3)
    assert new.w45 == pytest.approx(4.0061, abs=1e-3)


def test_backprop_reduces_error():
    new, _ = mlp.backprop_step(W0, x1=1, x2=0, target=1, eta=0.1)
    old_err = 1 - mlp.forward(W0, 1, 0)["y5"]
    new_err = 1 - mlp.forward(new, 1, 0)["y5"]
    assert new_err == pytest.approx(0.476, abs=1e-3)
    assert new_err < old_err


# --- Kohonen SOM (R = 0, α = 0.6) ---------------------------------------------
def test_ksom_one_epoch_and_recall():
    W = np.array([[0.2, 0.8], [0.6, 0.4], [0.5, 0.7], [0.9, 0.3]])
    X = np.array([[1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 1]], float)
    W_new, trace = ksom.train_epoch(W, X, alpha=0.6)
    assert [t["winner"] + 1 for t in trace] == [2, 1, 2, 1]
    np.testing.assert_allclose(
        W_new, [[0.032, 0.968], [0.096, 0.304], [0.680, 0.112], [0.984, 0.048]], atol=1e-9
    )
    assert list(ksom.recall(W_new, X) + 1) == [2, 1, 2, 1]


# --- LVQ (α = 0.1) ------------------------------------------------------------
def test_lvq_one_epoch():
    W = np.array([[0, 0, 1, 1], [1, 0, 0, 0]], float)      # classes 1 and 2
    X = np.array([[0, 0, 0, 1], [1, 1, 0, 0], [0, 1, 1, 0]], float)
    T = np.array([2, 1, 1])
    W_new, trace = lvq.train_epoch(W, np.array([1, 2]), X, T, alpha=0.1)
    assert [t["correct"] for t in trace] == [False, False, True]
    np.testing.assert_allclose(W_new, [[0, 0.1, 1.09, 0.9], [1, -0.1, 0, 0]])


# --- BAM (DUET 2020-21) --------------------------------------------------------
A = np.array([[1, -1, -1], [-1, 1, -1], [-1, -1, 1]])
B = np.array([[1, -1], [-1, 1], [1, -1]])


def test_bam_weight_matrix_and_recall():
    W = bam.weight_matrix(A, B)
    np.testing.assert_array_equal(W, [[1, -1], [-3, 3], [1, -1]])
    np.testing.assert_array_equal(A[0] @ W, [3, -3])
    np.testing.assert_array_equal(bam.recall_forward(W, A[0]), B[0])   # A1 → B1 ✓
    np.testing.assert_array_equal(B[0] @ W.T, [2, -6, 2])
    # B1 → A fails: B1 = B3, so the memory cannot tell A1 from A3
    np.testing.assert_array_equal(bam.recall_backward(W, B[0]), [1, -1, 1])


# --- ART-1 (DUET 2018-19, Q3b) ---------------------------------------------------
def test_art1_reset_then_learn():
    Bm = np.array([[.2, 0, .2], [.5, .8, .2], [.5, .5, .2], [.5, .8, .2], [.1, 0, .2]])
    Tm = np.array([[1, 1, 1, 1, 1], [0, 1, 1, 1, 0], [1, 1, 1, 1, 1]], float)
    s = np.array([0, 1, 1, 1, 1], float)
    B_new, T_new, info = art1.present(s, Bm, Tm, L=2, rho=0.8)
    np.testing.assert_allclose(info["y"], [1.6, 2.1, 0.8])
    assert info["resets"][0]["J"] + 1 == 2                   # Y2 wins first, fails vigilance
    assert info["resets"][0]["ratio"] == pytest.approx(0.75)
    assert info["winner"] + 1 == 1                           # Y1 then passes (ratio 1.0)
    np.testing.assert_allclose(T_new[0], [0, 1, 1, 1, 1])
    np.testing.assert_allclose(B_new[:, 0], [0, 0.4, 0.4, 0.4, 0.4])
