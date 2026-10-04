# 03 — Probability Distributions and Density Estimation

> Handwritten: [pp. 2–4](../handwritten-notes/neural-network-handwritten-notes.pdf) (Lecture 04) · Exam: DUET 2018–19, Q2(a)

## Probability distribution

A probability distribution describes how likely each possible outcome of a random variable is.

### Discrete data: probability mass function (PMF)

Used when outcomes are countable. Example: the number of heads `X` in two coin flips can only be 0, 1 or 2, so each probability can be listed in a table.

```text
P(X = x) = p(x),     Σ p(x) = 1
```

| x (heads) | 0 | 1 | 2 |
|---|---:|---:|---:|
| p(x) | 1/4 | 1/2 | 1/4 |

### Continuous data: probability density function (PDF)

Used when the variable can take any value in a range (height, time, temperature). A PDF `f(x)` satisfies:

1. `f(x) ≥ 0` for all x,
2. the total area under the curve is 1: `∫_{−∞}^{∞} f(x) dx = 1`.

The probability of any *single* point is 0. Instead, we compute the probability of falling in an interval as the area under the curve:

```text
P(a ≤ X ≤ b) = ∫_a^b f(x) dx
```

Common PDFs: normal, standard normal, Student's t, chi-square and continuous uniform distributions.

## Q: Describe the styles of probability density estimation

Density estimation means estimating the unknown `f(x)` from a data sample. There are three styles:

| Style | Idea | Example | Strength | Weakness |
|---|---|---|---|---|
| **Parametric** | Assume a distribution family and estimate its parameters | Fit a Gaussian by **Maximum Likelihood Estimation** (μ = sample mean, σ² = sample variance) | Few parameters; efficient; works with little data | Poor if the assumed form does not fit the data |
| **Non-parametric** | No assumption about the form; let the data define the shape | **Kernel density estimation** (Parzen windows), histograms, k-nearest neighbours | Fits arbitrary distributions | Number of "parameters" grows with the dataset; slow on new data; needs much data |
| **Semi-parametric** | A flexible combination of parametric components | **Mixture models**, e.g., a Gaussian mixture fitted with EM | Best of both: flexible but compact | Choosing the number of components; local optima |

The same trade-off appears again in clustering ([04](04-clustering-and-k-means.md)): distribution-based clustering is semi-parametric density estimation with a Gaussian mixture.
