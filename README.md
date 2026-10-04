# Neural Networks — MSc Course Notes and From-Scratch Implementations

My worked solutions for **CSE 6106: Neural Network** (M.Sc. in CSE, DUET), typed up from my handwritten notes and backed by from-scratch NumPy implementations that **reproduce every hand calculation as an automated test**.

| | |
|---|---|
| **Course** | CSE 6106: Neural Network (3 credits) — M.Sc. in CSE, Dhaka University of Engineering & Technology (DUET) |
| **Semester** | 1st semester, 2025–2026 |
| **Course teacher** | Prof. Dr. Momotaz Begum, Head of the Department of Computer Science and Engineering, DUET |
| **Grade** | [TODO: grade] |

## What's inside

| # | Topic | Worked problem | Code | Verified |
|---|---|---|---|:---:|
| 01 | [Biological neural networks](notes/01-biological-neural-networks.md) | Working principles of the BNN | — | — |
| 02 | [Perceptron learning](notes/02-perceptron.md) | One epoch of the perceptron rule; convergence | [`perceptron.py`](src/nncourse/perceptron.py) | ✅ |
| 03 | [Probability density estimation](notes/03-probability-density-estimation.md) | PMF vs. PDF; parametric, non-parametric, semi-parametric | — | — |
| 04 | [Clustering and K-means](notes/04-clustering-and-k-means.md) | 8 points, 3 clusters (Manhattan); 4 medicines, 2 clusters | [`kmeans.py`](src/nncourse/kmeans.py) | ✅ |
| 05 | [Backpropagation](notes/05-backpropagation.md) | Full forward/backward pass and update of a 2-2-1 sigmoid MLP | [`mlp.py`](src/nncourse/mlp.py) | ✅ |
| 06 | [Kohonen self-organizing map](notes/06-kohonen-self-organizing-map.md) | 4 vectors → 2 clusters, one epoch + recall | [`ksom.py`](src/nncourse/ksom.py) | ✅ |
| 07 | [Learning vector quantization](notes/07-learning-vector-quantization.md) | LVQ1, one epoch with push/pull updates | [`lvq.py`](src/nncourse/lvq.py) | ✅ |
| 08 | [Bidirectional associative memory](notes/08-bidirectional-associative-memory.md) | Store 3 pairs; forward and backward recall; crosstalk analysis | [`bam.py`](src/nncourse/bam.py) | ✅ |
| 09 | [Adaptive resonance theory (ART-1)](notes/09-adaptive-resonance-theory.md) | Vigilance reset, resonance and learning | [`art1.py`](src/nncourse/art1.py) | ✅ |

Each note gives the question, a step-by-step solution, the intuition behind the algorithm, and a **"Corrections to the handwritten version"** section listing every slip found when the hand calculations were checked in code.

## Beyond the exam answers

Checking the hand calculations in code led to a few results that go past the standard answer:

- **K-means:** the exam asks for the centers after two iterations, but the algorithm has not converged there. A4 changes cluster in iteration 3 ([04](notes/04-clustering-and-k-means.md)).
- **Backpropagation:** the hidden unit u3 gets a delta about 28× smaller than u4 because it is saturated (y ≈ 0.98), a small-scale view of the sigmoid's vanishing gradient ([05](notes/05-backpropagation.md)).
- **BAM:** backward recall B1 → A1 fails. Expanding `B1 Wᵀ = Σ_k (B1·B_k) A_k` shows exactly why: crosstalk from the repeated pattern B1 = B3 and the opposite pattern B2 = −B1, ending in a stable but spurious state ([08](notes/08-bidirectional-associative-memory.md)).
- **ART-1:** adds the bottom-up weight update (step 13) that was missing from my handwritten solution, and explains why ART templates can only shrink, which is what keeps learned categories stable ([09](notes/09-adaptive-resonance-theory.md)).

## Run the verification

```bash
git clone https://github.com/shahidul-shabuz/MSc-Neural-Networks.git
cd MSc-Neural-Networks
pip install -r requirements.txt
pytest -q
```

Expected output: `11 passed`. The implementations depend only on NumPy and are written for readability, with each update rule matching the formula in the corresponding note.

## Repository structure

```
MSc-Neural-Networks/
├── notes/                  # Typed solutions, one file per topic
├── src/nncourse/           # From-scratch NumPy implementations
├── tests/                  # Each test reproduces a worked example from the notes
├── handwritten-notes/      # Scan of my original handwritten notes (personal details redacted)
├── requirements.txt
└── pyproject.toml
```

## Past exam coverage (DUET 2018–19, CSE 6106)

| Question | Topic | Covered in |
|---|---|---|
| 1(a) | Working principles of the BNN | [01](notes/01-biological-neural-networks.md) |
| 1(a) | ANN vs. BNN comparison | — |
| 1(b) | Three-layer linear network, solve for inputs | — |
| 2(a) | Styles of probability density estimation | [03](notes/03-probability-density-estimation.md) |
| 2(a) | Linear discriminant functions | — |
| 2(b) | KSOM clustering | [06](notes/06-kohonen-self-organizing-map.md) (same method, textbook data) |
| 3(a) | Operating principle of ART | [09](notes/09-adaptive-resonance-theory.md) |
| 3(b) | ART-1 worked problem | [09](notes/09-adaptive-resonance-theory.md) |
| 4(a) | K-means pros/cons; elbow technique | [04](notes/04-clustering-and-k-means.md) |
| 4(b) | K-means on medicines | [04](notes/04-clustering-and-k-means.md) |
| 5(a), 5(b) | Ensembles; learning paradigms | — |
| 6(a) | LVQ; types of clustering | [07](notes/07-learning-vector-quantization.md), [04](notes/04-clustering-and-k-means.md) |
| 6(b) | SVM, curse of dimensionality, feature extraction | — |

## Paper presentation

As part of the course, I presented:

> F. Ghazouani, P. Vera, S. Ruan, "Efficient brain tumor segmentation using Swin transformer and enhanced local self-attention," *International Journal of Computer Assisted Radiology and Surgery*. DOI: [10.1007/s11548-023-03024-8](https://doi.org/10.1007/s11548-023-03024-8)

The paper combines Swin Transformer blocks for global context, enhanced local self-attention (ELSA) for fine detail, and spatial squeeze-and-excitation in a CNN decoder, evaluated on BraTS 2021. It is directly related to my M.Sc. research on attention-based brain tumor segmentation in multi-modal MRI.

## A note on preparation

The worked solutions are my own, from the handwritten notes in [`handwritten-notes/`](handwritten-notes/). The typed versions, the Python implementations and tests, and the extensions noted above were prepared with the help of an AI assistant (Claude) and reviewed by me. Points added during typing are marked in the notes.

Lecture slides, course readings and other students' materials are not included, as they belong to their authors.

## Author

**Md Shahidul Islam Shabuz** — Assistant Professor, Dept. of CSE, RSTU; M.Sc. candidate, DUET
[Website](https://shahidul-shabuz.github.io/) · [Google Scholar](https://scholar.google.com/citations?user=VDtX90sAAAAJ&hl=en) · [ORCID](https://orcid.org/0009-0008-8271-888X)

## License

Code: [MIT](LICENSE). Notes: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
