# Reproducing EP-B

`reproduce.py` reruns the node classification experiments of García-Durán and Niepert, "Learning
Graph Representations with Embedding Propagation" (NIPS 2017), with cloverleaf's EP set up as the
paper's EP-B:

- Glorot uniform initialisation, 128 dims per label type
- margin loss with one negative
- Adam at a constant learning rate of 0.001
- batches of 64 nodes, 200 epochs

See the script's docstring for the evaluation protocol and where to download the data.

```
python reproduce.py --data DIR --cache CACHE_DIR --out results.json
python reproduce.py --data DIR --datasets blogcatalog --max-nodes 5   # sample 5 neighbours
```

## Results

All results are transductive. Each is the mean ± standard deviation over 10 random splits, with
EP trained once per dataset and γ. The margin γ is the one the paper reports choosing; our own
cross validation picks γ = 1 for PPI and POS, which scores about the same.

**Cora, Citeseer, Pubmed** (accuracy, all neighbours; γ and the regulariser picked on validation):

| | cloverleaf | paper |
|---|---|---|
| Cora | 79.47 ± 0.88 | 78.05 ± 1.49 |
| Citeseer | 71.21 ± 1.52 | 71.01 ± 1.35 |
| Pubmed | 79.65 ± 1.77 | 79.56 ± 2.10 |

**BlogCatalog, PPI, POS** (micro / macro F1; Tr is the fraction of labelled nodes):

| | Tr | all neighbours | 5 sampled neighbours | paper |
|---|---|---|---|---|
| BlogCatalog (γ=1) | 0.1 | 29.04 / 13.81 | 33.94 / 19.16 | 35.05 / 19.08 |
| | 0.5 | 33.26 / 16.84 | 38.50 / 23.77 | 39.44 / 25.11 |
| | 0.9 | 34.11 / 17.37 | 39.62 / 24.98 | 40.41 / 25.97 |
| PPI (γ=5) | 0.1 | 18.08 / 13.92 | 18.10 / 14.04 | 17.82 / 13.80 |
| | 0.5 | 22.42 / 18.21 | 22.69 / 18.84 | 23.30 / 18.96 |
| | 0.9 | 23.99 / 19.29 | 24.19 / 20.33 | 24.74 / 20.36 |
| POS (γ=10) | 0.1 | 47.58 / 9.18 | 46.53 / 8.57 | 46.97 / 8.85 |
| | 0.5 | 50.21 / 10.14 | 49.21 / 10.38 | 49.52 / 10.45 |
| | 0.9 | 51.64 / 10.44 | 50.62 / 10.77 | 50.05 / 12.17 |

The majority-class baseline matches the paper's, which checks the evaluation protocol. On
BlogCatalog, PPI and POS at Tr = 0.1, micro / macro F1 is:

| | ours | paper |
|---|---|---|
| BlogCatalog | 16.19 / 2.45 | 16.51 / 2.51 |
| PPI | 6.71 / 1.69 | 6.15 / 1.58 |
| POS | 40.44 / 3.26 | 40.40 / 3.38 |

## Findings

- **Five of the six datasets reproduce with the paper's settings.** Cora, Citeseer, Pubmed, PPI and
  POS land within about one standard deviation of the paper. The one exception is POS macro F1 at
  Tr = 0.9, which comes out at 10.4–10.8 against the paper's 12.2.
- **BlogCatalog only reproduces with neighbour sampling.** Using every neighbour leaves it 5–8 points
  short. BlogCatalog is dense: the average degree is 65 and the largest is 3,992. Sampling 5
  neighbours per reconstruction closes all but about 1 point. On one split, 1, 2, 5, 10, 20 and 50
  sampled neighbours give micro F1 of 36.2, 36.7, 38.0, 37.1, 35.9 and 34.0, against 33.1 with every
  neighbour. The paper samples neighbours (its κ) in Figure 4, but doesn't say what it used for
  Table 3. Sampling makes no real difference on PPI and POS.
- **What didn't cause the BlogCatalog gap:**
  - Hogwild parallelism: a single-threaded run scores the same.
  - The classifier's regulariser: C beyond the paper's grid adds at most 0.3 points.
  - Underfitting: EP's final training loss is low.
- **A pitfall in the multi-label evaluation.** One-vs-rest gives a label missing from a training
  split a constant decision score of 0. That outranks every real label with a negative score, so
  the script ranks labels with `predict_proba` instead. Scoring with `decision_function` cost POS
  20 points of micro F1.
