# Embedding Propagation — Product Requirements Document

**Version:** 1
**Status:** Draft
**Date:** 2026-03-10

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Concept](#2-core-concept)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Feature Aggregation Models](#4-feature-aggregation-models)
5. [Loss Functions](#5-loss-functions)
6. [Negative Sampling](#6-negative-sampling)
7. [Subsampling and Regularization](#7-subsampling-and-regularization)
8. [Loss Weighting by Node Degree](#8-loss-weighting-by-node-degree)
9. [Training Loop](#9-training-loop)
10. [Warm-Start / Transfer Learning](#10-warm-start--transfer-learning)
11. [Inference Pipeline](#11-inference-pipeline)
12. [Configuration and Hyperparameters](#12-configuration-and-hyperparameters)
13. [Out of Scope (Future Work)](#13-out-of-scope-future-work)
14. [Changelog](#14-changelog)

---

## 1. Problem Statement

Many graph representation learning methods produce *node embeddings* — a fixed vector per node, learned end-to-end against the graph topology. These transductive methods cannot generalize: a model trained on one graph snapshot cannot embed new, previously unseen nodes without retraining.

Embedding Propagation (EP) addresses this by learning *feature embeddings* instead. Each node is described by a bag of discrete, named features (e.g. category tags, node type, text tokens). The model learns a vector for every feature, and constructs a node embedding on demand by aggregating its feature vectors. Because node identity is never part of the model, the approach is fully **inductive**: a node that appears after training can be embedded immediately, provided it carries known features.

EP is designed for graphs where:

- Nodes carry one or more discrete, categorical features.
- The graph topology provides a supervision signal: two nodes that are neighbors should be embedded closer together than two random nodes.
- The feature vocabulary is orders of magnitude smaller than the node count, making storage and training tractable.

---

## 2. Core Concept

### What is learned

EP learns a mapping `feature → vector`, stored as a dense embedding table indexed by feature ID. This table is the sole output of training.

### What is not learned

Node IDs are never embedded. There is no per-node parameter. Given a set of feature embeddings, any node's representation is computed deterministically (up to the stochastic subsampling choices during inference) by aggregating its feature vectors.

### Node embedding construction

Given a node `v` with features `F(v) = {f₁, f₂, …, fₙ}` and the feature embedding table `E`:

```
h(v) = Aggregate({ E[f] : f ∈ F(v) })
```

For the **averaged** aggregation model (the only model in scope for this document):

```
h(v) = (1 / |F(v)|) · Σ E[f]
```

### Positive reconstruction

The key insight of EP is that a node's embedding should be *predictable from its neighbors*. Formally, given node `v` and its neighborhood `N(v)`:

```
~h(v) = Aggregate({ h(u) : u ∈ N(v) })
       = Aggregate({ Aggregate({ E[f] : f ∈ F(u) }) : u ∈ N(v) })
```

This "reconstruction" of `v` from its neighbors is called `~h(v)`. The loss function pushes `h(v)` and `~h(v)` close together, and away from embeddings of random (negative) nodes.

For the averaged model, the reconstruction flattens to the mean of all feature embeddings across all neighbor nodes, weighted by how many times they appear — which makes it simply the mean feature embedding over the multi-set union of neighbor features.

---

## 3. Algorithm Overview

At a high level:

1. **Initialize** a feature embedding table at random (unit-normalized vectors).
2. For each training pass:
   a. Shuffle the training node set.
   b. Split into mini-batches.
   c. For each batch, run the forward pass in parallel for each node:
      - Compute `h(v)` from the node's own features.
      - Compute `~h(v)` from its neighbors' features.
      - Sample negative nodes; compute their embeddings `h(u)`.
      - Compute a scalar loss from `h(v)`, `~h(v)`, and `{h(u)}`.
      - Optionally scale the loss by the node's degree.
   d. Accumulate gradients across nodes in the batch (summing gradients for shared features).
   e. Optionally inject Gaussian noise into the gradients.
   f. Apply Adam update to the feature embeddings.
3. Periodically evaluate on a held-out validation set and report validation loss.
4. Return the trained feature embedding table.

The output of training is the feature embedding table. Node embeddings are not stored — they are computed on demand during inference.

---

## 4. Feature Aggregation Models

### 4.1 Averaged Feature Model (in scope)

The averaged model is the canonical, computationally cheap aggregator. It requires no additional parameters beyond the feature embedding table.

**Node embedding construction:** The feature embeddings for a node's features are summed and divided by the count. When features are subsampled (see §7), each sampled embedding is upweighted by the inverse sampling probability before averaging, ensuring an unbiased estimate of the full average.

**Neighbor reconstruction:** The same averaging logic is applied across all features from the sampled neighbor set. Each neighbor may also carry a scalar weight (from edge weights; see below). When weight-averaged reconstruction is enabled, each neighbor's contribution is scaled proportionally to its edge weight (converted from cumulative distribution form). When subsampling neighbors (see §7), a scale factor is applied to keep the estimate unbiased.

**Weighted neighbor sampling and averaging:** Two independent flags control neighbor handling during reconstruction:
- *Weighted sampling*: when enabled, neighbors are sampled with probability proportional to their edge weights (using weighted reservoir sampling). When disabled, uniform reservoir sampling is used.
- *Weighted averaging*: when enabled, each sampled neighbor's contribution is scaled by its edge weight. When disabled, all sampled neighbors contribute equally.

### 4.2 Attention Feature Model (excluded)

See §13.

---

## 5. Loss Functions

All losses share the same structure: an anchor node `v`, a positive target `~h(v)` (neighbor reconstruction or PPR-walk target), and a set of negative nodes `{h(uᵢ)}`. Each loss returns a non-negative scalar.

Losses that evaluate to zero (already satisfied) contribute no gradient. This sparsity is intentional: it prevents wasteful updates on already-correct examples.

### 5.1 Margin Loss

**Intent:** The FaceNet-style max-margin triplet loss. Pushes the positive closer than every negative by at least `γ` in Euclidean space.

**Distance metric:** Euclidean distance.

**Parameters:**
- `γ` (gamma, margin threshold): minimum required separation between positive and negative distances.
- `num_negatives`: number of negative samples per anchor.

**Formula:** For each negative `uᵢ`:
```
loss_i = max(0,  γ + d_E(~h(v), h(v))  −  d_E(~h(v), h(uᵢ)) )
```
Final loss = mean of positive (non-zero) per-negative losses. If all are satisfied, loss = 0.

### 5.2 StarSpace Loss

**Intent:** The StarSpace margin loss, which operates in cosine space instead of Euclidean space.

**Distance metric:** Cosine similarity (via dot product of L2-normalized vectors).

**Parameters:**
- `γ` (margin).
- `num_negatives`.

**Formula:** For each negative `uᵢ`:
```
loss_i = max(0,  γ  −  (cos(~h(v), h(v))  −  cos(h(v), h(uᵢ))) )
```
Final loss = mean of positive (non-zero) per-negative losses.

### 5.3 Contrastive Loss

**Intent:** Applies separate margin constraints to both the positive pair and each negative pair.

**Distance metric:** Cosine similarity.

**Parameters:**
- `pos_margin`: the positive pair cosine similarity must exceed this threshold (loss incurred when it falls below).
- `neg_margin`: each negative pair cosine similarity must stay below this threshold (loss incurred when it exceeds).
- `num_negatives`.

**Formula:**
```
pos_loss  = max(0,  pos_margin  −  cos(~h(v), h(v)) )
neg_loss_i = max(0,  cos(h(v), h(uᵢ))  −  neg_margin )
```
Final loss = mean of all non-zero terms (positive and negative losses pooled together).

### 5.4 Rank Loss

**Intent:** A listwise ranking loss. The positive reconstruction and all negatives form a ranked list. The loss maximizes the probability that the positive is ranked first (i.e., has the highest dot product score with `h(v)`).

**Distance metric:** Dot product (unnormalized).

**Parameters:**
- `τ` (tau): a probability threshold. If the softmax probability of the positive already exceeds `τ`, no loss is applied (similar in spirit to a margin).
- `num_negatives`.

**Formula:** Let `s_i = h(v) · h(uᵢ)` for negatives and `s_pos = h(v) · ~h(v)`.
Compute softmax over `[s₁, …, s_n, s_pos]`. Let `p` = softmax probability of the positive.
```
loss = -log(p)   if p < τ
loss = 0          otherwise
```

### 5.5 RankSpace Loss

**Intent:** A combined loss that applies both StarSpace and Rank Loss simultaneously. This has been found to outperform either alone for search-style tasks.

**Parameters:** Same as StarSpace and RankLoss (`γ`, `num_negatives`).

**Formula:**
```
loss = StarSpace(γ, num_negatives) + RankLoss(γ, num_negatives)
```

### 5.6 PPR Loss (Personalized PageRank Loss)

**Intent:** A variant of Margin Loss where the *positive target* is constructed differently. Instead of using the direct neighborhood reconstruction, a short random walk from the anchor node generates multiple target nodes. These are then averaged together (via the model's `construct_from_multiple_nodes`) to form the positive target `~h(v)`. This captures a richer, multi-hop notion of neighborhood similarity.

**Distance metric:** Euclidean (same as Margin Loss).

**Parameters:**
- `γ` (margin).
- `num_negatives` (also controls the number of walk-sampled positive targets).
- `restart_p`: probability of stopping the random walk at each step (i.e., the return probability in PPR).

**Walk procedure:** For each of the `num_negatives` positive samples: perform a random walk from the anchor, stopping at each step with probability `restart_p` (minimum 2 steps taken before the restart check applies, to ensure at least one hop). Walks that return to the anchor are retried. If no valid walk node is found, the anchor itself is used as fallback. The resulting set of nodes is aggregated into a single positive representation.

**Loss computation:** Identical to Margin Loss once `~h(v)` is formed.

---

## 6. Negative Sampling

### 6.1 Strategy

Negatives are sampled from the *training node set only* (never from the validation hold-out). This is important: training and validation are partitioned at the node level before any sampling begins.

### 6.2 Random Negatives

The majority of negatives are drawn uniformly at random from the training node index. This is the baseline and the only source of negatives when `hard_negatives = 0`.

### 6.3 Hard Negatives via Random Walk

When `hard_negatives > 0`, the sampler first attempts to find hard negatives by performing random walks from the anchor node, then fills the remaining slots with uniform random negatives.

**Walk procedure for hard negatives:**
- Walk at most 10 steps.
- At each step, uniformly sample one of the current node's neighbors.
- After at least 2 steps, stop with probability `p = 0.25` if the current node is not the anchor.
- A candidate is accepted as a hard negative only if it is not the anchor and is *not* a direct neighbor of the anchor (to avoid selecting weak positives).

Up to `2 × num_hard_negatives` walk attempts are made before falling back to random sampling. Duplicates are rejected.

**Design note:** Hard negatives via random walk have proven empirically difficult to use without hurting validation loss (neighbors-of-neighbors are often weak positives). They are included as a configurable option but default to zero.

---

## 7. Subsampling and Regularization

### 7.1 Feature Dropout (`max_features`)

During both training and positive construction, when `max_features` is set, not all features of a node are used. Features are subsampled at each forward pass.

The `max_features` parameter is expressed as either:
- An integer `k`: exactly `k` features are randomly selected (reservoir sampling without replacement).
- A probability `p ∈ (0, 1)`: the number of features to keep is drawn from Binomial(n, p); at least one feature is always kept.
- Unset (All): all features are used.

**Unbiased reweighting:** When fewer than all features are used, the contribution of each selected feature is scaled by `n / k` (for fixed-k) or `1 / p` (for probability) so the expected value of the averaged embedding equals the full average. This ensures the gradient signal is unbiased.

### 7.2 Neighbor Subsampling (`max_nodes`)

When `max_nodes` is set, the neighbor set used for the positive reconstruction `~h(v)` is subsampled. The same `Sample` type governs this: integer, probability, or All.

- **Uniform reservoir sampling** (default): each neighbor is equally likely to be selected.
- **Weighted reservoir sampling** (when `weighted_neighbor_sampling = true`): selection probability is proportional to edge weight, using the A-Res algorithm (key = `uniform_random ^ (1/weight)`).

The same scale factor is applied to neighbor contributions to keep the reconstruction unbiased.

### 7.3 Gradient Noise Regularization

When `noise > 0`, zero-mean Gaussian noise is added to the accumulated gradient vector of each feature before the optimizer step. The noise magnitude follows an **exponential decay schedule** that starts at the configured `noise` level and decays toward zero across the total number of training steps. This acts as a form of stochastic regularization similar to gradient perturbation, helping to prevent overfitting when validation loss starts to diverge from training loss.

---

## 8. Loss Weighting by Node Degree

High-degree nodes (hubs) contribute many training examples and may dominate gradient updates. EP provides optional loss scaling to compensate.

Three modes:

| Mode | Formula | Intent |
|---|---|---|
| `None` | `loss` | No reweighting (default). |
| `DegreeLog` | `loss / ln(1 + degree(v))` | Downweight hubs logarithmically. |
| `DegreeExponential(w)` | `loss / degree(v)^w` | Downweight hubs by a configurable power `w`. |

The scaling is applied per-node before gradient accumulation, so hub nodes generate weaker gradient signals relative to low-degree nodes.

---

## 9. Training Loop

### 9.1 Data Split

Before training begins, all nodes are randomly shuffled and partitioned into a **training set** and a **validation set**. The split is controlled by `valid_pct` (fraction of nodes held out). The validation set is never used to generate gradients; it is only used for reporting.

### 9.2 Mini-Batch Parallelism

Training nodes are shuffled at the start of each pass and chunked into mini-batches of size `batch_size`. Each batch is processed in parallel: within a batch, each node's forward pass and gradient computation run concurrently.

Because feature embeddings are shared across nodes in a batch, gradients from multiple nodes that touch the same feature are *accumulated* (summed) before the optimizer step. This is effectively a synchronous mini-batch SGD step over the feature embedding table.

### 9.3 Optimizer: Adam

Feature embeddings are updated using **Adam** with fixed hyperparameters:
- `β₁ = 0.9` (first moment decay)
- `β₂ = 0.999` (second moment decay)
- `ε = 1e-8` (numerical stability)

The optimizer maintains per-feature first-moment (momentum) and second-moment (variance) buffers. Updates follow the standard bias-corrected Adam formula:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g
v_t = β₂ · v_{t-1} + (1 - β₂) · g²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ ← θ - α · m̂_t / (√v̂_t + ε)
```

The time step `t` is the training pass number (not the global step), which means the bias correction terms change only once per pass, not per batch. This is a deliberate simplification.

Updates are applied with Hogwild-style lock-free parallelism: concurrent writes to different feature slots do not synchronize. This is correct in expectation when batch members rarely share features.

### 9.4 Learning Rate Schedule: Cosine Decay with Warmup

The learning rate follows a **cosine decay with linear warmup** schedule:

- **Warmup phase** (first 20% of total steps): LR rises linearly from `α / 100` to `α`.
- **Decay phase** (remaining 80%): LR decays from `α` down toward `α / 100` following a cosine curve.

```
# Warmup:
lr(step) = α_min + (step / warmup_steps) · (α - α_min)

# Decay:
lr(step) = α_min + 0.5 · (α - α_min) · (1 + cos(π · step / max_steps))
```

where `α_min = α / 100`.

### 9.5 Validation

At the end of each pass, if a validation set exists, the forward pass is run (without gradient tracking) on all validation nodes using a fixed random seed for reproducibility. The mean loss over all validation nodes is reported.

Validation batches use the same sampler type as training (random walk hard sampler initialized on the validation node subset). The validation error is not used to trigger any automatic action (e.g., early stopping); it is informational only.

### 9.6 Numerical Stability

- Losses evaluating to exactly `0.0` are skipped and do not contribute gradients.
- Losses evaluating to NaN (which can occur due to underflows in softmax or degenerate distance computations) are silently dropped.
- Gradients containing NaN or infinite values are not applied.

---

## 10. Warm-Start / Transfer Learning

EP supports incremental training via an optional `feature_embeddings` input. When provided, training begins from this pre-trained embedding table rather than from random initialization. The optimizer's moment buffers are always initialized fresh regardless of whether warm-start embeddings are provided.

Use cases:
- **Incremental updates:** When new nodes or features are added to the graph, the existing feature embeddings can be passed as the warm start. New features (unknown to the old table) would need to be initialized separately outside the module (or handled by expanding the embedding table before passing it in).
- **Fine-tuning:** Training on a general graph first, then specializing to a downstream graph.
- **Hyperparameter search:** Continuing from a checkpoint rather than training from scratch.

The warm-start path does not alter any other aspect of training. The `valid_pct` split, learning rate schedule, and optimizer state are all re-initialized.

---

## 11. Inference Pipeline

After training, the feature embedding table is used to generate node embeddings on demand. This is the inference step and is separate from training.

### 11.1 Aggregators

Three aggregator strategies convert a node's feature list and the learned embedding table into a node embedding vector:

#### 11.1.1 Average Aggregator

The simplest aggregator. Takes the unweighted arithmetic mean of the feature embeddings for a node's feature set.

```
node_emb = (1/|F(v)|) · Σ_{f ∈ F(v)} E[f]
```

No additional parameters. Equivalent to the training-time node embedding construction (without subsampling).

#### 11.1.2 Weighted Aggregator (SIF-style)

Applies a smooth IDF-like weight to each feature before averaging:

```
w(f) = α / (α + P(f))
node_emb = Σ_{f ∈ F(v)} w(f) · E[f]  /  Σ_{f ∈ F(v)} w(f)
```

where `P(f)` is the unigram probability of feature `f` (i.e., its frequency across all nodes divided by the total feature count), and `α` is a smoothing parameter. Rarer features receive higher weight. This is inspired by the Smooth Inverse Frequency (SIF) sentence embedding method.

The unigram probabilities are computed from the full feature store before inference.

#### 11.1.3 Attention Aggregator (excluded)

See §13.

### 11.2 Node Embedding Construction

The inference pipeline is stateless: each node is embedded independently by passing its feature list through the chosen aggregator. Inference can be parallelized trivially across nodes.

---

## 12. Configuration and Hyperparameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | `0.9` | Peak learning rate for the cosine-decay schedule. |
| `loss` | enum | `MarginLoss(1.0, 1)` | Loss function variant and its parameters. See §5. |
| `batch_size` | int | `50` | Mini-batch size (number of nodes per gradient step). |
| `dims` (d_model) | int | `100` | Dimensionality of node embeddings (and feature embeddings, for the averaged model). |
| `passes` | int | `100` | Number of full passes over the training nodes. |
| `seed` | int | constant | Random seed for reproducibility (shuffling, sampling, noise). |
| `max_nodes` | int, float, or None | All | Max neighbors to use for reconstruction. Int = fixed count; float ∈ (0,1) = Binomial probability; None = all. |
| `weighted_neighbor_sampling` | bool | `false` | Sample neighbors proportionally to edge weight during reconstruction. |
| `weighted_neighbor_averaging` | bool | `false` | Weight neighbor contributions by edge weight during reconstruction. |
| `max_features` | int, float, or None | All | Max features per node per forward pass. Same encoding as `max_nodes`. |
| `valid_pct` | float | `0.1` | Fraction of nodes held out for validation. |
| `hard_negatives` | int | `0` | Number of hard negatives generated via random walk per anchor. |
| `loss_weighting` | enum | `None` | Per-node loss scaling: `None`, `DegreeLog`, or `DegreeExponential(w)`. |
| `noise` | float | `0.0` | Initial standard deviation of Gaussian gradient noise. Decays exponentially to ~0 over training. |
| `indicator` | bool | `true` | Whether to display a progress bar during training. |

### Loss-specific parameters

| Loss | Parameters |
|---|---|
| `MarginLoss` | `gamma: float`, `num_negatives: int` |
| `StarSpace` | `gamma: float`, `num_negatives: int` |
| `Contrastive` | `pos_margin: float`, `neg_margin: float`, `num_negatives: int` |
| `RankLoss` | `tau: float`, `num_negatives: int` |
| `RankSpace` | `gamma: float`, `num_negatives: int` |
| `PPR` | `gamma: float`, `num_negatives: int`, `restart_p: float` |

---

## 13. Out of Scope (Future Work)

### Attention-Based Feature Aggregation

An attention model exists that replaces simple averaging with scaled multi-head self-attention over a node's features. This supports full attention (all features attend to each other), random attention (attend over a subsampled feature subset), and sliding-window attention (useful for ordered, text-like feature sequences). The output dimensionality is `num_heads × (2 × d_k + d_model)`.

Attention is excluded from this document because it is computationally expensive on CPU (two orders of magnitude slower than averaging), requires significant additional hyperparameters (number of heads, attention key dimension, window size), and is expected to be addressed separately in a future iteration of this PRD.

---

## 14. Changelog

| Version | Date | Notes |
|---|---|---|
| v1 | 2026-03-10 | Initial PRD. Covers averaged model, all 6 loss functions, negative sampling, subsampling, degree weighting, training loop, warm-start, and inference. Attention excluded pending separate treatment. |
