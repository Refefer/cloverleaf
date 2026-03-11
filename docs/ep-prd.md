# Embedding Propagation — Product Requirements Document

**Version:** 3
**Status:** Draft
**Date:** 2026-03-10

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Core Concept](#2-core-concept)
3. [Algorithm Overview](#3-algorithm-overview)
4. [Model — Node Embedding Construction h(v)](#4-model--node-embedding-construction-hv)
5. [PositiveSampler — Positive Target Construction ~h(v)](#5-positivesampler--positive-target-construction-hv)
6. [ReconstructionAggregator — Combining Neighbor Features](#6-reconstructionaggregator--combining-neighbor-features)
7. [NegativeSampler](#7-negativesampler)
8. [Loss Functions](#8-loss-functions)
9. [FeatureDropout](#9-featuredropout)
10. [Loss Weighting by Node Degree](#10-loss-weighting-by-node-degree)
11. [Training Loop](#11-training-loop)
12. [Warm-Start / Transfer Learning](#12-warm-start--transfer-learning)
13. [Inference Pipeline](#13-inference-pipeline)
14. [Configuration and Hyperparameters](#14-configuration-and-hyperparameters)
15. [Out of Scope (Future Work)](#15-out-of-scope-future-work)
16. [Changelog](#16-changelog)

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

EP learns a mapping `feature → vector`, stored as a dense embedding table indexed by feature ID. The **Model** (§4) determines how these feature vectors are combined to produce a node embedding `h(v)`. For parameterized models (§15), the model's own weights are learned alongside the feature embedding table and can be serialized/deserialized independently.

### What is not learned

Node IDs are never embedded. There is no per-node parameter. Given a trained feature embedding table (and model weights, if applicable), any node's representation is computed on demand from its features.

### Training signal

The core supervision signal is the *positive reconstruction* `~h(v)`: an estimate of `h(v)` constructed from the node's neighborhood, using the **PositiveSampler** (§5) and **ReconstructionAggregator** (§6). The loss function pushes `h(v)` and `~h(v)` together and away from **negative** node embeddings drawn by the **NegativeSampler** (§7).

---

## 3. Algorithm Overview

1. **Initialize** the feature embedding table at random (unit-normalized vectors). Initialize any Model parameters.
2. For each training pass:
   a. Shuffle the training node set.
   b. Split into mini-batches.
   c. For each batch, run the forward pass in parallel for each node `v`:
      - Compute `h(v)` using the **Model** (§4).
      - Compute `~h(v)` using the **PositiveSampler** (§5) and **ReconstructionAggregator** (§6).
      - Sample `num_negatives` nodes using the **NegativeSampler** (§7); compute their embeddings.
      - Compute a scalar loss (§8) from `h(v)`, `~h(v)`, and the negative embeddings.
      - Optionally scale the loss by node degree (§10).
   d. Accumulate gradients across the batch.
   e. Optionally inject gradient noise (§9.3).
   f. Apply the **Optimizer** (§11.3) update, scaled by the **LRScheduler** (§11.4).
3. Periodically evaluate on a held-out validation set.
4. Return the feature embedding table and any serializable Model parameters.

---

## 4. Model — Node Embedding Construction h(v)

The **Model** defines how a node's feature embeddings are combined into a single node embedding vector `h(v)`. It is parameterized by a **FeatureDropout** policy (§9) that controls feature subsampling during training.

The Model is a first-class serializable component: its parameters (beyond the feature embedding table) can be saved and restored independently, enabling warm-start and transfer learning for parameterized models (§12).

### 4.1 Model Enum

```
Model::Averaged
Model::Parameterized   [future — see §15]
```

### 4.2 Averaged

The canonical, parameter-free model. Feature embeddings for the selected features are summed and divided by the (possibly scaled) count. No learnable weights beyond the feature embedding table.

```
h(v) = (1 / |F̃(v)|) · Σ_{f ∈ F̃(v)} scale(f) · E[f]
```

where `F̃(v)` is the (possibly subsampled) feature set and `scale(f)` is the unbiased reweighting factor from the FeatureDropout policy (§9).

When serialized, only the feature embedding table needs to be stored.

### 4.3 Parameterized (Future)

See §15.

---

## 5. PositiveSampler — Positive Target Construction ~h(v)

The **PositiveSampler** defines how the positive target `~h(v)` is constructed. It is responsible for selecting the set of neighbor nodes (or walk-reachable nodes) that are passed to the **ReconstructionAggregator** (§6).

Decoupling positive sampling from the loss function means any loss (§8) can be paired with any positive construction strategy.

### 5.1 PositiveSampler Enum

```
PositiveSampler::LocalNeighborhood { ... }
PositiveSampler::RandomWalk { ... }
```

### 5.2 LocalNeighborhood

Selects from the direct one-hop neighbors of the anchor node. This is the standard EP strategy.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_nodes` | `FeatureDropout` | All | How many neighbors to include. Uses the FeatureDropout enum (§9) for fixed-count or probabilistic subsampling. |
| `weighted_sampling` | bool | `false` | When subsampling neighbors, select proportionally to edge weight (A-Res weighted reservoir sampling) rather than uniformly. |
| `weighted_averaging` | bool | `false` | Weight each selected neighbor's contribution by its edge weight when aggregating. When false, all selected neighbors contribute equally. |

**Unbiased reweighting:** When fewer than all neighbors are used, each selected neighbor's contribution is upscaled by the inverse sampling factor so the expected value of `~h(v)` equals the full-neighborhood aggregate.

### 5.3 RandomWalk

Constructs `~h(v)` from nodes reached by short random walks from the anchor, rather than direct neighbors. This captures a multi-hop neighborhood signal and is more robust to noise in immediate connections.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `num_walk_samples` | int | — | Number of walk-sampled nodes to aggregate into `~h(v)`. |
| `restart_p` | float | — | Probability of halting the walk at each step (PPR-style restart). Higher values bias toward closer neighborhoods. |
| `max_steps` | int | `10` | Hard cap on walk length to prevent unbounded walks in dense graphs. |

**Walk procedure:** For each sample, perform a random walk from the anchor. At least 2 steps are taken before the restart check applies (ensuring at least one hop). A node is accepted only if it is not the anchor. If no valid node is found after the walk, the anchor's direct neighbors are used as fallback. The resulting node set is passed to the ReconstructionAggregator.

**Note:** This strategy subsumes the `PPR` loss variant from v1/v2. `PPR` loss is now expressed as `Loss::MarginLoss` + `PositiveSampler::RandomWalk` and should not be used as a separate loss type.

---

## 6. ReconstructionAggregator — Combining Neighbor Features

The **ReconstructionAggregator** defines how the set of nodes selected by the PositiveSampler is combined into a single `~h(v)` vector. It is applied after neighbor selection and operates on the feature embeddings of the selected nodes.

### 6.1 ReconstructionAggregator Enum

```
ReconstructionAggregator::Averaged
ReconstructionAggregator::Weighted { alpha: float }
```

### 6.2 Averaged

All features across all selected neighbor nodes are pooled and averaged uniformly (respecting the FeatureDropout scaling from §9 and the per-neighbor weights from the PositiveSampler).

```
~h(v) = (1 / Z) · Σ_{u ∈ neighbors} w(u) · Σ_{f ∈ F̃(u)} scale(f) · E[f]
```

where `w(u)` is the neighbor weight (from PositiveSampler), `scale(f)` is the feature dropout reweighting, and `Z` is the normalizing sum of weights.

### 6.3 Weighted (SIF-style)

Features are weighted by their inverse document frequency before aggregation. Rarer features across the full feature store receive higher weight, reducing the influence of high-frequency features that carry little discriminative signal.

```
w(f) = α / (α + P(f))
~h(v) = Σ_f w(f) · E[f]  /  Σ_f w(f)
```

where `P(f)` is the unigram probability of feature `f` computed from the full feature store, and `α` is a smoothing hyperparameter.

---

## 7. NegativeSampler

The **NegativeSampler** defines how negative nodes are drawn per anchor. Negatives are always sampled from the **training node set only** (the validation hold-out is never used as a negative source).

The count of negatives requested per anchor is the EP-level `num_negatives` parameter (§14). The NegativeSampler does not control this count — it only controls the selection strategy.

### 7.1 NegativeSampler Enum

```
NegativeSampler::UniformWithHardNegatives { hard_negatives: usize }
```

_(Single variant for now; additional strategies are future work.)_

### 7.2 UniformWithHardNegatives

Fills the requested `num_negatives` slots using a two-phase strategy:

1. **Hard negatives (first `hard_negatives` slots):** Random-walk from the anchor, accepting only nodes that are not the anchor and not a direct neighbor of the anchor. Up to `2 × hard_negatives` walk attempts are made. Restart probability is fixed at `0.25`; max walk steps is `10`.
2. **Random negatives (remaining slots):** Drawn uniformly from the full training node index.

When `hard_negatives = 0`, all slots are filled with uniform random negatives.

**Design note:** Hard negatives via random walk have proven empirically difficult to improve results — neighbors-of-neighbors are often weak positives. They are included as a configurable option but default to `0`.

---

## 8. Loss Functions

All losses share a uniform interface: `compute(anchor: h(v), positive: ~h(v), negatives: [h(u₁), …, h(uₙ)]) → scalar`. Loss functions have no knowledge of how many negatives to request (that is `num_negatives` at the EP level) and no knowledge of how the positive was constructed (that is the PositiveSampler).

Losses that evaluate to zero (already satisfied) contribute no gradient.

### 8.1 Margin Loss

**Distance metric:** Euclidean.
**Parameters:** `gamma: float`

For each negative `uᵢ`:
```
loss_i = max(0,  γ + d_E(~h(v), h(v))  −  d_E(~h(v), h(uᵢ)) )
```
Final loss = mean of non-zero per-negative losses.

### 8.2 StarSpace Loss

**Distance metric:** Cosine similarity (dot product of L2-normalized vectors).
**Parameters:** `gamma: float`

For each negative `uᵢ`:
```
loss_i = max(0,  γ  −  (cos(~h(v), h(v))  −  cos(h(v), h(uᵢ))) )
```
Final loss = mean of non-zero per-negative losses.

### 8.3 Contrastive Loss

**Distance metric:** Cosine similarity.
**Parameters:** `pos_margin: float`, `neg_margin: float`

```
pos_loss   = max(0,  pos_margin  −  cos(~h(v), h(v)) )
neg_loss_i = max(0,  cos(h(v), h(uᵢ))  −  neg_margin )
```
Final loss = mean of all non-zero terms pooled together.

### 8.4 Rank Loss

**Distance metric:** Dot product.
**Parameters:** `tau: float`

Compute softmax over `[h(v)·h(u₁), …, h(v)·h(uₙ), h(v)·~h(v)]`. Let `p` = softmax probability of the positive.
```
loss = -log(p)   if p < τ
loss = 0          otherwise
```

### 8.5 RankSpace Loss

**Parameters:** `gamma: float`

```
loss = StarSpace(γ)(h(v), ~h(v), negatives) + RankLoss(γ)(h(v), ~h(v), negatives)
```

### 8.6 PPR Loss — Deprecated

The PPR loss variant from v1/v2 (which embedded positive-construction logic inside a loss function) is superseded by `Loss::MarginLoss` + `PositiveSampler::RandomWalk`. It should not be used in new implementations.

---

## 9. FeatureDropout

**FeatureDropout** controls how many features of a node are used in each forward pass, for both node embedding construction (§4) and positive target construction (§5). It applies uniformly to both `h(v)` and `~h(v)` construction.

### 9.1 FeatureDropout Enum

```
FeatureDropout::All
FeatureDropout::Fixed { k: usize }
FeatureDropout::Probability { p: float }
```

### 9.2 All

Every feature of the node is used. No subsampling. Scale factor = 1.

### 9.3 Fixed

Exactly `k` features are drawn without replacement (reservoir sampling). If the node has fewer than `k` features, all are used.

Unbiased scale factor: `n / k`, where `n` is the total feature count. Each selected feature's embedding is multiplied by this factor before averaging, so the expected result equals the full average.

### 9.4 Probability

The number of features to include is drawn from `Binomial(n, p)`. At least one feature is always retained.

Unbiased scale factor: `1 / p`.

### 9.5 Gradient Noise Regularization

When `noise > 0`, zero-mean Gaussian noise is added to accumulated gradients before the optimizer step. The noise magnitude follows an **exponential decay schedule** starting at `noise` and decaying toward zero over all training steps. This helps prevent overfitting when validation loss diverges from training loss.

---

## 10. Loss Weighting by Node Degree

High-degree nodes may dominate gradient updates. EP provides optional per-node loss scaling before gradient accumulation.

| Mode | Formula | Intent |
|---|---|---|
| `None` | `loss` | No reweighting (default). |
| `DegreeLog` | `loss / ln(1 + degree(v))` | Downweight hubs logarithmically. |
| `DegreeExponential(w)` | `loss / degree(v)^w` | Downweight hubs by a configurable power. |

---

## 11. Training Loop

### 11.1 Data Split

All nodes are randomly shuffled and partitioned into a **training set** and a **validation set** by `valid_pct`. The validation set is never used for gradient computation.

### 11.2 Mini-Batch Parallelism

Training nodes are shuffled each pass and chunked into mini-batches. Nodes within a batch are processed in parallel. Gradients for shared features are accumulated (summed) across the batch before the optimizer step.

### 11.3 Optimizer

The **Optimizer** is a first-class configuration choice.

#### Optimizer Enum

```
Optimizer::Adam { beta1: float, beta2: float, eps: float }
```

_(Additional optimizers are future work.)_

#### Adam

Standard bias-corrected Adam:

```
m_t = β₁ · m_{t-1} + (1 - β₁) · g
v_t = β₂ · v_{t-1} + (1 - β₂) · g²
m̂_t = m_t / (1 - β₁^t)
v̂_t = v_t / (1 - β₂^t)
θ ← θ - α · m̂_t / (√v̂_t + ε)
```

Default hyperparameters: `β₁ = 0.9`, `β₂ = 0.999`, `ε = 1e-8`. Updates applied with lock-free (Hogwild-style) parallelism across features.

**Note:** The current implementation uses the training pass number as the time step `t` (updated once per pass, not per batch). This is a deliberate simplification.

### 11.4 LR Scheduler

The **LRScheduler** is a first-class configuration choice.

#### LRScheduler Enum

```
LRScheduler::CosineDecayWithWarmup { warmup_pct: float }
LRScheduler::ExponentialDecay { decay: float }
LRScheduler::Constant
```

#### CosineDecayWithWarmup (default)

Linear warmup from `α / 100` to `α` over the first `warmup_pct` fraction of total steps, then cosine decay back toward `α / 100`.

```
# Warmup (step ≤ warmup_steps):
lr(step) = α_min + (step / warmup_steps) · (α - α_min)

# Decay (step > warmup_steps):
lr(step) = α_min + 0.5 · (α - α_min) · (1 + cos(π · step / max_steps))
```

Default: `warmup_pct = 0.2` (first 20% of steps).

#### ExponentialDecay

```
lr(step) = max(α_min,  α · decay^step)
```

The decay rate controls how quickly the learning rate falls. `α_min` serves as a floor.

#### Constant

Learning rate is fixed at `α` for the entire training run. No scheduling.

### 11.5 Validation

At the end of each pass, the forward pass is run (no gradient tracking) on all validation nodes with a fixed random seed. Mean validation loss is reported but does not trigger any automatic action.

### 11.6 Numerical Stability

- Losses of exactly `0.0` are skipped (no gradient).
- NaN losses are silently dropped.
- Gradients containing NaN or infinite values are not applied.

---

## 12. Warm-Start / Transfer Learning

EP accepts optional pre-trained state to initialize training:

- **Feature embedding table:** When provided, training starts from these embeddings rather than random initialization.
- **Model parameters:** For parameterized models (§15), the model's serialized weights can also be provided as a warm start. For `Model::Averaged`, there are no additional model parameters to restore.

In both cases, the Optimizer's moment buffers and the LRScheduler state are always re-initialized from scratch.

Use cases:
- **Incremental updates:** Add new nodes/features without retraining from scratch.
- **Fine-tuning:** Specialize a general-purpose embedding to a downstream graph.
- **Hyperparameter search:** Continue from a checkpoint.

---

## 13. Inference Pipeline

After training, the feature embedding table (and any Model parameters) are used to generate node embeddings on demand.

### 13.1 Aggregators

Three aggregator strategies convert a node's feature list into a node embedding:

#### Average Aggregator

```
node_emb = (1/|F(v)|) · Σ_{f ∈ F(v)} E[f]
```

Equivalent to `Model::Averaged` at inference time (without subsampling).

#### Weighted Aggregator (SIF-style)

```
w(f) = α / (α + P(f))
node_emb = Σ_f w(f) · E[f]  /  Σ_f w(f)
```

Unigram probabilities `P(f)` are computed from the full feature store before inference.

#### Attention Aggregator (excluded)

See §15.

### 13.2 Construction

Inference is stateless and trivially parallelizable across nodes.

---

## 14. Configuration and Hyperparameters

### EP-Level Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `alpha` | float | `0.9` | Peak learning rate. |
| `loss` | `Loss` enum | `MarginLoss(1.0)` | Loss function. See §8. |
| `num_negatives` | int | `1` | Negatives per anchor per forward pass. |
| `model` | `Model` enum | `Averaged` | How h(v) is computed. See §4. |
| `positive_sampler` | `PositiveSampler` enum | `LocalNeighborhood { All, false, false }` | How ~h(v) is constructed. See §5. |
| `reconstruction_aggregator` | `ReconstructionAggregator` enum | `Averaged` | How neighbor features are combined. See §6. |
| `negative_sampler` | `NegativeSampler` enum | `UniformWithHardNegatives { 0 }` | How negatives are selected. See §7. |
| `feature_dropout` | `FeatureDropout` enum | `All` | Feature subsampling per forward pass. See §9. |
| `optimizer` | `Optimizer` enum | `Adam { 0.9, 0.999, 1e-8 }` | Gradient optimizer. See §11.3. |
| `lr_scheduler` | `LRScheduler` enum | `CosineDecayWithWarmup { 0.2 }` | Learning rate schedule. See §11.4. |
| `batch_size` | int | `50` | Mini-batch size. |
| `dims` | int | `100` | Embedding dimensionality. |
| `passes` | int | `100` | Number of full training passes. |
| `seed` | int | constant | Random seed. |
| `valid_pct` | float | `0.1` | Fraction of nodes held out for validation. |
| `loss_weighting` | enum | `None` | Per-node degree-based loss scaling. See §10. |
| `noise` | float | `0.0` | Initial gradient noise std dev. See §9.5. |
| `indicator` | bool | `true` | Show progress bar. |

### Loss-Specific Parameters

| Loss | Parameters |
|---|---|
| `MarginLoss` | `gamma: float` |
| `StarSpace` | `gamma: float` |
| `Contrastive` | `pos_margin: float`, `neg_margin: float` |
| `RankLoss` | `tau: float` |
| `RankSpace` | `gamma: float` |
| `PPR` | **Deprecated** — use `MarginLoss` + `PositiveSampler::RandomWalk` |

### PositiveSampler-Specific Parameters

| Variant | Parameters |
|---|---|
| `LocalNeighborhood` | `max_nodes: FeatureDropout`, `weighted_sampling: bool`, `weighted_averaging: bool` |
| `RandomWalk` | `num_walk_samples: int`, `restart_p: float`, `max_steps: int` |

### ReconstructionAggregator-Specific Parameters

| Variant | Parameters |
|---|---|
| `Averaged` | _(none)_ |
| `Weighted` | `alpha: float` |

---

## 15. Out of Scope (Future Work)

### Parameterized Model

The original EP paper framed the algorithm around a pluggable **Model** that maps a node's features to an embedding. While the paper explored only averaged features, the Model abstraction supports future implementations with learnable weights — for example, a linear projection, a small MLP, or an attention mechanism over features.

A `Model::Parameterized` variant would:
- Own a set of learnable parameter tensors (separate from the feature embedding table).
- Serialize and deserialize those parameters independently of the feature embeddings.
- Accept a warm-start at training time (restoring previously trained weights).
- Be expressible through the same `Model` trait interface used by `Model::Averaged`.

### Attention-Based Aggregation

An attention model that replaces simple averaging with scaled multi-head self-attention over a node's features. Computationally expensive on CPU (two orders of magnitude slower than averaging). To be addressed in a separate PRD iteration.

### Additional Optimizers and Schedulers

The `Optimizer` and `LRScheduler` enums have extension points for SGD, momentum, polynomial decay, etc. No immediate plans.

### Additional NegativeSampler Variants

Mining-based negative selection (e.g., online hard negative mining using current embedding distances) could improve training efficiency. Currently deferred.

---

## 16. Changelog

| Version | Date | Notes |
|---|---|---|
| v1 | 2026-03-10 | Initial PRD. Covers averaged model, all 6 loss functions, negative sampling, subsampling, degree weighting, training loop, warm-start, and inference. Attention excluded pending separate treatment. |
| v2 | 2026-03-10 | Moved `num_negatives` out of loss function parameters into EP-level config. Loss functions now define only loss-specific parameters and operate on a fixed interface. PPR's positive-construction count renamed `num_walk_samples`. |
| v3 | 2026-03-10 | Introduced first-class enums: `PositiveSampler` (LocalNeighborhood, RandomWalk), `NegativeSampler` (UniformWithHardNegatives), `ReconstructionAggregator` (Averaged, Weighted), `FeatureDropout` (All, Fixed, Probability), `Optimizer` (Adam), `LRScheduler` (CosineDecayWithWarmup, ExponentialDecay, Constant), and `Model` (Averaged; Parameterized as future work). `max_nodes` and `weighted_*` parameters moved into `PositiveSampler::LocalNeighborhood`. PPR loss deprecated in favor of `MarginLoss` + `PositiveSampler::RandomWalk`. Model established as a serializable component to support future parameterized variants. |
