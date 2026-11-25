# EP on the GPU

A PyTorch implementation of cloverleaf's Embedding Propagation, for graphs where training on the
CPU is too slow, attention pooling in particular.  It reads cloverleaf's graph and feature files
and writes files cloverleaf loads, so embedding new nodes, alignment, ANN search and evaluation
all stay in cloverleaf.

## Install

From the repository root, `make gpu` builds cloverleaf into a virtualenv and installs torch and the
other requirements into it (see the main README).  Or, with cloverleaf already installed:

```
pip install -r requirements.txt
```

## Quickstart

```
cd examples
python quickstart.py
```

trains on a small synthetic query/product graph, then embeds a query the graph has never seen,
aligns it and retrieves products, all in cloverleaf.  `examples/` has the other examples.

## Training

`python -m ep_gpu.train --edges GRAPH --features FEATURES --out PREFIX` trains with the defaults;
`--help` lists every option and the `Config` class in `ep_gpu/train.py` documents them.
`examples/train.sh` is a starting point for large graphs.  Which settings suit a graph depends
on the task, so start from the guidance below and choose by a retrieval evaluation on held-out
nodes, not by training or validation loss, which have ranked models differently from retrieval.

**Loss.**  StarSpace (a margin on cosine, or on distance in the Poincaré ball) is the robust
default for retrieval.  The margin loss (EP-B's) suits embeddings used as features, such as node
classification.  InfoNCE puts more weight on the top of the ranking but overfits: pair it with
`--valid-edges` and `--keep-best-probe` to stop at the best pass.  The listwise losses (ListNet,
ListMLE) train against personalised PageRank lists, for multi-hop relevance.

**Negatives.**  Random negatives give the broadest separation from unrelated nodes.  To retrieve
one node type for another (items for queries, papers for authors), draw each side's negatives
from the other side (`--negative-types`), which trains the ranking retrieval uses.  A few hard
negatives (`--hard-negatives`) alongside random ones sharpen the top of the ranking at some cost
to recall; because margin losses average only violated margins, even one carries much of the
gradient.

**Geometry.**  Euclidean space (cosine) suits graphs without a strong hierarchy.  On graphs with
a hierarchy or a heavily skewed popularity (a few hubs, a long tail of rare nodes), the Poincaré
ball (`--geometry poincare`) can place general, popular nodes near the origin and specific ones
further out, and holds up at far lower widths.  Lower `--curvature` is flatter and closer to
Euclidean; higher favours the hierarchy at the head.  Margins are then in distance units, so
scale `--gamma` up from cosine values.

**Pooling.**  Mean pooling is the safe default.  Learned weightings (attention, `--attention-mode
weights`, `--feature-importance`) help when nodes carry many features of uneven value, such as
long documents, and tend to memorise identifier-like features (a node's own id, codes in a
title) otherwise, most of all in the Poincaré ball.  If you use attention there, use
`--attention-keeps-norm` and a strong `--attention-dropout` (0.6 or more).

**Alignment.**  Neighbourhood alignment (cloverleaf's `NeighborhoodAligner`, then
`EmbeddingAligner(method="mean_shift")` for nodes without edges) helps Euclidean models,
most of all ones trained with random negatives.  Poincaré models already place nodes by their
neighbourhoods, and averaging a node with neighbours at different distances from the origin
blurs that, so use their embeddings unaligned; cloverleaf's aligners also average in Euclidean
space, which isn't the ball's own mean.

**Width.**  Euclidean models usually want 100 or more dimensions; Poincaré ones often do as well
with 32-64, a several-fold smaller feature table.

**Scale and optimisation.**  For large graphs, row-wise Adagrad with a bfloat16 table
(`--optimizer rowwise_adagrad --table-dtype bfloat16`) needs a fraction of Adam's memory and
takes a larger learning rate (around 0.1-0.5); use large batches (thousands of nodes) there and
small ones (tens to hundreds) on graphs of a few thousand nodes.  The cosine schedule is a good
default.  Sample a dozen or so neighbours per reconstruction (`--max-nodes`) and about half of
each node's features (`--max-features`) as regularisation, and use `--weighted-averaging` when
edge weights carry signal.  `--graph-cache` saves the parsed graph for the next run.

## Output

`PREFIX.node-embeddings` (every node, pooled from all its features), `PREFIX.feature-embeddings`
(the feature table), `PREFIX.aggregator` (the pooling, as a cloverleaf `FeatureAggregator` file)
and `PREFIX.pt` (the model).  In cloverleaf:

```python
import cloverleaf as cl
features = cl.NodeEmbeddings.load("PREFIX.feature-embeddings", cl.Distance.Cosine)
embedder = cl.NodeEmbedder(cl.FeatureAggregator.load("PREFIX.aggregator"))
new_node = embedder.embed_adhoc([("query", "red"), ("query", "shoes")], features, strict=False)
```

embeds a node from its features exactly as the trained model does, with mean or attention
pooling (`tests/test_cloverleaf_parity.py` checks this; `ep_gpu.embed_feature_sets` does the same
in PyTorch).  After `NeighborhoodAligner`, `EmbeddingAligner(k, method="mean_shift")` places such
a node in the aligned space by how its nearest nodes moved.

## What matches cloverleaf

* Files are read by cloverleaf itself (`Graph.load`, `FeatureSet.load_into`), so nodes, features
  and their ids are exactly cloverleaf's: nodes numbered in order of first appearance, features
  namespaced by node type (or one shared namespace), and a node without features gets its own
  `("node", id)` feature.  A token repeated on a line counts once (`dedup=True`), and edge
  weights are each node's transition probabilities.  `--shared-namespace NAME` loads with
  `FeatureNamespace.node_type_and_shared(NAME)`, each token getting a feature in NAME as well as
  its node type's, and starts the node-type features at zero: each is an offset from the shared
  one, so the same word on two node types starts out identical and diverges only as training
  needs.  `--prune-min-count` is cloverleaf's `prune_min_count`.
* h(v) pools a node's own features; h~(v) pools all its sampled neighbours' features together,
  so a feature shared by several neighbours counts once per neighbour.
* Negatives are uniform random nodes other than the anchor, embedded from their own features.
* Losses: margin (Euclidean), StarSpace, rank and RankSpace, each averaging only its non-zero
  terms, as `src/algos/ep/loss.rs` does.
* `max_features` (a fraction or a count), `max_nodes`, weighted sampling and weighted averaging,
  where averaging skips the edge weights for neighbours already sampled by weight.
* Attention pooling (`--pooling attention`) is cloverleaf's attention model: multi-head
  self-attention over a node's features with query, key, value and output projections shared by
  every feature, the node embedding being the features' mean plus the projected mean attention
  context (the output projection starts at zero, so training starts from mean pooling).  A
  reconstruction pools each sampled neighbour from its own features, then takes the neighbours'
  weighted mean (`reconstruction="nested"`, the default for attention, with
  `neighbour_pooling="mean"`).  Weights move both ways through the `PREFIX.aggregator` file.
* Adam on the feature table with sparse updates; "cosine" (20% warmup from lr/100, then cosine
  decay to lr/100) or "constant" learning rates; "unit" or "glorot" initialisation.
* `update_frequency`: "batch" (an update after every batch) or "epoch" (every batch of a pass is
  computed against the same embeddings, their gradients are summed, and one update applies the
  sum at the pass's end; the schedule counts one step per pass).  Epoch updates keep a float32
  gradient table the size of the feature table, on the table's device.  With `epoch_after K`
  (GPU only), the first K passes make batch updates and the rest epoch updates; the schedule
  runs over passes, so the rate doesn't jump at the switch, and the optimiser state carries over.

With mean pooling, cloverleaf's averaged `NodeEmbedder` rebuilds the exported node embeddings
from the exported feature table exactly, and features-only EP on Cora scores the same on both
(77.32 ± 1.29 GPU, 77.43 ± 1.13 CPU; `examples/parity_cora.py`).  With attention pooling,
cloverleaf embeds nodes from the exported feature table and aggregator to within 1e-5 of the
GPU model.

## What differs

* Nodes with more neighbours than `max_nodes` are sampled with replacement (cloverleaf samples
  without).  `hard_negatives` here are the pool nodes nearest the reconstruction (cloverleaf's
  come from random walks).
* `neighbour_pooling="attention"` (a nested reconstruction's neighbours combined by attention
  rather than their weighted mean) and flat attention reconstructions are GPU only.
* GPU only: the "sawtooth" learning rate (`lr_cycles` equal cycles, each a linear decay from `lr`
  to `lr`/100, the first after the warmup), row-wise optimisers and a bfloat16 table, InfoNCE and
  the listwise losses, zero-initialised offsets with `--shared-namespace`, and `epoch_after`; and
  the options below: midpoint pooling in the Poincaré ball, `negative_types`, `margin_sq`, the
  attention variants and `feature_importance` (cloverleaf has the Poincaré geometry itself, with
  tangent pooling and `max_tangent_norm`).

## Speed and precision

* Each batch looks up every distinct feature once, and attention projects each distinct
  feature's query, key and value once, applying the output projection after pooling.
* Negatives come from a pool of `negative_pool` nodes embedded once per batch.
* `precision` is "bf16" by default: the forward pass runs under bfloat16 autocast while the
  feature table, optimiser state and losses stay float32.  Attention heads are widened to a
  multiple of 8 so the fused attention kernels apply.
* The attention weights train with their own learning rate, `dense_lr` (1e-3), on the same
  schedule as `lr`; a rate tuned for sparse feature updates is far too large for dense matrices.
* `examples/profile_step.py` profiles a few training steps.

## Memory

The feature table and its optimiser state dominate GPU memory.  Per feature, with d dims:

| `optimizer` | state | with a float32 table | with `table_dtype bfloat16` |
|---|---|---|---|
| `adam` (default; cloverleaf's optimiser) | two full moments | 12d bytes | 10d |
| `rowwise_adam` | full first moment, one second moment per row | 8d + 4 | 6d + 4 |
| `rowwise_adagrad` | one accumulated squared gradient per row | 4d + 4 | 2d + 4 |

`rowwise_adagrad` is the usual choice for large embedding tables (FBGEMM's and TorchRec's
default).  A bfloat16 table's updates are computed in float32 and written back with stochastic
rounding, so steps smaller than bfloat16's spacing still move the weights on average.  The peak
GPU memory of a run is printed when it finishes.

## Large graphs

Node and feature ids are stored as int32 whenever they fit, halving the largest index arrays.
`graph_device cpu` keeps the graph in host memory and samples there, copying each batch's ids
to the GPU; `table_device cpu` keeps the feature table and its optimiser state in pinned host
memory, copying each batch's distinct rows to the GPU and their gradients back.  With both, GPU
memory holds only the dense attention weights and one batch's activations, whatever the graph's
size.

## Losses and reconstruction

* `loss`: margin, starspace, rank, rankspace (as cloverleaf) or `infonce`: cross entropy over
  cosine similarities divided by a temperature (`gamma`).  With `pool_negatives`, each node's
  softmax runs over the whole negative pool with one matrix multiply.
* `listnet` / `listmle`: no reconstruction.  Each node is scored against its own personalised
  PageRank list, which cloverleaf builds (`cloverleaf.ppr_lists`, as `PprRankLearner` uses, with
  `ppr_k`, `ppr_walks`, `ppr_steps`, `ppr_beta`, `ppr_compression` and `ppr_other_type`), and
  against negatives from the pool.  `ppr_steps` below 1 is the restart probability; a whole
  number is a fixed walk length, and on a bipartite graph an odd one always ends on the other
  side, so no walks are wasted.  `ppr_sample` draws that many of each list's `ppr_k` entries every
  time the node is used (`ppr_sample_by` weight or uniform), so a long list doesn't become a
  fixed set of targets to memorise.  Logits are cosine similarities over the temperature
  (`gamma`).  ListNet is cross entropy against the list's weights; top-k ListMLE is the
  likelihood of the listed nodes coming first, in order.  This trains embeddings the way
  retrieval uses them, node against node, and needs cloverleaf installed.
* `neighbour_pooling` ("mean" by default) combines a nested reconstruction's neighbours.  A
  weighted mean keeps h~(v) in the same space as the node embeddings it's compared with;
  attention there learns its own transform, which lowers the training loss but makes nodes
  worse at retrieving their neighbours.

## Geometry: the Poincaré ball

`--geometry poincare` (`ep_gpu/geometry.py`) puts the node embeddings in the Poincaré ball of
curvature `-curvature`.  The feature table and pooling don't change: a node's features pool to a
vector, which is read as a tangent vector at the origin and mapped into the ball with the
exponential map, `exp0(v) = tanh(sqrt(c)|v|) v / (sqrt(c)|v|)`.  Pairs are scored by negative
hyperbolic distance, and the margin loss uses that distance; StarSpace, InfoNCE and the listwise
losses use it as their similarity (logits are `-distance / gamma`).  Every step is
differentiable and the parameters stay Euclidean, so the optimisers apply unchanged.  Distance
grows exponentially towards the ball's edge, so a node's distance from the origin can encode how
general it is: popular, general nodes end up near the origin, specific ones further out.

* `curvature` (default 1).  Lower curvature is flatter (`c -> 0` is Euclidean); 0.1-1 is the
  useful range.
* `max_tangent_norm` clips the pooled tangent vector before the map, keeping points away from
  the edge, where gradients vanish and float32 loses precision.
* `poincare_pooling`: `tangent` (default) averages the features' vectors and maps the mean;
  `midpoint` maps every feature into the ball and takes their weighted gyromidpoint;
  `neighbour_midpoint` averages features as `tangent` but combines a nested reconstruction's
  neighbours by their gyromidpoint.  The geometry-aware means haven't beaten `tangent`: many
  scattered word vectors average towards the origin, where every long title would look generic.
* The rank and rankspace losses score by dot product, so they need Euclidean space.

When to use it, and how to set the curvature, margin and width: see Training.
* With tangent pooling the exported aggregator records the geometry, and cloverleaf's
  `NodeEmbedder` maps new nodes into the ball exactly as the model does; search the node
  embeddings with `cloverleaf.PoincareDistance(curvature)`.

Graph reconstruction on a balanced tree (branching 3, depth 6; mean average precision of each
node's neighbours, InfoNCE over distance with one sampled neighbour per reconstruction):

| | 2 dims | 5 dims |
|---|---|---|
| Euclidean, best loss | 0.434 | 0.597 |
| Poincaré | 0.696 | 0.873 |

## Negatives

* `hard_negatives k`: also use the `k` pool nodes nearest h~(v) as negatives, alongside the
  `negatives` random ones (`--negatives 0` for hard negatives only).  The anchor's own neighbours
  are never picked: the nearest nodes are often real neighbours, and pushing them away would
  fight the positive term.  Hard negatives sharpen the top of the ranking (NDCG) at some cost to
  separation from random nodes (HitRate).  Because margin losses average only the negatives that
  violate the margin, even one hard negative carries much of the gradient.
* `negative_types "query=product;product=query"`: for each anchor node type, the node types its
  negatives may be; types left out may draw any type.  The pool is drawn per allowed type in
  equal shares, so a rare type always has candidates.  On a bipartite graph, giving each side the
  other side's nodes as negatives trains the ranking retrieval uses.
* `loss margin_sq`: max(0, m + d(t, h)^2 - d(t, hu)^2), the margin ranking loss of hyperbolic
  recommenders; `adaptive_margin` makes m a geometry-aware margin, `gamma * sigmoid(.)` of the
  pair's distances from the origin.

## Pooling options

* `pooling attention` adds `out(ctx)`, the output projection of the tokens' attention context,
  to their mean.  As the value and output projections are linear, that term is one matrix every
  node shares applied to an attention-weighted mean of the node's features.
  * `attention_keeps_norm` lets it change only the pooled vector's direction.  In the Poincaré
    ball, free, the shared projection learns to stretch every node outwards (larger distances
    satisfy margins cheaply), so a node's distance from the origin stops following its features.
  * `attention_dropout p` drops the added term's coordinates in training, so the shared
    projection can't co-adapt into one shift that fits the training nodes.
  * `attention_mode weights` keeps only the per-node importance: the node is a convex
    combination of its own feature embeddings, weighted by the attention each receives, with no
    value or output projection.
* `feature_importance`: one learned scalar per feature, weighting it by `exp(s)` in every mean it
  is pooled into (a learned IDF); starts at 0, the plain mean.

Every learned weighting starts as the plain mean and can only move away from it as far as
training rewards.  Whether that helps depends on the data.  With random negatives the objective
rewards emphasising whatever tells a node apart from the others, so learned weights drift
towards identifier-like features (a node's own id, codes in its title) and memorise.  Held-out
link prediction (HitRate@10 of 10% held-out edges against 1,000 random nodes, 64 dims, margin
loss; change against mean pooling):

| | Euclidean, ids + words | Euclidean, words | Poincaré, ids + words | Poincaré, words |
|---|---|---|---|---|
| Cora, importance | +0.020 | +0.042 | +0.002 | +0.016 |
| Cora, attention | -0.063 | -0.046 | -0.037 | +0.021 |
| Citeseer, importance | -0.001 | -0.002 | -0.014 | -0.024 |
| Citeseer, attention + dropout 0.6 | -0.013 | -0.009 | +0.014 | +0.001 |
| Pubmed, importance | +0.139 | +0.063 | -0.017 | +0.026 |
| Pubmed, attention | +0.041 | +0.067 | -0.068 | +0.055 |

Learned weighting helps where nodes carry many features of uneven value (Pubmed) and hurts in the
Poincaré ball when nodes carry identifiers to memorise.  Mean pooling is the safe default.

## Export

The model (`PREFIX.pt`) is written first, then the node and feature embeddings with
`cloverleaf.write_embeddings` (rows formatted in parallel with the GIL released), and the
aggregator with `cloverleaf.FeatureAggregator.save` when cloverleaf reproduces the model's node
embeddings: mean or `transform` attention pooling, in Euclidean space or the Poincaré ball with
tangent pooling (the aggregator records the curvature and clip), without `attention_keeps_norm`
or feature importance.

## Tests

`pytest tests` (CPU, a few seconds; needs cloverleaf).
