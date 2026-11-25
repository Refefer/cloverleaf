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

The recommended configuration, also in `examples/train.sh`:

```
python -m ep_gpu.train --edges GRAPH --features FEATURES --out PREFIX \
    --dims 128 --lr 0.3 --loss starspace --gamma 0.1 --negatives 12 \
    --max-nodes 12 --max-features 0.5 --weighted-averaging --valid-pct 0.01 --passes 100 \
    --pooling attention --heads 4 --neighbour-pooling mean --batch-size 2048 \
    --optimizer rowwise_adagrad --table-dtype bfloat16 \
    --lr-schedule sawtooth --lr-cycles 4 --graph-cache PREFIX.graph.pt
```

Attention pooling with mean-combined neighbours and StarSpace retrieved best in our experiments;
row-wise Adagrad with a bfloat16 table needs a fraction of Adam's memory; and the sawtooth
schedule (4 cycles of linear decay, restarting at the peak rate) reaches a lower validation loss
than cosine in the same passes.  Tune `dims`, `lr`, `gamma`, `negatives`, `max_nodes` and
`max_features` for your graph.  `python -m ep_gpu.train --help` lists every option; the `Config`
class in `ep_gpu/train.py` documents them.

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

* Files: nodes numbered in order of first appearance, features namespaced by node type (or one
  shared namespace), and a node without features gets its own `("node", id)` feature.
  `--shared-namespace NAME` (GPU only) gives each token a feature in NAME as well as its node
  type's, and starts the node-type features at zero: each is an offset from the shared one, so
  the same word on two node types starts out identical and diverges only as training needs.
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
  the listwise losses, feature pruning on load (`--prune-min-count`), shared feature namespaces
  (`--shared-namespace`) and `epoch_after`; and the options below: the Poincaré geometry,
  `negative_types`, `margin_sq`, the attention variants and `feature_importance`.

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

A starting point: the recommended configuration with `--geometry poincare --curvature 0.3
--loss starspace --gamma 1.0` (the margin is now in distance units) and mean pooling (see
Pooling options).  In our tests hyperbolic models held up at far lower widths than Euclidean ones,
so try 32-64.
* The aggregator file isn't written for a Poincaré model (cloverleaf's `FeatureAggregator` pools
  without the map); load its node embeddings with a Poincaré distance to search them.

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

The model (`PREFIX.pt`) is written first, then the node and feature embeddings, and the
aggregator only when cloverleaf's `FeatureAggregator` reproduces the model's node embeddings:
Euclidean space, mean or `transform` attention, no `attention_keeps_norm` and no feature
importance.  Embeddings are written with `cloverleaf.write_embeddings` when the installed
cloverleaf has it: rows are formatted in parallel with the GIL released, tens of times faster
than the pure-Python fallback.

## Tests

`pytest tests` (CPU, a few seconds).  `tests/test_cloverleaf_parity.py` runs when cloverleaf is
installed.
