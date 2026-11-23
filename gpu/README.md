# EP on the GPU

A PyTorch implementation of cloverleaf's Embedding Propagation, for experiments that are too slow
on the CPU, attention pooling in particular.  It reads cloverleaf's graph and feature files and
writes embeddings cloverleaf loads, so alignment, ANN search and evaluation still use cloverleaf.

```
pip install -r requirements.txt
python -m ep_gpu.train --edges GRAPH --features FEATURES --out PREFIX \
    --loss margin --gamma 1 --negatives 5 --max-nodes 10 --max-features 0.5 \
    --pooling attention --heads 4 --batch-size 1024 --passes 100
```

writes `PREFIX.node-embeddings` (every node, pooled from all its features),
`PREFIX.feature-embeddings` (the feature table), `PREFIX.aggregator` (the pooling, as a cloverleaf
`FeatureAggregator` file) and `PREFIX.pt` (the model).  New nodes are embedded from their features
by cloverleaf, `NodeEmbedder(FeatureAggregator.load(PREFIX.aggregator))` over the feature table,
with mean or attention pooling, or in PyTorch with `ep_gpu.embed_feature_sets`; the two agree.

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
  gradient table the size of the feature table, on the table's device.

With mean pooling, cloverleaf's averaged `NodeEmbedder` rebuilds the exported node embeddings
from the exported feature table exactly, and features-only EP on Cora scores the same on both
(77.32 ± 1.29 GPU, 77.43 ± 1.13 CPU; `examples/parity_cora.py`).

## What differs

* Nodes with more neighbours than `max_nodes` are sampled with replacement (cloverleaf samples
  without); hard negatives aren't implemented.
* `neighbour_pooling="attention"` (a nested reconstruction's neighbours combined by attention
  rather than their weighted mean) and flat attention reconstructions are GPU only.

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

## Export

Embeddings are written with `cloverleaf.write_embeddings` when the installed cloverleaf has it:
rows are formatted in parallel with the GIL released, tens of times faster than the pure-Python
fallback.

## Tests

`pytest tests` (CPU, a few seconds).  `tests/test_cloverleaf_parity.py` runs when cloverleaf is
installed.
