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
`PREFIX.feature-embeddings` (the feature table) and `PREFIX.pt` (the model).  New nodes are
embedded from their features with `ep_gpu.embed_feature_sets`.

## What matches cloverleaf

* Files: nodes numbered in order of first appearance, features namespaced by node type (or one
  shared namespace), and a node without features gets its own `("node", id)` feature.
* h(v) pools a node's own features; h~(v) pools all its sampled neighbours' features together,
  so a feature shared by several neighbours counts once per neighbour.
* Negatives are uniform random nodes other than the anchor, embedded from their own features.
* Losses: margin (Euclidean), StarSpace, rank and RankSpace, each averaging only its non-zero
  terms, as `src/algos/ep/loss.rs` does.
* `max_features` (a fraction or a count), `max_nodes`, weighted sampling and weighted averaging,
  where averaging skips the edge weights for neighbours already sampled by weight.
* Adam on the feature table with sparse updates; "cosine" (20% warmup from lr/100, then cosine
  decay to lr/100) or "constant" learning rates; "unit" or "glorot" initialisation.

With mean pooling, cloverleaf's averaged `NodeEmbedder` rebuilds the exported node embeddings
from the exported feature table exactly, and features-only EP on Cora scores the same on both
(77.32 ± 1.29 GPU, 77.43 ± 1.13 CPU; `examples/parity_cora.py`).

## What differs

* Attention: cloverleaf stores per-head query and key slices in every feature's embedding.  Here
  the query, key, value and output projections are shared across features, so the table stays
  one `dims`-wide vector per feature (several times less memory).  Attention is
  self-attention over a node's tokens followed by their weighted mean; the output projection
  starts at zero, so training starts from mean pooling.
* Nodes with more neighbours than `max_nodes` are sampled with replacement (cloverleaf samples
  without); hard negatives aren't implemented.
* Attention reconstructs a node in two levels (`reconstruction`, "auto" by default): each
  sampled neighbour is pooled from its own features, then the neighbours are pooled together.
  Cloverleaf attends over all the neighbours' features at once, which on a GPU runs out of
  memory at useful batch sizes.  Mean pooling keeps cloverleaf's flat reconstruction.

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
