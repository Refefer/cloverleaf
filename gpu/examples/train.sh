#!/bin/bash
# A starting point for GPU EP on a large graph: StarSpace, mean pooling, row-wise Adagrad with a
# bfloat16 table, and the cosine schedule.  The choices that depend on the task (negatives,
# geometry, pooling, width, margin) are discussed under Training in ../README.md; pass them as
# extra options, e.g. for item retrieval on a skewed query/item graph:
#
#   ./train.sh GRAPH FEATURES PREFIX --geometry poincare --gamma 1.0 --dims 64 \
#       --negatives 6 --hard-negatives 6 --negative-types "query=item;item=query"
#
#   ./train.sh GRAPH FEATURES PREFIX [more ep_gpu.train options]
#
# Writes PREFIX.node-embeddings, PREFIX.feature-embeddings, PREFIX.aggregator and PREFIX.pt.
set -euo pipefail
if [ $# -lt 3 ]; then
  echo "usage: $0 GRAPH FEATURES PREFIX [options]" >&2
  exit 2
fi
edges=$(realpath "$1") features=$(realpath "$2") prefix=$(realpath -m "$3")
shift 3
mkdir -p "$(dirname "$prefix")"
cd "$(dirname "$0")/.."
python -m ep_gpu.train --edges "$edges" --features "$features" --out "$prefix" \
  --dims 128 --loss starspace --gamma 0.1 --negatives 12 \
  --max-nodes 12 --max-features 0.5 --weighted-averaging --valid-pct 0.01 --passes 100 \
  --optimizer rowwise_adagrad --table-dtype bfloat16 --lr 0.3 --batch-size 4096 \
  --graph-cache "$prefix.graph.pt" \
  "$@"
