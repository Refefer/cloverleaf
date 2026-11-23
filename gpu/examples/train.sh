#!/bin/bash
# The recommended GPU EP configuration from the command line: attention pooling with
# mean-combined neighbours, StarSpace, row-wise Adagrad with a bfloat16 table, and the sawtooth
# learning rate (4 cycles).  Tune dims, lr, gamma, negatives, max_nodes and max_features for
# your graph; the rest are good defaults.
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
  --dims 128 --lr 0.3 --loss starspace --gamma 0.1 --negatives 12 \
  --max-nodes 12 --max-features 0.5 --weighted-averaging --valid-pct 0.01 --passes 100 \
  --pooling attention --heads 4 --neighbour-pooling mean --batch-size 2048 \
  --optimizer rowwise_adagrad --table-dtype bfloat16 \
  --lr-schedule sawtooth --lr-cycles 4 \
  --graph-cache "$prefix.graph.pt" \
  "$@"
