# GPU EP examples

Run them from this directory with torch and cloverleaf installed (`make gpu` at the repository
root sets both up).

| Example | What it shows |
|---|---|
| `quickstart.py` | End to end on a synthetic query/product graph: train with a general starting configuration, export, then in cloverleaf embed a query the graph has never seen from its words, align it with `EmbeddingAligner(method="mean_shift")` and retrieve products.  About a minute on a CPU. |
| `train.sh` | A starting configuration from the command line for your own graph and feature files; add task-specific options as Training in ../README.md describes. |
| `parity_cora.py` | GPU and CPU EP learn equally well: features-only EP on Cora with the EP paper's protocol, on both. |
| `profile_step.py` | Where a training step's time goes, from the PyTorch profiler. |
