import pytest
import torch

from ep_gpu import data, sampling
from ep_gpu.train import Config, node_embeddings, train


def test_parse_negative_types():
    assert sampling.parse_negative_types("query=product; product=product,query") == \
        {"query": ["product"], "product": ["product", "query"]}
    for bad in ["", "query", "query=", "=product", "q=a;q=b"]:
        with pytest.raises(ValueError):
            sampling.parse_negative_types(bad)


def test_negative_types_pool_and_mask():
    types = ["query"] * 90 + ["product"] * 10            # products are rare
    nt = sampling.NegativeTypes(types, {"query": ["product"]}, "cpu")
    pool = nt.pool(100, torch.Generator().manual_seed(0))
    # Unmapped products may draw anything, so both types are pooled, in equal shares.
    assert (nt.type_of[pool] == nt.type_of[95]).sum() == 50
    allowed = nt.allowed_mask(torch.tensor([0, 95]), pool)
    assert (allowed[0] == (nt.type_of[pool] == nt.type_of[95])).all()      # a query: products only
    assert allowed[1].all()                                                 # a product: anything
    with pytest.raises(ValueError, match="not in the graph"):
        sampling.NegativeTypes(types, {"query": ["user"]}, "cpu")


def test_pick_allowed_draws_only_allowed_positions():
    allowed = torch.tensor([[False, True, False, True], [True, False, False, False]])
    pick = sampling.pick_allowed(allowed, 50, torch.Generator().manual_seed(0))
    assert allowed.gather(1, pick).all()
    assert sampling.pick_allowed(allowed, 0).shape == (2, 0)
    with pytest.raises(ValueError, match="no allowed negative"):
        sampling.pick_allowed(torch.tensor([[False, False]]), 1)


@pytest.fixture
def bipartite(tmp_path):
    """Two groups of queries, each clicking its own group of products."""
    path = tmp_path / "g.edges"
    with open(path, "w") as f:
        for g in range(2):
            for q in range(8):
                for p in range(6):
                    f.write(f"query\tq{g}_{q}\tproduct\tp{g}_{p}\t1\n")
    return data.load(str(path))


@pytest.mark.parametrize("geometry, kwargs", [
    ("euclidean", dict(negatives=2, hard_negatives=2)),
    ("poincare", dict(negatives=2, hard_negatives=2, curvature=0.3)),
    ("poincare", dict(negatives=0, hard_negatives=3)),
    ("euclidean", dict(negatives=3, negative_types="query=product;product=query")),
    ("poincare", dict(negatives=2, hard_negatives=2, negative_types="query=product")),
])
def test_mixed_and_typed_negatives_separate_the_groups(bipartite, geometry, kwargs):
    cfg = Config(dims=8, passes=40, batch_size=4, lr=0.1, negative_pool=16, loss="starspace",
                 gamma=0.5, geometry=geometry, device="cpu", **kwargs)
    model = train(bipartite, cfg, log=lambda m: None)
    X = node_embeddings(model, bipartite)
    names = bipartite.node_names
    group = torch.tensor([int(n[1]) for n in names])
    is_query = torch.tensor([t == "query" for t in bipartite.node_types])
    S = model.space.pairwise_similarity(X[is_query], X[~is_query])
    same = group[is_query][:, None] == group[~is_query][None, :]
    assert S[same].mean() > S[~same].mean()


def test_negative_types_are_validated():
    with pytest.raises(ValueError, match="negative_types"):
        from ep_gpu.train import check_loss
        check_loss(Config(loss="starspace", negative_types="query=product", negative_pool=None))
