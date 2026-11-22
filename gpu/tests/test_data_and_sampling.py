import pytest
import torch

from ep_gpu import data, sampling


@pytest.fixture
def tiny(tmp_path):
    """a - b - c - a triangle plus c - d; d has no features."""
    edges = tmp_path / "g.edges"
    edges.write_text("n\ta\tn\tb\t1\n"
                     "n\tb\tn\tc\t1\n"
                     "n\tc\tn\ta\t3\n"
                     "n\tc\tm\td\t1\n")
    feats = tmp_path / "g.features"
    feats.write_text("n\ta\tx y\n"
                     "n\tb\ty z w\n"
                     "n\tc\tx\n"
                     "n\tzz\tq\n")   # not in the graph: ignored
    return str(edges), str(feats)


def neighbours(g, v):
    return set(g.adj_idx[g.adj_ptr[v]:g.adj_ptr[v + 1]].tolist())


def features(g, v):
    return {g.feature_keys[f] for f in g.feat_idx[g.feat_ptr[v]:g.feat_ptr[v + 1]].tolist()}


def test_nodes_are_numbered_in_order_of_appearance(tiny):
    g = data.load(*tiny)
    assert list(zip(g.node_types, g.node_names)) == [("n", "a"), ("n", "b"), ("n", "c"), ("m", "d")]


def test_undirected_adjacency(tiny):
    g = data.load(*tiny)
    assert neighbours(g, 0) == {1, 2}
    assert neighbours(g, 2) == {0, 1, 3}
    assert neighbours(g, 3) == {2}


def test_directed_adjacency(tiny):
    g = data.load(*tiny, directed=True)
    assert neighbours(g, 0) == {1}
    assert neighbours(g, 3) == set()


def test_features_are_namespaced_by_node_type_and_missing_nodes_get_their_own(tiny):
    g = data.load(*tiny)
    assert features(g, 0) == {("n", "x"), ("n", "y")}
    assert features(g, 3) == {("node", "3")}
    assert ("n", "q") not in g.feature_keys


def test_single_namespace(tiny):
    g = data.load(*tiny, namespace="feat")
    assert features(g, 1) == {("feat", "y"), ("feat", "z"), ("feat", "w")}


def test_sample_all_features(tiny):
    g = data.load(*tiny)
    idx, mask = sampling.sample_features(g, torch.tensor([0, 1, 2, 3]))
    for row, v in enumerate([0, 1, 2, 3]):
        got = {g.feature_keys[f] for f in idx[row][mask[row]].tolist()}
        assert got == features(g, v)


@pytest.mark.parametrize("max_features,expected", [(0.5, [1, 2, 1, 1]), (2, [2, 2, 1, 1]), (1, [1, 1, 1, 1])])
def test_sample_feature_subsets(tiny, max_features, expected):
    g = data.load(*tiny)
    nodes = torch.tensor([0, 1, 2, 3])
    for seed in range(20):
        gen = torch.Generator().manual_seed(seed)
        idx, mask = sampling.sample_features(g, nodes, max_features, gen)
        assert mask.sum(dim=1).tolist() == expected
        for row, v in enumerate(nodes.tolist()):
            sampled = [g.feature_keys[f] for f in idx[row][mask[row]].tolist()]
            assert len(set(sampled)) == len(sampled), "sampled without replacement"
            assert set(sampled) <= features(g, v)


def test_sample_all_neighbours(tiny):
    g = data.load(*tiny)
    nbrs, w, mask, sampled = sampling.sample_neighbours(g, torch.tensor([0, 2, 3]))
    assert set(nbrs[1][mask[1]].tolist()) == {0, 1, 3}
    assert set(nbrs[2][mask[2]].tolist()) == {2}
    assert not sampled.any()
    # c - a has weight 3
    assert dict(zip(nbrs[0][mask[0]].tolist(), w[0][mask[0]].tolist())) == {1: 1.0, 2: 3.0}


def test_sampled_neighbours_are_real_neighbours(tiny):
    g = data.load(*tiny)
    nodes = torch.tensor([0, 1, 2, 3])
    for seed in range(20):
        nbrs, _, mask, sampled = sampling.sample_neighbours(g, nodes, max_nodes=2,
                                                            generator=torch.Generator().manual_seed(seed))
        for row, v in enumerate(nodes.tolist()):
            assert set(nbrs[row][mask[row]].tolist()) <= neighbours(g, v)
        # Only c has more than 2 neighbours, so only c is sampled; the rest take all of theirs.
        assert sampled.tolist() == [False, False, True, False]
        assert mask.sum(dim=1).tolist() == [2, 2, 2, 1]


def test_weighted_sampling_follows_edge_weights(tmp_path):
    path = tmp_path / "star.edges"
    path.write_text("n\thub\tn\tx\t1\nn\thub\tn\ty\t3\nn\thub\tn\tz\t0.0001\n")
    g = data.load(str(path))
    hub = torch.zeros(20000, dtype=torch.long)
    nbrs, _, mask, _ = sampling.sample_neighbours(g, hub, max_nodes=1, weighted=True,
                                                  generator=torch.Generator().manual_seed(0))
    picked = nbrs[mask]
    share_y = (picked == g.node_id("n", "y")).float().mean().item()
    assert abs(share_y - 0.75) < 0.02, share_y


def test_negatives_are_never_the_anchor():
    nodes = torch.arange(5).repeat(200)
    neg = sampling.sample_negatives(5, nodes, 3, torch.Generator().manual_seed(0))
    assert (neg != nodes[:, None]).all()
    assert neg.min() >= 0 and neg.max() < 5
    assert set(neg.flatten().tolist()) == set(range(5))


def test_neighbourhood_tokens_count_each_neighbours_features(tiny):
    g = data.load(*tiny)
    idx, w = sampling.neighbourhood_tokens(g, torch.tensor([2]))
    # c's neighbours a (x, y), b (y, z, w) and d (node 3): y counts twice.
    counts = {}
    for f, wt in zip(idx[0].tolist(), w[0].tolist()):
        if wt > 0:
            counts[g.feature_keys[f]] = counts.get(g.feature_keys[f], 0) + wt
    assert counts == {("n", "x"): 1, ("n", "y"): 2, ("n", "z"): 1, ("n", "w"): 1, ("node", "3"): 1}


def test_weighted_averaging_uses_edge_weights(tiny):
    g = data.load(*tiny)
    idx, w = sampling.neighbourhood_tokens(g, torch.tensor([0]), weighted_averaging=True)
    # a's neighbours: b (weight 1), c (weight 3).
    per_feature = {}
    for f, wt in zip(idx[0].tolist(), w[0].tolist()):
        if wt > 0:
            per_feature[g.feature_keys[f]] = per_feature.get(g.feature_keys[f], 0) + wt
    assert per_feature == {("n", "y"): 1, ("n", "z"): 1, ("n", "w"): 1, ("n", "x"): 3}


def test_negatives_from_a_pool_are_never_the_anchor():
    pool = torch.tensor([0, 1, 2, 3, 3, 4])
    nodes = torch.arange(5).repeat(100)
    pick = sampling.pick_from_pool(pool, nodes, 4, torch.Generator().manual_seed(0))
    assert (pool[pick] != nodes[:, None]).all()
    assert pick.min() >= 0 and pick.max() < len(pool)


def test_load_cached_reuses_the_parsed_graph(tiny, tmp_path):
    cache = str(tmp_path / "g.cache")
    first = data.load_cached(cache, *tiny)
    again = data.load_cached(cache, *tiny)
    assert again.node_names == first.node_names
    assert torch.equal(again.adj_idx, first.adj_idx) and torch.equal(again.feat_idx, first.feat_idx)
    # A different namespace isn't served from the cache.
    other = data.load_cached(cache, *tiny, namespace="feat")
    assert other.feature_keys != first.feature_keys


def test_neighbour_features_one_row_per_neighbour(tiny):
    g = data.load(*tiny)
    fidx, fmask, w = sampling.neighbour_features(g, torch.tensor([0, 3]))
    k = w.shape[1]
    # a's neighbours b and c; d's only neighbour is c, so its second slot is padding.
    rows_a = [{g.feature_keys[f] for f in fidx[i][fmask[i]].tolist()} for i in range(k)]
    assert sorted(map(sorted, rows_a)) == sorted(map(sorted, [features(g, 1), features(g, 2)]))
    assert w[1].tolist() == [1.0, 0.0]
    assert not fmask[k + 1].any(), "a padded neighbour contributes no features"
