"""Keeping the graph and the feature table in host memory: same results, on CUDA."""
import pytest
import torch

from ep_gpu import data
from ep_gpu.model import EPModel
from ep_gpu.train import Config, node_embeddings, train

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def two_cliques(tmp_path):
    lines = [f"n\t{c * 10 + a}\tn\t{c * 10 + b}\t1" for c in range(2) for a in range(10) for b in range(a + 1, 10)]
    lines.append("n\t0\tn\t10\t1")
    path = tmp_path / "cliques.edges"
    path.write_text("\n".join(lines) + "\n")
    return data.load(str(path))


@cuda
@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_table_in_host_memory_gives_the_same_embeddings_and_gradients(pooling):
    torch.manual_seed(0)
    on_gpu = EPModel(30, 16, pooling=pooling, heads=2, seed=0).cuda()
    offloaded = EPModel(30, 16, pooling=pooling, heads=2, seed=0).cuda()
    offloaded.load_state_dict(on_gpu.state_dict())
    offloaded.features.cpu()
    offloaded.device = "cuda"
    f = torch.randint(0, 30, (4, 5), device="cuda")
    w = torch.rand(4, 5, device="cuda")
    outs = []
    for m in (on_gpu, offloaded):
        out = m.embed(f, w)
        (out ** 2).sum().backward()
        outs.append((out.detach().cpu(), m.features.weight.grad.to_dense().cpu()))
    assert offloaded.features.weight.device.type == "cpu"
    assert torch.allclose(outs[0][0], outs[1][0], atol=1e-5)
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-5)


@cuda
@pytest.mark.parametrize("pooling", ["mean", "attention"])
def test_training_with_graph_and_table_in_host_memory(two_cliques, pooling):
    cfg = Config(dims=16, passes=60, batch_size=8, lr=0.05, negatives=3, pooling=pooling, heads=2,
                 graph_device="cpu", table_device="cpu", optimizer="rowwise_adagrad",
                 table_dtype="bfloat16", device="cuda", seed=3)
    model = train(two_cliques, cfg, log=lambda m: None)
    assert model.features.weight.device.type == "cpu" and model.features.weight.is_pinned()
    e = torch.nn.functional.normalize(node_embeddings(model, two_cliques).float(), dim=1)
    d = 1 - e @ e.T
    within = (d[:10, :10].sum() + d[10:, 10:].sum()) / 180
    assert float(d[:10, 10:].mean() - within) > 0.3


def test_unknown_placement_is_rejected(two_cliques):
    with pytest.raises(ValueError):
        train(two_cliques, Config(dims=4, passes=1, table_device="disk", device="cpu"), log=lambda m: None)
