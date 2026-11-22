import pytest
import torch

from ep_gpu.optim import SparseTableOptimizer, stochastic_round_bf16


def sparse_grad(n, d, rows, values):
    return torch.sparse_coo_tensor(torch.tensor([rows]), torch.tensor(values, dtype=torch.float32), (n, d),
                                   check_invariants=True)


def test_adam_matches_torch_sparse_adam():
    torch.manual_seed(0)
    ours = torch.nn.Parameter(torch.randn(6, 3))
    ref = torch.nn.Parameter(ours.detach().clone())
    opt = SparseTableOptimizer(ours, "adam", lr=0.1)
    ref_opt = torch.optim.SparseAdam([ref], lr=0.1)
    for step in range(5):
        rows = [step % 6, (step + 2) % 6]
        values = torch.randn(2, 3).tolist()
        ours.grad = sparse_grad(6, 3, rows, values)
        ref.grad = sparse_grad(6, 3, rows, values)
        opt.step()
        ref_opt.step()
        assert torch.allclose(ours, ref, atol=1e-6), step


def test_rowwise_adagrad_formula():
    w = torch.nn.Parameter(torch.zeros(3, 2))
    opt = SparseTableOptimizer(w, "rowwise_adagrad", lr=0.5)
    w.grad = sparse_grad(3, 2, [1], [[3.0, 4.0]])
    opt.step()
    # G = mean(9, 16) = 12.5; step = 0.5 * g / sqrt(12.5)
    expected = -0.5 * torch.tensor([3.0, 4.0]) / 12.5 ** 0.5
    assert torch.allclose(w[1], expected, atol=1e-6)
    assert torch.equal(w[0], torch.zeros(2)) and torch.equal(w[2], torch.zeros(2))


def test_rowwise_adam_first_step_scales_by_the_rows_rms():
    w = torch.nn.Parameter(torch.zeros(2, 2))
    opt = SparseTableOptimizer(w, "rowwise_adam", lr=0.1)
    w.grad = sparse_grad(2, 2, [0], [[3.0, 4.0]])
    opt.step()
    # Bias-corrected m = g and v = mean(g^2), so the step is lr * g / rms(g).
    expected = -0.1 * torch.tensor([3.0, 4.0]) / 12.5 ** 0.5
    assert torch.allclose(w[0], expected, atol=1e-5)


@pytest.mark.parametrize("kind,extra", [("adam", 2 * 6 * 4), ("rowwise_adam", 6 * 4 + 4), ("rowwise_adagrad", 4)])
def test_state_size_per_row(kind, extra):
    opt = SparseTableOptimizer(torch.nn.Parameter(torch.zeros(10, 6)), kind)
    assert opt.state_bytes() == 10 * extra


def test_unknown_optimizer_is_rejected():
    with pytest.raises(ValueError):
        SparseTableOptimizer(torch.nn.Parameter(torch.zeros(2, 2)), "sgd")


def test_stochastic_rounding_is_unbiased_and_lands_on_neighbours():
    x = torch.full((200000,), 1.0 + 2 ** -10)       # between bf16 neighbours 1 and 1 + 2^-7
    r = stochastic_round_bf16(x, torch.Generator().manual_seed(0)).float()
    assert set(r.unique().tolist()) == {1.0, 1.0 + 2 ** -7}
    assert abs(r.mean().item() - x[0].item()) < 2e-5
    neg = stochastic_round_bf16(-x, torch.Generator().manual_seed(1)).float()
    assert abs(neg.mean().item() + x[0].item()) < 2e-5


def test_bfloat16_table_keeps_small_updates_on_average():
    w = torch.nn.Parameter(torch.ones(1, 4000, dtype=torch.bfloat16))
    opt = SparseTableOptimizer(w, "rowwise_adagrad", lr=1e-3, generator=torch.Generator().manual_seed(0))
    for _ in range(10):
        w.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.ones(1, 4000, dtype=torch.bfloat16), (1, 4000))
        opt.step()
    # Each step moves 1e-3 / sqrt(k), far below bf16's spacing (2^-7) near 1, yet on average the
    # weights move by the full amount: sum_k 1e-3 / sqrt(k) for k = 1..10.
    expected = 1 - sum(1e-3 / k ** 0.5 for k in range(1, 11))
    assert abs(w.float().mean().item() - expected) < 5e-4
