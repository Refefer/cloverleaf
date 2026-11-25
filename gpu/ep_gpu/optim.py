"""Optimiser for the feature table: lazy, sparse updates with a choice of how much state to keep.

The table is the bulk of training memory, and Adam keeps two more copies of it.  Per feature, with
d dims and a float32 table:

  adam             full first and second moments              12d bytes  (torch's SparseAdam)
  rowwise_adam     full first moment, one second moment/row   8d + 4
  rowwise_adagrad  one accumulated squared gradient per row   4d + 4     (FBGEMM's and TorchRec's
                                                                          default for embeddings)

A bfloat16 table halves the 4d.  Its updates are computed in float32 and written back with
stochastic rounding, which keeps small steps on average instead of rounding them all away.

Only rows with a gradient are updated (and, for Adam, their moments decayed), as SparseAdam does.

EpochGradients sums a whole pass's gradients so one step can apply them at its end (cloverleaf's
update_frequency "epoch").
"""
import torch

KINDS = ("adam", "rowwise_adam", "rowwise_adagrad")


def stochastic_round_bf16(x, generator=None):
    """float32 -> bfloat16, rounding each value's magnitude up with probability equal to how far
    it lies toward the next representable value, so the result equals x in expectation."""
    bits = x.contiguous().view(torch.int32)
    noise = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32, generator=generator)
    # Adding below the kept 16 bits carries into them with exactly that probability; clearing the
    # low bits then makes the conversion exact.
    return ((bits + noise) & -65536).view(torch.float32).to(torch.bfloat16)


class TableOptimizers:
    """Several sparse tables (the feature embeddings, and their importance scalars) stepped as one,
    on the same learning rate."""

    def __init__(self, *optimizers):
        self.optimizers = optimizers

    @property
    def lr(self):
        return self.optimizers[0].lr

    @lr.setter
    def lr(self, value):
        for o in self.optimizers:
            o.lr = value

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad()

    def step(self):
        for o in self.optimizers:
            o.step()

    def state_bytes(self):
        return sum(o.state_bytes() for o in self.optimizers)


class SparseTableOptimizer:
    def __init__(self, weight, kind="adam", lr=0.01, betas=(0.9, 0.999), eps=1e-8, generator=None):
        if kind not in KINDS:
            raise ValueError(f"optimizer must be one of {', '.join(KINDS)}, got {kind!r}")
        n, d = weight.shape
        device = weight.device
        self.weight, self.kind, self.lr = weight, kind, lr
        self.beta1, self.beta2 = betas
        self.eps, self.steps, self.generator = eps, 0, generator
        self.m = torch.zeros(n, d, device=device) if kind != "rowwise_adagrad" else None
        self.v = torch.zeros(n, d, device=device) if kind == "adam" else torch.zeros(n, device=device)

    def zero_grad(self):
        self.weight.grad = None

    def state_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.m, self.v) if t is not None)

    @torch.no_grad()
    def step(self):
        grad = self.weight.grad
        if grad is None:
            return
        grad = grad.coalesce()
        self.steps += 1
        self._update(grad.indices()[0], grad.values().float())

    @torch.no_grad()
    def _update(self, rows, g):
        """Updates distinct `rows` with their float32 gradients `g`, as step `self.steps`."""
        if self.kind == "rowwise_adagrad":
            v = self.v[rows] + g.pow(2).mean(dim=1)
            self.v[rows] = v
            update = g / (v.sqrt() + self.eps)[:, None]
        else:
            assert self.m is not None
            m = self.m[rows] * self.beta1 + (1 - self.beta1) * g
            self.m[rows] = m
            if self.kind == "adam":
                v = self.v[rows] * self.beta2 + (1 - self.beta2) * g.pow(2)
                denom = (v / (1 - self.beta2 ** self.steps)).sqrt() + self.eps
            else:
                v = self.v[rows] * self.beta2 + (1 - self.beta2) * g.pow(2).mean(dim=1)
                denom = ((v / (1 - self.beta2 ** self.steps)).sqrt() + self.eps)[:, None]
            self.v[rows] = v
            update = m / (1 - self.beta1 ** self.steps) / denom
        new = self.weight[rows].float() - self.lr * update
        if self.weight.dtype == torch.bfloat16:
            new = stochastic_round_bf16(new, self.generator)
        self.weight[rows] = new.to(self.weight.dtype)


class EpochGradients:
    """Sums the feature table's sparse gradients over a pass; `apply` then takes one optimiser
    step with the sums of every row that received any, leaving untouched rows (and, for Adam,
    their moments) alone, as a batch step does.  Keeps a float32 table the size of the weights,
    on the same device, and applies in chunks of rows so the step's temporaries stay small."""

    def __init__(self, weight, chunk_rows=1 << 16):
        self.sums = torch.zeros(weight.shape, device=weight.device)
        self.touched = torch.zeros(weight.shape[0], dtype=torch.bool, device=weight.device)
        self.chunk_rows = chunk_rows

    def state_bytes(self):
        return self.sums.numel() * self.sums.element_size() + self.touched.numel()

    @torch.no_grad()
    def add(self, grad):
        """Adds a sparse gradient (the table's .grad after a backward pass)."""
        if grad is None:
            return
        grad = grad.coalesce()
        rows = grad.indices()[0]
        self.sums.index_add_(0, rows, grad.values().float())
        self.touched[rows] = True

    @torch.no_grad()
    def apply(self, optimizer):
        """One step of `optimizer` with the summed gradients, then resets the sums."""
        rows = self.touched.nonzero().squeeze(1)
        if len(rows) == 0:
            return
        optimizer.steps += 1
        for s in range(0, len(rows), self.chunk_rows):
            chunk = rows[s:s + self.chunk_rows]
            optimizer._update(chunk, self.sums[chunk])
            self.sums[chunk] = 0
        self.touched.zero_()
