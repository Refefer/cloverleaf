"""The space node embeddings live in, and how two of them are compared.

The feature table and pooling are the same in every space: a node's features are pooled (mean or
attention) into one vector.  The space decides what that vector becomes and how embeddings are
scored:

  euclidean  the pooled vector is the embedding.  Losses score pairs by cosine similarity, and the
             margin loss by Euclidean distance, as cloverleaf does.
  poincare   the pooled vector is a tangent vector at the origin of the Poincaré ball of curvature
             -c, and the embedding is its exponential map there:
               exp0(v) = tanh(sqrt(c) |v|) v / (sqrt(c) |v|)
             Pairs are scored by negative hyperbolic distance,
               d(x, y) = arcosh(1 + 2c |x - y|^2 / ((1 - c|x|^2)(1 - c|y|^2))) / sqrt(c),
             and the margin loss uses that distance.  Every step is differentiable and the
             parameters stay Euclidean, so the usual optimisers apply unchanged (no Riemannian
             gradient rescaling or projection after updates).

             How features (and, nested, neighbours) pool, `pooling`:
               tangent   a weighted mean of the tangent vectors, mapped into the ball after: the
                         choice hyperbolic GCNs make for aggregation.  Works with attention.
               midpoint  each feature is mapped into the ball first, and they pool to their
                         weighted gyromidpoint (Ungar), the ball's own weighted mean:
                           m = 1/2 (x) sum_i w_i l_i x_i / sum_i w_i (l_i - 1),  l_i = 2 / (1 - c|x_i|^2)
                         ((x) Möbius scalar multiplication).  It respects hyperbolic geometry: on
                         a tree the midpoint of a node's children lies near the node, as geodesics
                         between siblings pass by their parent.  Mean pooling only.
               neighbour_midpoint
                         features pool as with tangent, so a node's words just place it, but a
                         nested reconstruction h~(v) is the gyromidpoint of its neighbours' own
                         embeddings: the "parent" of them.  A bag of many scattered words averages
                         towards the origin under midpoint pooling, where every long title would
                         look generic; neighbours are real points, and their midpoint is the
                         hierarchy EP should learn.  Needs the nested reconstruction.

             max_tangent_norm clips the pooled tangent vector's norm before exp0, as hyperbolic
             metric-learning models do: points then stay well inside the ball, away from the edge
             where gradients vanish and float32 loses precision.

  lorentz    the hyperboloid (Lorentz) model of the same hyperbolic space: points x in R^(d+1) with
             <x, x>_L = -1/c and x_0 > 0, where <x, y>_L = -x_0 y_0 + x_1 y_1 + ... + x_d y_d.  The
             pooled vector is a tangent vector at the origin o = (1/sqrt(c), 0, ..., 0), mapped in
             by
               exp0(v) = (cosh(2 sqrt(c) |v|) / sqrt(c), sinh(2 sqrt(c) |v|) v / (sqrt(c) |v|)),
             at the ball's scale: the ball's metric is twice the Euclidean one at its origin, so
             its exp0 takes v a distance 2|v| out, and so does this one.  Pairs are scored by
             negative hyperbolic distance
               d(x, y) = arcosh(-c <x, y>_L) / sqrt(c).
             The two models are isometric (x -> x_1..d / (1 + sqrt(c) x_0) maps the hyperboloid
             onto the ball, the origin to the origin), so with tangent pooling a Lorentz model
             and a Poincaré one compute the same distances from the same parameters, and the same
             settings (learning rate, gamma, max_tangent_norm) carry over.  They differ only
             numerically.  The ball loses precision in 1 - c|x|^2 towards its edge and clamps
             points at distance 2 artanh(1 - 1e-5) / sqrt(c), about 12.2 / sqrt(c), from the
             origin.  The hyperboloid has no edge, but its coordinates grow like cosh, and the
             inner product of two far-out points is a difference of huge, nearly equal terms:
             in float32 it's noise past a radius of about 5 / sqrt(c).  So its points are float64
             (only a batch's exist at a time; the parameters stay float32), which keeps distances
             accurate to a radius near 30 / sqrt(c), well beyond the ball's clamp.  Tangent
             pooling only; models export in Poincaré coordinates, which cloverleaf reads.

Pooling is written as sums, so it can run as a scatter over a batch's tokens: each token
contributes mean_terms(x) = (numerator, denominator), weighted, and mean_finish turns their sums
into the pooled vector.  to_space then maps that into the space (exp0 for tangent pooling; the
midpoint is already there).

Hyperbolic distance grows exponentially towards the ball's edge, so a tree's levels fit in few
dimensions, which Euclidean space needs many for.
"""
import math

import torch
import torch.nn.functional as F


class _ArithmeticMean:
    """Pooling as the weighted arithmetic mean of the vectors themselves."""
    pools_in_space = False          # features pool to a point of the space
    neighbours_in_space = False     # a nested reconstruction pools neighbours' points in the space

    def feature_points(self, rows):
        return rows

    def mean_terms(self, x):
        return x, torch.ones_like(x[..., :1])

    def mean_finish(self, num, den):
        return num / den.clamp(min=1e-12)


class Euclidean(_ArithmeticMean):
    name = "euclidean"

    def to_space(self, v):
        return v

    def distance(self, x, y):
        """Broadcasting over leading dimensions."""
        return torch.linalg.vector_norm(x - y, dim=-1)

    def pairwise_distance(self, x, pool):
        """(B, d) against (P, d) -> (B, P)."""
        return torch.cdist(x, pool)

    def origin_distance(self, x):
        return torch.linalg.vector_norm(x, dim=-1)

    def similarity(self, x, y):
        """Cosine similarity, broadcasting over leading dimensions."""
        return (F.normalize(x, dim=-1) * F.normalize(y, dim=-1)).sum(-1)

    def pairwise_similarity(self, x, pool):
        """(B, d) against (P, d) -> (B, P), one matrix multiply."""
        return F.normalize(x, dim=-1) @ F.normalize(pool, dim=-1).T


POINCARE_POOLING = ("tangent", "midpoint", "neighbour_midpoint")


class Poincare:
    name = "poincare"

    def __init__(self, curvature=1.0, pooling="tangent", max_tangent_norm=None, boundary=1e-5):
        """boundary: points are kept at least this far (relatively) inside the ball's edge,
        where float32 can no longer tell 1 - c|x|^2 from zero."""
        if not curvature > 0:
            raise ValueError(f"curvature must be > 0, got {curvature}")
        if pooling not in POINCARE_POOLING:
            raise ValueError(f"poincare pooling must be one of {', '.join(POINCARE_POOLING)}, got {pooling!r}")
        if max_tangent_norm is not None and not max_tangent_norm > 0:
            raise ValueError(f"max_tangent_norm must be > 0, got {max_tangent_norm}")
        self.pooling = pooling
        self.pools_in_space = pooling == "midpoint"
        self.neighbours_in_space = pooling != "tangent"
        self.max_tangent_norm = max_tangent_norm
        if not 0 < boundary < 1:
            raise ValueError(f"boundary must be in (0, 1), got {boundary}")
        self.c = float(curvature)
        self.sqrt_c = math.sqrt(self.c)
        # exp0 maps a tangent vector of norm s to radius tanh(sqrt(c) s) / sqrt(c); capping
        # sqrt(c) s here keeps the radius at most (1 - boundary) / sqrt(c).
        self.max_scaled_norm = math.atanh(1 - boundary)

    def exp0(self, v):
        """exp0(v), in float32, after clipping |v| to max_tangent_norm.  A zero vector maps to the
        origin with a zero gradient."""
        v = v.float()
        norm = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-30)
        if self.max_tangent_norm is not None:
            clipped = norm.clamp(max=self.max_tangent_norm)
            v, norm = v * (clipped / norm), clipped
        scaled = (self.sqrt_c * norm).clamp(max=self.max_scaled_norm)
        return torch.tanh(scaled) / (self.sqrt_c * norm) * v

    def to_space(self, v):
        """A pooled vector as a point of the ball: exp0 of a tangent mean; a midpoint already is."""
        return v.float() if self.pools_in_space else self.exp0(v)

    def feature_points(self, rows):
        return self.exp0(rows) if self.pools_in_space else rows

    def mean_terms(self, x):
        if not self.pools_in_space:
            return x, torch.ones_like(x[..., :1])
        lam = 2 / (1 - self.c * (x * x).sum(-1, keepdim=True)).clamp(min=1e-15)
        return lam * x, lam - 1

    def mean_finish(self, num, den):
        m = num / den.clamp(min=1e-12)
        if not self.pools_in_space:
            return m
        # 1/2 (x) m: halves m's hyperbolic distance from the origin.
        norm = torch.sqrt((m * m).sum(-1, keepdim=True) + 1e-30)
        scaled = (self.sqrt_c * norm).clamp(max=1 - 1e-7)
        return torch.tanh(0.5 * torch.atanh(scaled)) / (self.sqrt_c * norm) * m

    def origin_distance(self, x):
        """d(0, x) = 2 artanh(sqrt(c) |x|) / sqrt(c)."""
        norm = torch.sqrt((x * x).sum(-1) + 1e-30)
        return 2 * torch.atanh((self.sqrt_c * norm).clamp(max=1 - 1e-7)) / self.sqrt_c

    def ball_mean(self, x, w):
        """The weighted gyromidpoint of points x (B, T, d) with weights w (B, T), 0 for padding;
        rows without points give the origin."""
        x = x.float()
        lam = 2 / (1 - self.c * (x * x).sum(-1, keepdim=True)).clamp(min=1e-15)
        num = (lam * x * w[..., None]).sum(1)
        den = ((lam - 1) * w[..., None]).sum(1)
        m = num / den.clamp(min=1e-12)
        norm = torch.sqrt((m * m).sum(-1, keepdim=True) + 1e-30)
        scaled = (self.sqrt_c * norm).clamp(max=1 - 1e-7)
        return torch.tanh(0.5 * torch.atanh(scaled)) / (self.sqrt_c * norm) * m

    def log0(self, x):
        """The inverse of to_space: the tangent vector at the origin that maps to x."""
        x = x.float()
        norm = torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-30)
        scaled = (self.sqrt_c * norm).clamp(max=1 - 1e-7)
        return torch.atanh(scaled) / (self.sqrt_c * norm) * x

    def _arcosh_1p(self, u):
        """arcosh(1 + u) / sqrt(c) for u >= 0.  arcosh's slope is infinite at 1, so u is floored
        just above 0: there the clamp passes no gradient, rather than 0 * inf = nan for pairs
        that coincide (e.g. an anchor against itself in a masked-out pool slot)."""
        u = u.clamp(min=1e-9)
        return torch.log1p(u + torch.sqrt(u * (u + 2))) / self.sqrt_c

    def distance(self, x, y):
        """Broadcasting over leading dimensions."""
        diff2 = ((x - y) ** 2).sum(-1)
        denom = (1 - self.c * (x * x).sum(-1)) * (1 - self.c * (y * y).sum(-1))
        return self._arcosh_1p(2 * self.c * diff2 / denom.clamp(min=1e-15))

    def pairwise_distance(self, x, pool):
        """(B, d) against (P, d) -> (B, P), through one matrix multiply."""
        xx, yy = (x * x).sum(-1), (pool * pool).sum(-1)
        diff2 = (xx[:, None] + yy[None, :] - 2 * x @ pool.T).clamp(min=0)
        denom = (1 - self.c * xx)[:, None] * (1 - self.c * yy)[None, :]
        return self._arcosh_1p(2 * self.c * diff2 / denom.clamp(min=1e-15))

    def similarity(self, x, y):
        return -self.distance(x, y)

    def pairwise_similarity(self, x, pool):
        return -self.pairwise_distance(x, pool)


class Lorentz(_ArithmeticMean):
    """The hyperboloid model; see the module docstring.  Points have one more coordinate than the
    pooled tangent vectors: x_0 first."""
    name = "lorentz"
    pooling = "tangent"

    def __init__(self, curvature=1.0, max_tangent_norm=None, max_scaled_norm=40.0):
        """max_scaled_norm: 2 sqrt(c) |v| is capped here, far beyond any useful radius, so cosh
        and sinh stay finite in float32 (their squares too, which distances multiply)."""
        if not curvature > 0:
            raise ValueError(f"curvature must be > 0, got {curvature}")
        if max_tangent_norm is not None and not max_tangent_norm > 0:
            raise ValueError(f"max_tangent_norm must be > 0, got {max_tangent_norm}")
        self.c = float(curvature)
        self.sqrt_c = math.sqrt(self.c)
        self.max_tangent_norm = max_tangent_norm
        self.max_scaled_norm = max_scaled_norm

    def exp0(self, v):
        """exp0(v), in float64, after clipping |v| to max_tangent_norm: (..., d) -> (..., d+1).  A
        zero vector maps to the origin with a zero gradient."""
        v = v.double()
        norm = torch.sqrt((v * v).sum(-1, keepdim=True) + 1e-30)
        if self.max_tangent_norm is not None:
            clipped = norm.clamp(max=self.max_tangent_norm)
            v, norm = v * (clipped / norm), clipped
        scaled = (2 * self.sqrt_c * norm).clamp(max=self.max_scaled_norm)
        x0 = torch.cosh(scaled) / self.sqrt_c
        return torch.cat([x0, torch.sinh(scaled) / (self.sqrt_c * norm) * v], dim=-1)

    def to_space(self, v):
        return self.exp0(v)

    def log0(self, x):
        """The inverse of to_space: the tangent vector at the origin that maps to x."""
        xs = x[..., 1:].double()
        norm = torch.sqrt((xs * xs).sum(-1, keepdim=True) + 1e-30)
        return (torch.asinh(self.sqrt_c * norm) / (2 * self.sqrt_c * norm) * xs).float()

    def to_poincare(self, x):
        """The same points in the Poincaré ball of the same curvature (the isometry that keeps
        the origin): x_1..d / (1 + sqrt(c) x_0)."""
        x = x.double()
        return (x[..., 1:] / (1 + self.sqrt_c * x[..., :1])).float()

    def _arcosh(self, z):
        """arcosh(z) / sqrt(c) for z >= 1, as arcosh(1 + u) with u floored just above 0: there the
        clamp passes no gradient, rather than 0 * inf = nan for pairs that coincide."""
        u = (z - 1).clamp(min=1e-12)
        return (torch.log1p(u + torch.sqrt(u * (u + 2))) / self.sqrt_c).float()

    def origin_distance(self, x):
        """d(o, x) = arcosh(sqrt(c) x_0) / sqrt(c)."""
        return self._arcosh(self.sqrt_c * x[..., 0].double())

    def distance(self, x, y):
        """Broadcasting over leading dimensions."""
        x, y = x.double(), y.double()
        inner = -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(-1)
        return self._arcosh(-self.c * inner)

    def pairwise_distance(self, x, pool):
        """(B, d+1) against (P, d+1) -> (B, P), through one matrix multiply."""
        x, pool = x.double(), pool.double()
        inner = -x[:, :1] * pool[:, 0][None, :] + x[:, 1:] @ pool[:, 1:].T
        return self._arcosh(-self.c * inner)

    def similarity(self, x, y):
        return -self.distance(x, y)

    def pairwise_similarity(self, x, pool):
        return -self.pairwise_distance(x, pool)


EUCLIDEAN = Euclidean()
GEOMETRIES = ("euclidean", "poincare", "lorentz")


def make(name, curvature=1.0, pooling="tangent", max_tangent_norm=None):
    if name == "euclidean":
        return EUCLIDEAN
    if name == "poincare":
        return Poincare(curvature, pooling, max_tangent_norm)
    if name == "lorentz":
        if pooling != "tangent":
            raise ValueError("geometry 'lorentz' pools in the tangent space only")
        return Lorentz(curvature, max_tangent_norm)
    raise ValueError(f"geometry must be one of {', '.join(GEOMETRIES)}, got {name!r}")
