//! The space node embeddings live in, and how two of them compare.
//!
//! Feature embeddings and pooling are the same in every space: a node's features pool (mean,
//! weighted mean or attention) into one vector.  The geometry decides what that vector becomes:
//!
//! * [`Geometry::Euclidean`]: the pooled vector is the embedding, compared by cosine.
//! * [`Geometry::Poincare`]: the pooled vector is a tangent vector at the origin of the Poincaré
//!   ball of curvature -c, and the embedding is its exponential map there,
//!   `exp0(v) = tanh(sqrt(c) |v|) v / (sqrt(c) |v|)`.  Embeddings compare by hyperbolic distance,
//!   `d(x, y) = arcosh(1 + 2c |x - y|^2 / ((1 - c|x|^2)(1 - c|y|^2))) / sqrt(c)`, which grows
//!   exponentially towards the ball's edge, so hierarchies (general nodes near the origin,
//!   specific ones further out) fit in few dimensions.
//!
//! The arithmetic matches the GPU trainer's (`gpu/ep_gpu/geometry.py`), clamps included, so
//! embeddings trained on either side embed and score the same way here.
use crate::autograd::ANode;
use crate::error::Error;

/// A Poincaré ball's curvature magnitude c: the ball has curvature -c and radius 1 / sqrt(c).
/// Always finite and positive.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Curvature(f32);

impl Curvature {
    pub fn new(c: f32) -> Result<Self, Error> {
        if c.is_finite() && c > 0. {
            Ok(Curvature(c))
        } else {
            Err(Error::InvalidArgument(format!("curvature must be finite and positive, got {}", c)))
        }
    }

    pub fn value(&self) -> f32 {
        self.0
    }

    /// The hyperbolic distance between two points of the ball.
    pub fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        let c = self.0 as f64;
        let (mut diff2, mut xx, mut yy) = (0f64, 0f64, 0f64);
        for (a, b) in x.iter().zip(y.iter()) {
            let (a, b) = (*a as f64, *b as f64);
            diff2 += (a - b) * (a - b);
            xx += a * a;
            yy += b * b;
        }
        let denom = ((1. - c * xx) * (1. - c * yy)).max(1e-15);
        (arcosh_1p(2. * c * diff2 / denom) / c.sqrt()) as f32
    }
}

/// arcosh(1 + u), floored at u = 1e-9 as the training side is, where arcosh's slope is infinite.
fn arcosh_1p(u: f64) -> f64 {
    let u = u.max(1e-9);
    (u + (u * (u + 2.)).sqrt()).ln_1p()
}

/// The Poincaré ball an embedding space maps into: its curvature and, optionally, a cap on the
/// pooled tangent vector's norm before the map, which keeps points away from the edge where
/// gradients vanish.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PoincareBall {
    curvature: Curvature,
    max_tangent_norm: Option<f32>
}

impl PoincareBall {
    /// Points stay at least this far (relatively) inside the edge, where float32 can no longer
    /// tell 1 - c|x|^2 from zero.
    pub const BOUNDARY: f32 = 1e-5;

    pub fn new(curvature: Curvature, max_tangent_norm: Option<f32>) -> Result<Self, Error> {
        if let Some(r) = max_tangent_norm {
            if !(r.is_finite() && r > 0.) {
                return Err(Error::InvalidArgument(format!(
                    "max_tangent_norm must be finite and positive, got {}", r)))
            }
        }
        Ok(PoincareBall { curvature, max_tangent_norm })
    }

    pub fn curvature(&self) -> Curvature {
        self.curvature
    }

    pub fn max_tangent_norm(&self) -> Option<f32> {
        self.max_tangent_norm
    }

    /// The cap on sqrt(c) |v| that keeps exp0(v) inside the boundary.
    fn max_scaled_norm() -> f32 {
        (1. - Self::BOUNDARY).atanh()
    }

    /// exp0(v) in place, after clipping |v| to `max_tangent_norm`.  Zero stays at the origin.
    pub fn exp0(&self, v: &mut [f32]) {
        let sqrt_c = self.curvature.0.sqrt();
        let mut norm = (v.iter().map(|x| x * x).sum::<f32>() + 1e-30).sqrt();
        let mut scale = 1f32;
        if let Some(r) = self.max_tangent_norm {
            if norm > r {
                scale = r / norm;
                norm = r;
            }
        }
        let scaled = (sqrt_c * norm).min(Self::max_scaled_norm());
        let factor = scale * scaled.tanh() / (sqrt_c * norm);
        v.iter_mut().for_each(|x| *x *= factor);
    }

    /// exp0 on an autograd node, as [`PoincareBall::exp0`].
    pub fn exp0_node(&self, v: &ANode) -> ANode {
        let sqrt_c = self.curvature.0.sqrt();
        let mut norm = (v.pow(2f32).sum() + 1e-30f32).sqrt();
        let mut v = v.clone();
        if let Some(r) = self.max_tangent_norm {
            let clipped = norm.minimum(r);
            v = &v * &(&clipped / &norm);
            norm = clipped;
        }
        let scaled = (&norm * sqrt_c).minimum(Self::max_scaled_norm());
        &v * &(scaled.tanh() / (&norm * sqrt_c))
    }

    /// The hyperbolic distance on autograd nodes, as [`Curvature::distance`].
    pub fn distance_node(&self, x: &ANode, y: &ANode) -> ANode {
        let c = self.curvature.0;
        let diff2 = (x - y).pow(2f32).sum();
        let denom = ((1f32 - x.pow(2f32).sum() * c) * (1f32 - y.pow(2f32).sum() * c)).maximum(1e-15);
        let u = (diff2 * (2f32 * c) / denom).maximum(1e-9);
        let inner = (&u * &(&u + 2f32)).sqrt();
        (1f32 + u + inner).ln() / c.sqrt()
    }
}

/// The space node embeddings live in.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub enum Geometry {
    /// Pooled vectors are the embeddings, compared by cosine.
    #[default]
    Euclidean,

    /// Pooled vectors map into a Poincaré ball, compared by hyperbolic distance.
    Poincare(PoincareBall)
}

impl Geometry {
    /// "euclidean" (no curvature or clip) or "poincare" (curvature 1 unless given).
    pub fn new(name: &str, curvature: Option<f32>, max_tangent_norm: Option<f32>) -> Result<Self, Error> {
        match name {
            "euclidean" if curvature.is_none() && max_tangent_norm.is_none() => Ok(Geometry::Euclidean),
            "euclidean" => Err(Error::InvalidArgument(
                "curvature and max_tangent_norm apply only to geometry \"poincare\"".into())),
            "poincare" => Ok(Geometry::Poincare(PoincareBall::new(
                Curvature::new(curvature.unwrap_or(1.))?, max_tangent_norm)?)),
            other => Err(Error::InvalidArgument(format!(
                "geometry must be \"euclidean\" or \"poincare\", got {:?}", other)))
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Geometry::Euclidean => "euclidean",
            Geometry::Poincare(_) => "poincare"
        }
    }

    /// Maps a pooled vector into the space, in place.
    pub fn to_space(&self, v: &mut [f32]) {
        if let Geometry::Poincare(ball) = self {
            ball.exp0(v);
        }
    }

    /// Maps a pooled autograd node into the space.
    pub fn to_space_node(&self, v: ANode) -> ANode {
        match self {
            Geometry::Euclidean => v,
            Geometry::Poincare(ball) => ball.exp0_node(&v)
        }
    }

    /// The distance between two embeddings in the space: Euclidean or hyperbolic.
    pub fn distance(&self, x: &[f32], y: &[f32]) -> f32 {
        match self {
            Geometry::Euclidean => x.iter().zip(y.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt(),
            Geometry::Poincare(ball) => ball.curvature().distance(x, y)
        }
    }

    /// The distance node embeddings in this space are searched and ranked by.
    pub fn node_distance(&self) -> crate::distance::Distance {
        match self {
            Geometry::Euclidean => crate::distance::Distance::Cosine,
            Geometry::Poincare(ball) => crate::distance::Distance::Poincare(ball.curvature())
        }
    }
}

impl std::fmt::Display for Geometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Geometry::Euclidean => write!(f, "euclidean"),
            Geometry::Poincare(ball) => match ball.max_tangent_norm() {
                Some(r) => write!(f, "poincare<c={},max_tangent_norm={}>", ball.curvature().value(), r),
                None => write!(f, "poincare<c={}>", ball.curvature().value())
            }
        }
    }
}

#[cfg(test)]
mod geometry_tests {
    use super::*;
    use crate::autograd::Variable;

    fn ball(c: f32) -> PoincareBall {
        PoincareBall::new(Curvature::new(c).unwrap(), None).unwrap()
    }

    #[test]
    fn exp0_lands_in_the_ball_and_matches_its_autograd_version() {
        for c in [0.3f32, 1., 2.] {
            let b = ball(c);
            for scale in [0.01f32, 0.5, 3., 1e3] {
                let v: Vec<f32> = (0..6).map(|i| scale * (i as f32 - 2.5)).collect();
                let mut x = v.clone();
                b.exp0(&mut x);
                let norm = x.iter().map(|a| a * a).sum::<f32>().sqrt();
                assert!(norm < 1. / c.sqrt(), "c {} scale {}: |x| {}", c, scale, norm);
                let node = b.exp0_node(&Variable::new(v)).value();
                for (a, n) in x.iter().zip(node.iter()) {
                    assert!((a - n).abs() < 1e-5, "{} vs {}", a, n);
                }
            }
        }
    }

    #[test]
    fn distance_matches_the_known_radius_and_its_autograd_version() {
        // d(0, (r, 0)) = 2 artanh(r) at c = 1.
        let d = Curvature::new(1.).unwrap().distance(&[0., 0.], &[0.5, 0.]);
        assert!((d - 2. * 0.5f32.atanh()).abs() < 1e-5);
        let b = ball(0.7);
        let (x, y) = (vec![0.1f32, -0.4, 0.3], vec![-0.5f32, 0.2, 0.6]);
        let plain = b.curvature().distance(&x, &y);
        let node = b.distance_node(&Variable::new(x.clone()), &Variable::new(y.clone())).value()[0];
        assert!((plain - node).abs() < 1e-4, "{} vs {}", plain, node);
        assert!(b.curvature().distance(&x, &x) < 1e-3);
    }

    #[test]
    fn clipping_bounds_the_distance_from_the_origin() {
        let b = PoincareBall::new(Curvature::new(1.).unwrap(), Some(1.5)).unwrap();
        let mut x = vec![10f32, -20., 5.];
        b.exp0(&mut x);
        let d0 = b.curvature().distance(&[0., 0., 0.], &x);
        assert!((d0 - 3.).abs() < 1e-3, "d(0, exp0(v)) = 2 |v| at c = 1, clipped to 1.5: {}", d0);
    }

    #[test]
    fn invalid_settings_are_rejected() {
        assert!(Curvature::new(0.).is_err() && Curvature::new(f32::NAN).is_err());
        assert!(PoincareBall::new(Curvature::new(1.).unwrap(), Some(-1.)).is_err());
        assert!(Geometry::new("euclidean", Some(1.), None).is_err());
        assert!(Geometry::new("spherical", None, None).is_err());
        assert_eq!(Geometry::new("euclidean", None, None).unwrap(), Geometry::Euclidean);
        assert_eq!(Geometry::new("poincare", None, None).unwrap(), Geometry::Poincare(ball(1.)));
    }
}
