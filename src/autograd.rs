use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::OnceLock;

use candle_core::{backprop::GradStore, CpuStorage, CustomOp2, Device, Layout, Shape, Tensor, Var};

fn unwrap_tensor(result: candle_core::Result<Tensor>, context: &str) -> Tensor {
    result.unwrap_or_else(|err| panic!("Candle autograd error in {context}: {err}"))
}

fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        #[cfg(feature = "cuda")]
        {
            return Device::new_cuda(0).expect("Candle CUDA device");
        }

        #[cfg(all(not(feature = "cuda"), feature = "metal"))]
        {
            return Device::new_metal(0).expect("Candle Metal device");
        }

        #[cfg(all(not(feature = "cuda"), not(feature = "metal")))]
        {
            Device::Cpu
        }
    })
}


#[derive(Clone)]
pub struct ANode {
    tensor: Tensor,
}

impl ANode {
    fn from_tensor(tensor: Tensor) -> Self {
        Self { tensor }
    }

    fn as_tensor(&self) -> &Tensor {
        &self.tensor
    }

    /// The values, flattened in row-major order.
    pub fn value(&self) -> Vec<f32> {
        flatten(&self.tensor, "value")
    }

    /// The shape: empty for a scalar, [n] for a vector, [rows, cols] for a matrix.
    pub fn shape(&self) -> Vec<usize> {
        self.tensor.dims().to_vec()
    }

    /// Stacks equal-length vectors as the rows of a matrix.
    pub fn stack(rows: &[ANode]) -> ANode {
        let tensors: Vec<&Tensor> = rows.iter().map(|r| r.as_tensor()).collect();
        Self::from_tensor(unwrap_tensor(Tensor::stack(&tensors, 0), "stack"))
    }

    /// Matrix product of two matrices.  On the CPU it runs on the calling thread: candle's own
    /// matmul spreads every product over all cores, which for the small matrices of one node's
    /// forward pass, already run in parallel across nodes, costs far more than it saves.
    pub fn matmul(&self, rhs: &Self) -> Self {
        let product = if self.tensor.device().is_cpu() {
            self.tensor.contiguous().and_then(|a| rhs.tensor.contiguous().and_then(|b| a.apply_op2(&b, SerialMatMul)))
        } else {
            self.tensor.matmul(rhs.as_tensor())
        };
        Self::from_tensor(unwrap_tensor(product, "matmul"))
    }

    /// The transpose of a matrix.
    pub fn t(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.t(), "transpose"))
    }

    /// `len` entries from `start` along dimension `dim`.
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.narrow(dim, start, len), "narrow"))
    }

    /// The same values as a vector.
    pub fn flatten(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.flatten_all(), "flatten"))
    }

    /// Concatenates matrices with the same number of rows side by side.
    pub fn hcat(parts: &[ANode]) -> ANode {
        let tensors: Vec<&Tensor> = parts.iter().map(|p| p.as_tensor()).collect();
        Self::from_tensor(unwrap_tensor(Tensor::cat(&tensors, 1), "hcat"))
    }

    /// Softmax over each row of a matrix.  The row maxima are subtracted first for stability;
    /// they're constants to the gradient, which is unchanged by the shift.
    pub fn softmax_rows(&self) -> Self {
        let max = unwrap_tensor(self.tensor.max_keepdim(1), "softmax max").detach();
        let shifted = unwrap_tensor(self.tensor.broadcast_sub(&max), "softmax shift");
        let e = unwrap_tensor(shifted.exp(), "softmax exp");
        let z = unwrap_tensor(e.sum_keepdim(1), "softmax sum");
        Self::from_tensor(unwrap_tensor(e.broadcast_div(&z), "softmax div"))
    }

    pub fn slice(&self, start: usize, len: usize) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.narrow(0, start, len), "slice"))
    }

    pub fn dot(&self, rhs: &Self) -> Self {
        Self::from_tensor(unwrap_tensor(
            unwrap_tensor((&self.tensor).mul(rhs.as_tensor()), "dot mul").sum_all(),
            "dot sum",
        ))
    }

    pub fn sum(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.sum_all(), "sum"))
    }

    /// `self` to the power `exp`.  Squares and square roots, the distances' and norms' powers,
    /// use Candle's own ops: its general power goes through `powf` per element, many times
    /// slower than a multiply.
    pub fn pow(&self, exp: f32) -> Self {
        if exp == 2.0 {
            return self.sqr()
        }
        if exp == 0.5 {
            return self.sqrt()
        }
        Self::from_tensor(unwrap_tensor(self.tensor.powf(exp as f64), "pow"))
    }

    pub fn sqr(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.sqr(), "sqr"))
    }

    pub fn sqrt(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.sqrt(), "sqrt"))
    }

    pub fn exp(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.exp(), "exp"))
    }

    pub fn exp_approx(&self) -> Self {
        self.exp()
    }

    pub fn ln(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.log(), "ln"))
    }

    pub fn tanh(&self) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.tanh(), "tanh"))
    }

    pub fn maximum(&self, min: f32) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.maximum(min as f64), "maximum"))
    }

    /// Clamps from above: min(self, max), with no gradient where the clamp applies.
    pub fn minimum(&self, max: f32) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.minimum(max as f64), "minimum"))
    }
}

pub struct Variable;

impl Variable {
    #[cfg(test)]
    pub fn new(data: Vec<f32>) -> ANode {
        Self::pooled(data.as_slice())
    }

    pub fn pooled(data: &[f32]) -> ANode {
        let var = Var::from_slice(data, data.len(), device()).expect("Candle variable");
        ANode::from_tensor(var.into_inner())
    }

    /// A rows x cols matrix variable from row-major `data`.
    pub fn matrix(data: &[f32], rows: usize, cols: usize) -> ANode {
        let var = Var::from_slice(data, (rows, cols), device()).expect("Candle matrix variable");
        ANode::from_tensor(var.into_inner())
    }
}

pub struct Constant;

impl Constant {
    pub fn new(data: Vec<f32>) -> ANode {
        ANode::from_tensor(
            Tensor::from_slice(data.as_slice(), data.len(), device()).expect("Candle constant"),
        )
    }

    pub fn scalar(value: f32) -> ANode {
        ANode::from_tensor(
            Tensor::from_slice(&[value], 1, device()).expect("Candle scalar constant"),
        )
    }

    /// A rows x cols matrix constant from row-major `data`.
    pub fn matrix(data: &[f32], rows: usize, cols: usize) -> ANode {
        ANode::from_tensor(
            Tensor::from_slice(data, (rows, cols), device()).expect("Candle matrix constant"),
        )
    }
}

/// A single-threaded f32 matrix product, with its gradient, for small CPU matrices.
struct SerialMatMul;

fn serial_matmul(a: &Tensor, b: &Tensor) -> candle_core::Result<Tensor> {
    a.contiguous()?.apply_op2_no_bwd(&b.contiguous()?, &SerialMatMul)
}

impl CustomOp2 for SerialMatMul {
    fn name(&self) -> &'static str {
        "serial-matmul"
    }

    fn cpu_fwd(&self, s1: &CpuStorage, l1: &Layout, s2: &CpuStorage, l2: &Layout) -> candle_core::Result<(CpuStorage, Shape)> {
        let (m, k) = l1.shape().dims2()?;
        let (k2, n) = l2.shape().dims2()?;
        if k != k2 {
            candle_core::bail!("serial-matmul: ({}, {}) x ({}, {})", m, k, k2, n)
        }
        let slice = |s: &CpuStorage, l: &Layout| -> candle_core::Result<Vec<f32>> {
            let CpuStorage::F32(v) = s else { candle_core::bail!("serial-matmul needs f32") };
            match l.contiguous_offsets() {
                Some((start, end)) => Ok(v[start..end].to_vec()),
                None => candle_core::bail!("serial-matmul needs contiguous inputs")
            }
        };
        let (a, b) = (slice(s1, l1)?, slice(s2, l2)?);
        let mut out = vec![0f32; m * n];
        for (row, a_row) in out.chunks_exact_mut(n.max(1)).zip(a.chunks_exact(k.max(1))) {
            for (a_ip, b_row) in a_row.iter().zip(b.chunks_exact(n.max(1))) {
                row.iter_mut().zip(b_row).for_each(|(o, b_pj)| *o += a_ip * b_pj);
            }
        }
        Ok((CpuStorage::F32(out), Shape::from((m, n))))
    }

    fn bwd(&self, a: &Tensor, b: &Tensor, _res: &Tensor, grad: &Tensor) -> candle_core::Result<(Option<Tensor>, Option<Tensor>)> {
        Ok((Some(serial_matmul(grad, &b.t()?)?), Some(serial_matmul(&a.t()?, grad)?)))
    }
}

fn flatten(tensor: &Tensor, context: &str) -> Vec<f32> {
    match tensor.rank() {
        0 => vec![tensor.to_scalar::<f32>().unwrap_or_else(|e| panic!("Candle {context}: {e}"))],
        1 => tensor.to_vec1::<f32>().unwrap_or_else(|e| panic!("Candle {context}: {e}")),
        _ => unwrap_tensor(tensor.flatten_all(), context).to_vec1::<f32>()
            .unwrap_or_else(|e| panic!("Candle {context}: {e}"))
    }
}

pub struct Graph {
    grads: Option<GradStore>,
}

impl Graph {
    pub fn new() -> Self {
        Self { grads: None }
    }

    pub fn backward(&mut self, loss: &ANode) {
        self.grads = Some(loss.as_tensor().backward().expect("Candle backward"));
    }

    pub fn get_grad(&self, node: &ANode) -> Option<Vec<f32>> {
        self.grads
            .as_ref()
            .and_then(|grads| grads.get(node.as_tensor()))
            .map(|grad| flatten(grad, "gradient"))
    }

    pub fn print_graph(_loss: &ANode) {}
}

pub fn use_shared_pool(_enabled: bool) {}

pub trait ANodeVecOps {
    fn sum_all(self) -> ANode;
    fn concat(self) -> ANode;
}

impl ANodeVecOps for Vec<ANode> {
    fn sum_all(self) -> ANode {
        let mut it = self.into_iter();
        let first = it.next().unwrap_or_else(|| Constant::scalar(0.0));
        it.fold(first, |acc, node| acc + node)
    }

    fn concat(self) -> ANode {
        let tensors = self
            .iter()
            .map(|node| {
                if node.as_tensor().rank() == 0 {
                    unwrap_tensor(node.as_tensor().reshape(1), "concat scalar reshape")
                } else {
                    node.as_tensor().clone()
                }
            })
            .collect::<Vec<_>>();

        if tensors.is_empty() {
            return Constant::new(Vec::new());
        }

        ANode::from_tensor(unwrap_tensor(Tensor::cat(&tensors, 0), "concat"))
    }
}

fn binary_op(
    lhs: &ANode,
    rhs: &ANode,
    op: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>,
    context: &str,
) -> ANode {
    ANode::from_tensor(unwrap_tensor(op(lhs.as_tensor(), rhs.as_tensor()), context))
}

fn scalar_op(
    lhs: &ANode,
    rhs: f32,
    op: fn(&Tensor, f64) -> candle_core::Result<Tensor>,
    context: &str,
) -> ANode {
    ANode::from_tensor(unwrap_tensor(op(lhs.as_tensor(), rhs as f64), context))
}


impl Div<ANode> for f32 {
    type Output = ANode;

    fn div(self, rhs: ANode) -> Self::Output {
        Constant::scalar(self) / rhs
    }
}

impl Div<&ANode> for f32 {
    type Output = ANode;

    fn div(self, rhs: &ANode) -> Self::Output {
        Constant::scalar(self) / rhs
    }
}

impl Neg for ANode {
    type Output = ANode;

    fn neg(self) -> Self::Output {
        scalar_op(&self, 0.0, |lhs, _| lhs.affine(-1.0, 0.0), "neg")
    }
}

impl Neg for &ANode {
    type Output = ANode;

    fn neg(self) -> Self::Output {
        scalar_op(self, 0.0, |lhs, _| lhs.affine(-1.0, 0.0), "neg")
    }
}


macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $tensor_op:path, $context:literal) => {
        impl $trait for ANode {
            type Output = ANode;

            fn $method(self, rhs: Self) -> Self::Output {
                binary_op(&self, &rhs, $tensor_op, $context)
            }
        }

        impl $trait<&ANode> for ANode {
            type Output = ANode;

            fn $method(self, rhs: &ANode) -> Self::Output {
                binary_op(&self, rhs, $tensor_op, $context)
            }
        }

        impl $trait<ANode> for &ANode {
            type Output = ANode;

            fn $method(self, rhs: ANode) -> Self::Output {
                binary_op(self, &rhs, $tensor_op, $context)
            }
        }

        impl $trait<&ANode> for &ANode {
            type Output = ANode;

            fn $method(self, rhs: &ANode) -> Self::Output {
                binary_op(self, rhs, $tensor_op, $context)
            }
        }
    };
}

macro_rules! impl_scalar_rhs_op {
    ($trait:ident, $method:ident, $op:expr, $context:literal) => {
        impl $trait<f32> for ANode {
            type Output = ANode;

            fn $method(self, rhs: f32) -> Self::Output {
                scalar_op(&self, rhs, $op, $context)
            }
        }

        impl $trait<f32> for &ANode {
            type Output = ANode;

            fn $method(self, rhs: f32) -> Self::Output {
                scalar_op(self, rhs, $op, $context)
            }
        }
    };
}

macro_rules! impl_commutative_scalar_lhs_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: ANode) -> Self::Output {
                rhs $op self
            }
        }

        impl $trait<&ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: &ANode) -> Self::Output {
                rhs $op self
            }
        }
    };
}

macro_rules! impl_scalar_lhs_op {
    ($trait:ident, $method:ident, $op:expr, $context:literal) => {
        impl $trait<ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: ANode) -> Self::Output {
                scalar_op(&rhs, self, $op, $context)
            }
        }

        impl $trait<&ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: &ANode) -> Self::Output {
                scalar_op(rhs, self, $op, $context)
            }
        }
    };
}

impl_binary_op!(Add, add, Tensor::broadcast_add, "add");
impl_scalar_rhs_op!(Add, add, |lhs, s| lhs.affine(1.0, s), "add scalar");
impl_commutative_scalar_lhs_op!(Add, add, +);

impl_binary_op!(Sub, sub, Tensor::broadcast_sub, "sub");
impl_scalar_rhs_op!(Sub, sub, |lhs, s| lhs.affine(1.0, -s), "sub scalar");
impl_scalar_lhs_op!(Sub, sub, |lhs, s| lhs.affine(-1.0, s), "scalar sub");

impl_binary_op!(Mul, mul, Tensor::broadcast_mul, "mul");
impl_scalar_rhs_op!(Mul, mul, |lhs, s| lhs.affine(s, 0.0), "mul scalar");
impl_commutative_scalar_lhs_op!(Mul, mul, *);

impl_binary_op!(Div, div, Tensor::broadcast_div, "div");
impl_scalar_rhs_op!(Div, div, |lhs, s| lhs.affine(1.0 / s, 0.0), "div scalar");

#[cfg(test)]
mod pow_tests {
    use super::*;

    /// Squares and square roots take the fast path; values and gradients match the general
    /// power.
    #[test]
    fn test_pow_fast_path_matches_powf() {
        let data = [0.25f32, 1.5, 3.0, 7.75];
        for exp in [2f32, 0.5] {
            let fast = Variable::new(data.to_vec());
            let general = Variable::new(data.to_vec());
            let general_out = ANode::from_tensor(unwrap_tensor(general.tensor.powf(exp as f64), "pow"));
            let fast_out = fast.pow(exp);
            for (a, b) in fast_out.value().iter().zip(general_out.value().iter()) {
                assert!((a - b).abs() <= 1e-6 * b.abs().max(1.), "pow({}) {} vs {}", exp, a, b);
            }
            let (mut gf, mut gg) = (Graph::new(), Graph::new());
            gf.backward(&fast_out.sum());
            gg.backward(&general_out.sum());
            let (df, dg) = (gf.get_grad(&fast).unwrap(), gg.get_grad(&general).unwrap());
            for (a, b) in df.iter().zip(dg.iter()) {
                assert!((a - b).abs() <= 1e-5 * b.abs().max(1.), "d pow({}) {} vs {}", exp, a, b);
            }
        }
        // Other exponents still work.
        assert!((Variable::new(vec![2.0]).pow(3.0).value()[0] - 8.0).abs() < 1e-5);
    }
}

#[cfg(test)]
mod matrix_op_tests {
    use super::*;

    /// Checks the gradient of `f` (a scalar function of one matrix variable) against central
    /// finite differences.
    fn check_gradient(data: &[f32], rows: usize, cols: usize, f: impl Fn(&ANode) -> ANode) {
        let x = Variable::matrix(data, rows, cols);
        let mut graph = Graph::new();
        graph.backward(&f(&x));
        let grad = graph.get_grad(&x).expect("a gradient");
        let h = 1e-2f32;
        for i in 0..data.len() {
            let (mut up, mut down) = (data.to_vec(), data.to_vec());
            up[i] += h;
            down[i] -= h;
            let numeric = (f(&Constant::matrix(&up, rows, cols)).value()[0]
                - f(&Constant::matrix(&down, rows, cols)).value()[0]) / (2. * h);
            assert!((grad[i] - numeric).abs() < 2e-3, "entry {}: autograd {} vs numeric {}", i, grad[i], numeric);
        }
    }

    #[test]
    fn test_matmul_transpose_and_narrow() {
        let a = Constant::matrix(&[1., 2., 3., 4., 5., 6.], 2, 3);
        let b = Constant::matrix(&[1., 0., 0., 1., 1., 1.], 3, 2);
        assert_eq!(a.matmul(&b).value(), vec![4., 5., 10., 11.]);
        assert_eq!(a.t().shape(), vec![3, 2]);
        assert_eq!(a.narrow(1, 1, 2).value(), vec![2., 3., 5., 6.]);
        assert_eq!(ANode::hcat(&[a.narrow(1, 0, 1), a.narrow(1, 2, 1)]).value(), vec![1., 3., 4., 6.]);
        assert_eq!(ANode::stack(&[Constant::new(vec![1., 2.]), Constant::new(vec![3., 4.])]).shape(), vec![2, 2]);
        let w = Constant::matrix(&[0.5, -1., 2., 0.25, 1., -0.5], 3, 2);
        check_gradient(&[0.3, -0.7, 1.1, 0.2, 0.9, -0.4], 2, 3, |x| x.matmul(&w).pow(2.).sum());
        check_gradient(&[0.3, -0.7, 1.1, 0.2, 0.9, -0.4], 2, 3, |x| x.t().matmul(x).narrow(0, 1, 2).sum());
    }

    /// The serial product equals candle's, including on transposed (non-contiguous) inputs.
    #[test]
    fn test_serial_matmul_matches_candle() {
        let a: Vec<f32> = (0..7 * 13).map(|i| ((i * 37 % 11) as f32 - 5.) / 3.).collect();
        let b: Vec<f32> = (0..13 * 5).map(|i| ((i * 17 % 7) as f32 - 3.) / 2.).collect();
        let (ta, tb) = (Constant::matrix(&a, 7, 13), Constant::matrix(&b, 13, 5));
        let expected = ta.as_tensor().matmul(tb.as_tensor()).unwrap().flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(ta.matmul(&tb).value().iter().zip(expected.iter()).all(|(x, y)| (x - y).abs() < 1e-4));
        let expected_t = tb.as_tensor().t().unwrap().matmul(&ta.as_tensor().t().unwrap()).unwrap()
            .flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert!(tb.t().matmul(&ta.t()).value().iter().zip(expected_t.iter()).all(|(x, y)| (x - y).abs() < 1e-4));
    }

    #[test]
    fn test_softmax_rows() {
        let s = Constant::matrix(&[1., 2., 3., 1000., 1000., 1000.], 2, 3).softmax_rows().value();
        let e = [1f32.exp(), 2f32.exp(), 3f32.exp()];
        let z: f32 = e.iter().sum();
        for i in 0..3 {
            assert!((s[i] - e[i] / z).abs() < 1e-6);
            assert!((s[3 + i] - 1. / 3.).abs() < 1e-6, "large logits stay finite");
        }
        let target = Constant::matrix(&[0.1, 0.5, -0.3, 0.7, -0.2, 0.4], 2, 3);
        check_gradient(&[0.3, -0.7, 1.1, 0.2, 0.9, -0.4], 2, 3, |x| (x.softmax_rows() * &target).sum());
    }

    #[test]
    fn test_flattened_values_and_gradients() {
        let x = Variable::matrix(&[1., 2., 3., 4.], 2, 2);
        assert_eq!(x.value(), vec![1., 2., 3., 4.]);
        let mut graph = Graph::new();
        graph.backward(&(&x * 3f32).sum());
        assert_eq!(graph.get_grad(&x).unwrap(), vec![3.; 4]);
    }
}
