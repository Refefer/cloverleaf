use std::ops::{Add, Div, Mul, Neg, Sub};

use candle_core::{backprop::GradStore, Device, Tensor, Var};

fn unwrap_tensor(result: candle_core::Result<Tensor>, context: &str) -> Tensor {
    result.unwrap_or_else(|err| panic!("Candle autograd error in {context}: {err}"))
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

    pub fn value(&self) -> Vec<f32> {
        if self.tensor.rank() == 0 {
            vec![self.tensor.to_scalar::<f32>().expect("Candle scalar value")]
        } else {
            self.tensor.to_vec1::<f32>().expect("Candle 1d value")
        }
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

    pub fn pow(&self, exp: f32) -> Self {
        Self::from_tensor(unwrap_tensor(self.tensor.powf(exp as f64), "pow"))
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
}

pub struct Variable;

impl Variable {
    #[cfg(test)]
    pub fn new(data: Vec<f32>) -> ANode {
        Self::pooled(data.as_slice())
    }

    pub fn pooled(data: &[f32]) -> ANode {
        let var = Var::from_slice(data, data.len(), &Device::Cpu).expect("Candle variable");
        ANode::from_tensor(var.into_inner())
    }
}

pub struct Constant;

impl Constant {
    pub fn new(data: Vec<f32>) -> ANode {
        ANode::from_tensor(
            Tensor::from_slice(data.as_slice(), data.len(), &Device::Cpu).expect("Candle constant"),
        )
    }

    pub fn scalar(value: f32) -> ANode {
        ANode::from_tensor(
            Tensor::from_slice(&[value], 1, &Device::Cpu).expect("Candle scalar constant"),
        )
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
            .map(|grad| {
                if grad.rank() == 0 {
                    vec![grad.to_scalar::<f32>().expect("Candle scalar gradient")]
                } else {
                    grad.to_vec1::<f32>().expect("Candle 1d gradient")
                }
            })
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

fn binary_op(lhs: &ANode, rhs: &ANode, op: fn(&Tensor, &Tensor) -> candle_core::Result<Tensor>, context: &str) -> ANode {
    ANode::from_tensor(unwrap_tensor(op(lhs.as_tensor(), rhs.as_tensor()), context))
}

fn scalar_op(lhs: &ANode, rhs: f32, op: fn(&Tensor, f64) -> candle_core::Result<Tensor>, context: &str) -> ANode {
    ANode::from_tensor(unwrap_tensor(op(lhs.as_tensor(), rhs as f64), context))
}

impl Add for ANode {
    type Output = ANode;

    fn add(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, Tensor::broadcast_add, "add")
    }
}

impl Add<&ANode> for ANode {
    type Output = ANode;

    fn add(self, rhs: &ANode) -> Self::Output {
        binary_op(&self, rhs, Tensor::broadcast_add, "add")
    }
}

impl Add<ANode> for &ANode {
    type Output = ANode;

    fn add(self, rhs: ANode) -> Self::Output {
        binary_op(self, &rhs, Tensor::broadcast_add, "add")
    }
}

impl Add<&ANode> for &ANode {
    type Output = ANode;

    fn add(self, rhs: &ANode) -> Self::Output {
        binary_op(self, rhs, Tensor::broadcast_add, "add")
    }
}

impl Add<f32> for ANode {
    type Output = ANode;

    fn add(self, rhs: f32) -> Self::Output {
        scalar_op(&self, rhs, |lhs, s| lhs.affine(1.0, s), "add scalar")
    }
}

impl Add<f32> for &ANode {
    type Output = ANode;

    fn add(self, rhs: f32) -> Self::Output {
        scalar_op(self, rhs, |lhs, s| lhs.affine(1.0, s), "add scalar")
    }
}

impl Add<ANode> for f32 {
    type Output = ANode;

    fn add(self, rhs: ANode) -> Self::Output {
        rhs + self
    }
}

impl Add<&ANode> for f32 {
    type Output = ANode;

    fn add(self, rhs: &ANode) -> Self::Output {
        rhs + self
    }
}

impl Sub for ANode {
    type Output = ANode;

    fn sub(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, Tensor::broadcast_sub, "sub")
    }
}

impl Sub<&ANode> for ANode {
    type Output = ANode;

    fn sub(self, rhs: &ANode) -> Self::Output {
        binary_op(&self, rhs, Tensor::broadcast_sub, "sub")
    }
}

impl Sub<ANode> for &ANode {
    type Output = ANode;

    fn sub(self, rhs: ANode) -> Self::Output {
        binary_op(self, &rhs, Tensor::broadcast_sub, "sub")
    }
}

impl Sub<&ANode> for &ANode {
    type Output = ANode;

    fn sub(self, rhs: &ANode) -> Self::Output {
        binary_op(self, rhs, Tensor::broadcast_sub, "sub")
    }
}

impl Sub<f32> for ANode {
    type Output = ANode;

    fn sub(self, rhs: f32) -> Self::Output {
        scalar_op(&self, rhs, |lhs, s| lhs.affine(1.0, -s), "sub scalar")
    }
}

impl Sub<f32> for &ANode {
    type Output = ANode;

    fn sub(self, rhs: f32) -> Self::Output {
        scalar_op(self, rhs, |lhs, s| lhs.affine(1.0, -s), "sub scalar")
    }
}

impl Sub<ANode> for f32 {
    type Output = ANode;

    fn sub(self, rhs: ANode) -> Self::Output {
        scalar_op(&rhs, self, |lhs, s| lhs.affine(-1.0, s), "scalar sub")
    }
}

impl Sub<&ANode> for f32 {
    type Output = ANode;

    fn sub(self, rhs: &ANode) -> Self::Output {
        scalar_op(rhs, self, |lhs, s| lhs.affine(-1.0, s), "scalar sub")
    }
}

impl Mul for ANode {
    type Output = ANode;

    fn mul(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, Tensor::broadcast_mul, "mul")
    }
}

impl Mul<&ANode> for ANode {
    type Output = ANode;

    fn mul(self, rhs: &ANode) -> Self::Output {
        binary_op(&self, rhs, Tensor::broadcast_mul, "mul")
    }
}

impl Mul<ANode> for &ANode {
    type Output = ANode;

    fn mul(self, rhs: ANode) -> Self::Output {
        binary_op(self, &rhs, Tensor::broadcast_mul, "mul")
    }
}

impl Mul<&ANode> for &ANode {
    type Output = ANode;

    fn mul(self, rhs: &ANode) -> Self::Output {
        binary_op(self, rhs, Tensor::broadcast_mul, "mul")
    }
}

impl Mul<f32> for ANode {
    type Output = ANode;

    fn mul(self, rhs: f32) -> Self::Output {
        scalar_op(&self, rhs, |lhs, s| lhs.affine(s, 0.0), "mul scalar")
    }
}

impl Mul<f32> for &ANode {
    type Output = ANode;

    fn mul(self, rhs: f32) -> Self::Output {
        scalar_op(self, rhs, |lhs, s| lhs.affine(s, 0.0), "mul scalar")
    }
}

impl Mul<ANode> for f32 {
    type Output = ANode;

    fn mul(self, rhs: ANode) -> Self::Output {
        rhs * self
    }
}

impl Mul<&ANode> for f32 {
    type Output = ANode;

    fn mul(self, rhs: &ANode) -> Self::Output {
        rhs * self
    }
}

impl Div for ANode {
    type Output = ANode;

    fn div(self, rhs: Self) -> Self::Output {
        binary_op(&self, &rhs, Tensor::broadcast_div, "div")
    }
}

impl Div<&ANode> for ANode {
    type Output = ANode;

    fn div(self, rhs: &ANode) -> Self::Output {
        binary_op(&self, rhs, Tensor::broadcast_div, "div")
    }
}

impl Div<ANode> for &ANode {
    type Output = ANode;

    fn div(self, rhs: ANode) -> Self::Output {
        binary_op(self, &rhs, Tensor::broadcast_div, "div")
    }
}

impl Div<&ANode> for &ANode {
    type Output = ANode;

    fn div(self, rhs: &ANode) -> Self::Output {
        binary_op(self, rhs, Tensor::broadcast_div, "div")
    }
}

impl Div<f32> for ANode {
    type Output = ANode;

    fn div(self, rhs: f32) -> Self::Output {
        scalar_op(&self, rhs, |lhs, s| lhs.affine(1.0 / s, 0.0), "div scalar")
    }
}

impl Div<f32> for &ANode {
    type Output = ANode;

    fn div(self, rhs: f32) -> Self::Output {
        scalar_op(self, rhs, |lhs, s| lhs.affine(1.0 / s, 0.0), "div scalar")
    }
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
