use std::ops::{Add, Div, Mul, Neg, Sub};
use std::sync::OnceLock;

use candle_core::{backprop::GradStore, Device, Tensor, Var};

fn unwrap_tensor(result: candle_core::Result<Tensor>, context: &str) -> Tensor {
    result.unwrap_or_else(|err| panic!("Candle autograd error in {context}: {err}"))
}

fn device() -> &'static Device {
    static DEVICE: OnceLock<Device> = OnceLock::new();
    DEVICE.get_or_init(|| {
        #[cfg(feature = "metal")]
        {
            return Device::new_metal(0).expect("Candle Metal device");
        }

        #[cfg(not(feature = "metal"))]
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
        let var = Var::from_slice(data, data.len(), device()).expect("Candle variable");
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

macro_rules! impl_scalar_lhs_via_constant_op {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait<ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: ANode) -> Self::Output {
                Constant::scalar(self) $op rhs
            }
        }

        impl $trait<&ANode> for f32 {
            type Output = ANode;

            fn $method(self, rhs: &ANode) -> Self::Output {
                Constant::scalar(self) $op rhs
            }
        }
    };
}

macro_rules! impl_unary_op {
    ($trait:ident, $method:ident, $op:expr, $context:literal) => {
        impl $trait for ANode {
            type Output = ANode;

            fn $method(self) -> Self::Output {
                scalar_op(&self, 0.0, $op, $context)
            }
        }

        impl $trait for &ANode {
            type Output = ANode;

            fn $method(self) -> Self::Output {
                scalar_op(self, 0.0, $op, $context)
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
impl_scalar_lhs_via_constant_op!(Div, div, /);

impl_unary_op!(Neg, neg, |lhs, _| lhs.affine(-1.0, 0.0), "neg");
