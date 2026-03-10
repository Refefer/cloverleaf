//! Helpers and utilities for Candle operations
//!
//! This module provides common tensor operations used throughout the cloverleaf codebase,
//! abstracting away some of Candle's boilerplate for cleaner code.

use candle_core::{Device, Result, Tensor, Var};

/// Create a scalar tensor from f32
#[inline]
pub fn scalar(value: f32, device: &Device) -> Tensor {
    Tensor::from_slice(&[value], 1usize, device).unwrap()
}

/// Create a tensor from a slice of f32
#[inline]
pub fn from_slice(values: &[f32], device: &Device) -> Tensor {
    Tensor::from_slice(values, values.len(), device).unwrap()
}

/// Create a variable from a slice of f32
#[inline]
pub fn var(values: &[f32], device: &Device) -> Var {
    Var::from_slice(values, values.len(), device).unwrap()
}

/// Compute Euclidean distance between two tensors
pub fn euclidean_distance(tensor1: &Tensor, tensor2: &Tensor) -> Result<Tensor> {
    let diff = tensor1.sub(tensor2)?;
    let squared = diff.powf(2.0)?;
    let sum = squared.sum_all()?;
    sum.sqrt()
}

/// Compute dot product of two tensors
pub fn dot(tensor1: &Tensor, tensor2: &Tensor) -> Result<Tensor> {
    tensor1.mul(tensor2)?.sum_all()
}

/// Compute L2 norm of a tensor
pub fn l2norm(tensor: &Tensor) -> Result<Tensor> {
    let squared = tensor.powf(2.0)?;
    let sum = squared.sum_all()?;
    sum.sqrt()
}

/// Compute inverse L2 normalization (unit vector)
pub fn il2norm(tensor: &Tensor) -> Result<Tensor> {
    let norm = l2norm(tensor)?;
    tensor.div(&norm)
}

/// Compute cosine similarity
pub fn cosine(tensor1: &Tensor, tensor2: &Tensor) -> Result<Tensor> {
    let norm1 = il2norm(tensor1)?;
    let norm2 = il2norm(tensor2)?;
    dot(&norm1, &norm2)
}

/// Compute log-sum-exp trick for numerical stability
pub fn log_sum_exp(tensor: &Tensor, dim: usize) -> Result<Tensor> {
    let max = tensor.max_all()?;
    let exp_max = max.exp();
    let exp_diff = tensor.sub(&max)?.exp()?;
    let sum_exp = exp_diff.sum(dim)?;
    let log_sum = sum_exp.log()?;
    log_sum.add(&max)
}

/// Compute sum of all elements
pub fn sum_all(tensor: &Tensor) -> Result<Tensor> {
    tensor.sum_all()
}

/// Extract tensor to Vec<f32>
pub fn to_vec(tensor: &Tensor) -> Result<Vec<f32>> {
    tensor.to_vec1::<f32>()
}

/// Get scalar value from tensor
pub fn to_scalar(tensor: &Tensor) -> Result<f32> {
    tensor.to_scalar()
}
