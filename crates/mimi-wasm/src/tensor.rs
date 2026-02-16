//! Lightweight tensor abstraction for Mimi codec.
//!
//! Based on ndarray with shape (batch, channels, time) for 3D tensors.

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, ArrayView3, Axis};
use std::ops::{Add, Mul};

/// 3D tensor with shape (batch, channels, time).
#[derive(Clone, Debug)]
pub struct Tensor3 {
    pub data: Array3<f32>,
}

impl Tensor3 {
    /// Create a new tensor from a 3D array.
    pub fn new(data: Array3<f32>) -> Self {
        Self { data }
    }

    /// Create a tensor filled with zeros.
    pub fn zeros(shape: (usize, usize, usize)) -> Self {
        Self {
            data: Array3::zeros(shape),
        }
    }

    /// Get the shape as (batch, channels, time).
    pub fn shape(&self) -> (usize, usize, usize) {
        let s = self.data.shape();
        (s[0], s[1], s[2])
    }

    /// Get batch size.
    pub fn batch(&self) -> usize {
        self.data.shape()[0]
    }

    /// Get number of channels.
    pub fn channels(&self) -> usize {
        self.data.shape()[1]
    }

    /// Get time dimension.
    pub fn time(&self) -> usize {
        self.data.shape()[2]
    }

    /// Get a view of the tensor.
    pub fn view(&self) -> ArrayView3<f32> {
        self.data.view()
    }

    /// Apply activation function element-wise.
    pub fn elu(&self, alpha: f32) -> Self {
        let data = self.data.mapv(|x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) });
        Self { data }
    }

    /// Slice along the time dimension.
    pub fn slice_time(&self, start: usize, len: usize) -> Self {
        let sliced = self
            .data
            .slice_axis(Axis(2), ndarray::Slice::from(start..start + len))
            .to_owned();
        Self { data: sliced }
    }

    /// Concatenate along the time axis.
    pub fn concat_time(tensors: &[&Self]) -> Self {
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let data = ndarray::concatenate(Axis(2), &views).expect("concat failed");
        Self { data }
    }
}

impl Add for Tensor3 {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self {
            data: self.data + other.data,
        }
    }
}

impl Add for &Tensor3 {
    type Output = Tensor3;

    fn add(self, other: Self) -> Tensor3 {
        Tensor3 {
            data: &self.data + &other.data,
        }
    }
}

impl Mul<f32> for &Tensor3 {
    type Output = Tensor3;

    fn mul(self, scalar: f32) -> Tensor3 {
        Tensor3 {
            data: &self.data * scalar,
        }
    }
}

/// 2D tensor with shape (channels, time).
#[derive(Clone, Debug)]
pub struct Tensor2 {
    pub data: Array2<f32>,
}

impl Tensor2 {
    pub fn new(data: Array2<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(shape: (usize, usize)) -> Self {
        Self {
            data: Array2::zeros(shape),
        }
    }

    pub fn view(&self) -> ArrayView2<f32> {
        self.data.view()
    }
}

/// 1D tensor (vector).
#[derive(Clone, Debug)]
pub struct Tensor1 {
    pub data: Array1<f32>,
}

impl Tensor1 {
    pub fn new(data: Array1<f32>) -> Self {
        Self { data }
    }

    pub fn zeros(len: usize) -> Self {
        Self {
            data: Array1::zeros(len),
        }
    }

    pub fn view(&self) -> ArrayView1<f32> {
        self.data.view()
    }
}
