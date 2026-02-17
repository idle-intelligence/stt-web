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
        let mut data = self.data.clone();
        let slice = data.as_slice_mut().unwrap();
        for v in slice.iter_mut() {
            if *v <= 0.0 {
                *v = alpha * (v.exp() - 1.0);
            }
        }
        Self { data }
    }

    /// Apply ELU activation in-place, avoiding allocation.
    pub fn elu_inplace(&mut self, alpha: f32) {
        let slice = self.data.as_slice_mut().unwrap();
        for v in slice.iter_mut() {
            if *v <= 0.0 {
                *v = alpha * (v.exp() - 1.0);
            }
        }
    }

    /// Slice along the time dimension.
    /// Always returns a contiguous (standard layout) array.
    pub fn slice_time(&self, start: usize, len: usize) -> Self {
        let sliced = self
            .data
            .slice_axis(Axis(2), ndarray::Slice::from(start..start + len));
        // as_standard_layout + to_owned ensures C-contiguous memory
        let data = sliced.as_standard_layout().to_owned();
        Self { data }
    }

    /// Concatenate along the time axis.
    /// Always returns a contiguous (standard layout) array.
    pub fn concat_time(tensors: &[&Self]) -> Self {
        let views: Vec<_> = tensors.iter().map(|t| t.data.view()).collect();
        let data = ndarray::concatenate(Axis(2), &views).expect("concat failed");
        // Ensure contiguous layout
        if data.is_standard_layout() {
            Self { data }
        } else {
            Self { data: data.as_standard_layout().to_owned() }
        }
    }

    /// Multiply each channel by a per-channel scale vector (in-place).
    /// scale shape: (channels,), self shape: (batch, channels, time)
    pub fn scale_channels_inplace(&mut self, scale: &Array1<f32>) {
        let (batch, channels, time) = self.shape();
        assert_eq!(scale.len(), channels);
        let slice = self.data.as_slice_mut().unwrap();
        let scale_s = scale.as_slice().unwrap();
        for b in 0..batch {
            let b_off = b * channels * time;
            for c in 0..channels {
                let s = scale_s[c];
                let c_off = b_off + c * time;
                for t in 0..time {
                    slice[c_off + t] *= s;
                }
            }
        }
    }

    /// In-place addition: self += other
    pub fn add_assign(&mut self, other: &Tensor3) {
        self.data += &other.data;
    }

    /// Transpose axes 1 and 2: (batch, channels, time) → (batch, time, channels)
    pub fn transpose_12(&self) -> Self {
        let (batch, d1, d2) = self.shape();
        let mut out = Array3::<f32>::zeros((batch, d2, d1));
        let src = self.data.as_slice().unwrap();
        let dst = out.as_slice_mut().unwrap();
        for b in 0..batch {
            let src_base = b * d1 * d2;
            let dst_base = b * d2 * d1;
            for i in 0..d1 {
                for j in 0..d2 {
                    dst[dst_base + j * d1 + i] = src[src_base + i * d2 + j];
                }
            }
        }
        Self { data: out }
    }

    /// Apply GELU activation element-wise.
    pub fn gelu(&self) -> Self {
        let data = self.data.mapv(|x| {
            // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            let c = (2.0_f32 / std::f32::consts::PI).sqrt();
            x * 0.5 * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
        });
        Self { data }
    }

    /// Softmax over the last dimension (axis 2), using raw slices for speed.
    pub fn softmax_last(&self) -> Self {
        let (batch, d1, d2) = self.shape();
        let mut out = self.data.clone();
        let slice = out.as_slice_mut().unwrap();
        for b in 0..batch {
            let b_off = b * d1 * d2;
            for i in 0..d1 {
                let row_off = b_off + i * d2;
                let row = &mut slice[row_off..row_off + d2];
                // Find max for numerical stability
                let mut max_val = f32::NEG_INFINITY;
                for &v in row.iter() {
                    if v > max_val {
                        max_val = v;
                    }
                }
                // Exp and sum
                let mut sum = 0.0f32;
                for v in row.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                // Normalize
                if sum > 0.0 {
                    let inv_sum = 1.0 / sum;
                    for v in row.iter_mut() {
                        *v *= inv_sum;
                    }
                }
            }
        }
        Self { data: out }
    }

    /// Batched matrix multiply: (batch, M, K) @ (batch, K, N) → (batch, M, N)
    pub fn matmul(&self, other: &Self) -> Self {
        let (b1, m, k1) = self.shape();
        let (b2, k2, n) = other.shape();
        assert_eq!(b1, b2, "batch size mismatch");
        assert_eq!(k1, k2, "inner dimension mismatch");
        let mut out = Array3::<f32>::zeros((b1, m, n));
        for b in 0..b1 {
            let a = self.data.index_axis(Axis(0), b);
            let b_mat = other.data.index_axis(Axis(0), b);
            let result = a.dot(&b_mat);
            out.index_axis_mut(Axis(0), b).assign(&result);
        }
        Self { data: out }
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
