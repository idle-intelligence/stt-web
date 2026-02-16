//! 1D convolution operations for audio processing.
//!
//! Implements Conv1d and ConvTranspose1d with streaming support.

use crate::tensor::{Tensor1, Tensor3};
use ndarray::{Array1, Array2, Array3};

/// 1D convolution layer.
#[derive(Clone, Debug)]
pub struct Conv1d {
    /// Weight tensor: (out_channels, in_channels, kernel_size)
    pub weight: Array3<f32>,
    /// Bias vector: (out_channels,)
    pub bias: Option<Array1<f32>>,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub causal: bool,
    /// Buffered input for streaming (per batch).
    buffer: Vec<Array2<f32>>, // Vec of (in_channels, buffer_time)
}

impl Conv1d {
    /// Create a new Conv1d layer.
    pub fn new(
        weight: Array3<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        dilation: usize,
        groups: usize,
        causal: bool,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            dilation,
            groups,
            causal,
            buffer: Vec::new(),
        }
    }

    /// Get output channels.
    pub fn out_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    /// Get input channels.
    pub fn in_channels(&self) -> usize {
        self.weight.shape()[1]
    }

    /// Get kernel size.
    pub fn kernel_size(&self) -> usize {
        self.weight.shape()[2]
    }

    /// Calculate padding for causal convolution.
    fn padding(&self) -> usize {
        if self.causal {
            (self.kernel_size() - 1) * self.dilation
        } else {
            ((self.kernel_size() - 1) * self.dilation) / 2
        }
    }

    /// Forward pass (non-streaming).
    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        let (batch, in_c, time) = input.shape();
        assert_eq!(in_c, self.in_channels(), "Input channels mismatch");

        let padding = self.padding();
        let out_time = (time + padding - self.dilation * (self.kernel_size() - 1)) / self.stride;
        let mut output = Array3::<f32>::zeros((batch, self.out_channels(), out_time));

        // Simplified convolution (not optimized, but correct for small models)
        for b in 0..batch {
            for out_c in 0..self.out_channels() {
                for t_out in 0..out_time {
                    let mut sum = 0.0;
                    let t_in_start = t_out * self.stride;

                    for in_c in 0..self.in_channels() {
                        for k in 0..self.kernel_size() {
                            let t_in = t_in_start + k * self.dilation;
                            if t_in >= padding && t_in - padding < time {
                                let input_val = input.data[[b, in_c, t_in - padding]];
                                let weight_val = self.weight[[out_c, in_c, k]];
                                sum += input_val * weight_val;
                            }
                        }
                    }

                    if let Some(ref bias) = self.bias {
                        sum += bias[out_c];
                    }
                    output[[b, out_c, t_out]] = sum;
                }
            }
        }

        Tensor3::new(output)
    }

    /// Initialize streaming buffer for a given batch size.
    pub fn init_buffer(&mut self, batch_size: usize) {
        let buffer_len = if self.causal { self.padding() } else { 0 };
        self.buffer = (0..batch_size)
            .map(|_| Array2::zeros((self.in_channels(), buffer_len)))
            .collect();
    }

    /// Streaming forward (feed incremental data).
    pub fn step(&mut self, input: &Tensor3) -> Option<Tensor3> {
        // For now, just use the non-streaming version
        // A full streaming implementation would buffer and emit partial outputs
        Some(self.forward(input))
    }

    /// Reset streaming state.
    pub fn reset(&mut self) {
        for buf in &mut self.buffer {
            buf.fill(0.0);
        }
    }
}

/// 1D transposed convolution (upsampling).
#[derive(Clone, Debug)]
pub struct ConvTranspose1d {
    /// Weight tensor: (in_channels, out_channels, kernel_size)
    pub weight: Array3<f32>,
    /// Bias vector: (out_channels,)
    pub bias: Option<Array1<f32>>,
    pub stride: usize,
    pub causal: bool,
    buffer: Vec<Array2<f32>>,
}

impl ConvTranspose1d {
    pub fn new(
        weight: Array3<f32>,
        bias: Option<Array1<f32>>,
        stride: usize,
        causal: bool,
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            causal,
            buffer: Vec::new(),
        }
    }

    pub fn in_channels(&self) -> usize {
        self.weight.shape()[0]
    }

    pub fn out_channels(&self) -> usize {
        self.weight.shape()[1]
    }

    pub fn kernel_size(&self) -> usize {
        self.weight.shape()[2]
    }

    /// Forward pass (transposed convolution).
    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        let (batch, in_c, time_in) = input.shape();
        assert_eq!(in_c, self.in_channels(), "Input channels mismatch");

        let time_out = time_in * self.stride;
        let mut output = Array3::<f32>::zeros((batch, self.out_channels(), time_out));

        // Simplified transposed convolution
        for b in 0..batch {
            for in_c in 0..in_c {
                for t_in in 0..time_in {
                    let input_val = input.data[[b, in_c, t_in]];
                    for k in 0..self.kernel_size() {
                        let t_out = t_in * self.stride + k;
                        if t_out < time_out {
                            for out_c in 0..self.out_channels() {
                                let weight_val = self.weight[[in_c, out_c, k]];
                                output[[b, out_c, t_out]] += input_val * weight_val;
                            }
                        }
                    }
                }
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            for b in 0..batch {
                for c in 0..self.out_channels() {
                    for t in 0..time_out {
                        output[[b, c, t]] += bias[c];
                    }
                }
            }
        }

        Tensor3::new(output)
    }

    pub fn init_buffer(&mut self, batch_size: usize) {
        self.buffer = (0..batch_size)
            .map(|_| Array2::zeros((self.in_channels(), 0)))
            .collect();
    }

    pub fn reset(&mut self) {
        for buf in &mut self.buffer {
            buf.fill(0.0);
        }
    }
}

/// Downsampling convolution (used between encoder and quantizer).
#[derive(Clone, Debug)]
pub struct ConvDownsample {
    conv: Conv1d,
}

impl ConvDownsample {
    pub fn new(weight: Array3<f32>, bias: Option<Array1<f32>>, stride: usize) -> Self {
        let conv = Conv1d::new(weight, bias, stride, 1, 1, true);
        Self { conv }
    }

    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        self.conv.forward(input)
    }
}

/// Upsampling transposed convolution (used after dequantizer).
#[derive(Clone, Debug)]
pub struct ConvUpsample {
    conv: ConvTranspose1d,
}

impl ConvUpsample {
    pub fn new(weight: Array3<f32>, bias: Option<Array1<f32>>, stride: usize) -> Self {
        let conv = ConvTranspose1d::new(weight, bias, stride, true);
        Self { conv }
    }

    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        self.conv.forward(input)
    }
}
