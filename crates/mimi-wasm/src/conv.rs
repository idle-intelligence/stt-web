//! 1D convolution operations for audio processing.
//!
//! Implements Conv1d and ConvTranspose1d with streaming support.

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3};

/// 1D convolution layer.
#[derive(Clone, Debug)]
pub struct Conv1d {
    /// Weight tensor: (out_channels, in_channels, kernel_size)
    pub weight: Array3<f32>,
    /// Pre-reshaped weight matrix: (out_channels, in_channels * kernel_size)
    weight_mat: Array2<f32>,
    /// Bias vector: (out_channels,)
    pub bias: Option<Array1<f32>>,
    pub stride: usize,
    pub dilation: usize,
    pub groups: usize,
    pub causal: bool,
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
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
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = weight.shape()[2];
        let col_rows = in_channels * kernel_size;

        // Pre-compute the reshaped weight matrix
        let weight_mat = weight
            .view()
            .into_shape_with_order((out_channels, col_rows))
            .expect("weight reshape")
            .to_owned();

        Self {
            weight,
            weight_mat,
            bias,
            stride,
            dilation,
            groups,
            causal,
            out_channels,
            in_channels,
            kernel_size,
            buffer: Vec::new(),
        }
    }

    /// Get output channels.
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get input channels.
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get kernel size.
    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    /// Calculate padding for causal convolution.
    fn padding(&self) -> usize {
        if self.causal {
            (self.kernel_size - 1) * self.dilation
        } else {
            ((self.kernel_size - 1) * self.dilation) / 2
        }
    }

    /// Forward pass using im2col + GEMM.
    ///
    /// Converts the convolution into a matrix multiply:
    ///   weight_mat: (out_channels, in_channels * kernel_size)
    ///   col_mat:    (in_channels * kernel_size, out_time)
    ///   output = weight_mat . col_mat → (out_channels, out_time)
    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        let (batch, in_c, time) = input.shape();
        assert_eq!(in_c, self.in_channels, "Input channels mismatch");

        let padding = self.padding();
        let ks = self.kernel_size;
        let out_c = self.out_channels;
        let stride = self.stride;
        let dilation = self.dilation;
        let out_time = (time + padding - dilation * (ks - 1)) / stride;
        let col_rows = in_c * ks;

        // Special case: kernel_size=1, stride=1, dilation=1 → direct matmul
        if ks == 1 && stride == 1 && dilation == 1 {
            return self.forward_1x1(input, batch, in_c, time, out_c);
        }

        let mut output = Array3::<f32>::zeros((batch, out_c, out_time));

        // Get raw slice access to input data for faster indexing
        let input_slice = input.data.as_slice().unwrap();

        for b in 0..batch {
            let b_input_offset = b * in_c * time;

            // Build im2col matrix: (col_rows, out_time) using raw slices
            let mut col = Array2::<f32>::zeros((col_rows, out_time));
            let col_slice = col.as_slice_mut().unwrap();

            if dilation == 1 && stride == 1 {
                // Highly optimized path for stride=1, dilation=1: bulk memcpy
                // For each (ic, k), the valid output range is contiguous in input
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        // t_in = t_out + k, valid when t_out + k >= padding
                        // AND t_out + k - padding < time
                        let t_start = if k >= padding { 0 } else { padding - k };
                        let t_end_limit = if time + padding >= k { time + padding - k } else { 0 };
                        let t_end = out_time.min(t_end_limit);
                        if t_start < t_end {
                            let len = t_end - t_start;
                            let src_start = ic_input_base + t_start + k - padding;
                            let dst_start = col_row_base + t_start;
                            col_slice[dst_start..dst_start + len]
                                .copy_from_slice(&input_slice[src_start..src_start + len]);
                        }
                    }
                }
            } else if dilation == 1 {
                // Optimized path for dilation=1 with stride > 1: bulk copy per row
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        // For stride > 1: t_in = t_out * stride + k
                        // valid when: t_out * stride + k >= padding
                        // AND t_out * stride + k - padding < time
                        let t_start = if k >= padding { 0 } else { (padding - k + stride - 1) / stride };
                        let t_end = if time + padding > k {
                            ((time + padding - k - 1) / stride + 1).min(out_time)
                        } else {
                            0
                        };
                        for t_out in t_start..t_end {
                            let t_in = t_out * stride + k;
                            col_slice[col_row_base + t_out] =
                                input_slice[ic_input_base + t_in - padding];
                        }
                    }
                }
            } else {
                // General path with dilation
                for t_out in 0..out_time {
                    let t_base = t_out * stride;
                    for ic in 0..in_c {
                        let col_offset = ic * ks;
                        let ic_input_base = b_input_offset + ic * time;
                        for k in 0..ks {
                            let t_in = t_base + k * dilation;
                            if t_in >= padding && t_in - padding < time {
                                let col_idx = (col_offset + k) * out_time + t_out;
                                col_slice[col_idx] =
                                    input_slice[ic_input_base + t_in - padding];
                            }
                        }
                    }
                }
            }

            // GEMM: (out_channels, col_rows) . (col_rows, out_time) → (out_channels, out_time)
            let result = self.weight_mat.dot(&col);

            // Copy result + bias using raw slices
            let result_slice = result.as_slice().unwrap();
            let output_slice = output.as_slice_mut().unwrap();
            let b_output_offset = b * out_c * out_time;

            match &self.bias {
                Some(bias) => {
                    let bias_slice = bias.as_slice().unwrap();
                    for oc in 0..out_c {
                        let bias_val = bias_slice[oc];
                        let r_base = oc * out_time;
                        let o_base = b_output_offset + oc * out_time;
                        for t in 0..out_time {
                            output_slice[o_base + t] = result_slice[r_base + t] + bias_val;
                        }
                    }
                }
                None => {
                    let src = &result_slice[..out_c * out_time];
                    let dst = &mut output_slice[b_output_offset..b_output_offset + out_c * out_time];
                    dst.copy_from_slice(src);
                }
            }
        }

        Tensor3::new(output)
    }

    /// Optimized forward pass for 1x1 convolution (no im2col needed).
    fn forward_1x1(&self, input: &Tensor3, batch: usize, in_c: usize, time: usize, out_c: usize) -> Tensor3 {
        let mut output = Array3::<f32>::zeros((batch, out_c, time));

        for b in 0..batch {
            // Input slice for this batch: (in_c, time) viewed as (in_c, time)
            // weight_mat is (out_c, in_c) for ks=1
            // output = weight_mat @ input_batch → (out_c, time)
            let input_view = input.data.index_axis(ndarray::Axis(0), b);
            let result = self.weight_mat.dot(&input_view);

            let result_slice = result.as_slice().unwrap();
            let output_slice = output.as_slice_mut().unwrap();
            let b_offset = b * out_c * time;

            match &self.bias {
                Some(bias) => {
                    let bias_slice = bias.as_slice().unwrap();
                    for oc in 0..out_c {
                        let bias_val = bias_slice[oc];
                        let base = oc * time;
                        let o_base = b_offset + base;
                        for t in 0..time {
                            output_slice[o_base + t] = result_slice[base + t] + bias_val;
                        }
                    }
                }
                None => {
                    let n = out_c * time;
                    output_slice[b_offset..b_offset + n].copy_from_slice(&result_slice[..n]);
                }
            }
        }

        Tensor3::new(output)
    }

    /// Initialize streaming buffer for a given batch size.
    pub fn init_buffer(&mut self, batch_size: usize) {
        let buffer_len = if self.causal { self.padding() } else { 0 };
        self.buffer = (0..batch_size)
            .map(|_| Array2::zeros((self.in_channels, buffer_len)))
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
