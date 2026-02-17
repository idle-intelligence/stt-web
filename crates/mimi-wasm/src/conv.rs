//! 1D convolution operations for audio processing.
//!
//! Implements Conv1d with streaming support.

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3};

/// 1D convolution layer.
#[derive(Clone, Debug)]
pub struct Conv1d {
    /// Weight tensor: (out_channels, in_channels, kernel_size)
    pub weight: Array3<f32>,
    /// Pre-reshaped weight matrix: (out_channels, in_channels * kernel_size)
    weight_mat: Array2<f32>,
    /// Per-kernel-position weight matrices for direct conv: weight[:, :, k] → (out_c, in_c)
    kernel_weights: Vec<Array2<f32>>,
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
    /// Whether left padding has been applied on the first streaming step.
    left_pad_applied: bool,
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

        // Pre-compute the reshaped weight matrix (for im2col path)
        let weight_mat = weight
            .view()
            .into_shape_with_order((out_channels, col_rows))
            .expect("weight reshape")
            .to_owned();

        // Pre-compute per-kernel-position weight matrices (for direct conv path)
        // Used for both stride=1 and strided convolutions when dilation=1
        let kernel_weights = if dilation == 1 && kernel_size > 1 {
            (0..kernel_size)
                .map(|k| {
                    weight
                        .slice(ndarray::s![.., .., k])
                        .to_owned()
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            weight,
            weight_mat,
            kernel_weights,
            bias,
            stride,
            dilation,
            groups,
            causal,
            out_channels,
            in_channels,
            kernel_size,
            buffer: Vec::new(),
            left_pad_applied: false,
        }
    }

    /// Calculate padding for causal convolution.
    fn padding(&self) -> usize {
        if self.causal {
            (self.kernel_size - 1) * self.dilation
        } else {
            ((self.kernel_size - 1) * self.dilation) / 2
        }
    }

    /// Forward pass for 1D convolution.
    ///
    /// Uses one of three strategies:
    /// - kernel_size=1: direct matmul (no im2col)
    /// - stride=1, dilation=1: direct conv via partial matmuls (avoids huge im2col allocation)
    /// - general: im2col + GEMM
    pub fn forward(&self, input: &Tensor3) -> Tensor3 {
        let (batch, in_c, time) = input.shape();
        assert_eq!(in_c, self.in_channels, "Input channels mismatch");

        let ks = self.kernel_size;
        let out_c = self.out_channels;

        // Special case: kernel_size=1, stride=1, dilation=1 → direct matmul
        if ks == 1 && self.stride == 1 && self.dilation == 1 {
            return self.forward_1x1(input, batch, in_c, time, out_c);
        }

        // Direct conv via partial matmuls: avoids allocating huge im2col matrix.
        // Only use when im2col would be large (>1M floats = 4MB) and in_c is
        // large enough that per-kernel GEMMs are efficient.
        let padding = self.padding();
        let stride = self.stride;
        let dilation = self.dilation;
        let out_time = (time + padding - dilation * (ks - 1)) / stride;
        let col_size = in_c * ks * out_time;

        if stride == 1 && dilation == 1 && !self.kernel_weights.is_empty()
            && col_size > 1_000_000 && in_c >= 4
        {
            return self.forward_direct(input, batch, in_c, time, out_c);
        }

        // General im2col path
        self.forward_im2col(input, batch, in_c, time, out_c)
    }

    /// Direct convolution via partial matmuls (stride=1, dilation=1).
    ///
    /// Instead of building a (in_c*ks, out_time) im2col matrix, performs ks
    /// separate matmuls with offset input views and accumulates results.
    /// This eliminates the massive im2col allocation that dominates memory
    /// for long sequences (e.g. 553MB for layer[0] at 30s audio).
    fn forward_direct(
        &self,
        input: &Tensor3,
        batch: usize,
        in_c: usize,
        time: usize,
        out_c: usize,
    ) -> Tensor3 {
        let ks = self.kernel_size;
        let padding = self.padding();
        let out_time = time; // stride=1, dilation=1 → out_time = time

        let mut output = Array3::<f32>::zeros((batch, out_c, out_time));

        // For very thin matrices (in_c < 4), fall back to im2col since
        // per-kernel-position GEMMs would have too much overhead.
        if in_c < 4 {
            return self.forward_im2col(input, batch, in_c, time, out_c);
        }

        for b in 0..batch {
            let input_b = input.data.index_axis(ndarray::Axis(0), b); // (in_c, time)

            for k in 0..ks {
                let weight_k = &self.kernel_weights[k]; // (out_c, in_c)

                // Valid output range: t_out where input[t_out + k - padding] is in bounds
                let t_start = padding.saturating_sub(k);
                let t_end = out_time.min(time + padding - k);

                if t_start >= t_end {
                    continue;
                }

                let input_start = t_start + k - padding;
                let len = t_end - t_start;

                // Slice input view: (in_c, len) — no allocation, just a view
                let input_slice =
                    input_b.slice(ndarray::s![.., input_start..input_start + len]);

                // GEMM: (out_c, in_c) @ (in_c, len) → (out_c, len)
                let result = weight_k.dot(&input_slice);

                // Accumulate into output
                let mut out_slice =
                    output.slice_mut(ndarray::s![b, .., t_start..t_end]);
                out_slice += &result;
            }
        }

        // Add bias
        if let Some(ref bias) = self.bias {
            let bias_slice = bias.as_slice().unwrap();
            let output_slice = output.as_slice_mut().unwrap();
            for b in 0..batch {
                for oc in 0..out_c {
                    let bias_val = bias_slice[oc];
                    let base = (b * out_c + oc) * out_time;
                    for t in 0..out_time {
                        output_slice[base + t] += bias_val;
                    }
                }
            }
        }

        Tensor3::new(output)
    }

    /// Forward pass using im2col + GEMM (for stride > 1 or dilation > 1).
    fn forward_im2col(
        &self,
        input: &Tensor3,
        batch: usize,
        in_c: usize,
        time: usize,
        out_c: usize,
    ) -> Tensor3 {
        let padding = self.padding();
        let ks = self.kernel_size;
        let stride = self.stride;
        let dilation = self.dilation;
        let out_time = (time + padding - dilation * (ks - 1)) / stride;
        let col_rows = in_c * ks;

        let mut output = Array3::<f32>::zeros((batch, out_c, out_time));
        let input_slice = input.data.as_slice().unwrap();

        for b in 0..batch {
            let b_input_offset = b * in_c * time;

            let mut col = Array2::<f32>::zeros((col_rows, out_time));
            let col_slice = col.as_slice_mut().unwrap();

            if dilation == 1 && stride == 1 {
                // Bulk memcpy path for stride=1, dilation=1
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        let t_start = if k >= padding { 0 } else { padding - k };
                        let t_end_limit =
                            if time + padding >= k { time + padding - k } else { 0 };
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
                // Stride > 1 path
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        let t_start = if k >= padding {
                            0
                        } else {
                            (padding - k + stride - 1) / stride
                        };
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

            let result = self.weight_mat.dot(&col);

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
                    let dst =
                        &mut output_slice[b_output_offset..b_output_offset + out_c * out_time];
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
    pub fn init_buffer(&mut self, _batch_size: usize) {
        // For streaming, buffer starts empty; left padding is applied on first step.
        self.buffer = vec![Array2::zeros((self.in_channels, 0))];
        self.left_pad_applied = false;
    }

    /// Initialize streaming state (alias for init_buffer(1)).
    pub fn init_streaming(&mut self) {
        self.init_buffer(1);
    }

    /// Forward pass without any padding (used by streaming step).
    ///
    /// Applies convolution directly — caller is responsible for handling padding
    /// via the buffer mechanism.
    fn forward_no_padding(&self, input: &Tensor3) -> Tensor3 {
        let (batch, in_c, time) = input.shape();
        assert_eq!(in_c, self.in_channels, "Input channels mismatch");

        let ks = self.kernel_size;
        let out_c = self.out_channels;

        // kernel_size=1: no padding distinction needed
        if ks == 1 && self.stride == 1 && self.dilation == 1 {
            return self.forward_1x1(input, batch, in_c, time, out_c);
        }

        // General im2col path with padding=0
        let stride = self.stride;
        let dilation = self.dilation;
        // With padding=0: out_time = (time - dilation*(ks-1) - 1) / stride + 1
        let effective_kernel = dilation * (ks - 1) + 1;
        if time < effective_kernel {
            // Not enough input for even one output frame
            return Tensor3::zeros((batch, out_c, 0));
        }
        let out_time = (time - effective_kernel) / stride + 1;
        let col_rows = in_c * ks;

        let mut output = Array3::<f32>::zeros((batch, out_c, out_time));
        let input_slice = input.data.as_slice().unwrap();

        for b in 0..batch {
            let b_input_offset = b * in_c * time;

            let mut col = Array2::<f32>::zeros((col_rows, out_time));
            let col_slice = col.as_slice_mut().unwrap();

            // No padding, so all accesses are direct (no zero-padding region)
            if dilation == 1 && stride == 1 {
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        let src_start = ic_input_base + k;
                        let dst_start = col_row_base;
                        col_slice[dst_start..dst_start + out_time]
                            .copy_from_slice(&input_slice[src_start..src_start + out_time]);
                    }
                }
            } else if dilation == 1 {
                // Stride > 1, no dilation
                for ic in 0..in_c {
                    let ic_input_base = b_input_offset + ic * time;
                    let col_ic_base = ic * ks;
                    for k in 0..ks {
                        let col_row = col_ic_base + k;
                        let col_row_base = col_row * out_time;
                        for t_out in 0..out_time {
                            let t_in = t_out * stride + k;
                            col_slice[col_row_base + t_out] =
                                input_slice[ic_input_base + t_in];
                        }
                    }
                }
            } else {
                // General: dilation > 1
                for t_out in 0..out_time {
                    let t_base = t_out * stride;
                    for ic in 0..in_c {
                        let col_offset = ic * ks;
                        let ic_input_base = b_input_offset + ic * time;
                        for k in 0..ks {
                            let t_in = t_base + k * dilation;
                            if t_in < time {
                                let col_idx = (col_offset + k) * out_time + t_out;
                                col_slice[col_idx] = input_slice[ic_input_base + t_in];
                            }
                        }
                    }
                }
            }

            let result = self.weight_mat.dot(&col);

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
                    let dst =
                        &mut output_slice[b_output_offset..b_output_offset + out_c * out_time];
                    dst.copy_from_slice(src);
                }
            }
        }

        Tensor3::new(output)
    }

    /// Streaming forward: feed incremental data, returns output when enough has accumulated.
    ///
    /// On the first call, applies causal left-padding. Each call concatenates buffered state
    /// with new input, produces as many output frames as possible, and saves the remainder.
    /// Returns `None` if not enough data has accumulated for any output frames.
    pub fn step(&mut self, input: &Tensor3) -> Option<Tensor3> {
        let (batch, _in_c, _time) = input.shape();
        assert_eq!(batch, 1, "Streaming only supports batch_size=1");

        let in_c = self.in_channels;

        // Build combined buffer: [left_pad] + [prev_buffer] + [new_input]
        // Directly into a single contiguous allocation to avoid intermediate clones
        let buf_time = if self.buffer.is_empty() { 0 } else { self.buffer[0].shape()[1] };
        let pad_time = if !self.left_pad_applied && self.causal {
            self.padding()
        } else {
            0
        };
        let (_, _, input_time) = input.shape();
        let total_time = pad_time + buf_time + input_time;

        let mut combined_data = Array3::<f32>::zeros((1, in_c, total_time));
        {
            let dst = combined_data.as_slice_mut().unwrap();
            let mut write_pos = 0;

            // Left padding (zeros, already initialized) — just advance position
            if pad_time > 0 {
                self.left_pad_applied = true;
                write_pos += pad_time;
            } else if !self.left_pad_applied {
                self.left_pad_applied = true;
            }

            // Copy buffer
            if buf_time > 0 {
                let buf_2d = &self.buffer[0];
                let buf_src = buf_2d.as_standard_layout();
                let buf_slice = buf_src.as_slice().unwrap();
                // Buffer layout: (in_c, buf_time) — copy per channel
                for c in 0..in_c {
                    let dst_off = c * total_time + write_pos;
                    let src_off = c * buf_time;
                    dst[dst_off..dst_off + buf_time]
                        .copy_from_slice(&buf_slice[src_off..src_off + buf_time]);
                }
                write_pos += buf_time;
            }

            // Copy new input
            if input_time > 0 {
                let inp_slice = input.data.as_slice().unwrap();
                for c in 0..in_c {
                    let dst_off = c * total_time + write_pos;
                    let src_off = c * input_time;
                    dst[dst_off..dst_off + input_time]
                        .copy_from_slice(&inp_slice[src_off..src_off + input_time]);
                }
            }
        }

        let combined = Tensor3::new(combined_data);
        let seq_len = total_time;
        let stride = self.stride;
        let dilation = self.dilation;
        let kernel = (self.kernel_size - 1) * dilation + 1;

        let num_frames = (seq_len + stride).saturating_sub(kernel) / stride;

        if num_frames == 0 {
            // Not enough data — save everything as buffer
            let view = combined.data.index_axis(ndarray::Axis(0), 0);
            self.buffer = vec![view.as_standard_layout().to_owned()];
            return None;
        }

        let offset = num_frames * stride;
        let in_len = (num_frames - 1) * stride + kernel;

        // Split: take [0..in_len] for conv, save [offset..] as buffer
        let conv_input = combined.slice_time(0, in_len);
        if seq_len > offset {
            // Save remaining as buffer (directly from combined to avoid double-copy)
            let remaining_len = seq_len - offset;
            let mut buf = Array2::<f32>::zeros((in_c, remaining_len));
            let src = combined.data.as_slice().unwrap();
            let dst = buf.as_slice_mut().unwrap();
            for c in 0..in_c {
                let src_off = c * seq_len + offset;
                let dst_off = c * remaining_len;
                dst[dst_off..dst_off + remaining_len]
                    .copy_from_slice(&src[src_off..src_off + remaining_len]);
            }
            self.buffer = vec![buf];
        } else {
            self.buffer = vec![Array2::zeros((in_c, 0))];
        }

        // Apply conv without any padding (padding handled by buffer)
        let output = self.forward_no_padding(&conv_input);
        let (_b, _c, out_time) = output.shape();
        if out_time == 0 {
            None
        } else {
            Some(output)
        }
    }

    /// Reset streaming state.
    pub fn reset(&mut self) {
        self.buffer = vec![Array2::zeros((self.in_channels, 0))];
        self.left_pad_applied = false;
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

    pub fn step(&mut self, input: &Tensor3) -> Option<Tensor3> {
        self.conv.step(input)
    }

    pub fn init_streaming(&mut self) {
        self.conv.init_streaming();
    }

    pub fn reset(&mut self) {
        self.conv.reset();
    }
}

