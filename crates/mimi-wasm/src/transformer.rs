//! Small transformer used inside Mimi encoder.
//!
//! 8 layers, 8 heads, 512 dim, causal, with RoPE positional embeddings.

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3, Axis};

/// Multi-head self-attention layer with separate Q/K/V/O projections.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Array2<f32>, // (d_model, d_model)
    pub k_proj: Array2<f32>, // (d_model, d_model)
    pub v_proj: Array2<f32>, // (d_model, d_model)
    pub o_proj: Array2<f32>, // (d_model, d_model)
    // KV cache for streaming
    pub k_cache: Option<Array3<f32>>,
    pub v_cache: Option<Array3<f32>>,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        o_proj: Array2<f32>,
    ) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        let head_dim = d_model / num_heads;
        Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            k_cache: None,
            v_cache: None,
        }
    }

    /// Forward pass (non-streaming, full self-attention with RoPE).
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, d_model, seq_len) = x.shape();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let half_dim = head_dim / 2;

        // Transpose: (B, C, T) → (B, T, C)
        let xt = x.transpose_12();

        // Project Q, K, V separately
        let q_proj_t = self.q_proj.t();
        let k_proj_t = self.k_proj.t();
        let v_proj_t = self.v_proj.t();

        let mut q = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));
        let mut k = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));
        let mut v = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));

        let q_slice = q.as_slice_mut().unwrap();
        let k_slice = k.as_slice_mut().unwrap();
        let v_slice = v.as_slice_mut().unwrap();

        for b in 0..batch {
            let bt_slice = xt.data.index_axis(Axis(0), b); // (T, d_model)
            let q_all = bt_slice.dot(&q_proj_t); // (T, d_model)
            let k_all = bt_slice.dot(&k_proj_t);
            let v_all = bt_slice.dot(&v_proj_t);

            let q_all_s = q_all.as_slice().unwrap();
            let k_all_s = k_all.as_slice().unwrap();
            let v_all_s = v_all.as_slice().unwrap();

            // Reshape: (T, d_model) → (H, T, head_dim) stored as (B*H, T, D)
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let bh_off = bh * seq_len * head_dim;
                let h_off = h * head_dim;
                for t in 0..seq_len {
                    let t_src = t * d_model + h_off;
                    let t_dst = bh_off + t * head_dim;
                    q_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&q_all_s[t_src..t_src + head_dim]);
                    k_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&k_all_s[t_src..t_src + head_dim]);
                    v_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&v_all_s[t_src..t_src + head_dim]);
                }
            }
        }

        // RoPE: pre-compute cos/sin table for all positions
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / 10000.0_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Pre-compute cos/sin for all (t, i) pairs
        let mut cos_table = vec![0.0f32; seq_len * half_dim];
        let mut sin_table = vec![0.0f32; seq_len * half_dim];
        for t in 0..seq_len {
            for i in 0..half_dim {
                let freq = t as f32 * inv_freq[i];
                cos_table[t * half_dim + i] = freq.cos();
                sin_table[t * half_dim + i] = freq.sin();
            }
        }

        // Apply RoPE to Q and K using raw slices
        let q_s = q.as_slice_mut().unwrap();
        let k_s = k.as_slice_mut().unwrap();
        let total_bh = batch * num_heads;

        for bh in 0..total_bh {
            let bh_off = bh * seq_len * head_dim;
            for t in 0..seq_len {
                let t_off = bh_off + t * head_dim;
                let ct_off = t * half_dim;
                for i in 0..half_dim {
                    let cos_val = cos_table[ct_off + i];
                    let sin_val = sin_table[ct_off + i];

                    let qi0 = t_off + 2 * i;
                    let q_even = q_s[qi0];
                    let q_odd = q_s[qi0 + 1];
                    q_s[qi0] = q_even * cos_val - q_odd * sin_val;
                    q_s[qi0 + 1] = q_even * sin_val + q_odd * cos_val;

                    let ki0 = t_off + 2 * i;
                    let k_even = k_s[ki0];
                    let k_odd = k_s[ki0 + 1];
                    k_s[ki0] = k_even * cos_val - k_odd * sin_val;
                    k_s[ki0 + 1] = k_even * sin_val + k_odd * cos_val;
                }
            }
        }

        // Attention scores: Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = Array3::<f32>::zeros((batch * num_heads, seq_len, seq_len));
        for bh in 0..(batch * num_heads) {
            let q_slice = q.index_axis(Axis(0), bh);
            let k_slice = k.index_axis(Axis(0), bh);
            let attn = q_slice.dot(&k_slice.t()) * scale;
            scores.index_axis_mut(Axis(0), bh).assign(&attn);
        }

        // Causal mask: upper triangle = -inf
        for bh in 0..(batch * num_heads) {
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[bh, i, j]] = f32::NEG_INFINITY;
                }
            }
        }

        // Softmax over last dim
        let scores = Tensor3::new(scores).softmax_last();

        // Weighted sum: scores @ V → (B*H, T, D)
        let attn_out = scores.matmul(&Tensor3::new(v));

        // Reshape: (B*H, T, D) → (B, T, d_model) using raw slices
        let mut output = Array3::<f32>::zeros((batch, seq_len, d_model));
        let attn_s = attn_out.data.as_slice().unwrap();
        let out_s = output.as_slice_mut().unwrap();
        for b in 0..batch {
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let src_bh_off = bh * seq_len * head_dim;
                let h_off = h * head_dim;
                for t in 0..seq_len {
                    let src_off = src_bh_off + t * head_dim;
                    let dst_off = b * seq_len * d_model + t * d_model + h_off;
                    out_s[dst_off..dst_off + head_dim]
                        .copy_from_slice(&attn_s[src_off..src_off + head_dim]);
                }
            }
        }

        // Output projection: (B, T, d_model) @ o_proj.T → (B, T, d_model)
        let o_proj_t = self.o_proj.t();
        let mut projected = Array3::<f32>::zeros((batch, seq_len, d_model));
        for b in 0..batch {
            let slice = output.index_axis(Axis(0), b);
            let result = slice.dot(&o_proj_t);
            projected.index_axis_mut(Axis(0), b).assign(&result);
        }

        // Transpose back: (B, T, d_model) → (B, d_model, T)
        Tensor3::new(projected).transpose_12()
    }

    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

/// Feed-forward network.
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub fc1: Array2<f32>, // (dim_ff, d_model)
    pub fc2: Array2<f32>, // (d_model, dim_ff)
    pub bias1: Option<Array1<f32>>,
    pub bias2: Option<Array1<f32>>,
}

impl FeedForward {
    pub fn new(
        fc1: Array2<f32>,
        fc2: Array2<f32>,
        bias1: Option<Array1<f32>>,
        bias2: Option<Array1<f32>>,
    ) -> Self {
        Self { fc1, fc2, bias1, bias2 }
    }

    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, _d_model, _time) = x.shape();
        let xt = x.transpose_12(); // (B, T, d_model)
        let time = xt.data.shape()[1];
        let out_dim = self.fc2.shape()[0];

        let fc1_t = self.fc1.t(); // (d_model, dim_ff)
        let fc2_t = self.fc2.t(); // (dim_ff, d_model)

        let mut result = Array3::<f32>::zeros((batch, time, out_dim));

        for b in 0..batch {
            let slice = xt.data.index_axis(Axis(0), b); // (T, d_model)

            // First linear: (T, d_model) @ (d_model, dim_ff) → (T, dim_ff)
            let mut hidden = slice.dot(&fc1_t);

            if let Some(ref bias) = self.bias1 {
                let dim_ff = hidden.shape()[1];
                for i in 0..time {
                    for j in 0..dim_ff {
                        hidden[[i, j]] += bias[j];
                    }
                }
            }

            // GELU activation (hoist constant)
            const GELU_COEFF: f32 = 0.7978845608; // sqrt(2/pi)
            let h_slice = hidden.as_slice_mut().unwrap();
            for v in h_slice.iter_mut() {
                let x = *v;
                *v = x * 0.5 * (1.0 + (GELU_COEFF * (x + 0.044715 * x * x * x)).tanh());
            }

            // Second linear: (T, dim_ff) @ (dim_ff, d_model) → (T, d_model)
            let mut output = hidden.dot(&fc2_t);

            if let Some(ref bias) = self.bias2 {
                for i in 0..time {
                    for j in 0..out_dim {
                        output[[i, j]] += bias[j];
                    }
                }
            }

            result.index_axis_mut(Axis(0), b).assign(&output);
        }

        // Transpose back: (B, T, d_model) → (B, d_model, T)
        Tensor3::new(result).transpose_12()
    }
}

/// Layer normalization.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, channels, time) = x.shape();
        let mut out = x.data.clone();
        let slice = out.as_slice_mut().unwrap();
        let w = self.weight.as_slice().unwrap();
        let bias = self.bias.as_slice().unwrap();

        // Data layout: [b, c, t] → index = b * channels * time + c * time + t
        for b in 0..batch {
            let b_off = b * channels * time;
            for t in 0..time {
                // Mean over channels (stride = time between channel elements)
                let mut mean = 0.0f32;
                for c in 0..channels {
                    mean += slice[b_off + c * time + t];
                }
                mean /= channels as f32;

                // Variance over channels
                let mut var = 0.0f32;
                for c in 0..channels {
                    let diff = slice[b_off + c * time + t] - mean;
                    var += diff * diff;
                }
                var /= channels as f32;

                // Normalize and apply weight/bias
                let inv_std = 1.0 / (var + self.eps).sqrt();
                for c in 0..channels {
                    let idx = b_off + c * time + t;
                    slice[idx] = w[c] * (slice[idx] - mean) * inv_std + bias[c];
                }
            }
        }

        Tensor3::new(out)
    }
}

/// Single transformer layer.
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    pub attn: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    /// Per-dimension layer scale applied after attention: (d_model,)
    pub layer_scale_1: Option<Array1<f32>>,
    /// Per-dimension layer scale applied after feedforward: (d_model,)
    pub layer_scale_2: Option<Array1<f32>>,
}

impl TransformerLayer {
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        // norm_first = true: norm -> attn -> scale -> residual -> norm -> ff -> scale -> residual
        let normed = self.norm1.forward(x);
        let mut attn_out = self.attn.forward(&normed);
        if let Some(ref scale) = self.layer_scale_1 {
            attn_out = attn_out.scale_channels(scale);
        }
        let x = x + &attn_out;

        let normed = self.norm2.forward(&x);
        let mut ff_out = self.ff.forward(&normed);
        if let Some(ref scale) = self.layer_scale_2 {
            ff_out = ff_out.scale_channels(scale);
        }
        let x = &x + &ff_out;

        x
    }

    pub fn reset(&mut self) {
        self.attn.reset();
    }
}

/// Full transformer stack.
#[derive(Clone, Debug)]
pub struct Transformer {
    pub layers: Vec<TransformerLayer>,
    pub d_model: usize,
}

impl Transformer {
    pub fn new(layers: Vec<TransformerLayer>, d_model: usize) -> Self {
        Self { layers, d_model }
    }

    /// Forward pass through all layers.
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x);
        }
        x
    }

    /// Reset all layer states (clear KV cache).
    pub fn reset(&mut self) {
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
