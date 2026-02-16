//! Small transformer used inside Mimi encoder.
//!
//! 8 layers, 8 heads, 512 dim, causal, with RoPE positional embeddings.

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3};

/// Multi-head self-attention layer.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Array2<f32>, // (d_model, d_model)
    pub k_proj: Array2<f32>,
    pub v_proj: Array2<f32>,
    pub out_proj: Array2<f32>,
    // KV cache for streaming
    pub k_cache: Option<Array3<f32>>, // (batch, heads, seq_len, head_dim)
    pub v_cache: Option<Array3<f32>>,
}

impl MultiHeadAttention {
    pub fn new(
        d_model: usize,
        num_heads: usize,
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        out_proj: Array2<f32>,
    ) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        let head_dim = d_model / num_heads;
        Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            k_cache: None,
            v_cache: None,
        }
    }

    /// Forward pass (non-streaming).
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, _d_model, seq_len) = x.shape();
        // Simplified self-attention (not optimized)
        // In a real implementation, we'd project, reshape, compute attention, and project back
        // For now, return input unchanged as placeholder
        x.clone()
    }

    pub fn reset(&mut self) {
        self.k_cache = None;
        self.v_cache = None;
    }
}

/// Feed-forward network.
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub fc1: Array2<f32>, // (d_model, dim_ff)
    pub fc2: Array2<f32>, // (dim_ff, d_model)
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
        // Simplified: return input unchanged as placeholder
        x.clone()
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
        // Simplified: return input unchanged as placeholder
        // Real implementation would normalize over the channel dimension
        x.clone()
    }
}

/// Single transformer layer.
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    pub attn: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    pub layer_scale: Option<f32>,
}

impl TransformerLayer {
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        // Simplified transformer layer
        // norm_first = true: norm -> attn -> residual -> norm -> ff -> residual
        let normed = self.norm1.forward(x);
        let attn_out = self.attn.forward(&normed);
        let x = x + &attn_out;

        let normed = self.norm2.forward(&x);
        let ff_out = self.ff.forward(&normed);
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
