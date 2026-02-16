//! STT transformer model: embeddings, attention, RoPE, KV cache.
//!
//! Decoder-only transformer architecture:
//! - Per-codebook audio embeddings (32 codebooks, summed per frame)
//! - Text token embedding
//! - RoPE positional embeddings (theta=100000)
//! - Multi-head attention (MHA, 16 heads)
//! - SwiGLU feed-forward network
//! - KV cache for autoregressive decoding
//! - Sliding window attention (750)
//!
//! All linear projections use Q4Linear from the gguf module.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::activation::{silu, softmax};
use burn::tensor::Tensor;

use crate::gguf::{EmbeddingStore, Q4Linear};
use crate::SttConfig;

// ---------------------------------------------------------------------------
// RoPE — Rotary Position Embeddings
// ---------------------------------------------------------------------------

/// Rotary Position Embeddings with precomputed cos/sin tables.
pub struct RoPE {
    cos: Tensor<Wgpu, 2>,
    sin: Tensor<Wgpu, 2>,
}

impl RoPE {
    /// Create RoPE with precomputed frequencies.
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64, device: &WgpuDevice) -> Self {
        let half_dim = head_dim / 2;

        // Inverse frequencies: 1 / (theta^(2i/d))
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / (theta as f32).powf((2 * i) as f32 / head_dim as f32))
            .collect();

        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();

        // Outer product: freqs[i, j] = positions[i] * inv_freq[j]
        let mut freqs = vec![0.0f32; max_seq_len * half_dim];
        for i in 0..max_seq_len {
            for j in 0..half_dim {
                freqs[i * half_dim + j] = positions[i] * inv_freq[j];
            }
        }

        let freqs = Tensor::<Wgpu, 1>::from_floats(freqs.as_slice(), device)
            .reshape([max_seq_len, half_dim]);

        let cos = freqs.clone().cos();
        let sin = freqs.sin();

        RoPE { cos, sin }
    }

    /// Apply rotary embeddings to Q and K tensors.
    ///
    /// q, k shape: [batch, seq, heads, head_dim]
    pub fn apply(
        &self,
        q: Tensor<Wgpu, 4>,
        k: Tensor<Wgpu, 4>,
        offset: usize,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let seq_len = q.dims()[1];
        let [_max_len, half_dim] = self.cos.dims();
        let cos = self
            .cos
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);
        let sin = self
            .sin
            .clone()
            .slice([offset..offset + seq_len, 0..half_dim]);

        let q_rot = self.apply_rotation(q, cos.clone(), sin.clone());
        let k_rot = self.apply_rotation(k, cos, sin);
        (q_rot, k_rot)
    }

    fn apply_rotation(
        &self,
        x: Tensor<Wgpu, 4>,
        cos: Tensor<Wgpu, 2>,
        sin: Tensor<Wgpu, 2>,
    ) -> Tensor<Wgpu, 4> {
        let [batch, seq, heads, head_dim] = x.dims();
        let half_dim = head_dim / 2;

        // Reshape to separate interleaved pairs: [batch, seq, heads, half_dim, 2]
        let x_pairs = x.reshape([batch, seq, heads, half_dim, 2]);

        let x_r: Tensor<Wgpu, 4> = x_pairs
            .clone()
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 0..1])
            .reshape([batch, seq, heads, half_dim]);
        let x_i: Tensor<Wgpu, 4> = x_pairs
            .slice([0..batch, 0..seq, 0..heads, 0..half_dim, 1..2])
            .reshape([batch, seq, heads, half_dim]);

        // Broadcast cos/sin: [seq, half_dim] -> [1, seq, 1, half_dim]
        let cos: Tensor<Wgpu, 4> = cos
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim(2);
        let sin: Tensor<Wgpu, 4> = sin
            .unsqueeze_dim::<3>(0)
            .unsqueeze_dim(2);

        // Apply rotation: [x_r * cos - x_i * sin, x_r * sin + x_i * cos]
        let out_r = x_r.clone() * cos.clone() - x_i.clone() * sin.clone();
        let out_i = x_r * sin + x_i * cos;

        // Interleave back
        let out_r: Tensor<Wgpu, 5> = out_r.unsqueeze_dim(4);
        let out_i: Tensor<Wgpu, 5> = out_i.unsqueeze_dim(4);
        let out = Tensor::cat(vec![out_r, out_i], 4);
        out.reshape([batch, seq, heads, head_dim])
    }
}

// ---------------------------------------------------------------------------
// KVCache
// ---------------------------------------------------------------------------

/// KV Cache for autoregressive decoding (dynamic concatenation mode).
pub struct KVCache {
    pub k: Option<Tensor<Wgpu, 4>>,
    pub v: Option<Tensor<Wgpu, 4>>,
}

impl KVCache {
    pub fn new() -> Self {
        Self { k: None, v: None }
    }

    /// Update cache with new K, V and return full sequences.
    pub fn update(
        &mut self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        let k_full = match &self.k {
            None => {
                self.k = Some(k.clone());
                k
            }
            Some(cache) => {
                let full = Tensor::cat(vec![cache.clone(), k], 2);
                self.k = Some(full.clone());
                full
            }
        };
        let v_full = match &self.v {
            None => {
                self.v = Some(v.clone());
                v
            }
            Some(cache) => {
                let full = Tensor::cat(vec![cache.clone(), v], 2);
                self.v = Some(full.clone());
                full
            }
        };
        (k_full, v_full)
    }

    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|k| k.dims()[2]).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }
}

/// Collection of KV caches for all layers.
pub struct LayerCaches {
    caches: Vec<KVCache>,
}

impl LayerCaches {
    pub fn new(n_layers: usize) -> Self {
        Self {
            caches: (0..n_layers).map(|_| KVCache::new()).collect(),
        }
    }

    pub fn get_mut(&mut self, layer: usize) -> Option<&mut KVCache> {
        self.caches.get_mut(layer)
    }

    pub fn seq_len(&self) -> usize {
        self.caches.first().map(|c| c.seq_len()).unwrap_or(0)
    }

    pub fn reset(&mut self) {
        for cache in &mut self.caches {
            cache.reset();
        }
    }
}

// ---------------------------------------------------------------------------
// Masking
// ---------------------------------------------------------------------------

/// Apply causal mask to attention scores (same-length Q and K).
fn apply_causal_mask(scores: Tensor<Wgpu, 4>, seq_len: usize) -> Tensor<Wgpu, 4> {
    let device = scores.device();
    let mut mask_data = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask_data[i * seq_len + j] = f32::NEG_INFINITY;
        }
    }
    let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<Wgpu, 2> = mask.reshape([seq_len, seq_len]);
    let mask: Tensor<Wgpu, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

/// Apply causal mask with different Q/K lengths (for KV cache).
fn apply_causal_mask_with_offset(
    scores: Tensor<Wgpu, 4>,
    q_len: usize,
    kv_len: usize,
    offset: usize,
) -> Tensor<Wgpu, 4> {
    if q_len == 1 {
        return scores;
    }
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if j > actual_pos {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<Wgpu, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<Wgpu, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

/// Apply sliding window mask with different Q/K lengths (for KV cache).
fn apply_sliding_window_mask_with_offset(
    scores: Tensor<Wgpu, 4>,
    q_len: usize,
    kv_len: usize,
    window: usize,
    offset: usize,
) -> Tensor<Wgpu, 4> {
    if offset + q_len <= window + 1 {
        return scores;
    }
    let device = scores.device();
    let mut mask_data = vec![0.0f32; q_len * kv_len];
    for i in 0..q_len {
        let actual_pos = offset + i;
        for j in 0..kv_len {
            if actual_pos.abs_diff(j) > window {
                mask_data[i * kv_len + j] = f32::NEG_INFINITY;
            }
        }
    }
    let mask: Tensor<Wgpu, 1> = Tensor::from_floats(mask_data.as_slice(), &device);
    let mask: Tensor<Wgpu, 2> = mask.reshape([q_len, kv_len]);
    let mask: Tensor<Wgpu, 4> = mask.unsqueeze_dim::<3>(0).unsqueeze_dim(0);
    scores + mask
}

// ---------------------------------------------------------------------------
// RmsNorm wrapper
// ---------------------------------------------------------------------------

/// RMSNorm layer wrapping burn::nn::RmsNorm for GGUF weight loading.
pub struct RmsNormLayer {
    pub inner: burn::nn::RmsNorm<Wgpu>,
}

impl RmsNormLayer {
    pub fn forward<const D: usize>(&self, x: Tensor<Wgpu, D>) -> Tensor<Wgpu, D> {
        self.inner.forward(x)
    }
}

// ---------------------------------------------------------------------------
// Q4Attention
// ---------------------------------------------------------------------------

/// Multi-head attention with Q4-quantized weight projections.
///
/// Supports both MHA (n_heads == n_kv_heads) and GQA configurations.
pub struct Q4Attention {
    wq: Q4Linear,
    wk: Q4Linear,
    wv: Q4Linear,
    wo: Q4Linear,
    n_heads: usize,
    n_kv_heads: usize,
    head_dim: usize,
    scale: f32,
    sliding_window: Option<usize>,
}

impl Q4Attention {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        wq: Q4Linear,
        wk: Q4Linear,
        wv: Q4Linear,
        wo: Q4Linear,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        sliding_window: Option<usize>,
    ) -> Self {
        Self {
            wq,
            wk,
            wv,
            wo,
            n_heads,
            n_kv_heads,
            head_dim,
            scale: (head_dim as f32).powf(-0.5),
            sliding_window,
        }
    }

    /// Forward pass with KV cache.
    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();
        let offset = cache.seq_len();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = cache.update(k, v);
        let total_seq_len = cache.seq_len();

        // Expand K, V for GQA if needed
        let (k, v) = self.expand_kv(k, v);

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        // Causal masking
        let scores = apply_causal_mask_with_offset(scores, seq_len, total_seq_len, offset);

        // Sliding window masking
        let scores = if let Some(window) = self.sliding_window {
            apply_sliding_window_mask_with_offset(scores, seq_len, total_seq_len, window, offset)
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.wo.forward(out)
    }

    /// Forward pass without KV cache (for prefill).
    pub fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let [batch, seq_len, _] = x.dims();

        let q = self.wq.forward(x.clone());
        let k = self.wk.forward(x.clone());
        let v = self.wv.forward(x);

        let q = q.reshape([batch, seq_len, self.n_heads, self.head_dim]);
        let k = k.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);
        let v = v.reshape([batch, seq_len, self.n_kv_heads, self.head_dim]);

        let (q, k) = rope.apply(q, k, offset);

        let q = q.swap_dims(1, 2);
        let k = k.swap_dims(1, 2);
        let v = v.swap_dims(1, 2);

        let (k, v) = self.expand_kv(k, v);

        let k_t = k.swap_dims(2, 3);
        let scores = q.matmul(k_t) * self.scale;

        let scores = apply_causal_mask(scores, seq_len);
        let scores = if let Some(window) = self.sliding_window {
            apply_sliding_window_mask_with_offset(scores, seq_len, seq_len, window, offset)
        } else {
            scores
        };

        let attn = softmax(scores, 3);
        let out = attn.matmul(v);

        let out = out.swap_dims(1, 2);
        let out = out.reshape([batch, seq_len, self.n_heads * self.head_dim]);
        self.wo.forward(out)
    }

    fn expand_kv(
        &self,
        k: Tensor<Wgpu, 4>,
        v: Tensor<Wgpu, 4>,
    ) -> (Tensor<Wgpu, 4>, Tensor<Wgpu, 4>) {
        if self.n_heads == self.n_kv_heads {
            return (k, v);
        }
        let repeat_factor = self.n_heads / self.n_kv_heads;
        let [batch, n_kv_heads, seq, head_dim] = k.dims();

        let k = k
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        let v = v
            .unsqueeze_dim::<5>(2)
            .repeat_dim(2, repeat_factor)
            .reshape([batch, n_kv_heads * repeat_factor, seq, head_dim]);
        (k, v)
    }
}

// ---------------------------------------------------------------------------
// Q4FeedForward (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU MLP with Q4-quantized weights.
///
/// Computes `w2(silu(w1(x)) * w3(x))`.
pub struct Q4FeedForward {
    w1: Q4Linear,
    w2: Q4Linear,
    w3: Q4Linear,
}

impl Q4FeedForward {
    pub fn new(w1: Q4Linear, w2: Q4Linear, w3: Q4Linear) -> Self {
        Self { w1, w2, w3 }
    }

    pub fn forward(&self, x: Tensor<Wgpu, 3>) -> Tensor<Wgpu, 3> {
        let gate = silu(self.w1.forward(x.clone()));
        let up = self.w3.forward(x);
        self.w2.forward(gate * up)
    }
}

// ---------------------------------------------------------------------------
// Q4TransformerBlock
// ---------------------------------------------------------------------------

/// Pre-LN transformer block with Q4 weights.
///
/// Architecture: RMSNorm → self-attention → residual → RMSNorm → SwiGLU → residual
pub struct Q4TransformerBlock {
    attention_norm: RmsNormLayer,
    attention: Q4Attention,
    ffn_norm: RmsNormLayer,
    ffn: Q4FeedForward,
}

impl Q4TransformerBlock {
    pub fn new(
        attention_norm: RmsNormLayer,
        attention: Q4Attention,
        ffn_norm: RmsNormLayer,
        ffn: Q4FeedForward,
    ) -> Self {
        Self {
            attention_norm,
            attention,
            ffn_norm,
            ffn,
        }
    }

    pub fn forward_with_cache(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        cache: &mut KVCache,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward_with_cache(x, rope, cache);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }

    pub fn forward(
        &self,
        x: Tensor<Wgpu, 3>,
        rope: &RoPE,
        offset: usize,
    ) -> Tensor<Wgpu, 3> {
        let residual = x.clone();
        let x = self.attention_norm.forward(x);
        let x = self.attention.forward(x, rope, offset);
        let x = x + residual;

        let residual = x.clone();
        let x = self.ffn_norm.forward(x);
        let x = self.ffn.forward(x);
        x + residual
    }
}

// ---------------------------------------------------------------------------
// SttModel
// ---------------------------------------------------------------------------

/// The complete STT transformer model.
///
/// Decoder-only architecture consuming audio tokens (32 codebooks) and
/// previous text token, producing next text token logits.
pub struct SttModel {
    audio_emb: Vec<EmbeddingStore>,
    text_emb: EmbeddingStore,
    layers: Vec<Q4TransformerBlock>,
    rope: RoPE,
    out_norm: RmsNormLayer,
    text_linear: Q4Linear,
    config: SttConfig,
    device: WgpuDevice,
}

impl SttModel {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        audio_emb: Vec<EmbeddingStore>,
        text_emb: EmbeddingStore,
        layers: Vec<Q4TransformerBlock>,
        rope: RoPE,
        out_norm: RmsNormLayer,
        text_linear: Q4Linear,
        config: SttConfig,
        device: WgpuDevice,
    ) -> Self {
        Self {
            audio_emb,
            text_emb,
            layers,
            rope,
            out_norm,
            text_linear,
            config,
            device,
        }
    }

    /// Single-step forward pass for autoregressive decoding.
    ///
    /// `audio_tokens`: 32 token IDs (one per codebook)
    /// `text_token`: previous text token (or padding token at start)
    /// `cache`: mutable KV cache (updated in place)
    ///
    /// Returns logits [1, 1, vocab_size].
    pub fn forward(
        &self,
        audio_tokens: &[u32],
        text_token: u32,
        cache: &mut LayerCaches,
    ) -> Tensor<Wgpu, 3> {
        // Sum audio embeddings for all codebooks
        let mut input =
            Tensor::<Wgpu, 3>::zeros([1, 1, self.config.hidden_size], &self.device);
        for (i, &token) in audio_tokens.iter().enumerate() {
            let emb = self.audio_emb[i].embed_id(token, &self.device);
            input = input + emb.unsqueeze_dim::<3>(0);
        }

        // Add text embedding
        let text_emb = self.text_emb.embed_id(text_token, &self.device);
        input = input + text_emb.unsqueeze_dim::<3>(0);

        // Run transformer layers with KV cache
        let mut x = input;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(c) = cache.get_mut(i) {
                x = layer.forward_with_cache(x, &self.rope, c);
            }
        }

        // Output norm + text linear → logits
        let x = self.out_norm.forward(x);
        self.text_linear.forward(x)
    }

    /// Create a new LayerCaches for this model.
    pub fn create_cache(&self) -> LayerCaches {
        LayerCaches::new(self.config.num_layers)
    }

    /// Reset KV cache (for new utterance).
    pub fn reset_cache(cache: &mut LayerCaches) {
        cache.reset();
    }

    pub fn config(&self) -> &SttConfig {
        &self.config
    }

    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }
}
