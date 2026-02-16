//! Kyutai STT 1B — browser-native speech-to-text.
//!
//! Decoder-only transformer consuming Mimi audio codec tokens (32 codebooks at 12.5Hz)
//! and producing text tokens on a delayed parallel stream (6 frames / 480ms offset).
//!
//! Uses Burn's wgpu backend for GPU inference — works natively (Vulkan/Metal) and
//! in the browser (WASM + WebGPU).

pub mod model;

#[cfg(feature = "wgpu")]
pub mod gguf;

pub mod stream;

#[cfg(feature = "wasm")]
pub mod web;

use burn::backend::wgpu::{Wgpu, WgpuDevice};

/// Backend type alias — wgpu with f32 floats and i32 ints.
pub type Backend = Wgpu<f32, i32>;

/// Model configuration matching `kyutai/stt-1b-en_fr`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SttConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads (queries).
    pub num_heads: usize,
    /// Number of key-value heads (for GQA).
    pub num_kv_heads: usize,
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Text vocabulary size.
    pub vocab_size: usize,
    /// Number of audio codebooks (Mimi).
    pub num_codebooks: usize,
    /// Audio codebook vocabulary size.
    pub audio_vocab_size: usize,
    /// Delayed-streams text offset in frames.
    pub text_delay: usize,
    /// RoPE base frequency.
    pub rope_theta: f64,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

impl Default for SttConfig {
    fn default() -> Self {
        // Values for kyutai/stt-1b-en_fr — to be verified against model config.json
        Self {
            num_layers: 24,
            hidden_size: 2048,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 5632,
            vocab_size: 32000,
            num_codebooks: 32,
            audio_vocab_size: 2048,
            text_delay: 6,
            rope_theta: 10000.0,
            max_seq_len: 4096,
        }
    }
}
