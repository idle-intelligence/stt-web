//! Kyutai STT 1B — browser-native speech-to-text.
//!
//! Decoder-only transformer consuming Mimi audio codec tokens (32 codebooks at 12.5Hz)
//! and producing text tokens on a delayed parallel stream (6 frames / 480ms offset).
//!
//! Uses Burn's wgpu backend for GPU inference — works natively (Vulkan/Metal) and
//! in the browser (WASM + WebGPU).

#[cfg(feature = "wgpu")]
pub mod model;

#[cfg(feature = "wgpu")]
pub mod gguf;

#[cfg(feature = "wgpu")]
pub mod stream;

#[cfg(feature = "wasm")]
pub mod web;

/// Model configuration matching `kyutai/stt-1b-en_fr`.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct SttConfig {
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Hidden dimension.
    pub hidden_size: usize,
    /// Number of attention heads (queries).
    pub num_heads: usize,
    /// Number of key-value heads (for GQA; same as num_heads for MHA).
    pub num_kv_heads: usize,
    /// Feed-forward intermediate size (hidden_size * hidden_scale).
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
    /// Sliding window size for attention.
    pub sliding_window: usize,
    /// Text padding token ID.
    pub text_padding_id: u32,
}

impl Default for SttConfig {
    fn default() -> Self {
        // Verified values from kyutai/stt-1b-en_fr config.json
        Self {
            num_layers: 16,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            intermediate_size: 8448, // 2048 * 4.125
            vocab_size: 8000,
            num_codebooks: 32,
            audio_vocab_size: 2048,
            text_delay: 6,
            rope_theta: 100000.0,
            max_seq_len: 4096,
            sliding_window: 750,
            text_padding_id: 3,
        }
    }
}
