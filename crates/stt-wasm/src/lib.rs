//! STT 1B — browser-native speech-to-text.
//!
//! Decoder-only transformer consuming Mimi audio codec tokens (32 codebooks at 12.5Hz)
//! and producing text tokens on a delayed parallel stream (7 frames / 560ms offset).
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
    /// Feed-forward intermediate size.
    pub intermediate_size: usize,
    /// Text output vocabulary size (text_linear out dim).
    pub vocab_size: usize,
    /// Text input vocabulary size (text_emb rows).
    pub text_in_vocab_size: usize,
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
    /// Text start token ID (fed on first step; = text_in_vocab_size - 1).
    pub text_start_token: u32,
}

impl Default for SttConfig {
    fn default() -> Self {
        // Verified values from kyutai/stt-1b-en_fr config.json
        Self {
            num_layers: 16,
            hidden_size: 2048,
            num_heads: 16,
            num_kv_heads: 16,
            intermediate_size: 5632,
            vocab_size: 8000,
            text_in_vocab_size: 8001,
            num_codebooks: 32,
            audio_vocab_size: 2049,
            text_delay: 7,
            rope_theta: 100000.0,
            max_seq_len: 4096,
            sliding_window: 750,
            text_padding_id: 3,
            text_start_token: 8000, // text_in_vocab_size - 1
        }
    }
}
