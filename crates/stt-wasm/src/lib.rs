//! STT 1B — browser-native speech-to-text.
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

pub mod mimi_encoder;
pub mod mimi_remap;
pub mod tokenizer;

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
    /// Audio delay in seconds used by the reference to compute how many
    /// trailing silence frames to feed at end-of-turn (`audio_delay_seconds`
    /// in the model config, independent of `text_delay`).
    pub audio_delay_seconds: f64,
    /// Mimi codec frame rate in Hz.
    pub frame_rate_hz: f64,
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

impl SttConfig {
    /// Number of trailing silence frames to feed at end-of-turn, matching the
    /// reference's `n_suffix_chunks = ceil(audio_delay_seconds * frame_rate)`.
    /// This is distinct from `text_delay`: `text_delay` governs *when* text
    /// starts emitting mid-stream (`step_idx >= text_delay`), while this
    /// governs how long the delay pipeline must be drained after the last
    /// real audio frame so the model's lookahead gets to run out.
    pub fn flush_drain_frames(&self) -> usize {
        (self.audio_delay_seconds * self.frame_rate_hz).ceil() as usize
    }
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
            text_delay: 6,
            audio_delay_seconds: 0.5,
            frame_rate_hz: 12.5,
            rope_theta: 100000.0,
            max_seq_len: 4096,
            sliding_window: 750,
            text_padding_id: 3,
            text_start_token: 8000, // text_in_vocab_size - 1
        }
    }
}
