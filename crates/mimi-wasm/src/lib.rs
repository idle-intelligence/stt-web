//! Mimi audio codec compiled to WASM.
//!
//! Converts 16kHz mono PCM audio into discrete tokens at 12.5Hz
//! (32 codebook tokens per frame). Based on `rustymimi` from kyutai-labs/moshi,
//! stripped of pyo3/CUDA dependencies for wasm32-unknown-unknown target.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Error type for Mimi codec operations.
#[derive(Debug, thiserror::Error)]
pub enum MimiError {
    #[error("Failed to load weights: {0}")]
    WeightLoad(String),

    #[error("Encoding error: {0}")]
    Encode(String),
}

/// Mimi audio codec instance.
///
/// Encodes raw PCM audio into 32-codebook token frames at 12.5Hz.
/// Runs entirely on CPU — no GPU required.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct MimiCodec {
    // TODO: internal state from rustymimi port
    _placeholder: (),
}

// WASM API
#[cfg(feature = "wasm")]
#[wasm_bindgen]
impl MimiCodec {
    /// Create a new codec instance. Downloads/loads Mimi weights from the given URL.
    #[wasm_bindgen(constructor)]
    pub async fn new(weights_url: &str) -> Result<MimiCodec, JsError> {
        console_error_panic_hook::set_once();
        Self::create(weights_url).await.map_err(|e| JsError::new(&e.to_string()))
    }

    /// Feed a chunk of f32 PCM audio (16kHz mono).
    ///
    /// Returns token IDs as a flat array:
    /// `[frame0_tok0, frame0_tok1, ..., frame0_tok31, frame1_tok0, ...]`
    ///
    /// May return empty if not enough audio has accumulated for a full frame.
    #[wasm_bindgen(js_name = feedAudio)]
    pub fn feed_audio_js(&mut self, samples: &[f32]) -> Vec<u32> {
        self.feed_audio(samples)
    }

    /// Reset internal state (e.g., when user stops and restarts recording).
    #[wasm_bindgen]
    pub fn reset(&mut self) {
        self.reset_inner();
    }
}

// Shared implementation (native + WASM)
impl MimiCodec {
    async fn create(_weights_source: &str) -> Result<MimiCodec, MimiError> {
        todo!("Port rustymimi: load weights, initialize encoder/decoder/quantizer")
    }

    pub fn feed_audio(&mut self, _samples: &[f32]) -> Vec<u32> {
        todo!("Buffer audio, run Mimi encoder when enough samples accumulated")
    }

    fn reset_inner(&mut self) {
        todo!("Clear audio buffer, reset encoder/quantizer state")
    }
}

// Native API
#[cfg(not(feature = "wasm"))]
impl MimiCodec {
    /// Create a new codec instance (native path).
    pub async fn new(weights_path: &str) -> Result<MimiCodec, MimiError> {
        Self::create(weights_path).await
    }

    pub fn reset(&mut self) {
        self.reset_inner();
    }
}
