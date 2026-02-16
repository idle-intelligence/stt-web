//! Mimi audio codec compiled to WASM.
//!
//! Converts 16kHz mono PCM audio into discrete tokens at 12.5Hz
//! (32 codebook tokens per frame). Based on `rustymimi` from kyutai-labs/moshi,
//! stripped of pyo3/CUDA dependencies for wasm32-unknown-unknown target.

mod conv;
mod quantization;
mod seanet;
mod tensor;
mod transformer;
mod weights;

use conv::{ConvDownsample, Conv1d};
use ndarray::Array3;
use quantization::SplitResidualVectorQuantizer;
use seanet::SeaNetEncoder;
use tensor::Tensor3;
use transformer::Transformer;

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Error type for Mimi codec operations.
#[derive(Debug, thiserror::Error)]
pub enum MimiError {
    #[error("Failed to load weights: {0}")]
    WeightLoad(String),

    #[error("Encoding error: {0}")]
    Encode(String),

    #[error("Network error: {0}")]
    Network(String),
}

/// Mimi codec configuration.
#[derive(Clone, Debug)]
pub struct MimiConfig {
    pub sample_rate: usize,      // Input sample rate (typically 24000 Hz)
    pub frame_rate: f64,          // Output frame rate (12.5 Hz)
    pub num_codebooks: usize,     // Number of codebooks (32)
    pub codebook_bins: usize,     // Bins per codebook (2048)
    pub dimension: usize,         // Internal dimension (512)
}

impl Default for MimiConfig {
    fn default() -> Self {
        Self {
            sample_rate: 24000,
            frame_rate: 12.5,
            num_codebooks: 32,
            codebook_bins: 2048,
            dimension: 512,
        }
    }
}

/// Mimi audio codec instance.
///
/// Encodes raw PCM audio into 32-codebook token frames at 12.5Hz.
/// Runs entirely on CPU — no GPU required.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub struct MimiCodec {
    config: MimiConfig,
    encoder: SeaNetEncoder,
    encoder_transformer: Transformer,
    downsample: ConvDownsample,
    quantizer: SplitResidualVectorQuantizer,
    // Audio buffer for streaming (accumulates until we have enough for one frame)
    audio_buffer: Vec<f32>,
    // Samples per frame (sample_rate / frame_rate)
    samples_per_frame: usize,
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

    /// Feed a chunk of f32 PCM audio (24kHz mono for Mimi).
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
    async fn create(weights_source: &str) -> Result<MimiCodec, MimiError> {
        let config = MimiConfig::default();

        // Fetch weights (WASM: fetch from URL, Native: read from file)
        let weights_data = Self::fetch_weights(weights_source).await?;

        // Parse safetensors
        let tensors = weights::load_safetensors(&weights_data)?;

        // Build encoder (placeholder - needs actual weight loading)
        // For now, create empty/stub components
        let encoder = Self::build_encoder(&tensors, &config)?;
        let encoder_transformer = Self::build_transformer(&tensors, &config)?;
        let downsample = Self::build_downsample(&tensors, &config)?;
        let quantizer = Self::build_quantizer(&tensors, &config)?;

        let samples_per_frame = (config.sample_rate as f64 / config.frame_rate) as usize;

        Ok(MimiCodec {
            config,
            encoder,
            encoder_transformer,
            downsample,
            quantizer,
            audio_buffer: Vec::new(),
            samples_per_frame,
        })
    }

    #[cfg(feature = "wasm")]
    async fn fetch_weights(url: &str) -> Result<Vec<u8>, MimiError> {
        use wasm_bindgen::JsCast;
        use wasm_bindgen_futures::JsFuture;

        let window = web_sys::window().ok_or_else(|| MimiError::Network("No window".into()))?;
        let resp_value = JsFuture::from(window.fetch_with_str(url))
            .await
            .map_err(|e| MimiError::Network(format!("{:?}", e)))?;
        let resp: web_sys::Response = resp_value
            .dyn_into()
            .map_err(|_| MimiError::Network("Invalid response".into()))?;

        let array_buffer = JsFuture::from(
            resp.array_buffer()
                .map_err(|e| MimiError::Network(format!("{:?}", e)))?,
        )
        .await
        .map_err(|e| MimiError::Network(format!("{:?}", e)))?;

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let mut data = vec![0u8; uint8_array.length() as usize];
        uint8_array.copy_to(&mut data);

        Ok(data)
    }

    #[cfg(not(feature = "wasm"))]
    async fn fetch_weights(path: &str) -> Result<Vec<u8>, MimiError> {
        std::fs::read(path).map_err(|e| MimiError::Network(e.to_string()))
    }

    fn build_encoder(
        _tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<SeaNetEncoder, MimiError> {
        // TODO: Load actual weights from safetensors
        // For now, return a placeholder
        Err(MimiError::WeightLoad(
            "Encoder weight loading not yet implemented".into(),
        ))
    }

    fn build_transformer(
        _tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<Transformer, MimiError> {
        // TODO: Load actual weights
        Err(MimiError::WeightLoad(
            "Transformer weight loading not yet implemented".into(),
        ))
    }

    fn build_downsample(
        _tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<ConvDownsample, MimiError> {
        // TODO: Load actual weights
        Err(MimiError::WeightLoad(
            "Downsample weight loading not yet implemented".into(),
        ))
    }

    fn build_quantizer(
        _tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<SplitResidualVectorQuantizer, MimiError> {
        // TODO: Load actual weights
        Err(MimiError::WeightLoad(
            "Quantizer weight loading not yet implemented".into(),
        ))
    }

    /// Feed audio samples and return tokens when a full frame is ready.
    pub fn feed_audio(&mut self, samples: &[f32]) -> Vec<u32> {
        // Accumulate samples
        self.audio_buffer.extend_from_slice(samples);

        let mut all_tokens = Vec::new();

        // Process complete frames
        while self.audio_buffer.len() >= self.samples_per_frame {
            // Extract one frame worth of audio
            let frame_samples: Vec<f32> =
                self.audio_buffer.drain(..self.samples_per_frame).collect();

            // Encode frame
            match self.encode_frame(&frame_samples) {
                Ok(tokens) => {
                    all_tokens.extend(tokens);
                }
                Err(_) => {
                    // For now, return zeros on error (placeholder)
                    all_tokens.extend(vec![0u32; self.config.num_codebooks]);
                }
            }
        }

        all_tokens
    }

    /// Encode a single frame of audio into 32 tokens.
    fn encode_frame(&mut self, samples: &[f32]) -> Result<Vec<u32>, MimiError> {
        // Reshape to (batch=1, channels=1, time)
        let time = samples.len();
        let mut data = Array3::<f32>::zeros((1, 1, time));
        for (i, &sample) in samples.iter().enumerate() {
            data[[0, 0, i]] = sample;
        }
        let input = Tensor3::new(data);

        // SEANet encoder
        let encoded = self.encoder.forward(&input);

        // Encoder transformer
        let transformed = self.encoder_transformer.forward(&encoded);

        // Downsample
        let downsampled = self.downsample.forward(&transformed);

        // Quantize to tokens
        let codes = self.quantizer.encode(&downsampled);

        // Extract tokens for batch 0, frame 0 (all 32 codebooks)
        let mut tokens = Vec::with_capacity(self.config.num_codebooks);
        for q in 0..self.config.num_codebooks {
            tokens.push(codes[[0, q, 0]]);
        }

        Ok(tokens)
    }

    fn reset_inner(&mut self) {
        self.audio_buffer.clear();
        self.encoder.reset();
        self.encoder_transformer.reset();
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
