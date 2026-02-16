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

use conv::{Conv1d, ConvDownsample};
use ndarray::{Array1, Array2, Array3};
use quantization::{ResidualVectorQuantizer, SplitResidualVectorQuantizer, VectorQuantizer};
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

        // Use global fetch() which works in both Window and Worker contexts.
        let global = js_sys::global();
        let fetch_fn = js_sys::Reflect::get(&global, &"fetch".into())
            .map_err(|e| MimiError::Network(format!("No fetch function: {e:?}")))?;
        let fetch_fn: js_sys::Function = fetch_fn
            .dyn_into()
            .map_err(|_| MimiError::Network("fetch is not a function".into()))?;
        let resp_promise = fetch_fn
            .call1(&JsValue::UNDEFINED, &JsValue::from_str(url))
            .map_err(|e| MimiError::Network(format!("fetch call failed: {e:?}")))?;
        let resp_value = JsFuture::from(js_sys::Promise::from(resp_promise))
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

    /// Load a conv weight from safetensors, handling weight_norm decomposition.
    ///
    /// Tries pre-merged `.weight` first; falls back to `.weight_g` + `.weight_v`.
    fn load_conv_weight(
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<(Array3<f32>, Option<Array1<f32>>), MimiError> {
        let weight = match tensors.tensor(&format!("{prefix}.weight")) {
            Ok(view) => weights::to_array3(view)?,
            Err(_) => {
                // Weight norm decomposition: weight = weight_v * (weight_g / ||weight_v||)
                let weight_g_view =
                    weights::get_tensor(tensors, &format!("{prefix}.weight_g"))?;
                let weight_v =
                    weights::to_array3(weights::get_tensor(tensors, &format!("{prefix}.weight_v"))?)?;
                let (out_c, in_c, kernel) =
                    (weight_v.shape()[0], weight_v.shape()[1], weight_v.shape()[2]);

                // weight_g may be 3D (out_c,1,1) or 1D (out_c,)
                let weight_g_shape = weight_g_view.shape();
                let weight_g_flat: Vec<f32> = weight_g_view
                    .data()
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect();

                let mut weight = Array3::<f32>::zeros((out_c, in_c, kernel));
                for o in 0..out_c {
                    let g = if weight_g_shape.len() == 1 {
                        weight_g_flat[o]
                    } else {
                        weight_g_flat[o] // (out_c, 1, 1) → index o*1*1 = o
                    };
                    // L2 norm of weight_v for output channel o
                    let mut norm_sq = 0.0f32;
                    for i in 0..in_c {
                        for k in 0..kernel {
                            let v = weight_v[[o, i, k]];
                            norm_sq += v * v;
                        }
                    }
                    let scale = g / norm_sq.sqrt().max(1e-12);
                    for i in 0..in_c {
                        for k in 0..kernel {
                            weight[[o, i, k]] = weight_v[[o, i, k]] * scale;
                        }
                    }
                }
                weight
            }
        };
        let bias = weights::get_optional_bias(tensors, &format!("{prefix}.bias"))?;
        Ok((weight, bias))
    }

    /// Build the SEANet encoder from safetensors weights.
    ///
    /// Sequential layout (from Python nn.Sequential):
    ///   0: init_conv
    ///   For each of 4 ratios [4,5,6,8] (reversed from decoder order [8,6,5,4]):
    ///     {idx}: ResidualBlock  (1 per layer, n_residual_layers=1)
    ///     {idx+1}: ELU (no weights)
    ///     {idx+2}: Downsample Conv1d
    ///   {last-1}: ELU (no weights)
    ///   {last}: final_conv
    fn build_encoder(
        tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<SeaNetEncoder, MimiError> {
        // Encoder ratios (reversed from decoder [8,6,5,4])
        let ratios_reversed: [usize; 4] = [4, 5, 6, 8];

        // Init conv at encoder.model.0 (1 → 64, kernel=7)
        let (init_w, init_b) =
            Self::load_conv_weight(tensors, "encoder.model.0.conv.conv")?;
        let init_conv = Conv1d::new(init_w, init_b, 1, 1, 1, true);

        let mut layers = Vec::new();
        // Sequential index tracking:
        //   0 = init_conv
        //   then for each ratio: resblock, ELU, downsample_conv
        let mut seq_idx: usize = 1;

        for &ratio in &ratios_reversed {
            // Residual block (n_residual_layers = 1, dilation_base = 2^j, j=0 → dilation=1)
            let block_prefix = format!("encoder.model.{seq_idx}");

            // Block Sequential: [ELU, conv1, ELU, conv2] → indices 1 and 3 have weights
            let (w1, b1) = Self::load_conv_weight(
                tensors,
                &format!("{block_prefix}.block.1.conv.conv"),
            )?;
            let conv1 = Conv1d::new(w1, b1, 1, 1, 1, true);

            let (w2, b2) = Self::load_conv_weight(
                tensors,
                &format!("{block_prefix}.block.3.conv.conv"),
            )?;
            let conv2 = Conv1d::new(w2, b2, 1, 1, 1, true);

            // Shortcut (true_skip=false → learned shortcut conv, kernel=1)
            let (ws, bs) = Self::load_conv_weight(
                tensors,
                &format!("{block_prefix}.shortcut.conv.conv"),
            )?;
            let shortcut = Some(Conv1d::new(ws, bs, 1, 1, 1, true));

            let residual_block = seanet::ResidualBlock {
                conv1,
                conv2,
                shortcut,
            };

            seq_idx += 1; // past the resblock

            // ELU activation at seq_idx (no weights, skip)
            // Downsample conv at seq_idx + 1
            let ds_prefix = format!("encoder.model.{}", seq_idx + 1);
            let (ds_w, ds_b) =
                Self::load_conv_weight(tensors, &format!("{ds_prefix}.conv.conv"))?;
            let downsample = Conv1d::new(ds_w, ds_b, ratio, 1, 1, true);

            layers.push(seanet::EncoderLayer {
                residual_blocks: vec![residual_block],
                downsample,
            });

            seq_idx += 2; // past ELU + downsample
        }

        // Final conv: ELU at seq_idx (no weights), conv at seq_idx + 1
        let final_prefix = format!("encoder.model.{}", seq_idx + 1);
        let (final_w, final_b) =
            Self::load_conv_weight(tensors, &format!("{final_prefix}.conv.conv"))?;
        let final_conv = Conv1d::new(final_w, final_b, 1, 1, 1, true);

        Ok(SeaNetEncoder {
            init_conv,
            layers,
            final_conv,
        })
    }

    /// Build the 8-layer encoder transformer from safetensors weights.
    fn build_transformer(
        tensors: &safetensors::SafeTensors,
        config: &MimiConfig,
    ) -> Result<Transformer, MimiError> {
        let d_model = config.dimension; // 512
        let num_heads = 8;
        let num_layers = 8;
        let eps = 1e-5;

        let mut layers = Vec::new();

        for i in 0..num_layers {
            let prefix = format!("encoder_transformer.transformer.layers.{i}");

            // Self-attention (combined QKV in_proj)
            let in_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.in_proj.weight"))?,
            )?;
            let out_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.out_proj.weight"))?,
            )?;
            let attn = transformer::MultiHeadAttention::new(d_model, num_heads, in_proj, out_proj);

            // LayerNorm 1 — try .weight then .alpha
            let norm1_weight = match tensors.tensor(&format!("{prefix}.norm1.weight")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => weights::to_array1(
                    weights::get_tensor(tensors, &format!("{prefix}.norm1.alpha"))?,
                )?,
            };
            let norm1_bias = match tensors.tensor(&format!("{prefix}.norm1.bias")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => Array1::zeros(d_model),
            };
            let norm1 = transformer::LayerNorm::new(norm1_weight, norm1_bias, eps);

            // LayerNorm 2
            let norm2_weight = match tensors.tensor(&format!("{prefix}.norm2.weight")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => weights::to_array1(
                    weights::get_tensor(tensors, &format!("{prefix}.norm2.alpha"))?,
                )?,
            };
            let norm2_bias = match tensors.tensor(&format!("{prefix}.norm2.bias")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => Array1::zeros(d_model),
            };
            let norm2 = transformer::LayerNorm::new(norm2_weight, norm2_bias, eps);

            // Feed-forward MLP
            let fc1 = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.mlp.linear1.weight"))?,
            )?;
            let fc2 = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.mlp.linear2.weight"))?,
            )?;
            let bias1 =
                weights::get_optional_bias(tensors, &format!("{prefix}.mlp.linear1.bias"))?;
            let bias2 =
                weights::get_optional_bias(tensors, &format!("{prefix}.mlp.linear2.bias"))?;
            let ff = transformer::FeedForward::new(fc1, fc2, bias1, bias2);

            // Layer scales (optional)
            let layer_scale_1 =
                match tensors.tensor(&format!("{prefix}.layer_scale_1.scale")) {
                    Ok(view) => Some(weights::to_array1(view)?),
                    Err(_) => None,
                };
            let layer_scale_2 =
                match tensors.tensor(&format!("{prefix}.layer_scale_2.scale")) {
                    Ok(view) => Some(weights::to_array1(view)?),
                    Err(_) => None,
                };

            layers.push(transformer::TransformerLayer {
                attn,
                ff,
                norm1,
                norm2,
                layer_scale_1,
                layer_scale_2,
            });
        }

        Ok(Transformer::new(layers, d_model))
    }

    /// Build the ConvDownsample from safetensors weights.
    ///
    /// The downsample sits between the encoder transformer and quantizer.
    /// Stride is inferred from kernel size: stride = kernel_size / 2.
    fn build_downsample(
        tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<ConvDownsample, MimiError> {
        let (weight, bias) =
            Self::load_conv_weight(tensors, "downsample.conv.conv")?;
        let kernel_size = weight.shape()[2];
        let stride = kernel_size / 2;
        Ok(ConvDownsample::new(weight, bias, stride))
    }

    /// Build the split residual vector quantizer from safetensors weights.
    ///
    /// rvq_first: 1 semantic codebook with input/output projections
    /// rvq_rest: 31 acoustic codebooks with input/output projections
    ///
    /// Codebook normalization: embedding = embedding_sum / max(cluster_usage, 1.0)
    fn build_quantizer(
        tensors: &safetensors::SafeTensors,
        config: &MimiConfig,
    ) -> Result<SplitResidualVectorQuantizer, MimiError> {
        // Load rvq_first (1 semantic codebook)
        let first_input_proj = weights::to_array2(
            weights::get_tensor(tensors, "quantizer.rvq_first.input_proj.weight")?,
        )?;
        let first_output_proj = weights::to_array2(
            weights::get_tensor(tensors, "quantizer.rvq_first.output_proj.weight")?,
        )?;
        let first_codebook = Self::load_codebook(
            tensors,
            "quantizer.rvq_first.vq.layers.0._codebook",
        )?;
        let rvq_first = ResidualVectorQuantizer::new(
            first_input_proj,
            first_output_proj,
            vec![VectorQuantizer::new(first_codebook)],
        );

        // Load rvq_rest (31 acoustic codebooks)
        let rest_input_proj = weights::to_array2(
            weights::get_tensor(tensors, "quantizer.rvq_rest.input_proj.weight")?,
        )?;
        let rest_output_proj = weights::to_array2(
            weights::get_tensor(tensors, "quantizer.rvq_rest.output_proj.weight")?,
        )?;
        let n_rest = config.num_codebooks - 1; // 31
        let mut rest_quantizers = Vec::with_capacity(n_rest);
        for i in 0..n_rest {
            let codebook = Self::load_codebook(
                tensors,
                &format!("quantizer.rvq_rest.vq.layers.{i}._codebook"),
            )?;
            rest_quantizers.push(VectorQuantizer::new(codebook));
        }
        let rvq_rest = ResidualVectorQuantizer::new(
            rest_input_proj,
            rest_output_proj,
            rest_quantizers,
        );

        Ok(SplitResidualVectorQuantizer::new(
            rvq_first,
            rvq_rest,
            config.num_codebooks,
        ))
    }

    /// Load and normalize a single codebook from safetensors.
    ///
    /// embedding = embedding_sum / max(cluster_usage, 1.0) (per-row normalization)
    fn load_codebook(
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<Array2<f32>, MimiError> {
        let embedding_sum = weights::to_array2(
            weights::get_tensor(tensors, &format!("{prefix}.embedding_sum"))?,
        )?;
        let cluster_usage = weights::to_array1(
            weights::get_tensor(tensors, &format!("{prefix}.cluster_usage"))?,
        )?;

        let (num_bins, dim) = (embedding_sum.shape()[0], embedding_sum.shape()[1]);
        let mut codebook = Array2::<f32>::zeros((num_bins, dim));
        for bin in 0..num_bins {
            let usage = cluster_usage[bin].max(1.0);
            for d in 0..dim {
                codebook[[bin, d]] = embedding_sum[[bin, d]] / usage;
            }
        }
        Ok(codebook)
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
