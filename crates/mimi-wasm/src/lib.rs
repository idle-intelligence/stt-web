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

    /// Load a conv weight+bias from safetensors.
    fn load_conv(
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<(Array3<f32>, Option<Array1<f32>>), MimiError> {
        let weight = weights::to_array3(
            weights::get_tensor(tensors, &format!("{prefix}.weight"))?,
        )?;
        let bias = weights::get_optional_bias(tensors, &format!("{prefix}.bias"))?;
        Ok((weight, bias))
    }

    /// Build the SEANet encoder from safetensors weights.
    ///
    /// Actual tensor layout (from mimi safetensors):
    ///   encoder.layers.0.conv  → init conv [64, 1, 7]
    ///   encoder.layers.{1,4,7,10}.block.{1,3}.conv → residual blocks
    ///   encoder.layers.{3,6,9,12}.conv → downsample convs
    ///   encoder.layers.14.conv → final conv [512, 1024, 3]
    ///
    /// Ratios [4,5,6,8] with strides matching kernel_size/2.
    fn build_encoder(
        tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<SeaNetEncoder, MimiError> {
        // Init conv: encoder.layers.0.conv (1→64, kernel=7)
        let (init_w, init_b) = Self::load_conv(tensors, "encoder.layers.0.conv")?;
        let init_conv = Conv1d::new(init_w, init_b, 1, 1, 1, true);

        // Residual block indices and downsample indices (from inspecting safetensors):
        // res_block at layers.1, downsample at layers.3 (stride=4, kernel=8)
        // res_block at layers.4, downsample at layers.6 (stride=5, kernel=10)
        // res_block at layers.7, downsample at layers.9 (stride=6, kernel=12)
        // res_block at layers.10, downsample at layers.12 (stride=8, kernel=16)
        let layer_plan: [(usize, usize); 4] = [
            (1, 3),   // res_block idx, downsample idx
            (4, 6),
            (7, 9),
            (10, 12),
        ];

        let mut layers = Vec::new();
        for &(res_idx, ds_idx) in &layer_plan {
            // Residual block: block.1 = first conv (with ELU before), block.3 = second conv
            let (w1, b1) = Self::load_conv(
                tensors,
                &format!("encoder.layers.{res_idx}.block.1.conv"),
            )?;
            let conv1 = Conv1d::new(w1, b1, 1, 1, 1, true);

            let (w2, b2) = Self::load_conv(
                tensors,
                &format!("encoder.layers.{res_idx}.block.3.conv"),
            )?;
            let conv2 = Conv1d::new(w2, b2, 1, 1, 1, true);

            // No shortcut conv in this model (true_skip=true, identity when dims match)
            let residual_block = seanet::ResidualBlock {
                conv1,
                conv2,
                shortcut: None,
            };

            // Downsample conv: stride = kernel_size / 2
            let (ds_w, ds_b) = Self::load_conv(
                tensors,
                &format!("encoder.layers.{ds_idx}.conv"),
            )?;
            let kernel_size = ds_w.shape()[2];
            let stride = kernel_size / 2;
            let downsample = Conv1d::new(ds_w, ds_b, stride, 1, 1, true);

            layers.push(seanet::EncoderLayer {
                residual_blocks: vec![residual_block],
                downsample,
            });
        }

        // Final conv: encoder.layers.14.conv [512, 1024, 3]
        let (final_w, final_b) = Self::load_conv(tensors, "encoder.layers.14.conv")?;
        let final_conv = Conv1d::new(final_w, final_b, 1, 1, 1, true);

        Ok(SeaNetEncoder {
            init_conv,
            layers,
            final_conv,
        })
    }

    /// Build the 8-layer encoder transformer from safetensors weights.
    ///
    /// Actual tensor names:
    ///   encoder_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    ///   encoder_transformer.layers.{i}.input_layernorm.{weight,bias}
    ///   encoder_transformer.layers.{i}.post_attention_layernorm.{weight,bias}
    ///   encoder_transformer.layers.{i}.mlp.fc1.weight / fc2.weight
    ///   encoder_transformer.layers.{i}.self_attn_layer_scale.scale
    ///   encoder_transformer.layers.{i}.mlp_layer_scale.scale
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
            let prefix = format!("encoder_transformer.layers.{i}");

            // Separate Q/K/V/O projections
            let q_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.q_proj.weight"))?,
            )?;
            let k_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.k_proj.weight"))?,
            )?;
            let v_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.v_proj.weight"))?,
            )?;
            let o_proj = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.self_attn.o_proj.weight"))?,
            )?;
            let attn = transformer::MultiHeadAttention::new(
                d_model, num_heads, q_proj, k_proj, v_proj, o_proj,
            );

            // input_layernorm (pre-attention norm)
            let norm1_weight = weights::to_array1(
                weights::get_tensor(tensors, &format!("{prefix}.input_layernorm.weight"))?,
            )?;
            let norm1_bias = match tensors.tensor(&format!("{prefix}.input_layernorm.bias")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => Array1::zeros(d_model),
            };
            let norm1 = transformer::LayerNorm::new(norm1_weight, norm1_bias, eps);

            // post_attention_layernorm (pre-FFN norm)
            let norm2_weight = weights::to_array1(
                weights::get_tensor(tensors, &format!("{prefix}.post_attention_layernorm.weight"))?,
            )?;
            let norm2_bias = match tensors.tensor(&format!("{prefix}.post_attention_layernorm.bias")) {
                Ok(view) => weights::to_array1(view)?,
                Err(_) => Array1::zeros(d_model),
            };
            let norm2 = transformer::LayerNorm::new(norm2_weight, norm2_bias, eps);

            // Feed-forward MLP: mlp.fc1 / mlp.fc2
            let fc1 = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.mlp.fc1.weight"))?,
            )?;
            let fc2 = weights::to_array2(
                weights::get_tensor(tensors, &format!("{prefix}.mlp.fc2.weight"))?,
            )?;
            let ff = transformer::FeedForward::new(fc1, fc2, None, None);

            // Layer scales
            let layer_scale_1 =
                match tensors.tensor(&format!("{prefix}.self_attn_layer_scale.scale")) {
                    Ok(view) => Some(weights::to_array1(view)?),
                    Err(_) => None,
                };
            let layer_scale_2 =
                match tensors.tensor(&format!("{prefix}.mlp_layer_scale.scale")) {
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
    /// Actual tensor: downsample.conv.weight [512, 512, 4]
    /// Stride is inferred from kernel size: stride = kernel_size / 2.
    fn build_downsample(
        tensors: &safetensors::SafeTensors,
        _config: &MimiConfig,
    ) -> Result<ConvDownsample, MimiError> {
        let weight = weights::to_array3(
            weights::get_tensor(tensors, "downsample.conv.weight")?,
        )?;
        let bias = weights::get_optional_bias(tensors, "downsample.conv.bias")?;
        let kernel_size = weight.shape()[2];
        let stride = kernel_size / 2;
        Ok(ConvDownsample::new(weight, bias, stride))
    }

    /// Build the split residual vector quantizer from safetensors weights.
    ///
    /// Actual tensor names:
    ///   quantizer.semantic_residual_vector_quantizer.{input,output}_proj.weight [dim, dim, 1]
    ///   quantizer.semantic_residual_vector_quantizer.layers.0.codebook.{embed_sum, cluster_usage}
    ///   quantizer.acoustic_residual_vector_quantizer.{input,output}_proj.weight [dim, dim, 1]
    ///   quantizer.acoustic_residual_vector_quantizer.layers.{0-30}.codebook.{embed_sum, cluster_usage}
    fn build_quantizer(
        tensors: &safetensors::SafeTensors,
        config: &MimiConfig,
    ) -> Result<SplitResidualVectorQuantizer, MimiError> {
        let sem = "quantizer.semantic_residual_vector_quantizer";
        let aco = "quantizer.acoustic_residual_vector_quantizer";

        // Semantic (rvq_first): 1 codebook
        // input_proj is 3D [256, 512, 1] (1x1 conv) — squeeze to 2D [256, 512]
        let first_input_proj = Self::load_proj(tensors, &format!("{sem}.input_proj.weight"))?;
        let first_output_proj = Self::load_proj(tensors, &format!("{sem}.output_proj.weight"))?;
        let first_codebook = Self::load_codebook(tensors, &format!("{sem}.layers.0.codebook"))?;
        let rvq_first = ResidualVectorQuantizer::new(
            first_input_proj,
            first_output_proj,
            vec![VectorQuantizer::new(first_codebook)],
        );

        // Acoustic (rvq_rest): 31 codebooks
        let rest_input_proj = Self::load_proj(tensors, &format!("{aco}.input_proj.weight"))?;
        let rest_output_proj = Self::load_proj(tensors, &format!("{aco}.output_proj.weight"))?;
        let n_rest = config.num_codebooks - 1; // 31
        let mut rest_quantizers = Vec::with_capacity(n_rest);
        for i in 0..n_rest {
            let codebook = Self::load_codebook(
                tensors,
                &format!("{aco}.layers.{i}.codebook"),
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

    /// Load a projection weight that may be 3D [out, in, 1] (1x1 conv) or 2D [out, in].
    /// Returns 2D [out, in].
    fn load_proj(
        tensors: &safetensors::SafeTensors,
        name: &str,
    ) -> Result<Array2<f32>, MimiError> {
        let view = weights::get_tensor(tensors, name)?;
        let shape = view.shape();
        if shape.len() == 3 && shape[2] == 1 {
            // 3D [out, in, 1] → squeeze to 2D [out, in]
            let arr3 = weights::to_array3(view)?;
            let (d0, d1, _) = (arr3.shape()[0], arr3.shape()[1], arr3.shape()[2]);
            let mut arr2 = Array2::<f32>::zeros((d0, d1));
            for i in 0..d0 {
                for j in 0..d1 {
                    arr2[[i, j]] = arr3[[i, j, 0]];
                }
            }
            Ok(arr2)
        } else {
            weights::to_array2(view)
        }
    }

    /// Load and normalize a single codebook from safetensors.
    ///
    /// embedding = embed_sum / max(cluster_usage, 1.0)
    fn load_codebook(
        tensors: &safetensors::SafeTensors,
        prefix: &str,
    ) -> Result<Array2<f32>, MimiError> {
        let embedding_sum = weights::to_array2(
            weights::get_tensor(tensors, &format!("{prefix}.embed_sum"))?,
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

    /// Encode an entire waveform at once (batch mode).
    ///
    /// Processes all audio through the encoder pipeline in a single pass,
    /// avoiding the frame-boundary artifacts of per-frame encoding.
    /// Returns a flat Vec of token IDs: [frame0_tok0..tok31, frame1_tok0..tok31, ...]
    pub fn encode_all(&mut self, samples: &[f32]) -> Vec<u32> {
        let time = samples.len();
        let mut data = Array3::<f32>::zeros((1, 1, time));
        let data_slice = data.as_slice_mut().unwrap();
        data_slice[..time].copy_from_slice(samples);
        let input = Tensor3::new(data);

        // Full pipeline in one shot
        let encoded = self.encoder.forward(&input);
        let transformed = self.encoder_transformer.forward(&encoded);
        let downsampled = self.downsample.forward(&transformed);
        let codes = self.quantizer.encode(&downsampled);

        // codes shape: (batch=1, num_codebooks=32, num_frames)
        let num_frames = codes.shape()[2];
        let mut all_tokens = Vec::with_capacity(num_frames * self.config.num_codebooks);
        for f in 0..num_frames {
            for q in 0..self.config.num_codebooks {
                all_tokens.push(codes[[0, q, f]]);
            }
        }
        all_tokens
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
