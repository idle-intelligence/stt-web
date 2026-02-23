//! Mimi audio encoder wrapper using mimi-rs (candle-based).
//!
//! Provides streaming Mimi encoding (audio → codec tokens) for both
//! native tests and WASM bindings. Uses the shared mimi-rs library
//! with automatic key remapping for the full Mimi model.

/// Wraps mimi-rs MimiModel + streaming encoder state for the STT pipeline.
pub struct MimiEncoder {
    model: mimi_rs::mimi::MimiModel,
    state: mimi_rs::mimi::MimiEncoderState,
    num_codebooks: usize,
}

impl MimiEncoder {
    /// Load from safetensors bytes with key remapping for the full Mimi model.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let remapped = crate::mimi_remap::remap_mimi_weights(data);
        let vb = candle_nn::VarBuilder::from_buffered_safetensors(
            remapped,
            candle_core::DType::F32,
            &candle_core::Device::Cpu,
        )
        .map_err(|e| format!("Failed to create VarBuilder: {e}"))?;

        let cfg = mimi_rs::config::MimiConfig::mimi_v1_0_0();
        let mut model = mimi_rs::mimi::MimiModel::load_encoder_only(vb, &cfg)
            .map_err(|e| format!("Failed to load Mimi model: {e}"))?;

        // Quantize encoder transformer to Q8_0 for faster inference
        model
            .quantize_encoder_transformer(candle_core::quantized::GgmlDType::Q8_0)
            .map_err(|e| format!("Failed to quantize encoder transformer: {e}"))?;

        let state = model
            .init_encoder_state(1, &candle_core::Device::Cpu)
            .map_err(|e| format!("Failed to init encoder state: {e}"))?;

        Ok(Self {
            model,
            state,
            num_codebooks: cfg.num_codebooks,
        })
    }

    /// Number of codebooks (32 for standard Mimi).
    pub fn num_codebooks(&self) -> usize {
        self.num_codebooks
    }

    /// Feed audio samples and return tokens as a flat Vec<u32>.
    ///
    /// Output format: `[frame0_tok0..tok31, frame1_tok0..tok31, ...]`
    /// May return empty if not enough audio has accumulated.
    pub fn feed_audio(&mut self, samples: &[f32]) -> Vec<u32> {
        if samples.is_empty() {
            return Vec::new();
        }

        let tensor = match candle_core::Tensor::from_vec(
            samples.to_vec(),
            (1, 1, samples.len()),
            &candle_core::Device::Cpu,
        ) {
            Ok(t) => t,
            Err(_) => return Vec::new(),
        };

        // Streaming encode: audio → latent
        let latent = match self.model.encode_streaming(&tensor, &mut self.state) {
            Ok(l) => l,
            Err(_) => return Vec::new(),
        };

        // Check if any frames were produced
        let n_frames = latent.dim(2).unwrap_or(0);
        if n_frames == 0 {
            return Vec::new();
        }

        // Quantize latent → token IDs [1, n_q, T']
        let codes = match self.model.quantize_to_codes(&latent) {
            Ok(c) => c,
            Err(_) => return Vec::new(),
        };

        // Flatten [1, n_q, T'] to frame-major order: [T' * n_q]
        let codes_flat = match codes.squeeze(0) {
            Ok(c) => match c.to_vec2::<u32>() {
                Ok(v) => v,
                Err(_) => return Vec::new(),
            },
            Err(_) => return Vec::new(),
        };

        let n_q = codes_flat.len();
        let mut tokens = Vec::with_capacity(n_frames * n_q);
        for f in 0..n_frames {
            for q in 0..n_q {
                tokens.push(codes_flat[q][f]);
            }
        }
        tokens
    }

    /// Batch-encode all audio at once (non-streaming).
    ///
    /// Creates a fresh encoder state, processes all samples in one call.
    /// Output format matches `feed_audio`: frame-major flat `Vec<u32>`.
    pub fn encode_all(&mut self, samples: &[f32]) -> Vec<u32> {
        // Reset state for batch mode
        self.reset();
        self.feed_audio(samples)
    }

    /// Reset streaming state for a new recording session.
    pub fn reset(&mut self) {
        if let Ok(new_state) = self
            .model
            .init_encoder_state(1, &candle_core::Device::Cpu)
        {
            self.state = new_state;
        }
    }
}
