//! Delayed-streams decoding loop.
//!
//! Audio and text run as parallel time-aligned streams.
//! Text is delayed by `delay` frames (7 frames = 560ms for stt-1b-en_fr).
//! Each step: model receives current audio frame + previous text token,
//! predicts next text token.

use burn::backend::wgpu::Wgpu;
use burn::tensor::{Int, Tensor};

use crate::model::{LayerCaches, SttModel};
use crate::SttConfig;

/// Streaming STT decoder implementing delayed-streams logic.
pub struct SttStream {
    config: SttConfig,
    cache: LayerCaches,
    frame_count: usize,
    last_text_token: u32,
}

impl SttStream {
    /// Create a new streaming decoder.
    pub fn new(config: SttConfig, num_layers: usize) -> Self {
        Self {
            cache: LayerCaches::new(num_layers),
            last_text_token: config.text_start_token,
            frame_count: 0,
            config,
        }
    }

    /// Feed one frame of Mimi tokens (32 u32 values).
    ///
    /// Returns decoded text token if past the delay offset.
    pub async fn feed_frame(
        &mut self,
        audio_tokens: &[u32],
        model: &SttModel,
    ) -> Option<u32> {
        self.frame_count += 1;

        // During delay: run model to fill KV cache but don't emit text.
        // Still extract and track the predicted token so we feed the model's
        // own predictions (not text_start_token repeatedly) on subsequent frames.
        if self.frame_count <= self.config.text_delay {
            let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
            let _ = self.extract_token(logits).await; // updates last_text_token
            return None;
        }

        // Normal forward path
        let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
        self.extract_token(logits).await
    }

    /// Extract token from logits via argmax + async readback.
    async fn extract_token(&mut self, logits: Tensor<Wgpu, 3>) -> Option<u32> {
        let pred = logits.argmax(2);
        let token = match Tensor::<Wgpu, 3, Int>::into_data_async(pred).await {
            Ok(data) => match data.to_vec::<i32>() {
                Ok(vec) => vec[0] as u32,
                Err(e) => {
                    Self::log(&format!(
                        "[stt] GPU readback to_vec FAILED (frame {}): {:?}",
                        self.frame_count, e
                    ));
                    self.config.text_padding_id
                }
            },
            Err(e) => {
                Self::log(&format!(
                    "[stt] GPU readback into_data_async FAILED (frame {}): {:?}",
                    self.frame_count, e
                ));
                self.config.text_padding_id
            }
        };
        self.last_text_token = token;
        Some(token)
    }

    /// Log to console (WASM) or stderr (native).
    fn log(msg: &str) {
        #[cfg(target_family = "wasm")]
        web_sys::console::warn_1(&msg.into());
        #[cfg(not(target_family = "wasm"))]
        eprintln!("{}", msg);
    }

    /// Flush remaining text after VAD detects end of speech.
    pub async fn flush(&mut self, model: &SttModel) -> Vec<u32> {
        let mut tokens = Vec::new();
        let zero_audio = vec![0u32; self.config.num_codebooks];

        for _ in 0..self.config.text_delay {
            let logits = model.forward(&zero_audio, self.last_text_token, &mut self.cache);
            if let Some(token) = self.extract_token(logits).await {
                tokens.push(token);
            }
        }

        tokens
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.last_text_token = self.config.text_start_token;
        self.cache.reset();
    }
}
