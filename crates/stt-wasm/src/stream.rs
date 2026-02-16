//! Delayed-streams decoding loop.
//!
//! Audio and text run as parallel time-aligned streams.
//! Text is delayed by `delay` frames (6 frames = 480ms for stt-1b-en_fr).
//! Each step: model receives current audio frame + previous text token,
//! predicts next text token.
//!
//! "Flush trick": after VAD detects end-of-speech, run remaining buffered
//! frames faster than real-time to eliminate tail latency.

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
    audio_buffer: Vec<Vec<u32>>,
}

impl SttStream {
    /// Create a new streaming decoder.
    pub fn new(config: SttConfig, num_layers: usize) -> Self {
        Self {
            cache: LayerCaches::new(num_layers),
            last_text_token: config.text_padding_id,
            frame_count: 0,
            audio_buffer: Vec::new(),
            config,
        }
    }

    /// Feed one frame of Mimi tokens (32 u32 values).
    ///
    /// Returns decoded text token if past the delay offset.
    /// Uses `into_data_async` for WASM compatibility.
    pub async fn feed_frame(
        &mut self,
        audio_tokens: &[u32],
        model: &SttModel,
    ) -> Option<u32> {
        self.audio_buffer.push(audio_tokens.to_vec());
        self.frame_count += 1;

        // Text prediction starts after the delay offset
        if self.frame_count <= self.config.text_delay {
            // During delay: run model to fill KV cache but don't predict text
            let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
            // Discard logits during delay period
            let _ = logits;
            return None;
        }

        // Past delay: predict text token
        let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
        let pred = logits.argmax(2);

        // Async readback for WASM compatibility
        let token = match Tensor::<Wgpu, 3, Int>::into_data_async(pred).await {
            Ok(data) => match data.to_vec::<i32>() {
                Ok(vec) => vec[0] as u32,
                Err(_) => self.config.text_padding_id,
            },
            Err(_) => self.config.text_padding_id,
        };

        self.last_text_token = token;
        Some(token)
    }

    /// Flush remaining text after VAD detects end of speech.
    ///
    /// The "flush trick": continue running model steps with zero audio tokens
    /// to drain the remaining `text_delay` frames of buffered predictions.
    pub async fn flush(&mut self, model: &SttModel) -> Vec<u32> {
        let mut tokens = Vec::new();
        let zero_audio = vec![0u32; self.config.num_codebooks];

        for _ in 0..self.config.text_delay {
            let logits = model.forward(&zero_audio, self.last_text_token, &mut self.cache);
            let pred = logits.argmax(2);

            let token = match Tensor::<Wgpu, 3, Int>::into_data_async(pred).await {
                Ok(data) => match data.to_vec::<i32>() {
                    Ok(vec) => vec[0] as u32,
                    Err(_) => self.config.text_padding_id,
                },
                Err(_) => self.config.text_padding_id,
            };

            self.last_text_token = token;
            tokens.push(token);
        }

        tokens
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.last_text_token = self.config.text_padding_id;
        self.audio_buffer.clear();
        self.cache.reset();
    }
}
