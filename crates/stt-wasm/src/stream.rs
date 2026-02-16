//! Delayed-streams decoding loop.
//!
//! Audio and text run as parallel time-aligned streams.
//! Text is delayed by `delay` frames (6 frames = 480ms for stt-1b-en_fr).
//! Each step: model receives current audio frame + previous text token,
//! predicts next text token.

use crate::model::SttModel;
use crate::SttConfig;

/// Streaming STT decoder implementing delayed-streams logic.
pub struct SttStream {
    model: SttModel,
    config: SttConfig,
    frame_count: usize,
    text_position: usize,
    last_text_token: u32,
    audio_buffer: Vec<Vec<u32>>,
}

impl SttStream {
    /// Create a new streaming decoder.
    pub fn new(model: SttModel, config: SttConfig) -> Self {
        Self {
            model,
            config,
            frame_count: 0,
            text_position: 0,
            last_text_token: 0, // padding token
            audio_buffer: Vec::new(),
        }
    }

    /// Feed one frame of Mimi tokens (32 u32 values).
    ///
    /// Returns decoded text if any new tokens were produced.
    /// Text output is delayed by `config.text_delay` frames behind audio input.
    pub async fn feed_frame(&mut self, _audio_tokens: &[u32]) -> Option<String> {
        todo!("Buffer audio frame, run model if past delay offset, decode text token")
    }

    /// Flush remaining text after VAD detects end of speech.
    ///
    /// The "flush trick": run remaining frames faster than real-time
    /// to eliminate tail latency from the delay offset.
    pub async fn flush(&mut self) -> String {
        todo!("Process remaining buffered frames, collect all pending text tokens")
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.text_position = 0;
        self.last_text_token = 0;
        self.audio_buffer.clear();
        self.model.reset_cache();
    }
}
