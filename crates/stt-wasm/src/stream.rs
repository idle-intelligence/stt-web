//! Delayed-streams decoding loop.
//!
//! Audio and text run as parallel time-aligned streams.
//! Text is delayed by `delay` frames (7 frames = 560ms for stt-1b-en_fr).
//! Each step: model receives current audio frame + previous text token,
//! predicts next text token.
//!
//! Supports pipelined execution: `submit_frame()` queues GPU work (forward +
//! argmax) without awaiting the result, and `resolve_pending()` awaits the
//! readback. This allows CPU work (e.g. Mimi encode) to overlap with GPU
//! execution between calls.

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
    /// Pending argmax tensor from a `submit_frame()` call awaiting GPU readback.
    /// When present, `resolve_pending()` must be called before the next frame
    /// to complete the readback and update `last_text_token`.
    pending_argmax: Option<PendingFrame>,
}

/// A submitted but not-yet-resolved frame result.
struct PendingFrame {
    /// Argmax tensor on GPU, awaiting readback.
    argmax: Tensor<Wgpu, 3, Int>,
    /// Whether this frame is past the delay (i.e. should emit a token).
    emits: bool,
}

impl SttStream {
    /// Create a new streaming decoder with a pre-allocated KV cache from the model.
    pub fn new(config: SttConfig, cache: LayerCaches) -> Self {
        Self {
            last_text_token: config.text_start_token,
            frame_count: 0,
            config,
            cache,
            pending_argmax: None,
        }
    }

    /// Feed one frame of Mimi tokens (32 u32 values).
    ///
    /// Returns decoded text token if past the delay offset.
    /// This is the simple all-in-one path used by tests, warmup, and flush.
    pub async fn feed_frame(
        &mut self,
        audio_tokens: &[u32],
        model: &SttModel,
    ) -> Option<u32> {
        // Resolve any pending frame first (shouldn't happen in normal
        // feed_frame usage, but be safe).
        self.resolve_pending().await;

        self.submit_frame(audio_tokens, model);
        self.resolve_pending_as_token().await
    }

    /// Queue GPU work for one frame without awaiting the result.
    ///
    /// Runs `model.forward()` (queues GPU commands) and `argmax()` (also GPU),
    /// but does NOT call `into_data_async()`. The caller must later call
    /// `resolve_pending()` to complete the readback.
    ///
    /// Panics if there is already a pending frame (must resolve first).
    pub fn submit_frame(
        &mut self,
        audio_tokens: &[u32],
        model: &SttModel,
    ) {
        debug_assert!(
            self.pending_argmax.is_none(),
            "submit_frame called with unresolved pending frame"
        );

        self.frame_count += 1;
        let emits = self.frame_count > self.config.text_delay;

        let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
        let argmax = logits.argmax(2);

        self.pending_argmax = Some(PendingFrame { argmax, emits });
    }

    /// Await the GPU readback of a pending frame, update `last_text_token`,
    /// and return the emitted token (if past the delay).
    ///
    /// Returns `None` if no pending frame or if the frame is in the delay
    /// warmup period.
    pub async fn resolve_pending_as_token(&mut self) -> Option<u32> {
        let pending = self.pending_argmax.take()?;
        let token = self.readback_argmax(pending.argmax).await;
        self.last_text_token = token;
        if pending.emits { Some(token) } else { None }
    }

    /// Await any pending GPU readback and update `last_text_token`.
    ///
    /// No-op if no frame is pending. Used to drain the pipeline.
    pub async fn resolve_pending(&mut self) {
        if let Some(pending) = self.pending_argmax.take() {
            let token = self.readback_argmax(pending.argmax).await;
            self.last_text_token = token;
        }
    }

    /// Returns true if there is a pending frame awaiting readback.
    pub fn has_pending(&self) -> bool {
        self.pending_argmax.is_some()
    }

    /// Perform the async GPU readback on an argmax tensor.
    async fn readback_argmax(&self, pred: Tensor<Wgpu, 3, Int>) -> u32 {
        match Tensor::<Wgpu, 3, Int>::into_data_async(pred).await {
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
        }
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
        // Drain any pending frame from the pipeline first.
        let mut tokens = Vec::new();
        if let Some(token) = self.resolve_pending_as_token().await {
            tokens.push(token);
        }

        let zero_audio = vec![0u32; self.config.num_codebooks];

        for _ in 0..self.config.text_delay {
            if let Some(token) = self.feed_frame(&zero_audio, model).await {
                tokens.push(token);
            }
        }

        tokens
    }

    /// Reset all state for a new utterance.
    pub fn reset(&mut self) {
        self.frame_count = 0;
        self.last_text_token = self.config.text_start_token;
        self.pending_argmax = None;
        self.cache.reset();
    }

    /// Reset state but keep GPU KV cache buffers allocated.
    ///
    /// Used after warmup to avoid first-frame re-allocation overhead.
    pub fn reset_keep_buffers(&mut self) {
        self.frame_count = 0;
        self.last_text_token = self.config.text_start_token;
        self.pending_argmax = None;
        self.cache.reset_keep_buffers();
    }
}
