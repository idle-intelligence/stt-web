//! Delayed-streams decoding loop.
//!
//! Audio and text run as parallel time-aligned streams.
//! Text is delayed by `delay` frames (6 frames = 480ms for stt-1b-en_fr).
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
    /// GPU-resident argmax from the previous frame, used for autoregressive
    /// feedback WITHOUT reading back to CPU. Shape [1] Int on GPU.
    /// This eliminates the 236ms WebGPU buffer mapping latency per frame.
    last_text_argmax: Option<Tensor<Wgpu, 1, Int>>,
    /// Pending argmax tensor from a `submit_frame()` call awaiting GPU readback.
    /// When present, `resolve_pending()` must be called before the next frame
    /// to complete the readback and update `last_text_token`.
    pending_argmax: Option<PendingFrame>,
    /// Accumulated argmax tensors awaiting batch readback for text output.
    /// Frames are appended during `submit_frame_gpu()` and read back all
    /// at once via `resolve_batch()`.
    batch_pending: Vec<PendingFrame>,
    /// Last two predictions during the delay period, for limit-cycle detection.
    /// Q4 quantization can cause degenerate oscillations (e.g. 260↔263) that
    /// the F32 reference model doesn't exhibit.
    delay_prev: [u32; 2],
}

/// A submitted but not-yet-resolved frame result.
pub struct PendingFrame {
    /// Argmax tensor on GPU, awaiting readback.
    pub argmax: Tensor<Wgpu, 3, Int>,
    /// Whether this frame is past the delay (i.e. should emit a token).
    pub emits: bool,
}

impl SttStream {
    /// Create a new streaming decoder with a pre-allocated KV cache from the model.
    pub fn new(config: SttConfig, cache: LayerCaches) -> Self {
        Self {
            last_text_token: config.text_start_token,
            last_text_argmax: None,
            frame_count: 0,
            config,
            cache,
            pending_argmax: None,
            batch_pending: Vec::new(),
            delay_prev: [u32::MAX; 2],
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
        let emits = self.frame_count >= self.config.text_delay;

        let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
        let argmax = logits.argmax(2);

        self.pending_argmax = Some(PendingFrame { argmax, emits });
    }

    /// Queue GPU work using GPU-resident text token (no readback needed).
    ///
    /// Like `submit_frame()` but uses `forward_with_gpu_token()` — the
    /// autoregressive text token stays on GPU as an argmax tensor, avoiding
    /// the 236ms WebGPU buffer mapping latency between frames.
    ///
    /// The argmax result is appended to `batch_pending` for later readback
    /// via `resolve_batch()` (text output only).
    pub fn submit_frame_gpu(
        &mut self,
        audio_tokens: &[u32],
        model: &SttModel,
    ) {
        self.frame_count += 1;
        let emits = self.frame_count >= self.config.text_delay;

        let logits = model.forward_with_gpu_token(
            audio_tokens,
            self.last_text_argmax.take(),
            self.last_text_token,
            &mut self.cache,
        );
        let argmax = logits.argmax(2);

        // Keep argmax on GPU for the next frame's autoregressive input.
        self.last_text_argmax = Some(argmax.clone().reshape([1]));

        // Queue for batch readback (text output).
        self.batch_pending.push(PendingFrame { argmax, emits });
    }

    /// Batch-read all accumulated argmax tensors and return emitted tokens.
    ///
    /// Called once at the end of `feed_audio()` instead of per-frame.
    /// A single readback of the last pending argmax pays the 236ms mapAsync
    /// cost once, not per frame.
    pub async fn resolve_batch(&mut self) -> Vec<u32> {
        let pending = std::mem::take(&mut self.batch_pending);
        let mut tokens = Vec::new();

        for frame in pending {
            let token = self.readback_argmax(frame.argmax).await;
            self.last_text_token = token;
            if frame.emits {
                tokens.push(token);
            } else {
                // Q4 limit-cycle detection (mirrors resolve_pending_as_token)
                let prev = self.delay_prev;
                self.delay_prev = [prev[1], token];
                if token != self.config.text_padding_id
                    && token != 0
                    && token == prev[0]
                    && token != prev[1]
                {
                    Self::log(&format!(
                        "[stt] delay limit-cycle detected: {}↔{}, forcing padding",
                        prev[1], token,
                    ));
                    self.last_text_token = self.config.text_padding_id;
                    self.last_text_argmax = None;
                }
            }
        }

        tokens
    }

    /// Take all pending frames for external (non-blocking) readback.
    ///
    /// Used by the WASM bindings to decouple GPU readback from the audio
    /// processing loop. The caller receives owned `PendingFrame`s and can
    /// resolve them via `spawn_local` without blocking `feedAudio`.
    pub fn take_batch_pending(&mut self) -> Vec<PendingFrame> {
        std::mem::take(&mut self.batch_pending)
    }

    /// Set `last_text_token` from an externally resolved readback.
    ///
    /// Used by the WASM bindings after concatenated batch readback in flush,
    /// so that `feed_frame()` (used by the delay-pipeline drain) gets the
    /// correct autoregressive input without a separate GPU readback.
    pub fn set_last_text_token(&mut self, token: u32) {
        self.last_text_token = token;
    }

    /// Clear GPU-resident autoregressive state (for limit-cycle breaking from spawn_local).
    pub fn clear_gpu_autoregressive(&mut self) {
        self.last_text_token = self.config.text_padding_id;
        self.last_text_argmax = None;
    }

    /// Read back the GPU-resident argmax to synchronize `last_text_token`.
    ///
    /// When using the GPU-resident pipeline (`submit_frame_gpu`), the
    /// autoregressive feedback stays on GPU via `last_text_argmax`.
    /// But `flush()` → `feed_frame()` needs the CPU-side `last_text_token`.
    /// Call this before flush to ensure correctness.
    pub async fn sync_last_text_token(&mut self) {
        if let Some(ref argmax) = self.last_text_argmax {
            match argmax.clone().into_data_async().await {
                Ok(data) => {
                    if let Ok(vec) = data.to_vec::<i32>() {
                        if !vec.is_empty() {
                            self.last_text_token = vec[0] as u32;
                        }
                    }
                }
                Err(e) => {
                    Self::log(&format!(
                        "[stt] sync_last_text_token readback failed: {:?}",
                        e
                    ));
                }
            }
        }
    }

    /// Await the GPU readback of a pending frame, update `last_text_token`,
    /// and return the emitted token (if past the delay).
    ///
    /// Returns `None` if no pending frame or if the frame is in the delay
    /// warmup period.
    pub async fn resolve_pending_as_token(&mut self) -> Option<u32> {
        let pending = self.pending_argmax.take()?;
        let token = self.readback_argmax(pending.argmax).await;

        if pending.emits {
            self.last_text_token = token;
            Some(token)
        } else {
            // Delay period: feed back the model's own prediction, matching the
            // reference PyTorch/Candle implementations. The model expects to see
            // its previous prediction as text input (autoregressive), even during
            // the delay when tokens aren't emitted to the user.
            //
            // Q4 quantization can cause degenerate limit cycles (e.g. 260↔263)
            // that the F32 reference model doesn't exhibit. Detect oscillation
            // and break it by forcing padding for one frame.
            let prev = self.delay_prev;
            self.delay_prev = [prev[1], token];

            if token != self.config.text_padding_id
                && token != 0
                && token == prev[0]
                && token != prev[1]
            {
                // A↔B oscillation detected (e.g. 260→263→260): break the cycle.
                Self::log(&format!(
                    "[stt] delay limit-cycle detected: {}↔{}, forcing padding",
                    prev[1], token,
                ));
                self.last_text_token = self.config.text_padding_id;
            } else {
                self.last_text_token = token;
            }
            None
        }
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
                Ok(vec) => vec.first().copied().unwrap_or(self.config.text_padding_id as i32) as u32,
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

    /// Flush remaining text after end of speech.
    pub async fn flush(&mut self, model: &SttModel) -> Vec<u32> {
        // Callers should pre-drain via resolve_batch() before calling flush.
        debug_assert!(
            self.batch_pending.is_empty(),
            "flush() called with unresolved batch_pending; caller should resolve_batch() first"
        );
        let mut tokens = self.resolve_batch().await;
        if let Some(token) = self.resolve_pending_as_token().await {
            tokens.push(token);
        }

        // Feed zero-audio frames to drain the delay pipeline.
        // Use feed_frame (old path) since flush is not perf-critical
        // and needs per-frame readback for the text tokens.
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
        self.last_text_argmax = None;
        self.pending_argmax = None;
        self.batch_pending.clear();
        self.delay_prev = [u32::MAX; 2];
        self.cache.reset();
    }

    /// Reset state but keep GPU KV cache buffers allocated.
    ///
    /// Used after warmup to avoid first-frame re-allocation overhead.
    pub fn reset_keep_buffers(&mut self) {
        self.frame_count = 0;
        self.last_text_token = self.config.text_start_token;
        self.last_text_argmax = None;
        self.pending_argmax = None;
        self.batch_pending.clear();
        self.delay_prev = [u32::MAX; 2];
        self.cache.reset_keep_buffers();
    }
}

/// Perform async GPU readback on an argmax tensor (free function).
///
/// Same logic as `SttStream::readback_argmax` but doesn't require `&self`,
/// so it can be called from a `spawn_local` closure that only holds owned
/// tensors and shared state (no borrow on `SttStream`).
pub async fn readback_argmax_free(pred: Tensor<Wgpu, 3, Int>, padding_id: u32) -> u32 {
    match Tensor::<Wgpu, 3, Int>::into_data_async(pred).await {
        Ok(data) => match data.to_vec::<i32>() {
            Ok(vec) => vec.first().copied().unwrap_or(padding_id as i32) as u32,
            Err(e) => {
                log_warn(&format!("[stt] GPU readback to_vec FAILED: {:?}", e));
                padding_id
            }
        },
        Err(e) => {
            log_warn(&format!(
                "[stt] GPU readback into_data_async FAILED: {:?}",
                e
            ));
            padding_id
        }
    }
}

/// Log a warning to console (WASM) or stderr (native).
fn log_warn(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::warn_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    eprintln!("{}", msg);
}
