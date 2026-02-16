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

fn wasm_log(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    eprintln!("{msg}");
}

async fn read_scalar(t: Tensor<Wgpu, 1>) -> f32 {
    match Tensor::<Wgpu, 1>::into_data_async(t).await {
        Ok(d) => d.to_vec::<f32>().unwrap_or_default().first().copied().unwrap_or(f32::NAN),
        Err(_) => f32::NAN,
    }
}

/// How many frames past delay to run full diagnostics.
const DIAG_FRAMES: usize = 5;

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
            last_text_token: config.text_start_token,
            frame_count: 0,
            audio_buffer: Vec::new(),
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
        self.audio_buffer.push(audio_tokens.to_vec());
        self.frame_count += 1;

        // During delay: run model to fill KV cache but don't emit text.
        // Still extract and track the predicted token so we feed the model's
        // own predictions (not text_start_token repeatedly) on subsequent frames.
        if self.frame_count <= self.config.text_delay {
            let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
            let _ = self.extract_token(logits).await; // updates last_text_token
            wasm_log(&format!(
                "[stream] warmup frame={}/{} text={} cache_len={}",
                self.frame_count, self.config.text_delay,
                self.last_text_token, self.cache.seq_len()
            ));
            return None;
        }

        let frames_past_delay = self.frame_count - self.config.text_delay;

        // For the first few frames past delay, run with full diagnostics
        if frames_past_delay <= DIAG_FRAMES {
            let (logits, diag) =
                model.forward_diag(audio_tokens, self.last_text_token, &mut self.cache);

            // Readback all diagnostic norms
            let input_n = read_scalar(diag.input_norm).await;
            let after_norm_n = read_scalar(diag.layer0_after_norm).await;
            let q_n = read_scalar(diag.layer0_attn.q_norm).await;
            let k_n = read_scalar(diag.layer0_attn.k_norm).await;
            let v_n = read_scalar(diag.layer0_attn.v_norm).await;
            let scores_n = read_scalar(diag.layer0_attn.scores_norm).await;
            let attn_out_n = read_scalar(diag.layer0_attn.attn_out_norm).await;
            let post_attn_n = read_scalar(diag.layer0_post_attn_norm).await;
            let final_n = read_scalar(diag.final_pre_logits_norm).await;

            // Readback per-layer norms (single GPU readback for all 16 layers)
            let layer_norms_str = match Tensor::<Wgpu, 1>::into_data_async(diag.per_layer_norms).await {
                Ok(d) => match d.to_vec::<f32>() {
                    Ok(norms) => norms.iter().enumerate()
                        .map(|(i, n)| format!("L{}={:.0}", i, n))
                        .collect::<Vec<_>>().join(" "),
                    Err(_) => "ERR".to_string(),
                },
                Err(_) => "ERR".to_string(),
            };

            wasm_log(&format!(
                "[DIAG frame={}] input={:.2} | L0: norm={:.2} Q={:.2} K={:.2} V={:.2} scores={:.2} attn_out={:.2} | final={:.2} cache={}",
                self.frame_count, input_n, after_norm_n, q_n, k_n, v_n, scores_n, attn_out_n, final_n, diag.layer0_attn.cache_seq_len
            ));
            wasm_log(&format!(
                "[DIAG frame={}] per-layer norms: {}",
                self.frame_count, layer_norms_str
            ));

            // Also log top-5 logits
            let logits_data = Tensor::<Wgpu, 3>::into_data_async(logits.clone()).await;
            if let Ok(data) = logits_data {
                if let Ok(vals) = data.to_vec::<f32>() {
                    let mut indexed: Vec<(usize, f32)> =
                        vals.iter().copied().enumerate().collect();
                    indexed.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
                    wasm_log(&format!(
                        "[DIAG frame={}] top5={:?}",
                        self.frame_count, top5
                    ));
                }
            }

            return self.extract_token(logits).await;
        }

        // Normal path (no diagnostics)
        let logits = model.forward(audio_tokens, self.last_text_token, &mut self.cache);
        let result = self.extract_token(logits).await;
        if let Some(token) = result {
            wasm_log(&format!(
                "[stream] frame={} token={} prev_text={} cache={}",
                self.frame_count, token, self.last_text_token, self.cache.seq_len()
            ));
        }
        result
    }

    /// Extract token from logits via argmax + async readback.
    async fn extract_token(&mut self, logits: Tensor<Wgpu, 3>) -> Option<u32> {
        let pred = logits.argmax(2);
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
        self.audio_buffer.clear();
        self.cache.reset();
    }
}
