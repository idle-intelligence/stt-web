//! WASM bindings for STT using Q4 GGUF weights and wgpu (WebGPU) backend.
//!
//! Provides JavaScript-callable APIs for GPU-accelerated Q4 inference
//! in browsers with WebGPU support.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

use burn::backend::wgpu::WgpuDevice;

use crate::gguf::Q4ModelLoader;
use crate::model::SttModel;
use crate::stream::SttStream;
use crate::tokenizer::SpmDecoder;
use crate::SttConfig;

/// Device initialized by `initWgpuDevice()` — used by `SttEngine` instances.
static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn wasm_log(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    let _ = msg;
}

/// Cross-platform millisecond timer.
fn now_ms() -> f64 {
    #[cfg(target_family = "wasm")]
    {
        js_sys::Date::now()
    }
    #[cfg(not(target_family = "wasm"))]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs_f64()
            * 1000.0
    }
}

/// Cumulative metrics tracked across the session.
struct SessionMetrics {
    first_audio_ms: f64,
    first_token_ms: f64,
    has_audio: bool,
    has_token: bool,
    total_frames: usize,
    total_tokens: usize,
    // Cumulative timing
    total_mimi_ms: f64,
    total_stt_ms: f64,
    total_ms: f64,
}

impl SessionMetrics {
    fn new() -> Self {
        Self {
            first_audio_ms: 0.0,
            first_token_ms: 0.0,
            has_audio: false,
            has_token: false,
            total_frames: 0,
            total_tokens: 0,
            total_mimi_ms: 0.0,
            total_stt_ms: 0.0,
            total_ms: 0.0,
        }
    }

    fn reset(&mut self) {
        *self = Self::new();
    }

    fn ttfb_ms(&self) -> f64 {
        if self.has_token && self.has_audio {
            self.first_token_ms - self.first_audio_ms
        } else {
            -1.0
        }
    }
}

use crate::mimi_encoder::MimiEncoder;

// ---- WASM entry points ----

/// Initialize panic hook for better error messages in browser console.
#[cfg_attr(target_family = "wasm", wasm_bindgen(start))]
pub fn start() {
    console_error_panic_hook::set_once();
}

/// Initialize the WebGPU device asynchronously.
///
/// **Must** be called (and awaited) before creating `SttEngine`.
/// Requests the adapter's full limits (especially `max_compute_invocations_per_workgroup`).
#[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = initWgpuDevice))]
pub async fn init_wgpu_device() {
    use burn::backend::wgpu::{init_device, RuntimeOptions, WgpuSetup};

    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .expect("No WebGPU adapter found");

    let info = adapter.get_info();
    let adapter_limits = adapter.limits();
    wasm_log(&format!(
        "[wgpu] Adapter: {} ({:?}), backend: {:?}",
        info.name, info.device_type, info.backend
    ));
    wasm_log(&format!(
        "[wgpu] Adapter limits: max_compute_invocations_per_workgroup={}, max_buffer_size={}",
        adapter_limits.max_compute_invocations_per_workgroup,
        adapter_limits.max_buffer_size,
    ));

    // Request device with the adapter's full limits — not spec defaults
    let features = adapter.features() - wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("stt-wgpu"),
            required_features: features,
            required_limits: adapter_limits,
            memory_hints: wgpu::MemoryHints::MemoryUsage,
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("Failed to create WebGPU device");

    wasm_log(&format!(
        "[wgpu] Device created: max_compute_invocations_per_workgroup={}",
        device.limits().max_compute_invocations_per_workgroup,
    ));

    let setup = WgpuSetup {
        instance,
        adapter,
        device,
        queue,
        backend: info.backend,
    };

    let wgpu_device = init_device(setup, RuntimeOptions::default());
    WGPU_DEVICE.set(wgpu_device).ok();
}

/// Browser-facing STT engine combining Mimi codec + STT transformer.
///
/// This is the single entry point that the Web Worker calls.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct SttEngine {
    model: Option<SttModel>,
    stream: Option<SttStream>,
    mimi: Option<MimiEncoder>,
    tokenizer: Option<SpmDecoder>,
    config: SttConfig,
    device: WgpuDevice,
    shard_bufs: Vec<Vec<u8>>,
    metrics: SessionMetrics,
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl SttEngine {
    /// Create a new SttEngine instance.
    ///
    /// Call `initWgpuDevice()` first, then create this, then load weights.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(constructor))]
    pub fn new() -> Self {
        console_error_panic_hook::set_once();
        let device = WGPU_DEVICE
            .get()
            .cloned()
            .unwrap_or_else(WgpuDevice::default);
        Self {
            model: None,
            stream: None,
            mimi: None,
            tokenizer: None,
            config: SttConfig::default(),
            device,
            shard_bufs: Vec::new(),
            metrics: SessionMetrics::new(),
        }
    }

    /// Append a model weight shard (for multi-shard GGUF loading).
    ///
    /// Call this for each shard before calling `loadModel`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = appendModelShard))]
    pub fn append_model_shard(&mut self, shard: &[u8]) {
        self.shard_bufs.push(shard.to_vec());
        wasm_log(&format!(
            "[stt] Shard appended ({} bytes, {} total shards)",
            shard.len(),
            self.shard_bufs.len()
        ));
    }

    /// Load the STT model from previously appended shards.
    ///
    /// Uses two-phase loading: parse GGUF → drop reader → finalize tensors on GPU.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModel))]
    pub fn load_model(&mut self) -> Result<(), JsError> {
        if self.shard_bufs.is_empty() {
            return Err(JsError::new("No shards appended. Call appendModelShard first."));
        }

        wasm_log("[stt] Phase 1: Parsing GGUF and loading Q4 tensors...");

        let shards = std::mem::take(&mut self.shard_bufs);
        let parts = {
            let mut loader = Q4ModelLoader::from_shards(shards)
                .map_err(|e| JsError::new(&format!("Failed to parse GGUF: {e}")))?;
            loader
                .load_deferred(&self.device, &self.config)
                .map_err(|e| JsError::new(&format!("Failed to load Q4 model: {e}")))?
            // loader (and its shard data) dropped here
        };

        wasm_log("[stt] Phase 2: Finalizing model on GPU...");

        let model = parts
            .finalize(&self.device)
            .map_err(|e| JsError::new(&format!("Failed to finalize model: {e}")))?;

        let stream = SttStream::new(self.config.clone(), model.create_cache());

        self.model = Some(model);
        self.stream = Some(stream);

        wasm_log("[stt] Model loaded successfully");
        Ok(())
    }

    /// Initialize the Mimi audio codec from pre-fetched weight bytes.
    ///
    /// Loads the full Mimi model via mimi-rs (candle), with automatic key
    /// remapping from the standard Mimi safetensors naming convention.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadMimi))]
    pub fn load_mimi(&mut self, data: &[u8]) -> Result<(), JsError> {
        wasm_log(&format!("[stt] Loading Mimi codec ({} bytes)...", data.len()));
        let mimi = MimiEncoder::from_bytes(data)
            .map_err(|e| JsError::new(&format!("Failed to load Mimi: {e}")))?;
        self.mimi = Some(mimi);
        wasm_log("[stt] Mimi codec loaded (mimi-rs, Q8_0 transformer)");
        Ok(())
    }

    /// Load the SentencePiece tokenizer from pre-fetched `.model` bytes.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadTokenizer))]
    pub fn load_tokenizer(&mut self, data: &[u8]) -> Result<(), JsError> {
        wasm_log(&format!("[stt] Loading tokenizer ({} bytes)...", data.len()));
        let decoder = SpmDecoder::from_bytes(data);
        wasm_log(&format!("[stt] Tokenizer loaded: {} vocab entries", decoder.vocab_len()));
        self.tokenizer = Some(decoder);
        Ok(())
    }

    /// Feed PCM audio samples (f32, 24kHz mono for Mimi).
    ///
    /// Returns decoded transcript text if any new tokens were produced.
    /// Audio goes through: Mimi codec → STT transformer → text tokens → detokenize.
    /// Per-call timing is stored in metrics (retrieve via `getMetrics()`).
    ///
    /// **Pipelining:** The last frame's GPU work (forward + argmax) is left
    /// pending at the end of each call. On the *next* call, Mimi encode (CPU)
    /// runs first while the GPU finishes, then we resolve the pending readback.
    /// This overlaps ~20ms of CPU work with the GPU readback latency.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = feedAudio))]
    pub async fn feed_audio(&mut self, samples: &[f32]) -> Result<String, JsError> {
        let t_start = now_ms();

        // Track first audio arrival for TTFB
        if !self.metrics.has_audio && !samples.is_empty() {
            self.metrics.has_audio = true;
            self.metrics.first_audio_ms = t_start;
        }

        let mimi = self
            .mimi
            .as_mut()
            .ok_or_else(|| JsError::new("Mimi not loaded. Call loadMimi first."))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        // --- Mimi encode (CPU) ---
        // This runs while any pending GPU work from the previous call executes
        // on the GPU hardware in parallel.
        let t_mimi_start = now_ms();
        let tokens = mimi.feed_audio(samples);
        let t_mimi_end = now_ms();

        // --- Resolve pending GPU readback from previous call ---
        // By now the GPU has had ~20ms of Mimi CPU time to finish, so this
        // await should resolve near-instantly.
        let mut text_tokens = Vec::new();
        if let Some(token) = stream.resolve_pending_as_token().await {
            if !self.metrics.has_token
                && token != self.config.text_padding_id
                && token != 0
            {
                self.metrics.has_token = true;
                self.metrics.first_token_ms = now_ms();
            }
            text_tokens.push(token);
        }

        // --- STT forward (pipelined) ---
        let num_codebooks = self.config.num_codebooks;
        let mimi_frames = tokens.len() / num_codebooks;

        let t_stt_start = now_ms();

        // Build list of frame slices
        let frame_starts: Vec<usize> = (0..tokens.len())
            .step_by(num_codebooks)
            .filter(|&s| s + num_codebooks <= tokens.len())
            .collect();
        let n_frames = frame_starts.len();

        for (i, &frame_start) in frame_starts.iter().enumerate() {
            let frame = &tokens[frame_start..frame_start + num_codebooks];
            let is_last = i == n_frames - 1;

            // Submit GPU work (forward + argmax, no await)
            stream.submit_frame(frame, model);

            if is_last {
                // Leave the last frame pending — it will be resolved on the
                // next feed_audio call, overlapping with Mimi encode.
                // Don't await here.
            } else {
                // For non-last frames, resolve immediately (we need the token
                // for the next frame's input).
                if let Some(token) = stream.resolve_pending_as_token().await {
                    if !self.metrics.has_token
                        && token != self.config.text_padding_id
                        && token != 0
                    {
                        self.metrics.has_token = true;
                        self.metrics.first_token_ms = now_ms();
                    }
                    text_tokens.push(token);
                }
            }
        }
        let t_stt_end = now_ms();
        let t_end = now_ms();

        // Update metrics
        let mimi_ms = t_mimi_end - t_mimi_start;
        let stt_ms = t_stt_end - t_stt_start;
        let total_call_ms = t_end - t_start;
        self.metrics.total_frames += mimi_frames;
        self.metrics.total_tokens += text_tokens.len();
        self.metrics.total_mimi_ms += mimi_ms;
        self.metrics.total_stt_ms += stt_ms;
        self.metrics.total_ms += total_call_ms;

        // Decode token IDs to text
        let text = if let Some(ref tok) = self.tokenizer {
            tok.decode(&text_tokens)
        } else {
            // Fallback: return token IDs as comma-separated string
            text_tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
        };

        Ok(text)
    }

    /// Get timing metrics from the session.
    ///
    /// Returns a JS object: `{ mimi_encode_ms, stt_forward_ms, total_ms,
    ///   total_frames, total_tokens, ttfb_ms }`
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = getMetrics))]
    pub fn get_metrics(&self) -> JsValue {
        let obj = js_sys::Object::new();
        let set = |k: &str, v: f64| {
            js_sys::Reflect::set(&obj, &JsValue::from_str(k), &JsValue::from_f64(v)).ok();
        };
        set("mimi_encode_ms", self.metrics.total_mimi_ms);
        set("stt_forward_ms", self.metrics.total_stt_ms);
        set("total_ms", self.metrics.total_ms);
        set("total_frames", self.metrics.total_frames as f64);
        set("total_tokens", self.metrics.total_tokens as f64);
        set("ttfb_ms", self.metrics.ttfb_ms());
        obj.into()
    }

    /// Flush remaining text after end of speech.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn flush(&mut self) -> Result<String, JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        let tokens = stream.flush(model).await;

        let text = if let Some(ref tok) = self.tokenizer {
            tok.decode(&tokens)
        } else {
            tokens.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(",")
        };

        Ok(text)
    }

    /// Reset all state for a new recording session.
    ///
    /// Uses `reset_keep_buffers` for the STT stream to preserve GPU KV cache
    /// allocations from warmup. A full `reset()` drops the GPU tensors, forcing
    /// expensive re-allocation on the first frame of the next recording.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub fn reset(&mut self) {
        if let Some(stream) = &mut self.stream {
            stream.reset_keep_buffers();
        }
        if let Some(mimi) = &mut self.mimi {
            mimi.reset();
        }
        self.metrics.reset();
    }

    /// Run warmup passes to pre-compile WebGPU shader pipelines.
    ///
    /// Feeds 10 dummy frames through the STT transformer with varied audio
    /// tokens, exercising all shader variants and the delay→emit transition
    /// (text_delay = 7). Uses `reset_keep_buffers()` afterwards to keep GPU
    /// KV cache buffers allocated, avoiding re-allocation on first real frame.
    ///
    /// Call after `loadModel()` + `loadMimi()`. Reduces TTFB by ~150-400ms.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn warmup(&mut self) -> Result<(), JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        let t0 = now_ms();

        // Run 10 forward passes (past text_delay=7) to exercise:
        // - All Q4 matmul / RmsNorm / softmax / RoPE shader variants
        // - KV cache lazy allocation and ring buffer writes
        // - The delay→emit code path transition (frame 8+)
        // Use varied token values to avoid any zero-optimized GPU paths.
        let num_cb = self.config.num_codebooks;
        for i in 0..10u32 {
            let audio_tokens: Vec<u32> = (0..num_cb)
                .map(|cb| (i * 7 + cb as u32 * 13 + 1) % 2048)
                .collect();
            let _ = stream.feed_frame(&audio_tokens, model).await;
        }

        // Also warm up Mimi with a silent chunk if loaded
        if let Some(mimi) = &mut self.mimi {
            let silence = vec![0.0f32; 1920]; // one Mimi frame
            let _ = mimi.feed_audio(&silence);
        }

        // Reset state but keep GPU KV cache buffers allocated.
        // A full reset() would drop the GPU tensors, forcing lazy
        // re-allocation on the first real frame.
        stream.reset_keep_buffers();
        if let Some(mimi) = &mut self.mimi {
            mimi.reset();
        }
        self.metrics.reset();

        let elapsed = now_ms() - t0;
        wasm_log(&format!("[stt] Warmup complete ({elapsed:.0}ms, 10 frames)"));
        Ok(())
    }

    /// Check if the model is loaded and ready.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = isReady))]
    pub fn is_ready(&self) -> bool {
        self.model.is_some() && self.mimi.is_some() && self.tokenizer.is_some()
    }
}

impl Default for SttEngine {
    fn default() -> Self {
        Self::new()
    }
}
