//! WASM bindings for STT using Q4 GGUF weights and wgpu (WebGPU) backend.
//!
//! Provides JavaScript-callable APIs for GPU-accelerated Q4 inference
//! in browsers with WebGPU support.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::OnceLock;

use burn::backend::wgpu::WgpuDevice;
use wasm_bindgen_futures::{future_to_promise, JsFuture};

use crate::gguf::Q4ModelLoader;
use crate::model::SttModel;
use crate::stream::{readback_argmax_free, SttStream};
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
    /// Token readbacks from spawn_local: (token_id, emits).
    token_sink: Rc<RefCell<Vec<(u32, bool)>>>,
    /// Last spawn_local's promise, awaited by flush.
    readback_promise: Option<js_sys::Promise>,
    /// Limit-cycle signal from spawn_local → next feedAudio.
    cycle_flag: Rc<Cell<bool>>,
    /// Delay-period prediction history for limit-cycle detection (shared with spawn_local).
    delay_prev_rc: Rc<RefCell<[u32; 2]>>,
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
            token_sink: Rc::new(RefCell::new(Vec::new())),
            readback_promise: None,
            cycle_flag: Rc::new(Cell::new(false)),
            delay_prev_rc: Rc::new(RefCell::new([u32::MAX; 2])),
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
    /// **Non-blocking readback via spawn_local:** dispatches GPU work synchronously,
    /// then spawns an async task to read back the argmax tensors. Text arrives
    /// one chunk late (~80ms). The pipeline never stalls on desktop — Firefox's
    /// 100ms mapAsync poll timer has zero impact since readback is fully decoupled.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = feedAudio))]
    pub fn feed_audio(&mut self, samples: &[f32]) -> Result<String, JsError> {
        let t_start = now_ms();

        // --- Step 0: Check limit-cycle flag from previous spawn_local ---
        if self.cycle_flag.get() {
            self.cycle_flag.set(false);
            if let Some(stream) = self.stream.as_mut() {
                stream.clear_gpu_autoregressive();
            }
        }

        // --- Step 1: Drain token_sink (from previous spawn_local) ---
        let drained: Vec<(u32, bool)> = std::mem::take(&mut *self.token_sink.borrow_mut());

        // Track first audio arrival for TTFB
        if !self.metrics.has_audio && !samples.is_empty() {
            self.metrics.has_audio = true;
            self.metrics.first_audio_ms = t_start;
        }

        let text_tokens: Vec<u32> = drained
            .iter()
            .filter(|(_, emits)| *emits)
            .map(|(token, _)| *token)
            .collect();

        // Track TTFB
        for &token in &text_tokens {
            if !self.metrics.has_token
                && token != self.config.text_padding_id
                && token != 0
            {
                self.metrics.has_token = true;
                self.metrics.first_token_ms = now_ms();
            }
        }

        // --- Step 2: Mimi encode (CPU) ---
        let mimi = self
            .mimi
            .as_mut()
            .ok_or_else(|| JsError::new("Mimi not loaded. Call loadMimi first."))?;

        let t_mimi_start = now_ms();
        let tokens = mimi.feed_audio(samples);
        let t_mimi_end = now_ms();

        let num_codebooks = self.config.num_codebooks;

        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded. Call loadModel first."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        let total_mimi_frames = tokens.len() / num_codebooks;

        // --- Step 3: STT forward pass (GPU dispatch, no readback) ---
        let t_stt_start = now_ms();

        let frame_starts: Vec<usize> = (0..tokens.len())
            .step_by(num_codebooks)
            .filter(|&s| s + num_codebooks <= tokens.len())
            .collect();

        for &frame_start in &frame_starts {
            let frame = &tokens[frame_start..frame_start + num_codebooks];
            stream.submit_frame_gpu(frame, model);
        }

        let t_stt_end = now_ms();

        // --- Step 4: Take pending frames for async readback ---
        let pending_frames = stream.take_batch_pending();

        let t_end = now_ms();

        // --- Step 5: Spawn async readback via future_to_promise ---
        if !pending_frames.is_empty() {
            let sink = Rc::clone(&self.token_sink);
            let flag = Rc::clone(&self.cycle_flag);
            let delay_prev = Rc::clone(&self.delay_prev_rc);
            let padding_id = self.config.text_padding_id;
            let prev_promise = self.readback_promise.take();

            let promise = future_to_promise(async move {
                // Await previous readback to maintain frame ordering
                if let Some(prev) = prev_promise {
                    let _ = JsFuture::from(prev).await;
                }

                for frame in pending_frames {
                    let token = readback_argmax_free(frame.argmax, padding_id).await;

                    // Q4 limit-cycle detection on non-emitting (delay) frames
                    if !frame.emits {
                        let mut dp = delay_prev.borrow_mut();
                        let prev = *dp;
                        *dp = [prev[1], token];
                        if token != padding_id
                            && token != 0
                            && token == prev[0]
                            && token != prev[1]
                        {
                            web_sys::console::warn_1(
                                &format!(
                                    "[stt] delay limit-cycle detected: {}↔{}, forcing padding",
                                    prev[1], token,
                                )
                                .into(),
                            );
                            flag.set(true);
                        }
                    }

                    sink.borrow_mut().push((token, frame.emits));
                }

                Ok(JsValue::UNDEFINED)
            });

            self.readback_promise = Some(promise);
        }

        // Update metrics
        let mimi_ms = t_mimi_end - t_mimi_start;
        let stt_ms = t_stt_end - t_stt_start;
        let total_call_ms = t_end - t_start;
        self.metrics.total_frames += total_mimi_frames;
        self.metrics.total_tokens += text_tokens.len();
        self.metrics.total_mimi_ms += mimi_ms;
        self.metrics.total_stt_ms += stt_ms;
        self.metrics.total_ms += total_call_ms;

        // Decode drained tokens to text
        let text = if let Some(ref tok) = self.tokenizer {
            tok.decode(&text_tokens)
        } else {
            text_tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
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
    ///
    /// Awaits the last spawn_local readback, drains the token sink, syncs the
    /// GPU-resident text token back to CPU, then drains the delay pipeline
    /// with zero-audio frames.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn flush(&mut self) -> Result<String, JsError> {
        // 1. Await last spawn_local (ensures all readbacks complete)
        if let Some(promise) = self.readback_promise.take() {
            let _ = JsFuture::from(promise).await;
        }

        // 2. Drain token_sink → collect emitting tokens
        let drained: Vec<(u32, bool)> = std::mem::take(&mut *self.token_sink.borrow_mut());
        let mut tokens: Vec<u32> = drained
            .iter()
            .filter(|(_, emits)| *emits)
            .map(|(token, _)| *token)
            .collect();

        // 3. Update last_text_token from last drained token (fallback if sync
        //    can't read GPU argmax, e.g. after clear_gpu_autoregressive)
        if let Some(&(last_token, _)) = drained.last() {
            self.stream
                .as_mut()
                .ok_or_else(|| JsError::new("Stream not initialized."))?
                .set_last_text_token(last_token);
        }

        // 4. Sync GPU-resident argmax → CPU (overwrites step 3 if available)
        self.stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?
            .sync_last_text_token()
            .await;

        // 5. Flush delay pipeline (feeds zero-audio frames via old per-frame path)
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;
        tokens.extend(stream.flush(model).await);

        // 6. Decode all accumulated tokens
        let text = if let Some(ref tok) = self.tokenizer {
            tok.decode(&tokens)
        } else {
            tokens
                .iter()
                .map(|t| t.to_string())
                .collect::<Vec<_>>()
                .join(",")
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
        // Fresh Rcs — stale spawn_local closures write to orphaned Rcs (harmless)
        self.token_sink = Rc::new(RefCell::new(Vec::new()));
        self.readback_promise = None;
        self.cycle_flag = Rc::new(Cell::new(false));
        self.delay_prev_rc = Rc::new(RefCell::new([u32::MAX; 2]));
    }

    /// Run warmup passes to pre-compile WebGPU shader pipelines.
    ///
    /// Feeds 10 dummy frames through the STT transformer with varied audio
    /// tokens, exercising all shader variants and the delay→emit transition
    /// (text_delay = 6). Uses `reset_keep_buffers()` afterwards to keep GPU
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

        // Run 10 forward passes (past text_delay=6) to exercise:
        // - All Q4 matmul / RmsNorm / softmax / RoPE shader variants
        // - KV cache lazy allocation and ring buffer writes
        // - The delay→emit code path transition (frame 6+)
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
