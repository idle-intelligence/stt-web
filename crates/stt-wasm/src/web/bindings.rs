//! WASM bindings for Kyutai STT using Q4 GGUF weights and wgpu (WebGPU) backend.
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
use crate::SttConfig;

/// Device initialized by `initWgpuDevice()` — used by `SttEngine` instances.
static WGPU_DEVICE: OnceLock<WgpuDevice> = OnceLock::new();

fn wasm_log(msg: &str) {
    #[cfg(target_family = "wasm")]
    web_sys::console::log_1(&msg.into());
    #[cfg(not(target_family = "wasm"))]
    let _ = msg;
}

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
    mimi: Option<mimi_wasm::MimiCodec>,
    config: SttConfig,
    device: WgpuDevice,
    shard_bufs: Vec<Vec<u8>>,
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
            config: SttConfig::default(),
            device,
            shard_bufs: Vec::new(),
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

        let stream = SttStream::new(self.config.clone(), self.config.num_layers);

        self.model = Some(model);
        self.stream = Some(stream);

        wasm_log("[stt] Model loaded successfully");
        Ok(())
    }

    /// Initialize the Mimi audio codec from a weights URL.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadMimi))]
    pub async fn load_mimi(&mut self, weights_url: &str) -> Result<(), JsError> {
        wasm_log(&format!("[stt] Loading Mimi codec from {weights_url}..."));
        let mimi = mimi_wasm::MimiCodec::new(weights_url)
            .await
            .map_err(|e| JsError::new(&format!("Failed to load Mimi: {e:?}")))?;
        self.mimi = Some(mimi);
        wasm_log("[stt] Mimi codec loaded");
        Ok(())
    }

    /// Feed PCM audio samples (f32, 24kHz mono for Mimi).
    ///
    /// Returns transcript text tokens if any new tokens were produced.
    /// Audio goes through: Mimi codec → STT transformer → text tokens.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = feedAudio))]
    pub async fn feed_audio(&mut self, samples: &[f32]) -> Result<Vec<u32>, JsError> {
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

        // Encode audio to Mimi tokens (batch mode — processes all at once)
        let tokens = mimi.encode_all(samples);

        if !tokens.is_empty() {
            let preview: Vec<u32> = tokens.iter().take(32).copied().collect();
            wasm_log(&format!(
                "[stt] Mimi: {} samples → {} tokens, first frame: {:?}",
                samples.len(),
                tokens.len(),
                preview
            ));
        }

        // Process complete frames (32 tokens per frame)
        let num_codebooks = self.config.num_codebooks;
        let mut text_tokens = Vec::new();

        for frame_start in (0..tokens.len()).step_by(num_codebooks) {
            if frame_start + num_codebooks > tokens.len() {
                break;
            }
            let frame = &tokens[frame_start..frame_start + num_codebooks];

            if let Some(token) = stream.feed_frame(frame, model).await {
                text_tokens.push(token);
            }
        }

        Ok(text_tokens)
    }

    /// Flush remaining text after end of speech.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn flush(&mut self) -> Result<Vec<u32>, JsError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| JsError::new("Model not loaded."))?;
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| JsError::new("Stream not initialized."))?;

        let tokens = stream.flush(model).await;
        Ok(tokens)
    }

    /// Reset all state for a new recording session.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub fn reset(&mut self) {
        if let Some(stream) = &mut self.stream {
            stream.reset();
        }
        if let Some(mimi) = &mut self.mimi {
            mimi.reset();
        }
    }

    /// Check if the model is loaded and ready.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = isReady))]
    pub fn is_ready(&self) -> bool {
        self.model.is_some() && self.mimi.is_some()
    }

    /// Run a diagnostic forward pass with known inputs to verify correctness.
    ///
    /// Uses the same tokens as the native test. On native Metal, this produces:
    ///   argmax=3, top5=[(3, 5.93), (0, 3.96), (270, -3.12), ...]
    /// If WebGPU produces different results, there's a GPU computation bug.
    #[cfg_attr(target_family = "wasm", wasm_bindgen)]
    pub async fn diagnose(&self) -> Result<String, JsError> {
        use burn::backend::wgpu::Wgpu;
        use burn::tensor::Tensor;
        use crate::model::LayerCaches;

        let model = self.model.as_ref()
            .ok_or_else(|| JsError::new("Model not loaded"))?;

        let test_frame: Vec<u32> = vec![
            326, 955, 1016, 546, 1200, 400, 800, 1500,
            100, 200, 300, 400, 500, 600, 700, 800,
            900, 1000, 1100, 1200, 1300, 1400, 1500, 1600,
            1700, 1800, 1900, 2000, 50, 150, 250, 350,
        ];
        let text_token = self.config.text_start_token;

        // --- Test 1: Single forward with fresh cache ---
        let mut cache = LayerCaches::new(self.config.num_layers);
        let logits = model.forward(&test_frame, text_token, &mut cache);
        let logits_data = Tensor::<Wgpu, 3>::into_data_async(logits).await
            .map_err(|e| JsError::new(&format!("readback failed: {e}")))?;
        let vals: Vec<f32> = logits_data.to_vec::<f32>()
            .map_err(|e| JsError::new(&format!("to_vec failed: {e}")))?;

        let mut indexed: Vec<(usize, f32)> = vals.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top5: Vec<(usize, f32)> = indexed.into_iter().take(5).collect();
        let argmax = top5[0].0;
        let min_val = vals.iter().cloned().fold(f32::INFINITY, f32::min);
        let max_val = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut result = format!(
            "=== DIAGNOSE: single forward pass ===\n\
             Expected (native Metal): argmax=3, max=5.93\n\
             Got: argmax={argmax}, max={max_val:.4}, min={min_val:.4}\n\
             top5={top5:?}\n"
        );

        // --- Test 2: Two forwards with different inputs, fresh caches ---
        let frame2: Vec<u32> = vec![
            1, 2, 3, 4, 5, 6, 7, 8,
            9, 10, 11, 12, 13, 14, 15, 16,
            17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32,
        ];

        let mut cache1 = LayerCaches::new(self.config.num_layers);
        let mut cache2 = LayerCaches::new(self.config.num_layers);

        let logits1 = model.forward(&test_frame, text_token, &mut cache1);
        let logits2 = model.forward(&frame2, text_token, &mut cache2);

        let d1 = Tensor::<Wgpu, 3>::into_data_async(logits1).await
            .map_err(|e| JsError::new(&format!("readback1 failed: {e}")))?;
        let d2 = Tensor::<Wgpu, 3>::into_data_async(logits2).await
            .map_err(|e| JsError::new(&format!("readback2 failed: {e}")))?;

        let v1: Vec<f32> = d1.to_vec().unwrap_or_default();
        let v2: Vec<f32> = d2.to_vec().unwrap_or_default();

        let sum_diff: f32 = v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        let max_diff: f32 = v1.iter().zip(v2.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);

        result += &format!(
            "\n=== Two-input diff (should be >1000 if varying) ===\n\
             sum_diff={sum_diff:.4}, max_diff={max_diff:.6}\n\
             FROZEN={}\n",
            if sum_diff < 1.0 { "YES (BUG!)" } else { "no (OK)" }
        );

        // --- Test 3: Per-layer norm check ---
        // Run through layers one by one to find where the signal dies
        let dim = self.config.hidden_size;
        let mut sum_a = vec![0.0f32; dim];
        let mut sum_b = vec![0.0f32; dim];
        for (i, &token) in test_frame.iter().enumerate() {
            model.audio_emb_ref()[i].embed_id_add_cpu(token, &mut sum_a);
        }
        model.text_emb_ref().embed_id_add_cpu(text_token, &mut sum_a);
        for (i, &token) in frame2.iter().enumerate() {
            model.audio_emb_ref()[i].embed_id_add_cpu(token, &mut sum_b);
        }
        model.text_emb_ref().embed_id_add_cpu(text_token, &mut sum_b);

        let emb_diff: f32 = sum_a.iter().zip(sum_b.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        result += &format!("\nEmbedding diff (should be large): {emb_diff:.4}\n");

        wasm_log(&result);
        Ok(result)
    }
}

impl Default for SttEngine {
    fn default() -> Self {
        Self::new()
    }
}
