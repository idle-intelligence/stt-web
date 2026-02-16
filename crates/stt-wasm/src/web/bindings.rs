//! WASM bindings for Kyutai STT using Q4 GGUF weights and wgpu (WebGPU) backend.
//!
//! Provides JavaScript-callable APIs for GPU-accelerated Q4 inference
//! in browsers with WebGPU support.

#[cfg(target_family = "wasm")]
use wasm_bindgen::prelude::*;

use std::sync::OnceLock;

use burn::backend::wgpu::{Wgpu, WgpuDevice};

use crate::Backend;

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
    todo!("Create wgpu instance, request adapter, create device with full limits")
}

/// Browser-facing STT engine combining Mimi codec + STT transformer.
///
/// This is the single entry point that the Web Worker calls.
#[cfg_attr(target_family = "wasm", wasm_bindgen)]
pub struct SttEngine {
    // TODO: MimiCodec + SttStream
}

#[cfg_attr(target_family = "wasm", wasm_bindgen)]
impl SttEngine {
    /// Append a model weight shard (for multi-shard GGUF loading).
    ///
    /// Call this for each shard before calling `loadModel`.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = appendModelShard))]
    pub fn append_model_shard(&mut self, _shard: Vec<u8>) {
        todo!("Buffer shard for later assembly")
    }

    /// Load the model from previously appended shards.
    ///
    /// Uses two-phase loading: parse GGUF → drop reader → finalize tensors on GPU.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = loadModel))]
    pub async fn load_model(&mut self) -> Result<(), JsError> {
        todo!("Two-phase GGUF loading: parse shards, drop reader, finalize on GPU")
    }

    /// Feed PCM audio samples (f32, 16kHz mono).
    ///
    /// Returns transcript text if any new tokens were produced.
    /// Audio goes through: Mimi codec → STT transformer → text.
    #[cfg_attr(target_family = "wasm", wasm_bindgen(js_name = feedAudio))]
    pub async fn feed_audio(&mut self, _samples: &[f32]) -> Result<Option<String>, JsError> {
        todo!("Mimi encode → STT decode → return text")
    }

    /// Flush remaining text after end of speech.
    pub async fn flush(&mut self) -> Result<String, JsError> {
        todo!("Run flush trick to eliminate tail latency")
    }

    /// Reset all state for a new recording session.
    pub fn reset(&mut self) {
        todo!("Reset Mimi codec + STT stream")
    }
}
