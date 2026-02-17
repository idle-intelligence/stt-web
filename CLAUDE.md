# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Port Kyutai's `stt-1b-en_fr` speech-to-text model (~1B params) to run 100% client-side in a web browser using Rust → WASM + WebGPU. Decoder-only transformer consuming Mimi audio codec tokens (32 codebooks at 12.5Hz) and producing text tokens on a delayed parallel stream (6 frames / 480ms offset). English + French, CC-BY 4.0.

## Build Commands

```bash
# Native (development and testing)
cargo build --release --features "wgpu,cli"
cargo run --release --features "wgpu,cli" -- --audio test.wav

# WASM (browser)
wasm-pack build crates/stt-wasm --target web --no-default-features --features wasm
wasm-pack build crates/mimi-wasm --target web --features wasm

# Tests
cargo test --features "wgpu"

# Lint (both targets)
cargo clippy --features "wgpu,cli" -- -D warnings
cargo clippy --no-default-features --features wasm --target wasm32-unknown-unknown -- -D warnings

# E2E browser test
bunx playwright test tests/e2e_browser.spec.ts

# Quantization and evaluation (Python)
python scripts/quantize.py   # safetensors → Q4 GGUF
python scripts/eval.py       # WER comparison

# Dev server (HTTPS required for WebGPU)
# Generate self-signed cert first: scripts/gen-cert.sh
bun web/serve.mjs
```

## Repo Structure

```
crates/
  mimi-wasm/          # Mimi audio codec compiled to WASM (CPU-only, ~25M params)
  stt-wasm/           # STT transformer in Burn+wgpu
    src/
      lib.rs          # wasm-bindgen entry points (SttEngine)
      model.rs        # transformer: embeddings, attention, RoPE, KV cache
      gguf.rs         # Q4 GGUF weight loader + WGSL dequant shaders
      stream.rs       # delayed-streams decoding loop
      wgsl/           # custom WebGPU compute shaders
web/
  index.html          # standalone demo page (no bundler)
  worker.js           # Web Worker: loads WASM, orchestrates pipeline
  audio-processor.js  # AudioWorklet: mic → 16kHz mono PCM chunks
  stt-client.js       # optional JS embedding API
scripts/              # quantize.py, eval.py, gen-cert.sh
tests/reference/      # shared test fixtures (wav, tokens, transcripts)
```

## Reference Repos (in `refs/`, gitignored)

- **`refs/moshi/`** (`kyutai-labs/moshi`) — Mimi codec Rust code (`rust/moshi-core/`), Candle-based STT model (`rust/src/lm.rs`, `transformer.rs`). Source for Mimi port and model architecture.
- **`refs/delayed-streams-modeling/`** (`kyutai-labs/delayed-streams-modeling`) — Official PyTorch implementation. Ground truth for delayed-streams logic (`moshi/models/lm.py`). Also has `stt-rs/` (Candle CLI).
- **`refs/voxtral-mini-realtime-rs/`** (`TrevorS/voxtral-mini-realtime-rs`) — **Architectural template.** Burn 0.20 + wgpu + WASM + Q4 GGUF in browser. Copy patterns for: GGUF reader, Q4 WGSL shaders, two-phase weight loading, ShardedCursor, WASM bindings, web worker, KV cache, RoPE, feature flags.

## Critical Constraints

1. **No sync GPU readback in WASM.** Always `into_data_async().await`. Never `.into_data()`.
2. **2GB single ArrayBuffer limit in WASM.** Use `ShardedCursor` (Vec<Vec<u8>>) for multi-shard GGUF reading.
3. **4GB WASM address space.** Use two-phase weight loading: parse GGUF → drop reader → finalize tensors on GPU.
4. **WebGPU workgroup size limit: 256 invocations.** Apply the cubecl-wgpu patch from `refs/voxtral-mini-realtime-rs/patches/cubecl-wgpu-0.9.0/`.
5. **WebGPU requires HTTPS.** Dev server must use self-signed cert for localhost.
6. **All inference in a Web Worker.** Main thread only does UI and mic capture.
7. **Model weights fetched at runtime** from HuggingFace, cached via browser Cache API. Never committed to repo.
8. **Q4 WGSL shaders:** Use naive kernel for WASM (tiled kernel is native-only). See `refs/voxtral-mini-realtime-rs/src/gguf/shader_naive.wgsl`.

## Key Dependencies (follow voxtral-rs versions)

- `burn = "0.20"` (features: std, wgpu)
- `cubecl = "0.9"` (with vendored patch for workgroup limits)
- `wasm-bindgen = "0.2"`, `js-sys`, `web-sys` (WASM target)
- `wgpu = "26"` (direct device init in WASM)
- `hound = "3.5"` (WAV I/O), `rubato = "1.0"` (resampling)
- `tokenizers = "0.22"` (native-only, has C deps)

## Feature Flags

```toml
[features]
default = ["wgpu", "native-tokenizer"]
wgpu = ["burn/wgpu"]
native-tokenizer = ["tokenizers"]   # C deps, not WASM-compatible
wasm = ["wgpu", "dep:wgpu", "wasm-bindgen", "wasm-bindgen-futures",
        "js-sys", "web-sys", "console_error_panic_hook", "getrandom/wasm_js"]
cli = ["clap", "indicatif"]
hub = ["hf-hub"]
```

## Architecture: Inference Pipeline

```
Microphone → AudioWorklet (16kHz mono PCM)
  → Web Worker
    → Mimi codec [WASM, CPU] (PCM → 32 codebook tokens per frame at 12.5Hz)
      → STT Transformer [WASM+WebGPU] (audio tokens + prev text token → next text token)
        → Detokenizer → text → postMessage → UI
```

**Delayed-streams decoding:** Audio and text run as parallel time-aligned streams. Text is delayed by 6 frames (480ms). Each step: model receives current audio frame + previous text token, predicts next text token. "Flush trick": after VAD detects end-of-speech, run remaining frames faster than real-time to eliminate tail latency.

**Q4 weight path:** GGUF file → parse header/tensors → store Q4 blocks as raw bytes on GPU → dequantize via WGSL compute shader → matmul. Embeddings: Q4 on GPU + CPU byte copy for token lookups.

## Worker ↔ Main Thread Protocol

```
Main → Worker: { type: 'load' | 'audio' | 'stop' | 'reset', ... }
Worker → Main: { type: 'status' | 'transcript' | 'error', ... }
```

## Quality Bar

- WER within 2% absolute of PyTorch f32 on LibriSpeech test-clean
- Real-time factor ≥ 1x on 2020+ laptop with integrated GPU
- Page load to transcribing < 10s (after model cached)
- Chrome and Edge required; Firefox best-effort

