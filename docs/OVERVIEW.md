# STT 1B → Browser (WASM + WebGPU)

## What we're building

A browser-native, fully client-side speech-to-text engine based on the `stt-1b-en_fr` model. No server, no API keys, no data leaves the device. The user opens a web page, the model downloads once (~500MB, cached), and from then on they have real-time streaming transcription from their microphone — entirely in their browser tab.

## Why this model

STT 1B is the sweet spot:

| Model | Params | Browser-viable? | Streaming? | Quality |
|---|---|---|---|---|
| Whisper base | 73M | ✅ trivial (~200MB) | ❌ chunked | decent |
| Moonshine Base | ~400M | ✅ easy (~150MB) | ✅ | good |
| **STT 1B** | **1B** | **✅ feasible (~500MB Q4)** | **✅ native streaming** | **excellent** |
| Voxtral Realtime 4B | 4B | 🟡 tight (~2.5GB Q4) | ✅ | SOTA |

STT 1B uses "delayed streams modeling" — audio and text are modeled as parallel time-aligned streams. This means the model is *natively* streaming: it starts outputting text ~500ms after speech begins, with no chunking hacks. It includes a semantic VAD (voice activity detection) that predicts when the user has stopped talking. English + French. CC-BY 4.0 license.

## Architecture

```
Microphone (browser)
  → AudioWorklet (16kHz mono PCM)
    → Web Worker
      → Mimi codec [WASM, CPU] (audio samples → 12.5Hz token frames)
        → STT Transformer [WASM+WebGPU] (audio tokens → text tokens)
          → Detokenizer → text string
            → postMessage to main thread → UI
```

### Component breakdown

**1. Mimi audio codec (~25M params)**
- Converts raw audio into discrete tokens at 12.5Hz (32 codebook tokens per frame)
- Already implemented in pure Rust (`rustymimi` in `kyutai-labs/moshi`)
- Small enough to run on CPU via WASM. No GPU needed for this part

**2. STT Transformer (~1B params)**
- Decoder-only transformer consuming the Mimi tokens
- Predicts text tokens on a parallel stream, offset by ~6 frames (500ms) from audio
- This is the expensive part — must run on WebGPU
- Needs: embeddings, multi-head attention with RoPE, KV cache, linear head

**3. Delayed streams logic**
- The model doesn't do "listen then transcribe" (encoder-decoder like Whisper)
- Instead, audio and text streams run in parallel, with text delayed by a fixed offset
- At each step: consume the next audio frame, predict the text token for N frames ago
- Padding tokens fill the text stream until real text appears
- The semantic VAD predicts end-of-speech probability at each frame

## Key reference implementations

### `kyutai-labs/delayed-streams-modeling` + `kyutai-labs/moshi`
- The official implementation. PyTorch (research), Rust/Candle (production), MLX (Apple)
- The Rust/Candle code is the closest to what we need, but Candle has no WebGPU backend
- `rustymimi` is reusable as-is for the Mimi codec
- Weight format: safetensors (bf16), Candle q8

### `TrevorS/voxtral-mini-realtime-rs`
- Solved the exact same class of problem: STT transformer → Rust → WASM + WebGPU
- Uses the Burn framework with its `wgpu` backend (works native + browser)
- Q4 GGUF quantization with custom WGSL dequant shaders
- Solved all the hard WASM constraints: 2GB allocation limit, async GPU readback, workgroup size limits
- **This is our architectural template.** Many patterns copy directly

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| ML framework | **Burn** (`burn-rs`) | Has native `wgpu` backend → runs on WebGPU in browser. Candle doesn't |
| GPU compute | **wgpu** (via Burn) | Cross-platform: Vulkan/Metal/DX12 native, WebGPU in browser. Same code path |
| WASM toolchain | **wasm-pack** + `wasm-bindgen` | Standard Rust→WASM pipeline. Generates JS bindings automatically |
| Quantization | **Q4** (GGUF or custom) | 1B params × 4 bits ÷ 8 = ~500MB. Fits in browser memory comfortably |
| Audio codec | **rustymimi** (existing Rust) | Already done. Compile to WASM, runs on CPU |
| Audio capture | **AudioWorklet API** | Low-latency mic input in the browser. Feeds PCM chunks to the worker |
| Threading | **Web Worker** | All inference runs off the main thread. UI stays responsive |
| Model hosting | **HuggingFace Hub** | Free CDN, CORS-friendly, standard for ML models. Cached via Cache API |

## Quantization math

```
1B params at Q4:
  1,000,000,000 × 4 bits = 4,000,000,000 bits = 500,000,000 bytes ≈ 500 MB

WASM constraints:
  - Single ArrayBuffer max: 2GB (4GB with memory64, but not universal) ✅ fine
  - Total memory budget: ~4GB practical                                ✅ fine
  - WebGPU buffer max: implementation-dependent, usually 1-2GB         ✅ fine

For comparison:
  - Voxtral 4B Q4 = ~2.5GB (proven to work in browser)
  - Moonshine Base = ~150MB
  - Our target 500MB is comfortably in the middle
```

## Risks and mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Q4 quantization hurts 1B model accuracy too much | Medium | Eval WER early. Fall back to Q8 (~1GB) — still browser-viable |
| Burn's wgpu backend missing ops needed by the model | Medium | voxtral-rs already proved the path for attention + RoPE + KV cache. Copy their custom WGSL shaders. File issues upstream for anything else |
| Mimi codec too slow in WASM-CPU | Low | Mimi is only ~25M params, runs at 12.5Hz. Even WASM-CPU should handle real-time. If not, port Mimi to Burn+wgpu too |
| Delayed-streams logic is subtle to port correctly | Medium | The PyTorch reference is clean. Port line-by-line, verify frame-by-frame against reference outputs on test audio |
| WebGPU workgroup size limits (256 invocations) | Low | voxtral-rs already patched `cubecl-wgpu` for this. Reuse their patch |
| Browser memory pressure on lower-end devices | Low | 500MB model + ~200MB working memory. Any machine with 4GB+ RAM is fine. Show a "your device may not support this" warning if needed |

## Deliverables

The project produces two things:

### 1. Model artifact
- `stt-1b-en_fr-q4.gguf` (or shards) uploaded to HuggingFace
- Verified: WER on LibriSpeech test-clean within acceptable delta of original

### 2. Code repository
```
stt-web/
├── CLAUDE.md                  # project context for Claude Code agents
├── Cargo.toml                 # workspace: mimi-wasm + stt-wasm crates
├── crates/
│   ├── mimi-wasm/             # Mimi codec compiled to WASM
│   │   ├── src/lib.rs         # stripped-down rustymimi, wasm-bindgen exports
│   │   └── Cargo.toml
│   └── stt-wasm/              # STT transformer in Burn+wgpu
│       ├── src/
│       │   ├── lib.rs         # wasm-bindgen entry points
│       │   ├── model.rs       # transformer: embeddings, attention, RoPE, KV cache
│       │   ├── gguf.rs        # Q4 weight loader
│       │   ├── wgsl/          # custom compute shaders (Q4 dequant + matmul)
│       │   └── stream.rs      # delayed-streams decoding logic
│       └── Cargo.toml
├── web/
│   ├── index.html             # standalone demo page
│   ├── worker.js              # web worker: loads WASM, orchestrates pipeline
│   ├── audio-processor.js     # AudioWorklet: mic → PCM chunks
│   └── stt-client.js          # optional: clean JS API for embedding
├── scripts/
│   ├── quantize.py            # convert safetensors → Q4 GGUF
│   ├── eval.py                # WER evaluation script
│   ├── strip_mimi.py          # encoder-only f16 Mimi weights
│   ├── download-assets.sh     # fetch Mimi weights + tokenizer for local dev
│   ├── build.sh                # assemble _site/ for Pages
│   └── serve.py                # local server with COOP/COEP headers
├── tests/
│   └── reference/             # test audio files + expected transcripts
│                              # (Rust integration tests live under crates/*/tests/)
└── README.md
```

The `web/` directory is self-contained. Copy it to any static file server and it works.
