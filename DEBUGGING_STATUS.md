# E2E Testing Status

## Root Cause Found: Mimi Encoder Frame-Boundary Bug

**The "frozen logits" bug was a misdiagnosis.** The STT model was producing padding tokens because the Mimi encoder was feeding it garbage tokens — not because of any WebGPU issue.

### The Bug

`MimiCodec::feed_audio()` splits audio into independent 1920-sample chunks and processes each through `encode_frame()` with zero-padding at boundaries. Conv1d kernels (size 7–16) span frame boundaries, so independent processing produces semantically wrong tokens. The STT model correctly outputs padding (token 3) for garbage input.

### The Fix

Added `encode_all()` to `MimiCodec` — processes the entire waveform through the encoder pipeline in a single pass, avoiding frame-boundary artifacts.

### Verification

| Test | Result |
|------|--------|
| Reference Mimi tokens → STT → text | 0% WER, perfect transcript |
| PyTorch Mimi tokens (bria.wav) → STT → text | Correct transcript |
| Rust `encode_all()` (loona.wav, 1.05s) → STT | "Luna." |
| Rust `encode_all()` (bria.wav, 30s) → STT | Full correct transcript |
| Rust `feed_audio()` (streaming, broken) → STT | All padding tokens |

## Remaining Issues

### 1. Conv1d Performance (BLOCKING)

The naive O(n^4) Conv1d in `crates/mimi-wasm/src/conv.rs` is far too slow:
- 1.05s audio → 4.34s encoding (release mode, native)
- 30s audio → 60+ seconds encoding
- Would be even slower in WASM/CPU in browser

**Needs:** im2col + GEMM optimization, or port streaming conv from refs/moshi.

### 2. Streaming Mode Still Broken

`feed_audio()` (frame-by-frame) still produces wrong tokens. `encode_all()` works but is batch-only — incompatible with real-time mic input in the browser.

**Needs:** Port `StreamableConv1d` with `state_prev_xs` buffers from reference implementations (refs/moshi/rust/moshi-core/src/conv.rs or refs/xn/pocket-tts/src/conv.rs).

### 3. Browser Testing Blocked

Can't meaningfully test in browser until either:
- Conv1d is fast enough for `encode_all()` batch mode (temporary), or
- Streaming conv is fixed for real-time frame-by-frame encoding

## E2E Test Commands

```bash
# Reference tokens → STT → text (fast, no Mimi needed)
cargo test -p stt-wasm --release --features wgpu --test e2e_transcript -- --nocapture

# Real WAV → Mimi → STT → text (slow, needs mimi.safetensors)
cargo test -p stt-wasm --release --features wgpu --test e2e_wav -- --nocapture

# PyTorch Mimi tokens → STT (needs /tmp/mimi_bria_ref.json)
cargo test -p stt-wasm --release --features wgpu --test e2e_pytorch_mimi -- --nocapture
```

## Test Audio Files

| File | Duration | Content |
|------|----------|---------|
| web/test-loona.wav | 1.05s | "Luna" (short utterance test) |
| web/test-bria.wav | ~30s | Forest/nature description (long form) |
| web/test-crepes-fr.wav | ~30s | French audio |
