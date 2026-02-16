# Agent 2: Mimi Audio Codec → WASM

You are Agent 2 on a multi-agent team. Read the root `CLAUDE.md` first for full project context.

## Your Goal

Port the Mimi audio codec to a standalone WASM-compatible Rust crate at `crates/mimi-wasm/`.

## What Mimi Does

Mimi is a neural audio codec (~25M params). It converts 16kHz mono PCM audio into discrete tokens:
- Input: f32 PCM samples at 16kHz
- Output: 32 codebook token IDs per frame at 12.5Hz (one frame per 1280 audio samples)
- Architecture: SEANet encoder → small transformer → Residual Vector Quantizer (RVQ, 32 codebooks)
- Runs on CPU (small enough, no GPU needed)

## Key References

Study these files in order:

1. **`/refs/moshi/rust/mimi-pyo3/src/lib.rs`** — Python bindings showing the API surface. Strip the pyo3 parts.
2. **`/refs/moshi/rust/moshi-core/src/`** — The core Mimi Rust implementation:
   - `encodec.rs` or `mimi.rs` — top-level codec
   - `seanet.rs` — SEANet convolutional encoder/decoder
   - `transformer.rs` — small transformer used inside Mimi
   - `quantization.rs` — Residual Vector Quantizer
   - Look for the streaming interface (how audio is fed incrementally)
3. **`/refs/delayed-streams-modeling/` Python code** — if you need to understand expected token output for testing

## Critical Constraints

- **No C dependencies.** Cannot use candle, pyo3, CUDA, or anything that won't compile to `wasm32-unknown-unknown`.
- **Pure Rust tensor ops.** Mimi is small enough (~25M params at 12.5Hz) to run without a GPU framework. Implement Conv1d, ConvTranspose1d, etc. in pure Rust.
- **Streaming.** Must support incremental feeding: call `feed_audio()` with small chunks, get tokens back when enough audio has accumulated.
- **Weights.** Mimi weights are ~100MB (safetensors). They must be fetched at runtime, not embedded. The `safetensors` crate (already in Cargo.toml) can parse them.

## Current State

`crates/mimi-wasm/` exists with a stub `lib.rs`. The Cargo.toml has `safetensors = "0.5"` already added. Fill in the actual implementation.

## Architecture to Port

```
PCM audio (16kHz mono f32)
  → SEANet Encoder (strided convolutions, downsample 320x → 50Hz internal)
    → Encoder Transformer (8 layers, 512 dim, causal)
      → RVQ Quantizer (first 1 codebook at 50Hz, then split into 32 at 12.5Hz)
        → 32 token IDs per frame at 12.5Hz
```

The exact architecture details are in the moshi-core source code. Read it carefully.

## Files to Create/Modify

- `crates/mimi-wasm/src/lib.rs` — public API (replace the stub)
- `crates/mimi-wasm/src/tensor.rs` — lightweight f32 tensor type (3D: batch × channels × time)
- `crates/mimi-wasm/src/conv.rs` — Conv1d, ConvTranspose1d with streaming support
- `crates/mimi-wasm/src/seanet.rs` — SEANet encoder
- `crates/mimi-wasm/src/transformer.rs` — small transformer for Mimi (not the STT transformer)
- `crates/mimi-wasm/src/quantization.rs` — RVQ encoder (encode only, no decode needed)
- `crates/mimi-wasm/src/weights.rs` — safetensors weight loading
- `crates/mimi-wasm/Cargo.toml` — add any needed deps

## Verification

```bash
cargo check -p mimi-wasm                          # native check
cargo check -p mimi-wasm --target wasm32-unknown-unknown --features wasm  # WASM check
```

## Coordination

- Agent 3 (Model) and Agent 4 (Web) consume your output tokens. The `SttEngine` in `crates/stt-wasm/src/web/bindings.rs` wraps your `MimiCodec`.
- Your `feed_audio()` returns `Vec<u32>` — flat array of `[frame0_tok0..frame0_tok31, frame1_tok0..frame1_tok31, ...]`.

## Done When

- `cargo check -p mimi-wasm` passes
- `wasm-pack build crates/mimi-wasm --target web --features wasm` succeeds (or gets close)
- The architecture is correct even if numerical tuning is needed later
