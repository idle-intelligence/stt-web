# Agent 3: STT Transformer in Burn + wgpu

You are Agent 3 on a multi-agent team. Read the root `CLAUDE.md` first for full project context. **This is the critical path — the hardest and most important agent.**

## Your Goal

Implement the Kyutai STT 1B transformer in Burn with the wgpu backend, working both natively and in WASM+WebGPU. Everything lives in `crates/stt-wasm/`.

## Model Architecture (from config.json)

**Main transformer:**
- `dim`: 2048, `num_layers`: 16, `num_heads`: 16
- `hidden_scale`: 4.125 → FFN hidden = 2048 × 4.125 = 8448
- `norm`: rms_norm_f32, `gating`: silu (SwiGLU)
- `positional_embedding`: rope, `max_period`: 100000.0
- `context`: 750 (sliding window size)
- `causal`: true

**Audio input:**
- `n_q`: 32 codebooks, `card`: 2048 (audio vocab size per codebook)
- `delays`: all zeros (audio codebooks are not delayed relative to each other)

**Text output:**
- `text_card`: 8000 (text vocabulary size)
- `existing_text_padding_id`: 3

**Depformer (text prediction head):**
- `depformer_dim`: 1024, `depformer_num_heads`: 16, `depformer_num_layers`: 6
- `depformer_multi_linear`: true (separate linear heads per step)
- `depformer_weights_per_step`: true
- `depformer_pos_emb`: none

**Streaming config:**
- `stt_config.audio_delay_seconds`: 0.5 (= 6 frames at 12.5Hz)

## Key References (STUDY FIRST)

**voxtral-rs — your architectural template:**
- `/refs/voxtral-mini-realtime-rs/src/models/layers/` — attention.rs, rope.rs, kv_cache.rs, rms_norm.rs, swiglu.rs, decoder_layer.rs
- `/refs/voxtral-mini-realtime-rs/src/gguf/` — reader.rs, loader.rs, model.rs, tensor.rs, linear.rs, op.rs, shader_naive.wgsl

**Kyutai STT — the model you're porting:**
- `/refs/delayed-streams-modeling/stt-rs/src/` — Candle-based STT (main.rs, or whatever files exist)
- `/refs/moshi/rust/moshi-core/src/lm.rs` — the LM model
- `/refs/moshi/rust/moshi-core/src/transformer.rs` — transformer layers

**PyTorch reference:**
- `/refs/delayed-streams-modeling/moshi/models/lm.py` — ground truth for delayed-streams logic

## Files to Implement

1. **`src/model.rs`** — Replace the stubs:
   - Audio embeddings: 32 codebook embedding tables (card=2048 each), summed per frame
   - Text token embedding (text_card=8000)
   - Transformer blocks: RMSNorm → self-attention (causal, RoPE, sliding window 750) → RMSNorm → SwiGLU FFN
   - Note: this model uses MHA (16 heads, 16 kv_heads) not GQA
   - Depformer: 6 smaller transformer layers (1024 dim) that predict text tokens
   - KV cache for autoregressive decoding

2. **`src/gguf.rs`** — Replace the stubs:
   - GGUF v2/v3 parser (copy pattern from voxtral-rs `reader.rs`)
   - ShardedCursor for multi-shard reading
   - Two-phase loading: parse → drop reader → finalize on GPU
   - Q4 dequant via WGSL compute shader (naive kernel for WASM)
   - Adapt tensor names to match Agent 1's GGUF output

3. **`src/wgsl/shader_naive.wgsl`** — Copy from voxtral-rs and adapt

4. **`src/stream.rs`** — Delayed-streams decoding:
   - Buffer audio frames from Mimi
   - After 6 frames (500ms delay), start predicting text
   - Each step: model gets audio frame + previous text token → predict next text
   - Flush trick: when speech ends, process remaining frames fast

5. **`src/web/bindings.rs`** — WASM bindings:
   - `initWgpuDevice()` — create wgpu device with full limits (copy from voxtral-rs)
   - `SttEngine` — appendModelShard, loadModel, feedAudio, flush, reset

6. **`src/lib.rs`** — Update SttConfig defaults to match the real model config above

## Critical Constraints

1. **No sync GPU readback in WASM.** Always `into_data_async().await`. Never `.into_data()`.
2. **2GB single ArrayBuffer limit.** Use ShardedCursor (Vec<Vec<u8>>).
3. **4GB WASM address space.** Two-phase weight loading.
4. **WebGPU 256 workgroup limit.** cubecl-wgpu patch already applied.
5. **Use naive WGSL kernel for WASM.** Tiled kernel is native-only.

## Verification

```bash
cargo check -p stt-wasm --features "wgpu"         # basic check
cargo check -p stt-wasm --features "wgpu,cli"      # with CLI
cargo clippy -p stt-wasm --features "wgpu,cli" -- -D warnings  # lint
```

## Coordination

- **Depends on Agent 1** for Q4 GGUF weights (tensor names must match)
- **Depends on Agent 2** for `mimi-wasm` crate (MimiCodec type used in SttEngine)
- **Agent 4 (Web)** consumes your WASM bindings via worker.js

## Done When

- `cargo check -p stt-wasm --features "wgpu"` passes with real implementation (not todo!())
- Model architecture matches the config.json above
- GGUF loading, WGSL shaders, streaming decoder, and WASM bindings are implemented
- `wasm-pack build crates/stt-wasm --target web --no-default-features --features wasm` succeeds
