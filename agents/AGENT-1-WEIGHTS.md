# Agent 1: Model Weights — Q4 GGUF Quantization

You are Agent 1 on a multi-agent team. Read the root `CLAUDE.md` first for full project context.

## Your Goal

Produce Q4-quantized weights for `kyutai/stt-1b-en_fr` that can be loaded by the browser GGUF reader.

## Model Details (from HuggingFace config.json)

```json
{
  "dim": 2048, "num_layers": 16, "num_heads": 16,
  "hidden_scale": 4.125, "context": 750, "max_period": 100000.0,
  "n_q": 32, "card": 2048, "text_card": 8000,
  "existing_text_padding_id": 3,
  "gating": "silu", "norm": "rms_norm_f32", "positional_embedding": "rope",
  "depformer_dim": 1024, "depformer_num_heads": 16, "depformer_num_layers": 6,
  "depformer_multi_linear": true, "depformer_pos_emb": "none",
  "depformer_weights_per_step": true,
  "stt_config": { "audio_delay_seconds": 0.5, "audio_silence_prefix_seconds": 0.0 },
  "model_type": "stt",
  "tokenizer_name": "tokenizer_en_fr_audio_8000.model"
}
```

## Tasks

1. **`scripts/quantize.py`** — Download `kyutai/stt-1b-en_fr` safetensors from HuggingFace. Quantize all linear weight tensors to Q4_0 (4-bit, block_size=32). Write as GGUF format. Shard into ≤512MB files if total exceeds 512MB.

2. **`scripts/eval.py`** — Load both the original f32 and Q4 quantized model. Run inference on LibriSpeech test-clean. Compute WER for both. Report delta. Target: within 2% absolute.

## Q4_0 Format

Each block of 32 values:
- 1 × f16 scale factor (2 bytes)
- 32 × 4-bit quantized values (16 bytes)
- Total: 18 bytes per block of 32 values

## Key References

- `/refs/voxtral-mini-realtime-rs/` — check for any quantization scripts
- GGUF spec: https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
- HuggingFace model: `kyutai/stt-1b-en_fr`
- The `gguf-py` Python package can help with GGUF writing

## Output Files

- `scripts/quantize.py` (replace the stub)
- `scripts/eval.py` (replace the stub)

## Coordination

- Agent 3 (Model) depends on your GGUF output. The tensor names you write must match what the Rust GGUF reader expects.
- Document the exact tensor naming convention you use in the GGUF file.
- The model has two components: the main transformer (16 layers, 2048 dim) and the depformer (6 layers, 1024 dim, per-step weights).

## Done When

- Q4 GGUF weight file(s) exist (or script can produce them)
- WER eval is documented
- Tensor naming convention is documented
