# Benchmark Results

## Test Files

| File | Duration | Language | Description |
|------|----------|----------|-------------|
| test-loona.wav | ~1s | EN | Short clip |
| test-bria.wav | ~30s | EN | Medium clip |
| test-crepes-fr.wav | ~30s | FR | French clip |

## Mimi Encoder Performance

### Batch Mode (`encode_all`)

_Measured on native (cargo test, release mode, Apple Silicon)_

| File | Duration | Encode Time | Frames |
|------|----------|-------------|--------|
| test-loona.wav | ~1s | ~5.8s | 13 |

### Streaming Mode (`feed_audio`)

_Measured on native (cargo test, release mode, Apple Silicon)_

| File | Chunk Size | Total Encode | Frames | Tokens Match Batch? |
|------|-----------|-------------|--------|---------------------|
| test-loona.wav | 480 (20ms) | ~54s | 14 | 13/13 (100%) |
| test-loona.wav | 960 (40ms) | ~42s | 14 | 13/13 (100%) |
| test-loona.wav | 1920 (80ms) | ~35s | 14 | 13/13 (100%) |
| test-loona.wav | 2400 (100ms) | ~31s | 14 | 13/13 (100%) |
| test-loona.wav | 4800 (200ms) | ~19s | 14 | 13/13 (100%) |
| test-loona.wav | 9600 (400ms) | ~13s | 14 | 13/13 (100%) |

**Note:** Streaming produces 14 frames vs batch 13 — the extra frame comes from
flushing buffered data. All 13 comparable frames match exactly.

**Note:** Streaming is much slower than batch because each step() call recomputes
convolutions on overlapping state buffers. Larger chunks amortize this overhead.

## STT End-to-End Performance (Browser, WebGPU)

_Not yet measured — run the dev server and test manually to populate this section._

## Bug Fixes Applied

1. **Conv1d padding mismatch (ROOT CAUSE of 0% token match)**
   - Streaming step() used `effective_kernel - stride` for initial causal padding
   - Batch forward() used `(kernel_size - 1) * dilation`
   - For stride>1 convs (e.g., kernel=8, stride=4): batch pads 7, streaming pads 4
   - Fix: step() now uses `self.padding()` (same formula as batch)
   - Result: 100% frame match across all chunk sizes

2. **tokenizer.js .trim() strips streaming word spaces**
   - `.trim()` removed leading ▁-encoded spaces in streaming mode
   - Fix: removed `.trim()` from decode()

## Architecture Notes

- Mimi total downsampling: 960x (encoder) * 2x (downsample conv) = 1920x
- At 24kHz: 1920 samples = 1 Mimi frame (80ms)
- AudioWorklet sends ~1920 sample chunks at 24kHz (one frame per chunk)
- File feed sends 2400 sample chunks (~1.25 frames per chunk)
- text_delay = 7 frames (560ms) — first 7 Mimi frames produce no text
- STT transformer: 16 layers, 32 codebooks per frame
- KV cache context window: 250 timesteps (~10s at internal 25Hz)
