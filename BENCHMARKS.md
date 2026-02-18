# Benchmarks

## Hardware

All benchmarks run natively (`cargo test --release --features wgpu`) unless noted.

| | |
|---|---|
| **CPU/GPU** | Apple M2 (10-core GPU) |
| **RAM** | 16 GB unified |
| **Backend** | wgpu (Metal) |
| **Model** | stt-1b-en\_fr-q4.gguf (Q4 quantized, ~600 MB) |

## Streaming Pipeline (Mimi + STT)

End-to-end: WAV → Mimi encoder (CPU) → STT transformer (GPU) → text.
Budget is **80 ms/frame** for real-time at 12.5 Hz.

| File | Duration | Frames | Mimi avg | STT avg | Total avg | RTF | Transcript excerpt |
|------|----------|--------|----------|---------|-----------|-----|-------------------|
| test-loona.wav | 1.1s | 14 | 19.3 ms | 68.9 ms | 88.3 ms | 1.51x | "Luna." |
| test-bria-3s.wav | 3.0s | 38 | 19.8 ms | 49.8 ms | 69.5 ms | 1.00x | "you're rabbit named Luna..." |
| test-bria-5s.wav | 5.0s | 63 | 20.3 ms | 49.8 ms | 70.1 ms | 0.96x | "in the heart of an ancient forest..." |
| test-bria-10s.wav | 10.0s | 125 | 20.6 ms | 50.0 ms | 70.6 ms | 0.92x | "in the heart of an ancient forest..." |
| test-bria-10s-noisy.wav | 10.0s | 125 | 20.9 ms | 50.0 ms | 70.9 ms | 0.92x | _(noisy, no output)_ |
| test-bria.wav | 30.0s | 375 | 21.1 ms | 50.9 ms | 72.1 ms | 0.91x | "in the heart of an ancient forest..." |
| test-crepes-fr-10s.wav | 10.0s | 125 | 20.5 ms | 50.0 ms | 70.4 ms | 0.92x | _(French, no output)_ |
| test-crepes-fr.wav | 30.0s | 375 | 21.0 ms | 50.9 ms | 72.0 ms | 0.91x | _(French, no output)_ |
| test-silence-60s.wav | 60.0s | 750 | 20.7 ms | 52.5 ms | 73.1 ms | 0.92x | _(silence)_ |
| test-bria-120s.wav | 120.0s | 1500 | 21.4 ms | 54.5 ms | 75.9 ms | 0.95x | "in the heart of an ancient forest..." |

**RTF** = Real-Time Factor (wall time / audio duration). Values < 1.0 are faster than real-time.

### Observations

- **Steady state ~70 ms/frame** (Mimi ~21 ms + STT ~50 ms), well within the 80 ms budget.
- **Short clips are slower** (1.1s → 1.51x RTF) due to GPU pipeline warmup on the first few frames. GPU warmup pass (added to loading) reduces this in the browser pipeline.
- **Long clips stay within budget** (120s → 0.95x RTF) thanks to pre-allocated KV cache with `slice_assign`. STT latency remains ~55 ms even at 1500 frames (previously degraded to ~62 ms with `Tensor::cat`).
- **Flush adds ~350–400 ms** to drain the 7-frame text delay after end-of-speech.
- **Mimi is consistent** at ~21 ms/frame regardless of duration (CPU-bound, no state growth).
- **French and noisy clips** produce no text — the model appears to suppress output when confidence is low or language detection fails. Needs investigation.

### Changes (perf/rtf-and-ttfb-optimization)

1. **Pre-allocated KV cache** — replaced `Tensor::cat` (O(n) per step, O(n²) total) with pre-allocated ring buffer using `slice_assign` (O(1) per step). 120s RTF improved from 1.06x to 0.95x.
2. **GPU warmup** — 2 dummy forward passes after model load pre-compile all WGSL shader pipelines. Reduces browser TTFB by ~150-400 ms.
3. **AudioWorklet buffer** — reduced from 1920 to 480 samples (80ms → 20ms). Reduces browser TTFB by ~60 ms.

## Mimi Encoder (Isolated)

Mimi codec only, no STT. Shows the effect of chunk size on streaming overhead.

### Batch vs Streaming

| File | Mode | Chunk Size | Total Time | Frames | Notes |
|------|------|-----------|------------|--------|-------|
| test-loona.wav (1s) | batch | all | ~5.8s | 13 | Single `encode_all` call |
| test-loona.wav (1s) | stream | 480 (20ms) | ~54s | 14 | Many small chunks → high overhead |
| test-loona.wav (1s) | stream | 960 (40ms) | ~42s | 14 | |
| test-loona.wav (1s) | stream | 1920 (80ms) | ~35s | 14 | 1 Mimi frame per chunk |
| test-loona.wav (1s) | stream | 4800 (200ms) | ~19s | 14 | |
| test-loona.wav (1s) | stream | 9600 (400ms) | ~13s | 14 | Diminishing returns |

Streaming produces 14 frames vs batch 13 — the extra frame comes from flushing buffered
data at end-of-input. All 13 comparable frames match exactly. Larger chunks amortize
the per-step convolution overhead.

## Reference Constants

| Constant | Value | Notes |
|----------|-------|-------|
| Mimi frame size | 1920 samples | 80 ms at 24 kHz |
| Mimi frame rate | 12.5 Hz | 1 frame every 80 ms |
| Mimi downsampling | 1920x | 960x encoder × 2x downsample conv |
| Text delay | 7 frames | 560 ms before first text token |
| KV cache window | 250 timesteps | ~10s at internal 25 Hz rate |
| Transformer layers | 16 | 32 codebook inputs per frame |
| Real-time budget | 80 ms/frame | 1000 ms / 12.5 Hz |
