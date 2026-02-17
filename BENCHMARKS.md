# Benchmark Results

## Test Files

| File | Duration | Language | Description |
|------|----------|----------|-------------|
| test-loona.wav | ~1s | EN | Short clip |
| test-bria.wav | ~30s | EN | Medium clip |
| test-crepes-fr.wav | ~30s | FR | French clip |

## Mimi Encoder Performance

_Measured on native (cargo test, release mode, Apple Silicon)_

### Batch Mode (`encode_all`)

| File | Duration | Encode Time | Frames |
|------|----------|-------------|--------|
| test-loona.wav | ~1s | ~5.8s | 13 |

### Streaming Mode (`feed_audio`)

| File | Chunk Size | Total Encode | Frames | Tokens Match Batch? |
|------|-----------|-------------|--------|---------------------|
| test-loona.wav | 480 (20ms) | ~54s | 14 | 13/13 (100%) |
| test-loona.wav | 960 (40ms) | ~42s | 14 | 13/13 (100%) |
| test-loona.wav | 1920 (80ms) | ~35s | 14 | 13/13 (100%) |
| test-loona.wav | 2400 (100ms) | ~31s | 14 | 13/13 (100%) |
| test-loona.wav | 4800 (200ms) | ~19s | 14 | 13/13 (100%) |
| test-loona.wav | 9600 (400ms) | ~13s | 14 | 13/13 (100%) |

Streaming produces 14 frames vs batch 13 — the extra frame comes from flushing buffered
data at end-of-input. All 13 comparable frames match exactly. Larger chunk sizes amortize
the per-step convolution overhead, significantly improving throughput.

## STT End-to-End Performance (Browser, WebGPU)

_Not yet measured. Start the dev server (`bun web/serve.mjs`) and record results here._

Suggested columns: File | Audio Duration | Total Time | RTF | Device | Transcript excerpt

## Reference

Key timing constants:

| Constant | Value | Notes |
|----------|-------|-------|
| Mimi frame size | 1920 samples | At 24kHz = 80ms per frame |
| Mimi frame rate | 12.5 Hz | 1000ms / 80ms |
| Mimi downsampling | 1920× | 960× encoder × 2× downsample conv |
| Text delay | 7 frames | 560ms before first text token |
| KV cache window | 250 timesteps | ~10s at internal 25Hz rate |
| Transformer layers | 16 | 32 codebook inputs per frame |
