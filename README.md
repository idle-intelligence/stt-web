# stt-web

Browser-native speech-to-text running 100% client-side via Rust/WASM + WebGPU.

[**Try the demo →**](https://idle-intelligence.github.io/stt-web/web/)

> **Disclaimer:** This is an experimental port. The model weights are Q4-quantized from [kyutai/stt-1b-en_fr](https://huggingface.co/kyutai/stt-1b-en_fr) (CC-BY 4.0). Transcription quality may differ from the original PyTorch implementation. This project is not affiliated with or endorsed by Kyutai Labs.

## Status

Work in progress. The pipeline runs end-to-end in Chrome/Edge with WebGPU:

```
Microphone → AudioWorklet (24kHz mono) → Mimi codec (WASM/CPU) → STT transformer (WASM/WebGPU) → text
```

## Requirements

- Chrome 113+ or Edge 113+ (WebGPU required)
- HTTPS (required for WebGPU; dev server uses self-signed cert)

## Quick Start

```bash
# 1. Quantize model weights (requires Python + PyTorch)
python scripts/quantize.py

# 2. Build WASM packages
wasm-pack build crates/mimi-wasm --target web --features wasm
wasm-pack build crates/stt-wasm --target web --no-default-features --features wasm

# 3. Generate self-signed cert for HTTPS dev server
scripts/gen-cert.sh

# 4. Start dev server
bun web/serve.mjs
```

## Architecture

- **Mimi codec** (`crates/mimi-wasm/`): SEANet encoder + transformer + RVQ quantizer. Runs on CPU via ndarray. Converts 24kHz mono PCM → 32 codebook tokens per frame at 12.5Hz.
- **STT transformer** (`crates/stt-wasm/`): 16-layer decoder-only transformer with Q4-quantized weights. Runs on GPU via Burn + wgpu. Predicts text tokens from audio tokens.
- **Web UI** (`web/`): AudioWorklet captures mic at 24kHz, Web Worker orchestrates pipeline, SentencePiece decoder produces text.

## Testing

```bash
# Generate reference data from PyTorch model
source .venv/bin/activate
python scripts/test_reference.py

# Verify Q4 quantization quality
python scripts/test_quantization.py

# Test JS tokenizer
node scripts/test_tokenizer.mjs

# Verify Q4 roundtrip encoding
python scripts/test_q4_roundtrip.py
```
