# Quantization and Evaluation Scripts

This directory contains Python scripts for quantizing the STT model and evaluating the quantized weights.

## Setup

Install Python dependencies:

```bash
pip install -r scripts/requirements.txt
```

You may also need to install the `moshi` package from source:

```bash
pip install git+https://github.com/kyutai-labs/moshi.git
```

## Scripts

### `quantize.py` - Model Quantization

Converts the full-precision `kyutai/stt-1b-en_fr` model to Q4 GGUF format.

**Usage:**

```bash
# Download and quantize the model
python scripts/quantize.py

# Custom output path
python scripts/quantize.py --output models/custom-path.gguf

# Adjust shard size (default 512MB)
python scripts/quantize.py --max-shard-size 256
```

**Options:**
- `--model-id`: HuggingFace model ID (default: `kyutai/stt-1b-en_fr`)
- `--output`: Output GGUF file path (default: `models/stt-1b-en_fr-q4.gguf`)
- `--max-shard-size`: Maximum shard size in MB (default: 512)
- `--cache-dir`: HuggingFace cache directory (default: `models/.cache`)

**What it does:**

1. Downloads the model from HuggingFace
2. Loads all safetensors weight files
3. Quantizes linear layer weights to Q4_0 (4-bit with block size 32)
4. Keeps normalization layers as F32 for precision
5. Writes GGUF v3 format with metadata
6. Shards into ≤512MB files if needed (for browser compatibility)

**Output:**

- `models/stt-1b-en_fr-q4.gguf` - Single GGUF file (if < 512MB)
- `models/stt-1b-en_fr-q4-shards/shard-aa`, `shard-ab`, ... - Sharded files (if > 512MB)

### `eval.py` - WER Evaluation

Evaluates Word Error Rate (WER) for the quantized model compared to the original.

**Usage:**

```bash
# Evaluate on LibriSpeech test-clean (default)
python scripts/eval.py

# Custom dataset
python scripts/eval.py --dataset librispeech_asr --subset clean --split test

# Evaluate on full dataset (not just first 100 samples)
python scripts/eval.py --max-samples 0

# Use custom GGUF file
python scripts/eval.py --gguf models/custom-q4.gguf
```

**Options:**
- `--model-id`: HuggingFace model ID for F32 reference (default: `kyutai/stt-1b-en_fr`)
- `--gguf`: Path to Q4 GGUF file (default: `models/stt-1b-en_fr-q4.gguf`)
- `--dataset`: Dataset name (default: `librispeech_asr`)
- `--subset`: Dataset subset (default: `clean`)
- `--split`: Dataset split (default: `test`)
- `--max-samples`: Max samples to evaluate, 0 = all (default: 100)

**What it does:**

1. Loads LibriSpeech test-clean dataset
2. Runs inference with F32 PyTorch model
3. Runs inference with Q4 GGUF model (requires Rust CLI)
4. Computes WER for both
5. Reports delta and pass/fail against 2% threshold
6. Saves results to `models/eval_results.txt`

**Note:** `eval.py` is currently a stub — inference paths are not yet implemented.

**Target:** WER degradation ≤ 2% absolute compared to F32 model

## Tensor Naming Convention

See `docs/TENSOR_NAMING.md` for complete documentation of the GGUF tensor naming scheme.

## Q4_0 Format

Each quantized tensor uses Q4_0 format:

- Block size: 32 elements
- Per block: 1 × f16 scale (2 bytes) + 32 × 4-bit values (16 bytes) = 18 bytes
- Compression ratio: ~7.1× vs F32 (18 bytes for 32 values vs 128 bytes)

## Notes

1. **Dependencies**: The eval script requires `moshi` for PyTorch inference, which may need to be installed from source.

2. **GGUF inference**: The Q4 inference path in `eval.py` is a stub pending the Rust CLI being wired up.

3. **Sharding**: Files are sharded to stay under the browser's 512MB ArrayBuffer limit. The Rust GGUF reader supports multi-shard loading via `ShardedCursor`.

4. **Metadata**: The GGUF file includes model hyperparameters in the metadata section for runtime configuration.
