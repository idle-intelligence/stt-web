#!/usr/bin/env python3
"""Convert kyutai/stt-1b-en_fr safetensors weights to Q4_0 GGUF format.

Usage:
    python scripts/quantize.py [--output OUTPUT_PATH] [--max-shard-size MB]

Downloads the model from HuggingFace, quantizes weight matrices to Q4_0
(4-bit with block size 32), and saves as GGUF. Shards into ≤512MB files
for browser ArrayBuffer safety margin.

Output: models/stt-1b-en_fr-q4.gguf (or shard-aa, shard-ab, ...)

## Actual safetensors Tensor Names

The GGUF file preserves the exact safetensors names from kyutai/stt-1b-en_fr:

**Text Embedding:**
- `text_emb.weight` - [8001, 2048]

**Audio Embeddings** (32 codebooks):
- `emb.{0-31}.weight` - [2049, 2048] for each codebook

**Transformer Layers** (16 layers, i=0..15):
- `transformer.layers.{i}.norm1.alpha` - [2048] (F32, not quantized)
- `transformer.layers.{i}.self_attn.in_proj_weight` - [6144, 2048] (Q4)
- `transformer.layers.{i}.self_attn.out_proj.weight` - [2048, 2048] (Q4)
- `transformer.layers.{i}.norm2.alpha` - [2048] (F32, not quantized)
- `transformer.layers.{i}.mlp.gating.linear_in.weight` - [11264, 2048] (Q4)
- `transformer.layers.{i}.mlp.gating.linear_out.weight` - [2048, 5632] (Q4)

**Output:**
- `out_norm.alpha` - [2048] (F32, not quantized)
- `text_linear.weight` - [8000, 2048] (Q4)

**VAD Heads (skipped):**
- `extra_heads.{0-3}.weight` - [6, 2048] (not needed for STT inference)

Quantization strategy:
- Q4_0: All weight matrices including embeddings (in_proj, out_proj, linear_in, linear_out, text_linear, emb.*, text_emb)
- F32: Norm parameters only (norm1.alpha, norm2.alpha, out_norm.alpha)
- Skipped: extra_heads.* (VAD not used)
"""

import argparse
import struct
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open

# GGUF constants
GGUF_MAGIC = 0x46554747  # "GGUF"
GGUF_VERSION = 3
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2

# Q4_0 block size
Q4_BLOCK_SIZE = 32


def quantize_q4_0(tensor: torch.Tensor) -> bytes:
    """Quantize a tensor to Q4_0 format.

    Q4_0 format: For each block of 32 values:
    - 1 × f16 scale factor (2 bytes)
    - 32 × 4-bit quantized values (16 bytes)
    - Total: 18 bytes per block

    Args:
        tensor: Float tensor to quantize (must have size divisible by 32)

    Returns:
        Raw bytes in Q4_0 format
    """
    # Flatten tensor
    data = tensor.flatten().float().numpy()

    # Pad to multiple of 32 if needed
    remainder = len(data) % Q4_BLOCK_SIZE
    if remainder != 0:
        padding = Q4_BLOCK_SIZE - remainder
        data = np.pad(data, (0, padding), mode='constant', constant_values=0.0)

    num_blocks = len(data) // Q4_BLOCK_SIZE
    output = bytearray()

    for i in range(num_blocks):
        block = data[i * Q4_BLOCK_SIZE:(i + 1) * Q4_BLOCK_SIZE]

        # Compute scale factor (max absolute value)
        abs_max = np.abs(block).max()
        scale = abs_max / 7.0 if abs_max > 0 else 1.0

        # Quantize to 4-bit signed integers (-8..7)
        quantized = np.round(block / scale).astype(np.int8)
        quantized = np.clip(quantized, -8, 7)

        # Pack scale as f16
        scale_f16 = np.float16(scale)
        output.extend(struct.pack('<e', scale_f16))

        # Pack pairs of 4-bit values into bytes
        for j in range(0, Q4_BLOCK_SIZE, 2):
            v0 = int(quantized[j]) & 0x0F
            v1 = int(quantized[j + 1]) & 0x0F
            byte = (v1 << 4) | v0
            output.append(byte)

    return bytes(output)


def write_gguf_string(f, s: str):
    """Write a string in GGUF format (u64 length + UTF-8 bytes)."""
    encoded = s.encode('utf-8')
    f.write(struct.pack('<Q', len(encoded)))
    f.write(encoded)


def write_gguf_metadata(f, metadata: Dict):
    """Write GGUF metadata key-value pairs."""
    # For now, write minimal metadata
    f.write(struct.pack('<Q', len(metadata)))  # num key-value pairs

    for key, value in metadata.items():
        write_gguf_string(f, key)

        # Determine type and write
        if isinstance(value, str):
            f.write(struct.pack('<I', 8))  # GGUF_TYPE_STRING
            write_gguf_string(f, value)
        elif isinstance(value, int):
            f.write(struct.pack('<I', 6))  # GGUF_TYPE_INT64
            f.write(struct.pack('<q', value))
        elif isinstance(value, float):
            f.write(struct.pack('<I', 5))  # GGUF_TYPE_FLOAT64
            f.write(struct.pack('<d', value))
        else:
            raise ValueError(f"Unsupported metadata type: {type(value)}")


def write_gguf_header(f, tensors: List[Tuple[str, torch.Tensor, int]], metadata: Dict):
    """Write GGUF v3 header."""
    # Magic and version
    f.write(struct.pack('<I', GGUF_MAGIC))
    f.write(struct.pack('<I', GGUF_VERSION))

    # Tensor count and metadata KV count
    f.write(struct.pack('<Q', len(tensors)))

    # Write metadata
    write_gguf_metadata(f, metadata)

    # Write tensor index
    offset = 0
    for name, tensor, ggml_type in tensors:
        write_gguf_string(f, name)

        # Number of dimensions
        ndims = len(tensor.shape)
        f.write(struct.pack('<I', ndims))

        # Dimensions (reversed for GGUF)
        for dim in reversed(tensor.shape):
            f.write(struct.pack('<Q', dim))

        # GGML type
        f.write(struct.pack('<I', ggml_type))

        # Offset from start of data section
        f.write(struct.pack('<Q', offset))

        # Calculate size and update offset
        num_elements = tensor.numel()
        if ggml_type == GGML_TYPE_Q4_0:
            # Pad to Q4_BLOCK_SIZE
            num_elements_padded = ((num_elements + Q4_BLOCK_SIZE - 1) // Q4_BLOCK_SIZE) * Q4_BLOCK_SIZE
            byte_size = (num_elements_padded // Q4_BLOCK_SIZE) * 18
        elif ggml_type == GGML_TYPE_F32:
            byte_size = num_elements * 4
        elif ggml_type == GGML_TYPE_F16:
            byte_size = num_elements * 2
        else:
            raise ValueError(f"Unsupported GGML type: {ggml_type}")

        offset += byte_size

    # Align to 32 bytes
    header_end = f.tell()
    alignment = 32
    padding = (alignment - (header_end % alignment)) % alignment
    f.write(b'\x00' * padding)


def should_quantize(name: str) -> bool:
    """Determine if a tensor should be quantized to Q4_0 or kept as F32.

    Returns True for weight matrices to quantize to Q4_0.
    Returns False for norm parameters (keep F32) and VAD heads (skip).
    """
    # Skip VAD heads entirely
    if name.startswith('extra_heads.'):
        return False

    # Keep norm alpha parameters as F32 (small 1D tensors)
    if '.alpha' in name:
        return False

    # Quantize all weight matrices to Q4_0, including embeddings
    # (emb.*.weight, text_emb.weight, in_proj_weight, out_proj.weight,
    #  linear_in.weight, linear_out.weight, text_linear.weight)
    if '.weight' in name:
        return True

    return False


def load_and_quantize_model(model_dir: Path, output_path: Path, max_shard_mb: int = 512):
    """Load safetensors model and quantize to GGUF."""
    print(f"Loading model from {model_dir}")

    # Find safetensors files
    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    print(f"Found {len(safetensors_files)} safetensors files")

    # Load all tensors
    all_tensors = {}
    for st_file in safetensors_files:
        print(f"  Loading {st_file.name}...")
        with safe_open(st_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                all_tensors[key] = f.get_tensor(key)

    print(f"Loaded {len(all_tensors)} tensors")

    # Map safetensors names to GGUF names and prepare for quantization
    gguf_tensors: List[Tuple[str, torch.Tensor, int]] = []

    for name, tensor in sorted(all_tensors.items()):
        # Skip VAD heads (not needed for STT inference)
        if name.startswith('extra_heads.'):
            print(f"  SKIP: {name} {list(tensor.shape)}")
            continue

        # Map tensor names (keep safetensors names as-is)
        gguf_name = name

        # Determine if we should quantize
        if should_quantize(name):
            ggml_type = GGML_TYPE_Q4_0
            print(f"  Q4: {name} {list(tensor.shape)}")
        else:
            ggml_type = GGML_TYPE_F32
            print(f"  F32: {name} {list(tensor.shape)}")

        gguf_tensors.append((gguf_name, tensor, ggml_type))

    # Prepare metadata
    metadata = {
        "general.architecture": "kyutai-stt",
        "general.name": "stt-1b-en_fr",
        "stt.context_length": 750,
        "stt.embedding_length": 2048,
        "stt.block_count": 16,
        "stt.feed_forward_length": 5632,
        "stt.attention.head_count": 16,
        "stt.vocab_size": 8000,
        "stt.audio_vocab_size": 2049,
        "stt.rope.freq_base": 100000.0,
    }

    # Write GGUF file
    print(f"\nWriting GGUF to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        # Write header with tensor index
        write_gguf_header(f, gguf_tensors, metadata)

        # Write tensor data
        total_bytes = 0
        for name, tensor, ggml_type in gguf_tensors:
            if ggml_type == GGML_TYPE_Q4_0:
                data = quantize_q4_0(tensor)
            elif ggml_type == GGML_TYPE_F32:
                data = tensor.flatten().float().numpy().tobytes()
            elif ggml_type == GGML_TYPE_F16:
                data = tensor.flatten().half().numpy().tobytes()
            else:
                raise ValueError(f"Unsupported type: {ggml_type}")

            f.write(data)
            total_bytes += len(data)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\nWrote {file_size_mb:.2f} MB to {output_path}")

    # Shard if needed
    if file_size_mb > max_shard_mb:
        print(f"\nSharding into {max_shard_mb}MB chunks...")
        shard_dir = output_path.parent / f"{output_path.stem}-shards"
        shard_dir.mkdir(exist_ok=True)

        max_shard_bytes = max_shard_mb * 1024 * 1024

        with open(output_path, 'rb') as f_in:
            shard_idx = 0
            while True:
                chunk = f_in.read(max_shard_bytes)
                if not chunk:
                    break

                # Use naming convention: shard-aa, shard-ab, ...
                shard_name = f"shard-{chr(97 + shard_idx // 26)}{chr(97 + shard_idx % 26)}"
                shard_path = shard_dir / shard_name

                with open(shard_path, 'wb') as f_out:
                    f_out.write(chunk)

                shard_size_mb = len(chunk) / (1024 * 1024)
                print(f"  {shard_name}: {shard_size_mb:.2f} MB")
                shard_idx += 1

        print(f"\nCreated {shard_idx} shards in {shard_dir}")


def main():
    parser = argparse.ArgumentParser(description="Quantize Kyutai STT model to Q4 GGUF")
    parser.add_argument(
        "--model-id",
        default="kyutai/stt-1b-en_fr",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/stt-1b-en_fr-q4.gguf"),
        help="Output GGUF file path"
    )
    parser.add_argument(
        "--max-shard-size",
        type=int,
        default=512,
        help="Maximum shard size in MB"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("models/.cache"),
        help="HuggingFace cache directory"
    )

    args = parser.parse_args()

    # Download model from HuggingFace
    print(f"Downloading {args.model_id} from HuggingFace...")
    model_dir = Path(snapshot_download(
        repo_id=args.model_id,
        cache_dir=args.cache_dir,
        allow_patterns=["*.safetensors", "config.json"],
    ))

    # Quantize and save
    load_and_quantize_model(model_dir, args.output, args.max_shard_size)

    print("\nDone! Quantization complete.")
    print(f"\nGGUF file: {args.output}")
    if (args.output.parent / f"{args.output.stem}-shards").exists():
        print(f"Shards: {args.output.parent / f'{args.output.stem}-shards'}/")


if __name__ == "__main__":
    main()
