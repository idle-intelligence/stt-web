#!/usr/bin/env python3
"""Strip Mimi decoder weights and convert to f16.

Reads the full mimi.safetensors (f32, ~367MB) and produces a smaller
encoder-only file in f16, keeping only the prefixes needed for inference:
  encoder.*, encoder_transformer.*, downsample.*, quantizer.*

Usage:
    python scripts/strip_mimi.py [--input PATH] [--output PATH]

Output: models/mimi-encoder-f16.safetensors
"""

import argparse
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# Tensor prefixes to keep (encoder-side only; drop decoder.* and decoder_transformer.*)
KEEP_PREFIXES = (
    "encoder.",
    "encoder_transformer.",
    "downsample.",
    "quantizer.",
)


def should_keep(name: str) -> bool:
    return any(name.startswith(p) for p in KEEP_PREFIXES)


def strip_and_convert(input_path: Path, output_path: Path) -> None:
    input_size_mb = input_path.stat().st_size / (1024 * 1024)
    print(f"Input:  {input_path}  ({input_size_mb:.1f} MB)")

    kept: dict[str, np.ndarray] = {}
    total_params_before = 0
    total_params_kept = 0

    with safe_open(str(input_path), framework="numpy") as f:
        keys = list(f.keys())
        for name in keys:
            tensor = f.get_tensor(name)
            total_params_before += tensor.size
            if should_keep(name):
                # Convert to float16
                kept[name] = tensor.astype(np.float16)
                total_params_kept += tensor.size

    tensor_count_before = len(keys)
    tensor_count_kept = len(kept)

    print(f"\nBefore: {tensor_count_before} tensors, {total_params_before / 1e6:.2f}M params")
    print(f"After:  {tensor_count_kept} tensors, {total_params_kept / 1e6:.2f}M params")
    print(f"\nKept prefixes: {KEEP_PREFIXES}")

    dropped = [k for k in keys if not should_keep(k)]
    print(f"Dropped {len(dropped)} tensors (decoder.*, decoder_transformer.*, upsample.*)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {output_path} ...")
    save_file(kept, str(output_path))

    output_size_mb = output_path.stat().st_size / (1024 * 1024)
    reduction = (1 - output_size_mb / input_size_mb) * 100
    print(f"\nInput size:  {input_size_mb:.1f} MB")
    print(f"Output size: {output_size_mb:.1f} MB  ({reduction:.0f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description="Strip Mimi decoder weights and convert to f16")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("models/mimi.safetensors"),
        help="Path to full mimi.safetensors (default: models/mimi.safetensors)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/mimi-encoder-f16.safetensors"),
        help="Output path (default: models/mimi-encoder-f16.safetensors)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    strip_and_convert(args.input, args.output)
    print("\nDone.")


if __name__ == "__main__":
    main()
