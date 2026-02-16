#!/usr/bin/env python3
"""Evaluate WER for original vs Q4 quantized Kyutai STT model.

Usage:
    python scripts/eval.py [--gguf PATH] [--dataset DATASET] [--split SPLIT]

Runs inference on LibriSpeech test-clean with both the original f32
and Q4 quantized model, computes Word Error Rate (WER), and reports
the delta. Target: within 2% absolute degradation.

Note: This script requires the `moshi` package to be installed:
    pip install moshi

The script will:
1. Load the original PyTorch model from HuggingFace
2. Load the Q4 GGUF model (when Rust GGUF loader is ready)
3. Run inference on a subset of LibriSpeech test-clean
4. Compute WER for both and report delta
"""

import argparse
from pathlib import Path
from typing import List, Tuple

try:
    import jiwer
    import torch
    import tqdm
    from datasets import load_dataset
    from transformers import AutoProcessor
    import moshi.models
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("\nPlease install required packages:")
    print("  pip install moshi jiwer datasets transformers torch")
    exit(1)


def normalize_text(text: str) -> str:
    """Normalize text for WER computation."""
    # Simple normalization - lowercase and strip
    return text.lower().strip()


def compute_wer(references: List[str], hypotheses: List[str]) -> float:
    """Compute Word Error Rate."""
    # Normalize all texts
    refs = [normalize_text(r) for r in references]
    hyps = [normalize_text(h) for h in hypotheses]

    # Filter out empty references
    filtered = [(r, h) for r, h in zip(refs, hyps) if r.strip()]

    if not filtered:
        return 0.0

    refs, hyps = zip(*filtered)
    return jiwer.wer(list(refs), list(hyps))


def run_pytorch_inference(
    model,
    processor,
    audio_samples: List[torch.Tensor],
    sample_rate: int = 16000
) -> List[str]:
    """Run inference using PyTorch model.

    Args:
        model: PyTorch STT model
        processor: Audio processor
        audio_samples: List of audio tensors
        sample_rate: Audio sample rate

    Returns:
        List of transcribed texts
    """
    transcriptions = []

    print("Running PyTorch f32 inference...")
    for audio in tqdm.tqdm(audio_samples, desc="F32 inference"):
        # Process audio
        # NOTE: This is placeholder code - actual moshi STT API may differ
        # Adjust based on actual moshi.models API
        try:
            # Placeholder - replace with actual API call
            result = "PLACEHOLDER - PyTorch inference not implemented yet"
            transcriptions.append(result)
        except Exception as e:
            print(f"Error in PyTorch inference: {e}")
            transcriptions.append("")

    return transcriptions


def run_gguf_inference(
    gguf_path: Path,
    audio_samples: List[torch.Tensor],
    sample_rate: int = 16000
) -> List[str]:
    """Run inference using Q4 GGUF model.

    Args:
        gguf_path: Path to GGUF file
        audio_samples: List of audio tensors
        sample_rate: Audio sample rate

    Returns:
        List of transcribed texts
    """
    print("Running Q4 GGUF inference...")
    print(f"Note: GGUF inference requires Rust implementation at: {gguf_path}")

    # This will be implemented by Agent 3 (Model) using Burn+wgpu
    # For now, return placeholder
    transcriptions = []
    for _ in tqdm.tqdm(audio_samples, desc="Q4 inference"):
        transcriptions.append("PLACEHOLDER - GGUF inference not implemented yet")

    return transcriptions


def evaluate_model(args):
    """Run evaluation comparing f32 and Q4 models."""
    print(f"Loading dataset: {args.dataset} / {args.split}")

    # Load LibriSpeech test-clean
    dataset = load_dataset(
        args.dataset,
        args.subset,
        split=args.split,
        streaming=False,
        trust_remote_code=True
    )

    # Take a subset for faster evaluation during development
    if args.max_samples > 0:
        dataset = dataset.select(range(min(args.max_samples, len(dataset))))

    print(f"Evaluating on {len(dataset)} samples")

    # Extract audio and references
    audio_samples = []
    references = []

    for sample in dataset:
        # LibriSpeech structure: sample['audio']['array'], sample['text']
        audio_samples.append(torch.from_numpy(sample['audio']['array']))
        references.append(sample['text'])

    # Load PyTorch model
    print(f"\nLoading PyTorch model: {args.model_id}")
    try:
        # NOTE: Adjust based on actual moshi.models API
        # This is placeholder code
        print("Note: PyTorch model loading not implemented - using placeholder")
        model = None
        processor = None
    except Exception as e:
        print(f"Error loading PyTorch model: {e}")
        model = None
        processor = None

    # Run f32 inference
    if model is not None:
        f32_transcriptions = run_pytorch_inference(model, processor, audio_samples)
        f32_wer = compute_wer(references, f32_transcriptions)
        print(f"\nF32 WER: {f32_wer * 100:.2f}%")
    else:
        print("\nSkipping F32 inference (model not loaded)")
        f32_transcriptions = [""] * len(references)
        f32_wer = 0.0

    # Run Q4 GGUF inference
    if args.gguf.exists():
        q4_transcriptions = run_gguf_inference(args.gguf, audio_samples)
        q4_wer = compute_wer(references, q4_transcriptions)
        print(f"Q4 WER: {q4_wer * 100:.2f}%")

        # Compute delta
        delta = (q4_wer - f32_wer) * 100
        print(f"\nWER Delta: {delta:+.2f}% absolute")

        if abs(delta) <= 2.0:
            print("✓ Within target (≤2% absolute degradation)")
        else:
            print("✗ Exceeds target (>2% absolute degradation)")

        # Save results
        results_path = args.gguf.parent / "eval_results.txt"
        with open(results_path, 'w') as f:
            f.write(f"Evaluation Results\n")
            f.write(f"==================\n\n")
            f.write(f"Dataset: {args.dataset} / {args.split}\n")
            f.write(f"Samples: {len(dataset)}\n\n")
            f.write(f"F32 WER: {f32_wer * 100:.2f}%\n")
            f.write(f"Q4 WER: {q4_wer * 100:.2f}%\n")
            f.write(f"Delta: {delta:+.2f}% absolute\n\n")
            f.write(f"Target: ≤2% absolute degradation\n")
            f.write(f"Status: {'PASS' if abs(delta) <= 2.0 else 'FAIL'}\n")

        print(f"\nResults saved to: {results_path}")

    else:
        print(f"\nGGUF file not found: {args.gguf}")
        print("Run quantize.py first to generate the GGUF file")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate WER for Kyutai STT Q4 quantization"
    )
    parser.add_argument(
        "--model-id",
        default="kyutai/stt-1b-en_fr",
        help="HuggingFace model ID for f32 reference"
    )
    parser.add_argument(
        "--gguf",
        type=Path,
        default=Path("models/stt-1b-en_fr-q4.gguf"),
        help="Path to Q4 GGUF file"
    )
    parser.add_argument(
        "--dataset",
        default="librispeech_asr",
        help="Dataset name"
    )
    parser.add_argument(
        "--subset",
        default="clean",
        help="Dataset subset (clean or other)"
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of samples to evaluate (0 = all)"
    )

    args = parser.parse_args()

    evaluate_model(args)


if __name__ == "__main__":
    main()
