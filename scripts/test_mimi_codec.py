#!/usr/bin/env python3
"""Test Mimi codec output: compare PyTorch Mimi vs our streaming Mimi.

Since our Rust Mimi can't easily run a standalone test from Python,
this script instead tests:

1. Whether the reference Mimi tokens match what PyTorch produces
   when streaming frame-by-frame (vs batch mode)
2. Sample rate requirements (24kHz, NOT 16kHz)
3. Frame size requirements (1920 samples per frame at 24kHz)

Key finding to verify: the browser AudioWorklet captures at 16kHz,
but Mimi expects 24kHz. If no resampling is happening, this is a bug.

Usage:
    source .venv/bin/activate
    python scripts/test_mimi_codec.py
"""

import json
import sys
from pathlib import Path

import julius
import moshi.models
import moshi.models.loaders
import sphn
import torch


def main():
    base = Path(__file__).resolve().parent.parent
    ref_dir = base / "tests" / "reference"

    # Load reference Mimi tokens
    with open(ref_dir / "mimi_tokens.json") as f:
        ref_data = json.load(f)
    ref_tokens = ref_data["tokens"]
    print(f"Reference: {len(ref_tokens)} frames, {ref_data['num_codebooks']} codebooks")

    # Load Mimi model
    print("\nLoading Mimi model...")
    info = moshi.models.loaders.CheckpointInfo.from_hf_repo("kyutai/stt-1b-en_fr")
    mimi = info.get_mimi(device="cpu")

    print(f"Mimi config:")
    print(f"  sample_rate: {mimi.sample_rate}")
    print(f"  frame_rate: {mimi.frame_rate}")
    print(f"  frame_size: {mimi.frame_size}")
    print(f"  Expected samples per frame: {mimi.frame_size}")

    # Load test audio
    audio_path = "refs/delayed-streams-modeling/audio/bria.mp3"
    print(f"\nLoading audio: {audio_path}")
    audio, input_sr = sphn.read(audio_path)
    audio = torch.from_numpy(audio).to("cpu")
    print(f"  Raw: {audio.shape}, sr={input_sr}")

    # Resample to Mimi's expected rate
    audio_24k = julius.resample_frac(audio, input_sr, mimi.sample_rate)
    print(f"  Resampled to {mimi.sample_rate}Hz: {audio_24k.shape}")

    # Pad to frame boundary
    if audio_24k.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio_24k.shape[-1] % mimi.frame_size
        audio_24k = torch.nn.functional.pad(audio_24k, (0, to_pad))

    n_frames = audio_24k.shape[-1] // mimi.frame_size
    print(f"  Padded: {audio_24k.shape}, {n_frames} frames")

    # Test 1: Stream frame-by-frame and compare to reference
    print(f"\n=== Test 1: Streaming Mimi vs Reference ===")

    chunks = torch.split(audio_24k[:, None], mimi.frame_size, dim=-1)
    streaming_tokens = []

    with mimi.streaming(1):
        for chunk in chunks:
            audio_tokens = mimi.encode(chunk)
            frame = audio_tokens[0, :, 0].cpu().tolist()
            streaming_tokens.append([int(t) for t in frame])

    # Compare
    assert len(streaming_tokens) == len(ref_tokens), (
        f"Frame count mismatch: streaming={len(streaming_tokens)}, ref={len(ref_tokens)}"
    )

    mismatches = 0
    for i, (stream_frame, ref_frame) in enumerate(zip(streaming_tokens, ref_tokens)):
        if stream_frame != ref_frame:
            mismatches += 1
            if mismatches <= 5:
                print(f"  Frame {i}: stream={stream_frame[:5]}... vs ref={ref_frame[:5]}...")

    if mismatches == 0:
        print(f"  PASS: All {len(ref_tokens)} frames match!")
    else:
        print(f"  FAIL: {mismatches}/{len(ref_tokens)} frames differ")

    # Test 2: Show what happens with 16kHz audio (the bug)
    print(f"\n=== Test 2: Wrong Sample Rate (16kHz) ===")
    audio_16k = julius.resample_frac(audio, input_sr, 16000)
    print(f"  16kHz audio: {audio_16k.shape}")

    # Pad
    if audio_16k.shape[-1] % mimi.frame_size != 0:
        to_pad = mimi.frame_size - audio_16k.shape[-1] % mimi.frame_size
        audio_16k = torch.nn.functional.pad(audio_16k, (0, to_pad))

    n_frames_16k = audio_16k.shape[-1] // mimi.frame_size
    print(f"  With 24kHz frame_size={mimi.frame_size}: only {n_frames_16k} frames "
          f"(vs {n_frames} with 24kHz audio)")
    print(f"  That's {n_frames_16k / n_frames:.1%} of the correct frame count")
    print(f"  Audio at wrong rate: tokens will be meaningless garbage")

    # Actually encode 16kHz audio through Mimi to show the tokens are different
    chunks_16k = torch.split(audio_16k[:, None], mimi.frame_size, dim=-1)
    wrong_tokens = []

    with mimi.streaming(1):
        for chunk in chunks_16k:
            audio_tokens = mimi.encode(chunk)
            frame = audio_tokens[0, :, 0].cpu().tolist()
            wrong_tokens.append([int(t) for t in frame])

    # Compare first 10 frames
    print(f"\n  First 5 frames comparison (24kHz vs 16kHz):")
    for i in range(min(5, len(wrong_tokens))):
        ref_frame = ref_tokens[i][:4] if i < len(ref_tokens) else ["N/A"]
        wrong_frame = wrong_tokens[i][:4]
        print(f"    Frame {i}: 24kHz={ref_frame} vs 16kHz={wrong_frame}")

    # Test 3: Show correct streaming config
    print(f"\n=== Test 3: Correct Streaming Configuration ===")
    print(f"  Mimi sample rate: {mimi.sample_rate} Hz")
    print(f"  Mimi frame size: {mimi.frame_size} samples")
    print(f"  Mimi frame rate: {mimi.frame_rate} Hz")
    print(f"  Samples per frame: {mimi.frame_size} (at {mimi.sample_rate}Hz)")
    print(f"  Browser AudioContext should be: sampleRate={mimi.sample_rate}")
    print(f"  OR: resample 16kHz→24kHz before feeding to Mimi")

    # Test 4: Token range validation
    print(f"\n=== Test 4: Token Range Validation ===")
    all_tokens = [t for frame in ref_tokens for t in frame]
    min_tok, max_tok = min(all_tokens), max(all_tokens)
    print(f"  Token range: [{min_tok}, {max_tok}]")
    print(f"  Expected range: [0, 2047] (2048 bins)")
    valid = 0 <= min_tok and max_tok <= 2047
    print(f"  Valid: {'PASS' if valid else 'FAIL'}")

    print(f"\n=== Summary ===")
    print(f"  Streaming Mimi matches reference: {'PASS' if mismatches == 0 else 'FAIL'}")
    print(f"  Token range valid: {'PASS' if valid else 'FAIL'}")

    return 0 if mismatches == 0 and valid else 1


if __name__ == "__main__":
    sys.exit(main())
