#!/usr/bin/env python3
"""Evaluate WER for original vs Q4 quantized Kyutai STT model.

Usage:
    python scripts/eval.py

Runs inference on LibriSpeech test-clean with both the original f32
and Q4 quantized model, computes Word Error Rate (WER), and reports
the delta. Target: within 2% absolute degradation.
"""

# TODO: Implement WER evaluation
# 1. Load original kyutai/stt-1b-en_fr model
# 2. Load Q4 GGUF quantized model
# 3. Run inference on LibriSpeech test-clean
# 4. Compute WER for both
# 5. Report delta

raise NotImplementedError("WER evaluation not yet implemented")
