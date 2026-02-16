#!/usr/bin/env python3
"""Convert kyutai/stt-1b-en_fr safetensors weights to Q4_0 GGUF format.

Usage:
    python scripts/quantize.py

Downloads the model from HuggingFace, quantizes all linear layers to Q4_0
(4-bit with block size 32), and saves as GGUF. Shards into ≤512MB files
for browser ArrayBuffer safety margin.

Output: models/stt-1b-en_fr-q4.gguf (or shard-aa, shard-ab, ...)
"""

# TODO: Implement quantization pipeline
# 1. Download kyutai/stt-1b-en_fr safetensors from HuggingFace
# 2. Load model weights
# 3. Quantize linear layers to Q4_0 (block_size=32)
# 4. Write GGUF format (header, metadata, tensor data)
# 5. Shard if >512MB

raise NotImplementedError("Quantization pipeline not yet implemented")
