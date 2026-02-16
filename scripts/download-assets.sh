#!/bin/bash
# Download Mimi codec weights and tokenizer for local development.
#
# These are small enough to fetch directly; the STT model weights
# need to be quantized first via quantize.py.
#
# Usage: ./scripts/download-assets.sh

set -euo pipefail

MODELS_DIR="models"
mkdir -p "$MODELS_DIR"

echo "Downloading Mimi codec weights..."
if [ ! -f "$MODELS_DIR/mimi.safetensors" ]; then
    curl -L -o "$MODELS_DIR/mimi.safetensors" \
        "https://huggingface.co/kyutai/mimi/resolve/main/model.safetensors"
    echo "  Done: $MODELS_DIR/mimi.safetensors ($(du -h "$MODELS_DIR/mimi.safetensors" | cut -f1))"
else
    echo "  Already exists: $MODELS_DIR/mimi.safetensors"
fi

echo "Downloading tokenizer..."
if [ ! -f "$MODELS_DIR/tokenizer_spm_32k_3.model" ]; then
    curl -L -o "$MODELS_DIR/tokenizer_spm_32k_3.model" \
        "https://huggingface.co/kyutai/stt-1b-en_fr/resolve/main/tokenizer_spm_32k_3.model"
    echo "  Done: $MODELS_DIR/tokenizer_spm_32k_3.model"
else
    echo "  Already exists: $MODELS_DIR/tokenizer_spm_32k_3.model"
fi

echo ""
echo "Assets ready in $MODELS_DIR/"
echo ""
echo "Next steps:"
echo "  1. Quantize the STT model:  python scripts/quantize.py"
echo "  2. Generate TLS cert:       ./scripts/gen-cert.sh"
echo "  3. Start dev server:        bun web/serve.mjs"
echo "  4. Open Chrome:             https://localhost:8443"
