#!/bin/bash
# Generate a self-signed TLS certificate for localhost HTTPS dev server.
# WebGPU requires a secure context (HTTPS), so this is needed for local dev.
#
# Usage: ./scripts/gen-cert.sh
# Output: /tmp/stt-key.pem, /tmp/stt-cert.pem (valid for 7 days)

set -euo pipefail

openssl req -x509 \
    -newkey ec \
    -pkeyopt ec_paramgen_curve:prime256v1 \
    -keyout /tmp/stt-key.pem \
    -out /tmp/stt-cert.pem \
    -days 7 \
    -nodes \
    -subj "/CN=localhost"

echo "Generated self-signed cert:"
echo "  Key:  /tmp/stt-key.pem"
echo "  Cert: /tmp/stt-cert.pem"
echo ""
echo "Start the dev server with: bun web/serve.mjs"
