#!/usr/bin/env bash
# build.sh — Build whisper.cpp to WASM + SIMD using Emscripten
#
# Requirements:
#   - Emscripten (emcc) in PATH, or EMSDK_PATH set to the emsdk directory
#   - cmake
#
# Usage:
#   ./whisper/build.sh
#
# Output:
#   whisper/pkg/libwhisper.js       — Emscripten JS glue (with embedded WASM)
#   whisper/pkg/libwhisper.worker.js — pthread worker (Emscripten >= 3.1.58 embeds this)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG_DIR="$SCRIPT_DIR/pkg"
SRC_DIR="$SCRIPT_DIR/src"
BUILD_DIR="$SCRIPT_DIR/build"

# ── Emscripten setup ────────────────────────────────────────────────────────

if ! command -v emcc &>/dev/null; then
    # Try emsdk in temp dir (set by CI / install step)
    EMSDK_PATH="${EMSDK_PATH:-/private/tmp/claude-501/-Users-tc-Code-idle-intelligence-stt-web--claude-worktrees-mobile-fallback/emsdk}"
    if [ -f "$EMSDK_PATH/emsdk_env.sh" ]; then
        echo "Activating emsdk from $EMSDK_PATH..."
        source "$EMSDK_PATH/emsdk_env.sh"
    else
        echo "ERROR: emcc not found. Install Emscripten:"
        echo "  brew install emscripten"
        echo "  OR: git clone https://github.com/emscripten-core/emsdk && cd emsdk && ./emsdk install latest && ./emsdk activate latest && source ./emsdk_env.sh"
        exit 1
    fi
fi

echo "Using emcc: $(command -v emcc)"
emcc --version | head -1

# ── whisper.cpp source ──────────────────────────────────────────────────────

WHISPER_SRC="${WHISPER_SRC:-/private/tmp/claude-501/-Users-tc-Code-idle-intelligence-stt-web--claude-worktrees-mobile-fallback/whisper-cpp-src}"

if [ ! -d "$WHISPER_SRC" ]; then
    echo "Cloning whisper.cpp..."
    git clone --depth=1 https://github.com/ggerganov/whisper.cpp "$WHISPER_SRC"
fi

echo "Using whisper.cpp source: $WHISPER_SRC"

# ── Configure ───────────────────────────────────────────────────────────────

mkdir -p "$BUILD_DIR" "$PKG_DIR"

echo "Configuring whisper.cpp with Emscripten..."
emcmake cmake -S "$WHISPER_SRC" \
    -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DWHISPER_BUILD_TESTS=OFF \
    -DWHISPER_BUILD_EXAMPLES=OFF \
    -DGGML_NATIVE=OFF \
    -DGGML_CPU_AARCH64=OFF \
    -DGGML_SIMD=ON \
    2>&1

# ── Build whisper library ───────────────────────────────────────────────────

echo "Building whisper.cpp library..."
cmake --build "$BUILD_DIR" --target whisper -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

# ── Build our custom WASM module ────────────────────────────────────────────

echo "Building custom WASM wrapper..."

WHISPER_LIB="$BUILD_DIR/src/libwhisper.a"
if [ ! -f "$WHISPER_LIB" ]; then
    WHISPER_LIB="$BUILD_DIR/libwhisper.a"
fi

GGML_LIB="$BUILD_DIR/ggml/src/libggml.a"
if [ ! -f "$GGML_LIB" ]; then
    GGML_LIB="$BUILD_DIR/libggml.a"
fi

# Collect all .a files from the build
ALL_LIBS=$(find "$BUILD_DIR" -name "*.a" | tr '\n' ' ')
echo "Linking against: $ALL_LIBS"

emcc \
    "$SRC_DIR/whisper_wasm.cpp" \
    $ALL_LIBS \
    -I "$WHISPER_SRC/include" \
    -I "$WHISPER_SRC/ggml/include" \
    -O3 \
    -msimd128 \
    --bind \
    -s WASM=1 \
    -s USE_PTHREADS=1 \
    -s PTHREAD_POOL_SIZE=4 \
    -s PTHREAD_POOL_SIZE_STRICT=0 \
    -s INITIAL_MEMORY=512MB \
    -s MAXIMUM_MEMORY=2048MB \
    -s ALLOW_MEMORY_GROWTH=1 \
    -s FORCE_FILESYSTEM=1 \
    -s MODULARIZE=1 \
    -s EXPORT_NAME="WhisperModule" \
    -s SINGLE_FILE=0 \
    -s EXPORTED_RUNTIME_METHODS='["FS", "cwrap", "ccall", "HEAPU8"]' \
    -o "$PKG_DIR/libwhisper.js"

echo ""
echo "Build complete!"
echo "  $PKG_DIR/libwhisper.js"
echo "  $PKG_DIR/libwhisper.wasm"
ls -lh "$PKG_DIR/"
