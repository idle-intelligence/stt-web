#!/usr/bin/env bash
# Builds the stt-wasm package and assembles the deployed site into _site/.
#
# then rewrites the `?v=` build tag to ENGINE_BUILD on the wasm module URL
# in web/worker.js, and checks the built wasm carries no local build path.
#
# Requires wasm-pack, wasm-bindgen-cli (version matching Cargo.lock's
# wasm-bindgen exactly: `cargo install wasm-bindgen-cli --version <ver>
# --locked`).
#
# Usage: ENGINE_BUILD=<tag> scripts/build.sh
# ENGINE_BUILD defaults to "dev" for local builds; CI passes the commit sha.
# It is required on every real deploy - a rebuild with no tag bump keeps
# browsers running the cached module.
set -euo pipefail

ENGINE_BUILD="${ENGINE_BUILD:-dev}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Building stt-wasm (wasm feature) for ENGINE_BUILD=$ENGINE_BUILD"
# Per-target flags (+simd128, the getrandom_backend cfg) live in
# .cargo/config.toml. Only the machine-dependent --remap-path-prefix flags
# are passed inline, so compiled-in source paths carry no home directory.
# They go in with --config, which merges with the config file (RUSTFLAGS
# would replace it); wasm-pack forwards the extra args to `cargo build`
# with no `--` separator. The cfg(all()) key puts them after any user-level
# cfg(all()) remap, so they win (rustc applies the last matching remap);
# with --target they still apply only to the wasm, not to host build
# scripts.
# This moves to `trim-paths = "all"` in the release profile once cargo
# stabilizes it.
CARGO_HOME_DIR="${CARGO_HOME:-$HOME/.cargo}"
REMAP="target.'cfg(all())'.rustflags=[\"--remap-path-prefix=$HOME=/home\", \"--remap-path-prefix=$CARGO_HOME_DIR=/cargo\"]"
wasm-pack build crates/stt-wasm --target web --release --no-default-features --features wasm \
  --config "$REMAP"

WASM="crates/stt-wasm/pkg/stt_wasm_bg.wasm"

# --- Local-path / user-name leak check on the built wasm ---
LEAKS=$(strings "$WASM" | grep -F -e "$HOME" -e "Code/" -e ".claude/" -e "/Users/" || true)
USER_HITS=$(strings "$WASM" | grep -Fw -e "$(id -un)" || true)
if [ -n "$LEAKS$USER_HITS" ]; then
    echo "error: $WASM contains local paths or the user name:" >&2
    printf '%s\n%s\n' "$LEAKS" "$USER_HITS" | grep -v '^$' | head -20 >&2
    exit 1
fi
echo "==> no local paths in built wasm"

# --- Assemble the deployed site into _site/ ---
echo "==> Assembling _site"
rm -rf _site
mkdir -p _site/pkg _site/web
cp crates/stt-wasm/pkg/stt_wasm.js crates/stt-wasm/pkg/stt_wasm_bg.wasm crates/stt-wasm/pkg/package.json _site/pkg/
cp web/index.html web/worker.js web/audio-processor.js web/stt-client.js \
   web/apple-touch-icon.png web/favicon.ico web/test-bria.wav _site/web/

# --- Rewrite the ?v= build tag to ENGINE_BUILD on the wasm loading URL ---
echo "==> Rewriting ENGINE_BUILD tag to $ENGINE_BUILD"
sed -i.bak "s/const ENGINE_BUILD = \"[^\"]*\";/const ENGINE_BUILD = \"$ENGINE_BUILD\";/" \
  _site/web/worker.js
rm -f _site/web/worker.js.bak

COUNT="$(grep -c "const ENGINE_BUILD = \"$ENGINE_BUILD\";" _site/web/worker.js)"
if [ "$COUNT" -ne 1 ]; then
    echo "error: expected 1 ENGINE_BUILD assignment rewritten to $ENGINE_BUILD, found $COUNT" >&2
    exit 1
fi

# --- Target features: the wasm keeps SIMD ---
echo "==> Checking wasm target features"
python3 scripts/check_wasm_features.py "$WASM" +simd128

echo "==> Wrote _site"
