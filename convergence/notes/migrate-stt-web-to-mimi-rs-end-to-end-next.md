---
run: migrate-stt-web-to-mimi-rs-end-to-end
completed: 2026-05-03
verdict: PASS
---

# Walls hit
- **Model files absent from repo at test time**: `cargo test` exits 0 with 14/15 tests silently skipping when `models/stt-1b-en_fr-q4.gguf`, `models/mimi.safetensors`, `models/tokenizer.model`, and `web/test-bria.wav` are not present. The loop cannot download from HuggingFace. Any run requiring e2e verification must either rely on models already on disk or have the owner symlink them before the verification step. Document model file expectations explicitly in the queue, and detect silent skips proactively rather than after the fact.
- **Pre-existing usize overflow in mimi-rs/src/transformer.rs:144**: `test_e2e_wav_bria` panics in batch mode for long audio (num_keys < num_queries causes underflow). Triggered for the first time by this run because models were previously absent. Not introduced by the run but now a live failing test in the suite.

# Anti-patterns observed
- **Absolute-path symlinks committed into project repo** (b101b39): `convergence/prompts`, `convergence/templates`, `convergence/tools` were committed as symlinks to `/Users/tc/Code/nordcoop/convergence/...`. Breaks portability; should use relative symlinks or not be committed at all.
- **Hollow-pass risk on `cargo test`**: The test suite does not fail when model-dependent tests skip — it exits 0. Any verification task that relies on `cargo test` exit code alone will give a false green. Future plans should explicitly grep for skip messages and treat them as inconclusive, not passing. (This run did catch it correctly on re-inspection, but the detection was manual.)

# Suggested next goals

1. **Fix usize overflow in mimi-rs transformer (batch mode)** — The panic at `mimi-rs/src/transformer.rs:144` (`let shift = num_keys - num_queries` underflows when `num_keys < num_queries`) causes `test_e2e_wav_bria` to fail in batch mode on long audio. Fix: saturating subtraction or explicit guard. Success shape: `cargo test --features wgpu -- --test-threads=1` exits 0 with no panics; `test_e2e_wav_bria` passes and produces a valid RTF reading; no encode-path regression.

2. **Promote mimi-rs to a remote git dependency** — `stt-web/crates/stt-wasm/Cargo.toml` currently pins mimi-rs via `git = "file:///Users/tc/Code/idle-intelligence/mimi-rs"`. This is machine-local and makes the repo non-reproducible. Success shape: mimi-rs is published to crates.io or mirrored to GitHub (idle-intelligence/mimi-rs), stt-web's dep updated to the remote URL or a semver `version = "0.1.x"` from crates.io, and `cargo build --features wgpu,cli` exits 0 resolving from the remote source.

3. **Add reconstruction-quality assertions to mimi-rs decode tests** — The 5 round-trip tests added in `1d4f530` verify tensor shapes only. A real consumer of `decode_from_codes` needs to know the output is acoustically valid, not just correctly shaped. Success shape: at least one test encodes a known sine-wave or short speech snippet, decodes it, and asserts cosine similarity or normalized MSE is above a reasonable threshold (e.g. > 0.95 cosine similarity); all 5 existing shape tests continue to pass.
