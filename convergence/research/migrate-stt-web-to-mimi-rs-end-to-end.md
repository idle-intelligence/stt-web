# Survey: migrate-stt-web-to-mimi-rs-end-to-end

## Goal

Finish the mimi-rs decode-path additions currently sitting as ~119 uncommitted lines on
`feat/optional-encoder` in `/Users/tc/Code/idle-intelligence/mimi-rs`, commit + merge to main,
then update stt-web to point at the merged version and verify the end-to-end STT pipeline still
produces correct transcripts with no RTF regression.

**Interpretation note:** The goal states "replace its current inlined/older audio codec" but that
migration already happened — PR #9 (`feat/migrate-to-mimi-rs`, merged, commit `2d09f6a`) deleted
`mimi-wasm` and replaced it with `mimi-rs`. stt-web is already on mimi-rs for encoding. The
`mimi-wasm` crate was deleted at `6b6d11b`. No old codec remains to remove (grep confirms zero
`mimi-wasm` references in current source). The *actual* remaining work is:
1. Commit the decode path (`decode_from_codes`, `dequantize_codes`, `dequantize_codes_n`, `decode_n`
   variants) that is staged but uncommitted in mimi-rs.
2. Merge `feat/optional-encoder` → `main` in mimi-rs.
3. Update stt-web's path dependency to track the merged main (or cut a tagged version).
4. Confirm STT build, tests, and browser run still work (the decode API is additive; STT only uses
   encoding today).

## Project context

`stt-web` is a 100% browser-side speech-to-text system (Rust → WASM + WebGPU). The inference
pipeline is: AudioWorklet (mic PCM) → Web Worker → Mimi encoder (CPU, candle) → STT transformer
(GPU, burn+wgpu) → text. `mimi-rs` is a sibling repo at `../mimi-rs` (local path dep). The project
uses `wasm-pack`, `bun` (but user prefers `node` for the dev server), and targets Chrome/Edge as
primary browsers. Commits should be small and atomic. `cargo clippy` before committing is required.
No WASM sync GPU readback. Model weights fetched from HuggingFace at runtime, not committed.
WASM binaries deployed to gh-pages `/pkg`, not HuggingFace.

## Closest analogous work

- PR #9 / branch `feat/migrate-to-mimi-rs` (commits `7c2b35c`–`cc742c2`, merged `2d09f6a`): the
  encoder migration that this goal is completing. Established `MimiEncoder` wrapper at
  `crates/stt-wasm/src/mimi_encoder.rs`, key-remapping at `mimi_remap.rs`, and the local path dep
  pattern. Tests were updated at `cc742c2`.
- `refs/xn/wasm-pocket-tts/`: contains a prior TTS+mimi decode pipeline in WASM — useful reference
  for how `MimiState` / `decode_from_latent` / `decode_from_codes` were wired up in the analogous
  TTS direction.

## Hard rules surfaced

From `stt-web/CLAUDE.md`:
- No sync GPU readback in WASM — always `into_data_async().await`.
- 2GB single ArrayBuffer limit in WASM; use `ShardedCursor`.
- 4GB WASM address space; two-phase weight loading (parse → drop reader → finalize on GPU).
- WebGPU workgroup size limit 256; apply cubecl-wgpu patch.
- All inference in a Web Worker.
- `refs/` directories are READ-ONLY. Never modify files under `refs/`.

From user `~/.claude/CLAUDE.md`:
- `accelerate`/`metal`/`cuda` features belong in consumer crate, not the library.
- `cargo clippy` before committing.
- Commit early and often; small, focused, atomic commits.
- Never open PRs on external/upstream repos without explicit user authorization.

From project MEMORY.md:
- WASM binaries go to gh-pages `/pkg`, never `.d.ts` files.
- Use `node`, not `bun`, for the dev server.
- Never add "Generated with Claude Code" to PR descriptions.
- Don't generate self-signed certs.

## Prior-run carryover

No `convergence/notes/*-next.md` files found. No prior-run walls to carry over.

## Open questions for mesh

1. **Does stt-web actually need the decode path?** STT encodes audio → tokens. The decode direction
   (tokens → audio) is for TTS. The goal asks to "verify STT produces correct transcription via the
   mimi-rs path" — which it already does for encoding. Mesh must decide if the decoder additions are
   purely a mimi-rs library improvement (additive, no stt-web changes needed beyond bumping the dep)
   or if there's a planned feature in stt-web that exercises decoding.

2. **mimi-rs version strategy:** `Cargo.toml` currently uses `path = "../../../mimi-rs"` with a
   comment "switch to git after mimi-rs PR merges." After merging, should stt-web switch to a `git`
   dep with a specific revision/tag, or stay on a local path? The user hasn't cut a git tag for
   mimi-rs yet (version is `0.1.0`, no tags visible).

3. **WASM build compatibility:** The ~119 new lines in mimi-rs add `quantizer.rs` decode methods
   using `candle-core` tensor ops. mimi-rs has no WASM target itself (it's candle-based, CPU-only).
   But stt-wasm depends on it and must build for `wasm32-unknown-unknown`. Need to confirm the new
   decode path compiles clean for WASM (no new non-WASM deps introduced).

4. **RTF baseline:** The goal requires "benchmark RTF within 5% of baseline." There is no recorded
   RTF baseline in `BENCHMARKS.md` for the current state. Mesh needs to decide whether to
   (a) record a baseline before changes, (b) treat the existing BENCHMARKS.md numbers as baseline,
   or (c) skip if the changes are provably additive (no hot-path changes).

5. **Test coverage for decode path:** mimi-rs has existing tests. The new decode path (`VectorQuantizer::decode`, `ResidualVectorQuantizer::decode_n`, `SplitResidualVectorQuantizer::decode_n`) needs round-trip test coverage (encode → decode → check reconstruction quality). Does mesh plan to write these tests in mimi-rs, or only verify via stt-web integration?

## Estimated shape

`single-phase` — the mimi-rs changes are already written (just uncommitted), the stt-web dep bump
is trivial, and verification is a standard build + test cycle with no architectural unknowns.
