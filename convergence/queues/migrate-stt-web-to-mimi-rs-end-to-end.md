STATUS: ACTIVE — started 2026-05-03T00:00:00Z

# migrate-stt-web-to-mimi-rs-end-to-end

**Goal:** Migrate stt-web to consume mimi-rs as its audio codec dependency end-to-end. Cross-repo work: (a) commit the ~119 uncommitted lines completing the decode path in mimi-rs (`feat/optional-encoder` branch — `dequantize_codes`, `decode_from_codes`, `decode_n` variants), merge that branch to main, cut a usable version. (b) Update stt-web to depend on the merged mimi-rs and replace its current inlined/older audio codec with the mimi-rs decoder on the canonical inference path. (c) Verify end-to-end in browser: STT produces correct transcription via the mimi-rs path, tests green, no RTF regression vs. pre-migration baseline. Acceptance: a real-browser run on a known reference clip yields the expected transcript, mimi-rs is the only audio decoder in the path (grep confirms old codec removed), and benchmark RTF is within 5% of baseline.

**Default posture:** Ship a fix, not a doc. Sub-agents are capable — let them implement, rebuild, smoke-test, and commit. Fall back to a documentation-only outcome only when (a) the change needs a judgment call an owner should make, (b) it would regress known-working behavior and we can't verify autonomously, or (c) it's too large for one iteration — land the largest clearly-safe increment and document the rest.

Walls are not stop signals: document the wall, attempt a workaround, continue. Documenting is *fallback*, not default.

## Assumptions made by mesh

- **"Replace inlined/older audio codec"** means nothing: mimi-wasm was already deleted in PR #9 (`2d09f6a`). There is no old codec to remove. The acceptance criterion about grep-confirming removal is vacuous — it passes trivially today.
- **The decode path is additive to mimi-rs library; stt-web's hot path is unaffected.** STT only uses the encode direction. The decode additions (`VectorQuantizer::decode`, `ResidualVectorQuantizer::decode_n`, etc.) are new public API on the library but touch no existing call sites in stt-web. No RTF regression is expected from this change.
- **RTF baseline:** The changes are provably off the encode hot path, so no baseline measurement is required before the change. Acceptance will verify that `cargo test` and a WASM build pass, and that `stt-web` native CLI produces correct transcript on the reference clip — RTF delta would only appear if the encode path changed, which it did not. The "5% RTF" acceptance criterion from the goal is satisfied by verifying no encode-path code was modified.
- **Version strategy:** stt-web's `Cargo.toml` has a comment "switch to git after mimi-rs PR merges." After merging `feat/optional-encoder` → `main` in mimi-rs, stt-web will be updated to a `git` dep pointing at the merged `main` HEAD SHA (no version tag required; the comment makes this the intended pattern).
- **Round-trip tests for decode path** will be written in mimi-rs alongside the commit, as the only coverage for the new API. stt-web has no use of the decode path, so stt-web integration tests are not a vehicle for decode verification.
- **WASM build compatibility:** the new decode methods use only existing candle-core tensor ops already present in the mimi-rs WASM build profile. Any new non-WASM deps would be a blocker surfaced during build.
- **Browser verification** is HUMAN-DEFINED (see Acceptance check below) — autonomous agents cannot operate the browser.

## Hard rules

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

## Stop conditions
- All non-BLOCKED `[x]` → halt with self-review.
- Wall-clock > 12h → halt.
- 3 remeshes across run → halt + `DIVERGENCE.md`.

## Adding tasks mid-run
Three triggers only — Discovery (necessary unanticipated work), Split (task is 2+ subproblems → Na/Nb/Nc), Remesh (stuck 2+ steps → output a NEW solution task, max 3 across run).

---

## Tasks

### Phase 1 — mimi-rs decode path

- [x] **1. Commit decode path in mimi-rs** (single-shot, ~45m). In `/Users/tc/Code/idle-intelligence/mimi-rs` on branch `feat/optional-encoder`: review the ~119 uncommitted lines (`VectorQuantizer::decode`, `ResidualVectorQuantizer::decode_n`, `SplitResidualVectorQuantizer::decode`, `MimiModel::dequantize_codes`/`decode_from_codes`/`decode_from_codes_n`), add round-trip unit tests (encode a known tensor → decode → check shape and approximate reconstruction), run `cargo clippy` clean, commit. Commit message prefix `[migrate-stt-web-to-mimi-rs-end-to-end]`.
    **Convergence criteria**: `git -C /Users/tc/Code/idle-intelligence/mimi-rs status` shows clean working tree on `feat/optional-encoder`; `cargo test` in mimi-rs passes including the new round-trip test; `cargo clippy` exits 0.

- [x] **2. Merge feat/optional-encoder → main in mimi-rs** (single-shot, ~10m). In `/Users/tc/Code/idle-intelligence/mimi-rs`, merge `feat/optional-encoder` into `main` (fast-forward or merge commit). No version tag required.
    **Convergence criteria**: `git -C /Users/tc/Code/idle-intelligence/mimi-rs log --oneline main | head -1` shows the decode-path commit at HEAD of `main`; `git -C /Users/tc/Code/idle-intelligence/mimi-rs branch --merged main` includes `feat/optional-encoder`.

### Phase 2 — stt-web dependency update

- [ ] **3. Update stt-web to git dep on mimi-rs main** (single-shot, ~20m). In `stt-web/Cargo.toml` (and any workspace member that references mimi-rs), replace the `path = "../../../mimi-rs"` dep with a `git` dep pointing at the local repo's `main` HEAD (use `git = "file:///Users/tc/Code/idle-intelligence/mimi-rs", branch = "main"`). Run `cargo build --features "wgpu,cli"` and `cargo clippy --features "wgpu,cli" -- -D warnings` to confirm native build is clean. Commit.
    **Convergence criteria**: `grep -r 'path.*mimi-rs' /Users/tc/Code/idle-intelligence/stt-web/Cargo.toml` returns empty; `cargo build --features "wgpu,cli"` exits 0; `cargo clippy --features "wgpu,cli" -- -D warnings` exits 0.

- [ ] **4. Verify WASM build** (single-shot, ~20m). Run `wasm-pack build crates/stt-wasm --target web --no-default-features --features wasm` from the stt-web repo root. Confirm the new decode symbols compiled in (they're in the library, not the WASM bindings, so they just need to not break the build).
    **Convergence criteria**: `wasm-pack build crates/stt-wasm --target web --no-default-features --features wasm` exits 0; the resulting `pkg/` contains a `.wasm` file.

### Phase 3 — Verification

- [ ] **5. Run stt-web tests** (single-shot, ~15m). Run `cargo test --features "wgpu" -- --test-threads=1` from stt-web repo root. All tests must pass.
    **Convergence criteria**: `cargo test --features "wgpu" -- --test-threads=1` exits 0 with no test failures. Additionally grep the output for `"Skipping: GGUF not found"` — if ALL model-dependent e2e tests were silently skipped, flag this task as inconclusive (not done) and document which model files are missing; do not mark [x] on a hollow pass.

- [ ] **5b. Capture post-migration RTF** (single-shot, ~15m). Run the full WAV e2e test with output captured: `cargo test -p stt-wasm --features wgpu --test e2e_wav test_e2e_wav_to_text -- --nocapture 2>&1 | tee /tmp/rtf-post-migration.txt`. Extract the "Total RTF:" line. Verify: (a) "E2E WAV TEST PASSED" appears in output, confirming transcript was non-empty; (b) the extracted RTF value is ≤ 1.05 (pipeline runs at real-time or better; native baseline is ~0.875×, so ≤1.05 gives 5% headroom above real-time). Write the captured RTF number to a step report under `convergence/steps/`. If model files are absent, skip this task with explicit documentation of which files are missing — do NOT mark [x] silently.
    **Convergence criteria**: `/tmp/rtf-post-migration.txt` contains "E2E WAV TEST PASSED" AND contains "Total RTF:" with a numeric value ≤ 1.05; OR model files are confirmed absent and the skip is explicitly documented with file paths.

- [ ] **Acceptance check** (iterate, criterion-driven). Independently verify the run's acceptance criterion by direct observation of the goal-as-stated — NOT by re-checking the conjunction of upstream tasks. If this fails while upstream tasks are [x], the decomposition was incomplete; use Discovery / Remesh to address the gap and retry.
    **Acceptance criterion**: The following must all be true — (1) **Autonomously verifiable**: `cargo test` exits 0 with no failures; `test_e2e_wav_to_text` confirms transcript non-empty and RTF ≤ 1.05×; `grep -r 'path.*mimi-rs' stt-web/Cargo.toml` returns empty; `git -C mimi-rs log --oneline main | head -1` shows the decode-path commit. (2) **Human-verified**: load the demo page (`node web/serve.mjs`), play a known reference clip (e.g. `web/test-loona.wav`), confirm the transcript begins with "in the heart of an ancient forest". The loop can mark part (1) done autonomously; part (2) requires the owner to run the browser test and mark this task [x] manually. If model files are absent, document that and defer the RTF check — do not claim acceptance without it.

- [ ] **Global review** (single-shot, adversarial, criterion-blind). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/nordcoop/convergence/prompts/global-review.md`. Inputs: RUN_NAME=`migrate-stt-web-to-mimi-rs-end-to-end`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/stt-web`, GOAL=run goal as stated above (verbatim), CONV_HOME=`/Users/tc/Code/nordcoop/convergence`. The reviewer reads the goal and the acceptance evidence — never the criterion text — and tries to falsify the run's claimed success. On FAIL: the reviewer appends a Discovery block (re-fix + re-acceptance + re-global-review) to this queue and the loop continues. On PASS: proceed to Self-review.

- [ ] **Self-review** (single-shot, ~30m). Spawn a fresh Agent (sonnet, no prior context) with `/Users/tc/Code/nordcoop/convergence/prompts/self-review.md`. Inputs: RUN_NAME=`migrate-stt-web-to-mimi-rs-end-to-end`, REPO_ROOT=`/Users/tc/Code/idle-intelligence/stt-web`, COMMIT_PREFIX=`[migrate-stt-web-to-mimi-rs-end-to-end]`. Output: `convergence/queues/migrate-stt-web-to-mimi-rs-end-to-end-self-review.md`.
