# Agent 5: Integration & Testing

You are Agent 5 on a multi-agent team. Read the root `CLAUDE.md` first for full project context.

**You start after Agents 1-4 have delivered their pieces.**

## Your Goal

Wire everything together, run end-to-end tests, benchmark, and write the README.

## Tasks

### 1. Integration
- Verify Mimi WASM (Agent 2) and STT WASM (Agent 3) work together in the web page (Agent 4)
- Debug WASM module initialization order, WebGPU device sharing, memory issues
- Ensure streaming pipeline is correct: audio timing, frame boundaries, text synchronization

### 2. End-to-End Testing
- `tests/e2e_browser.spec.ts` (Playwright):
  - Load page in headless Chrome with WebGPU flags
  - Inject a .wav file as if it were mic input
  - Assert transcript matches expected output
  - Test: start/stop/restart recording cycle
  - Test: long audio (>30 seconds)

### 3. WER Evaluation
- Run Agent 1's `scripts/eval.py` comparing:
  - PyTorch f32 → baseline WER
  - Q4 GGUF native (Burn+wgpu) → Q4 WER
  - Q4 GGUF browser (WASM+WebGPU) → browser WER (should match native)

### 4. Benchmarking
- Model load time (cold + warm)
- Tokens per second during inference
- Real-time factor (must be ≥ 1.0)
- Peak GPU/WASM memory usage

### 5. Documentation
- `README.md` — what it is, live demo link, quick start, architecture diagram, build instructions, performance numbers

## Verification

```bash
cargo test --features "wgpu"
bunx playwright test tests/e2e_browser.spec.ts
```

## Done When

- All tests pass
- WER within 2% of PyTorch f32
- Real-time factor ≥ 1x
- README.md is complete
- Someone can clone, build, serve, and get working browser STT
