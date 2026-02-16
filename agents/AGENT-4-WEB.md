# Agent 4: Browser Demo (Web UI)

You are Agent 4 on a multi-agent team. Read the root `CLAUDE.md` first for full project context.

## Your Goal

Build the browser demo that captures mic audio, feeds it to the WASM engine, and displays streaming text. Everything lives in `web/`.

## Current State

Several files already have working implementations:
- **`web/worker.js`** — DONE. Full implementation with Cache API, shard loading, progress reporting.
- **`web/audio-processor.js`** — DONE. AudioWorklet with 1280-sample buffering, stop signal handling.
- **`web/serve.mjs`** — DONE. HTTPS dev server with CORS headers, shard discovery, range requests.

Files still needing work:
- **`web/index.html`** — Has basic skeleton, needs full implementation
- **`web/stt-client.js`** — Has API skeleton, needs full implementation

## Key Reference

Study the voxtral-rs web demo:
- `/refs/voxtral-mini-realtime-rs/web/index.html` — full page with model loading UI, recording, status
- `/refs/voxtral-mini-realtime-rs/web/voxtral-client.js` — client wrapper with mic handling

## Tasks

### `web/index.html` — Full demo page
- Standalone HTML, no bundler, monospace styling
- On page load: check WebGPU, create Worker, send `{ type: 'load' }`
- Show download progress during model loading (worker sends progress updates)
- Record/Stop button: toggle mic recording via AudioWorklet
- Forward audio chunks from AudioWorklet → Worker via postMessage
- Display streaming transcript text as it arrives
- Reset button: clear transcript, reset engine state
- Error handling: no WebGPU, mic denied, model download failure

### `web/stt-client.js` — Clean embedding API
- `SttClient` class wrapping Worker + AudioWorklet lifecycle
- `init()` — create worker, load model, resolve when ready
- `startRecording()` — getUserMedia, create AudioContext at 16kHz, register worklet, connect
- `stopRecording()` — disconnect worklet, send 'stop' to worker, flush
- `reset()` — send 'reset' to worker
- `destroy()` — terminate worker, close AudioContext, stop media tracks
- Callbacks: onTranscript(text, isFinal), onStatus(text, isReady), onError(err)

## Worker ↔ Main Thread Protocol

Already implemented in `worker.js`:
```
Main → Worker:
  { type: 'load' }                          — init WASM + WebGPU, download model
  { type: 'audio', samples: Float32Array }  — feed audio chunk
  { type: 'stop' }                          — end of speech, flush
  { type: 'reset' }                         — clear state

Worker → Main:
  { type: 'status', text, ready?, progress?: { loaded, total } }
  { type: 'transcript', text, final? }
  { type: 'error', message }
```

## AudioWorklet → Main Thread Protocol

Already implemented in `audio-processor.js`:
```
Worklet → Main: { type: 'audio', samples: Float32Array }  — 1280-sample chunks
Main → Worklet: { type: 'stop' }                          — signal to stop
```

## Requirements

- No build tools, no npm — plain ES modules
- HTTPS required for WebGPU (dev server handles this)
- Cross-Origin headers already set in serve.mjs (COOP/COEP)
- Must work in Chrome and Edge. Firefox best-effort.

## Verification

Run the dev server and open in Chrome:
```bash
./scripts/gen-cert.sh
bun web/serve.mjs
# Open https://localhost:8443
```

The page should load, show status messages, and the UI should be functional (recording/stopping). Actual transcription requires Agents 1-3 to deliver their WASM modules.

## Coordination

- **Depends on Agent 2** (mimi-wasm) and **Agent 3** (stt-wasm) for the WASM packages
- The worker.js already imports from `/pkg/stt_wasm.js` (the wasm-pack output of stt-wasm)
- You can stub/mock the WASM parts until they're ready

## Done When

- `index.html` is a complete, functional page
- `stt-client.js` is a clean reusable API
- Page loads, requests mic, shows status, has recording controls
- All JS is correct and ready for integration once WASM modules are built
