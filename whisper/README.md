# Whisper WASM — CPU Fallback STT

A lightweight Whisper-based speech-to-text WASM package for browsers without WebGPU support.
Built from [whisper.cpp](https://github.com/ggerganov/whisper.cpp) with Emscripten + SIMD.

## Architecture

```
whisper/
├── build.sh            # Emscripten build script
├── src/
│   └── whisper_wasm.cpp # Custom C++ Emscripten bindings
├── worker.js           # Web Worker: loads WASM + model, runs transcription
├── whisper-client.js   # JS client API: transcribe(Float32Array) → Promise<string>
└── pkg/                # Built output (after running build.sh)
    ├── libwhisper.js   # Emscripten JS glue + embedded WASM loader
    └── libwhisper.wasm # WASM binary
```

## Build

### Prerequisites

- Emscripten 3.1.2+ (`emcc` in PATH)
- CMake 3.16+

```bash
# Option 1: Homebrew (macOS)
brew install emscripten

# Option 2: emsdk
git clone https://github.com/emscripten-core/emsdk
cd emsdk && ./emsdk install latest && ./emsdk activate latest
source ./emsdk_env.sh
```

### Build

```bash
chmod +x whisper/build.sh
./whisper/build.sh
```

Output goes to `whisper/pkg/`.

## Usage

```js
import { WhisperClient, WHISPER_TINY_Q5_1_URL } from './whisper/whisper-client.js';

const client = new WhisperClient('./whisper/worker.js');

// Load model (downloaded from HuggingFace, cached in browser)
await client.load(WHISPER_TINY_Q5_1_URL, (progress) => {
    console.log(`Loading: ${Math.round(progress * 100)}%`);
});

// Transcribe 16kHz mono Float32Array
const text = await client.transcribe(pcmFloat32Array, { lang: 'en', threads: 4 });
console.log(text);

// Cleanup
client.destroy();
```

## API

### `WhisperClient(workerUrl)`
Creates the worker. `workerUrl` must point to `worker.js`.

### `client.load(modelUrl, onProgress?): Promise<void>`
Downloads the ggml model and initialises Whisper. The model is ~31MB (tiny Q5_1).

### `client.transcribe(pcm: Float32Array, opts?): Promise<string>`
Transcribes 16 kHz mono PCM audio. Options: `lang` (BCP-47, default `'en'`), `threads` (default `4`).

### `client.unload(): void`
Frees the whisper context (keeps worker alive for reload).

### `client.destroy(): void`
Terminates the worker.

## Model

Default: **Whisper Tiny Q5_1** (~31 MB) from `ggerganov/whisper.cpp` on HuggingFace.
Fetched at runtime, cached in the browser's Cache API via Emscripten FS.

## Requirements

- SharedArrayBuffer (requires `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: credentialless` headers)
- WASM SIMD 128-bit (`-msimd128`) — supported in Chrome 91+, Firefox 89+, Safari 16.4+
- The existing `web/serve.mjs` already sets the required COOP/COEP headers
