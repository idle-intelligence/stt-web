/**
 * worker.js — Whisper WASM Web Worker
 *
 * Runs inside a Web Worker (requires COOP/COEP headers for SharedArrayBuffer / pthreads).
 * Manages the whisper.cpp WASM module and the loaded model.
 *
 * Message protocol (postMessage):
 *
 *   Main → Worker:
 *     { type: 'load', modelUrl: string }
 *       → downloads model, initialises WASM, replies { type: 'ready' }
 *
 *     { type: 'transcribe', id: string|number, pcm: Float32Array,
 *       lang?: string, threads?: number }
 *       → runs Whisper, replies { type: 'result', id, text: string }
 *       → on error:     { type: 'error',  id, message: string }
 *
 *     { type: 'unload' }
 *       → frees the whisper context
 *
 *   Worker → Main:
 *     { type: 'ready' }
 *     { type: 'progress', value: 0–1 }
 *     { type: 'result',   id, text }
 *     { type: 'error',    id?, message }
 */

/* global WhisperModule */

// Relative path to the built JS glue — resolved relative to the worker script URL.
// Adjust if your pkg/ output is in a different location.
const WASM_JS_URL = new URL('./pkg/libwhisper.js', self.location.href).href;

// Model file name inside the Emscripten virtual FS
const MODEL_FS_PATH = '/whisper.bin';

// ── State ────────────────────────────────────────────────────────────────────

let module = null;    // initialised WhisperModule
let ctxHandle = 0;   // whisper context handle (>0 when loaded)
let modelLoaded = false;

// ── Helpers ──────────────────────────────────────────────────────────────────

function post(msg, transfer) {
    if (transfer) {
        self.postMessage(msg, transfer);
    } else {
        self.postMessage(msg);
    }
}

async function fetchModel(url) {
    const response = await fetch(url);
    if (!response.ok) {
        throw new Error(`Failed to fetch model: ${response.status} ${response.statusText}`);
    }

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;

    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        received += value.length;
        if (total > 0) {
            post({ type: 'progress', value: received / total });
        }
    }

    const buffer = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
        buffer.set(chunk, offset);
        offset += chunk.length;
    }
    return buffer;
}

// ── WASM module initialisation ───────────────────────────────────────────────

async function initModule() {
    if (module !== null) return;

    // Import the Emscripten JS glue.
    // WhisperModule is a factory function that returns a Promise<Module>.
    await import(WASM_JS_URL);

    module = await WhisperModule({
        // Silence noisy Emscripten logging
        print:    (t) => {},
        printErr: (t) => console.warn('[whisper.cpp]', t),
        // Locate the .wasm file next to the .js glue
        locateFile: (path) => new URL(`./pkg/${path}`, self.location.href).href,
    });
}

// ── Load model ───────────────────────────────────────────────────────────────

async function loadModel(modelUrl) {
    if (modelLoaded) {
        // Already loaded — nothing to do
        post({ type: 'ready' });
        return;
    }

    try {
        // 1. Initialise WASM module
        post({ type: 'progress', value: 0 });
        await initModule();

        // 2. Fetch model weights
        const modelBytes = await fetchModel(modelUrl);

        // 3. Write into Emscripten virtual FS
        // Remove stale file if present
        try { module.FS.unlink(MODEL_FS_PATH); } catch (_) {}
        module.FS.writeFile(MODEL_FS_PATH, modelBytes);

        // 4. Initialise whisper context
        ctxHandle = module.whisper_init(MODEL_FS_PATH);
        if (ctxHandle === 0) {
            throw new Error('whisper_init() returned 0 — model file may be corrupt or wrong format');
        }

        modelLoaded = true;
        post({ type: 'ready' });
    } catch (err) {
        post({ type: 'error', message: `load failed: ${err.message}` });
    }
}

// ── Transcribe ───────────────────────────────────────────────────────────────

async function transcribe(id, pcm, lang = 'en', threads = 4) {
    if (!modelLoaded || ctxHandle === 0) {
        post({ type: 'error', id, message: 'Model not loaded. Send { type: "load", modelUrl } first.' });
        return;
    }

    try {
        // whisper_transcribe is synchronous — it runs in this Worker thread so
        // the main thread stays responsive.
        const text = module.whisper_transcribe(ctxHandle, pcm, lang, threads);
        post({ type: 'result', id, text: text.trim() });
    } catch (err) {
        post({ type: 'error', id, message: `transcribe failed: ${err.message}` });
    }
}

// ── Message handler ───────────────────────────────────────────────────────────

self.onmessage = async (event) => {
    const msg = event.data;
    if (!msg || !msg.type) return;

    switch (msg.type) {
        case 'load':
            if (!msg.modelUrl) {
                post({ type: 'error', message: 'load: missing modelUrl' });
                return;
            }
            await loadModel(msg.modelUrl);
            break;

        case 'transcribe':
            if (!msg.pcm || !(msg.pcm instanceof Float32Array)) {
                post({ type: 'error', id: msg.id, message: 'transcribe: pcm must be a Float32Array' });
                return;
            }
            await transcribe(msg.id, msg.pcm, msg.lang || 'en', msg.threads || 4);
            break;

        case 'unload':
            if (ctxHandle !== 0 && module !== null) {
                module.whisper_free(ctxHandle);
                ctxHandle = 0;
                modelLoaded = false;
            }
            break;

        default:
            console.warn('[whisper worker] unknown message type:', msg.type);
    }
};
