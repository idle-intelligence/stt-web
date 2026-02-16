/**
 * Web Worker: loads WASM modules, orchestrates the STT pipeline.
 *
 * All inference runs here -- never on the main thread.
 *
 * Protocol:
 *   Main -> Worker:
 *     { type: 'load' }                          -- initialize WASM + WebGPU, download model
 *     { type: 'audio', samples: Float32Array }   -- feed audio chunk
 *     { type: 'stop' }                           -- end of speech, trigger flush
 *     { type: 'reset' }                          -- clear state for new session
 *
 *   Worker -> Main:
 *     { type: 'status', text: string, ready?: boolean, progress?: { loaded, total } }
 *     { type: 'transcript', text: string, final?: boolean }
 *     { type: 'error', message: string }
 */

import { SpmDecoder } from './tokenizer.js';

let engine = null;
let sttWasm = null;
let tokenizer = null;

self.onmessage = async (e) => {
    const { type, ...data } = e.data;

    try {
        switch (type) {
            case 'load':
                await handleLoad();
                break;
            case 'audio':
                await handleAudio(data);
                break;
            case 'stop':
                await handleStop();
                break;
            case 'reset':
                handleReset();
                break;
            default:
                console.warn('[worker] Unknown message type:', type);
        }
    } catch (err) {
        self.postMessage({ type: 'error', message: err.message || String(err) });
    }
};

// ---------------------------------------------------------------------------
// Cache helpers
// ---------------------------------------------------------------------------

const CACHE_NAME = 'kyutai-stt-model-v1';

/**
 * Fetch a URL with caching via the Cache API.
 * Returns the Response body as an ArrayBuffer.
 * Reports download progress via postMessage.
 */
async function cachedFetch(url, label) {
    const cache = await caches.open(CACHE_NAME);

    // Check cache first.
    const cached = await cache.match(url);
    if (cached) {
        self.postMessage({ type: 'status', text: `${label} (cached)` });
        return await cached.arrayBuffer();
    }

    // Not cached -- download with progress tracking.
    const resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`Failed to fetch ${url}: ${resp.status} ${resp.statusText}`);
    }

    const contentLength = parseInt(resp.headers.get('Content-Length') || '0', 10);
    const reader = resp.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        chunks.push(value);
        loaded += value.byteLength;

        self.postMessage({
            type: 'status',
            text: label,
            progress: { loaded, total: contentLength },
        });
    }

    // Reassemble into a single ArrayBuffer.
    const buf = new Uint8Array(loaded);
    let offset = 0;
    for (const chunk of chunks) {
        buf.set(chunk, offset);
        offset += chunk.byteLength;
    }

    // Store in cache for next time.
    // We must reconstruct a Response because the original was consumed.
    try {
        const cacheResp = new Response(buf.buffer, {
            headers: { 'Content-Type': 'application/octet-stream' },
        });
        await cache.put(url, cacheResp);
    } catch (cacheErr) {
        // Cache API may fail in some contexts (e.g., storage quota).
        // Non-fatal -- just skip caching.
        console.warn('[worker] Could not cache:', cacheErr);
    }

    return buf.buffer;
}

// ---------------------------------------------------------------------------
// Message handlers
// ---------------------------------------------------------------------------

async function handleLoad() {
    // Configuration: HuggingFace model URLs
    const HF_BASE = 'https://huggingface.co/kyutai/stt-1b-en_fr-q4/resolve/main';
    const NUM_SHARDS = 4; // Adjust based on actual shard count
    const MIMI_WEIGHTS_URL = 'https://huggingface.co/kyutai/mimi/resolve/main/model.safetensors';
    const TOKENIZER_URL = `${HF_BASE}/tokenizer_spm_32k_3.model`;

    // 1. Import WASM module.
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    sttWasm = await import('/pkg/stt_wasm.js');
    await sttWasm.default();

    // 2. Initialize WebGPU device.
    self.postMessage({ type: 'status', text: 'Initializing WebGPU device...' });
    await sttWasm.initWgpuDevice();

    // 3. Create engine instance.
    engine = new sttWasm.SttEngine();

    // 4. Download model shards from HuggingFace.
    for (let i = 0; i < NUM_SHARDS; i++) {
        const name = `stt-1b-en_fr-q4-shard-${i}.gguf`;
        const url = `${HF_BASE}/${name}`;
        const label = `Downloading model shard ${i + 1}/${NUM_SHARDS}`;

        const buf = await cachedFetch(url, label);
        engine.appendModelShard(new Uint8Array(buf));
    }

    // 5. Load model weights from shards into WebGPU.
    self.postMessage({ type: 'status', text: 'Loading model into WebGPU...' });
    await engine.loadModel();

    // 6. Load Mimi codec weights.
    self.postMessage({ type: 'status', text: 'Loading Mimi codec...' });
    await engine.loadMimi(MIMI_WEIGHTS_URL);

    // 7. Load tokenizer.
    self.postMessage({ type: 'status', text: 'Loading tokenizer...' });
    tokenizer = new SpmDecoder();
    await tokenizer.load(TOKENIZER_URL);

    // 8. Signal ready.
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

async function handleAudio({ samples }) {
    if (!engine || !tokenizer) return;

    // Ensure we have a Float32Array.
    const audioData = samples instanceof Float32Array
        ? samples
        : new Float32Array(samples);

    // feedAudio returns Vec<u32> of text token IDs
    const tokenIds = await engine.feedAudio(audioData);

    if (tokenIds && tokenIds.length > 0) {
        const text = tokenizer.decode(Array.from(tokenIds));
        if (text) {
            self.postMessage({ type: 'transcript', text, final: false });
        }
    }
}

async function handleStop() {
    if (!engine || !tokenizer) return;

    // flush() returns Vec<u32> of remaining text token IDs
    const tokenIds = await engine.flush();
    const text = tokenIds && tokenIds.length > 0
        ? tokenizer.decode(Array.from(tokenIds))
        : '';

    self.postMessage({ type: 'transcript', text, final: true });
}

function handleReset() {
    if (!engine) {
        return;
    }
    engine.reset();
}
