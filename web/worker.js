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

let engine = null;
let sttWasm = null;

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
    // 1. Import WASM module.
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    sttWasm = await import('/pkg/stt_wasm.js');
    await sttWasm.default();

    // 2. Initialize WebGPU device.
    self.postMessage({ type: 'status', text: 'Initializing WebGPU device...' });
    await sttWasm.initWgpuDevice();

    // 3. Create engine instance.
    engine = new sttWasm.SttEngine();

    // 4. Discover model shards from the dev server.
    self.postMessage({ type: 'status', text: 'Discovering model shards...' });
    const shardsResp = await fetch('/api/shards');
    if (!shardsResp.ok) {
        throw new Error('Failed to fetch shard list from /api/shards');
    }
    const { shards } = await shardsResp.json();

    if (!shards || shards.length === 0) {
        throw new Error(
            'No model shards found on server. Place sharded GGUF files in models/stt-q4-shards/.'
        );
    }

    // 5. Download each shard (with caching and progress).
    for (let i = 0; i < shards.length; i++) {
        const name = shards[i];
        const label = `Downloading shard ${i + 1}/${shards.length} (${name})`;
        const url = `/models/stt-q4-shards/${name}`;

        const buf = await cachedFetch(url, label);
        engine.appendModelShard(new Uint8Array(buf));
    }

    // 6. Load model weights from shards into WebGPU.
    self.postMessage({ type: 'status', text: 'Loading model into WebGPU...' });
    await engine.loadModel();

    // 7. Signal ready.
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

async function handleAudio({ samples }) {
    if (!engine) return;

    // Ensure we have a Float32Array.
    const audioData = samples instanceof Float32Array
        ? samples
        : new Float32Array(samples);

    const text = await engine.feedAudio(audioData);
    if (text !== undefined && text !== null) {
        self.postMessage({ type: 'transcript', text, final: false });
    }
}

async function handleStop() {
    if (!engine) return;

    const text = await engine.flush();
    self.postMessage({ type: 'transcript', text: text || '', final: true });
}

function handleReset() {
    if (!engine) {
        return;
    }
    engine.reset();
}
