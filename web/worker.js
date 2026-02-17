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
let audioBuffer = [];  // Buffer audio chunks for batch encoding

// Serialize all engine access to prevent wasm-bindgen "recursive use" errors.
// The WASM engine uses &mut self, so only one call can be active at a time.
let busy = false;
const msgQueue = [];

async function drainQueue() {
    if (busy) return;
    busy = true;
    while (msgQueue.length > 0) {
        const { type, data } = msgQueue.shift();
        try {
            switch (type) {
                case 'load':
                    await handleLoad(data.config || {});
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
    }
    busy = false;
}

self.onmessage = (e) => {
    const { type, ...data } = e.data;
    msgQueue.push({ type, data });
    drainQueue();
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

async function handleLoad(config) {
    const base = (config.baseUrl || '').replace(/\/+$/, '');

    // 1. Import WASM module (cache-bust during dev).
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    const cacheBust = '?v=' + Date.now();
    sttWasm = await import(base + '/pkg/stt_wasm.js' + cacheBust);
    await sttWasm.default(base + '/pkg/stt_wasm_bg.wasm' + cacheBust);

    // 2. Initialize WebGPU device.
    self.postMessage({ type: 'status', text: 'Initializing WebGPU device...' });
    await sttWasm.initWgpuDevice();

    // 3. Create engine instance.
    engine = new sttWasm.SttEngine();

    // 4. Discover and download model shards.
    self.postMessage({ type: 'status', text: 'Discovering model shards...' });

    let shards;
    if (config.shardList && config.shardList.length > 0) {
        shards = config.shardList;
    } else {
        const shardResp = await fetch(base + '/api/shards');
        const shardData = await shardResp.json();
        shards = shardData.shards;
    }

    if (shards.length === 0) {
        throw new Error(
            'No model shards found. Run: python scripts/quantize.py'
        );
    }

    for (let i = 0; i < shards.length; i++) {
        const name = shards[i];
        const url = base + `/models/stt-1b-en_fr-q4-shards/${name}`;
        const label = `Downloading shard ${i + 1}/${shards.length}`;

        const buf = await cachedFetch(url, label);
        engine.appendModelShard(new Uint8Array(buf));
    }

    // 5. Load model weights from shards into WebGPU.
    self.postMessage({ type: 'status', text: 'Loading model into WebGPU...' });
    engine.loadModel();

    // 6. Load Mimi codec weights.
    self.postMessage({ type: 'status', text: 'Loading Mimi codec...' });
    const mimiUrl = config.mimiUrl || (base + '/models/mimi.safetensors');
    await engine.loadMimi(mimiUrl);

    // 7. Load tokenizer.
    self.postMessage({ type: 'status', text: 'Loading tokenizer...' });
    const tokenizerUrl = config.tokenizerUrl || (base + '/models/tokenizer.model');
    tokenizer = new SpmDecoder();
    await tokenizer.load(tokenizerUrl);

    // 8. Run diagnostic forward pass to compare with native Metal output.
    self.postMessage({ type: 'status', text: 'Running diagnostics...' });
    try {
        const diagResult = await engine.diagnose();
        console.log('[worker] DIAGNOSE OUTPUT:\n' + diagResult);
    } catch (diagErr) {
        console.warn('[worker] Diagnose failed:', diagErr);
    }

    // 9. Signal ready.
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

async function handleAudio({ samples }) {
    if (!engine || !tokenizer) return;

    // Buffer audio chunks — they'll be batch-encoded on stop.
    const audioData = samples instanceof Float32Array
        ? samples
        : new Float32Array(samples);

    audioBuffer.push(audioData);

    const totalSamples = audioBuffer.reduce((s, c) => s + c.length, 0);
    const duration = (totalSamples / 24000).toFixed(1);
    self.postMessage({ type: 'status', text: `Recording... ${duration}s` });
}

async function handleStop() {
    if (!engine || !tokenizer) return;

    // Concatenate all buffered audio into a single Float32Array.
    const totalSamples = audioBuffer.reduce((s, c) => s + c.length, 0);
    if (totalSamples === 0) {
        self.postMessage({ type: 'transcript', text: '', final: true });
        return;
    }

    const allAudio = new Float32Array(totalSamples);
    let offset = 0;
    for (const chunk of audioBuffer) {
        allAudio.set(chunk, offset);
        offset += chunk.length;
    }
    audioBuffer = [];

    const audioDuration = totalSamples / 24000;
    self.postMessage({ type: 'status', text: `Transcribing ${audioDuration.toFixed(1)}s of audio...` });
    console.log(`[worker] Batch encoding ${totalSamples} samples (${audioDuration.toFixed(1)}s)`);

    // Batch encode + STT in one call
    const t0 = performance.now();
    const tokenIds = await engine.feedAudio(allAudio);
    const t1 = performance.now();
    const ids = tokenIds ? Array.from(tokenIds) : [];
    console.log('[worker] feedAudio →', ids.length, 'tokens');

    // Flush remaining delayed tokens
    const flushIds = await engine.flush();
    const t2 = performance.now();
    const fids = flushIds ? Array.from(flushIds) : [];
    console.log('[worker] flush →', fids.length, 'tokens');

    const feedTime = (t1 - t0) / 1000;
    const flushTime = (t2 - t1) / 1000;
    const totalTime = (t2 - t0) / 1000;
    const rtf = {
        total: totalTime / audioDuration,
        feed: feedTime / audioDuration,
        flush: flushTime / audioDuration,
        audioDuration,
    };
    console.log(`[worker] RTF: total=${rtf.total.toFixed(3)} feed=${rtf.feed.toFixed(3)} flush=${rtf.flush.toFixed(3)}`);

    const allIds = ids.concat(fids);
    const text = allIds.length > 0 ? tokenizer.decode(allIds) : '';

    self.postMessage({ type: 'transcript', text, final: true, rtf });
    self.postMessage({ type: 'status', text: 'Ready' });
}

function handleReset() {
    audioBuffer = [];
    if (!engine) {
        return;
    }
    engine.reset();
}
