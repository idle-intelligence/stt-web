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
let totalSamples = 0;
let recordingStart = 0;

let lastMetricsSent = 0;    // performance.now() of last metrics message
const METRICS_INTERVAL_MS = 1000; // send metrics update every ~1s

// State transition logging (not per-frame — only major state changes)
function logState(msg) {
    console.log(`[worker] ${msg}`);
}

// Serialize all engine access to prevent wasm-bindgen "recursive use" errors.
// The WASM engine uses &mut self, so only one call can be active at a time.
let busy = false;
let stopped = false; // Set when 'stop' received — skip remaining queued audio
let audioChunkCount = 0; // Track chunks received per session
let tokenCount = 0;      // Track text tokens produced per session
const msgQueue = [];

async function drainQueue() {
    if (busy) return;
    busy = true;
    while (msgQueue.length > 0) {
        const { type, data } = msgQueue.shift();
        try {
            // Skip queued audio after stop was requested
            if (type === 'audio' && stopped) {
                continue;
            }
            switch (type) {
                case 'load':
                    logState('Loading model...');
                    await handleLoad(data.config || {});
                    break;
                case 'audio':
                    await handleAudio(data);
                    break;
                case 'stop':
                    logState(`Stop received (${audioChunkCount} chunks, ${tokenCount} tokens produced)`);
                    stopped = true;
                    await handleStop();
                    break;
                case 'reset':
                    logState('Reset — starting new session');
                    stopped = false;
                    audioChunkCount = 0;
                    tokenCount = 0;
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

    // 1. Import WASM module.
    self.postMessage({ type: 'status', text: 'Loading WASM module...' });
    sttWasm = await import(base + '/pkg/stt_wasm.js');
    await sttWasm.default(base + '/pkg/stt_wasm_bg.wasm');

    // 2. Initialize WebGPU device.
    self.postMessage({ type: 'status', text: 'Initializing WebGPU device...' });
    await sttWasm.initWgpuDevice();

    // 3. Create engine instance.
    engine = new sttWasm.SttEngine();

    // 4. Download model weights.
    self.postMessage({ type: 'status', text: 'Downloading model...' });

    let modelFiles;
    if (config.shardList && config.shardList.length > 0) {
        modelFiles = config.shardList;
    } else {
        const resp = await fetch(base + '/api/shards');
        const data = await resp.json();
        modelFiles = data.shards;
    }

    if (modelFiles.length === 0) {
        throw new Error(
            'No model files found. Run: python scripts/quantize.py'
        );
    }

    for (let i = 0; i < modelFiles.length; i++) {
        const name = modelFiles[i];
        const url = base + `/models/${name}`;
        const label = `Downloading model${modelFiles.length > 1 ? ` (${i + 1}/${modelFiles.length})` : ''}`;

        const buf = await cachedFetch(url, label);
        engine.appendModelShard(new Uint8Array(buf));
    }

    // 5. Load model weights into WebGPU.
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
    logState('Model loaded, ready to receive audio');
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

async function handleAudio({ samples }) {
    if (!engine || !tokenizer) return;

    const audioData = samples instanceof Float32Array
        ? samples
        : new Float32Array(samples);

    if (totalSamples === 0) {
        recordingStart = performance.now();
        lastMetricsSent = performance.now();
        logState('Receiving audio...');
    }
    totalSamples += audioData.length;
    audioChunkCount++;

    // Process chunk immediately (streaming encode)
    const tokenIds = await engine.feedAudio(audioData);
    const ids = tokenIds ? Array.from(tokenIds) : [];

    if (ids.length > 0) {
        tokenCount += ids.length;
        const text = tokenizer.decode(ids);
        // Log first few tokens and any non-empty text for debugging
        if (tokenCount <= 15 || text) {
            logState(`Tokens [${ids.join(',')}] → "${text}" (chunk #${audioChunkCount}, total tokens: ${tokenCount})`);
        }
        if (text) {
            self.postMessage({ type: 'transcript', text, final: false });
        }
    }

    const now = performance.now();
    if (now - lastMetricsSent >= METRICS_INTERVAL_MS) {
        lastMetricsSent = now;
        const m = engine.getMetrics();
        const audioDuration = totalSamples / 24000;
        const elapsed = (now - recordingStart) / 1000;
        self.postMessage({
            type: 'metrics',
            ttfb: m.ttfb_ms >= 0 ? m.ttfb_ms : null,
            framesPerSec: m.total_frames > 0 ? m.total_frames / elapsed : 0,
            avgFrameMs: m.total_frames > 0 ? m.total_ms / m.total_frames : 0,
            mimiMs: m.total_frames > 0 ? m.mimi_encode_ms / m.total_frames : 0,
            sttMs: m.total_frames > 0 ? m.stt_forward_ms / m.total_frames : 0,
            rtf: elapsed / audioDuration,
            audioDuration,
        });
    }

    const audioDur = (totalSamples / 24000).toFixed(1);
    const elapsed = ((performance.now() - recordingStart) / 1000).toFixed(1);
    self.postMessage({ type: 'status', text: `Processing... ${audioDur}s audio (${elapsed}s elapsed)` });
}

async function handleStop() {
    if (!engine || !tokenizer) return;

    if (totalSamples === 0) {
        logState('Stop: no audio received');
        self.postMessage({ type: 'transcript', text: '', final: true });
        return;
    }

    logState(`Flushing (${totalSamples} samples = ${(totalSamples/24000).toFixed(1)}s, ${tokenCount} tokens so far)...`);

    // Flush remaining delayed tokens
    const flushIds = await engine.flush();
    const fids = flushIds ? Array.from(flushIds) : [];

    if (fids.length > 0) {
        logState(`Flush produced ${fids.length} tokens: [${fids.join(',')}]`);
        const text = tokenizer.decode(fids);
        if (text) {
            self.postMessage({ type: 'transcript', text, final: false });
        } else {
            logState('Flush tokens decoded to empty string (all special?)');
        }
    } else {
        logState('Flush returned 0 tokens');
    }

    // Calculate final RTF and send Rust-side metrics
    const audioDuration = totalSamples / 24000;
    const totalTime = (performance.now() - recordingStart) / 1000;
    const rtf = {
        total: totalTime / audioDuration,
        audioDuration,
    };

    const m = engine.getMetrics();
    const avgMimiMs = m.total_frames > 0 ? m.mimi_encode_ms / m.total_frames : 0;
    const avgSttMs = m.total_frames > 0 ? m.stt_forward_ms / m.total_frames : 0;
    logState(`Done: ${audioDuration.toFixed(1)}s audio, ${audioChunkCount} chunks, ${m.total_frames} frames, ${tokenCount} tokens, RTF=${rtf.total.toFixed(3)}, avg Mimi=${avgMimiMs.toFixed(1)}ms avg STT=${avgSttMs.toFixed(1)}ms`);

    // Send final metrics
    self.postMessage({
        type: 'metrics',
        ttfb: m.ttfb_ms >= 0 ? m.ttfb_ms : null,
        framesPerSec: m.total_frames > 0 ? m.total_frames / totalTime : 0,
        avgFrameMs: m.total_frames > 0 ? m.total_ms / m.total_frames : 0,
        mimiMs: avgMimiMs,
        sttMs: avgSttMs,
        rtf: rtf.total,
        audioDuration,
        final: true,
    });

    self.postMessage({ type: 'transcript', text: '', final: true, rtf });
    totalSamples = 0;
    self.postMessage({ type: 'status', text: 'Ready', ready: true });
}

function handleReset() {
    totalSamples = 0;
    lastMetricsSent = 0;
    audioChunkCount = 0;
    tokenCount = 0;
    if (!engine) return;
    engine.reset();
}
