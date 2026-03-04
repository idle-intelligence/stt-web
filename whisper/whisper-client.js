/**
 * whisper-client.js
 *
 * Client-side API for the Whisper WASM web worker.
 *
 * Usage:
 *
 *   import { WhisperClient } from './whisper/whisper-client.js';
 *
 *   const client = new WhisperClient('./whisper/worker.js');
 *   await client.load('https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin');
 *
 *   // pcm: Float32Array, 16 kHz mono
 *   const text = await client.transcribe(pcmFloat32Array);
 *   console.log(text);
 *
 * API:
 *   new WhisperClient(workerUrl: string)
 *     Creates the worker. workerUrl must point to worker.js.
 *
 *   client.load(modelUrl: string, onProgress?: (value: number) => void): Promise<void>
 *     Downloads the ggml model and initialises Whisper.
 *     modelUrl: URL to a ggml-format whisper model (e.g. ggml-tiny-q5_1.bin).
 *     Resolves when ready. Rejects on error.
 *
 *   client.transcribe(pcm: Float32Array, opts?: { lang?: string, threads?: number }): Promise<string>
 *     Transcribes 16 kHz mono PCM audio.
 *     Returns the transcript string. Rejects on error.
 *
 *   client.unload(): void
 *     Frees the whisper context in the worker (model stays cached in IndexedDB if applicable).
 *
 *   client.destroy(): void
 *     Terminates the worker entirely.
 */

export class WhisperClient {
    /**
     * @param {string} workerUrl  URL to whisper/worker.js
     */
    constructor(workerUrl) {
        this._worker = new Worker(workerUrl, { type: 'module' });
        this._pending = new Map(); // id → { resolve, reject }
        this._nextId = 1;
        this._loadResolve = null;
        this._loadReject  = null;
        this._onProgress  = null;

        this._worker.onmessage = (event) => this._handleMessage(event.data);
        this._worker.onerror   = (event) => {
            const msg = event.message || 'Worker error';
            // Reject any pending load
            if (this._loadReject) {
                const reject = this._loadReject;
                this._loadResolve = null;
                this._loadReject  = null;
                reject(new Error(msg));
            }
            // Reject all pending transcriptions
            for (const [, { reject: rej }] of this._pending) {
                rej(new Error(msg));
            }
            this._pending.clear();
        };
    }

    // ── Public API ──────────────────────────────────────────────────────────

    /**
     * Load and initialise the Whisper model.
     * @param {string}   modelUrl    URL to ggml model file
     * @param {function} onProgress  optional progress callback (0–1)
     * @returns {Promise<void>}
     */
    load(modelUrl, onProgress) {
        this._onProgress = onProgress || null;
        return new Promise((resolve, reject) => {
            this._loadResolve = resolve;
            this._loadReject  = reject;
            this._worker.postMessage({ type: 'load', modelUrl });
        });
    }

    /**
     * Transcribe 16 kHz mono PCM audio.
     * @param {Float32Array} pcm     Audio samples at 16 kHz mono
     * @param {object}       opts
     * @param {string}       opts.lang     BCP-47 language code (default 'en')
     * @param {number}       opts.threads  CPU threads (default 4)
     * @returns {Promise<string>}
     */
    transcribe(pcm, opts = {}) {
        if (!(pcm instanceof Float32Array)) {
            return Promise.reject(new TypeError('pcm must be a Float32Array'));
        }
        const id = this._nextId++;
        return new Promise((resolve, reject) => {
            this._pending.set(id, { resolve, reject });
            // Transfer ownership of the buffer for zero-copy
            this._worker.postMessage(
                { type: 'transcribe', id, pcm, lang: opts.lang || 'en', threads: opts.threads || 4 },
                [pcm.buffer]
            );
        });
    }

    /**
     * Free the whisper context in the worker (keeps the Worker alive).
     */
    unload() {
        this._worker.postMessage({ type: 'unload' });
    }

    /**
     * Terminate the worker entirely.
     */
    destroy() {
        this._worker.terminate();
    }

    // ── Internal ────────────────────────────────────────────────────────────

    _handleMessage(msg) {
        if (!msg || !msg.type) return;

        switch (msg.type) {
            case 'ready':
                if (this._loadResolve) {
                    const resolve = this._loadResolve;
                    this._loadResolve = null;
                    this._loadReject  = null;
                    resolve();
                }
                break;

            case 'progress':
                if (this._onProgress) {
                    this._onProgress(msg.value);
                }
                break;

            case 'result': {
                const entry = this._pending.get(msg.id);
                if (entry) {
                    this._pending.delete(msg.id);
                    entry.resolve(msg.text);
                }
                break;
            }

            case 'error': {
                const errorMsg = msg.message || 'Unknown error';
                if (msg.id !== undefined) {
                    const entry = this._pending.get(msg.id);
                    if (entry) {
                        this._pending.delete(msg.id);
                        entry.reject(new Error(errorMsg));
                    }
                } else if (this._loadReject) {
                    const reject = this._loadReject;
                    this._loadResolve = null;
                    this._loadReject  = null;
                    reject(new Error(errorMsg));
                } else {
                    console.error('[WhisperClient] worker error:', errorMsg);
                }
                break;
            }

            default:
                // ignore unknown message types
        }
    }
}

// ── Convenience function ────────────────────────────────────────────────────

/**
 * Recommended Whisper Tiny Q5_1 model URL (~31 MB).
 * Hosted on HuggingFace, fetched at runtime.
 */
export const WHISPER_TINY_Q5_1_URL =
    'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny-q5_1.bin';
