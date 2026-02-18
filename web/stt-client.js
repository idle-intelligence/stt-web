/**
 * Optional JS embedding API for STT.
 *
 * Wraps the Web Worker + AudioWorklet into a clean interface
 * for embedding in other web pages.
 *
 * Usage:
 *   const stt = new SttClient({
 *       onTranscript: (text, isFinal) => console.log(text),
 *       onStatus: (text, isReady) => console.log(text),
 *       onError: (err) => console.error(err),
 *   });
 *   await stt.init();
 *   await stt.startRecording();
 *   // ... user speaks ...
 *   stt.stopRecording();
 *   stt.destroy();
 *
 * Model weights are fetched from HuggingFace by default.
 * Override with custom URLs if self-hosting:
 *   const stt = new SttClient({
 *       modelUrl: 'https://cdn.example.com/stt-1b-en_fr-q4.gguf',
 *       mimiUrl: 'https://cdn.example.com/mimi.safetensors',
 *       tokenizerUrl: 'https://cdn.example.com/tokenizer.model',
 *       onTranscript: (text, isFinal) => console.log(text),
 *   });
 */

export class SttClient {
    constructor(options = {}) {
        this.onTranscript = options.onTranscript || (() => {});
        this.onStatus = options.onStatus || (() => {});
        this.onError = options.onError || console.error;
        this.onMetrics = options.onMetrics || (() => {});

        // URL configuration for embedding
        this.baseUrl = (options.baseUrl || '').replace(/\/+$/, '');
        this.workerUrl = options.workerUrl || (this.baseUrl + '/worker.js');
        this.audioProcessorUrl = options.audioProcessorUrl || (this.baseUrl + '/audio-processor.js');

        // Optional overrides passed to the worker (defaults fetch from HuggingFace)
        this.modelUrl = options.modelUrl || null;
        this.shardList = options.shardList || null;
        this.mimiUrl = options.mimiUrl || null;
        this.tokenizerUrl = options.tokenizerUrl || null;

        this.worker = null;
        this.audioContext = null;
        this.workletNode = null;
        this.mediaStream = null;

        this._pendingResolve = null;
        this._pendingReject = null;
        this._ready = false;
    }

    /** Load model — returns when ready to record. */
    async init() {
        return new Promise((resolve, reject) => {
            this.worker = new Worker(this.workerUrl, { type: 'module' });

            this.worker.onmessage = (e) => this._handleWorkerMessage(e);
            this.worker.onerror = (err) => {
                const errMsg = err.message || String(err);
                this.onError(new Error(`Worker error: ${errMsg}`));
                if (this._pendingReject) {
                    this._pendingReject(new Error(errMsg));
                    this._pendingReject = null;
                    this._pendingResolve = null;
                }
            };

            this._pendingResolve = () => {
                this._ready = true;
                resolve();
            };
            this._pendingReject = reject;

            // Send load command to worker with URL config
            const config = { baseUrl: this.baseUrl };
            if (this.modelUrl) config.modelUrl = this.modelUrl;
            if (this.shardList) config.shardList = this.shardList;
            if (this.mimiUrl) config.mimiUrl = this.mimiUrl;
            if (this.tokenizerUrl) config.tokenizerUrl = this.tokenizerUrl;
            this.worker.postMessage({ type: 'load', config });
        });
    }

    /** Request mic permission and start streaming audio to the engine. */
    async startRecording() {
        if (!this._ready) {
            throw new Error('Client not initialized. Call init() first.');
        }

        // Reset engine state for a clean recording session
        this.worker.postMessage({ type: 'reset' });

        // Request microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 24000,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true,
            }
        });

        // Create AudioContext at 24kHz (Mimi codec's native rate)
        this.audioContext = new AudioContext({ sampleRate: 24000 });

        // Register AudioWorklet processor
        await this.audioContext.audioWorklet.addModule(this.audioProcessorUrl);

        // Create worklet node
        this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');

        // Create a direct AudioWorklet → Worker channel.
        // This bypasses the main thread, preventing Chrome from throttling
        // audio delivery when the tab is in the background.
        const channel = new MessageChannel();
        this.workletNode.port.postMessage(
            { type: 'port', port: channel.port1 },
            [channel.port1]
        );
        this.worker.postMessage(
            { type: 'audio-port', port: channel.port2 },
            [channel.port2]
        );

        // Connect mic → worklet → destination.
        // Web Audio API is pull-based: the destination pulls audio from its inputs
        // recursively. Without connecting the worklet to the destination, Chrome
        // will not call process() on the worklet and no audio is captured.
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        source.connect(this.workletNode);
        this.workletNode.connect(this.audioContext.destination);

        // NOTE: Background tab throttling (macOS App Nap / Chrome CPU deprioritization)
        // causes ~2-4x RTF degradation when the tab is on another desktop.
        // Audio keep-alive tricks interfere with the mic. A SharedWorker architecture
        // would be needed to fully solve this.
    }

    /** Stop recording and flush remaining text. */
    stopRecording() {
        if (!this._ready) {
            return;
        }

        // Send stop to worker immediately. The worker's queue drainer will skip
        // any remaining queued audio chunks once it sees 'stop'.
        this.worker.postMessage({ type: 'stop' });

        // Tell the worklet to stop processing (best-effort — it may already be
        // disconnected by the time it processes this).
        if (this.workletNode) {
            try {
                this.workletNode.port.postMessage({ type: 'stop' });
            } catch (_) {
                // Worklet may already be disconnected
            }
        }

        // Disconnect audio graph (stop receiving mic input)
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        // Stop media tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Close audio context after a brief delay to let the worklet flush
        const ctx = this.audioContext;
        this.audioContext = null;
        if (ctx) {
            setTimeout(() => ctx.close(), 200);
        }
    }

    /** Reset state for a new session (without reloading model). */
    reset() {
        if (!this._ready) {
            return;
        }

        this.worker.postMessage({ type: 'reset' });
    }

    /** Clean up worker, audio context, and mic stream. */
    destroy() {
        // Stop recording if active
        this.stopRecording();

        // Terminate worker
        if (this.worker) {
            this.worker.terminate();
            this.worker = null;
        }

        this._ready = false;
        this._pendingResolve = null;
        this._pendingReject = null;
    }

    /** Check if ready to record. */
    isReady() {
        return this._ready;
    }

    // Private methods

    _handleWorkerMessage(e) {
        const { type, ...data } = e.data;

        switch (type) {
            case 'status':
                this.onStatus(data.text, data.ready || false, data.progress);

                // Resolve init() when ready
                if (data.ready && this._pendingResolve) {
                    this._pendingResolve();
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
                break;

            case 'transcript':
                this.onTranscript(data.text, data.final || false, data.rtf || null);
                break;

            case 'metrics':
                this.onMetrics(data);
                break;

            case 'error':
                const err = new Error(data.message);
                this.onError(err);

                if (this._pendingReject) {
                    this._pendingReject(err);
                    this._pendingResolve = null;
                    this._pendingReject = null;
                }
                break;
        }
    }
}
