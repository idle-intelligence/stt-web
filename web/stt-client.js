/**
 * Optional JS embedding API for Kyutai STT.
 *
 * Wraps the Web Worker + AudioWorklet into a clean interface
 * for embedding in other web pages.
 *
 * Usage:
 *   const stt = new SttClient({
 *       modelUrl: 'https://huggingface.co/.../stt-1b-en_fr-q4.gguf',
 *       onTranscript: (text, isFinal) => console.log(text),
 *       onStatus: (text, isReady) => console.log(text),
 *       onError: (err) => console.error(err),
 *   });
 *   await stt.init();
 *   await stt.startRecording();
 *   // ... user speaks ...
 *   stt.stopRecording();
 *   stt.destroy();
 */

export class SttClient {
    constructor(options = {}) {
        this.onTranscript = options.onTranscript || (() => {});
        this.onStatus = options.onStatus || (() => {});
        this.onError = options.onError || console.error;

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
            this.worker = new Worker('./worker.js', { type: 'module' });

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

            // Send load command to worker
            this.worker.postMessage({ type: 'load' });
        });
    }

    /** Request mic permission and start streaming audio to the engine. */
    async startRecording() {
        if (!this._ready) {
            throw new Error('Client not initialized. Call init() first.');
        }

        // Request microphone access
        this.mediaStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                sampleRate: 16000,
                echoCancellation: true,
                noiseSuppression: true,
            }
        });

        // Create AudioContext at 16kHz (Mimi codec's expected rate)
        this.audioContext = new AudioContext({ sampleRate: 16000 });

        // Register AudioWorklet processor
        await this.audioContext.audioWorklet.addModule('./audio-processor.js');

        // Create worklet node
        this.workletNode = new AudioWorkletNode(this.audioContext, 'audio-processor');

        // Forward audio chunks from worklet to worker
        this.workletNode.port.onmessage = (e) => {
            if (e.data.type === 'audio') {
                this.worker.postMessage(
                    { type: 'audio', samples: e.data.samples },
                    [e.data.samples.buffer]
                );
            }
        };

        // Connect mic → worklet (no output needed, we just process)
        const source = this.audioContext.createMediaStreamSource(this.mediaStream);
        source.connect(this.workletNode);
    }

    /** Stop recording and flush remaining text. */
    stopRecording() {
        if (!this._ready) {
            return;
        }

        // Signal worklet to stop
        if (this.workletNode) {
            this.workletNode.port.postMessage({ type: 'stop' });
        }

        // Disconnect audio graph
        if (this.workletNode) {
            this.workletNode.disconnect();
            this.workletNode = null;
        }

        // Stop media tracks
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }

        // Close audio context
        if (this.audioContext) {
            this.audioContext.close();
            this.audioContext = null;
        }

        // Send stop to worker to flush
        this.worker.postMessage({ type: 'stop' });
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
                this.onTranscript(data.text, data.final || false);
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
