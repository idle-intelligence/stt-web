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
        this.modelUrl = options.modelUrl;
        this.onTranscript = options.onTranscript || (() => {});
        this.onStatus = options.onStatus || (() => {});
        this.onError = options.onError || console.error;
        this.worker = null;
        this.audioContext = null;
        this.workletNode = null;
        this.stream = null;
    }

    /** Load model — returns when ready to record. */
    async init() {
        // TODO: Create worker, send 'load' message, wait for ready
    }

    /** Request mic permission and start streaming audio to the engine. */
    async startRecording() {
        // TODO: getUserMedia, create AudioContext at 16kHz,
        // register AudioWorklet, connect mic → worklet → worker
    }

    /** Stop recording and flush remaining text. */
    stopRecording() {
        // TODO: Disconnect worklet, send 'stop' to worker
    }

    /** Reset state for a new session (without reloading model). */
    reset() {
        // TODO: Send 'reset' to worker
    }

    /** Clean up worker, audio context, and mic stream. */
    destroy() {
        // TODO: Terminate worker, close AudioContext, stop media tracks
    }
}
