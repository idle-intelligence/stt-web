/**
 * AudioWorklet processor: captures mic input, resamples to 24kHz, and sends
 * mono PCM chunks to the Web Worker.
 *
 * Registered as 'audio-processor'. The main thread creates an AudioWorkletNode
 * pointing to this processor, then forwards PCM chunks to the Web Worker.
 *
 * The AudioContext uses the device's default sample rate (commonly 48kHz or
 * 44.1kHz). This processor resamples to 24kHz using linear interpolation
 * before sending to the worker. When the native rate is already 24kHz,
 * the resampler degrades to a direct copy (step = 1.0).
 *
 * Output is buffered into 1920-sample chunks (~80ms at 24kHz) matching the
 * Mimi codec's frame size. Since Mimi needs exactly 1920 samples to produce
 * one frame of tokens, smaller buffers just cause extra postMessage +
 * WASM call overhead without reducing latency.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this._targetRate = 24000;
        // Buffer at target rate: 1920 samples = 80ms at 24kHz (Mimi frame size)
        this._buffer = new Float32Array(1920);
        this._writePos = 0;
        this._active = true;
        // Resampling state
        this._resamplePos = 0; // fractional position in source stream
        // Direct port to the Worker, bypassing the main thread.
        // When set, audio is sent through this port instead of this.port,
        // which avoids Chrome's main-thread throttling when the tab loses focus.
        this._directPort = null;

        // Listen for control signals from main thread.
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'stop') {
                this._active = false;
            } else if (e.data && e.data.type === 'port') {
                // Receive a MessagePort for direct Worker communication.
                this._directPort = e.data.port;
            }
        };
    }

    process(inputs, outputs, parameters) {
        // Send via direct port (to Worker) if available, otherwise via default
        // port (to main thread). The direct port bypasses Chrome's main-thread
        // throttling when the tab is in the background.
        const port = this._directPort || this.port;

        if (!this._active) {
            // Flush any remaining buffered samples before stopping.
            if (this._writePos > 0) {
                const remaining = this._buffer.slice(0, this._writePos);
                port.postMessage({ type: 'audio', samples: remaining }, [remaining.buffer]);
                this._writePos = 0;
            }
            // Signal that the worklet is done sending audio.
            port.postMessage({ type: 'done' });
            return false; // Remove processor from the graph.
        }

        const input = inputs[0];
        if (!input || input.length === 0) return true;
        const channel = input[0]; // mono (first channel)
        if (!channel || channel.length === 0) return true;

        // Resample from AudioContext rate to 24kHz using linear interpolation.
        // `sampleRate` is the AudioWorkletGlobalScope property (= AudioContext rate).
        // step = how many source samples per output sample.
        const step = sampleRate / this._targetRate;
        let srcPos = this._resamplePos;

        while (srcPos < channel.length) {
            const idx = Math.floor(srcPos);
            const frac = srcPos - idx;
            const s0 = channel[idx];
            const s1 = idx + 1 < channel.length ? channel[idx + 1] : s0;
            this._buffer[this._writePos++] = s0 + frac * (s1 - s0);

            if (this._writePos >= this._buffer.length) {
                const chunk = new Float32Array(this._buffer);
                port.postMessage({ type: 'audio', samples: chunk }, [chunk.buffer]);
                this._writePos = 0;
            }

            srcPos += step;
        }
        // Save fractional remainder for next process() call
        this._resamplePos = srcPos - channel.length;

        return true; // Keep processor alive.
    }
}

registerProcessor('audio-processor', AudioProcessor);
