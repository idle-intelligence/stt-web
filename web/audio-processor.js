/**
 * AudioWorklet processor: captures mic input and sends 24kHz mono PCM chunks.
 *
 * Registered as 'audio-processor'. The main thread creates an AudioWorkletNode
 * pointing to this processor, then forwards PCM chunks to the Web Worker.
 *
 * The AudioContext is created with { sampleRate: 24000 } to match the Mimi
 * codec's native sample rate. No resampling needed.
 *
 * Each process() call receives 128 samples at the context sample rate.
 * We buffer them into larger chunks (~1920 samples = 80ms at 24kHz) to
 * reduce postMessage overhead while keeping latency low.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Buffer samples to reduce postMessage overhead.
        // At 24kHz, 1920 samples = 80ms = one Mimi frame.
        this._buffer = new Float32Array(1920);
        this._writePos = 0;
        this._active = true;

        // Listen for stop signal from main thread.
        this.port.onmessage = (e) => {
            if (e.data && e.data.type === 'stop') {
                this._active = false;
            }
        };
    }

    process(inputs, outputs, parameters) {
        if (!this._active) {
            // Flush any remaining buffered samples before stopping.
            if (this._writePos > 0) {
                const remaining = this._buffer.slice(0, this._writePos);
                this.port.postMessage({ type: 'audio', samples: remaining }, [remaining.buffer]);
                this._writePos = 0;
            }
            return false; // Remove processor from the graph.
        }

        const input = inputs[0];
        if (!input || input.length === 0) {
            return true;
        }

        const channel = input[0]; // mono (first channel)
        if (!channel || channel.length === 0) {
            return true;
        }

        // Copy samples into the accumulation buffer.
        for (let i = 0; i < channel.length; i++) {
            this._buffer[this._writePos++] = channel[i];

            if (this._writePos >= this._buffer.length) {
                // Buffer full — send a copy to the main thread.
                const chunk = new Float32Array(this._buffer);
                this.port.postMessage({ type: 'audio', samples: chunk }, [chunk.buffer]);
                this._writePos = 0;
            }
        }

        return true; // Keep processor alive.
    }
}

registerProcessor('audio-processor', AudioProcessor);
