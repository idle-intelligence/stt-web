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
 * We buffer into 1920-sample chunks (~80ms at 24kHz) matching the Mimi
 * codec's frame size. Since Mimi needs exactly 1920 samples to produce
 * one frame of tokens, smaller buffers just cause extra postMessage +
 * WASM call overhead without reducing latency.
 */

class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        // Buffer samples to match Mimi's frame size (1920 samples = 80ms).
        // Mimi cannot produce output until it has a full frame, so buffering
        // less than 1920 just adds postMessage overhead without reducing
        // latency. This reduces postMessage calls by 4x vs the old 480.
        this._buffer = new Float32Array(1920);
        this._writePos = 0;
        this._active = true;
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
                // Buffer full — send to worker (or main thread as fallback).
                const chunk = new Float32Array(this._buffer);
                port.postMessage({ type: 'audio', samples: chunk }, [chunk.buffer]);
                this._writePos = 0;
            }
        }

        return true; // Keep processor alive.
    }
}

registerProcessor('audio-processor', AudioProcessor);
