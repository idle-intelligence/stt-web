/* tslint:disable */
/* eslint-disable */

/**
 * Mimi audio codec instance.
 *
 * Encodes raw PCM audio into 32-codebook token frames at 12.5Hz.
 * Runs entirely on CPU — no GPU required.
 */
export class MimiCodec {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Feed a chunk of f32 PCM audio (24kHz mono for Mimi).
     *
     * Returns token IDs as a flat array:
     * `[frame0_tok0, frame0_tok1, ..., frame0_tok31, frame1_tok0, ...]`
     *
     * May return empty if not enough audio has accumulated for a full frame.
     */
    feedAudio(samples: Float32Array): Uint32Array;
    /**
     * Create a new codec instance. Downloads/loads Mimi weights from the given URL.
     */
    constructor(weights_url: string);
    /**
     * Reset internal state (e.g., when user stops and restarts recording).
     */
    reset(): void;
}

/**
 * Browser-facing STT engine combining Mimi codec + STT transformer.
 *
 * This is the single entry point that the Web Worker calls.
 */
export class SttEngine {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Append a model weight shard (for multi-shard GGUF loading).
     *
     * Call this for each shard before calling `loadModel`.
     */
    appendModelShard(shard: Uint8Array): void;
    /**
     * Feed PCM audio samples (f32, 24kHz mono for Mimi).
     *
     * Returns decoded transcript text if any new tokens were produced.
     * Audio goes through: Mimi codec → STT transformer → text tokens → detokenize.
     * Per-call timing is stored in metrics (retrieve via `getMetrics()`).
     *
     * **Pipelining:** The last frame's GPU work (forward + argmax) is left
     * pending at the end of each call. On the *next* call, Mimi encode (CPU)
     * runs first while the GPU finishes, then we resolve the pending readback.
     * This overlaps ~20ms of CPU work with the GPU readback latency.
     */
    feedAudio(samples: Float32Array): Promise<string>;
    /**
     * Flush remaining text after end of speech.
     */
    flush(): Promise<string>;
    /**
     * Get timing metrics from the session.
     *
     * Returns a JS object: `{ mimi_encode_ms, stt_forward_ms, total_ms,
     *   total_frames, total_tokens, ttfb_ms }`
     */
    getMetrics(): any;
    /**
     * Check if the model is loaded and ready.
     */
    isReady(): boolean;
    /**
     * Initialize the Mimi audio codec from pre-fetched weight bytes.
     */
    loadMimi(data: Uint8Array): void;
    /**
     * Load the STT model from previously appended shards.
     *
     * Uses two-phase loading: parse GGUF → drop reader → finalize tensors on GPU.
     */
    loadModel(): void;
    /**
     * Load the SentencePiece tokenizer from pre-fetched `.model` bytes.
     */
    loadTokenizer(data: Uint8Array): void;
    /**
     * Create a new SttEngine instance.
     *
     * Call `initWgpuDevice()` first, then create this, then load weights.
     */
    constructor();
    /**
     * Reset all state for a new recording session.
     *
     * Uses `reset_keep_buffers` for the STT stream to preserve GPU KV cache
     * allocations from warmup. A full `reset()` drops the GPU tensors, forcing
     * expensive re-allocation on the first frame of the next recording.
     */
    reset(): void;
    /**
     * Run warmup passes to pre-compile WebGPU shader pipelines.
     *
     * Feeds 10 dummy frames through the STT transformer with varied audio
     * tokens, exercising all shader variants and the delay→emit transition
     * (text_delay = 7). Uses `reset_keep_buffers()` afterwards to keep GPU
     * KV cache buffers allocated, avoiding re-allocation on first real frame.
     *
     * Call after `loadModel()` + `loadMimi()`. Reduces TTFB by ~150-400ms.
     */
    warmup(): Promise<void>;
}

/**
 * Initialize the WebGPU device asynchronously.
 *
 * **Must** be called (and awaited) before creating `SttEngine`.
 * Requests the adapter's full limits (especially `max_compute_invocations_per_workgroup`).
 */
export function initWgpuDevice(): Promise<void>;

/**
 * Initialize panic hook for better error messages in browser console.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_sttengine_free: (a: number, b: number) => void;
    readonly initWgpuDevice: () => any;
    readonly sttengine_appendModelShard: (a: number, b: number, c: number) => void;
    readonly sttengine_feedAudio: (a: number, b: number, c: number) => any;
    readonly sttengine_flush: (a: number) => any;
    readonly sttengine_getMetrics: (a: number) => any;
    readonly sttengine_isReady: (a: number) => number;
    readonly sttengine_loadMimi: (a: number, b: number, c: number) => [number, number];
    readonly sttengine_loadModel: (a: number) => [number, number];
    readonly sttengine_loadTokenizer: (a: number, b: number, c: number) => [number, number];
    readonly sttengine_new: () => number;
    readonly sttengine_reset: (a: number) => void;
    readonly sttengine_warmup: (a: number) => any;
    readonly start: () => void;
    readonly __wbg_mimicodec_free: (a: number, b: number) => void;
    readonly mimicodec_feedAudio: (a: number, b: number, c: number) => [number, number];
    readonly mimicodec_new: (a: number, b: number) => any;
    readonly mimicodec_reset: (a: number) => void;
    readonly wasm_bindgen__closure__destroy__h89c6deb10213d14a: (a: number, b: number) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hb5e657e40791b622: (a: number, b: number, c: any, d: any) => void;
    readonly wasm_bindgen__convert__closures_____invoke__hae609bdb97fd6683: (a: number, b: number, c: any) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
