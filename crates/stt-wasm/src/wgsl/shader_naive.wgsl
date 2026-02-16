// Q4 dequantize + matmul — naive kernel (WASM/WebGPU compatible).
//
// One thread per output element. Compatible with WebGPU's 256 workgroup invocation limit.
// Based on voxtral-mini-realtime-rs/src/gguf/shader_naive.wgsl.
//
// TODO: Port from voxtral-rs reference implementation.
//
// Bindings:
//   @binding(0) q4_weights: array<u32>   — packed Q4 blocks
//   @binding(1) input: array<f32>         — input activation matrix
//   @binding(2) output: array<f32>        — output matrix
//   @binding(3) info: array<u32>          — [B, M, K, N, blocks_per_row]
