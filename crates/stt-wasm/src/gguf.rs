//! Q4 GGUF weight loader and WGSL dequantization shaders.
//!
//! Pipeline: GGUF file → parse header/tensors → store Q4 blocks as raw bytes
//! on GPU → dequantize via WGSL compute shader → matmul.
//!
//! Key patterns from voxtral-mini-realtime-rs:
//! - `ShardedCursor`: Read+Seek over Vec<Vec<u8>> for multi-shard GGUF (stays under 2GB allocation limit)
//! - Two-phase loading: parse GGUF, drop reader, finalize tensors (stays under 4GB address space)
//! - Naive WGSL kernel for WASM (tiled kernel is native-only)

use burn::prelude::*;

use crate::Backend;

/// GGUF file reader supporting multi-shard files.
///
/// Uses `ShardedCursor` (Vec<Vec<u8>>) to stay under WASM's 2GB
/// single-allocation limit.
pub struct GgufReader {
    // TODO: GGUF header, metadata, tensor descriptors
}

impl GgufReader {
    /// Parse a GGUF file from raw bytes.
    pub fn from_bytes(_data: Vec<u8>) -> anyhow::Result<Self> {
        todo!("Parse GGUF v2/v3 header, metadata, tensor descriptors")
    }

    /// Parse a multi-shard GGUF file (for browser: each shard < 512MB).
    pub fn from_shards(_shards: Vec<Vec<u8>>) -> anyhow::Result<Self> {
        todo!("ShardedCursor over multiple shards")
    }
}

/// Q4 model loader with two-phase loading for WASM memory constraints.
///
/// Phase 1: Parse GGUF, extract Q4 tensor descriptors
/// Phase 2: Drop reader, finalize tensors on GPU
pub struct Q4ModelLoader {
    // TODO: tensor descriptors, raw Q4 data
}

impl Q4ModelLoader {
    /// Phase 1: Parse GGUF and prepare tensor descriptors.
    pub fn from_reader(_reader: GgufReader) -> anyhow::Result<Self> {
        todo!("Extract Q4 tensor descriptors from GGUF")
    }

    /// Phase 2: Finalize model on GPU device.
    ///
    /// Transfers Q4 data to GPU buffers and sets up dequant shaders.
    /// The GGUF reader should be dropped before calling this to free memory.
    pub fn finalize(self, _device: &<Backend as burn::tensor::backend::Backend>::Device) -> anyhow::Result<Q4Model> {
        todo!("Transfer Q4 blocks to GPU, create dequant pipeline")
    }
}

/// Q4 quantized model ready for inference.
///
/// Stores Q4 blocks on GPU and uses WGSL compute shaders for
/// fused dequantize + matmul operations.
pub struct Q4Model {
    // TODO: Q4 GPU buffers, dequant pipelines
}

impl Q4Model {
    /// Q4 matmul: dequantize weights on-the-fly and multiply with input.
    ///
    /// Uses the naive WGSL kernel (compatible with WASM/WebGPU).
    pub fn q4_matmul(&self, _input: Tensor<Backend, 2>, _layer_name: &str) -> Tensor<Backend, 2> {
        todo!("Dispatch WGSL compute shader for fused Q4 dequant + matmul")
    }

    /// Q4 embedding lookup: dequantize specific rows from Q4 storage.
    ///
    /// Embeddings are Q4 on GPU + CPU byte copy for token lookups.
    pub fn embed_tokens(&self, _token_ids: &[u32]) -> Tensor<Backend, 2> {
        todo!("CPU-side Q4 dequant for embedding lookup")
    }
}

/// Read and seek over multiple byte buffers (shards), treating them as one
/// contiguous stream. Keeps each shard as a separate Vec<u8> to stay under
/// WASM's 2GB single-allocation limit.
pub struct ShardedCursor {
    shards: Vec<Vec<u8>>,
    shard_idx: usize,
    offset_in_shard: usize,
}

impl ShardedCursor {
    pub fn new(shards: Vec<Vec<u8>>) -> Self {
        Self {
            shards,
            shard_idx: 0,
            offset_in_shard: 0,
        }
    }
}

impl std::io::Read for ShardedCursor {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        todo!("Read across shard boundaries")
    }
}

impl std::io::Seek for ShardedCursor {
    fn seek(&mut self, _pos: std::io::SeekFrom) -> std::io::Result<u64> {
        todo!("Seek across shard boundaries")
    }
}
