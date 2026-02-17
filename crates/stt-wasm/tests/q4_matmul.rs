//! Integration tests for Q4 matmul shader.
//!
//! Tests the WGSL compute shader against CPU reference implementation
//! to verify correctness on real GPU hardware.

use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::tensor::{Tensor, TensorData};

use stt_wasm::gguf::{Q4Tensor, q4_matmul};

/// CPU reference dequantization for test verification.
fn dequantize_q4_0_cpu(raw: &[u8], num_elements: usize) -> Vec<f32> {
    let num_blocks = num_elements / 32;
    let mut output = vec![0.0f32; num_elements];
    for block_idx in 0..num_blocks {
        let offset = block_idx * 18;
        let bits = u16::from_le_bytes([raw[offset], raw[offset + 1]]);
        // f16 to f32
        let d = {
            let sign = ((bits >> 15) & 1) as u32;
            let exp = ((bits >> 10) & 0x1F) as u32;
            let mant = (bits & 0x3FF) as u32;
            let f32_exp = (exp as i32 - 15 + 127) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
        };
        let base = block_idx * 32;
        for i in 0..16 {
            let byte = raw[offset + 2 + i];
            output[base + i] = ((byte & 0x0F) as f32 - 8.0) * d;
            output[base + i + 16] = (((byte >> 4) & 0x0F) as f32 - 8.0) * d;
        }
    }
    output
}

fn device() -> WgpuDevice {
    WgpuDevice::default()
}

/// Convert f32 to f16 bits (IEEE 754 half-precision).
fn f32_to_f16_bits(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Inf/NaN
        return sign | 0x7C00 | ((mant != 0) as u16);
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00; // overflow → inf
    }
    if new_exp <= 0 {
        return sign; // underflow → 0
    }

    sign | ((new_exp as u16) << 10) | ((mant >> 13) as u16)
}

/// Quantize f32 values to Q4_0 format (same as quantize.py).
fn quantize_q4_block(values: &[f32]) -> Vec<u8> {
    assert_eq!(values.len(), 32);
    let abs_max = values.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale = if abs_max > 0.0 { abs_max / 7.0 } else { 1.0 };

    let mut block = Vec::with_capacity(18);
    // f16 scale
    let scale_f16 = f32_to_f16_bits(scale);
    block.extend_from_slice(&scale_f16.to_le_bytes());
    // Pack nibbles: byte[j] = (elem[j+16] + 8) << 4 | (elem[j] + 8)
    for j in 0..16 {
        let lo = (values[j] / scale).round().clamp(-8.0, 7.0) as i8;
        let hi = (values[j + 16] / scale).round().clamp(-8.0, 7.0) as i8;
        let lo_nibble = ((lo + 8) & 0x0F) as u8;
        let hi_nibble = ((hi + 8) & 0x0F) as u8;
        block.push((hi_nibble << 4) | lo_nibble);
    }
    block
}

/// Quantize a 2D f32 matrix to Q4_0 bytes.
fn quantize_matrix(data: &[f32], rows: usize, cols: usize) -> Vec<u8> {
    assert_eq!(data.len(), rows * cols);
    assert_eq!(cols % 32, 0);
    let blocks_per_row = cols / 32;
    let mut output = Vec::with_capacity(rows * blocks_per_row * 18);
    for r in 0..rows {
        for b in 0..blocks_per_row {
            let start = r * cols + b * 32;
            let block_data = &data[start..start + 32];
            output.extend_from_slice(&quantize_q4_block(block_data));
        }
    }
    output
}

#[test]
fn test_q4_matmul_identity_like() {
    // Simple test: weight matrix is near-identity (diagonal-like pattern)
    // Input should pass through approximately unchanged
    let device = device();
    let n = 64; // out_features
    let k = 64; // in_features

    // Create a diagonal-ish weight matrix
    let mut weights = vec![0.0f32; n * k];
    for i in 0..n.min(k) {
        weights[i * k + i] = 7.0; // Max quantizable value
    }

    let q4_bytes = quantize_matrix(&weights, n, k);
    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();

    // Input: [1, 1, k]
    let input_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1).collect();
    let input = Tensor::<Wgpu, 3>::from_data(
        TensorData::new(input_data.clone(), [1, 1, k]),
        &device,
    );

    // GPU matmul
    let output = q4_matmul(input, &q4_tensor);
    let output_data: Vec<f32> = output
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // CPU reference: dequantize weights then matmul
    let deq_weights = dequantize_q4_0_cpu(&q4_bytes, n * k);
    let mut expected = vec![0.0f32; n];
    for row in 0..n {
        for col in 0..k {
            expected[row] += input_data[col] * deq_weights[row * k + col];
        }
    }

    // Compare
    let max_err: f32 = output_data
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max error: {max_err}");
    println!("GPU output[0..4]: {:?}", &output_data[..4]);
    println!("CPU expected[0..4]: {:?}", &expected[..4]);

    assert!(
        max_err < 0.5,
        "Q4 matmul error too high: {max_err}"
    );
}

#[test]
fn test_q4_matmul_different_inputs() {
    // Test that different inputs produce different outputs
    let device = device();
    let n = 128;
    let k = 128;

    // Random-ish weight matrix
    let weights: Vec<f32> = (0..n * k)
        .map(|i| ((i * 7 + 13) % 15) as f32 - 7.0)
        .collect();

    let q4_bytes = quantize_matrix(&weights, n, k);
    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();

    // Input 1
    let input1_data: Vec<f32> = (0..k).map(|i| (i as f32) * 0.1 - 6.4).collect();
    let input1 = Tensor::<Wgpu, 3>::from_data(
        TensorData::new(input1_data.clone(), [1, 1, k]),
        &device,
    );

    // Input 2 (different values)
    let input2_data: Vec<f32> = (0..k).map(|i| ((i * 3 + 5) as f32) * 0.05 - 3.2).collect();
    let input2 = Tensor::<Wgpu, 3>::from_data(
        TensorData::new(input2_data.clone(), [1, 1, k]),
        &device,
    );

    let output1: Vec<f32> = q4_matmul(input1, &q4_tensor)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    let output2: Vec<f32> = q4_matmul(input2, &q4_tensor)
        .into_data()
        .to_vec::<f32>()
        .unwrap();

    // Check outputs are different
    let diff: f32 = output1
        .iter()
        .zip(output2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();

    println!("Output1[0..4]: {:?}", &output1[..4]);
    println!("Output2[0..4]: {:?}", &output2[..4]);
    println!("Sum of absolute differences: {diff}");

    assert!(
        diff > 1.0,
        "Different inputs produced same output! diff={diff}"
    );

    // Also verify against CPU reference
    let deq_weights = dequantize_q4_0_cpu(&q4_bytes, n * k);
    let mut cpu_out1 = vec![0.0f32; n];
    for row in 0..n {
        for col in 0..k {
            cpu_out1[row] += input1_data[col] * deq_weights[row * k + col];
        }
    }

    let max_err: f32 = output1
        .iter()
        .zip(cpu_out1.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("Max GPU vs CPU error: {max_err}");
    assert!(max_err < 1.0, "GPU output doesn't match CPU: max_err={max_err}");
}

#[test]
fn test_q4_matmul_larger_realistic() {
    // Test with dimensions closer to the actual model (2048 → 6144)
    let device = device();
    let n = 256; // reduced from 6144 for speed
    let k = 256; // reduced from 2048 for speed

    // Create deterministic weights
    let weights: Vec<f32> = (0..n * k)
        .map(|i| {
            let x = ((i * 17 + 31) % 256) as f32 / 256.0 - 0.5;
            x * 2.0 // range [-1, 1]
        })
        .collect();

    let q4_bytes = quantize_matrix(&weights, n, k);
    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n, k], &device).unwrap();

    // Three different inputs
    let inputs: Vec<Vec<f32>> = (0..3)
        .map(|seed| {
            (0..k)
                .map(|i| {
                    let x = ((i * (seed * 7 + 3) + seed * 11 + 7) % 200) as f32 / 100.0 - 1.0;
                    x
                })
                .collect()
        })
        .collect();

    let outputs: Vec<Vec<f32>> = inputs
        .iter()
        .map(|inp| {
            let tensor = Tensor::<Wgpu, 3>::from_data(
                TensorData::new(inp.clone(), [1, 1, k]),
                &device,
            );
            q4_matmul(tensor, &q4_tensor)
                .into_data()
                .to_vec::<f32>()
                .unwrap()
        })
        .collect();

    // All three outputs should be different
    for i in 0..3 {
        for j in (i + 1)..3 {
            let diff: f32 = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            println!("Diff between output {i} and {j}: {diff}");
            assert!(
                diff > 1.0,
                "Outputs {i} and {j} are the same! diff={diff}"
            );
        }
    }

    // Verify first output against CPU
    let deq_weights = dequantize_q4_0_cpu(&q4_bytes, n * k);
    let mut cpu_out = vec![0.0f32; n];
    for row in 0..n {
        for col in 0..k {
            cpu_out[row] += inputs[0][col] * deq_weights[row * k + col];
        }
    }

    let max_err: f32 = outputs[0]
        .iter()
        .zip(cpu_out.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("Max GPU vs CPU error (256x256): {max_err}");
    assert!(max_err < 2.0, "Large matrix: GPU doesn't match CPU: max_err={max_err}");
}

#[test]
fn test_embedding_sum_then_q4_matmul() {
    // Mimics the actual model forward pass:
    // 1. Create 32 embedding rows from from_data() (simulating EmbeddingStore)
    // 2. Sum them all + a text embedding
    // 3. Pass through Q4 matmul
    // 4. Verify different token combinations produce different outputs
    let device = device();
    let dim = 128;
    let n_out = 64;
    let n_codebooks = 32;

    // Create Q4 weights (simulating in_proj)
    let weights: Vec<f32> = (0..n_out * dim)
        .map(|i| ((i * 13 + 7) % 15) as f32 - 7.0)
        .collect();
    let q4_bytes = quantize_matrix(&weights, n_out, dim);
    let q4_tensor = Q4Tensor::from_q4_bytes(&q4_bytes, [n_out, dim], &device).unwrap();

    // Create fake embedding tables (simulating EmbeddingStore rows)
    // Each codebook has a few rows (simulating vocab_size=8)
    let vocab = 8;
    let emb_tables: Vec<Vec<Vec<f32>>> = (0..n_codebooks)
        .map(|cb| {
            (0..vocab)
                .map(|tok| {
                    (0..dim)
                        .map(|d| {
                            // Deterministic but unique per (cb, tok, d)
                            let seed = cb * 10000 + tok * 100 + d;
                            ((seed * 7 + 3) % 200) as f32 / 100.0 - 1.0
                        })
                        .collect()
                })
                .collect()
        })
        .collect();

    // Token set 1: all token 0
    let tokens1: Vec<usize> = vec![0; n_codebooks];
    // Token set 2: all token 1
    let tokens2: Vec<usize> = vec![1; n_codebooks];
    // Token set 3: mixed tokens
    let tokens3: Vec<usize> = (0..n_codebooks).map(|i| i % vocab).collect();

    let mut outputs = Vec::new();

    for tokens in [&tokens1, &tokens2, &tokens3] {
        // Sum embeddings (mimicking model.forward embedding sum)
        let mut input = Tensor::<Wgpu, 3>::zeros([1, 1, dim], &device);
        for (cb, &tok) in tokens.iter().enumerate() {
            let row = &emb_tables[cb][tok];
            let emb = Tensor::<Wgpu, 2>::from_data(
                TensorData::new(row.clone(), [1, dim]),
                &device,
            );
            input = input + emb.unsqueeze_dim::<3>(0);
        }

        // Q4 matmul (mimicking in_proj.forward(x))
        let out = q4_matmul(input, &q4_tensor);
        let out_data: Vec<f32> = out.into_data().to_vec::<f32>().unwrap();
        outputs.push(out_data);
    }

    // Verify all outputs are different
    for i in 0..3 {
        for j in (i + 1)..3 {
            let diff: f32 = outputs[i]
                .iter()
                .zip(outputs[j].iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            println!("Token set {i} vs {j}: diff={diff}");
            assert!(
                diff > 1.0,
                "Token sets {i} and {j} produced same output! diff={diff}"
            );
        }
    }
    println!("Output 0 first4: {:?}", &outputs[0][..4]);
    println!("Output 1 first4: {:?}", &outputs[1][..4]);
    println!("Output 2 first4: {:?}", &outputs[2][..4]);
    println!("PASS: Embedding sum → Q4 matmul produces distinct outputs");
}

#[test]
fn test_q4_matmul_with_actual_gguf() {
    // Load actual GGUF model weights and run a test
    let gguf_path = std::path::Path::new("../../models/stt-1b-en_fr-q4.gguf");
    if !gguf_path.exists() {
        println!("Skipping: GGUF file not found at {}", gguf_path.display());
        return;
    }

    let device = device();

    // Read the GGUF file
    let file_data = std::fs::read(gguf_path).unwrap();
    let cursor = std::io::Cursor::new(file_data);
    let mut reader = stt_wasm::gguf::GgufReader::open(cursor).unwrap();

    println!("GGUF: {} tensors", reader.tensor_count());

    // Load emb.0.weight as EmbeddingStore
    let emb0_info = reader.tensor_info("emb.0.weight").unwrap().clone();
    let emb0_shape: Vec<usize> = emb0_info.shape().iter().rev().map(|&d| d as usize).collect();
    let emb0_bytes = reader.tensor_data("emb.0.weight").unwrap();
    let emb0 = stt_wasm::gguf::EmbeddingStore::new(emb0_bytes, emb0_shape[0], emb0_shape[1]);
    println!("emb.0: shape={:?}", emb0_shape);

    // Look up several different tokens
    let test_tokens: Vec<u32> = vec![0, 1, 100, 326, 955, 2048];
    let mut embs = Vec::new();
    for &tok in &test_tokens {
        let mut data = vec![0.0f32; emb0_shape[1]];
        emb0.embed_id_add_cpu(tok, &mut data);
        let norm: f32 = data.iter().map(|v| v * v).sum::<f32>().sqrt();
        println!("  token {tok}: norm={norm:.4} first4={:?}", &data[..4]);
        embs.push(data);
    }

    // Verify all embeddings are distinct
    for i in 0..embs.len() {
        for j in (i + 1)..embs.len() {
            let cos: f32 = embs[i]
                .iter()
                .zip(embs[j].iter())
                .map(|(a, b)| a * b)
                .sum::<f32>()
                / (embs[i].iter().map(|v| v * v).sum::<f32>().sqrt()
                    * embs[j].iter().map(|v| v * v).sum::<f32>().sqrt()
                    + 1e-10);
            if (cos - 1.0).abs() < 1e-4 {
                panic!(
                    "Tokens {} and {} have identical embeddings! cos={cos}",
                    test_tokens[i], test_tokens[j]
                );
            }
        }
    }
    println!("All embeddings are distinct");

    // Load first layer in_proj as Q4Linear
    let in_proj_info = reader
        .tensor_info("transformer.layers.0.self_attn.in_proj_weight")
        .unwrap()
        .clone();
    let in_proj_shape: Vec<usize> = in_proj_info.shape().iter().rev().map(|&d| d as usize).collect();
    let in_proj_bytes = reader
        .tensor_data("transformer.layers.0.self_attn.in_proj_weight")
        .unwrap();
    println!("in_proj shape: {:?} ({} bytes)", in_proj_shape, in_proj_bytes.len());

    let q4 = Q4Tensor::from_q4_bytes(&in_proj_bytes, [in_proj_shape[0], in_proj_shape[1]], &device).unwrap();

    // Create two different embedding sums
    let dim = emb0_shape[1];
    let mut emb_326_data = vec![0.0f32; dim];
    emb0.embed_id_add_cpu(326, &mut emb_326_data);
    let mut emb_955_data = vec![0.0f32; dim];
    emb0.embed_id_add_cpu(955, &mut emb_955_data);

    let emb_326 = Tensor::<Wgpu, 2>::from_data(TensorData::new(emb_326_data, [1, dim]), &device);
    let emb_955 = Tensor::<Wgpu, 2>::from_data(TensorData::new(emb_955_data, [1, dim]), &device);

    let input1 = emb_326.unsqueeze_dim::<3>(0); // [1, 1, 2048]
    let input2 = emb_955.unsqueeze_dim::<3>(0); // [1, 1, 2048]

    let out1: Vec<f32> = q4_matmul(input1, &q4).into_data().to_vec::<f32>().unwrap();
    let out2: Vec<f32> = q4_matmul(input2, &q4).into_data().to_vec::<f32>().unwrap();

    let diff: f32 = out1.iter().zip(out2.iter()).map(|(a, b)| (a - b).abs()).sum();
    println!("in_proj output diff (token 326 vs 955): {diff}");
    println!("  out1 first4: {:?}", &out1[..4]);
    println!("  out2 first4: {:?}", &out2[..4]);

    assert!(
        diff > 1.0,
        "Different embeddings produced same in_proj output! diff={diff}"
    );
    println!("PASS: Real GGUF weights produce distinct outputs for different inputs");
}
