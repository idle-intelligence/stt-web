//! Residual Vector Quantizer (RVQ) for Mimi.
//!
//! Uses a split architecture: 1 semantic codebook (rvq_first) + 31 acoustic codebooks (rvq_rest).
//! Each sub-RVQ has input/output projections between model dimension (512) and codebook dimension (256).

#![allow(clippy::needless_range_loop)]

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3};

/// Single vector quantizer codebook.
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    /// Codebook: (num_bins, dim)
    pub codebook: Array2<f32>,
    /// Pre-computed squared norms for each codebook entry: (num_bins,)
    codebook_sq_norms: Array1<f32>,
}

impl VectorQuantizer {
    pub fn new(codebook: Array2<f32>) -> Self {
        // Pre-compute ||c||² for each codebook entry
        let num_bins = codebook.shape()[0];
        let dim = codebook.shape()[1];
        let mut codebook_sq_norms = Array1::<f32>::zeros(num_bins);
        let cb_slice = codebook.as_slice().unwrap();
        for i in 0..num_bins {
            let mut norm = 0.0f32;
            let base = i * dim;
            for d in 0..dim {
                let v = cb_slice[base + d];
                norm += v * v;
            }
            codebook_sq_norms[i] = norm;
        }
        Self { codebook, codebook_sq_norms }
    }

    /// Quantize vectors to nearest codebook entry.
    ///
    /// dist(x, c) = ||x||² - 2*x·c + ||c||²
    /// Since ||x||² is constant across codebook entries, we just need
    /// argmin_c(-2*x·c + ||c||²) = argmin_c(||c||² - 2*x·c)
    ///
    /// Uses matrix-vector product for time=1 (streaming) or GEMM for time>1.
    pub fn encode(&self, x: &Tensor3) -> (Tensor3, Array3<u32>) {
        let (batch, channels, time) = x.shape();
        let num_bins = self.codebook.shape()[0];
        let dim = self.codebook.shape()[1];

        assert_eq!(channels, dim, "Input channels must match codebook dimension");

        let mut quantized = Array3::<f32>::zeros((batch, channels, time));
        let mut indices = Array3::<u32>::zeros((batch, 1, time));

        let x_slice = x.data.as_slice().unwrap();
        let norms_slice = self.codebook_sq_norms.as_slice().unwrap();
        let cb_slice = self.codebook.as_slice().unwrap();

        for b in 0..batch {
            let b_offset = b * channels * time;
            let q_slice = quantized.as_slice_mut().unwrap();

            if time <= 2 {
                // Optimized path for streaming (time=1 or 2):
                // Extract contiguous x_vec, use ndarray dot for SIMD matvec
                let mut x_vec = Array1::<f32>::zeros(dim);
                for t in 0..time {
                    let xv = x_vec.as_slice_mut().unwrap();
                    for d in 0..dim {
                        xv[d] = x_slice[b_offset + d * time + t];
                    }

                    // Matvec: codebook @ x_vec → (num_bins,) dot products (SIMD)
                    let dots = self.codebook.dot(&x_vec);
                    let dots_s = dots.as_slice().unwrap();

                    // Find argmin of (||c||² - 2*x·c)
                    let mut min_dist = f32::INFINITY;
                    let mut best_idx = 0u32;
                    for i in 0..num_bins {
                        let dist = norms_slice[i] - 2.0 * dots_s[i];
                        if dist < min_dist {
                            min_dist = dist;
                            best_idx = i as u32;
                        }
                    }

                    // Store quantized vector in (B, C, T) layout
                    let cb_base = best_idx as usize * dim;
                    for d in 0..dim {
                        q_slice[b_offset + d * time + t] = cb_slice[cb_base + d];
                    }
                    indices[[b, 0, t]] = best_idx;
                }
            } else {
                // GEMM path for larger time dimensions
                let mut x_mat = Array2::<f32>::zeros((time, dim));
                let xm = x_mat.as_slice_mut().unwrap();
                for t in 0..time {
                    for d in 0..dim {
                        xm[t * dim + d] = x_slice[b_offset + d * time + t];
                    }
                }

                let dots = x_mat.dot(&self.codebook.t());
                let dots_slice = dots.as_slice().unwrap();

                for t in 0..time {
                    let mut min_dist = f32::INFINITY;
                    let mut best_idx = 0u32;
                    let row_base = t * num_bins;

                    for i in 0..num_bins {
                        let dist = norms_slice[i] - 2.0 * dots_slice[row_base + i];
                        if dist < min_dist {
                            min_dist = dist;
                            best_idx = i as u32;
                        }
                    }

                    let cb_base = best_idx as usize * dim;
                    for d in 0..dim {
                        q_slice[b_offset + d * time + t] = cb_slice[cb_base + d];
                    }
                    indices[[b, 0, t]] = best_idx;
                }
            }
        }

        (Tensor3::new(quantized), indices)
    }
}

/// Residual Vector Quantizer with input/output projections.
///
/// Projects from model dimension to codebook dimension before quantization.
#[derive(Clone, Debug)]
#[allow(dead_code)]
pub struct ResidualVectorQuantizer {
    /// Input projection: (codebook_dim, input_dim) e.g. (256, 512)
    pub input_proj: Array2<f32>,
    /// Output projection: (output_dim, codebook_dim) e.g. (512, 256)
    pub output_proj: Array2<f32>,
    /// Residual VQ layers
    pub quantizers: Vec<VectorQuantizer>,
}

impl ResidualVectorQuantizer {
    pub fn new(
        input_proj: Array2<f32>,
        output_proj: Array2<f32>,
        quantizers: Vec<VectorQuantizer>,
    ) -> Self {
        Self {
            input_proj,
            output_proj,
            quantizers,
        }
    }

    /// Project input from input_dim to codebook_dim, then encode with residual VQ.
    ///
    /// Returns tensor of shape (batch, n_quantizers, time) with token IDs.
    pub fn encode(&self, x: &Tensor3) -> Array3<u32> {
        let projected = self.project_input(x);
        let (batch, _channels, time) = projected.shape();
        let n_q = self.quantizers.len();
        let mut codes = Array3::<u32>::zeros((batch, n_q, time));
        let mut residual = projected;

        for (q_idx, quantizer) in self.quantizers.iter().enumerate() {
            let (quantized, indices) = quantizer.encode(&residual);
            for b in 0..batch {
                for t in 0..time {
                    codes[[b, q_idx, t]] = indices[[b, 0, t]];
                }
            }
            // Compute next residual: residual = residual - quantized
            residual = Tensor3::new(&residual.data - &quantized.data);
        }

        codes
    }

    /// Project input: (batch, input_dim, time) → (batch, codebook_dim, time)
    /// Uses matvec for time<=2 (streaming), GEMM for larger time.
    fn project_input(&self, x: &Tensor3) -> Tensor3 {
        let (batch, in_dim, time) = x.shape();
        let cb_dim = self.input_proj.shape()[0];
        let mut projected = Array3::<f32>::zeros((batch, cb_dim, time));

        let x_slice = x.data.as_slice().unwrap();
        let p_slice = projected.as_slice_mut().unwrap();

        for b in 0..batch {
            let b_offset = b * in_dim * time;
            let p_b_offset = b * cb_dim * time;

            if time <= 2 {
                // Streaming path: extract contiguous x_vec, use ndarray dot (SIMD)
                let mut x_vec = Array1::<f32>::zeros(in_dim);
                for t in 0..time {
                    let xv = x_vec.as_slice_mut().unwrap();
                    for d in 0..in_dim {
                        xv[d] = x_slice[b_offset + d * time + t];
                    }

                    // input_proj @ x_vec: (cb_dim, in_dim) @ (in_dim,) → (cb_dim,)
                    let result = self.input_proj.dot(&x_vec);
                    let r_s = result.as_slice().unwrap();
                    for d in 0..cb_dim {
                        p_slice[p_b_offset + d * time + t] = r_s[d];
                    }
                }
            } else {
                // GEMM path: transpose, multiply, transpose back
                let proj_t = self.input_proj.t();
                let mut x_mat = Array2::<f32>::zeros((time, in_dim));
                let xm = x_mat.as_slice_mut().unwrap();
                for t in 0..time {
                    for d in 0..in_dim {
                        xm[t * in_dim + d] = x_slice[b_offset + d * time + t];
                    }
                }
                let result = x_mat.dot(&proj_t);
                let r_slice = result.as_slice().unwrap();
                for t in 0..time {
                    for d in 0..cb_dim {
                        p_slice[p_b_offset + d * time + t] = r_slice[t * cb_dim + d];
                    }
                }
            }
        }
        Tensor3::new(projected)
    }
}

/// Split Residual Vector Quantizer.
///
/// Split architecture: rvq_first (1 semantic codebook) + rvq_rest (31 acoustic codebooks).
/// Each sub-RVQ independently projects and quantizes the same input.
#[derive(Clone, Debug)]
pub struct SplitResidualVectorQuantizer {
    pub rvq_first: ResidualVectorQuantizer,
    pub rvq_rest: ResidualVectorQuantizer,
    pub n_q: usize,
}

impl SplitResidualVectorQuantizer {
    pub fn new(
        rvq_first: ResidualVectorQuantizer,
        rvq_rest: ResidualVectorQuantizer,
        n_q: usize,
    ) -> Self {
        assert_eq!(
            rvq_first.quantizers.len() + rvq_rest.quantizers.len(),
            n_q,
            "Total quantizers must match n_q"
        );
        Self {
            rvq_first,
            rvq_rest,
            n_q,
        }
    }

    /// Encode using split residual quantization.
    ///
    /// Returns tensor of shape (batch, n_q, time) with token IDs.
    pub fn encode(&self, x: &Tensor3) -> Array3<u32> {
        let (batch, _channels, time) = x.shape();
        let mut codes = Array3::<u32>::zeros((batch, self.n_q, time));

        // Encode with rvq_first (semantic codebooks)
        let first_codes = self.rvq_first.encode(x);
        let n_first = self.rvq_first.quantizers.len();
        for b in 0..batch {
            for q in 0..n_first {
                for t in 0..time {
                    codes[[b, q, t]] = first_codes[[b, q, t]];
                }
            }
        }

        // Encode with rvq_rest (acoustic codebooks)
        let rest_codes = self.rvq_rest.encode(x);
        let n_rest = self.rvq_rest.quantizers.len();
        for b in 0..batch {
            for q in 0..n_rest {
                for t in 0..time {
                    codes[[b, n_first + q, t]] = rest_codes[[b, q, t]];
                }
            }
        }

        codes
    }
}
