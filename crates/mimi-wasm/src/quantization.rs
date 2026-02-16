//! Residual Vector Quantizer (RVQ) for Mimi.
//!
//! Uses a split architecture: 1 semantic codebook (rvq_first) + 31 acoustic codebooks (rvq_rest).
//! Each sub-RVQ has input/output projections between model dimension (512) and codebook dimension (256).

use crate::tensor::Tensor3;
use ndarray::{Array2, Array3};

/// Single vector quantizer codebook.
#[derive(Clone, Debug)]
pub struct VectorQuantizer {
    /// Codebook: (num_bins, dim)
    pub codebook: Array2<f32>,
}

impl VectorQuantizer {
    pub fn new(codebook: Array2<f32>) -> Self {
        Self { codebook }
    }

    /// Quantize vectors to nearest codebook entry.
    /// Returns (quantized_vectors, indices).
    pub fn encode(&self, x: &Tensor3) -> (Tensor3, Array3<u32>) {
        let (batch, channels, time) = x.shape();
        let num_bins = self.codebook.shape()[0];
        let dim = self.codebook.shape()[1];

        assert_eq!(channels, dim, "Input channels must match codebook dimension");

        let mut quantized = Array3::<f32>::zeros((batch, channels, time));
        let mut indices = Array3::<u32>::zeros((batch, 1, time));

        // For each batch and time step, find nearest codebook entry
        for b in 0..batch {
            for t in 0..time {
                let mut min_dist = f32::INFINITY;
                let mut best_idx = 0u32;

                // Extract input vector at (b, :, t)
                for bin_idx in 0..num_bins {
                    let mut dist = 0.0;
                    for c in 0..dim {
                        let diff = x.data[[b, c, t]] - self.codebook[[bin_idx, c]];
                        dist += diff * diff;
                    }

                    if dist < min_dist {
                        min_dist = dist;
                        best_idx = bin_idx as u32;
                    }
                }

                // Store quantized vector
                for c in 0..dim {
                    quantized[[b, c, t]] = self.codebook[[best_idx as usize, c]];
                }
                indices[[b, 0, t]] = best_idx;
            }
        }

        (Tensor3::new(quantized), indices)
    }
}

/// Residual Vector Quantizer with input/output projections.
///
/// Projects from model dimension to codebook dimension before quantization.
#[derive(Clone, Debug)]
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
    fn project_input(&self, x: &Tensor3) -> Tensor3 {
        let (batch, in_dim, time) = x.shape();
        let cb_dim = self.input_proj.shape()[0];
        let mut projected = Array3::<f32>::zeros((batch, cb_dim, time));
        for b in 0..batch {
            for t in 0..time {
                for c_out in 0..cb_dim {
                    let mut sum = 0.0;
                    for c_in in 0..in_dim {
                        sum += self.input_proj[[c_out, c_in]] * x.data[[b, c_in, t]];
                    }
                    projected[[b, c_out, t]] = sum;
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
