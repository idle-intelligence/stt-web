//! Residual Vector Quantizer (RVQ) for Mimi.
//!
//! Converts continuous embeddings into discrete codebook tokens.
//! Uses 32 codebooks with 2048 bins each.

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

/// Split Residual Vector Quantizer.
///
/// First quantizes at 50Hz (1 codebook), then splits into 32 codebooks at 12.5Hz.
#[derive(Clone, Debug)]
pub struct SplitResidualVectorQuantizer {
    pub quantizers: Vec<VectorQuantizer>,
    pub n_q: usize, // Number of codebooks (32 for Mimi)
    pub bins: usize, // Number of bins per codebook (2048)
}

impl SplitResidualVectorQuantizer {
    pub fn new(quantizers: Vec<VectorQuantizer>, n_q: usize, bins: usize) -> Self {
        assert_eq!(quantizers.len(), n_q, "Number of quantizers must match n_q");
        Self { quantizers, n_q, bins }
    }

    /// Encode using residual quantization.
    ///
    /// Returns tensor of shape (batch, n_q, time) with token IDs.
    pub fn encode(&self, x: &Tensor3) -> Array3<u32> {
        let (batch, _channels, time) = x.shape();
        let mut codes = Array3::<u32>::zeros((batch, self.n_q, time));

        let mut residual = x.clone();

        // Iteratively quantize residuals
        for (q_idx, quantizer) in self.quantizers.iter().enumerate() {
            let (quantized, indices) = quantizer.encode(&residual);

            // Store codes
            for b in 0..batch {
                for t in 0..time {
                    codes[[b, q_idx, t]] = indices[[b, 0, t]];
                }
            }

            // Compute next residual
            residual = &residual + &(&quantized * -1.0);
        }

        codes
    }

    /// Decode from token IDs back to continuous embeddings (not needed for encode-only STT).
    #[allow(dead_code)]
    pub fn decode(&self, codes: &Array3<u32>) -> Tensor3 {
        let (batch, n_q, time) = codes.dim();
        assert_eq!(n_q, self.n_q, "Code dimension mismatch");

        let dim = self.quantizers[0].codebook.shape()[1];
        let mut decoded = Array3::<f32>::zeros((batch, dim, time));

        for (q_idx, quantizer) in self.quantizers.iter().enumerate() {
            for b in 0..batch {
                for t in 0..time {
                    let code_idx = codes[[b, q_idx, t]] as usize;
                    for c in 0..dim {
                        decoded[[b, c, t]] += quantizer.codebook[[code_idx, c]];
                    }
                }
            }
        }

        Tensor3::new(decoded)
    }
}
