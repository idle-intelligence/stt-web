//! Small transformer used inside Mimi encoder.
//!
//! 8 layers, 8 heads, 512 dim, causal, with RoPE positional embeddings.

// Performance-critical inner loops use index-based iteration for clarity
// and to allow multi-array access with the same index variable.
#![allow(clippy::needless_range_loop)]
#![allow(clippy::excessive_precision)]

use crate::tensor::Tensor3;
use ndarray::{Array1, Array2, Array3, Axis};

/// Multi-head self-attention layer with separate Q/K/V/O projections.
#[derive(Clone, Debug)]
pub struct MultiHeadAttention {
    pub num_heads: usize,
    pub head_dim: usize,
    pub q_proj: Array2<f32>, // (d_model, d_model)
    pub k_proj: Array2<f32>, // (d_model, d_model)
    pub v_proj: Array2<f32>, // (d_model, d_model)
    pub o_proj: Array2<f32>, // (d_model, d_model)
    // Pre-transposed O projection for streaming: (d_model, d_model), contiguous
    o_proj_t: Array2<f32>,
    // Fused QKV projection: (3*d_model, d_model) — single matmul instead of 3
    qkv_proj: Array2<f32>,
    // Pre-transposed fused QKV: (d_model, 3*d_model), contiguous
    qkv_proj_t: Array2<f32>,
    // Pre-allocated KV cache for streaming (avoids per-step allocation)
    k_cache: Array3<f32>,   // (num_heads, MAX_CONTEXT, head_dim)
    v_cache: Array3<f32>,   // (num_heads, MAX_CONTEXT, head_dim)
    cache_len: usize,       // number of valid entries in cache
    // Pre-computed inverse frequencies for RoPE
    inv_freq: Vec<f32>,
    // Pre-allocated scratch buffers for streaming step (avoid per-call allocations)
    scratch_scores: Vec<f32>,   // reusable scores buffer (capacity grows as needed)
}

impl MultiHeadAttention {
    const MAX_CONTEXT: usize = 250;

    pub fn new(
        d_model: usize,
        num_heads: usize,
        q_proj: Array2<f32>,
        k_proj: Array2<f32>,
        v_proj: Array2<f32>,
        o_proj: Array2<f32>,
    ) -> Self {
        assert_eq!(d_model % num_heads, 0, "d_model must be divisible by num_heads");
        let head_dim = d_model / num_heads;
        let half_dim = head_dim / 2;

        // Pre-compute inverse frequencies for RoPE
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / 10000.0_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Pre-allocate KV cache (batch=1 assumed for streaming)
        let k_cache = Array3::<f32>::zeros((num_heads, Self::MAX_CONTEXT, head_dim));
        let v_cache = Array3::<f32>::zeros((num_heads, Self::MAX_CONTEXT, head_dim));

        // Fused QKV: stack [q_proj; k_proj; v_proj] → (3*d_model, d_model)
        let d = d_model;
        let mut qkv_proj = Array2::<f32>::zeros((3 * d, d));
        let q_s = q_proj.as_slice().unwrap();
        let k_s = k_proj.as_slice().unwrap();
        let v_s = v_proj.as_slice().unwrap();
        let qkv_s = qkv_proj.as_slice_mut().unwrap();
        qkv_s[..d * d].copy_from_slice(q_s);
        qkv_s[d * d..2 * d * d].copy_from_slice(k_s);
        qkv_s[2 * d * d..3 * d * d].copy_from_slice(v_s);

        // Pre-transpose projections for dot products (avoids per-call .t())
        let qkv_proj_t = qkv_proj.t().to_owned();
        let o_proj_t = o_proj.t().to_owned();

        Self {
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            o_proj_t,
            qkv_proj,
            qkv_proj_t,
            k_cache,
            v_cache,
            cache_len: 0,
            inv_freq,
            scratch_scores: Vec::with_capacity(Self::MAX_CONTEXT),
        }
    }

    /// Forward pass (non-streaming, full self-attention with RoPE).
    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, d_model, seq_len) = x.shape();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let half_dim = head_dim / 2;

        // Transpose: (B, C, T) → (B, T, C)
        let xt = x.transpose_12();

        // Project Q, K, V separately
        let q_proj_t = self.q_proj.t();
        let k_proj_t = self.k_proj.t();
        let v_proj_t = self.v_proj.t();

        let mut q = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));
        let mut k = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));
        let mut v = Array3::<f32>::zeros((batch * num_heads, seq_len, head_dim));

        let q_slice = q.as_slice_mut().unwrap();
        let k_slice = k.as_slice_mut().unwrap();
        let v_slice = v.as_slice_mut().unwrap();

        for b in 0..batch {
            let bt_slice = xt.data.index_axis(Axis(0), b); // (T, d_model)
            let q_all = bt_slice.dot(&q_proj_t); // (T, d_model)
            let k_all = bt_slice.dot(&k_proj_t);
            let v_all = bt_slice.dot(&v_proj_t);

            let q_all_s = q_all.as_slice().unwrap();
            let k_all_s = k_all.as_slice().unwrap();
            let v_all_s = v_all.as_slice().unwrap();

            // Reshape: (T, d_model) → (H, T, head_dim) stored as (B*H, T, D)
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let bh_off = bh * seq_len * head_dim;
                let h_off = h * head_dim;
                for t in 0..seq_len {
                    let t_src = t * d_model + h_off;
                    let t_dst = bh_off + t * head_dim;
                    q_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&q_all_s[t_src..t_src + head_dim]);
                    k_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&k_all_s[t_src..t_src + head_dim]);
                    v_slice[t_dst..t_dst + head_dim]
                        .copy_from_slice(&v_all_s[t_src..t_src + head_dim]);
                }
            }
        }

        // RoPE: pre-compute cos/sin table for all positions
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / 10000.0_f32.powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        // Pre-compute cos/sin for all (t, i) pairs
        let mut cos_table = vec![0.0f32; seq_len * half_dim];
        let mut sin_table = vec![0.0f32; seq_len * half_dim];
        for t in 0..seq_len {
            for i in 0..half_dim {
                let freq = t as f32 * inv_freq[i];
                cos_table[t * half_dim + i] = freq.cos();
                sin_table[t * half_dim + i] = freq.sin();
            }
        }

        // Apply RoPE to Q and K using raw slices
        let q_s = q.as_slice_mut().unwrap();
        let k_s = k.as_slice_mut().unwrap();
        let total_bh = batch * num_heads;

        for bh in 0..total_bh {
            let bh_off = bh * seq_len * head_dim;
            for t in 0..seq_len {
                let t_off = bh_off + t * head_dim;
                let ct_off = t * half_dim;
                for i in 0..half_dim {
                    let cos_val = cos_table[ct_off + i];
                    let sin_val = sin_table[ct_off + i];

                    let qi0 = t_off + 2 * i;
                    let q_even = q_s[qi0];
                    let q_odd = q_s[qi0 + 1];
                    q_s[qi0] = q_even * cos_val - q_odd * sin_val;
                    q_s[qi0 + 1] = q_even * sin_val + q_odd * cos_val;

                    let ki0 = t_off + 2 * i;
                    let k_even = k_s[ki0];
                    let k_odd = k_s[ki0 + 1];
                    k_s[ki0] = k_even * cos_val - k_odd * sin_val;
                    k_s[ki0 + 1] = k_even * sin_val + k_odd * cos_val;
                }
            }
        }

        // Attention scores: Q @ K^T / sqrt(head_dim), fused with causal mask + softmax
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut scores = Array3::<f32>::zeros((batch * num_heads, seq_len, seq_len));

        for bh in 0..(batch * num_heads) {
            let q_view = q.index_axis(Axis(0), bh);
            let k_view = k.index_axis(Axis(0), bh);
            // GEMM: (seq_len, head_dim) @ (head_dim, seq_len) → (seq_len, seq_len)
            let attn = q_view.dot(&k_view.t());
            scores.index_axis_mut(Axis(0), bh).assign(&attn);
        }

        // Fused: scale + causal mask + softmax (in-place, no clone)
        {
            let s = scores.as_slice_mut().unwrap();
            let total_bh = batch * num_heads;
            for bh in 0..total_bh {
                let bh_off = bh * seq_len * seq_len;
                for i in 0..seq_len {
                    let row_off = bh_off + i * seq_len;
                    let row = &mut s[row_off..row_off + seq_len];

                    // Scale + find max over causal positions (j <= i)
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..=i {
                        row[j] *= scale;
                        if row[j] > max_val {
                            max_val = row[j];
                        }
                    }

                    // Exp + sum for causal positions, zero upper triangle
                    let mut sum = 0.0f32;
                    for j in 0..=i {
                        row[j] = (row[j] - max_val).exp();
                        sum += row[j];
                    }
                    for j in (i + 1)..seq_len {
                        row[j] = 0.0;
                    }

                    // Normalize
                    if sum > 0.0 {
                        let inv_sum = 1.0 / sum;
                        for j in 0..=i {
                            row[j] *= inv_sum;
                        }
                    }
                }
            }
        }

        // Weighted sum: scores @ V → (B*H, T, D)
        let scores_t3 = Tensor3::new(scores);
        let attn_out = scores_t3.matmul(&Tensor3::new(v));

        // Reshape: (B*H, T, D) → (B, T, d_model) using raw slices
        let mut output = Array3::<f32>::zeros((batch, seq_len, d_model));
        let attn_s = attn_out.data.as_slice().unwrap();
        let out_s = output.as_slice_mut().unwrap();
        for b in 0..batch {
            for h in 0..num_heads {
                let bh = b * num_heads + h;
                let src_bh_off = bh * seq_len * head_dim;
                let h_off = h * head_dim;
                for t in 0..seq_len {
                    let src_off = src_bh_off + t * head_dim;
                    let dst_off = b * seq_len * d_model + t * d_model + h_off;
                    out_s[dst_off..dst_off + head_dim]
                        .copy_from_slice(&attn_s[src_off..src_off + head_dim]);
                }
            }
        }

        // Output projection: (B, T, d_model) @ o_proj.T → (B, d_model, T)
        // Fused with transpose to avoid intermediate allocation
        let o_proj_t = self.o_proj.t();
        let mut result_bct = Array3::<f32>::zeros((batch, d_model, seq_len));
        for b in 0..batch {
            let proj_result = output.index_axis(Axis(0), b).dot(&o_proj_t); // (T, d_model)
            // Transpose (T, d_model) → (d_model, T) directly into output
            let src = proj_result.as_slice().unwrap();
            let dst = result_bct.as_slice_mut().unwrap();
            let b_off = b * d_model * seq_len;
            for d in 0..d_model {
                let d_off = b_off + d * seq_len;
                for t in 0..seq_len {
                    dst[d_off + t] = src[t * d_model + d];
                }
            }
        }

        Tensor3::new(result_bct)
    }

    /// Incremental attention: process new timesteps while attending to all cached past.
    ///
    /// `position_offset` is the absolute position of the first new timestep (for RoPE).
    /// Uses pre-allocated KV cache to avoid per-step allocations.
    /// Context window capped at MAX_CONTEXT timesteps.
    pub fn step(&mut self, x: &Tensor3, position_offset: usize) -> Tensor3 {
        let (batch, d_model, new_seq_len) = x.shape();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;
        let half_dim = head_dim / 2;

        // 1. Fused QKV projection with inline transpose
        //    Input is (B, d_model, T) — we need (B, T, d_model) for matmul.
        //    For streaming (batch=1, T=1-2), do matvec directly from BCT layout
        //    to avoid allocating a transposed intermediate.

        let mut q = Array3::<f32>::zeros((num_heads, new_seq_len, head_dim));

        // Handle cache overflow: if cache would exceed MAX_CONTEXT, shift first
        let new_total = self.cache_len + new_seq_len;
        if new_total > Self::MAX_CONTEXT {
            let shift = new_total - Self::MAX_CONTEXT;
            let keep = self.cache_len - shift;
            if keep > 0 {
                let k_s = self.k_cache.as_slice_mut().unwrap();
                let row_stride = Self::MAX_CONTEXT * head_dim;
                for h in 0..num_heads {
                    let base = h * row_stride;
                    let src_start = base + shift * head_dim;
                    let dst_start = base;
                    unsafe {
                        std::ptr::copy(
                            k_s.as_ptr().add(src_start),
                            k_s.as_mut_ptr().add(dst_start),
                            keep * head_dim,
                        );
                    }
                }
                let v_s = self.v_cache.as_slice_mut().unwrap();
                for h in 0..num_heads {
                    let base = h * row_stride;
                    let src_start = base + shift * head_dim;
                    let dst_start = base;
                    unsafe {
                        std::ptr::copy(
                            v_s.as_ptr().add(src_start),
                            v_s.as_mut_ptr().add(dst_start),
                            keep * head_dim,
                        );
                    }
                }
            }
            self.cache_len = keep;
        }

        let cache_write_pos = self.cache_len;

        {
            let q_slice = q.as_slice_mut().unwrap();
            let k_cache_s = self.k_cache.as_slice_mut().unwrap();
            let v_cache_s = self.v_cache.as_slice_mut().unwrap();
            let cache_row_stride = Self::MAX_CONTEXT * head_dim;
            let x_s = x.data.as_slice().unwrap();
            let d3 = 3 * d_model;

            if new_seq_len <= 2 {
                // Streaming fast path: extract contiguous x_vec, use ndarray dot (SIMD).
                let mut x_vec = Array1::<f32>::zeros(d_model);

                for b in 0..batch {
                    let b_off = b * d_model * new_seq_len;
                    for t in 0..new_seq_len {
                        // Extract x_vec from (B, d_model, T) layout
                        let xv = x_vec.as_slice_mut().unwrap();
                        for d in 0..d_model {
                            xv[d] = x_s[b_off + d * new_seq_len + t];
                        }

                        // Fused QKV projection: (3*d_model, d_model) @ (d_model,)
                        let qkv_result = self.qkv_proj.dot(&x_vec);
                        let qkv_s = qkv_result.as_slice().unwrap();

                        // Scatter into Q (local) and K/V (cache)
                        for h in 0..num_heads {
                            let h_off = h * head_dim;
                            let q_dst = h * new_seq_len * head_dim + t * head_dim;
                            q_slice[q_dst..q_dst + head_dim]
                                .copy_from_slice(&qkv_s[h_off..h_off + head_dim]);
                            let cache_dst = h * cache_row_stride + (cache_write_pos + t) * head_dim;
                            k_cache_s[cache_dst..cache_dst + head_dim]
                                .copy_from_slice(&qkv_s[d_model + h_off..d_model + h_off + head_dim]);
                            v_cache_s[cache_dst..cache_dst + head_dim]
                                .copy_from_slice(&qkv_s[2 * d_model + h_off..2 * d_model + h_off + head_dim]);
                        }
                    }
                }
            } else {
                // Batch path: transpose + GEMM
                for b in 0..batch {
                    let b_off = b * d_model * new_seq_len;
                    // Build transposed input: (T, d_model) from (d_model, T)
                    let mut xt_buf = Array2::<f32>::zeros((new_seq_len, d_model));
                    let xt_s = xt_buf.as_slice_mut().unwrap();
                    for t in 0..new_seq_len {
                        for d in 0..d_model {
                            xt_s[t * d_model + d] = x_s[b_off + d * new_seq_len + t];
                        }
                    }

                    let qkv_all = xt_buf.dot(&self.qkv_proj_t);
                    let qkv_s = qkv_all.as_slice().unwrap();

                    for h in 0..num_heads {
                        let q_bh_off = h * new_seq_len * head_dim;
                        let h_off = h * head_dim;
                        let cache_h_base = h * cache_row_stride;

                        for t in 0..new_seq_len {
                            let t_base = t * d3;
                            let q_dst = q_bh_off + t * head_dim;
                            q_slice[q_dst..q_dst + head_dim]
                                .copy_from_slice(&qkv_s[t_base + h_off..t_base + h_off + head_dim]);
                            let cache_dst = cache_h_base + (cache_write_pos + t) * head_dim;
                            k_cache_s[cache_dst..cache_dst + head_dim]
                                .copy_from_slice(&qkv_s[t_base + d_model + h_off..t_base + d_model + h_off + head_dim]);
                            v_cache_s[cache_dst..cache_dst + head_dim]
                                .copy_from_slice(&qkv_s[t_base + 2 * d_model + h_off..t_base + 2 * d_model + h_off + head_dim]);
                        }
                    }
                }
            }
        }

        // Update cache length
        self.cache_len += new_seq_len;
        let total_kv_len = self.cache_len;

        // 3. Apply RoPE to Q and new K entries in cache (using pre-computed inv_freq)
        {
            let q_s = q.as_slice_mut().unwrap();
            let k_s = self.k_cache.as_slice_mut().unwrap();
            let cache_row_stride = Self::MAX_CONTEXT * head_dim;

            for t in 0..new_seq_len {
                let abs_pos = (position_offset + t) as f32;
                for i in 0..half_dim {
                    let freq = abs_pos * self.inv_freq[i];
                    let cos_val = freq.cos();
                    let sin_val = freq.sin();

                    for h in 0..num_heads {
                        // RoPE on Q
                        let q_idx = h * new_seq_len * head_dim + t * head_dim + 2 * i;
                        let q_even = q_s[q_idx];
                        let q_odd = q_s[q_idx + 1];
                        q_s[q_idx] = q_even * cos_val - q_odd * sin_val;
                        q_s[q_idx + 1] = q_even * sin_val + q_odd * cos_val;

                        // RoPE on K (in cache)
                        let k_idx = h * cache_row_stride + (cache_write_pos + t) * head_dim + 2 * i;
                        let k_even = k_s[k_idx];
                        let k_odd = k_s[k_idx + 1];
                        k_s[k_idx] = k_even * cos_val - k_odd * sin_val;
                        k_s[k_idx + 1] = k_even * sin_val + k_odd * cos_val;
                    }
                }
            }
        }

        // 4. Compute attention: Q @ K^T / sqrt(head_dim) using cache views
        let scale = 1.0 / (head_dim as f32).sqrt();
        let cache_row_stride = Self::MAX_CONTEXT * head_dim;

        // For small new_seq_len (typically 1-8), do attention inline with raw slices
        // to avoid allocating scores array via ndarray.
        // Reuse scratch_scores buffer to avoid per-query heap allocation.
        let mut attn_out = Array3::<f32>::zeros((num_heads, new_seq_len, head_dim));
        {
            let q_s = q.as_slice().unwrap();
            let k_s = self.k_cache.as_slice().unwrap();
            let v_s = self.v_cache.as_slice().unwrap();
            let out_s = attn_out.as_slice_mut().unwrap();

            // Causal offset: all cached positions precede new input
            let causal_offset = total_kv_len - new_seq_len;

            // Ensure scratch buffer is large enough (grows once, never shrinks)
            let max_attend = total_kv_len;
            if self.scratch_scores.len() < max_attend {
                self.scratch_scores.resize(max_attend, 0.0);
            }
            let scores_buf = &mut self.scratch_scores;

            for h in 0..num_heads {
                let q_h_base = h * new_seq_len * head_dim;
                let kv_h_base = h * cache_row_stride;
                let out_h_base = h * new_seq_len * head_dim;

                for qi in 0..new_seq_len {
                    let q_off = q_h_base + qi * head_dim;
                    let attend_len = (causal_offset + qi + 1).min(total_kv_len);

                    // Q @ K^T (dot products)
                    let mut max_val = f32::NEG_INFINITY;
                    for j in 0..attend_len {
                        let k_off = kv_h_base + j * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_s[q_off + d] * k_s[k_off + d];
                        }
                        let scaled = dot * scale;
                        scores_buf[j] = scaled;
                        if scaled > max_val {
                            max_val = scaled;
                        }
                    }

                    // Softmax
                    let mut sum = 0.0f32;
                    for j in 0..attend_len {
                        scores_buf[j] = (scores_buf[j] - max_val).exp();
                        sum += scores_buf[j];
                    }
                    if sum > 0.0 {
                        let inv_sum = 1.0 / sum;
                        for j in 0..attend_len {
                            scores_buf[j] *= inv_sum;
                        }
                    }

                    // Weighted sum: output[qi] = sum_j scores[j] * V[j]
                    let out_off = out_h_base + qi * head_dim;
                    // Zero out output for this query position (since attn_out is zeros, this is already done for first use)
                    for j in 0..attend_len {
                        let w = scores_buf[j];
                        if w > 0.0 {
                            let v_off = kv_h_base + j * head_dim;
                            for d in 0..head_dim {
                                out_s[out_off + d] += w * v_s[v_off + d];
                            }
                        }
                    }
                }
            }
        }

        // 5. Reshape (H, T, D) → (B, T, d_model) and output projection → (B, d_model, T)
        //    Fused into a single step to minimize intermediate allocations.
        let mut result_bct = Array3::<f32>::zeros((batch, d_model, new_seq_len));
        {
            let attn_s = attn_out.as_slice().unwrap();

            if new_seq_len <= 2 {
                // Streaming fast path: gather attention output into contiguous vec,
                // use ndarray dot (SIMD), scatter to BCT output.
                let dst = result_bct.as_slice_mut().unwrap();
                let mut attn_vec = Array1::<f32>::zeros(d_model);

                for t in 0..new_seq_len {
                    // Gather attention output for this timestep into contiguous array:
                    // attn_out is (H, T, D): attn_s[h * new_seq_len * head_dim + t * head_dim + d]
                    let av = attn_vec.as_slice_mut().unwrap();
                    for h in 0..num_heads {
                        let src_off = h * new_seq_len * head_dim + t * head_dim;
                        let dst_off = h * head_dim;
                        av[dst_off..dst_off + head_dim]
                            .copy_from_slice(&attn_s[src_off..src_off + head_dim]);
                    }

                    // o_proj @ attn_vec: SIMD-optimized dot
                    let proj_result = self.o_proj.dot(&attn_vec);
                    let r_s = proj_result.as_slice().unwrap();

                    // Write transposed: dst[d * new_seq_len + t]
                    for d in 0..d_model {
                        dst[d * new_seq_len + t] = r_s[d];
                    }
                }
            } else {
                // Batch path
                let mut output = Array2::<f32>::zeros((new_seq_len, d_model));
                let out_s = output.as_slice_mut().unwrap();
                for h in 0..num_heads {
                    let src_h_off = h * new_seq_len * head_dim;
                    let h_off = h * head_dim;
                    for t in 0..new_seq_len {
                        let src_off = src_h_off + t * head_dim;
                        let dst_off = t * d_model + h_off;
                        out_s[dst_off..dst_off + head_dim]
                            .copy_from_slice(&attn_s[src_off..src_off + head_dim]);
                    }
                }

                let proj_result = output.dot(&self.o_proj_t);
                let src = proj_result.as_slice().unwrap();
                let dst = result_bct.as_slice_mut().unwrap();
                for d in 0..d_model {
                    let d_off = d * new_seq_len;
                    for t in 0..new_seq_len {
                        dst[d_off + t] = src[t * d_model + d];
                    }
                }
            }
        }

        Tensor3::new(result_bct)
    }

    pub fn reset(&mut self) {
        self.cache_len = 0;
        // Zero out cache (optional, entries won't be read past cache_len)
    }
}

/// Feed-forward network with pre-transposed weights for efficiency.
#[derive(Clone, Debug)]
pub struct FeedForward {
    pub fc1: Array2<f32>,   // (dim_ff, d_model) — original weights
    pub fc2: Array2<f32>,   // (d_model, dim_ff) — original weights
    fc1_t: Array2<f32>,     // (d_model, dim_ff) — pre-transposed, contiguous
    fc2_t: Array2<f32>,     // (dim_ff, d_model) — pre-transposed, contiguous
    pub bias1: Option<Array1<f32>>,
    pub bias2: Option<Array1<f32>>,
    // Pre-allocated scratch buffer for streaming matvec (dim_ff)
    scratch_hidden: Vec<f32>,
}

impl FeedForward {
    pub fn new(
        fc1: Array2<f32>,
        fc2: Array2<f32>,
        bias1: Option<Array1<f32>>,
        bias2: Option<Array1<f32>>,
    ) -> Self {
        // Pre-transpose weights so matmuls use contiguous memory
        let fc1_t = fc1.t().to_owned();
        let fc2_t = fc2.t().to_owned();
        let dim_ff = fc1.shape()[0];
        Self { fc1, fc2, fc1_t, fc2_t, bias1, bias2, scratch_hidden: vec![0.0; dim_ff] }
    }

    pub fn forward(&mut self, x: &Tensor3) -> Tensor3 {
        let (batch, d_model, time) = x.shape();
        let out_dim = self.fc2.shape()[0];
        let dim_ff = self.fc1.shape()[0];

        // Output in (B, d_model, T) format directly to avoid final transpose
        let mut output = Array3::<f32>::zeros((batch, out_dim, time));

        for b in 0..batch {
            if time <= 2 {
                // Streaming path: extract contiguous x_vec, use ndarray dot (SIMD).
                // Re-uses scratch_hidden buffer for GELU to avoid per-timestep alloc.
                let src = x.data.as_slice().unwrap();
                let b_off = b * d_model * time;
                let hidden = &mut self.scratch_hidden;

                let mut x_vec = Array1::<f32>::zeros(d_model);

                for t in 0..time {
                    // Extract x_vec from (B, d_model, T) layout into contiguous buffer
                    let xv = x_vec.as_slice_mut().unwrap();
                    for d in 0..d_model {
                        xv[d] = src[b_off + d * time + t];
                    }

                    // fc1 @ x_vec: SIMD-optimized dot, copy result to scratch
                    let fc1_result = self.fc1.dot(&x_vec);
                    let fc1_s = fc1_result.as_slice().unwrap();
                    hidden[..dim_ff].copy_from_slice(fc1_s);

                    // Add bias1
                    if let Some(ref bias) = self.bias1 {
                        let bias_s = bias.as_slice().unwrap();
                        for j in 0..dim_ff {
                            hidden[j] += bias_s[j];
                        }
                    }

                    // GELU activation (on scratch buffer, no allocation)
                    const GELU_COEFF: f32 = 0.7978845608;
                    for v in hidden[..dim_ff].iter_mut() {
                        let x = *v;
                        *v = x * 0.5 * (1.0 + (GELU_COEFF * (x + 0.044715 * x * x * x)).tanh());
                    }

                    // fc2 @ hidden: use ndarray dot with view into scratch buffer
                    let h_view = ndarray::ArrayView1::from(&hidden[..dim_ff]);
                    let result = self.fc2.dot(&h_view);
                    let r_s = result.as_slice().unwrap();
                    let dst = output.as_slice_mut().unwrap();
                    let ob_off = b * out_dim * time;
                    if let Some(ref bias) = self.bias2 {
                        let bias_s = bias.as_slice().unwrap();
                        for d in 0..out_dim {
                            dst[ob_off + d * time + t] = r_s[d] + bias_s[d];
                        }
                    } else {
                        for d in 0..out_dim {
                            dst[ob_off + d * time + t] = r_s[d];
                        }
                    }
                }
            } else {
                // Batch path: transpose + GEMM
                let mut x_bt = Array2::<f32>::zeros((time, d_model));
                {
                    let src = x.data.as_slice().unwrap();
                    let dst = x_bt.as_slice_mut().unwrap();
                    let b_off = b * d_model * time;
                    for t in 0..time {
                        for d in 0..d_model {
                            dst[t * d_model + d] = src[b_off + d * time + t];
                        }
                    }
                }

                // Use pre-transposed weights for contiguous memory access
                let mut hidden = x_bt.dot(&self.fc1_t);

                if let Some(ref bias) = self.bias1 {
                    let bias_s = bias.as_slice().unwrap();
                    let h_s = hidden.as_slice_mut().unwrap();
                    for t in 0..time {
                        let t_off = t * dim_ff;
                        for j in 0..dim_ff {
                            h_s[t_off + j] += bias_s[j];
                        }
                    }
                }

                // GELU activation
                const GELU_COEFF: f32 = 0.7978845608;
                let h_slice = hidden.as_slice_mut().unwrap();
                for v in h_slice.iter_mut() {
                    let x = *v;
                    *v = x * 0.5 * (1.0 + (GELU_COEFF * (x + 0.044715 * x * x * x)).tanh());
                }

                // Use pre-transposed weights
                let result = hidden.dot(&self.fc2_t);

                // Transpose result (T, d_model) → output (d_model, T) directly
                {
                    let src = result.as_slice().unwrap();
                    let dst = output.as_slice_mut().unwrap();
                    let b_off = b * out_dim * time;
                    if let Some(ref bias) = self.bias2 {
                        let bias_s = bias.as_slice().unwrap();
                        for d in 0..out_dim {
                            let bias_val = bias_s[d];
                            let d_off = b_off + d * time;
                            for t in 0..time {
                                dst[d_off + t] = src[t * out_dim + d] + bias_val;
                            }
                        }
                    } else {
                        for d in 0..out_dim {
                            let d_off = b_off + d * time;
                            for t in 0..time {
                                dst[d_off + t] = src[t * out_dim + d];
                            }
                        }
                    }
                }
            }
        }

        Tensor3::new(output)
    }
}

/// Layer normalization.
#[derive(Clone, Debug)]
pub struct LayerNorm {
    pub weight: Array1<f32>,
    pub bias: Array1<f32>,
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(weight: Array1<f32>, bias: Array1<f32>, eps: f32) -> Self {
        Self { weight, bias, eps }
    }

    pub fn forward(&self, x: &Tensor3) -> Tensor3 {
        let (batch, channels, time) = x.shape();
        let src = x.data.as_slice().unwrap();
        let w = self.weight.as_slice().unwrap();
        let bias = self.bias.as_slice().unwrap();
        let inv_channels = 1.0 / channels as f32;
        let eps = self.eps;

        // Allocate output without cloning (just zeros, faster than clone for large arrays)
        let mut out = Array3::<f32>::zeros((batch, channels, time));
        let dst = out.as_slice_mut().unwrap();

        // Data layout: [b, c, t] → index = b * channels * time + c * time + t
        for b in 0..batch {
            let b_off = b * channels * time;
            for t in 0..time {
                // Mean over channels (stride = time between channel elements)
                let mut mean = 0.0f32;
                for c in 0..channels {
                    mean += src[b_off + c * time + t];
                }
                mean *= inv_channels;

                // Variance over channels
                let mut var = 0.0f32;
                for c in 0..channels {
                    let diff = src[b_off + c * time + t] - mean;
                    var += diff * diff;
                }
                var *= inv_channels;

                // Normalize and apply weight/bias
                let inv_std = 1.0 / (var + eps).sqrt();
                for c in 0..channels {
                    let idx = b_off + c * time + t;
                    dst[idx] = w[c] * (src[idx] - mean) * inv_std + bias[c];
                }
            }
        }

        Tensor3::new(out)
    }
}

/// Single transformer layer.
#[derive(Clone, Debug)]
pub struct TransformerLayer {
    pub attn: MultiHeadAttention,
    pub ff: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
    /// Per-dimension layer scale applied after attention: (d_model,)
    pub layer_scale_1: Option<Array1<f32>>,
    /// Per-dimension layer scale applied after feedforward: (d_model,)
    pub layer_scale_2: Option<Array1<f32>>,
}

impl TransformerLayer {
    pub fn forward(&mut self, x: &Tensor3) -> Tensor3 {
        // norm_first = true: norm -> attn -> scale -> residual -> norm -> ff -> scale -> residual
        let normed = self.norm1.forward(x);
        let mut attn_out = self.attn.forward(&normed);

        if let Some(ref scale) = self.layer_scale_1 {
            attn_out.scale_channels_inplace(scale);
        }
        attn_out.add_assign(x); // in-place residual: attn_out += x

        let normed = self.norm2.forward(&attn_out);
        let mut ff_out = self.ff.forward(&normed);

        if let Some(ref scale) = self.layer_scale_2 {
            ff_out.scale_channels_inplace(scale);
        }
        ff_out.add_assign(&attn_out); // in-place residual: ff_out += attn_out

        ff_out
    }

    /// Streaming step: incremental attention with KV cache.
    pub fn step(&mut self, x: &Tensor3, position_offset: usize) -> Tensor3 {
        let normed = self.norm1.forward(x);
        let mut attn_out = self.attn.step(&normed, position_offset);

        if let Some(ref scale) = self.layer_scale_1 {
            attn_out.scale_channels_inplace(scale);
        }
        attn_out.add_assign(x);

        let normed = self.norm2.forward(&attn_out);
        let mut ff_out = self.ff.forward(&normed);

        if let Some(ref scale) = self.layer_scale_2 {
            ff_out.scale_channels_inplace(scale);
        }
        ff_out.add_assign(&attn_out);

        ff_out
    }

    pub fn reset(&mut self) {
        self.attn.reset();
    }
}

/// Full transformer stack.
#[derive(Clone, Debug)]
pub struct Transformer {
    pub layers: Vec<TransformerLayer>,
    #[allow(dead_code)]
    pub d_model: usize,
    /// Current position for RoPE in streaming mode.
    pub position: usize,
}

impl Transformer {
    pub fn new(layers: Vec<TransformerLayer>, d_model: usize) -> Self {
        Self { layers, d_model, position: 0 }
    }

    /// Forward pass through all layers (non-streaming, full self-attention).
    pub fn forward(&mut self, x: &Tensor3) -> Tensor3 {
        if self.layers.is_empty() {
            return x.clone();
        }

        let mut x = self.layers[0].forward(x);
        for layer in &mut self.layers[1..] {
            x = layer.forward(&x);
        }
        x
    }

    /// Streaming step: process new timesteps using KV cache across all layers.
    pub fn step(&mut self, x: &Tensor3) -> Tensor3 {
        let (_, _, seq_len) = x.shape();
        let pos = self.position;

        let mut x = self.layers[0].step(x, pos);
        for layer in &mut self.layers[1..] {
            x = layer.step(&x, pos);
        }

        self.position += seq_len;
        x
    }

    /// Reset all layer states (clear KV cache and position).
    pub fn reset(&mut self) {
        self.position = 0;
        for layer in &mut self.layers {
            layer.reset();
        }
    }
}
