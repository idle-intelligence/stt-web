//! Key remapping for loading the full Mimi codec model into mimi-rs.
//!
//! The full Mimi model (e.g. `kyutai/mimi`) uses different tensor naming than
//! the pocket-tts model that mimi-rs was originally built for. This module
//! remaps the safetensors keys and combines separate Q/K/V attention projections
//! into the combined `in_proj` format that mimi-rs expects.
//!
//! Naming differences:
//!   - SEANet: `encoder.layers.{n}` → `encoder.model.{n}`
//!   - Transformer: separate `q_proj/k_proj/v_proj/o_proj` → combined `in_proj` + `out_proj`
//!   - Norms: `input_layernorm` → `norm1`, `post_attention_layernorm` → `norm2`
//!   - FFN: `mlp.fc1/fc2` → `linear1/linear2`
//!   - Layer scales: `self_attn_layer_scale` → `layer_scale_1`, `mlp_layer_scale` → `layer_scale_2`
//!   - ProjectedTransformer wrapper adds `.transformer.` in paths

use std::collections::HashMap;

/// Remap a single key from full-Mimi naming to mimi-rs naming.
///
/// Returns `None` for keys that should be skipped (decoder-only keys when
/// loading encoder-only, or unused keys).
fn remap_key(name: &str) -> Option<String> {
    let mut name = name.to_string();

    // SEANet: encoder.layers.{n} → encoder.model.{n}
    // (decoder uses same pattern)
    name = name.replace("encoder.layers.", "encoder.model.");
    name = name.replace("decoder.layers.", "decoder.model.");

    // ProjectedTransformer wraps StreamingTransformer, adding a .transformer. prefix
    // But only for the inner transformer layers, not the input/output projections.
    if name.starts_with("encoder_transformer.layers.") {
        name = name.replace("encoder_transformer.layers.", "encoder_transformer.transformer.layers.");
    }
    if name.starts_with("decoder_transformer.layers.") {
        name = name.replace("decoder_transformer.layers.", "decoder_transformer.transformer.layers.");
    }

    // Attention: o_proj → out_proj
    name = name.replace(".self_attn.o_proj.", ".self_attn.out_proj.");

    // Norms
    name = name.replace(".input_layernorm.", ".norm1.");
    name = name.replace(".post_attention_layernorm.", ".norm2.");

    // FFN
    name = name.replace(".mlp.fc1.", ".linear1.");
    name = name.replace(".mlp.fc2.", ".linear2.");

    // Layer scales
    name = name.replace(".self_attn_layer_scale.", ".layer_scale_1.");
    name = name.replace(".mlp_layer_scale.", ".layer_scale_2.");

    // Downsample/upsample: the ConvDownsample1d wraps StreamingConv1d which wraps CausalConv1d,
    // each adding a `.conv` prefix. Full Mimi has `downsample.conv.*` while mimi-rs expects
    // `downsample.conv.conv.*`.
    if name.starts_with("downsample.conv.") && !name.starts_with("downsample.conv.conv.") {
        name = name.replacen("downsample.conv.", "downsample.conv.conv.", 1);
    }
    if name.starts_with("upsample.convtr.") && !name.starts_with("upsample.convtr.convtr.") {
        name = name.replacen("upsample.convtr.", "upsample.convtr.convtr.", 1);
    }

    // Q/K/V projections: skip individual q_proj/k_proj/v_proj — they get combined into in_proj
    // by combine_qkv_projections() separately.
    if name.contains(".self_attn.q_proj.")
        || name.contains(".self_attn.k_proj.")
        || name.contains(".self_attn.v_proj.")
    {
        return None;
    }

    Some(name)
}

/// Remap full-Mimi safetensors to mimi-rs naming convention.
///
/// This handles:
/// 1. Key name remapping (SEANet, transformer, norms, FFN, layer scales)
/// 2. Combining separate Q/K/V projections into a single `in_proj` weight
/// 3. BF16 → BF16 passthrough (dtype conversion handled by candle VarBuilder)
pub fn remap_mimi_weights(buffer: &[u8]) -> Vec<u8> {
    let st = match safetensors::SafeTensors::deserialize(buffer) {
        Ok(st) => st,
        Err(_) => return buffer.to_vec(),
    };

    let mut views: Vec<(String, OwnedView)> = Vec::new();

    // First pass: collect Q/K/V weights for combining
    let qkv_weights = collect_qkv_weights(&st);

    // Second pass: remap all other weights
    for (name, tensor) in st.iter() {
        let remapped = match remap_key(name) {
            Some(r) => r,
            None => continue, // Skip Q/K/V (handled separately) and unused keys
        };

        views.push((
            remapped,
            OwnedView {
                data: tensor.data().to_vec(),
                shape: tensor.shape().to_vec(),
                dtype: tensor.dtype(),
            },
        ));
    }

    // Third pass: add combined in_proj weights
    for (prefix, qkv) in &qkv_weights {
        let combined_data = combine_weight_data(&qkv.q_data, &qkv.k_data, &qkv.v_data);
        let out_dim = qkv.q_shape[0]; // Each projection is [embed_dim, embed_dim]
        let in_dim = qkv.q_shape[1];
        let combined_shape = vec![3 * out_dim, in_dim];

        views.push((
            format!("{prefix}.self_attn.in_proj.weight"),
            OwnedView {
                data: combined_data,
                shape: combined_shape,
                dtype: qkv.dtype,
            },
        ));
    }

    let view_refs: Vec<(&str, safetensors::tensor::TensorView<'_>)> = views
        .iter()
        .map(|(name, v)| {
            (
                name.as_str(),
                safetensors::tensor::TensorView::new(v.dtype, v.shape.clone(), &v.data)
                    .expect("invalid tensor view"),
            )
        })
        .collect();

    safetensors::tensor::serialize(view_refs, &None).expect("serialization failed")
}

struct QkvWeights {
    q_data: Vec<u8>,
    q_shape: Vec<usize>,
    k_data: Vec<u8>,
    v_data: Vec<u8>,
    dtype: safetensors::Dtype,
}

/// Collect Q/K/V weight data for each transformer layer, keyed by remapped prefix.
fn collect_qkv_weights(st: &safetensors::SafeTensors) -> HashMap<String, QkvWeights> {
    let mut qkv_map: HashMap<String, QkvWeights> = HashMap::new();

    for (name, tensor) in st.iter() {
        let (proj_kind, layer_prefix) = if let Some(rest) = name.strip_suffix(".weight") {
            if rest.ends_with(".self_attn.q_proj") {
                ("q", rest.strip_suffix(".self_attn.q_proj").unwrap())
            } else if rest.ends_with(".self_attn.k_proj") {
                ("k", rest.strip_suffix(".self_attn.k_proj").unwrap())
            } else if rest.ends_with(".self_attn.v_proj") {
                ("v", rest.strip_suffix(".self_attn.v_proj").unwrap())
            } else {
                continue;
            }
        } else {
            continue;
        };

        // Remap the layer prefix (e.g. encoder_transformer.layers.0 →
        // encoder_transformer.transformer.layers.0)
        let remapped_prefix = if layer_prefix.starts_with("encoder_transformer.layers.") {
            layer_prefix.replace(
                "encoder_transformer.layers.",
                "encoder_transformer.transformer.layers.",
            )
        } else if layer_prefix.starts_with("decoder_transformer.layers.") {
            layer_prefix.replace(
                "decoder_transformer.layers.",
                "decoder_transformer.transformer.layers.",
            )
        } else {
            layer_prefix.to_string()
        };

        let entry = qkv_map.entry(remapped_prefix).or_insert_with(|| QkvWeights {
            q_data: Vec::new(),
            q_shape: Vec::new(),
            k_data: Vec::new(),
            v_data: Vec::new(),
            dtype: tensor.dtype(),
        });

        match proj_kind {
            "q" => {
                entry.q_data = tensor.data().to_vec();
                entry.q_shape = tensor.shape().to_vec();
            }
            "k" => {
                entry.k_data = tensor.data().to_vec();
            }
            "v" => {
                entry.v_data = tensor.data().to_vec();
            }
            _ => unreachable!(),
        }
    }

    qkv_map
}

/// Concatenate Q, K, V weight bytes into a single in_proj weight: [3*out, in].
fn combine_weight_data(q: &[u8], k: &[u8], v: &[u8]) -> Vec<u8> {
    let mut combined = Vec::with_capacity(q.len() + k.len() + v.len());
    combined.extend_from_slice(q);
    combined.extend_from_slice(k);
    combined.extend_from_slice(v);
    combined
}

struct OwnedView {
    data: Vec<u8>,
    shape: Vec<usize>,
    dtype: safetensors::Dtype,
}
