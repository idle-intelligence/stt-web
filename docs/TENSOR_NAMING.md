# GGUF Tensor Naming Convention for STT

This document describes the tensor naming scheme used in the Q4 GGUF quantized weights for the `kyutai/stt-1b-en_fr` model.

## Overview

The model is a decoder-only transformer with:
- **Main transformer**: 16 layers, 2048 dim, 16 heads
- **Depformer**: 6 layers, 1024 dim, 16 heads (per-step conditioning on audio)
- **Audio embeddings**: 32 codebook embeddings (Mimi codec tokens)
- **Text vocabulary**: 8000 tokens

## Model Architecture Parameters

```python
dim = 2048                    # Main transformer dimension
num_layers = 16               # Main transformer layers
num_heads = 16                # Attention heads
head_dim = dim // num_heads   # = 128
hidden_scale = 4.125          # FFN expansion ratio
hidden_dim = int(dim * hidden_scale)  # = 8448

depformer_dim = 1024          # Depformer dimension
depformer_num_layers = 6      # Depformer layers
depformer_num_heads = 16      # Depformer attention heads

n_q = 32                      # Number of audio codebooks
card = 2048                   # Codebook size
text_card = 8000              # Text vocabulary size

context = 750                 # Max context length
max_period = 100000.0         # RoPE base frequency
```

## Tensor Names

### Token Embeddings

```
tok_embeddings.weight         [text_card, dim] = [8000, 2048]
```

### Main Transformer Layers (×16)

For each layer `i` in `0..15`:

**Attention:**
```
layers.{i}.attention_norm.weight         [dim] = [2048]
layers.{i}.attention.wq.weight           [dim, num_heads * head_dim] = [2048, 2048]
layers.{i}.attention.wk.weight           [dim, num_heads * head_dim] = [2048, 2048]
layers.{i}.attention.wv.weight           [dim, num_heads * head_dim] = [2048, 2048]
layers.{i}.attention.wo.weight           [num_heads * head_dim, dim] = [2048, 2048]
```

**Feed-Forward (SiLU gating):**
```
layers.{i}.ffn_norm.weight               [dim] = [2048]
layers.{i}.feed_forward.w1.weight        [dim, hidden_dim] = [2048, 8448]
layers.{i}.feed_forward.w2.weight        [hidden_dim, dim] = [8448, 2048]
layers.{i}.feed_forward.w3.weight        [dim, hidden_dim] = [2048, 8448]
```

**SiLU FFN formula:** `w2(silu(w1(x)) * w3(x))`

### Depformer Layers (×6)

The depformer conditions each text token on the current audio frame (32 codebook tokens). It uses per-step weights (one linear layer per time step).

For each layer `i` in `0..5`:

**Per-step linear projection:**
```
depformer.layers.{i}.linear.weight       [n_q * card, depformer_dim] = [65536, 1024]
```

**Self-attention:**
```
depformer.layers.{i}.sa.in_proj_weight   [3 * depformer_dim, depformer_dim] = [3072, 1024]
depformer.layers.{i}.sa.out_proj.weight  [depformer_dim, depformer_dim] = [1024, 1024]
```

**Output normalization:**
```
depformer.layers.{i}.out_norm.weight     [depformer_dim] = [1024]
```

### Audio Embeddings (×32)

One embedding table per codebook:

```
emb.{i}.weight                [card, depformer_dim] = [2048, 1024]
```

For `i` in `0..31` (32 codebooks total).

### Output Projection

```
output_norm.weight            [dim] = [2048]
output.weight                 [dim, text_card] = [2048, 8000]
```

## Quantization Strategy

**Quantized to Q4_0** (4-bit, block size 32):
- All `.weight` tensors in transformer layers
- All `.weight` tensors in depformer layers
- Token embeddings (`tok_embeddings.weight`)
- Audio embeddings (`emb.{i}.weight`)
- Output projection (`output.weight`)

**Kept as F32**:
- All normalization layers (`*norm.weight`)

## GGUF Metadata

The GGUF file includes the following metadata keys:

```python
"general.architecture": "kyutai-stt"
"general.name": "stt-1b-en_fr"
"stt.context_length": 750
"stt.embedding_length": 2048
"stt.block_count": 16
"stt.feed_forward_length": 8448
"stt.attention.head_count": 16
"stt.depformer.embedding_length": 1024
"stt.depformer.block_count": 6
"stt.depformer.attention.head_count": 16
"stt.vocab_size": 8000
"stt.rope.freq_base": 100000.0
```

## Tensor Count

Total tensors in GGUF file:
- Token embeddings: 1
- Main transformer layers: 16 × 9 = 144
- Depformer layers: 6 × 4 = 24
- Audio embeddings: 32
- Output: 2

**Total: ~203 tensors**

## Notes

1. **Dimension ordering**: GGUF reverses dimension order compared to PyTorch. A PyTorch tensor with shape `[A, B]` becomes `[B, A]` in GGUF's dimension list.

2. **Weight transpose**: PyTorch Linear layers store weights as `[out_features, in_features]`, but Burn expects `[in_features, out_features]`. The Rust loader must transpose when building Linear modules.

3. **RoPE**: Rotary positional embeddings are applied in the model, not stored as weights. Only the base frequency (theta) is in metadata.

4. **KV cache**: Not stored in GGUF. Allocated dynamically during inference.

5. **Delayed streams**: The text stream is delayed by 6 frames (480ms at 12.5 Hz) relative to audio. This is a runtime decoding parameter, not a weight.
