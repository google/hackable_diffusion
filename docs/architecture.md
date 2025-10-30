# Architecture

This document provides a detailed overview of the neural network architectures
used in the Hackable Diffusion library. The core components are designed to be
flexible and customizable for various diffusion model setups.

The modules related to architecture are located in `lib/architecture/`.

[TOC]

## Overview

The `architecture` sub-library provides building blocks for constructing
diffusion models, with a primary focus on conditional models. The key components
include:

  * **Backbones**: U-Net and MLP architectures that form the main body of the
    denoising network.
  * **Attention Mechanisms**: Multi-head attention for self-attention and
    cross-attention, with support for modern techniques like RoPE.
  * **Conditioning Encoders**: A system to encode and combine various forms of
    conditioning information (e.g., time, class labels, text embeddings).
  * **Building Blocks**: Lower-level components like residual blocks,
    normalization layers, and embedding layers.
  * **Typing and Utilities**: A set of common types and utility functions to
    ensure consistency and clarity.

## Common Types (`arch_typing.py`)

Several important enums and type definitions are centralized in `arch_typing.py`
to standardize the architecture's configuration.

### `ConditioningMechanism`

This enum specifies how a conditioning signal is injected into the backbone.

  * `ADAPTIVE_NORM`: The conditioning embedding is used to modulate the scale
    and shift in an adaptive normalization layer (e.g., AdaLN).
  * `CROSS_ATTENTION`: The conditioning embedding is used as the key and value
    in a cross-attention layer, with the model's intermediate representation as
    the query.
  * `CONCATENATE`: The conditioning is concatenated to the input of a layer or
    module.
  * `SUM`: The conditioning is added to the input of a layer or module.

### `EmbeddingMergeMethod`

When multiple embeddings are directed to the same `ConditioningMechanism`, this
enum defines how they are combined.

  * `SUM`: Embeddings are summed element-wise. This requires them to have the
    same shape.
  * `CONCAT`: Embeddings are concatenated along the feature axis.

### Resampling and Skip Connections

  * `DownsampleType`: Methods for downsampling spatial feature maps, including
    `MAX_POOL` and `AVG_POOL`.
  * `UpsampleType`: Methods for upsampling, including `NEAREST` and `BILINEAR`.
  * `SkipConnectionMethod`: Defines how skip connections in residual blocks are
    handled, such as `UNNORMALIZED_ADD` and `NORMALIZED_ADD`.

## Backbones

Backbones are the main neural network architectures that perform the denoising
task.

### `Unet`

(`lib/architecture/unet.py`)

The `Unet` class implements a standard U-Net architecture, enhanced with
conditioning capabilities. It's the most common backbone for diffusion models on
images.

The architecture is highly configurable. Key parameters include:

  * `base_channels`: The number of channels in the first level of the U-Net.
  * `channels_multiplier`: A sequence of integers multiplying `base_channels` at
    each resolution level.
  * `num_residual_blocks`: Number of residual blocks at each resolution level.
  * `downsample_method` / `upsample_method`: The resampling methods to use.
  * `self_attention_bool`: A sequence of booleans indicating whether to use
    self-attention at each resolution.
  * `cross_attention_bool`: A sequence of booleans for using cross-attention.
  * `attention_num_heads`, `attention_head_dim`: Configuration for attention
    layers.

The `Unet` expects conditioning embeddings as a dictionary, keyed by
`ConditioningMechanism`.

#### Example Usage

```python
import jax
import jax.numpy as jnp
import flax.linen as nn
from hackable_diffusion.lib.architecture.unet import Unet
from hackable_diffusion.lib.architecture.arch_typing import ConditioningMechanism

key = jax.random.PRNGKey(0)
input_shape = (1, 64, 64, 3)
x = jnp.ones(input_shape)
adaptive_norm_emb = jnp.ones((1, 128))
cross_attention_emb = jnp.ones((1, 10, 256)) # 10 tokens, 256 dim

conditioning_embeddings = {
    ConditioningMechanism.ADAPTIVE_NORM: adaptive_norm_emb,
    ConditioningMechanism.CROSS_ATTENTION: cross_attention_emb,
}

unet = Unet(
    base_channels=128,
    channels_multiplier=(1, 2, 4),
    num_residual_blocks=(2, 2, 2),
    downsample_method='avg_pool',
    upsample_method='bilinear',
    self_attention_bool=(False, True, True),
    cross_attention_bool=(False, True, True),
    attention_num_heads=8,
    attention_head_dim=-1, # auto-inferred
    attention_normalize_qk=True,
    attention_use_rope=False,
    normalization_type='group_norm',
    normalization_num_groups=32,
    activation='silu',
    skip_connection_method='normalized_add',
)

variables = unet.init(key, x, conditioning_embeddings, is_training=False)
output = unet.apply(variables, x, conditioning_embeddings, is_training=False)

print(f"Output shape: {output.shape}")
# Output shape: (1, 64, 64, 3)
```

### `ConditionalMLP`

(`lib/architecture/mlp.py`)

For simpler, non-image data, a `ConditionalMLP` backbone is provided. It
processes the input `x`, combines it with conditioning embeddings, and passes it
through a series of dense layers. This module is mainly use for testing
purposes.

The conditioning mechanism is simpler here, limited to `SUM` or `CONCATENATE` of
the conditioning embeddings with the intermediate representation of `x`.

## Attention

### `MultiHeadAttention`

(`lib/architecture/attention.py`)

This module provides a flexible multi-head attention implementation that can
function as either self-attention or cross-attention.

  * **Self-attention**: If only input `x` is provided, it performs
    self-attention on `x`.
  * **Cross-attention**: If a conditioning tensor `c` is provided, it performs
    cross-attention where `x` provides the queries and `c` provides keys and
    values.

It supports modern features like:

  * **Rotary Positional Embeddings (RoPE)** via `use_rope=True`.
  * **QK Normalization** (`normalize_qk=True`) for stabilizing attention.

## Conditioning

The conditioning system is designed to handle multiple sources of conditioning
and route them to different parts of the model.

### `ConditioningEncoder`

(`lib/architecture/conditioning_encoder.py`)

This is the central module for managing conditioning. It takes raw time and a
dictionary of conditioning signals and produces a dictionary of embeddings ready
to be consumed by the backbone.

Its configuration is defined by three main arguments:

1.  `time_embedder`: A module to encode the diffusion timestep.
    `SinusoidalTimeEmbedder` is a common choice.
2.  `conditioning_embedders`: A dictionary of modules to encode various other
    conditioning signals (e.g., class labels, text). Examples include
    `LabelEmbedder` and `LinearEmbedder`.
3.  `conditioning_rules`: A dictionary that maps each embedding (time and
    others) to a `ConditioningMechanism`. This is how you tell the model *how*
    to use each piece of conditioning.

During training, `conditioning_dropout_rate` can be used to implement
classifier-free guidance by randomly zeroing out conditioning embeddings.

#### Example Usage

This example sets up a conditioning system for time and a class label. Time is
used for adaptive normalization, while the class label is split to be used for
both adaptive normalization and cross-attention.

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.architecture.conditioning_encoder import (
    ConditioningEncoder, SinusoidalTimeEmbedder, LabelEmbedder)
from hackable_diffusion.lib.architecture.arch_typing import (
    ConditioningMechanism, EmbeddingMergeMethod)

key = jax.random.PRNGKey(0)

# 1. Define embedders for time and conditioning signals
time_embedder = SinusoidalTimeEmbedder(
    activation='silu', embedding_dim=64, num_features=256)

# We define two embedders for the same label. One for AdaNorm, one for X-Attn.
label_embedder_adanorm = LabelEmbedder(
    num_classes=10, num_features=256, conditioning_key='label')
label_embedder_xattn = LabelEmbedder(
    num_classes=10, num_features=512, conditioning_key='label')


# 2. Instantiate the ConditioningEncoder
conditioning_encoder = ConditioningEncoder(
    time_embedder=time_embedder,
    conditioning_embedders={
        'label_adanorm': label_embedder_adanorm,
        'label_xattn': label_embedder_xattn,
    },
    embedding_merging_method=EmbeddingMergeMethod.SUM,
    conditioning_rules={
        'time': ConditioningMechanism.ADAPTIVE_NORM,
        'label_adanorm': ConditioningMechanism.ADAPTIVE_NORM,
        'label_xattn': ConditioningMechanism.CROSS_ATTENTION,
    },
    conditioning_dropout_rate=0.1,
)


# 3. Prepare inputs
time = jnp.array([0.5, 0.2])
conditioning_dict = {'label': jnp.array([3, 7])}

# 4. Apply the encoder
variables = conditioning_encoder.init(
    key, time, conditioning_dict, is_training=True
)
output_embeddings = conditioning_encoder.apply(
    variables, time, conditioning_dict, is_training=True,
    rngs={'dropout': key}
)

# 5. Inspect the output
adanorm_emb = output_embeddings[ConditioningMechanism.ADAPTIVE_NORM]
xattn_emb = output_embeddings[ConditioningMechanism.CROSS_ATTENTION]

# The adanorm embedding is the sum of time and label_adanorm embeddings
print(f"Adaptive Norm embedding shape: {adanorm_emb.shape}")
# Adaptive Norm embedding shape: (2, 256)

print(f"Cross Attention embedding shape: {xattn_emb.shape}")
# Cross Attention embedding shape: (2, 512)
```

**Assumptions**:

  * The keys in `conditioning_embedders` are user-defined names.
  * The `conditioning_key` within an embedder (e.g., `LabelEmbedder`) refers to
    a key in the `conditioning` dictionary passed to the `__call__` method.
  * The `conditioning_rules` must have keys matching the keys of
    `conditioning_embedders` plus a key for `'time'`.
  * When multiple embeddings are routed to the same mechanism (like `time` and
    `label_adanorm` both to `ADAPTIVE_NORM`), they are combined using
    `embedding_merging_method`. If summing, their feature dimensions must match.
