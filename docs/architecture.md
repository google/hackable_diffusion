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

This string specifies how a conditioning signal is injected into the backbone.

  * `adaptative_norm`: The conditioning embedding is used to modulate the scale
    and shift in an adaptive normalization layer (e.g., AdaLN).
  * `cross_attention`: The conditioning embedding is used as the key and value
    in a cross-attention layer, with the model's intermediate representation as
    the query.
  * `concatenate`: The conditioning is concatenated to the input of a layer or
    module.
  * `sum`: The conditioning is added to the input of a layer or module.

### Embedding Merge Functions

When multiple embeddings are directed to the same `ConditioningMechanism`,
a merge function defines how they are combined. Two frozen dataclasses are
provided in `conditioning_encoder.py`:

  * `SumEmbeddings()`: Embeddings are summed element-wise. This requires them
    to have the same shape.
  * `ConcatEmbeddings()`: Embeddings are concatenated along the feature axis.

The type alias `MergeEmbeddingsFn = Union[SumEmbeddings, ConcatEmbeddings]` is
available for type annotations.

### Resampling and Skip Connections

  * `DownsampleFn`: Callable classes for downsampling spatial feature maps, including
    `MaxPoolDownsample` and `AvgPoolDownsample`.
  * `UpsampleFn`: Callable classes for upsampling, including `ImageResizeUpsample`.
  * `SkipConnectionFn`: Callable classes for adding skip connections in residual blocks,
    such as `UnnormalizedAddSkip` and `NormalizedAddSkip`.

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
  * `downsample_fn` / `upsample_fn`: The resampling callable instances to use.
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
    'adaptive_norm': adaptive_norm_emb,
    'cross_attention': cross_attention_emb,
}

unet = Unet(
    base_channels=128,
    channels_multiplier=(1, 2, 4),
    num_residual_blocks=(2, 2, 2),
    downsample_fn=hd.architecture.AvgPoolDownsample(),
    upsample_fn=hd.architecture.ImageResizeUpsample(resize_method='bilinear'),
    self_attention_bool=(False, True, True),
    cross_attention_bool=(False, True, True),
    attention_heads_spec=hd.architecture.AttentionHeadsSpec(num_heads=8),
    attention_normalize_qk=True,
    attention_use_rope=False,
    uncond_norm_strategy=hd.architecture.GroupNormStrategy(num_groups=32),
    cond_norm_strategy=hd.architecture.ConditionalGroupNormStrategy(
        num_groups=32,
        use_shift=True,
    ),
    activation='silu',
    skip_connection_fn=hd.architecture.NormalizedAddSkip(),
)

variables = unet.init(key, x, conditioning_embeddings, is_training=False)
output = unet.apply(variables, x, conditioning_embeddings, is_training=False)

print(f"Output shape: {output.shape}")
# Output shape: (1, 64, 64, 3)
```

### `DiT`

(`lib/architecture/dit.py`)

The `DiT` class implements a **Diffusion Transformer** backbone based on
<https://arxiv.org/abs/2212.09748>. It uses adaptive layer norm zero
(adaLN-Zero) as the conditioning mechanism. The architecture consists of
repeated transformer blocks with optional encoder/decoder and absolute
positional encoding.

Key parameters:

  * `num_blocks`: Number of DiT blocks.
*   `block`: A DiT block module (e.g., `DiTBlockAdaLNZero`).
    *   `encoder`: Optional encoder (e.g., `Patchify` for image inputs).
    *   `decoder`: Optional decoder (e.g., `DePatchify` for image outputs).
    *   `absolute_posenc`: Optional positional encoding module.
    *   `use_padding_mask`: Whether to mask out padding tokens (for tokenized
        inputs).

The `DiT` expects an `ADAPTIVE_NORM` conditioning embedding. The `mnist_dit`
notebook demonstrates its usage.

## Diffusion Network

(`lib/diffusion_network.py`)

The **`DiffusionNetwork`** class is the primary entry point for constructing a
complete diffusion model. It composes a backbone (e.g., `Unet` or `DiT`) with a
`ConditioningEncoder` into a single Flax module that conforms to the
`BaseDiffusionNetwork` protocol.

  * **`DiffusionNetwork`**: Single-modal model. Takes `(time, xt,
    conditioning)` and internally runs the conditioning encoder, applies any
    input/time rescaling, and calls the backbone.
  * **`MultiModalDiffusionNetwork`**: Generalizes `DiffusionNetwork` to
    multi-modal PyTree data, allowing different prediction types and data
    dtypes per leaf.
  * **`SelfConditioningDiffusionNetwork`**: Adds self-conditioning, where the
    model receives its own previous prediction as an additional input.

These classes also support `InputRescaler` and `TimeRescaler` for
schedule-dependent input preprocessing (e.g., EDM preconditioning).

### `ConditionalMLP`

(`lib/architecture/mlp.py`)

For simpler, non-image data, a `ConditionalMLP` backbone is provided. It
processes the input `x`, combines it with conditioning embeddings, and passes it
through a series of dense layers. This module is mainly used for testing
purposes.

### `RiemannianConditionalBackbone`

(`lib/architecture/riemannian.py`)

A specialized wrapper for any `ConditionalBackbone` that handles Riemannian
manifold constraints. Its primary role is to ensure that the model's output
`velocity` is a valid **tangent vector** at the point `xt`.

This is achieved by applying the manifold's **`project`** operator to the raw
output of the underlying backbone.

#### Riemannian Projections

Each manifold defines a `project(x, v)` method that ensures the output $$v$$ is
a valid tangent vector at point $$x$$.

*   **Sphere ($$S^d$$)**: The projection is $$v_{\text{tangent}} = v - \langle
    x, v \rangle x$$, which removes the component of $$v$$ parallel to $$x$$.
*   **SO(3)**: The projection maps a $$3 \times 3$$ matrix $$V$$ to the tangent
    space $$T_R SO(3)$$ by computing the skew-symmetric part of the relative
    velocity in the Lie algebra: $$R \cdot \text{skew}(R^T V)$$, where
    $$\text{skew}(\Omega) = 0.5(\Omega - \Omega^T)$$.

By wrapping a standard neural network (e.g., a UNet) in this backbone, we can
learn complex velocity fields on manifolds using standard architectures.

#### Example Usage

```python
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.architecture.riemannian import RiemannianConditionalBackbone
from hackable_diffusion.lib.architecture.mlp import ConditionalMLP

# 1. Choose a manifold
manifold = manifolds.Sphere()

# 2. Create a standard backbone
mlp = ConditionalMLP(num_features=256, num_layers=3)

# 3. Wrap it in a RiemannianConditionalBackbone
model = RiemannianConditionalBackbone(
    backbone=mlp,
    manifold=manifold,
)
```

The conditioning mechanism is simpler here, limited to `SUM` or `CONCATENATE` of
the conditioning embeddings with the intermediate representation of `x`.

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
    ConditioningEncoder, SinusoidalTimeEmbedder, LabelEmbedder,
    SumEmbeddings)

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
    merge_embeddings_fn=SumEmbeddings(),
    conditioning_rules={
        'time': 'adaptive_norm',
        'label_adanorm': 'adaptive_norm',
        'label_xattn': 'cross_attention',
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
adanorm_emb = output_embeddings['adaptive_norm']
xattn_emb = output_embeddings['cross_attention']

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
    `merge_embeddings_fn`. If summing, their feature dimensions must match.
