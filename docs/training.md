# Training

This document details the loss functions used for training diffusion models in
the Hackable Diffusion library. The loss modules are designed with flexibility
to support various training objectives and weighting schemes.

The modules related to loss functions are located in `lib/training/`.

[TOC]

## Overview

The goal of a diffusion model is to learn to reverse the corruption process. The
loss function measures how well the model's prediction at a given timestep `t`
matches the true target. The library provides a modular system for defining
these loss functions, mirroring the structure of the `corruption` modules.

The main components are:

  * **`DiffusionLoss` Protocol**: An interface for all diffusion loss functions.
  * **Gaussian Losses**: Highly configurable loss functions for continuous data,
    based on Mean Squared Error (MSE) with advanced weighting.
  * **Discrete Losses**: Loss functions for discrete data, based on
    Cross-Entropy.

## `DiffusionLoss` Protocol

(`lib/hd_api.py`)

This protocol defines the basic interface for a diffusion loss. It's a callable
that takes the model's predictions, the ground truth targets, and the time, and
returns a batch-wise loss.

```python
def __call__(
    self,
    preds: TargetInfo,
    targets: TargetInfo,
    time: TimeArray,
) -> LossOutput:
  ...
```

  * `preds`: The dictionary of predictions from the model (e.g., `{'epsilon':
    ...}`).
  * `targets`: The dictionary of ground truth targets from the corruption
    process (e.g., `{'epsilon': ..., 'x0': ...}`).
  * `time`: The timesteps for the batch.
  * Returns: A tensor of losses for each item in the batch (shape `[B,]`). This
    allows for further processing (like masking) before taking the final mean.

## Gaussian Losses

(`lib/training/gaussian_loss.py`)

For continuous data corrupted by a `GaussianProcess`, the losses are based on a
weighted Mean Squared Error. The core logic is encapsulated in a powerful
function `compute_continuous_diffusion_loss`.

### `compute_continuous_diffusion_loss`

This function calculates `weight * (prediction - target)^2`. Its flexibility
comes from how the `weight` is constructed. The main arguments controlling this
are:

  * `prediction_type`: This tells the loss function which prediction to use from
    the `preds` dictionary (e.g., `'epsilon'`). The model must be configured to
    output this quantity. If we do not specify it (i.e, `None`) then we would
    revert to the prediction type used in the prediction (only if that
    prediction is unambiguous).
  * `loss_type`: This defines the actual parameterisation used in the loss. For
    example, you could have a model that predicts `epsilon`
    (`prediction_type='epsilon'`) but choose to compute a loss equivalent to an
    MSE on `x0` (`loss_type='x0'`). The function will automatically apply the
    correct scaling factor to the loss to make this equivalence hold. If
    `loss_type` is `None`, it defaults to `prediction_type`.
  * `convert_to_logsnr_schedule`: If `True`, it applies a weighting of
    `-d(logsnr)/dt`. This makes the loss integral independent of the specific
    time parameterization of the noise schedule, which is useful for fair
    comparisons between models trained with different schedules.
  * `weight_fn`: An optional, arbitrary function of time that applies an
    additional weighting to the loss.

This design allows for the implementation of many different diffusion loss
formulations from recent literature within a single framework.

### `SiD2Loss`

This is a concrete implementation of the sigmoid weighting from Simpler
Diffusion (SiD2) <https://arxiv.org/abs/2410.19324>. It computes an MSE in `x0`
space, converts it to logSNR time, and applies an additional sigmoid-based
weighting. This has been shown to improve training stability and performance.

#### Example

```python
import jax.numpy as jnp
from hackable_diffusion.lib.training.gaussian_loss import SiD2Loss
from hackable_diffusion.lib.corruption.schedules import CosineSchedule

# 1. Instantiate the loss. It requires the noise schedule.
schedule = CosineSchedule()
sid_loss = SiD2Loss(
    schedule=schedule,
    prediction_type='epsilon', # Our model predicts epsilon
    bias=0.0
)

# 2. Fake model predictions and targets
preds = {'epsilon': jnp.ones((1, 32, 32, 3))}
targets = {'epsilon': jnp.zeros((1, 32, 32, 3))}
time = jnp.array([0.5])

# 3. Compute loss
loss_batch = sid_loss(preds, targets, time)
loss_scalar = jnp.mean(loss_batch)

print(f"Batch loss shape: {loss_batch.shape}")
print(f"Scalar loss: {loss_scalar}")
```

## Discrete Losses

(`lib/training/discrete_loss.py`)

For discrete data, the training objective is typically to predict the
probability distribution of the original token. This is based on
<https://arxiv.org/abs/2406.04329>. The core logic is encapsulated in
`compute_discrete_diffusion_loss`.

### `compute_discrete_diffusion_loss`

This function calculates a weighted cross-entropy loss between the model's
predicted logits and the true underlying token for data from a
`CategoricalProcess`.

The cross-entropy loss is computed using
`optax.softmax_cross_entropy_with_integer_labels`. Its flexibility comes from
`weight_fn`, an optional, arbitrary function of time that applies a weighting to
the loss.

A key option is `use_mask`. If `True`, it will compute the loss only on the
tokens for which the mask (given by `mask_key`) is `True`. By default
`mask_key=is_corrupted`, which corresponds to the tokens which were corrupted by
the process, focusing the loss only on the tokens the model needs to predict.
The existing `dLLM` library is also using this strategy.

### `MD4Loss`

This loss function implements loss from "Masked Discrete Diffusion in Image
Tokenizers" <https://arxiv.org/abs/2406.04329>, Eq 5. It uses a specific
weighting function `weight_fn` derived from continuous-time formulation of
discrete diffusion, which computes `weight = -alpha_der / (1 - alpha)`. This
loss requires a `DiscreteSchedule`.

### `NoWeightDiscreteLoss`

This is a concrete implementation that computes discrete diffusion loss without
any weighting (i.e. `weight=1`).

## Riemannian Flow Matching Loss

Training a Riemannian Flow Matching (RFM) model requires a loss function that
respects the intrinsic geometry of the manifold $$(M, g)$$.

### Metric-Aware Loss

The **Riemannian Flow Matching loss** is defined as the squared norm of the
difference between the model's velocity prediction $$v_\theta$$ and the true
geodesic velocity $$u_t$$:

$$L(\theta) = \mathbb{E}_{t, x_0, x_1} [ \| v_{\theta}(x_t, t) - u_t(x_t | x_0, x_1) \|_{g}^2 ]$$

where the norm is induced by the Riemannian metric $$g$$ at point $$x_t$$:

$$\| v \|_{g} = \sqrt{g_{x_t}(v, v)}$$

### Implementation for Embedded Manifolds

For many manifolds implemented in this library (like the Sphere $$S^d$$ or
$$SO(3)$$), the Riemannian metric is induced by the standard Euclidean metric of
the ambient space $$\mathbb{R}^n$$. In these cases, the loss simplifies to:

$$L(\theta) = \mathbb{E}_{t, x_0, x_1} [ \| v_{\theta}(x_t, t) - u_t(x_t | x_0, x_1) \|_{2}^2 ]$$

**Crucially**, this equivalence only holds if $$v_{\theta}$$ and $$u_t$$ are
both valid **tangent vectors** (i.e., $$v, u \in T_{x_t} M$$). The library
ensures this via:

1.  **True Target**: The `RiemannianProcess` returns a $$u_t$$ that is
    mathematically guaranteed to be tangent to the manifold.
2.  **Model Forecast**: The **`RiemannianConditionalBackbone`** (see
    [Architecture docs](./architecture.md)) acts as a wrapper that projects the
    raw model output onto the tangent space $$T_{x_t} \mathcal{M}$$ before
    computing the loss.

By enforcing the tangent space constraint, the RFM objective can be optimized
using standard MSE loss while remaining geometrically rigorous.

## Time Sampling

(`lib/training/time_sampling.py`)

During training, the diffusion loss is computed at random timesteps sampled from
`[0, 1]`. The `TimeSampler` protocol defines how these timesteps are drawn. The
choice of time distribution can significantly affect training dynamics.

### `TimeSampler` Protocol

```python
def __call__(self, key: PRNGKey, data_spec: DataTree) -> TimeTree:
  ...
```

Takes a PRNG key and a data specification (used to infer batch size and shape),
and returns sampled timesteps. Supports both array and PyTree data.

### Implementations

*   **`UniformTimeSampler`**: Draws `t ~ Uniform(span)`. This is the simplest
    and most common strategy. The `axes` parameter controls which data axes
    share the same time (default: only batch axis).
*   **`LogitNormalTimeSampler`**: Draws `t` from a logit-normal distribution,
    following <https://arxiv.org/abs/2403.03206> (Eq. 19). Parameterized by
    `mean` and `scale`, this concentrates samples around specific noise levels,
    which can improve training efficiency.
*   **`UniformStratifiedTimeSampler`**: Stratified uniform sampling following
    <https://arxiv.org/abs/2107.00630> (I.1). Reduces variance by ensuring
    each example in the batch sees a different stratum of the time interval.
*   **`UnbalancedTimestepSampler`**: A multimodal-aware sampler from the
    JointDiT paper <https://arxiv.org/abs/2505.00482> (Section 3.1). It
    samples different times for different modalities, with a probability
    `p_equal` of synchronizing them.

### Example

```python
from hackable_diffusion.lib.training.time_sampling import (
    LogitNormalTimeSampler,
    UniformTimeSampler,
)
import jax
import jax.numpy as jnp

# Uniform sampling
uniform_sampler = UniformTimeSampler()
data_spec = jnp.zeros((8, 32, 32, 3))
times = uniform_sampler(jax.random.PRNGKey(0), data_spec)

# Logit-normal sampling (concentrates around t=0.5)
logit_sampler = LogitNormalTimeSampler(mean=0.0, scale=1.0)
times = logit_sampler(jax.random.PRNGKey(0), data_spec)
```
