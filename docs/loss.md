# Diffusion Loss Functions

This document details the loss functions used for training diffusion models in
the Hackable Diffusion library. The loss modules are designed with flexibility
to support various training objectives and weighting schemes.

The modules related to loss functions are located in `lib/loss/`.

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

(`lib/loss/base.py`)

This protocol defines the basic interface for a diffusion loss. It's a callable
that takes the model's predictions, the ground truth targets, and the time, and
returns a batch-wise loss.

```python
def __call__(
    self,
    preds: TargetInfoTree,
    targets: TargetInfoTree,
    time: TimeTree,
) -> LossOutputTree:
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

(`lib/loss/gaussian.py`)

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

This is a concrete implementation of the sigmoid weighting introduced in
<https://arxiv.org/abs/2303.00848> and use in Simpler Diffusion (SiD2)
<https://arxiv.org/abs/2410.19324>. It computes an MSE in `x0` space, converts
it to logSNR time, and applies an additional sigmoid-based weighting. This has
been shown to improve training stability and performance.

#### Example

```python
import jax.numpy as jnp
from hackable_diffusion.lib.loss.gaussian import SiD2Loss
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

(`lib/loss/discrete.py`)

For discrete data, the training objective is typically to predict the
probability distribution of the original token. This is based on
<https://arxiv.org/abs/2406.04329>.

### `DiffusionCrossEntropyLoss`

This loss function is designed for data from a `CategoricalProcess`. It
calculates a weighted cross-entropy loss between the model's predicted logits
and the true underlying token.

The loss has a specific weighting `alpha_der / (1 - alpha)` which is derived
from the continuous-time formulation of discrete diffusion. It can also
optionally apply an additional `weight_fn`.

A key option is `use_mask`. If `True`, it will ignore tokens that were *not*
corrupted (masked) at a given timestep, focusing the loss only on the tokens the
model needs to predict.
