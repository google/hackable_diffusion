# Corruption Processes

This document describes the corruption processes used in the Hackable Diffusion
library, which correspond to the forward process in diffusion models. These
modules are responsible for adding noise to the data.

The modules related to corruption are located in `lib/corruption/`.

[TOC]

## Overview

The "corruption" process defines how clean data `x0` is gradually transformed
into noise. This library provides a flexible framework for defining and using
various corruption processes for both continuous and discrete data.

The main components are:

  * **`CorruptionProcess` Protocol**: An interface that standardizes how
    corruption is applied.
  * **Schedules**: Functions that define the rate and nature of corruption over
    time `t`.
  * **Process Implementations**: Concrete classes like `GaussianProcess` for
    continuous data (e.g., images) and `CategoricalProcess` for discrete data
    (e.g., labels, tokens).

## `CorruptionProcess` Protocol

(`lib/corruption/base.py`)

The `CorruptionProcess` is a protocol (an interface) that all corruption
processes must implement. It defines the following key methods:

  * `corrupt(key, x0, time)`: Takes clean data `x0` and a time `t`, and returns
    the corrupted data `xt` along with a dictionary of potential training
    targets and metadata (`target_info`).
  * `sample_from_invariant(key, data_spec)`: Samples from the invariant
    distribution of the process (i.e., pure noise at `t=1`).
  * `convert_predictions(prediction, xt, time)`: Takes a model's prediction
    (e.g., predicted epsilon) and converts it into all other possible target
    parameterizations (e.g., predicted `x0`, score, etc.).
  * `get_schedule_info(time)`: Returns parameters of the schedule at a given
    time.

### `NestedProcess`

For handling complex data structures (pytrees), `NestedProcess` is a wrapper
that applies different corruption processes to different leaves of the pytree.
For example, you can use a `GaussianProcess` on an image and a
`CategoricalProcess` on its corresponding labels simultaneously.

## Schedules

(`lib/corruption/schedules.py`)

Schedules define how the corruption parameters change over the continuous time
interval `[0, 1]`.

### `GaussianSchedule`

For Gaussian processes, schedules define `alpha(t)` and `sigma(t)`. Common
implementations include:

  * `CosineSchedule`: A popular choice where `alpha(t) = cos(0.5 * pi * t)`.
  * `RFSchedule`: Rectified Flow schedule where `alpha(t) = 1 - t` and `sigma(t)
    = t`.
  * `LinearDiffusionSchedule`: The schedule from the original DDPM paper,
    parameterized by `beta_min` and `beta_max`.

### `DiscreteSchedule`

For discrete processes, schedules define `alpha(t)`, which is the probability of
*keeping* the original token at time `t`. Implementations include:

  * `LinearDiscreteSchedule`: `alpha(t) = 1 - t`.
  * `CosineDiscreteSchedule`: `alpha(t) = cos(0.5 * pi * t)`.

## `GaussianProcess`

(`lib/corruption/gaussian.py`)

This is the implementation for standard diffusion on continuous data. It defines
the corruption as:

`xt = alpha(t) * x0 + sigma(t) * epsilon`

where `epsilon` is standard Gaussian noise with unit variance.

### Prediction Parameterizations

A key feature of `GaussianProcess` is its handling of different prediction
targets. The denoising model can be trained to predict various quantities. The
`corrupt` method returns all of them in `target_info`, and `convert_predictions`
can switch between them.

The supported parameterizations are:

  * **`x0`**: Predict the original clean data.
  * **`epsilon`**: Predict the noise that was added.
  * **`score`**: Predict the score function (gradient of the log-density), which
    is `-epsilon / sigma(t)`.
  * **`velocity`**: The velocity field using in Flow Matching
    (<https://arxiv.org/abs/2210.02747>), Rectified Flow
    (<https://arxiv.org/abs/2209.03003>) and Stochastic Interpolants
    (<https://arxiv.org/abs/2303.08797>) implementations.
  * **`v`**: The `v-prediction` first introduced in Progressive Distillation
    (<https://arxiv.org/abs/2202.00512>).

This flexibility allows you to train a model with one objective (e.g.,
epsilon-prediction) but use a sampler that requires a different one (e.g.,
x0-prediction) without any code change in the sampler.

### Example Usage

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import CosineSchedule

key = jax.random.PRNGKey(0)

# 1. Define schedule and process
schedule = CosineSchedule()
process = GaussianProcess(schedule=schedule)

# 2. Create data and time
x0 = jnp.ones((1, 32, 32, 3))
time = jnp.array([0.5]) # Corrupt halfway

# 3. Apply corruption
key, subkey = jax.random.split(key)
xt, target_info = process.corrupt(subkey, x0, time)

print(f"Shape of xt: {xt.shape}")
print(f"Available targets: {target_info.keys()}")

# 4. Convert between predictions
# Suppose a model predicts epsilon
model_prediction = {'epsilon': jnp.zeros_like(x0)}

# Get all other parameterizations
all_predictions = process.convert_predictions(model_prediction, xt, time)
print(f"Converted predictions: {all_predictions.keys()}")

# You can access the predicted x0
predicted_x0 = all_predictions['x0']
```

## `CategoricalProcess`

(`lib/corruption/discrete.py`)

This process is designed for discrete data, such as integer class labels or
tokens. At each time `t`, it replaces original tokens with noise tokens with
probability `1 - alpha(t)`.

Key configuration parameters:

  * `schedule`: A `DiscreteSchedule` that defines `alpha(t)`.
  * `invariant_probs`: The probability distribution of the "noise" tokens that
    replace the original ones.
  * `num_categories`: The number of valid categories in the data.

The library provides convenient factory methods for common use cases:

  * `CategoricalProcess.uniform_process`: Corrupts tokens by replacing them with
    a token drawn uniformly from all possible categories.
  * `CategoricalProcess.masking_process`: Corrupts tokens by replacing them with
    a special "mask" token. This requires `num_categories` to be the vocabulary
    size, and the mask token will be integer `num_categories`.

### Example Usage (Masking)

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.schedules import LinearDiscreteSchedule

key = jax.random.PRNGKey(0)
num_classes = 10 #
mask_token_id = num_classes

# 1. Create a masking process
schedule = LinearDiscreteSchedule()
process = CategoricalProcess.masking_process(
    schedule=schedule,
    num_categories=num_classes,
)

# 2. Data and time
# Shape is (batch, sequence_length, 1)
x0 = jnp.array([1, 2, 3, 4]).reshape(1, 4, 1)
time = jnp.array([0.8]) # Corrupt heavily, most tokens should be masked

# 3. Corrupt data
key, subkey = jax.random.split(key)
xt, target_info = process.corrupt(subkey, x0, time)

print(f"Original x0: {x0.flatten()}")
print(f"Corrupted xt: {xt.flatten()}")
# With high probability, most tokens in xt will be `mask_token_id` (10)

# The target info contains the one-hot encoded version of x0
print(f"Logits target shape: {target_info['logits'].shape}")
# Logits target shape: (1, 4, 10)
```

**Assumptions**:

  * Discrete data is expected to be integer arrays with a trailing dimension
    of 1.
  * The model prediction for discrete data is expected to be logits over the
    categories. `convert_predictions` will then convert these logits to a
    predicted `x0` (via argmax).
