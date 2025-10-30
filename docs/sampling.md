# Sampling

This document provides a comprehensive guide to the sampling process in the
Hackable Diffusion library, which is the process of generating new data from a
trained diffusion model.

The modules related to sampling are located in `lib/sampling/`.

[TOC]

## Overview of the Sampling Process

Sampling is the reverse of the corruption process. It starts with pure noise (at
`t=1`) and iteratively denoises it over a series of discrete time steps to
produce a clean sample (at `t=0`). The sampling logic is designed to be highly
modular and "hackable", allowing different components to be easily swapped.

The sampling process is orchestrated by `DiffusionSampler`, which combines three
key components:

1.  **`TimeSchedule`**: Defines the sequence of discrete time steps `{t_N,
    t_{N-1}, ..., t_0}` for the denoising process.
2.  **`InferenceFn`**: The function that calls the trained model to make a
    denoising prediction at a single time step (see the [Inference
    Function](./inference.md) documentation).
3.  **`SamplerStep`**: An implementation of a specific sampling algorithm (e.g.,
    DDIM, SDE) that uses the model's prediction to compute the state at the next
    time step.

The overall flow for `N` steps is:

1.  Start with `x_{t_N}` drawn from the invariant noise distribution.
2.  For `i` from `N` down to `1`: a. Get the current time `t_i` and the next
    time `t_{i-1}` from the schedule. b. Call `inference_fn(x_{t_i}, t_i,
    conditioning)` to get a prediction (e.g., predicted `x0`). c. Use the
    `SamplerStep.update` method with the prediction to compute `x_{t_{i-1}}`
    from `x_{t_i}`.
3.  Return the final sample.

## Core Data Structures

(`lib/sampling/base.py`)

Two main data structures manage the state of the sampling loop:

  * **`StepInfo`**: A static container for all information related to a single
    step that can be pre-computed. This includes the step index, the continuous
    time `t`, and a JAX random key for that step.
  * **`DiffusionStep`**: The complete, dynamic state of the process at a given
    step. It contains the noisy data `xt` and the `StepInfo` for that step. This
    is the "state" that is carried over from one iteration of the sampling loop
    to the next.

## Time Scheduling

(`lib/sampling/time_scheduling.py`)

The `TimeSchedule` protocol is responsible for discretizing the `[0, 1]` time
interval.

  * `UniformTimeSchedule`: Creates linearly spaced time steps.
  * `EDMTimeSchedule`: Implements the non-uniform time step distribution from
    the EDM paper, which can improve sample quality. The `rho` parameter
    controls the density of steps near `t=0`.

Before the sampling loop begins, the time schedule is used to generate a list of
all `StepInfo` objects for the entire trajectory.

## Sampler Step Algorithms

(`lib/sampling/gaussian_step_sampler.py`,
`lib/sampling/discrete_step_sampler.py`)

The `SamplerStep` protocol defines the actual sampling algorithm. It
encapsulates the mathematical formula for taking one step of the reverse
process.

The key method is `update(prediction, current_step, next_step_info)`, which
computes the next `DiffusionStep`.

Implementations for **Gaussian** processes include:

  * **`DDIMStep`**: Implements the popular Denoising Diffusion Implicit Models
    sampler. It can be deterministic (`stoch_coeff=0.0`) or stochastic
    (`stoch_coeff > 0.0`).
  * **`SdeStep`**: A stochastic sampler based on discretizing the reverse-time
    Stochastic Differential Equation (SDE).
  * **`VelocityStep`**: A sampler that operates using the velocity prediction
    from the model.
  * **`HeunStep`**: A more accurate second-order solver.

Note that in our implementation we assume that one step corresponds to one NFE.
While this strong assumption allows you to make the identification `num_steps =
NFE` it requires more complex implementation for higher-order sampler such as
the `HeunStep`. In the case of that second order update we alternate between
*two* versions of the single step.

Implementations for **Discrete** processes use different logic to sample tokens
based on predicted logits.

## The Main Sampling Loop: `DiffusionSampler`

(`lib/sampling/sampling.py`)

`DiffusionSampler` is the main driver that brings all the components together.
It is initialized with a schedule, a stepper algorithm, and the number of steps.
Its `__call__` method executes the full sampling loop using `jax.lax.scan` for
performance.

### Example: Putting It All Together

This example demonstrates how to configure and run a complete sampling process.

```python
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import EDMTimeSchedule
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.schedules import CosineSchedule

# Assume `inference_fn` is already created (see Inference Function doc).
# def inference_fn(xt, time, conditioning): ...

# 1. Define the Corruption Process (needed by the sampler step)
# This must match the process used during training.
corruption_process = GaussianProcess(schedule=CosineSchedule())

# 2. Choose and configure the Time Schedule
# Use 50 steps with EDM-style spacing.
time_schedule = EDMTimeSchedule(rho=7.0)
num_steps = 50

# 3. Choose and configure the Sampler Step algorithm
# Use deterministic DDIM.
stepper = DDIMStep(
    corruption_process=corruption_process,
    stoch_coeff=0.0,
)

# 4. Create the main DiffusionSampler
sampler = DiffusionSampler(
    time_schedule=time_schedule,
    stepper=stepper,
    num_steps=num_steps,
)

# 5. Prepare for sampling
key = jax.random.PRNGKey(0)
batch_size = 4
image_shape = (32, 32, 3)

# Start with pure Gaussian noise
sample_key, noise_key = jax.random.split(key)
initial_noise = jax.random.normal(noise_key, (batch_size,) + image_shape)
conditioning = None # or {'label': ...}

# 6. Run the sampling loop
final_step, all_steps_trajectory = sampler(
    inference_fn=inference_fn,
    rng=sample_key,
    initial_noise=initial_noise,
    conditioning=conditioning,
)

# The final generated image is in `final_step.xt`
generated_images = final_step.xt

print(f"Shape of generated images: {generated_images.shape}")
# Shape of generated images: (4, 32, 32, 3)
```

This modular setup makes it easy to experiment with different samplers (e.g.,
swapping `DDIMStep` for `SdeStep`), time schedules, or number of steps with
minimal code changes.
