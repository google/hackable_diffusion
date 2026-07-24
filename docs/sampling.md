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
    denoising prediction at a single time step (see the
    [Inference Function](./inference.md) documentation).
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

(`lib/hd_api.py`)

Two main data structures manage the state of the sampling loop:

*   **`StepInfo`**: A static container for all information related to a single
    step that can be pre-computed. This includes the step index, the continuous
    time `t`, and a JAX random key for that step.
*   **`DiffusionStep`**: The complete, dynamic state of the process at a given
    step. It contains the noisy data `xt` and the `StepInfo` for that step. This
    is the "state" that is carried over from one iteration of the sampling loop
    to the next.

## Time Scheduling

(`lib/sampling/time_scheduling.py`)

The `TimeSchedule` protocol is responsible for discretizing the `[0, 1]` time
interval.

*   `UniformTimeSchedule`: Creates linearly spaced time steps.
*   `EDMTimeSchedule`: Implements the non-uniform time step distribution from
    the EDM paper, which can improve sample quality. The `rho` parameter
    controls the density of steps near `t=0`.

Before the sampling loop begins, the time schedule is used to generate a list of
all `StepInfo` objects for the entire trajectory.

## Sampler Step Algorithms

(`lib/sampling/gaussian_step_sampler.py`,
`lib/sampling/discrete_step_sampler.py`,
`lib/sampling/simplicial_step_sampler.py`)

The `SamplerStep` protocol defines the actual sampling algorithm. It
encapsulates the mathematical formula for taking one step of the reverse
process.

The key method is `update(prediction, current_step, next_step_info)`, which
computes the next `DiffusionStep`.

Implementations for **Gaussian** processes include:

*   **`DDIMStep`**: Implements the popular Denoising Diffusion Implicit Models
    sampler. It can be deterministic (`stoch_coeff=0.0`) or stochastic
    (`stoch_coeff > 0.0`).
*   **`AdjustedDDIMStep`**: An improved DDIM variant from
    <https://arxiv.org/abs/2403.06807> that adjusts the update with an
    estimated covariance term to reduce sampling error.
*   **`SdeStep`**: A stochastic sampler based on discretizing the reverse-time
    Stochastic Differential Equation (SDE).
*   **`VelocityStep`**: A sampler that operates using the velocity prediction
    from the model.
*   **`HeunStep`**: A more accurate second-order solver.

### Riemannian Sampling Theory

Generating samples from a Riemannian Flow Matching model involves solving a
time-dependent Ordinary Differential Equation (ODE) on the manifold $$M$$:

$$\frac{dx_t}{dt} = v_{\theta}(x_t, t), \quad x_1 \sim \text{Invariant}(M)$$

where $$v_{\theta}$$ is the learned velocity field. To solve this ODE while
remaining on the manifold, we use specialized integration schemes that respect
the manifold's intrinsic geometry.

#### Riemannian Euler Integration

Implementations for **Riemannian** processes include:

*   **`RiemannianFlowSamplerStep`**: Implements **Riemannian Euler
    integration**. Instead of a standard additive update $$(x + dt \cdot v)$$,
    this step uses the manifold's **exponential map** to move along the tangent
    vector $$v$$ while respecting the manifold's curvature:

    $$x_{t-\Delta t} = \text{Exp}_{x_t}(-\Delta t \cdot v_{\theta}(x_t, t))$$

    This ensures that the updated state $$x_{t-\Delta t}$$ remains perfectly on
    the manifold $$M$$ (e.g., still has unit norm on a sphere) without needing
    ad-hoc projection steps. This is mathematically equivalent to moving along
    the unique geodesic starting at $$x_t$$ with initial velocity $$-v_\theta$$.

#### Why use Riemannian Euler?

In contrast, a **Euclidean Euler** step followed by a projection: 1. $$x' =
x_t - \Delta t \cdot v_\theta$$ 2. $$x_{t-\Delta t} = \text{Project}(x')$$

can lead to numerical drift and artifacts, especially when the manifold is
highly curved or the step size is large. Riemannian Euler is the "natural"
first-order integrator for manifolds as it directly utilizes the Riemannian
metric's shortest paths.

Note that in our implementation we assume that one step corresponds to one NFE.
While this strong assumption allows you to make the identification `num_steps =
NFE` it requires more complex implementation for higher-order sampler such as
the `HeunStep`. In the case of that second order update we alternate between
*two* versions of the single step.

Implementations for **Discrete** processes use different logic to sample tokens
based on predicted logits.

### `SimplicialDDIMStep`

(`lib/sampling/simplicial_step_sampler.py`)

The simplicial DDIM step implements the reverse process for the
`SimplicialProcess`. Unlike the Gaussian DDIM which manipulates continuous
vectors, or the discrete DDIM which overwrites integer tokens, the simplicial
DDIM operates on **probability vectors on the simplex**. It takes a noisy
log-probability distribution `P_t` and denoises it to `P_s` for `s < t`.

#### The backward transition

Given the current state `P_t` at time `t` and a predicted clean token `x̂_0`,
the backward transition produces `P_s` at time `s < t` via:

```
P_s = W · P_t^κ + (1 - W) · V
```

where:

*   **`W`** is a Beta-distributed mixing weight: `W ~ Beta(κ·ε/(1-α_t),
    ε/(1-α_s) - κ·ε/(1-α_t))`
*   **`P_t^κ`** is the Beta-shrunk version of `P_t` (see below)
*   **`V`** is a fresh Dirichlet sample: `V ~ Dir((1-κ)·ε·π + (ε·h_s -
    κ·ε·h_t)·δ(x̂_0))`

The mixture is computed in log-space via `logaddexp(log_w + log_pt_kappa,
log_1_minus_w + log_v)`.

#### Beta-shrinkage (`log_beta_shrinkage`)

Beta-shrinkage is the key building block that allows "partial forgetting" of
information from `P_t` without fully resampling. Given `X ~ Dir(α)`, it produces
`Y ~ Dir(κ·α)` via the following identity:

1.  For each category `i`, sample `B_i ~ Beta(κ·α_i, (1-κ)·α_i)` independently.
2.  Compute `Y_i = B_i · X_i / Σ_j B_j · X_j`.

Then `Y ~ Dir(κ·α)`. This "shrinks" the concentration parameter by a factor `κ`,
making the distribution more diffuse. The implementation adds a `safety_epsilon`
to both Beta shape parameters to handle the degenerate case `κ → 0`.

#### The `churn` parameter

The `churn` attribute controls the stochasticity of the reverse step. It is
related to the theoretical parameter `κ` by `κ = 1 - churn`:

| `churn` | `κ`   | Behaviour                                                  |
| ------- | ----- | ---------------------------------------------------------- |
| `0.0`   | `1`   | **Deterministic DDIM**: maximum trust in `P_t`, no         |
:         :       : shrinkage. `P_s = W·P_t + (1-W)·δ(x̂_0)`. Fastest, least   :
:         :       : diverse.                                                   :
| `0.5`   | `0.5` | **Balanced**: `P_t` is partially shrunk, then mixed with a |
:         :       : fresh Dirichlet sample.                                    :
| `1.0`   | `0`   | **Fully stochastic**: ignores `P_t` entirely. `W → 0`,     |
:         :       : `P_s ≈ V ~ Dir(β_s)`. Most diverse, slowest convergence.   :

The `churn=0` case uses a fast path that skips both the Beta-shrinkage and the
Dirichlet sampling for `V`.

#### Configuration

*   `corruption_process`: The `SimplicialProcess` used during training. The
    sampler reads the schedule, invariant distribution, temperature, and
    post-corruption function from it.
*   `churn`: Float in `[0, 1]` (default `1.0`). Controls stochasticity.
*   `safety_epsilon`: Small constant for numerical stability at `churn=1.0`
    (default `1e-6`).

#### Example Usage

```python
from hackable_diffusion.lib.sampling.simplicial_step_sampler import (
    SimplicialDDIMStep,
)
from hackable_diffusion.lib.sampling.sampling import DiffusionSampler
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule
from hackable_diffusion.lib.corruption.simplicial import SimplicialProcess
from hackable_diffusion.lib.corruption.discrete import CosineDiscreteSchedule

# 1. Define the corruption process (must match training)
corruption_process = SimplicialProcess.uniform_process(
    schedule=CosineDiscreteSchedule(),
    num_categories=5,
    temperature=1.0,
)

# 2. Create the sampler step
stepper = SimplicialDDIMStep(
    corruption_process=corruption_process,
    churn=0.0,  # Deterministic DDIM
)

# 3. Create the full sampler
sampler = DiffusionSampler(
    time_schedule=UniformTimeSchedule(),
    stepper=stepper,
    num_steps=100,
)

# 4. Sample (assuming inference_fn is defined)
# initial_noise = corruption_process.sample_from_invariant(key, data_spec)
# final_step, trajectory = sampler(
#     inference_fn=inference_fn,
#     rng=key,
#     initial_noise=initial_noise,
#     conditioning=None,
# )
# generated_logits = final_step.xt  # shape (batch, ..., K), log-probs
# generated_tokens = jnp.argmax(generated_logits, axis=-1)
```

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
from hackable_diffusion.lib.corruption.gaussian import CosineSchedule

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

### Riemannian Sampling Example

For manifolds like the Sphere or SO(3), the setup is similar but uses
Riemannian-specific components.

```python
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption.riemannian import RiemannianProcess
from hackable_diffusion.lib.corruption.riemannian import LinearRiemannianSchedule
from hackable_diffusion.lib.sampling.riemannian_sampling import RiemannianFlowSamplerStep
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

# 1. Define manifold and process
manifold = manifolds.Sphere()
process = RiemannianProcess(

    manifold=manifold,
    schedule=LinearRiemannianSchedule(),
, schedule=LinearRiemannianSchedule()
)

# 2. Configure Sampler Step
stepper = RiemannianFlowSamplerStep(corruption_process=process)

# 3. Create the sampler
sampler = DiffusionSampler(
    time_schedule=UniformTimeSchedule(),
    stepper=stepper,
    num_steps=50,
)
```

This modular setup makes it easy to experiment with different samplers (e.g.,
swapping `DDIMStep` for `SdeStep`), time schedules, or number of steps with
minimal code changes.
