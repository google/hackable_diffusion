# Multimodal Diffusion

This document explains how to use Hackable Diffusion's "Nested" wrappers to
build multimodal diffusion models that operate on PyTree-structured data.

The multimodal wrappers are located in `lib/multimodal.py`.

[TOC]

## Overview

Hackable Diffusion's core protocols (`CorruptionProcess`, `SamplerStep`,
`DiffusionLoss`, etc.) are designed around single-modal arrays. To handle
multimodal data — where different parts of the input (e.g., image + labels,
continuous + discrete) require different diffusion treatments — the library
provides **Nested wrappers**.

Each wrapper takes a PyTree of single-modal components that matches the
structure of your data. When called, it dispatches each method to the
corresponding component-data pair.

## Available Wrappers

### Training

*   **`NestedProcess`**: Applies different corruption processes per modality.
*   **`NestedDiffusionLoss`**: Computes different loss functions per modality.
*   **`NestedTimeSampler`**: Samples timesteps independently per modality.

### Sampling

*   **`NestedSamplerStep`**: Runs different sampler algorithms per modality.
*   **`NestedTimeSchedule`**: Uses different time discretizations per modality.
*   **`NestedGuidanceFn`**: Applies different guidance functions per modality.

## Key Concept: Structure Matching

The **structure of your Nested wrapper must match the structure of your data**.
For example, if your data is a dictionary `{"image": ..., "label": ...}`, your
`NestedProcess` must also be keyed by `{"image": ..., "label": ...}`.

```python
data = {
    "image": jnp.zeros((batch, 32, 32, 3)),
    "label": jnp.zeros((batch, 1), dtype=jnp.int32),
}

process = NestedProcess(
    processes={
        "image": GaussianProcess(schedule=CosineSchedule()),
        "label": CategoricalProcess.masking_process(
            schedule=LinearDiscreteSchedule(), num_categories=10,
        ),
    }
)
```

## Example: Multimodal Training Setup

```python
from hackable_diffusion.lib.multimodal import (
    NestedProcess,
    NestedDiffusionLoss,
    NestedTimeSampler,
)
from hackable_diffusion.lib.corruption.gaussian import GaussianProcess
from hackable_diffusion.lib.corruption.discrete import CategoricalProcess
from hackable_diffusion.lib.corruption.gaussian import CosineSchedule
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.training.gaussian_loss import SiD2Loss
from hackable_diffusion.lib.training.discrete_loss import MD4Loss
from hackable_diffusion.lib.training.time_sampling import UniformTimeSampler

# 1. Define per-modality corruption processes
process = NestedProcess(
    processes={
        "image": GaussianProcess(schedule=gaussian.CosineSchedule()),
        "label": CategoricalProcess.masking_process(
            schedule=discrete.LinearDiscreteSchedule(), num_categories=10,
        ),
    }
)

# 2. Define per-modality losses
loss_fn = NestedDiffusionLoss(
    losses={
        "image": SiD2Loss(schedule=gaussian.CosineSchedule()),
        "label": MD4Loss(schedule=discrete.LinearDiscreteSchedule()),
    }
)

# 3. Define per-modality time sampling (optional — can also share time)
time_sampler = NestedTimeSampler(
    time_samplers={
        "image": UniformTimeSampler(),
        "label": UniformTimeSampler(),
    }
)
```

## Example: Multimodal Sampling Setup

```python
from hackable_diffusion.lib.multimodal import (
    NestedSamplerStep,
    NestedTimeSchedule,
)
from hackable_diffusion.lib.sampling.gaussian_step_sampler import DDIMStep
from hackable_diffusion.lib.sampling.discrete_step_sampler import DiscreteDDIMStep
from hackable_diffusion.lib.sampling.time_scheduling import UniformTimeSchedule

sampler_step = NestedSamplerStep(
    sampler_steps={
        "image": DDIMStep(
            corruption_process=gaussian_process,
            stoch_coeff=0.0,
        ),
        "label": DiscreteDDIMStep(
            corruption_process=categorical_process,
        ),
    }
)

time_schedule = NestedTimeSchedule(
    time_schedules={
        "image": UniformTimeSchedule(),
        "label": UniformTimeSchedule(),
    }
)
```

## How It Works

Internally, Nested wrappers use `jax_helpers.lenient_map` to traverse the data and
component PyTrees in parallel, calling the corresponding method on each
component with its matching data leaf. This means:

*   Any nesting depth works (dictionaries, named tuples, etc.).
*   Single-modal and multimodal code share the same protocols.
*   You can mix and match any combination of corruption processes, samplers, and
    losses.

The `mnist_multimodal` notebook provides a complete end-to-end example.
