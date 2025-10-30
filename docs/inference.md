# Inference Function

This document explains the concept of the "Inference Function" in the Hackable
Diffusion library. This function is the core component responsible for making a
single denoising prediction at each step of the sampling process.

The modules related to the inference function are located in `lib/inference/`.

[TOC]

## Overview

In a diffusion model sampler, at each timestep `t`, we need a function that
takes the noisy data `xt` and predicts some version of the clean data (e.g.,
predicts `x0` or the noise `epsilon`). The `InferenceFn` is the abstraction for
this single-step prediction function.

An `InferenceFn` can be a simple wrapper around a model call, or it can be a
more complex pipeline that includes classifier-free guidance, self-conditioning,
and other advanced techniques. The sampler itself doesn't need to know about
these details; it just calls the `InferenceFn`.

The main components are:

  * **`InferenceFn` Protocol**: The interface for the single-step prediction
    function.
  * **Guidance Functions**: Modules that implement classifier-free guidance.
  * **Projection Functions**: Modules that modify the model's output.
  * **`GuidedDiffusionInferenceFn`**: A concrete implementation that combines a
    base model call, guidance, and projection.

## `InferenceFn` Protocol

(`lib/inference/base.py`)

The `InferenceFn` is a protocol that defines a callable with the following
signature:

```python
def __call__(
    self, time: TimeTree, xt: DataTree, conditioning: Conditioning | None
) -> TargetInfoTree:
  ...
```

  * **Inputs**:
      * `time`: The current timestep(s).
      * `xt`: The noisy data at time `t`.
      * `conditioning`: A dictionary of conditioning signals.
  * **Output**:
      * A `TargetInfoTree` (a dictionary or pytree of dictionaries) containing
        the model's predictions. For a `GaussianProcess`, this would include
        keys like `'x0'`, `'epsilon'`, `'score'`, etc.

## Classifier-Free Guidance

(`lib/inference/guidance.py`)

Classifier-free guidance (<https://arxiv.org/abs/2207.12598>) is a technique to
improve sample quality by amplifying the effect of conditioning. It works by
combining the output of a conditional model call with that of an unconditional
call.

### `GuidanceFn` Protocol

This protocol defines the interface for guidance functions. The most common one
is `ScalarGuidanceFn`.

### `ScalarGuidanceFn`

This implements the standard guidance formula:

`guided_prediction = conditional + guidance_scale * (conditional -
unconditional)`

which can be rewritten as:

`guided_prediction = (1 + guidance_scale) * conditional - guidance_scale *
unconditional`

The `guidance` parameter corresponds to `guidance_scale`.

### `LimitedIntervalGuidanceFn`

This is a variant that applies guidance only within a specified time interval
`[lower, upper]`. Outside this interval, it returns the conditional prediction.
This can be useful for fine-tuning the sampling process. This is an
implementation of <https://arxiv.org/abs/2404.07724>.

## `GuidedDiffusionInferenceFn`

(`lib/inference/diffusion_inference.py`)

This class is the primary implementation of a guided `InferenceFn`. It composes
several pieces to form the final prediction logic.

It is configured with:

  * `base_inference_fn`: An `InferenceFn` that makes the raw model prediction.
    Typically, this is a wrapper around the Flax model's `apply` function.
  * `guidance_fn`: A `GuidanceFn` to apply classifier-free guidance.
  * `projection_fn`: A function to apply any final modifications to the guided
    output (e.g., for self-conditioning).

The execution flow within `GuidedDiffusionInferenceFn` is:

1.  Call `base_inference_fn` with the provided `conditioning` to get
    `cond_outputs`.
2.  Call `base_inference_fn` with `conditioning=None` to get `uncond_outputs`.
3.  Pass both sets of outputs to `guidance_fn` to get `guided_outputs`.
4.  Pass `guided_outputs` to `projection_fn` to get the final
    `projected_outputs`.
5.  Return `projected_outputs`.

### Example: Constructing an Inference Function

This example shows how to build a complete inference function stack.

```python
# This is a conceptual example. In practice, `base_inference_fn` would be
# created using wrappers from `lib.inference.wrappers`.

from hackable_diffusion.lib.inference.diffusion_inference import GuidedDiffusionInferenceFn
from hackable_diffusion.lib.inference.guidance import ScalarGuidanceFn

# Assume we have a `base_inference_fn` that wraps our model.
# This base_fn takes (time, xt, conditioning) and returns model predictions.
#
# def base_inference_fn(time, xt, conditioning):
#   ...
#   return model_predictions

# 1. Define the guidance function
# Use a guidance scale of 7.5
guidance_fn = ScalarGuidanceFn(guidance=7.5)

# 2. (Optional) Define a projection function. We'll use identity for simplicity.
from hackable_diffusion.lib.inference.projection import IdentityProjectionFn
projection_fn = IdentityProjectionFn()

# 3. Construct the guided inference function
# Assume `my_base_inference_fn` exists.
inference_fn = GuidedDiffusionInferenceFn(
    base_inference_fn=my_base_inference_fn,
    guidance_fn=guidance_fn,
    projection_fn=projection_fn,
)

# 4. Use the inference function
# A sampler would call this function in a loop.
# Let's simulate one such call:
#
# xt_at_step_i = ...
# time_at_step_i = ...
# my_conditioning = {'label': 3}
#
# final_prediction = inference_fn(
#     time=time_at_step_i,
#     xt=xt_at_step_i,
#     conditioning=my_conditioning,
# )
#
# predicted_x0 = final_prediction['x0']
# ... use predicted_x0 to compute x_t_minus_1
```
