# Copyright 2026 Hackable Diffusion Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Autoregressive diffusion sampler.

This module implements an autoregressive generation loop where each generation
step produces a fixed-length "canvas" of data via diffusion sampling. The canvas
is then post-processed and integrated into the running sampler state.

The overall sampling flow is:

                        conditioning
                           │
                           ▼
                ┌─────────────────────┐
                │ ARStateHandler      │
                │ .init_ar_state      │──── Build initial state,
                └─────────────────────┘
                           │
                  (SamplerState, batch_size)
                           │
                           ▼
            ┌──────────────────────────────┐
            │         AR Loop              │
            │  (up to max_num_canvases)    │
            │                              │
            │  ┌────────────────────────┐  │
            │  │  EarlyStoppingFn       │──┼──▶ break if done
            │  └────────────────────────┘  │
            │              │               │
            │              ▼               │
            │  ┌────────────────────────┐  │
            │  │  DiffusionProcess      │  │
            │  │  .sample_from_invariant│──┼──▶ initialize noisy canvas
            │  └────────────────────────┘  │
            │              │               │
            │              ▼               │
            │  ┌────────────────────────┐  │
            │  │  ARStateHandler        │  │
            │  │  .create_conditioning  │──┼──▶ extract diffusion
            │  │   _from_state          │  │    conditioning from state
            │  └────────────────────────┘  │
            │              │               │
            │              ▼               │
            │  ┌────────────────────────┐  │
            │  │  DiffusionSampler      │  │
            │  │  (canvas_sampler)      │──┼──▶ denoise canvas via
            │  └────────────────────────┘  │    diffusion sampling
            │              │               │
            │              ▼               │
            │  ┌────────────────────────┐  │
            │  │  ARStateHandler        │  │
            │  │  .update_ar_state      │──┼──▶ update state
            │  └────────────────────────┘  │
            │              │               │
            │              └───────────┐   │
            │                   next   │   │
            │                  canvas  │   │
            └──────────────────────────────┘
                           │
                           ▼
                ┌─────────────────────┐
                │ ARStateHandler      │
                │ .finalize_ar_state  │──── Extract generated output
                └─────────────────────┘
                           │
                           ▼
                     output data

The architecture is model-agnostic: all model-specific logic is injected via
the ``ARStateHandler`` base class, which encapsulates:

  - ``init_ar_state``: Initializes the state.
  - ``update_ar_state``: Handles canvas post-processing and state bookkeeping
      after each canvas is sampled.
  - ``finalize_ar_state``: Extracts the final generated output from the state.

An ``EarlyStoppingFn`` Protocol controls when to terminate the AR loop early.

The AR loop uses ``jax.lax.while_loop`` and terminates when ``max_num_canvases``
is reached or an early stopping condition is met.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Protocol

from hackable_diffusion.lib import corruption
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import inference
from hackable_diffusion.lib.sampling import sampling
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt
from kauldron.ktyping import Bool, PRNGKey

################################################################################
# MARK: Type aliases
################################################################################

SamplerState = dict[str, Any]
DataArray = hd_typing.DataArray
Conditioning = hd_typing.Conditioning
InferenceFn = inference.InferenceFn
DiffusionStepTree = sampling.DiffusionStepTree

################################################################################
# MARK: ARStateHandler
################################################################################


class ARStateHandler(Protocol):
  """Manages the sampler state lifecycle during AR sampling.

  Subclass this to inject model-specific logic for initializing,
  updating, and finalizing the autoregressive sampler state.

  Methods:
    init_ar_state: Creates the initial state from conditioning.
    update_ar_state: Post-processes a sampled canvas and updates the state
      (i.e. update KV cache for LLMs).
    finalize_ar_state: Extracts the final generated output from the
      state.
    create_conditioning_from_state: Extracts the subset of sampler
      state needed as conditioning for the diffusion sampler.
  """

  def init_ar_state(
      self,
      *,
      batch_size: int,
      conditioning: Conditioning,
      canvas_length: int,
      max_num_canvases: int,
  ) -> SamplerState:
    ...

  def update_ar_state(
      self,
      canvas_last_step: DiffusionStepTree,
      sampler_state: SamplerState,
  ) -> SamplerState:
    ...

  def finalize_ar_state(
      self,
      sampler_state: SamplerState,
  ) -> DataArray:
    ...

  def create_conditioning_from_state(
      self,
      sampler_state: SamplerState,
  ) -> Conditioning:
    ...


################################################################################
# MARK: EarlyStoppingFn
################################################################################


class EarlyStoppingFn(Protocol):
  """Determines whether to terminate the AR loop early.

  The function receives the full sampler state and must return a JAX
  boolean *scalar* (``True`` → stop).  The canonical implementation
  checks ``jnp.all(sampler_state['done'])`` where ``done`` is a
  per-batch-element boolean array.
  """

  def __call__(self, sampler_state: SamplerState) -> Bool['']:
    """Returns true when the AR loop should terminate."""


class DoneEarlyStoppingFn(EarlyStoppingFn):
  """Stops when every batch element is done."""

  def __call__(self, sampler_state: SamplerState) -> Bool['']:
    if 'done' not in sampler_state:
      raise ValueError(
          'DoneEarlyStoppingFn requires sampler_state["done"] to be set.'
      )
    return jnp.all(sampler_state['done'])


################################################################################
# MARK: Sampler
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class AutoregressiveDiffusionSampler:
  """Generates data by autoregressively sampling fixed-length canvases.

  Each iteration of the generation loop:
    1. Samples a canvas of ``canvas_length`` elements via diffusion.
    2. Passes the canvas to ``state_handler.update_ar_state`` for
       post-processing and state bookkeeping.
    3. Checks EarlyStoppingFn to decide whether to stop.

  After the loop, ``state_handler.finalize_ar_state`` extracts the final
  generated output.

  The loop is implemented via ``jax.lax.while_loop`` for JIT compatibility.

  Attributes:
    canvas_sampler: Diffusion sampler that denoises a single canvas.
    diffusion_process: Noise process used to initialize canvases.
    canvas_length: Number of elements per canvas.
    max_num_canvases: Maximum number of canvases to generate.
    state_handler: Manages the AR state lifecycle (init, update, finalize).
    early_stopping_fn: Determines whether to terminate the AR loop early.
    data_dtype: Data type of the generated output.
    data_shape: Additional dimensions of the generated output (e.g., spatial
      dimensions for images).
    use_lax_while_loop: An optional flag to choose between lax while loop or
      pythonic one.
  """

  canvas_sampler: sampling.DiffusionSampler
  diffusion_process: corruption.CategoricalProcess
  canvas_length: int
  max_num_canvases: int
  state_handler: ARStateHandler
  data_dtype: jnp.dtype
  data_shape: tuple[int, ...]
  early_stopping_fn: EarlyStoppingFn = DoneEarlyStoppingFn()
  use_lax_while_loop: bool = True

  @kt.typechecked
  def __call__(
      self,
      diffusion_inference_fn: inference.InferenceFn,
      batch_size: int,
      rng: PRNGKey,
      conditioning: Conditioning,
  ) -> tuple[DataArray, SamplerState]:
    """Generates data autoregressively via discrete diffusion.

    Uses ``jax.lax.while_loop`` for JIT compatibility with true early
    stopping.

    Args:
      diffusion_inference_fn: Model inference function called during diffusion
        sampling.
      batch_size: Batch size of the generation.
      rng: JAX PRNG key, split per canvas for reproducibility.
      conditioning: Conditioning for the generation (e.g., text prompts, images,
        or any modality-specific inputs).

    Returns:
      A tuple of (generated output, final sampler state).
    """

    sampler_state = self.state_handler.init_ar_state(
        batch_size=batch_size,
        conditioning=conditioning,
        canvas_length=self.canvas_length,
        max_num_canvases=self.max_num_canvases,
    )

    # Carry: (sampler_state, rng, step_counter)
    init_carry = (sampler_state, rng, jnp.int32(0))

    def _cond_fn(carry):
      sampler_state, _, step = carry
      should_stop = self.early_stopping_fn(sampler_state)
      should_continue = ~should_stop
      less_than_max_canvases = step < self.max_num_canvases
      return should_continue & less_than_max_canvases

    def _body_fn(carry):
      sampler_state, rng, step = carry

      # Propagate random number generator.
      rng, canvas_init_rng, canvas_sampler_rng = jax.random.split(rng, 3)

      # Create new canvas.
      initial_canvas = self.diffusion_process.sample_from_invariant(
          key=canvas_init_rng,
          data_spec=jnp.zeros(
              (
                  batch_size,
                  self.canvas_length,
              )
              + self.data_shape,
              dtype=self.data_dtype,
          ),
      )

      # Sample canvas via diffusion.
      # TODO: Implement returning the whole sampling trajectory.
      canvas_last_step, _ = self.canvas_sampler(
          inference_fn=diffusion_inference_fn,
          rng=canvas_sampler_rng,
          initial_noise=initial_canvas,
          conditioning=self.state_handler.create_conditioning_from_state(
              sampler_state=sampler_state
          ),
      )
      # Post-process canvas and update sampler state.
      sampler_state = self.state_handler.update_ar_state(
          canvas_last_step=canvas_last_step,
          sampler_state=sampler_state,
      )

      return (sampler_state, rng, step + 1)

    if self.use_lax_while_loop:
      sampler_state, _, _ = jax.lax.while_loop(_cond_fn, _body_fn, init_carry)
    else:
      carry = init_carry
      while _cond_fn(carry):
        carry = _body_fn(carry)
      sampler_state, _, _ = carry

    # Read-out the final output.
    output = self.state_handler.finalize_ar_state(sampler_state=sampler_state)
    return output, sampler_state
