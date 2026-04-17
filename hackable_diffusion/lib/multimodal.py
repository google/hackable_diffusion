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

"""Multimodal wrappers for hackable_diffusion.

Hackable Diffusion (HD) is primarily designed around single-modal protocols:
each component (corruption, sampler step, loss, guidance, etc.) operates on a
single data array. This module enables **multimodal diffusion** by providing
`Nested*` wrapper classes that lift every single-modal protocol to operate over
arbitrary PyTrees of data — e.g. ``{"image": ..., "text": ...}``.

Each `Nested*` class holds a PyTree of single-modal instances whose structure
mirrors the data tree, and delegates calls leaf-wise using
``utils.lenient_map``.  This means any combination of modalities and nesting
depths works out-of-the-box without modifying the underlying single-modal
implementations.

The module provides the following wrappers:

  Training:
    - ``NestedProcess``        — corruption / noise process
    - ``NestedDiffusionLoss``  — per-modality loss functions
    - ``NestedTimeSampler``    — independent per-modality time sampling
    - ``JointNestedTimeSampler`` — joint (shared) time sampling

  Sampling / Inference:
    - ``NestedSamplerStep``   — denoising step algorithm
    - ``NestedTimeSchedule``  — discrete time-step schedules
    - ``NestedGuidanceFn``    — classifier-free guidance
    - ``NestedProjectionFn``  — output projection / clamping

  Architecture:
    - ``NestedTimeEmbedder``  — per-modality time embeddings

  Network:
    - ``MultiModalDiffusionNetwork`` — see ``diffusion_network.py``
"""

from __future__ import annotations

import dataclasses
from typing import cast

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import time_sampling
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.corruption import base as corruption_base
from hackable_diffusion.lib.inference import guidance as guidance_lib
from hackable_diffusion.lib.inference import projection as projection_lib
from hackable_diffusion.lib.loss import base as loss_base
from hackable_diffusion.lib.sampling import base as sampling_base
from hackable_diffusion.lib.sampling import time_scheduling
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
LossOutputTree = hd_typing.LossOutputTree
ScheduleInfoTree = hd_typing.ScheduleInfoTree
TargetInfoTree = hd_typing.TargetInfoTree
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree


################################################################################
# MARK: NestedProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedProcess(corruption_base.CorruptionProcess):
  """Wrapper for a pytree of corruption processes mapped over the data.

  Enables using different corruption processes for different input modalities.

  Usage Example:
    ```
    process = NestedProcess(
        processes={
            "image": GaussianProcess(schedule=CosineSchedule()),
            "label": CategoricalProcess(
                schedule=..., invariant_probs=..., num_categories=10,
            ),
        }
    )
    ```

  Attributes:
    processes: A pytree of corruption processes matching the structure of the
      data.
  """

  processes: PyTree[corruption_base.CorruptionProcess]

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,
  ) -> DataTree:
    """Sample from the invariant distribution."""
    return utils.tree_map_with_key(
        lambda k, process, data: process.sample_from_invariant(k, data),
        key,
        self.processes,
        data_spec,
    )

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,
      time: TimeTree,
  ) -> tuple[DataTree, TargetInfoTree]:
    x0_structure = jax.tree.structure(x0)
    time_structure = jax.tree.structure(time)
    if x0_structure != time_structure:
      raise ValueError(
          f'x0 and time must have the same structure. Got: {x0_structure=} and'
          f' {time_structure=}'
      )
    xt_and_targets = utils.tree_map_with_key(
        lambda k, process, x, t: process.corrupt(k, x, t),
        key,
        self.processes,
        x0,
        time,
    )
    xt = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[0], x0, xt_and_targets
    )
    target_info = jax.tree.map(
        lambda x0, xt_and_targets: xt_and_targets[1], x0, xt_and_targets
    )
    return xt, target_info

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfoTree,
      xt: DataTree,
      time: TimeTree,
  ) -> TargetInfoTree:
    """Convert the prediction to the target type."""
    return jax.tree.map(
        lambda process, pred, xt, time: process.convert_predictions(
            pred, xt, time
        ),
        self.processes,
        prediction,
        xt,
        time,
    )

  @kt.typechecked
  def get_schedule_info(self, time: TimeTree) -> ScheduleInfoTree:
    """Get the schedule info for the given time."""
    return jax.tree.map(
        lambda process, t: process.get_schedule_info(t),
        self.processes,
        time,
    )


################################################################################
# MARK: NestedSamplerStep
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class NestedSamplerStep(sampling_base.SamplerStep):
  """Wrapper for a pytree of sampler steps mapped over the data.

  Usage Example:
    ```
    sampler_step = NestedSamplerStep(
        sampler_steps={
            "image": DDIMStep(),
            "label": DiscreteFlowMatchingStep(),
        }
    )
    ```

  Attributes:
    sampler_steps: A pytree of sampler steps matching the structure of the data.
  """

  sampler_steps: PyTree[sampling_base.SamplerStep]

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataTree,
      initial_step_info: sampling_base.StepInfoTree,
  ) -> sampling_base.DiffusionStepTree:
    return jax.tree.map(
        lambda stepper, init_noise, init_step_info: stepper.initialize(
            initial_noise=init_noise,
            initial_step_info=init_step_info,
        ),
        self.sampler_steps,
        initial_noise,
        initial_step_info,
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfoTree,
      current_step: sampling_base.DiffusionStepTree,
      next_step_info: sampling_base.StepInfoTree,
  ) -> sampling_base.DiffusionStepTree:
    return jax.tree.map(
        lambda stepper, pred, current, next_info: stepper.update(
            prediction=pred,
            current_step=current,
            next_step_info=next_info,
        ),
        self.sampler_steps,
        prediction,
        current_step,
        next_step_info,
    )

  @kt.typechecked
  def finalize(
      self,
      prediction: TargetInfoTree,
      current_step: sampling_base.DiffusionStepTree,
      last_step_info: sampling_base.StepInfoTree,
  ) -> sampling_base.DiffusionStepTree:
    return jax.tree.map(
        lambda stepper, pred, current, last_info: stepper.finalize(
            prediction=pred,
            current_step=current,
            last_step_info=last_info,
        ),
        self.sampler_steps,
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: NestedTimeSchedule
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedTimeSchedule(time_scheduling.TimeSchedule):
  """Wrapper to support a nested pytree of time schedules.

  The structure of the time schedule should match the structure of the data.

  Usage Example:
    ```
    time_schedule = NestedTimeSchedule(
        time_schedules={
            "image": UniformTimeSchedule(),
            "label": EDMTimeSchedule(rho=2.0),
        }
    )
    ```

  Attributes:
    time_schedules: A pytree of time schedules matching the structure of the
      data.
  """

  time_schedules: PyTree[time_scheduling.TimeSchedule]

  @kt.typechecked
  def all_step_infos(
      self,
      rng: PRNGKey,
      num_steps: int,
      data_spec: DataTree,
  ) -> sampling_base.StepInfoTree:
    def _call_schedule(rng, time_schedule, data_spec):
      return time_schedule.all_step_infos(rng, num_steps, data_spec)

    return utils.tree_map_with_key(
        _call_schedule, rng, self.time_schedules, data_spec
    )


################################################################################
# MARK: NestedDiffusionLoss
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedDiffusionLoss(loss_base.DiffusionLoss):
  """Wrapper for a pytree of loss functions mapped over the data.

  Enables using different loss functions for different input modalities.

  Usage Example:
    ```
    loss_fn = NestedDiffusionLoss(
        losses={
            "image": NoWeightGaussianLoss(prediction_type="x0"),
            "label": NoWeightDiscreteLoss(prediction_type="logits"),
        }
    )
    ```

  Attributes:
    losses: A pytree of loss functions matching the structure of the data.
  """

  losses: PyTree[loss_base.DiffusionLoss]

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfoTree,
      targets: TargetInfoTree,
      time: TimeTree,
  ) -> LossOutputTree:
    return jax.tree.map(
        lambda loss, target, pred, t: loss(
            preds=pred,
            targets=target,
            time=t,
        ),
        self.losses,
        targets,
        preds,
        time,
    )


################################################################################
# MARK: NestedTimeEmbedder
################################################################################


class NestedTimeEmbedder(nn.Module, conditioning_encoder.BaseTimeEmbedder):
  """Wrapper for a pytree of time embedders mapped over the time tree.

  Per-modality time embeddings are summed to produce a single embedding.

  Usage Example:
    ```
    time_embedder = NestedTimeEmbedder(
        time_embedders={
            "image": SinusoidalTimeEmbedder(
                activation="silu", embedding_dim=64, num_features=32,
            ),
            "label": SinusoidalTimeEmbedder(
                activation="silu", embedding_dim=64, num_features=32,
            ),
        }
    )
    ```

  Attributes:
    time_embedders: A pytree of time embedders matching the structure of the
      data.
  """

  time_embedders: PyTree[conditioning_encoder.BaseTimeEmbedder]

  @nn.compact
  @kt.typechecked
  def __call__(self, time: hd_typing.TimeTree) -> kt.Float['batch ...']:
    t_emb_tree = utils.lenient_map(
        lambda x, time_embedder: cast(nn.Module, time_embedder).copy()(x),
        time,
        self.time_embedders,
    )
    leaves, _ = jax.tree_util.tree_flatten(t_emb_tree)
    t_emb = jnp.sum(jnp.stack(leaves), axis=0)
    return t_emb


################################################################################
# MARK: NestedGuidanceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedGuidanceFn(guidance_lib.GuidanceFn):
  """Wrapper for a pytree of guidance functions mapped over the data.

  Usage Example:
    ```
    guidance_fn = NestedGuidanceFn(
        guidance_fns={
            "image": ScalarGuidanceFn(guidance=3.0),
            "label": ScalarGuidanceFn(guidance=1.0),
        }
    )
    ```

  Attributes:
    guidance_fns: A pytree of guidance functions matching the structure of the
      data.
  """

  guidance_fns: PyTree[guidance_lib.GuidanceFn]

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,
      conditioning: Conditioning,
      time: TimeTree,
      cond_outputs: TargetInfoTree,
      uncond_outputs: TargetInfoTree,
  ) -> TargetInfoTree:
    """Combine conditional and unconditional outputs."""
    return jax.tree.map(
        lambda guidance_fn, xt, time, cond_out, uncond_out: guidance_fn(
            xt=xt,
            conditioning=conditioning,
            time=time,
            cond_outputs=cond_out,
            uncond_outputs=uncond_out,
        ),
        self.guidance_fns,
        xt,
        time,
        cond_outputs,
        uncond_outputs,
    )


################################################################################
# MARK: NestedProjectionFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedProjectionFn(projection_lib.ProjectionFn):
  """Wrapper for a pytree of projection functions mapped over the data.

  Usage Example:
    ```
    projection_fn = NestedProjectionFn(
        projection_fns={
            "image": StaticThresholdProjectionFn(process=...),
            "label": IdentityProjectionFn(),
        }
    )
    ```

  Attributes:
    projection_fns: A pytree of projection functions matching the structure of
      the data.
  """

  projection_fns: PyTree[projection_lib.ProjectionFn]

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,
      conditioning: Conditioning,
      time: TimeTree,
      outputs: TargetInfoTree,
  ) -> TargetInfoTree:
    """Nested projection function."""
    return jax.tree.map(
        lambda projection_fn, xt, time, output: projection_fn(
            xt=xt,
            conditioning=conditioning,
            time=time,
            outputs=output,
        ),
        self.projection_fns,
        xt,
        time,
        outputs,
    )


################################################################################
# MARK: NestedTimeSampler
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedTimeSampler(time_sampling.TimeSampler):
  """Wrapper to support a nested pytree of time samplers.

  The structure of the samplers should match the structure of the data.

  Usage Example:
    ```
    time_sampler = NestedTimeSampler(
        samplers={
            "image": UniformTimeSampler(),
            "label": BetaTimeSampler(alpha=1.0, beta=1.0),
        }
    )
    ```

  Attributes:
    samplers: A pytree of time samplers matching the structure of the data.
  """

  samplers: PyTree[time_sampling.TimeSampler]

  @kt.typechecked
  def __call__(self, key: PRNGKey, data_spec: DataTree) -> TimeTree:
    def _call_sampler(key, sampler, data_spec):
      return sampler(key, data_spec)

    return utils.tree_map_with_key(_call_sampler, key, self.samplers, data_spec)


@dataclasses.dataclass(kw_only=True, frozen=True)
class JointNestedTimeSampler(time_sampling.TimeSampler):
  """Wrapper to support a nested pytree of time samplers.

  The structure of the samplers should match the structure of the data.
  Contrary to NestedTimeSampler, the samplers are called with a joint key.

  Usage Example:
    ```
    time_sampler = JointNestedTimeSampler(
        samplers={
            "image": UniformTimeSampler(),
            "label": BetaTimeSampler(alpha=1.0, beta=1.0),
        }
    )
    ```

  Attributes:
    samplers: A pytree of time samplers matching the structure of the data.
  """

  samplers: PyTree[time_sampling.TimeSampler]

  @kt.typechecked
  def __call__(self, key: PRNGKey, data_spec: DataTree) -> TimeTree:
    def _call_sampler(sampler, data_spec):
      return sampler(key, data_spec)

    return jax.tree.map(_call_sampler, self.samplers, data_spec)
