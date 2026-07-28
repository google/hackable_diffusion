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
`jax_helpers.lenient_map`. This means any combination of modalities and nesting
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
    - ``NestedFlaxLinenInferenceFn``  — Linen model wrapper
    - ``NestedGuidedDiffusionInferenceFn`` — guided inference

  Architecture:
    - ``NestedTimeEmbedder``  — per-modality time embeddings

  Network:
    - ``MultiModalDiffusionNetwork`` — see ``diffusion_network.py``
"""

from __future__ import annotations

import dataclasses
from typing import cast

import flax.linen as nn
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.inference import guidance as guidance_lib
from hackable_diffusion.lib.inference import projection as projection_lib
from hackable_diffusion.lib.sampling import time_scheduling
from hackable_diffusion.lib.training import time_sampling
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
LossOutputTree = hd_typing.LossOutputTree
ScheduleInfoTree = hd_typing.ScheduleInfoTree
TargetInfoTree = hd_typing.TargetInfoTree
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree
ShapeTree = hd_typing.ShapeTree
ConditioningShape = hd_typing.ConditioningShape


StepInfoTree = PyTree[hd_api.StepInfo]
DiffusionStepTree = PyTree[hd_api.DiffusionStep]


################################################################################
# MARK: NestedProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedSchedule(hd_api.CorruptionSchedule):
  """Wrapper for a pytree of corruption schedules.

  Enables using different corruption schedules for different input modalities.
  """

  schedules: PyTree[hd_api.CorruptionSchedule]  # pyrefly: ignore[not-a-type]

  def evaluate(self, time: TimeTree) -> ScheduleInfoTree:  # pyrefly: ignore[not-a-type]
    """Evaluate the schedule for a given time. Return a dictionary of info."""
    return jax.tree.map(
        lambda schedule, t: schedule.evaluate(t),
        self.schedules,
        time,
    )


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedProcess(hd_api.CorruptionProcess):
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

  processes: PyTree[hd_api.CorruptionProcess]  # pyrefly: ignore[not-a-type]

  @property
  def schedule(self) -> hd_api.CorruptionSchedule:
    return NestedSchedule(
        schedules=jax.tree.map(lambda p: p.schedule, self.processes)
    )

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataTree,  # pyrefly: ignore[not-a-type]
  ) -> DataTree:  # pyrefly: ignore[not-a-type]
    """Sample from the invariant distribution."""
    return jax_helpers.tree_map_with_key(
        lambda k, process, data: process.sample_from_invariant(k, data),
        key,
        self.processes,
        data_spec,
    )

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataTree,  # pyrefly: ignore[not-a-type]
      time: TimeTree,  # pyrefly: ignore[not-a-type]
  ) -> tuple[DataTree, TargetInfoTree]:  # pyrefly: ignore[not-a-type]
    x0_structure = jax.tree.structure(x0)
    time_structure = jax.tree.structure(time)
    if x0_structure != time_structure:
      raise ValueError(
          f'x0 and time must have the same structure. Got: {x0_structure=} and'
          f' {time_structure=}'
      )
    xt_and_targets = jax_helpers.tree_map_with_key(
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
      prediction: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      time: TimeTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
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




################################################################################
# MARK: NestedSamplerStep
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class NestedSamplerStep(hd_api.SamplerStep):
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

  sampler_steps: PyTree[hd_api.SamplerStep]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataTree,  # pyrefly: ignore[not-a-type]
      initial_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
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
      prediction: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      current_step: DiffusionStepTree,  # pyrefly: ignore[not-a-type]
      next_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
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
      prediction: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      current_step: DiffusionStepTree,  # pyrefly: ignore[not-a-type]
      last_step_info: StepInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> DiffusionStepTree:  # pyrefly: ignore[not-a-type]
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

  time_schedules: PyTree[time_scheduling.TimeSchedule]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def all_step_infos(
      self,
      rng: PRNGKey,
      num_steps: int,
      data_spec: DataTree,  # pyrefly: ignore[not-a-type]
  ) -> StepInfoTree:  # pyrefly: ignore[not-a-type]
    def _call_schedule(rng, time_schedule, data_spec):
      return time_schedule.all_step_infos(rng, num_steps, data_spec)

    return jax_helpers.tree_map_with_key(
        _call_schedule, rng, self.time_schedules, data_spec
    )


################################################################################
# MARK: NestedDiffusionLoss
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedDiffusionLoss(hd_api.DiffusionLoss):
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

  losses: PyTree[hd_api.DiffusionLoss]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(
      self,
      preds: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      targets: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      time: TimeTree,  # pyrefly: ignore[not-a-type]
  ) -> LossOutputTree:  # pyrefly: ignore[not-a-type]
    return jax.tree.map(
        lambda loss, pred, target, t: loss(
            preds=pred,
            targets=target,
            time=t,
        ),
        self.losses,
        preds,
        targets,
        time,
    )


################################################################################
# MARK: NestedTimeEmbedder
################################################################################


class NestedTimeEmbedder(nn.Module, conditioning_encoder.TimeEmbedder):
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

  time_embedders: PyTree[conditioning_encoder.TimeEmbedder]  # pyrefly: ignore[not-a-type]

  @nn.compact
  @kt.typechecked
  def __call__(self, time: hd_typing.TimeTree) -> kt.Float['batch ...']:  # pyrefly: ignore[not-a-type]
    t_emb_tree = jax_helpers.lenient_map(
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

  guidance_fns: PyTree[guidance_lib.GuidanceFn]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      cond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
      uncond_outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
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

  projection_fns: PyTree[projection_lib.ProjectionFn]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(
      self,
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      outputs: TargetInfoTree,  # pyrefly: ignore[not-a-type]
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
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
# MARK: NestedFlaxLinenInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedFlaxLinenInferenceFn(hd_api.InferenceFn):
  """Inference function for multimodal (nested/PyTree) data with nn.Module.

  This is the multimodal counterpart of ``wrappers.FlaxLinenInferenceFn``.
  It uses ``DataTree`` / ``TimeTree`` type annotations so that
  ``@kt.typechecked`` accepts PyTree inputs.

  Attributes:
    network: The Flax Linen module (e.g. ``MultiModalDiffusionNetwork``).
    params: The model parameters.
  """

  network: nn.Module
  params: PyTree  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""
    return self.network.apply(
        {'params': self.params},
        time=time,
        xt=xt,
        conditioning=conditioning,
        is_training=False,
    )


################################################################################
# MARK: NestedGuidedDiffusionInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class NestedGuidedDiffusionInferenceFn(hd_api.InferenceFn):
  """Guided diffusion inference function for multimodal (nested/PyTree) data.

  This is the multimodal counterpart of
  ``diffusion_inference.GuidedDiffusionInferenceFn``.  It accepts PyTree-valued
  ``time`` and ``xt`` arguments and delegates guidance and projection to
  tree-aware ``NestedGuidanceFn`` / ``NestedProjectionFn``.

  The ``base_inference_fn`` is expected to accept tree-valued inputs directly
  (e.g. a ``NestedFlaxLinenInferenceFn`` wrapping a ``MultiModalDiffusionNetwork``).

  Usage Example:
    ```
    nested_inference_fn = NestedGuidedDiffusionInferenceFn(
        base_inference_fn=NestedFlaxLinenInferenceFn(
            network=multi_modal_network, params=params
        ),
        guidance_fn=NestedGuidanceFn(
            guidance_fns={
                "image": ScalarGuidanceFn(guidance=3.0),
                "label": ScalarGuidanceFn(guidance=1.0),
            }
        ),
        projection_fn=NestedProjectionFn(
            projection_fns={
                "image": IdentityProjectionFn(),
                "label": IdentityProjectionFn(),
            }
        ),
    )
    ```

  Attributes:
    base_inference_fn: The base inference function (must accept tree-valued
      inputs).
    guidance_fn: A tree-aware guidance function.
    projection_fn: A tree-aware projection function.
  """

  base_inference_fn: NestedFlaxLinenInferenceFn
  guidance_fn: NestedGuidanceFn
  projection_fn: NestedProjectionFn

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs with guidance and projection."""

    cond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=conditioning,
    )
    uncond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=None,
    )

    guided_outputs = self.guidance_fn(
        xt=xt,
        conditioning=conditioning,  # pyrefly: ignore[bad-argument-type]
        time=time,
        cond_outputs=cond_outputs,
        uncond_outputs=uncond_outputs,
    )

    projected_outputs = self.projection_fn(
        xt=xt,
        conditioning=conditioning,  # pyrefly: ignore[bad-argument-type]
        time=time,
        outputs=guided_outputs,
    )
    return projected_outputs



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

  samplers: PyTree[time_sampling.TimeSampler]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(self, key: PRNGKey, data_spec: DataTree) -> TimeTree:  # pyrefly: ignore[not-a-type]
    def _call_sampler(key, sampler, data_spec):
      return sampler(key, data_spec)

    return jax_helpers.tree_map_with_key(
        _call_sampler, key, self.samplers, data_spec
    )


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

  samplers: PyTree[time_sampling.TimeSampler]  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(self, key: PRNGKey, data_spec: DataTree) -> TimeTree:  # pyrefly: ignore[not-a-type]
    def _call_sampler(sampler, data_spec):
      return sampler(key, data_spec)

    return jax.tree.map(_call_sampler, self.samplers, data_spec)


################################################################################
# MARK: NestedSelfConditioningDiffusionNetwork
################################################################################


class NestedSelfConditioningDiffusionNetwork(
    nn.Module, diffusion_network.DiffusionNetwork
):
  """Multi-modal DiffusionNetwork with self-conditioning on predicted logits.

  This class generalizes `SelfConditioningDiffusionNetwork` to PyTree data.
  It assumes ALL modalities in the PyTree are discrete and require
  self-conditioning.

  Attributes:
    backbone_network: The backbone network to use. Must accept PyTree inputs
      where each leaf has concatenated logits.
    conditioning_encoder: The conditioning encoder to use.
    prediction_type: PyTree of strings, all must be 'logits'.
    processes: A NestedProcess containing the corruption processes for each
      modality (used to get `num_categories`).
    self_cond_prob: Probability of applying self-conditioning during training.
    data_dtype: PyTree of dtypes for each modality.
    input_rescaler: Optional PyTree of input rescalers.
    time_rescaler: Optional PyTree of time rescalers.
    rng_collection: PRNG collection name for the self-conditioning mask.
  """

  backbone_network: diffusion_network.ConditionalBackbone
  conditioning_encoder: conditioning_encoder.ConditioningEncoder
  prediction_type: PyTree[str]  # pyrefly: ignore[not-a-type]
  processes: NestedProcess
  self_cond_prob: float = 0.5
  data_dtype: PyTree[DType] = jnp.float32  # pyrefly: ignore[not-a-type]
  input_rescaler: PyTree[diffusion_network.InputRescaler | None] | None = None  # pyrefly: ignore[not-a-type]
  time_rescaler: PyTree[diffusion_network.TimeRescaler | None] | None = None  # pyrefly: ignore[not-a-type]
  rng_collection: str = 'self_conditioning'

  def __post_init__(self):
    super().__post_init__()

    # Verify all prediction types are 'logits'
    def _check_logits(pred_type):
      if pred_type != 'logits':
        raise ValueError(
            f"All prediction types must be 'logits', got {pred_type}"
        )

    jax.tree.map(_check_logits, self.prediction_type)

  @nn.compact
  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
      is_training: bool,
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]

    # 1. Rescale time and input (handling PyTrees)
    if self.time_rescaler is not None:
      time_rescaled = jax_helpers.lenient_map(
          lambda t, tr: tr(t) if tr is not None else t, time, self.time_rescaler
      )
    else:
      time_rescaled = time

    if self.input_rescaler is not None:
      xt_rescaled = jax_helpers.lenient_map(
          lambda t, x, ir: ir(t, x) if ir is not None else x,
          time,
          xt,
          self.input_rescaler,
      )
    else:
      xt_rescaled = xt

    # 2. Encode conditioning
    conditioning_embeddings = self.conditioning_encoder(
        time=time_rescaled,
        conditioning=conditioning,
        is_training=is_training,
    )

    # 3. Create zero logits for each leaf in the data tree
    def _create_zero_logits(xt_leaf, process_leaf):
      # Assumes process_leaf has `num_categories`
      return jnp.zeros(
          xt_leaf.shape[:-1] + (process_leaf.num_categories,),
          dtype=xt_leaf.dtype,
      )

    zero_logits = jax.tree.map(
        _create_zero_logits, xt_rescaled, self.processes.processes
    )

    # 4. First pass: run with zero logits
    xt_with_zeros = jax.tree.map(
        lambda x, z: jnp.concatenate([x, z], axis=-1), xt_rescaled, zero_logits
    )

    backbone_module = self.backbone_network

    first_output = backbone_module(
        x=xt_with_zeros,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    x0_hat_logits = jax.tree.map(jax.lax.stop_gradient, first_output)

    # 5. Apply self-conditioning mask during training
    if is_training:
      # We assume a global mask for the entire batch across all modalities
      # Find a leaf to get the batch size
      flat_xt, _ = jax.tree_util.tree_flatten(xt)
      batch_size = flat_xt[0].shape[0]

      do_self_cond = (
          jax.random.uniform(
              self.make_rng(self.rng_collection), shape=(batch_size,)
          )
          < self.self_cond_prob
      )

      def _apply_mask(logits_leaf, zero_leaf):
        # Broadcast mask to match leaf dimensions
        mask = do_self_cond.reshape(
            (batch_size,) + (1,) * (logits_leaf.ndim - 1)
        )
        return jnp.where(mask, logits_leaf, zero_leaf)

      x0_hat_logits = jax.tree.map(_apply_mask, x0_hat_logits, zero_logits)

    # 6. Second pass: run with predicted logits concatenated
    xt_with_x0_hat_logits = jax.tree.map(
        lambda x, l: jnp.concatenate([x, l], axis=-1),
        xt_rescaled,
        x0_hat_logits,
    )

    backbone_outputs = backbone_module(
        x=xt_with_x0_hat_logits,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    # 7. Wrap outputs in prediction type structure
    return jax_helpers.lenient_map(
        lambda out, pred_type: {pred_type: out},
        backbone_outputs,
        self.prediction_type,
    )
