# Copyright 2025 Hackable Diffusion Authors.
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

"""Actual implementation of the sampling steps.

This module proposes various implementations but they all have in common
the core logic:

* An `initialize` function that takes a starting state and returns the
  first step of the diffusion process.
* An `update` function that takes the current state and returns the next step.
* A `finalize` function that takes the last state and returns the final
  state.

At every step, the update function takes the current state and returns the next
state. The update is also in charge of computing other auxiliary informations
such as volatility, drifts, etc.

The `InferenceFn is also called within the step and converted into the
relevant representation, for instance score, velocity, etc.
"""

import dataclasses
from typing import Protocol

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import discrete
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
from hackable_diffusion.lib.sampling import base
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

DiffusionStep = base.DiffusionStep
StepInfo = base.StepInfo
SamplerStep = base.SamplerStep

CategoricalProcess = discrete.CategoricalProcess
DiscreteSchedule = schedules.DiscreteSchedule

################################################################################
# MARK: Remasking strategy
################################################################################


class RemaskingFn(Protocol):
  """Remasking strategy protocol.

  We follow the original implementation of remasking in
  https://arxiv.org/abs/2503.00307.

  We return the probability of unmasking in the forward process. This is denoted
  σ(t) in https://arxiv.org/abs/2503.00307. The return probability has the same
  shape as the input time. Note that σ(t) in https://arxiv.org/abs/2503.00307 is
  in fact dependent of `s` (the next time) and `t` (the current time).
  """

  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    """Returns the probability of unmasking."""
    ...


@dataclasses.dataclass(kw_only=True, frozen=True)
class NoRemaskingFn(RemaskingFn):
  """No remasking strategy."""

  @typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    return jnp.zeros_like(s)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MaxCappedRemaskingFn(RemaskingFn):
  """Max-capped remasking strategy.

  This is the original implementation of remasking in
  https://arxiv.org/abs/2503.00307. We follow the implementation of the switch
  function in the paper with the "max-capped" schedule (see Section 4.1). On
  top of this remasking strategy, we consider a switch to turn on and off the
  remasking based on the time, see Section 4.2.

  Attributes:
    schedule: The schedule to use for the corruption process.
    max_cap: The maximum value of the remasking probability.
    switch_min: The minimum value of the switch function.
    switch_max: The maximum value of the switch function.
  """

  schedule: DiscreteSchedule
  max_cap: float = 0.0
  switch_min: float = 0.0
  switch_max: float = 1.0

  def __post_init__(self):
    if self.switch_min > self.switch_max:
      raise ValueError(
          'switch_min must be smaller than switch_max, got'
          f' {self.switch_min} > {self.switch_max}'
      )
    if self.max_cap < 0.0:
      raise ValueError(f'max_cap must be non-negative, got {self.max_cap}')

  @typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    alpha_s = self.schedule.alpha(s)
    alpha_t = self.schedule.alpha(t)
    return jnp.minimum((1.0 - alpha_s) / alpha_t, self.max_cap)


@dataclasses.dataclass(kw_only=True, frozen=True)
class RescaledRemaskingFn(RemaskingFn):
  """Rescaled remasking strategy.

  This is the original implementation of remasking in
  https://arxiv.org/abs/2503.00307. We follow the implementation of the switch
  function in the paper with the "rescaled" schedule (see Section 4.1). On top
  of this remasking strategy, we consider a switch to turn on and off the
  remasking based on the time, see Section 4.2.

  Attributes:
    schedule: The schedule to use for the corruption process.
    rescale_factor: The rescale factor to apply to the remasking probability.
    switch_min: The minimum value of the switch function.
    switch_max: The maximum value of the switch function.
  """

  def __post_init__(self):
    if self.switch_min > self.switch_max:
      raise ValueError(
          'switch_min must be smaller than switch_max, got'
          f' {self.switch_min} > {self.switch_max}'
      )
    if self.rescale_factor <= 0.0:
      raise ValueError(
          f'rescale_factor must be positive, got {self.rescale_factor}'
      )
    if self.rescale_factor >= 1.0:
      raise ValueError(
          f'rescale_factor must be smaller than 1.0, got {self.rescale_factor}'
      )

  schedule: DiscreteSchedule
  rescale_factor: float = 1.0
  switch_min: float = 0.0
  switch_max: float = 1.0

  @typechecked
  def __call__(self, s: TimeArray, t: TimeArray) -> TimeArray:
    alpha_s = self.schedule.alpha(s)
    alpha_t = self.schedule.alpha(t)
    return self.rescale_factor * jnp.minimum((1.0 - alpha_s) / alpha_t, 1.0)


################################################################################
# MARK: Corrupted mask function
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class CorruptedMaskFn(Protocol):
  """Corrupted mask function protocol.

  This function takes the current xt, i.e., the "noisy" data and returns a mask
  which indicates which regions are corrupted. The mask is True if the region is
  corrupted and False otherwise. In the mask setting, this simply corresponds to
  identifying the tokens which have the mask value. More complex masking schemes
  can be defined as in https://arxiv.org/abs/2410.06264.
  """

  def __call__(self, xt: DataArray) -> DataArray:
    """Returns the corrupted mask."""
    ...


@dataclasses.dataclass(kw_only=True, frozen=True)
class AllCorruptedMaskFn(CorruptedMaskFn):
  """Assume all tokens are corrupted."""

  @typechecked
  def __call__(self, xt: DataArray) -> DataArray:
    return jnp.ones_like(xt, dtype=jnp.bool_)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MaskValueCorruptedMaskFn(CorruptedMaskFn):
  """Corrupted mask function based on the mask value.

  Note that this function only makes sense for masking processes.
  """

  process: CategoricalProcess

  def __post_init__(self):
    """MaskValueCorruptedMaskFn only supports masking processes."""
    if not self.process.is_masking:
      raise ValueError(
          'MaskValueCorruptedMaskFn only supports masking processes.'
      )

  @typechecked
  def __call__(self, xt: DataArray) -> DataArray:
    mask_value = self.process.process_num_categories - 1
    return xt == mask_value


################################################################################
# MARK: UnMasking Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class UnMaskingStep(SamplerStep):
  """Unmasking step following https://arxiv.org/abs/2406.04329.

  Attributes:
    corruption_process: The corruption process to use.
    remasking_fn: The remasking function to use, see
      https://arxiv.org/abs/2503.00307v1. This is optional with the default
        being no remasking.
    corruption_mask_fn: The corrupted mask function to use. This is optional
      with the default being all tokens corrupted.
    temperature: The temperature to use. This is optional with the default being
      a temperature of 1.0.
  """

  corruption_process: CategoricalProcess
  remasking_fn: RemaskingFn = NoRemaskingFn()
  corruption_mask_fn: CorruptedMaskFn = AllCorruptedMaskFn()
  temperature: float = 1.0

  def __post_init__(self):
    """UnMaskingStep only supports masking processes, see update for more details."""
    if not self.corruption_process.is_masking:
      raise ValueError('UnMaskingStep only supports masking processes.')

  @property
  def mask_value(self) -> int:
    return self.corruption_process.num_categories - 1

  @property
  def unused_mask_value(self) -> int:
    return self.corruption_process.unused_mask_value

  @property
  def post_corruption_fn(self) -> discrete.PostCorruptionFn:
    return self.corruption_process.post_corruption_fn

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    xt = current_step.xt

    unused_mask = xt == self.unused_mask_value
    # The mask is True if the token is unused.

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)
    key = next_step_info.rng

    # Sample from p_{0|t}

    logits = self.corruption_process.convert_predictions(
        prediction,
        xt,
        time,
    )['logits']
    logits = logits / self.temperature

    key, subkey = jax.random.split(key)
    sample = jax.random.categorical(key=subkey, logits=logits)[..., None]
    # (bsz, *seq_len, 1)

    # Split xt into masked and unmasked regions

    currently_masked = self.corruption_mask_fn(xt)
    currently_unmasked = jnp.invert(currently_masked)

    # Denoising

    alpha_s = self.corruption_process.schedule.alpha(next_time)
    alpha_t = self.corruption_process.schedule.alpha(time)

    p_st = self.remasking_fn(s=next_time, t=time)

    prob = (alpha_s - (1.0 - p_st) * alpha_t) / (1.0 - alpha_t)
    # Denoising probability following https://arxiv.org/abs/2503.00307v1
    # If no remasking, p_st = 0, so prob = (alpha_s - alpha_t) / (1.0 - alpha_t)
    prob = jnp.broadcast_to(prob, currently_masked.shape)

    key, subkey = jax.random.split(key)
    to_unmask = currently_masked * jax.random.bernoulli(subkey, prob)

    new_xt = jnp.where(to_unmask, sample, xt)

    # Renoising following https://arxiv.org/abs/2503.00307
    key_noise, key_remask = jax.random.split(key)
    noise_sample = self.corruption_process.sample_from_invariant(
        key=key_noise,
        data_spec=xt,
    )

    p_st = jnp.broadcast_to(p_st, currently_unmasked.shape)
    to_remask = currently_unmasked * jax.random.bernoulli(key_remask, p_st)

    new_xt = jnp.where(to_remask, noise_sample, new_xt)
    new_xt = self.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_mask_value.
    new_xt = jnp.where(unused_mask, self.unused_mask_value, new_xt)

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class DiscreteDDIMStep(SamplerStep):
  """This is the discrete version of the DDIM step."""

  corruption_process: CategoricalProcess
  temperature: float = 1.0

  def __post_init__(self):
    """DiscreteDDIMStep does not support masking processes, see update for more details."""
    if self.corruption_process.is_masking:
      raise ValueError('DiscreteDDIMStep does not support masking processes.')

  @property
  def mask_value(self) -> int:
    return self.corruption_process.num_categories - 1

  @property
  def unused_mask_value(self) -> int:
    return self.corruption_process.unused_mask_value

  @property
  def post_corruption_fn(self) -> discrete.PostCorruptionFn:
    return self.corruption_process.post_corruption_fn

  @property
  def invariant_probs_vec(self) -> Float['M']:
    return self.corruption_process.invariant_probs_vec

  @property
  def process_num_categories(self) -> int:
    return self.corruption_process.process_num_categories

  @typechecked
  def initialize(
      self,
      initial_noise: DataArray,
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux=dict(),
    )

  @typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    xt = current_step.xt

    unused_mask = xt == self.unused_mask_value
    # The mask is True if the token is unused.

    time = current_step_info.time
    next_time = next_step_info.time
    time = utils.bcast_right(time, xt.ndim)
    next_time = utils.bcast_right(next_time, xt.ndim)
    key = next_step_info.rng

    # Sample from p_{0|t}
    logits = self.corruption_process.convert_predictions(
        prediction,
        xt,
        time,
    )['logits']
    logits = logits / self.temperature

    x0 = jax.random.categorical(key=key, logits=logits)[..., None]
    # (bsz, *seq_len, 1)
    key, _ = jax.random.split(key)

    # Compute the probability vector

    xt_oh = jax.nn.one_hot(xt[..., 0], num_classes=self.process_num_categories)
    x0_oh = jax.nn.one_hot(x0[..., 0], num_classes=self.process_num_categories)
    # (bsz, *seq_len, M)

    alpha_s = self.corruption_process.schedule.alpha(next_time)
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = jnp.broadcast_to(alpha_s, x0_oh.shape)
    alpha_t = jnp.broadcast_to(alpha_t, x0_oh.shape)
    ratio = alpha_t / alpha_s
    # (bsz, *seq_len, M)

    first_logit = jnp.log(
        ratio * xt_oh + (1.0 - ratio) * self.invariant_probs_vec[xt]
    )
    second_logit = jnp.log(
        alpha_s * x0_oh + (1.0 - alpha_s) * self.invariant_probs_vec
    )
    total_logit = first_logit + second_logit
    # Do not use this sampler for masking.
    # What could happen is xt is unmasked (assume at first position) so the
    # first logits (first_logit) is [value, -inf, ..., -inf]. Then assume
    # that the predictionfor x0 is different than xt
    # (can never happen in unmasking),assume that the second position is the
    # one chosen by the x0 predictor. Then we have for the second logits
    # (second_logit): [-inf, value, -inf, ..., -inf, value_mask].
    # So when we add them together we get [-inf, ..., -inf].
    # jax.random.categorical will then return the first position.
    # This is not what we want and this behavior should not be accepted.

    # Sample from the distribution defined by logits
    new_xt = jax.random.categorical(key=key, logits=total_logit)[..., None]
    new_xt = self.post_corruption_fn(new_xt)

    # Replace the unused tokens with the unused_mask_value.
    new_xt = jnp.where(unused_mask, self.unused_mask_value, new_xt)

    return DiffusionStep(
        xt=new_xt,
        step_info=next_step_info,
        aux=dict(),
    )

  @typechecked
  def finalize(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      last_step_info: StepInfo,
  ) -> DiffusionStep:
    return self.update(
        prediction,
        current_step,
        last_step_info,
    )
