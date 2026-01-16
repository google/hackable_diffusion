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

"""Discrete noise processes."""

from __future__ import annotations

import dataclasses
import enum
from typing import Protocol, Sequence

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import base
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

CorruptionProcess = base.CorruptionProcess
DiscreteSchedule = schedules.DiscreteSchedule

################################################################################
# MARK: Enums
################################################################################


class SamplingPrecisionMode(enum.StrEnum):
  """Sampling precision mode.

  See
  https://docs.jax.dev/en/latest/_autosummary/jax.random.choice.html#jax.random.choice
  for more details about how `mode` is used in random samplers.
  """

  HIGH = 'high'
  LOW = 'low'


################################################################################
# MARK: Projection functions
################################################################################


class PostCorruptionFn(Protocol):
  """Post corruption function protocol.

  The purpose of a post corruption function is to project the labels on a new
  space. For instance in the case of the adjacency graph, we use a symmetric
  projection function, so that the noisy labels are also symmetric. This is used
  in DiGress https://arxiv.org/abs/2209.14734u
  """

  def __call__(self, x: DataArray) -> DataArray:
    """Project the labels."""
    ...


class IdentityPostCorruptionFn(PostCorruptionFn):
  """Identity post corruption function."""

  def __call__(self, x: DataArray) -> DataArray:
    """Project the labels."""
    return x


class SymmetricPostCorruptionFn(PostCorruptionFn):
  """Symmetric projection function.

  This is used in DiGress https://arxiv.org/abs/2209.14734 in order to noise the
  adjacency graph. This function also zeroes out the diagonal entries, thereby
  removing any self-loop.
  """

  def __call__(self, x: DataArray) -> DataArray:
    """Project the labels."""
    if x.ndim != 4:
      raise ValueError(f'Expected 4D input, got {x.ndim=}.')
    if x.shape[1] != x.shape[2]:
      raise ValueError(
          f'Expected square input, got {x.shape[1]=} and {x.shape[2]=}.'
      )
    x_without_trail = x[..., 0]
    # Take the upper triangle of the input.
    x_without_trail_tri = jnp.triu(x_without_trail, k=1)
    # Symmetric projection is the sum of the upper triangle and its transpose.
    x_without_trail_tri_sym = x_without_trail_tri + jnp.transpose(
        x_without_trail_tri, axes=(0, 2, 1)
    )
    x_sym = x_without_trail_tri_sym[..., None]
    return x_sym


################################################################################
# MARK: CategoricalProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class CategoricalProcess(CorruptionProcess):
  """Discrete noise processes that corrupt towards a categorical distribution.

  We are mostly using two special cases of this process:
  - masking: invariant_probs = (0.0,) * K + (1.0,)
  - uniform: invariant_probs = (1.0 / K,) * K
  where K is the number of categories.

  Attributes:
    schedule: The schedule to use for the corruption process.
    invariant_probs: The invariant probability distribution of the process. At
      time one, the process will corrupt towards the distribution defined by
      invariant_probs.
    num_categories: The number of categories in the distribution. Note that this
      might be different from the length of invariant_probs, which might contain
      K+1 elements in the case of masking.
    unused_mask_value: If a token is unused then it should have this value. Note
      that we require that this unused_mask_value is not in the range of the
      vocabulary, i,e., unused_mask_value < 0 or unused_mask_value >=
      len(invariant_probs) (which is the same as process_num_categories).
    post_corruption_fn: The projection function to use for the corruption
      process. This is a function applied at the end of the corruption process.
      It projects the labels on a new space. For instance in the case of the
      adjacency graph, we use a symmetric projection function, so that the noisy
      labels are also symmetric.
    mode: The mode to use in `jax.random.choice` and `jax.random.bernoulli`. Can
      be set to "high" or "low" for how many bits to use in the Gumbel sampler.
      See https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.choice
      for more information.
  """

  schedule: DiscreteSchedule
  invariant_probs: Sequence[float]
  num_categories: int
  unused_mask_value: int = -1
  post_corruption_fn: PostCorruptionFn = IdentityPostCorruptionFn()
  mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH

  def __post_init__(self):
    if (
        self.unused_mask_value >= 0
        and self.unused_mask_value < self.process_num_categories
    ):
      raise ValueError(
          'unused_mask_value must be outside of the range of the vocabulary.'
          f' Got: {self.unused_mask_value=} and {self.num_categories=}'
      )

  ##############################################################################
  # MARK: Properties
  ##############################################################################

  @property
  def invariant_probs_vec(self) -> Float['M']:
    """Returns the invariant probability distribution as a vector."""
    return jnp.array(self.invariant_probs)

  @property
  def process_num_categories(self) -> int:
    """Returns the number of categories in the process.

    Note that this might be different from the number of categories in the
    distribution, which might contain K+1 elements in the case of masking.
    """
    return len(self.invariant_probs)

  @property
  def is_masking(self) -> bool:
    """Returns whether the process is masking."""
    if self.process_num_categories == self.num_categories:
      return False
    else:
      invariant_probs_masking = (0.0,) * self.num_categories + (1.0,)
      invariant_probs_masking_vec = jnp.array(invariant_probs_masking)
      return jnp.all(
          self.invariant_probs_vec == invariant_probs_masking_vec
      ).item()

  ##############################################################################
  # MARK: Methods
  ##############################################################################

  @typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,
  ) -> DataArray:
    """Sample from the invariant distribution."""
    return jax.random.choice(
        key,
        a=self.process_num_categories,
        p=self.invariant_probs_vec,
        shape=data_spec.shape,
        mode=self.mode,
    )

  @typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,
      time: TimeArray,
  ) -> tuple[DataArray, TargetInfo]:
    """Corrupt the data according to the schedule and invariant probs.

    The target information contains:
    - x0: The uncorrupted data.
    - logits: The logits of the corrupted data.
    - mask: The mask of the corruption.
    Note that the mask is True if the original data is present in x0. The mask
    is False if the original data is replaced by noise.

    Args:
      key: The random key.
      x0: The uncorrupted data.
      time: The time of the corruption.

    Returns:
      xt: The corrupted data.
      target_info: The target info for the corrupted data.
    """
    # Broadcast the time to a shape compatible with x0.
    time = utils.bcast_right(time, x0.ndim)

    # compute alpha
    alpha = self.schedule.alpha(time)

    # get the unused mask
    unused_mask = x0 == self.unused_mask_value
    # The mask is True if the token is unused.

    # corrupt x0 with probability alpha
    # We must have alpha of the same shape as x0, since each pixel can be
    # corrupted independently.
    alpha_bcast = jnp.broadcast_to(alpha, x0.shape)
    assert alpha_bcast.shape == x0.shape
    mask = jax.random.bernoulli(key, p=alpha_bcast, mode=self.mode)
    key, _ = jax.random.split(key)

    # compute noise vector
    noise = self.sample_from_invariant(key, data_spec=x0)

    # noise x0 with probability alpha
    xt = jnp.where(mask, x0, noise)  # mask = (xt == x0)
    xt = self.post_corruption_fn(xt)

    logits = jax.nn.one_hot(x0[..., 0], self.num_categories)
    target_info = {
        'x0': x0,  # Int[*b 1]; Uncorrupted input data.
        'logits': logits,  # Float[*b K] one-hot encoding of x0.
        'mask': mask,  # Bool[*b 1] mask of the corruption.
        'unused_mask': unused_mask,  # Bool[*b 1] mask of the unused tokens.
    }

    # Replace the unused tokens with the unused_mask_value.
    xt = jnp.where(unused_mask, self.unused_mask_value, xt)

    return xt, target_info

  @typechecked
  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,
      time: TimeArray,
  ) -> TargetInfo:
    del time  # Unused
    if len(prediction) != 1 or 'logits' not in prediction:
      raise KeyError(
          f'Only logits prediction is supported. Got: {prediction.keys()=}'
      )
    logits = prediction['logits']
    x0_pred = jnp.argmax(logits, axis=-1)
    x0_pred = jnp.expand_dims(x0_pred, axis=-1)
    return {
        'x0': x0_pred,  # Int[*b 1]; Argmax of the predicted distribution.
        'logits': logits,  # Float[*b K]; Raw logits
    }

  @typechecked
  def get_schedule_info(self, time: TimeArray) -> dict[str, TimeArray]:
    """Get the schedule info for the given time."""
    return self.schedule.evaluate(time)

  ##############################################################################
  # MARK: Factory methods
  ##############################################################################

  @classmethod
  def masking_process(
      cls,
      schedule: DiscreteSchedule,
      num_categories: int,
      unused_mask_value: int = -1,
      post_corruption_fn: PostCorruptionFn = IdentityPostCorruptionFn(),
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
  ) -> CategoricalProcess:
    """Create a CategoricalProcess from a schedule and invariant probs."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )

    invariant_probs = (0.0,) * num_categories + (1.0,)
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_mask_value=unused_mask_value,
        post_corruption_fn=post_corruption_fn,
        mode=mode,
    )

  @classmethod
  def uniform_process(
      cls,
      schedule: DiscreteSchedule,
      num_categories: int,
      unused_mask_value: int = -1,
      post_corruption_fn: PostCorruptionFn = IdentityPostCorruptionFn(),
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
  ) -> CategoricalProcess:
    """Create a CategoricalProcess from a schedule and invariant probs."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )
    invariant_probs = (1.0 / num_categories,) * num_categories
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_mask_value=unused_mask_value,
        post_corruption_fn=post_corruption_fn,
        mode=mode,
    )
