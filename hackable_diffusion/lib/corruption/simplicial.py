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

"""Simplicial noise processes."""

from __future__ import annotations

import dataclasses
import enum
from typing import Sequence

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import random_utils
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
SimplicialSchedule = schedules.SimplicialSchedule

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
# MARK: CategoricalProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimplicialProcess(CorruptionProcess):
  """Simplicial noise processes that corrupt towards a Dirichlet distribution.

  We are mostly using two special cases of this process:
  - masking: invariant_probs = (0.0,) * K + (1.0,)
  - uniform: invariant_probs = (1.0 / K,) * K
  where K is the number of categories.
  In that case denoting π this invariant probability distribution, the
  corruption process targets Dir(τ π) where τ is a temperature parameter and Dir
  is the Dirichlet distribution.
  For each 0 <= t <= 1, the forward process is given by:

    p_{t|0}(X(t)|X(0)) = Dir(τ(h(t) δ(X(0)) + π)) ,

  where h(t) is a function of the time t and δ(X(0)) is the one-hot encoding of
  X(0). X(t) represents the corrupted data which is a sample from the Dirichlet
  distribution p_{t|0} and is therefore a categorical distribution.

  The function h(t) is given by the formula:

    h(t) = α(t) / (1 - α(t))

  In particular, we have that h(0) = +inf and h(1) = 0.

  NOTE: We perform the corruption in log-space for numerical stability.

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
    temperature: The temperature parameter of the Dirichlet distribution. This
      parameter controls the sharpness of the distribution.
    mode: The mode to use in `jax.random.choice` and `jax.random.bernoulli`. Can
      be set to "high" or "low" for how many bits to use in the Gumbel sampler.
      See https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.choice
      for more information.
    safety_epsilon: A small constant added to the denominator of the h-function
      to avoid division by zero.
  """

  schedule: SimplicialSchedule
  invariant_probs: Sequence[float]
  num_categories: int
  unused_mask_value: int = -1
  temperature: float = 1.0
  mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH
  safety_epsilon: float = 1e-6

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
  # MARK: h-function
  ##############################################################################

  @typechecked
  def h(self, time: TimeArray) -> TimeArray:
    """Returns the h-function of the process."""
    denominator = 1.0 - self.schedule.alpha(time) + self.safety_epsilon
    return self.schedule.alpha(time) / denominator

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
    invariant_dirichlet_params = self.temperature * self.invariant_probs_vec
    # data_spec is [B, T, 1]
    # invariant_dirichlet_params is [B, T, K]
    # output is [B, T, K]
    return random_utils.log_dirichlet_fast(
        key, alpha=invariant_dirichlet_params, shape=data_spec.shape[:-1]
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
    # get the unused mask
    unused_mask = x0 == self.unused_mask_value

    # compute one-hot encoding of x0
    x0_oh = jax.nn.one_hot(x0[..., 0], self.process_num_categories)
    time = utils.bcast_right(time, x0.ndim)

    # compute Dirichlet parameters
    dirichlet_param = self.invariant_probs_vec + self.h(time) * x0_oh
    dirichlet_param = self.temperature * dirichlet_param
    xt = random_utils.log_dirichlet_fast(key, alpha=dirichlet_param)

    logits = x0_oh
    target_info = {
        'x0': x0,  # Int[*b 1]; Uncorrupted input data.
        'logits': logits,  # Float[*b K] one-hot encoding of x0.
    }

    # Replace the unused probabilities with the unused_mask_value.
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
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_mask_value: int = -1,
      temperature: float = 1.0,
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
  ) -> SimplicialProcess:
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
        temperature=temperature,
        mode=mode,
        safety_epsilon=safety_epsilon,
    )

  @classmethod
  def uniform_process(
      cls,
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_mask_value: int = -1,
      temperature: float = 1.0,
      mode: SamplingPrecisionMode = SamplingPrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
  ) -> SimplicialProcess:
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
        temperature=temperature,
        mode=mode,
        safety_epsilon=safety_epsilon,
    )
