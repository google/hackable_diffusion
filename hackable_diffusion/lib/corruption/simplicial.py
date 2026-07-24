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

"""Simplicial noise processes."""

from __future__ import annotations

import dataclasses
from typing import Protocol, Sequence

from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import discrete
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Constants
################################################################################

UNUSED_TOKEN = -1
LOGITS_INF = 1e9

################################################################################
# MARK: Type Aliases
################################################################################

Float = hd_typing.Float
PRNGKey = hd_typing.PRNGKey

DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

CorruptionProcess = hd_api.CorruptionProcess
PrecisionMode = jax_helpers.PrecisionMode
SimplicialSchedule = discrete.DiscreteSchedule
LinearSimplicialSchedule = discrete.LinearDiscreteSchedule
CosineSimplicialSchedule = discrete.CosineDiscreteSchedule
SquareCosineSimplicialSchedule = discrete.SquareCosineDiscreteSchedule
GeometricSimplicialSchedule = discrete.GeometricDiscreteSchedule
PolynomialSimplicialSchedule = discrete.PolynomialDiscreteSchedule


################################################################################
# MARK: Post-Corruption Functions
################################################################################


class SimplicialPostCorruptionFn(Protocol):
  """Post-corruption function protocol for simplicial (log-prob) data.

  The purpose of a post-corruption function is to project the noisy
  log-probability array onto a constrained subspace after each corruption or
  reverse-diffusion step.  The canonical example is symmetrising a
  simplex-valued edge-attribute matrix so that edge (i, j) and edge (j, i)
  share the same categorical distribution — the simplicial analogue of
  ``SymmetricPostCorruptionFn`` from the discrete process.

  Unlike the discrete variant, the input and output here are log-probability
  arrays of shape (*batch, num_categories), not integer token arrays.
  """

  def __call__(self, log_x: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Project the log-probability array."""
    ...


class IdentitySimplicialPostCorruptionFn(SimplicialPostCorruptionFn):
  """Identity post-corruption function (no projection)."""

  def __call__(self, log_x: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    return log_x


class SymmetricSimplicialPostCorruptionFn(SimplicialPostCorruptionFn):
  """Symmetric post-corruption function for simplex-valued edge matrices.

  This is the simplicial analogue of ``SymmetricPostCorruptionFn`` from the
  discrete process, used in DiGress https://arxiv.org/abs/2209.14734 in order
  to noise the adjacency graph.  This function also zeroes out the diagonal
  entries, thereby removing any self-loop.

  Input shape must be (batch, N, N, num_categories) where N is the number of
  nodes.
  """

  def __call__(self, log_x: DataArray) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Project the log-probability array to be symmetric."""

    if log_x.ndim != 4:
      raise ValueError(f'Expected 4D (B, N, N, K) input, got {log_x.ndim=}.')
    if log_x.shape[1] != log_x.shape[2]:
      raise ValueError(
          f'Spatial dimensions must be equal, got {log_x.shape[1]=} and'
          f' {log_x.shape[2]=}.'
      )

    _, n, _, num_categories = log_x.shape

    # Move the category axis before the spatial axes so that jnp.triu
    # operates on the last two (N, N) dimensions: (B, N, N, K) -> (B, K, N, N).
    # Doing so, the operations are identical to the discrete case,
    # see SymmetricPostCorruptionFn.
    log_x_bknn = jnp.moveaxis(log_x, -1, 1)
    log_x_tri = jnp.triu(log_x_bknn, k=1)
    log_x_sym = log_x_tri + jnp.transpose(log_x_tri, axes=(0, 1, 3, 2))
    log_y = jnp.moveaxis(log_x_sym, 1, -1)
    # Zero out diagonal entries: set to the no-edge log-probability vector
    # (all mass on category 0).
    no_edge_log = jnp.full((num_categories,), -LOGITS_INF, dtype=log_x.dtype)
    no_edge_log = no_edge_log.at[0].set(0.0)
    diag_mask = jnp.eye(n, dtype=jnp.bool_)[None, :, :, None]
    log_y = jnp.where(diag_mask, no_edge_log, log_y)
    return log_y


################################################################################
# MARK: SimplicialProcess
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class SimplicialProcess:
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
    unused_token: If a token is unused then it should have this value. Note that
      we require that this unused_token is not in the range of the vocabulary,
      i,e., unused_token < 0 or unused_token >= len(invariant_probs) (which is
      the same as process_num_categories).
    temperature: The temperature parameter of the Dirichlet distribution. This
      parameter controls the sharpness of the distribution.
    precision_mode: The precision mode to use in `jax.random.categorical` and
      other JAX random samplers. Can be set to "high" or "low" for how many bits
      to use in the Gumbel sampler. See
      https://jax.readthedocs.io/en/latest/jax.random.html#jax.random.choice for
        more information.
    safety_epsilon: A small constant added to the denominator of the h-function
      to avoid division by zero.
    post_corruption_fn: A projection applied to the corrupted log-prob array
      after each forward-corruption step and after each reverse-diffusion step.
      Used to enforce structural constraints such as symmetry of edge-attribute
      matrices.  Defaults to the identity (no projection).
  """

  schedule: SimplicialSchedule
  invariant_probs: Sequence[float]
  num_categories: int
  unused_token: int = UNUSED_TOKEN
  temperature: float = 1.0
  precision_mode: PrecisionMode = PrecisionMode.HIGH
  safety_epsilon: float = 1e-6
  post_corruption_fn: SimplicialPostCorruptionFn = (
      IdentitySimplicialPostCorruptionFn()
  )

  def __post_init__(self):
    if (
        self.unused_token >= 0
        and self.unused_token < self.process_num_categories
    ):
      raise ValueError(
          'unused_token must be outside of the range of the vocabulary.'
          f' Got: {self.unused_token=} and {self.num_categories=}'
      )

  ##############################################################################
  # MARK: Properties
  ##############################################################################

  @property
  def invariant_probs_vec(self) -> Float['M']:  # pyrefly: ignore[not-a-type, unknown-name]
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

  @kt.typechecked
  def h(self, time: TimeArray) -> TimeArray:  # pyrefly: ignore[not-a-type]
    """Returns the h-function of the process."""
    denominator = 1.0 - self.schedule.alpha(time) + self.safety_epsilon
    return self.schedule.alpha(time) / denominator

  ##############################################################################
  # MARK: Methods
  ##############################################################################

  @kt.typechecked
  def sample_from_invariant(
      self,
      key: PRNGKey,
      data_spec: DataArray,  # pyrefly: ignore[not-a-type]
  ) -> DataArray:  # pyrefly: ignore[not-a-type]
    """Sample from the invariant distribution."""
    invariant_dirichlet_params = self.temperature * self.invariant_probs_vec
    # data_spec is [B, T, 1]
    # invariant_dirichlet_params is [B, T, K]
    # output is [B, T, K]
    return fast_random.log_dirichlet_fast(
        key, alpha=invariant_dirichlet_params, shape=data_spec.shape[:-1]
    )

  @kt.typechecked
  def corrupt(
      self,
      key: PRNGKey,
      x0: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
  ) -> tuple[DataArray, TargetInfo]:  # pyrefly: ignore[not-a-type]
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
    unused_mask = x0 == self.unused_token

    # compute one-hot encoding of x0
    x0_oh = jax.nn.one_hot(x0[..., 0], self.process_num_categories)
    time = jax_helpers.bcast_right(time, x0.ndim)

    # compute Dirichlet parameters
    dirichlet_param = self.invariant_probs_vec + self.h(time) * x0_oh
    dirichlet_param = self.temperature * dirichlet_param
    xt = fast_random.log_dirichlet_fast(key, alpha=dirichlet_param)

    logits = x0_oh

    # Apply post-corruption projection (e.g. symmetrisation).
    xt = self.post_corruption_fn(xt)

    target_info = {
        'x0': x0,  # Int[*b 1]; Uncorrupted input data.
        'logits': logits,  # Float[*b K] one-hot encoding of x0.
    }

    # Replace the unused probabilities with the unused_token.
    xt = jnp.where(unused_mask, self.unused_token, xt)

    return xt, target_info

  @kt.typechecked
  def convert_predictions(
      self,
      prediction: TargetInfo,
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      time: TimeArray,  # pyrefly: ignore[not-a-type]
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

  ##############################################################################
  # MARK: Factory Methods
  ##############################################################################

  @classmethod
  def masking_process(
      cls,
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_token: int = UNUSED_TOKEN,
      temperature: float = 1.0,
      precision_mode: PrecisionMode = PrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
      post_corruption_fn: SimplicialPostCorruptionFn = IdentitySimplicialPostCorruptionFn(),
  ) -> SimplicialProcess:
    """Create a SimplicialProcess with masking invariant distribution."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )

    invariant_probs = (0.0,) * num_categories + (1.0,)
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_token=unused_token,
        temperature=temperature,
        precision_mode=precision_mode,
        safety_epsilon=safety_epsilon,
        post_corruption_fn=post_corruption_fn,
    )

  @classmethod
  def uniform_process(
      cls,
      schedule: SimplicialSchedule,
      num_categories: int,
      unused_token: int = UNUSED_TOKEN,
      temperature: float = 1.0,
      precision_mode: PrecisionMode = PrecisionMode.HIGH,
      safety_epsilon: float = 1e-6,
      post_corruption_fn: SimplicialPostCorruptionFn = IdentitySimplicialPostCorruptionFn(),
  ) -> SimplicialProcess:
    """Create a SimplicialProcess with uniform invariant distribution."""
    if num_categories < 1:
      raise ValueError(
          f'num_categories must be positive. Got: {num_categories=}'
      )
    invariant_probs = (1.0 / num_categories,) * num_categories
    return cls(
        schedule=schedule,
        invariant_probs=invariant_probs,
        num_categories=num_categories,
        unused_token=unused_token,
        temperature=temperature,
        precision_mode=precision_mode,
        safety_epsilon=safety_epsilon,
        post_corruption_fn=post_corruption_fn,
    )
