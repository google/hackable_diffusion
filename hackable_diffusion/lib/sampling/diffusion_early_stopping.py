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

"""Early stopping strategies for the diffusion sampler with early stopping."""

import dataclasses
from typing import Any, Protocol

from hackable_diffusion.lib.sampling import base
import jax
import jax.numpy as jnp
from kauldron.ktyping import Bool

################################################################################
# MARK: Type Aliases
################################################################################

DiffusionStep = base.DiffusionStep


################################################################################
# MARK: DiffusionEarlyStoppingFn
################################################################################


class DiffusionEarlyStoppingFn(Protocol):
  """Protocol for early stopping during the diffusion sampling loop.

  Implementations return a per-batch boolean indicating which batch elements
  should stop denoising.  The while_loop continues until *all* elements are
  done (``jnp.all(done)``) or the step budget is exhausted.
  """

  def should_stop(
      self,
      *,
      step: jax.Array,
      current_step: DiffusionStep,
      previous_step: DiffusionStep,
  ) -> Bool['B']:
    """Returns a bool array of shape ``[batch_size]``.

    Args:
      step: Scalar int — the current (0-based) update step index.
      current_step: The DiffusionStep produced by the latest update.
      previous_step: The DiffusionStep from the prior iteration.

    Returns:
      A bool array of shape ``[batch_size]`` where ``True`` means the
      corresponding element should stop.
    """
    ...


################################################################################
# MARK: DiffusionNoEarlyStop
################################################################################


class DiffusionNoEarlyStopFn(DiffusionEarlyStoppingFn):
  """Default: never stop early.  Equivalent to the original loop behavior."""

  def should_stop(
      self,
      *,
      step: jax.Array,
      current_step: DiffusionStep,
      previous_step: DiffusionStep,
  ) -> Bool['']:
    del step, previous_step
    batch_size = current_step.xt.shape[0]
    return jnp.zeros(batch_size, dtype=jnp.bool_)


################################################################################
# MARK: EntropyEarlyStop
################################################################################


@dataclasses.dataclass(frozen=True)
class DiffusionEntropyEarlyStopFn(DiffusionEarlyStoppingFn):
  """Stop denoising when the entropy of the logits is below a threshold.

  When the entropy is low, the denoiser has become very confident in its
  predictions, and further iterations are unlikely to yield significant
  improvements.

  Expects logits to be stored in ``current_step.aux[logits_key]`` with shape
  ``[*B, L, V]``.  The ``SamplerStep.update()`` implementation must populate
  this field for this strategy to work.

  Returns a per-batch boolean: ``True`` for each batch element whose mean
  per-token entropy is at or below the threshold.

  Attributes:
    entropy_threshold: The threshold below which denoising is considered
      converged.
    logits_key: The key in ``aux`` containing the logits.  Default is
      ``'logits'``.
  """

  entropy_threshold: float = 0.05
  # We must have logits.
  logits_key: str = 'logits'

  def should_stop(
      self,
      *,
      step: jax.Array,
      current_step: DiffusionStep,
      previous_step: DiffusionStep,
  ) -> Bool['']:
    del step, previous_step
    aux = current_step.aux
    logits = aux[self.logits_key]
    log_probs = jax.nn.log_softmax(logits)
    probs = jnp.exp(log_probs)
    # Guard against log(0) producing NaN in the entropy sum.
    log_probs = jnp.where(probs == 0, 0.0, log_probs)
    entropy_per_token = -jnp.sum(log_probs * probs, axis=-1)
    # Mean over the sequence (token) dimension, keeping batch dimension.
    entropy = jnp.mean(entropy_per_token, axis=-1)
    return entropy <= self.entropy_threshold
