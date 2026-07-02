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

from hackable_diffusion.lib import fast_random
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.corruption import simplicial
from hackable_diffusion.lib.sampling import base
import jax
import jax.numpy as jnp
import kauldron.ktyping as kt

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

SimplicialProcess = simplicial.SimplicialProcess
SimplicialSchedule = schedules.SimplicialSchedule


################################################################################
# MARK: Beta Shrinkage
################################################################################


@kt.typechecked
def log_beta_shrinkage(
    key: jax.Array,
    log_x: jax.Array,
    concentration: jax.Array,
    kappa: float,
    safety_epsilon: float = 0.0,
) -> jax.Array:
  """Beta shrinkage of a Dirichlet sample.

  Let log(X) such that X ~ Dir(concentration) and kappa in [0, 1]. Then this
  function returns log(Y) such that Y ~ Dir(kappa * concentration).

  To do so we leverage the following identity.
  Let B ~ Beta(a, b) with a = kappa * concentration and b = (1 - kappa) *
  concentration.
  Then B X / sum(B X) has the same distribution as Dir(kappa * concentration).
  We call this process "Beta-shrinkage".

  Args:
    key: the random key.
    log_x: the log-Dirichlet sample of shape (..., num_categories).
    concentration: the concentration scalar or array.
    kappa: the shrinkage parameter in [0, 1].
    safety_epsilon: a small positive value added to both Beta concentration
      parameters to avoid the degenerate Beta(0, b) at kappa=0. When
      safety_epsilon > 0, kappa=0 produces a Beta highly concentrated near zero
      rather than a point mass, keeping all computations finite.

  Returns:
    the shrunk log-sample.
  """
  if kappa == 1.0:
    return log_x
  alpha_vec = jnp.broadcast_to(concentration, log_x.shape)

  log_b, _ = fast_random.sample_log_beta_joint(
      key,
      kappa * alpha_vec + safety_epsilon,
      (1.0 - kappa) * alpha_vec + safety_epsilon,
      shape=alpha_vec.shape,
  )

  log_y = log_b + log_x
  log_y = log_y - jax.nn.logsumexp(log_y, axis=-1, keepdims=True)
  return log_y


################################################################################
# MARK: DDIM Step
################################################################################


@dataclasses.dataclass(frozen=True, kw_only=True)
class SimplicialDDIMStep(SamplerStep):
  """This is the simplicial version of the DDIM step."""

  corruption_process: SimplicialProcess
  churn: float = 1.0
  safety_epsilon: float = 1e-6

  @kt.typechecked
  def initialize(
      self,
      initial_noise: DataArray,  # pyrefly: ignore[not-a-type]
      initial_step_info: StepInfo,
  ) -> DiffusionStep:
    return DiffusionStep(
        xt=initial_noise,
        step_info=initial_step_info,
        aux={'logits': initial_noise},
    )

  @kt.typechecked
  def update(
      self,
      prediction: TargetInfo,
      current_step: DiffusionStep,
      next_step_info: StepInfo,
  ) -> DiffusionStep:

    current_step_info = current_step.step_info
    log_xt = current_step.xt  # Input is now logits (log-probabilities)

    time = current_step_info.time
    next_time = next_step_info.time

    # Broadcast time to match batch dimensions
    time = jax_helpers.bcast_right(time, log_xt.ndim)
    next_time = jax_helpers.bcast_right(next_time, log_xt.ndim)
    key = next_step_info.rng

    # Get logits
    logits = self.corruption_process.convert_predictions(
        prediction=prediction,
        xt=log_xt,
        time=time,
    )['logits']

    # Sample hard token
    key, sample_key = jax.random.split(key)
    sample_idx = jax.random.categorical(key=sample_key, logits=logits)
    num_cats = self.corruption_process.process_num_categories
    one_hot_mask = jax.nn.one_hot(sample_idx, num_cats, dtype=log_xt.dtype)
    log_sample_oh = jnp.where(one_hot_mask > 0.5, 0.0, -1e30)

    # Compute parameters
    eps = self.corruption_process.temperature
    alpha_t = self.corruption_process.schedule.alpha(time)
    alpha_s = self.corruption_process.schedule.alpha(next_time)

    bar_beta_t = eps / (1.0 - alpha_t)
    bar_beta_s = eps / (1.0 - alpha_s)

    target_shape = log_xt.shape[:-1] + (1,)

    if self.churn == 0.0:
      # Regular DDIM step
      shape_0 = bar_beta_t
      shape_1 = bar_beta_s - shape_0

      _, beta_key = jax.random.split(key)
      log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
          beta_key, shape_0, shape_1, shape=target_shape
      )

      term_1 = log_w + log_xt
      term_2 = log_1_minus_w + log_sample_oh
    else:
      # Churn step
      pi = self.corruption_process.invariant_probs_vec
      h_t = alpha_t / (1.0 - alpha_t)
      h_s = alpha_s / (1.0 - alpha_s)

      concentration = eps * (pi + h_t * one_hot_mask)

      key, f_key = jax.random.split(key)
      log_pt_kappa = log_beta_shrinkage(
          f_key,
          log_x=log_xt,
          concentration=concentration,
          kappa=1.0 - self.churn,
          safety_epsilon=self.safety_epsilon,
      )

      key, v_key = jax.random.split(key)
      alpha_v = (
          self.churn * eps * pi
          + (eps * h_s - (1.0 - self.churn) * eps * h_t) * one_hot_mask
      )
      log_v = fast_random.log_dirichlet_fast(v_key, alpha=alpha_v, shape=())

      # Sample W from Beta(kappa * bar_beta_t, bar_beta_s - kappa * bar_beta_t).
      # safety_epsilon is added to both parameters to handle the degenerate
      # case kappa=0 (churn=1), where Beta(0, b) is undefined in JAX.
      # With safety_epsilon, Beta(eps, b+eps) is concentrated near 0,
      # matching the correct asymptotic behaviour W->0 as churn->1.
      _, beta_key = jax.random.split(key)
      log_w, log_1_minus_w = fast_random.sample_log_beta_joint(
          beta_key,
          (1.0 - self.churn) * bar_beta_t + self.safety_epsilon,
          bar_beta_s - (1.0 - self.churn) * bar_beta_t + self.safety_epsilon,
          shape=target_shape,
      )

      term_1 = log_w + log_pt_kappa
      term_2 = log_1_minus_w + log_v

    new_xt = jnp.logaddexp(term_1, term_2)

    # Apply the post-corruption projection (e.g. symmetrisation) exactly as
    # DiscreteDDIMStep / IntegratedDiscreteDDIMStep do after sampling new_xt.
    new_xt = self.corruption_process.post_corruption_fn(new_xt)

    return DiffusionStep(
        xt=new_xt,  # Output is robust logits
        step_info=next_step_info,
        aux={'logits': logits},
    )

  @kt.typechecked
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
