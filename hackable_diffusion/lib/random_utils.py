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

"""Random utils.

`jax.random.gamma` is slow https://github.com/jax-ml/jax/issues/18984.
As Dirichlet and Beta samplers rely on Gamma samplers and are heavily used in
the case of a simplicial corruption process, we implement fast
versions of them here.

In particular, as `jax.random.gamma` relies on the Marsaglia-Tsang algorithm,
which is a rejection-sampling algorithm, we implement a faster but biased
version of this algorithm. We refer to
https://dl.acm.org/doi/10.1145/358407.358414 for more details on the
Marsaglia-Tsang algorithm.

Our main modification is to avoid the `while` loop in `jax.random.gamma`
by using an unrolled loop. We use `N=5` iterations (instead of the `while`
loop). This introduces a negligible error, but the samplers are 10x faster.
"""

import functools
from hackable_diffusion.lib import hd_typing
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################


PRNGKey = hd_typing.PRNGKey

################################################################################
# MARK: Fast Log-Gamma Sampler
################################################################################


@functools.partial(jax.jit, static_argnames=("shape",))
def log_gamma_fast(
    key: PRNGKey,
    alpha: jax.Array,
    shape: tuple[int, ...] | None = None,
) -> jax.Array:
  """Samples log(X) where X ~ Gamma(alpha, 1).

  Implement a fast (but biased) sampler for Gamma(alpha, 1). This is a
  modification of the Marsaglia-Tsang algorithm. We refer to
  https://dl.acm.org/doi/10.1145/358407.358414 for more details on the
  Marsaglia-Tsang algorithm. We avoid the `while` loop in `jax.random.gamma`
  by using an unrolled loop. We use `N=5` iterations (instead of the `while`
  loop). This introduces a negligible error, but the samplers are 10x faster.

  The main ingredient of the Marsaglia-Tsang algorithm is a rejection sampling
  procedure based on the fact that if X has density

      h(x)^(α-1) exp[-h(x)] h'(x) / Γ(α), (1)

  then h(X) is a Gamma(α, 1) random variable.
  We refer to https://www.tandfonline.com/doi/abs/10.1080/01621459.1984.10477088
  for more details.

  In https://dl.acm.org/doi/10.1145/358407.358414, the choice of the "magic"
  constants `c` and `d` (as well as the choice of a third order polynomial), are
  explained, see Section 2.2. In particular, the authors show that with this
  choice, they obtain a very good approximation of a Gaussian distribution for
  (1).

  Args:
    key: The PRNGKey.
    alpha: The shape parameter of the Gamma distribution. Can be broadcasted.
    shape: The shape of the output. If `None`, it is inferred from `alpha`.

  Returns:
    The log of the Gamma sample.
  """
  # Broadcast alpha
  if shape is None:
    shape = jnp.shape(alpha)
  alpha = jnp.broadcast_to(alpha, shape)

  # Boosting for small alphas
  mask_small = alpha < 1.0
  alpha_boosted = jnp.where(mask_small, alpha + 1.0, alpha)

  # Marsaglia-Tsang constants
  d = alpha_boosted - 1.0 / 3.0
  c = 1.0 / jnp.sqrt(9.0 * d)

  # Initialize the rejection sampling loop
  out = jnp.zeros(shape, dtype=alpha.dtype)
  needs_sample = jnp.ones(shape, dtype=jnp.bool_)

  # Unrolled rejection sampling loop (no while loop)
  def body_fn(i, val):
    del i
    key, out, needs_sample = val
    key, k_v, k_u = jax.random.split(key, 3)

    v_rand = jax.random.normal(k_v, shape)
    u_rand = jax.random.uniform(k_u, shape)

    v = 1.0 + c * v_rand
    v3 = v * v * v

    cond_pos = v > 0
    x2 = v_rand * v_rand

    safe_v3 = jnp.where(cond_pos, v3, 1.0)
    cond_log = jnp.log(u_rand) < (
        0.5 * x2 + d * (1.0 - safe_v3 + jnp.log(safe_v3))
    )

    is_valid = cond_pos & cond_log
    update_mask = needs_sample & is_valid

    # We calculate the linear sample here: d * v^3
    candidate_val = d * v3
    new_out = jnp.where(update_mask, candidate_val, out)
    new_needs = jnp.where(update_mask, False, needs_sample)

    return key, new_out, new_needs

  # Run the rejection sampling loop
  val = (key, out, needs_sample)
  for i in range(5):
    val = body_fn(i, val)
  key, final_out_linear, _ = val

  # Log of the linear sample
  log_sample_base = jnp.log(final_out_linear + 1e-30)

  # Correction for boosted alphas
  _, subkey = jax.random.split(key)
  log_u = jnp.log(jax.random.uniform(subkey, shape) + 1e-30)
  log_correction = (1.0 / alpha) * log_u
  final_log_res = jnp.where(
      mask_small, log_sample_base + log_correction, log_sample_base
  )

  return final_log_res


################################################################################
# MARK: Fast Log-Dirichlet Sampler / Dirichlet Sampler
################################################################################


@functools.partial(jax.jit, static_argnames=("shape",))
def log_dirichlet_fast(
    key: PRNGKey, alpha: jax.Array, shape: tuple[int, ...] | None = ()
) -> jax.Array:
  """Samples log(P) where P ~ Dirichlet(alpha).

  It uses the characterization of Dirichlet random variables as a normalized
  vector of Gamma random variables.

  Args:
    key: The PRNGKey.
    alpha: The shape parameters of the Dirichlet distribution.
    shape: The shape of the output.

  Returns:
    The log of the Dirichlet sample.
  """
  alpha = jnp.array(alpha)
  total_shape = shape + alpha.shape
  log_gammas = log_gamma_fast(key, alpha, shape=total_shape)
  return jax.nn.log_softmax(log_gammas, axis=-1)


@functools.partial(jax.jit, static_argnames=("shape",))
def dirichlet_fast(
    key: PRNGKey, alpha: jax.Array, shape: tuple[int, ...] | None = ()
) -> jax.Array:
  """Samples P where P ~ Dirichlet(alpha)."""
  return jnp.exp(log_dirichlet_fast(key, alpha, shape))


################################################################################
# MARK: Fast Log-Beta Sampler
################################################################################


@functools.partial(jax.jit, static_argnames=("shape",))
def sample_log_beta_joint(
    key: PRNGKey,
    alpha: jax.Array,
    beta: jax.Array,
    shape: tuple[int, ...] | None = (),
) -> tuple[jax.Array, jax.Array]:
  """Samples W ~ Beta(alpha, beta) and returns (log(W), log(1-W)).

  Uses the Gamma ratio trick: W = Ga / (Ga + Gb), where G_a and G_b are Gamma
  samples with shape `alpha` and `beta` respectively.

  Args:
    key: The PRNGKey.
    alpha: The first shape parameter of the Beta distribution.
    beta: The second shape parameter of the Beta distribution.
    shape: The shape of the output.

  Returns:
    The log of the Beta sample.
  """
  k1, k2 = jax.random.split(key)

  log_ga = log_gamma_fast(k1, alpha, shape)
  log_gb = log_gamma_fast(k2, beta, shape)

  log_sum_g = jnp.logaddexp(log_ga, log_gb)

  log_w = log_ga - log_sum_g
  log_1_minus_w = log_gb - log_sum_g

  return log_w, log_1_minus_w
