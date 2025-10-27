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

"""Tests for normalization layers."""

import flax.linen as nn
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import normalization
import jax
from jax import lax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type Aliases
################################################################################

NormalizationType = arch_typing.NormalizationType

################################################################################
# MARK: Tests
################################################################################


class NormalizationTest(parameterized.TestCase):

  def setUp(self):
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.x_shape = (2, 8, 8, 10)
    self.x = jax.random.normal(self.rng, self.x_shape)
    self.c_shape = (2, 32)
    self.c = jax.random.normal(self.rng, self.c_shape)
    self.num_groups = 5

  def test_unconditional_rmsnorm(self):
    """Tests unconditional RMSNorm."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=False,
    )
    params = norm_layer.init(self.rng, self.x)
    output_new = norm_layer.apply(params, self.x)

    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_rmsnorm_at_init(self):
    """Tests conditional normalization at init when scale=0 and shift=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=True,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    output = norm_layer.apply(params, self.x, self.c)
    self.assertEqual(output.shape, self.x_shape)

    # at init, scale=0 and shift=0, so output is same as in unconditional.
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)
    np.testing.assert_allclose(
        output,
        output_ref,
        rtol=1e-5,
        atol=1e-5,
        err_msg=(
            "Conditional output should be same as unconditional output at"
            " params init."
        ),
    )

  def test_conditional_rmsnorm_perturbed(self):
    """Tests conditional normalization when scale!=0 and shift!=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.RMS_NORM,
        conditional=True,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = jax.tree_util.tree_map(lambda x: x + 0.5, params)
    output_perturbed = norm_layer.apply(params_perturbed, self.x, self.c)

    # unconditional output
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + norm_layer.epsilon)

    self.assertEqual(output_perturbed.shape, self.x_shape)
    self.assertFalse(
        np.allclose(output_perturbed, output_ref, rtol=1e-5, atol=1e-5),
        msg=(
            "Conditional output should be different from unconditional output"
            " after perturbing params."
        ),
    )

  def test_unconditional_groupnorm(self):
    """Tests unconditional GroupNorm."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=False,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x)
    output_new = norm_layer.apply(params, self.x)

    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_groupnorm_at_init(self):
    """Tests conditional GroupNorm at init when scale=0 and shift=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=True,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    output_new = norm_layer.apply(params, self.x, self.c)

    # at init, scale=0 and shift=0, so output is same as in unconditional.
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)

  def test_conditional_groupnorm_perturbed(self):
    """Tests conditional GroupNorm when scale!=0 and shift!=0."""
    norm_layer = normalization.NormalizationLayer(
        normalization_method=NormalizationType.GROUP_NORM,
        conditional=True,
        num_groups=self.num_groups,
    )
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = jax.tree_util.tree_map(lambda x: x + 0.5, params)
    output = norm_layer.apply(params_perturbed, self.x, self.c)

    # unconditional output
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output.shape, self.x_shape)
    self.assertFalse(
        np.allclose(output, output_ref, rtol=1e-5, atol=1e-5, equal_nan=True),
        "Conditional output should be different from unconditional output after"
        " perturbing params.",
    )


if __name__ == "__main__":
  absltest.main()
