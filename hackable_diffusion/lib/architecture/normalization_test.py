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

"""Tests for normalization strategies."""

import dataclasses
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
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

PyTree = hd_typing.PyTree

################################################################################
# MARK: Helpers
################################################################################


def _pad_to_shape(
    arr: jnp.ndarray, target_shape: tuple[int, ...]
) -> jnp.ndarray:
  """Pads an array to a target shape."""
  return (
      jnp.zeros(target_shape, dtype=arr.dtype)
      .at[: arr.shape[0], : arr.shape[1], : arr.shape[2], : arr.shape[3]]
      .set(arr)
  )


def _perturb_params(params: PyTree, key: jax.Array) -> PyTree:  # pyrefly: ignore[not-a-type]
  leaves, treedef = jax.tree_util.tree_flatten(params)
  keys_list = jax.random.split(key, len(leaves))
  key_tree = jax.tree_util.tree_unflatten(treedef, keys_list)
  return jax.tree_util.tree_map(
      lambda p, k: p + 0.5 * jax.random.normal(k, p.shape),
      params,
      key_tree,
  )


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

    # Sequence lengths for testing padding invariance.
    unpadded_seq_len = 4
    small_seq_len = 6
    large_seq_len = 8
    x_shape_small = (
        self.x_shape[0],
        self.x_shape[1],
        small_seq_len,
        self.x_shape[3],
    )
    x_shape_large = (
        self.x_shape[0],
        self.x_shape[1],
        large_seq_len,
        self.x_shape[3],
    )

    x_slice = self.x[:, :, :unpadded_seq_len, :]

    self.x_small = _pad_to_shape(arr=x_slice, target_shape=x_shape_small)
    self.x_large = _pad_to_shape(arr=x_slice, target_shape=x_shape_large)
    self.unpadded_seq_len = unpadded_seq_len
    self.small_seq_len = small_seq_len
    self.large_seq_len = large_seq_len

  def test_unconditional_rmsnorm_at_init(self):
    """Tests unconditional RMSNorm at init."""
    strategy = normalization.RMSNormStrategy()
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x)

    output_new = norm_layer.apply(params, self.x)

    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + strategy.epsilon)

    self.assertEqual(output_new.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_rmsnorm_at_init(self):
    """Tests conditional normalization at init when scale=0 and shift=0."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    output = norm_layer.apply(params, self.x, self.c)
    self.assertEqual(output.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]

    # At init, scale=0 and shift=0, so output is same as in unconditional.
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + strategy.epsilon)
    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
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
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output_perturbed = norm_layer.apply(params_perturbed, self.x, self.c)

    # Compute unconditional output for comparison.
    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + strategy.epsilon)

    self.assertEqual(output_perturbed.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    self.assertFalse(
        np.allclose(output_perturbed, output_ref, rtol=1e-5, atol=1e-5),  # pyrefly: ignore[bad-argument-type]
        msg=(
            "Conditional output should be different from unconditional output"
            " after perturbing params."
        ),
    )

  def test_unconditional_groupnorm_at_init(self):
    """Tests unconditional GroupNorm at init."""
    strategy = normalization.GroupNormStrategy(num_groups=self.num_groups)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x)
    output_new = norm_layer.apply(params, self.x)

    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_groupnorm_at_init(self):
    """Tests conditional GroupNorm at init when scale=0 and shift=0."""
    strategy = normalization.ConditionalGroupNormStrategy(
        num_groups=self.num_groups,
        use_shift=True,
    )
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    output_new = norm_layer.apply(params, self.x, self.c)

    # At init, scale=0 and shift=0, so output is same as in unconditional.
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-5, atol=1e-5)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_groupnorm_perturbed(self):
    """Tests conditional GroupNorm when scale!=0 and shift!=0."""
    strategy = normalization.ConditionalGroupNormStrategy(
        num_groups=self.num_groups,
        use_shift=True,
    )
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output = norm_layer.apply(params_perturbed, self.x, self.c)

    # Compute unconditional output for comparison.
    norm_ref = nn.GroupNorm(num_groups=self.num_groups)
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    self.assertFalse(
        np.allclose(output, output_ref, rtol=1e-5, atol=1e-5, equal_nan=True),  # pyrefly: ignore[bad-argument-type]
        "Conditional output should be different from unconditional output after"
        " perturbing params.",
    )

  def test_rmsnorm_padding_invariance(self):
    """Tests RMSNorm padding invariance."""
    strategy = normalization.RMSNormStrategy()
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x_small)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large)
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        out_large[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        atol=1e-5,
    )

  def test_unconditional_layernorm_at_init(self):
    """Tests unconditional LayerNorm at init."""
    strategy = normalization.LayerNormStrategy()
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x)
    output_new = norm_layer.apply(params, self.x)

    norm_ref = nn.LayerNorm()
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-4, atol=1e-4)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_layernorm_at_init(self):
    """Tests conditional LayerNorm at init when scale=0 and shift=0."""
    strategy = normalization.ConditionalLayerNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    output_new = norm_layer.apply(params, self.x, self.c)

    norm_ref = nn.LayerNorm()
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output_new.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    np.testing.assert_allclose(output_new, output_ref, rtol=1e-4, atol=1e-4)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_layernorm_perturbed(self):
    """Tests conditional LayerNorm when scale!=0 and shift!=0."""
    strategy = normalization.ConditionalLayerNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output = norm_layer.apply(params_perturbed, self.x, self.c)

    norm_ref = nn.LayerNorm()
    params_ref = norm_ref.init(self.rng, self.x)
    output_ref = norm_ref.apply(params_ref, self.x)

    self.assertEqual(output.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    self.assertFalse(
        np.allclose(output, output_ref, rtol=1e-5, atol=1e-5, equal_nan=True),  # pyrefly: ignore[bad-argument-type]
        "Conditional output should be different from unconditional output after"
        " perturbing params.",
    )

  def test_layernorm_padding_invariance(self):
    """Tests LayerNorm padding invariance."""
    strategy = normalization.LayerNormStrategy()
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x_small)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large)
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        out_large[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        atol=1e-5,
    )

  def test_groupnorm_padding_non_invariance(self):
    """Tests GroupNorm padding non-invariance."""
    strategy = normalization.GroupNormStrategy(num_groups=self.num_groups)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x_small)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large)
    self.assertFalse(
        np.allclose(
            out_small[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
            out_large[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
            atol=1e-5,
        )
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="rmsnorm",
          strategy_factory=lambda: normalization.RMSNormStrategy(),
          mask_dim=1,
      ),
      dict(
          testcase_name="groupnorm",
          strategy_factory=lambda: normalization.GroupNormStrategy(
              num_groups=5
          ),
          mask_dim=10,
      ),
      dict(
          testcase_name="layernorm",
          strategy_factory=lambda: normalization.LayerNormStrategy(),
          mask_dim=1,
      ),
  )
  def test_masked_padding_invariance(self, strategy_factory, mask_dim):
    """Tests masked padding invariance."""
    strategy = strategy_factory()
    norm_layer = strategy.build_layer(name="Norm")

    mask_shape_small = (
        self.x_shape[0],
        self.x_shape[1],
        self.small_seq_len,
        mask_dim,
    )
    mask_shape_large = (
        self.x_shape[0],
        self.x_shape[1],
        self.large_seq_len,
        mask_dim,
    )

    mask_small = jnp.zeros(mask_shape_small, dtype=jnp.bool_)
    mask_small = mask_small.at[:, :, : self.unpadded_seq_len, :].set(True)

    mask_large = jnp.zeros(mask_shape_large, dtype=jnp.bool_)
    mask_large = mask_large.at[:, :, : self.unpadded_seq_len, :].set(True)

    params = norm_layer.init(self.rng, self.x_small, mask=mask_small)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(
        params_perturbed, self.x_small, mask=mask_small
    )
    out_large = norm_layer.apply(
        params_perturbed, self.x_large, mask=mask_large
    )
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        out_large[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        atol=1e-5,
    )

  def test_groupnorm_broadcastable_mask_fails(self):
    """Tests GroupNorm with a broadcastable mask raises ValueError."""
    strategy = normalization.GroupNormStrategy(num_groups=self.num_groups)
    norm_layer = strategy.build_layer(name="Norm")

    mask_shape_large = (self.x_shape[0], self.x_shape[1], self.large_seq_len, 1)
    mask_large = jnp.zeros(mask_shape_large, dtype=jnp.bool_)
    mask_large = mask_large.at[:, :, : self.unpadded_seq_len, :].set(True)

    with self.assertRaisesRegex(
        ValueError,
        "If using GroupNorm with a mask, the mask's last dimension must"
        " match the input's channel dimension. Otherwise, one cannot"
        " reshape the mask during the grouping operation.",
    ):
      norm_layer.init(self.rng, self.x_large, mask=mask_large)

  def test_rmsnorm_mask_equivalence(self):
    """Tests that RMSNorm produces same values for non-padding tokens with or without mask."""
    strategy = normalization.RMSNormStrategy()
    norm_layer = strategy.build_layer(name="Norm")

    mask_large = jnp.zeros(
        (self.x_shape[0], self.x_shape[1], self.large_seq_len, 1),
        dtype=jnp.bool_,
    )
    mask_large = mask_large.at[:, : self.unpadded_seq_len, :].set(True)

    params = norm_layer.init(self.rng, self.x_large)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_no_mask = norm_layer.apply(params_perturbed, self.x_large)
    out_masked = norm_layer.apply(
        params_perturbed, self.x_large, mask=mask_large
    )

    np.testing.assert_allclose(
        out_no_mask[:, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        out_masked[:, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        atol=1e-6,
    )

  def test_conditional_rmsnorm_scale_only_at_init(self):
    """Tests conditional RMSNorm with scale-only (no shift) at init."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=False)
    norm_layer = strategy.build_layer(name="ConditionalNorm")
    params = norm_layer.init(self.rng, self.x, self.c)
    output = norm_layer.apply(params, self.x, self.c)
    self.assertEqual(output.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]

    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + strategy.epsilon)
    np.testing.assert_allclose(output, output_ref, rtol=1e-5, atol=1e-5)  # pyrefly: ignore[no-matching-overload]

  def test_conditional_rmsnorm_scale_only_perturbed(self):
    """Tests conditional RMSNorm scale-only with perturbed params."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=False)
    norm_layer = strategy.build_layer(name="ConditionalNorm")
    params = norm_layer.init(self.rng, self.x, self.c)
    params_perturbed = _perturb_params(params=params, key=self.rng)
    output_perturbed = norm_layer.apply(params_perturbed, self.x, self.c)

    x2 = jnp.mean(self.x**2, -1, keepdims=True)
    output_ref = self.x * lax.rsqrt(x2 + strategy.epsilon)

    self.assertEqual(output_perturbed.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]
    self.assertFalse(
        np.allclose(output_perturbed, output_ref, rtol=1e-5, atol=1e-5),  # pyrefly: ignore[bad-argument-type]
    )

  def test_conditional_scale_only_projects_to_ch(self):
    """Tests that scale-only conditioning projects to ch (not ch*2)."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=False)
    norm_layer = strategy.build_layer(name="ConditionalNorm")
    params = norm_layer.init(self.rng, self.x, self.c)
    # Under the new _NormalizationLayer, conditioning params are nested.
    dense_kernel = params["params"]["conditioning"]["Dense_Scale"]["kernel"]
    expected_shape = (self.c_shape[-1], self.x_shape[-1])
    self.assertEqual(dense_kernel.shape, expected_shape)

  def test_conditional_scale_shift_projects_to_ch_times_2(self):
    """Tests that scale+shift conditioning projects to ch * 2."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="ConditionalNorm")
    params = norm_layer.init(self.rng, self.x, self.c)
    # Under the new _NormalizationLayer, conditioning params are nested.
    dense_kernel = params["params"]["conditioning"]["Dense_ScaleShift"]["kernel"]
    expected_shape = (self.c_shape[-1], self.x_shape[-1] * 2)
    self.assertEqual(dense_kernel.shape, expected_shape)

  def test_conditional_rmsnorm_scale_only_padding_invariance(self):
    """Tests scale-only conditional RMSNorm padding invariance."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=False)
    norm_layer = strategy.build_layer(name="ConditionalNorm")
    c_small = jax.random.normal(self.rng, self.c_shape)
    params = norm_layer.init(self.rng, self.x_small, c_small)
    params_perturbed = _perturb_params(params=params, key=self.rng)

    out_small = norm_layer.apply(params_perturbed, self.x_small, c_small)
    out_large = norm_layer.apply(params_perturbed, self.x_large, c_small)
    np.testing.assert_allclose(
        out_small[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        out_large[:, :, : self.unpadded_seq_len, :],  # pyrefly: ignore[bad-index]
        atol=1e-5,
    )

  def test_conditional_norm_requires_conditioning(self):
    """Tests that conditional norm raises when c is not provided."""
    strategy = normalization.ConditionalRMSNormStrategy(use_shift=True)
    norm_layer = strategy.build_layer(name="Norm")
    params = norm_layer.init(self.rng, self.x, self.c)
    with self.assertRaisesRegex(
        ValueError,
        "Conditioning 'c' must be provided for a conditional norm layer.",
    ):
      norm_layer.apply(params, self.x)

  @parameterized.product(
      strategy_factories=[
          (
              lambda: normalization.RMSNormStrategy(),
              lambda: normalization.ConditionalRMSNormStrategy(
                  use_shift=True
              ),
          ),
          (
              lambda: normalization.LayerNormStrategy(),
              lambda: normalization.ConditionalLayerNormStrategy(
                  use_shift=True,
              ),
          ),
          (
              lambda: normalization.GroupNormStrategy(num_groups=5),
              lambda: normalization.ConditionalGroupNormStrategy(
                  num_groups=5,
                  use_shift=True,
              ),
          ),
      ],
      conditional=[False, True],
      dtype=[jnp.float32, jnp.bfloat16],
  )
  def test_output_dtype(self, strategy_factories, conditional, dtype):
    """Tests that the output dtype matches the configured dtype."""
    uncond_factory, cond_factory = strategy_factories
    if conditional:
      strategy = dataclasses.replace(cond_factory(), dtype=dtype)
    else:
      strategy = dataclasses.replace(uncond_factory(), dtype=dtype)

    norm_layer = strategy.build_layer(name="Norm")

    x = self.x.astype(dtype)
    if conditional:
      c = self.c.astype(dtype)
      params = norm_layer.init(self.rng, x, c)
      output = norm_layer.apply(params, x, c)
    else:
      params = norm_layer.init(self.rng, x)
      output = norm_layer.apply(params, x)

    self.assertEqual(output.dtype, dtype)  # pyrefly: ignore[missing-attribute]
    self.assertEqual(output.shape, self.x_shape)  # pyrefly: ignore[missing-attribute]


if __name__ == "__main__":
  absltest.main()
