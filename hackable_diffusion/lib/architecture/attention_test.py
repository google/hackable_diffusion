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

"""Tests for the attention module."""

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import test_utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import attention
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type aliases
################################################################################

Float = hd_typing.Float

RoPEPositionType = arch_typing.RoPEPositionType
INVALID_INT = arch_typing.INVALID_INT

################################################################################
# MARK: Attention Tests
################################################################################


class AttentionTest(parameterized.TestCase):

  def setUp(self):
    """Sets up the test class."""
    super().setUp()
    self.rng = jax.random.PRNGKey(0)
    self.batch_size = 2
    self.seq_len_q = 16
    self.seq_len_kv = 64  # Perfect square for square RoPE tests
    self.dim = 128
    self.head_dim = 32
    self.num_heads = self.dim // self.head_dim

    self.x = jnp.ones((self.batch_size, self.seq_len_q, self.dim))
    self.c = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))

  @parameterized.named_parameters(
      ("head_dim_not_specified", INVALID_INT, 16),
      ("num_heads_not_specified", 32, INVALID_INT),
  )
  def test_attention_dims_factory(self, head_dim: int, num_heads: int):
    """Tests the factory when head_dim or num_heads is specified.

    More precisely, we test that the factory returns the correct head dimension
    and number of heads when head_dim or num_heads is specified.

    Args:
      head_dim: The head dimension.
      num_heads: The number of heads.
    """
    if head_dim == INVALID_INT:
      head_dim_predicted = self.dim // num_heads
    else:
      head_dim_predicted = head_dim
    if num_heads == INVALID_INT:
      num_heads_predicted = self.dim // head_dim
    else:
      num_heads_predicted = num_heads

    factory = attention.attention_dims_factory(
        head_dim=head_dim, num_heads=num_heads
    )
    head_dim, num_heads = factory(self.x)
    self.assertEqual(head_dim, head_dim_predicted)
    self.assertEqual(num_heads, num_heads_predicted)

  @parameterized.named_parameters(
      ("zero_num_heads", 0, INVALID_INT),
      ("negative_num_heads", -4, INVALID_INT),
      ("zero_head_dim", INVALID_INT, 0),
      ("negative_head_dim", INVALID_INT, -4),
  )
  def test_attention_dims_factory_raises_error_on_non_positive_args(
      self, num_heads: int, head_dim: int
  ):
    """Tests that the factory raises errors for non-positive num_heads and head_dim.

    Args:
      num_heads: The number of heads.
      head_dim: The head dimension.
    """
    with self.assertRaisesRegex(
        ValueError,
        "(Head dimension|Number of heads) must be positive or INVALID_INT.",
    ):
      attention.attention_dims_factory(head_dim=head_dim, num_heads=num_heads)

  def test_attention_dims_factory_raises_error_on_invalid_arguments(self):
    """Tests that the factory raises errors for invalid arguments.

    More precisely, we test that the factory raises an error when head_dim AND
    num_heads are NOT specified.
    """
    with self.assertRaisesRegex(
        ValueError, "Either head_dim or num_heads must be specified."
    ):
      attention.attention_dims_factory(
          head_dim=INVALID_INT, num_heads=INVALID_INT
      )

  def test_attention_dims_factory_raises_error_on_both_valid_arguments(self):
    """Tests that the factory raises errors for invalid arguments.

    More precisely, we test that the factory raises an error when both head_dim
    AND num_heads are specified.
    """
    with self.assertRaisesRegex(
        ValueError, "Either head_dim or num_heads must be INVALID_INT."
    ):
      attention.attention_dims_factory(
          head_dim=self.head_dim, num_heads=self.num_heads
      )

  @parameterized.named_parameters(
      ("num_heads_does_not_divide_embedding_dim", INVALID_INT, 17),
      ("head_dim_does_not_divide_embedding_dim", 17, INVALID_INT),
  )
  def test_attention_dims_factory_raises_error_on_non_divisible_embedding_dim(
      self,
      head_dim: int,
      num_heads: int,
  ):
    """Tests that the factory raises errors for non-divisible embedding dim."""
    with self.assertRaisesRegex(
        ValueError, ".* is not divisible by (head_dim|num_heads) .*"
    ):
      attention.attention_dims_factory(head_dim=head_dim, num_heads=num_heads)(
          self.x
      )

  # MARK: MultiHeadAttention tests

  @parameterized.named_parameters(
      ("self_attention_linear", None, True, RoPEPositionType.LINEAR),
      ("self_attention_square", None, True, RoPEPositionType.SQUARE),
      ("cross_attention_linear", "c", True, RoPEPositionType.LINEAR),
      ("cross_attention_square", "c", True, RoPEPositionType.SQUARE),
      ("self_attention_no_rope", None, False, RoPEPositionType.LINEAR),
      ("cross_attention_no_rope", "c", False, RoPEPositionType.LINEAR),
  )
  def test_multi_head_attention_output_shape(
      self,
      context: Float["batch sequence2 dim1"] | None,
      use_rope: bool,
      rope_position_type: RoPEPositionType,
  ):
    """Tests the output shape of MultiHeadAttention."""
    c = self.c if context == "c" else None
    module = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        use_rope=use_rope,
        rope_position_type=rope_position_type,
    )
    x_curr = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x_curr, c)
    output = module.apply(variables, x_curr, c)
    self.assertEqual(output.shape, x_curr.shape)

  def test_multi_head_attention_zero_init_output(self):
    """Tests that zero_init_output=True initializes output to zeros."""
    module = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        zero_init_output=True,
    )
    variables = module.init(self.rng, self.x, self.c)

    # 1. Check that the kernel and bias of the output projection are zeros.
    leaves_with_paths = test_utils.get_leaves_with_paths(variables)
    for path, leaf in leaves_with_paths.items():
      path_split = path.split("/")
      last_key = path_split[-1]
      params_name = path_split[1]
      if params_name == "Dense_Output":
        self.assertIn(last_key, ["kernel", "bias"])
        if last_key == "kernel":
          zero_kernel = jnp.zeros(shape=(self.dim, self.dim))
          self.assertTrue(jnp.allclose(leaf, zero_kernel))
        elif last_key == "bias":
          zero_bias = jnp.zeros(shape=(self.dim,))
          self.assertTrue(jnp.allclose(leaf, zero_bias))
        else:
          self.fail(f"Unknown leaf key: {last_key}")

    # 2. Check that the output is zeros.
    output = module.apply(variables, self.x, self.c)
    zeros_output = jnp.zeros_like(self.x)
    self.assertTrue(jnp.allclose(output, zeros_output))

  @parameterized.named_parameters(
      ("qk_norm", True),
      ("no_qk_norm", False),
  )
  def test_multi_head_attention_params_shape(self, normalize_qk: bool):
    """Tests that MultiHeadAttention has the correct parameters."""
    module = attention.MultiHeadAttention(
        num_heads=self.num_heads,
        use_rope=True,
        rope_position_type=RoPEPositionType.SQUARE,
        normalize_qk=normalize_qk,
    )
    variables = module.init(self.rng, self.x, self.c)

    # Check that the variables have the correct shape.
    leaves_with_paths = test_utils.get_leaves_with_paths(variables)
    if normalize_qk:
      self.assertLen(leaves_with_paths, 9)
    else:
      self.assertLen(leaves_with_paths, 8)

    for path, leaf in leaves_with_paths.items():
      path_split = path.split("/")
      params_name = path_split[1]
      last_key = path_split[-1]
      if params_name in ["Dense_K", "Dense_Q", "Dense_V", "Dense_Output"]:
        self.assertIn(last_key, ["kernel", "bias"])
        if last_key == "kernel":
          self.assertEqual(leaf.shape, (self.dim, self.dim))
        elif last_key == "bias":
          self.assertEqual(leaf.shape, (self.dim,))
        else:
          self.fail(f"Unknown leaf key: {last_key}")
      elif params_name == "norm_qk_scale":
        self.assertEqual(leaf.shape, (1, 1, 1, 1))
      else:
        self.fail(f"Unknown params name: {params_name}")


if __name__ == "__main__":
  absltest.main()
