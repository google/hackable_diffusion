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

"""Tests for the attention strategies and layers."""

from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import test_helpers
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import sequence_embedders
import jax
from jax.experimental.pallas import tpu as pltpu
import jax.experimental.pallas.ops.tpu.flash_attention as flash
import jax.numpy as jnp
import kauldron.ktyping as kt
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Type aliases
################################################################################

Float = hd_typing.Float
LinearRoPEPositions = sequence_embedders.LinearRoPEPositions
SquareRoPEPositions = sequence_embedders.SquareRoPEPositions
RoPEPositionsFn = sequence_embedders.RoPEPositionsFn

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

  # MARK: AttentionHeadsSpec tests

  def test_heads_spec_infer_num_heads(self):
    heads_spec = attention.AttentionHeadsSpec(head_dim=16)
    head_dim, num_heads = heads_spec.resolve(self.dim)
    self.assertEqual(head_dim, 16)
    self.assertEqual(num_heads, 8)

  def test_heads_spec_infer_head_dim(self):
    heads_spec = attention.AttentionHeadsSpec(num_heads=32)
    head_dim, num_heads = heads_spec.resolve(self.dim)
    self.assertEqual(head_dim, 4)
    self.assertEqual(num_heads, 32)

  def test_heads_spec_no_args_raises(self):
    with self.assertRaisesRegex(
        ValueError, 'At least num_heads or head_dim must be provided.'
    ):
      attention.AttentionHeadsSpec()

  def test_heads_spec_both_specified(self):
    heads_spec = attention.AttentionHeadsSpec(num_heads=4, head_dim=32)
    head_dim, num_heads = heads_spec.resolve(128)
    self.assertEqual(head_dim, 32)
    self.assertEqual(num_heads, 4)

  def test_heads_spec_both_specified_mismatch_raises(self):
    heads_spec = attention.AttentionHeadsSpec(num_heads=4, head_dim=32)
    with self.assertRaisesRegex(
        ValueError, 'Expected embedding_dim=128.*got 256.'
    ):
      heads_spec.resolve(256)

  def test_heads_spec_non_divisible_head_dim_raises(self):
    heads_spec = attention.AttentionHeadsSpec(head_dim=17)
    with self.assertRaisesRegex(
        ValueError, 'Embedding dim 128 is not divisible by head_dim 17.'
    ):
      heads_spec.resolve(self.dim)

  def test_heads_spec_non_divisible_num_heads_raises(self):
    heads_spec = attention.AttentionHeadsSpec(num_heads=17)
    with self.assertRaisesRegex(
        ValueError, 'Embedding dim 128 is not divisible by num_heads 17.'
    ):
      heads_spec.resolve(self.dim)

  # MARK: MultiHeadAttention tests

  def test_multi_head_attention_mask_invariance(self):
    """Tests that masked tokens do not affect the attention output."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
    )

    # Create an initial input sequence
    rng1, rng2 = jax.random.split(self.rng)
    x_original = jnp.ones((self.batch_size, self.seq_len_q, self.dim))

    # Create a mask: Keep the first half of the sequence, mask out the second
    # half. Shape needs to be (batch_size, seq_len_q)
    half_seq = self.seq_len_q // 2
    single_mask = jnp.arange(self.seq_len_q) < half_seq
    mask = jnp.broadcast_to(single_mask, (self.batch_size, self.seq_len_q))

    # Initialize variables
    variables = module.init(rng1, x_original, c=None, mask=mask)

    # Get the output using the original sequence with the mask
    output_original = module.apply(variables, x_original, c=None, mask=mask)

    # Corrupt the masked tokens in the input sequence
    # We add random noise only to the tokens where mask == False
    noise = jax.random.normal(rng2, x_original.shape)
    x_corrupted = jnp.where(
        jnp.expand_dims(mask, -1), x_original, x_original + noise
    )

    # Get the output using the corrupted sequence with the SAME mask
    output_corrupted = module.apply(variables, x_corrupted, c=None, mask=mask)

    # We check that the outputs of the valid tokens are the same for the
    # original and corrupted sequences.
    valid_output_original = output_original[mask]
    valid_output_corrupted = output_corrupted[mask]

    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        valid_output_original,
        valid_output_corrupted,
        atol=1e-5,
    )

    # Ensure that WITHOUT the mask, the corrupted tokens DO change the valid
    # outputs.
    output_original_no_mask = module.apply(
        variables, x_original, c=None, mask=None
    )
    output_corrupted_no_mask = module.apply(
        variables, x_corrupted, c=None, mask=None
    )

    valid_output_original_no_mask = output_original_no_mask[mask]
    valid_output_corrupted_no_mask = output_corrupted_no_mask[mask]

    self.assertFalse(
        jnp.allclose(
            valid_output_original_no_mask,  # pyrefly: ignore[bad-argument-type]
            valid_output_corrupted_no_mask,  # pyrefly: ignore[bad-argument-type]
            atol=1e-5,
        ),
        msg="Outputs should differ when the mask is removed.",
    )

  def test_multi_head_cross_attention_different_lengths_and_mask(self):
    """Test cross-attention with different seq len and key masking."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
    )

    rng1, rng2 = jax.random.split(self.rng)

    # x (queries) has length 16
    x = jax.random.normal(rng1, (self.batch_size, self.seq_len_q, self.dim))

    # c (keys/values) has length 64
    c_original = jax.random.normal(
        rng2, (self.batch_size, self.seq_len_kv, self.dim)
    )

    # Mask applies to c (length 64). Keep first half, mask second half.
    half_seq_kv = self.seq_len_kv // 2

    # Explicitly cast to boolean for strict type safety
    single_mask = (jnp.arange(self.seq_len_kv) < half_seq_kv).astype(jnp.bool_)
    mask = jnp.broadcast_to(single_mask, (self.batch_size, self.seq_len_kv))

    variables = module.init(rng2, x, c_original, mask=mask)

    # Check Output Shape
    output_original = module.apply(variables, x, c_original, mask=mask)
    self.assertEqual(output_original.shape, x.shape)  # pyrefly: ignore[missing-attribute]

    # Check Mask Invariance on Keys
    noise = jax.random.normal(rng1, c_original.shape)
    c_corrupted = jnp.where(
        jnp.expand_dims(mask, -1), c_original, c_original + noise
    )

    output_corrupted = module.apply(variables, x, c_corrupted, mask=mask)

    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        output_original,
        output_corrupted,
        atol=1e-5,
    )

    # Ensure that WITHOUT the mask, the corrupted keys DO change the outputs.
    output_original_no_mask = module.apply(variables, x, c_original, mask=None)
    output_corrupted_no_mask = module.apply(
        variables, x, c_corrupted, mask=None
    )

    self.assertFalse(
        jnp.allclose(
            output_original_no_mask, output_corrupted_no_mask, atol=1e-5  # pyrefly: ignore[bad-argument-type]
        ),
        msg=(
            "Outputs should differ when the mask is removed and keys are"
            " altered."
        ),
    )

  @parameterized.named_parameters(
      ("self_attention_linear", None, True, LinearRoPEPositions()),
      ("self_attention_square", None, True, SquareRoPEPositions()),
      ("cross_attention_linear", "c", True, LinearRoPEPositions()),
      ("cross_attention_square", "c", True, SquareRoPEPositions()),
      ("self_attention_no_rope", None, False, LinearRoPEPositions()),
      ("cross_attention_no_rope", "c", False, LinearRoPEPositions()),
  )
  def test_multi_head_attention_output_shape(
      self,
      context: Float["batch sequence2 dim1"] | None,
      use_rope: bool,
      rope_positions_fn: RoPEPositionsFn,
  ):
    """Tests the output shape of MultiHeadAttention."""
    c = self.c if context == "c" else None
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        use_rope=use_rope,
        rope_positions_fn=rope_positions_fn,
    )
    x_curr = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x_curr, c)
    output = module.apply(variables, x_curr, c)
    self.assertEqual(output.shape, x_curr.shape)  # pyrefly: ignore[missing-attribute]

  def test_multi_head_attention_zero_init_output(self):
    """Test zero_init_output=True initializes output to zeros."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        zero_init_output=True,
    )
    variables = module.init(self.rng, self.x, self.c)

    # 1. Check that the kernel and bias of the output projection are zeros.
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
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
    self.assertTrue(jnp.allclose(output, zeros_output))  # pyrefly: ignore[bad-argument-type]

  @parameterized.named_parameters(
      ("qk_norm", True),
      ("no_qk_norm", False),
  )
  def test_multi_head_attention_params_shape(self, normalize_qk: bool):
    """Tests MultiHeadAttention has the correct parameters."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        use_rope=True,
        rope_positions_fn=SquareRoPEPositions(),
        normalize_qk=normalize_qk,
    )
    variables = module.init(self.rng, self.x, self.c)

    # Check that the variables have the correct shape.
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)
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

  @parameterized.named_parameters(
      dict(
          testcase_name="self_attention_wrong_mask_shape",
          pass_context=False,
          invalid_seq_len=42,
          expected_regex=(
              r"In self-attention, mask shape \(\d+, \d+\) does not match"
              r" expected shape \(\d+, \d+\)"
          ),
      ),
      dict(
          testcase_name="cross_attention_wrong_mask_shape",
          pass_context=True,
          invalid_seq_len=42,
          expected_regex=(
              "is not shape-compatible with 'batch sequence1|sequence2'"
          ),
      ),
      dict(
          testcase_name="cross_attention_but_mask_has_x_shape",
          pass_context=True,
          invalid_seq_len=16,
          expected_regex=(
              r"In cross-attention, mask shape \(\d+, \d+\) does not match"
              r" expected shape \(\d+, \d+\)"
          ),
      ),
  )
  def test_multi_head_attention_invalid_mask_shape_raises_error(
      self, pass_context: bool, invalid_seq_len: int, expected_regex: str
  ):
    """Tests that an invalid mask shape raises a ValueError."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(num_heads=self.num_heads)
    )

    c = self.c if pass_context else None

    # Create the mask with the intentionally incorrect shape
    invalid_mask = jnp.ones((self.batch_size, invalid_seq_len), dtype=jnp.bool_)

    # Verify that calling the module with this mask triggers the shape exception
    with self.assertRaisesRegex(
        (ValueError, kt.KTypeCheckError), expected_regex
    ):
      module.init(self.rng, self.x, c, mask=invalid_mask)

  # MARK: Dropout Tests

  def test_multi_head_attention_dropout_disabled_during_evaluation(self):
    """Verifies dropout is inactive when is_training=False (evaluation mode)."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        dropout_rate=0.5,
    )

    rng1, rng2 = jax.random.split(self.rng)
    x_rand = jax.random.normal(
        rng1, (self.batch_size, self.seq_len_q, self.dim)
    )

    variables = module.init(rng2, x_rand, c=None)

    output_eval_1 = module.apply(variables, x_rand, c=None, is_training=False)
    output_eval_2 = module.apply(variables, x_rand, c=None, is_training=False)

    np.testing.assert_allclose(  # pyrefly: ignore[no-matching-overload]
        output_eval_1,
        output_eval_2,
        atol=1e-6,
    )

  def test_multi_head_attention_dropout_active_during_training(self):
    """Verifies dropout alters outputs when is_training=True."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        dropout_rate=0.5,
    )

    rng1, rng2, rng_dropout1, rng_dropout2 = jax.random.split(self.rng, 4)
    x_rand = jax.random.normal(
        rng1, (self.batch_size, self.seq_len_q, self.dim)
    )

    variables = module.init(rng2, x_rand, c=None)

    output_train_1 = module.apply(
        variables,
        x_rand,
        c=None,
        is_training=True,
        rngs={"dropout": rng_dropout1},
    )
    output_train_2 = module.apply(
        variables,
        x_rand,
        c=None,
        is_training=True,
        rngs={"dropout": rng_dropout2},
    )

    self.assertFalse(jnp.allclose(output_train_1, output_train_2, atol=1e-5))  # pyrefly: ignore[bad-argument-type]

  def test_multi_head_attention_dropout_scales_retained_activations(self):
    """Verifies dropout scales active entries by 1 / (1 - rate) during training."""
    rate = 0.5
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        dropout_rate=rate,
    )

    rng1, rng2, rng_dropout = jax.random.split(self.rng, 3)
    x_rand = jax.random.normal(
        rng1, (self.batch_size, self.seq_len_q, self.dim)
    )

    variables = module.init(rng2, x_rand, c=None)

    output_eval = module.apply(variables, x_rand, c=None, is_training=False)
    output_train = module.apply(
        variables,
        x_rand,
        c=None,
        is_training=True,
        rngs={"dropout": rng_dropout},
    )

    max_train_val = float(jnp.max(jnp.abs(output_train)))  # pyrefly: ignore[bad-argument-type]
    max_eval_val = float(jnp.max(jnp.abs(output_eval)))  # pyrefly: ignore[bad-argument-type]

    self.assertGreater(max_train_val, max_eval_val)

  # MARK: use_bias tests

  @parameterized.named_parameters(
      ("with_bias", True),
      ("no_bias", False),
  )
  def test_multi_head_attention_use_bias(self, use_bias):
    """Verifies that use_bias controls bias in all projections."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        use_bias=use_bias,
    )
    variables = module.init(self.rng, self.x, c=None)
    leaves_with_paths = test_helpers.get_leaves_with_paths(variables)

    bias_paths = [p for p in leaves_with_paths if "bias" in p]
    if use_bias:
      self.assertLen(bias_paths, 4)
    else:
      self.assertEmpty(bias_paths)

  def test_multi_head_attention_no_bias_output_shape(self):
    """Verifies output shape is correct when use_bias=False."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        use_bias=False,
    )
    variables = module.init(self.rng, self.x, c=None)
    output = module.apply(variables, self.x, c=None, is_training=False)
    self.assertEqual(output.shape, self.x.shape)  # pyrefly: ignore[missing-attribute]

  def test_multi_head_attention_no_bias_param_shapes(self):
    """Verifies parameter shapes when use_bias=False."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        use_bias=False,
    )
    variables = module.init(self.rng, self.x, c=None)
    variables_shapes = test_helpers.get_pytree_shapes(variables)  # pyrefly: ignore[bad-argument-type]

    expected = {
        "params": {
            "Dense_Q": {"kernel": (self.dim, self.dim)},
            "Dense_K": {"kernel": (self.dim, self.dim)},
            "Dense_V": {"kernel": (self.dim, self.dim)},
            "Dense_Output": {"kernel": (self.dim, self.dim)},
        }
    }
    self.assertDictEqual(expected, variables_shapes)

  # MARK: qk_norm_method tests

  @parameterized.named_parameters(
      ("l2", "l2"),
      ("rms_norm", "rms_norm"),
  )
  def test_qk_norm_method_output_shape(self, qk_norm_method):
    """Verifies output shape is correct for each qk_norm_method."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method=qk_norm_method,
    )
    variables = module.init(self.rng, self.x, c=None)
    output = module.apply(variables, self.x, c=None, is_training=False)
    self.assertEqual(output.shape, self.x.shape)  # pyrefly: ignore[missing-attribute]

  def test_qk_norm_l2_param_shapes(self):
    """Verifies L2 QK normalization creates a norm_qk_scale parameter."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method="l2",
    )
    variables = module.init(self.rng, self.x, c=None)
    leaves = test_helpers.get_leaves_with_paths(variables)
    # L2 method should have a norm_qk_scale param
    self.assertIn("params/norm_qk_scale", leaves)
    self.assertEqual(leaves["params/norm_qk_scale"].shape, (1, 1, 1, 1))
    # Should NOT have RMSNorm_Q/K
    rms_paths = [p for p in leaves if "RMSNorm" in p]
    self.assertEmpty(rms_paths)

  def test_qk_norm_rms_norm_param_shapes(self):
    """Verifies RMSNorm QK normalization creates RMSNorm_Q/K scale params."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method="rms_norm",
    )
    variables = module.init(self.rng, self.x, c=None)
    leaves = test_helpers.get_leaves_with_paths(variables)
    # RMSNorm method should have RMSNorm_Q/scale and RMSNorm_K/scale
    self.assertIn("params/RMSNorm_Q/scale", leaves)
    self.assertIn("params/RMSNorm_K/scale", leaves)
    self.assertEqual(leaves["params/RMSNorm_Q/scale"].shape, (self.head_dim,))
    self.assertEqual(leaves["params/RMSNorm_K/scale"].shape, (self.head_dim,))
    # Should NOT have norm_qk_scale
    self.assertNotIn("params/norm_qk_scale", leaves)

  def test_qk_norm_rms_norm_with_rope(self):
    """Verifies RMSNorm QK norm works with RoPE (norm before RoPE)."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method="rms_norm",
        use_rope=True,
        rope_positions_fn=SquareRoPEPositions(),
    )
    x = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x, c=None)
    output = module.apply(variables, x, c=None, is_training=False)
    self.assertEqual(output.shape, x.shape)  # pyrefly: ignore[missing-attribute]

  def test_qk_norm_l2_with_rope(self):
    """Verifies L2 QK norm works with RoPE (norm before RoPE)."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method="l2",
        use_rope=True,
        rope_positions_fn=SquareRoPEPositions(),
    )
    x = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x, c=None)
    output = module.apply(variables, x, c=None, is_training=False)
    self.assertEqual(output.shape, x.shape)  # pyrefly: ignore[missing-attribute]

  def test_qk_norm_disabled_has_no_norm_params(self):
    """Verifies that normalize_qk=False creates no norm params."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=False,
    )
    variables = module.init(self.rng, self.x, c=None)
    leaves = test_helpers.get_leaves_with_paths(variables)
    norm_paths = [p for p in leaves if "norm_qk" in p or "RMSNorm" in p]
    self.assertEmpty(norm_paths)

  def test_qk_norm_invalid_method_raises_error(self):
    """Verifies that an invalid qk_norm_method raises ValueError."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads
        ),
        normalize_qk=True,
        qk_norm_method="invalid_method",  # type: ignore
    )
    with self.assertRaisesRegex(
        ValueError, "Unsupported QK normalization method"
    ):
      module.init(self.rng, self.x, c=None)

  @parameterized.named_parameters(
      # Format: (name, normalize_qk, qk_norm_method, seq_len_q, seq_len_kv, dim)
      # 1. Aligned Baseline
      ("aligned_no_norm", False, "l2", 128, 128, 128),
      ("aligned_l2_norm", True, "l2", 128, 128, 128),
      # 2. Mixed Boundaries (covers 1 below/above and 10 above/below in
      # cross-attention)
      ("mixed_boundary_1_l2", True, "l2", 127, 129, 128),
      ("mixed_boundary_10_rms", True, "rms_norm", 138, 118, 128),
      # 3. Extreme / Small Sizes (also covers dim 64)
      ("minimum_size", True, "l2", 1, 1, 64),
      ("small_context_dim_64", True, "l2", 8, 32, 64),
      # 4. Large Context
      ("large_context", True, "l2", 256, 512, 128),
  )
  def test_flash_attention_correctness(
      self, normalize_qk, qk_norm_method, seq_len_q, seq_len_kv, dim
  ):
    """Verifies that the Flash Attention forward and backward passes match Naive."""

    # Setup inputs
    rng_init, rng_eval, rng_grad = jax.random.split(self.rng, 3)
    x = jax.random.normal(rng_eval, (self.batch_size, seq_len_q, dim))
    c = jax.random.normal(rng_eval, (self.batch_size, seq_len_kv, dim))

    # Compile two modules: one with naive, one with flash
    module_naive = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads, use_flash_attention=False
        ),
        normalize_qk=normalize_qk,
        qk_norm_method=qk_norm_method,
    )
    module_flash = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads, use_flash_attention=True
        ),
        normalize_qk=normalize_qk,
        qk_norm_method=qk_norm_method,
    )

    variables = module_naive.init(rng_init, x, c)

    # Define forward wrapper to use with jax.vjp
    def forward_fn(module, q, k):
      return module.apply(variables, q, k, is_training=False)

    # 1. Forward and VJP: Naive Attention
    out_naive, vjp_fn_naive = jax.vjp(
        lambda q, k: forward_fn(module_naive, q, k), x, c
    )

    # 2. Forward and VJP: Flash Attention under TPU interpret mode
    with pltpu.force_tpu_interpret_mode():
      out_flash, vjp_fn_flash = jax.vjp(
          lambda q, k: forward_fn(module_flash, q, k), x, c
      )

    # Assert Forward Correctness
    np.testing.assert_allclose(out_flash, out_naive, atol=1e-5, rtol=1e-5)

    # 3. Propagate incoming gradients backward
    g = jax.random.normal(rng_grad, out_naive.shape)
    dx_naive, dc_naive = vjp_fn_naive(g)

    with pltpu.force_tpu_interpret_mode():
      dx_flash, dc_flash = vjp_fn_flash(g)

    # Assert VJP Correctness (Gradients of inputs)
    np.testing.assert_allclose(dx_flash, dx_naive, atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(dc_flash, dc_naive, atol=1e-5, rtol=1e-5)

  def test_flash_attention_fails_without_tpu_or_interpreter(self):
    """Verifies that Flash Attention fails when run on CPU without interpreter."""
    spec = attention.AttentionHeadsSpec(
        num_heads=self.num_heads,
        use_flash_attention=True,
    )
    module = attention.MultiHeadAttention(attention_heads_spec=spec)
    x = jnp.ones((self.batch_size, self.seq_len_q, self.dim))
    # We force interpret mode to None, which disables CPU interpreter emulation.
    with pltpu.force_tpu_interpret_mode(None):  # type: ignore[wrong-arg-types]
      with self.assertRaises((ValueError, RuntimeError)):
        variables = module.init(self.rng, x, None)
        module.apply(variables, x, None)

  def test_flash_attention_invalid_block_k_raises_error(self):
    """Verifies that block_k not divisible by 128 raises error in Flash Attention."""
    invalid_block_sizes = flash.BlockSizes(
        block_q=128,
        block_k_major=128,
        block_k=64,  # 64 is not a multiple of 128
        block_b=1,
    )
    spec = attention.AttentionHeadsSpec(
        num_heads=self.num_heads,
        use_flash_attention=True,
        block_sizes=invalid_block_sizes,
    )

    module = attention.MultiHeadAttention(attention_heads_spec=spec)
    x = jnp.ones((self.batch_size, self.seq_len_q, self.dim))
    mask = jnp.ones((self.batch_size, self.seq_len_q), dtype=jnp.bool_)

    with self.assertRaises((NotImplementedError, ValueError, RuntimeError)):
      variables = module.init(self.rng, x, None, mask=mask)
      module.apply(variables, x, None, mask=mask)

  def test_flash_attention_with_dropout_raises_error(self):
    """Verifies Flash Attention w/ dropout during training raises ValueError."""
    module = attention.MultiHeadAttention(
        attention_heads_spec=attention.AttentionHeadsSpec(
            num_heads=self.num_heads,
            use_flash_attention=True,
        ),
        dropout_rate=0.1,
    )

    # Verify exception is raised during initialization (with is_training=True)
    with self.assertRaisesRegex(
        ValueError, "Flash attention is not supported with dropout."
    ):
      module.init(self.rng, self.x, None, is_training=True)

    # Initialize with is_training=False under interpret mode, then verify
    # exception during apply with is_training=True
    with pltpu.force_tpu_interpret_mode():
      variables = module.init(self.rng, self.x, None, is_training=False)
    with self.assertRaisesRegex(
        ValueError, "Flash attention is not supported with dropout."
    ):
      module.apply(variables, self.x, None, is_training=True)


if __name__ == "__main__":
  absltest.main()
