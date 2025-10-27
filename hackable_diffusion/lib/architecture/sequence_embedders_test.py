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

"""Tests for sequence embedders."""

from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import sequence_embedders
import jax
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Type Aliases
################################################################################

RoPEPositionType = arch_typing.RoPEPositionType
INVALID_INT = arch_typing.INVALID_INT

################################################################################
# MARK: Tests
################################################################################


def _get_invalid_num_features_params():
  """Generates parameters for testing invalid num_features."""
  params = []
  modes = ["sinusoidal_embedding", "random_fourier_embedding"]
  feature_values = [
      ("default", INVALID_INT),
      ("zero", 0),
      ("negative", -10),
  ]
  for mode in modes:
    for name_feature, value_feature in feature_values:
      params.append(
          (f"{mode}_{name_feature}_num_features", mode, value_feature)
      )
  return params


class SequenceEmbeddersTest(parameterized.TestCase):

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
      ("sinusoidal_embedding", "sinusoidal_embedding"),
      ("random_fourier_embedding", "random_fourier_embedding"),
  )
  def test_sequence_embedding(self, embedding_type: str):
    """Tests the output shape of SinusoidalSequenceEmbedding."""
    if embedding_type == "sinusoidal_embedding":
      module = sequence_embedders.SinusoidalSequenceEmbedding(
          num_features=self.dim
      )
    elif embedding_type == "random_fourier_embedding":
      module = sequence_embedders.RandomFourierSequenceEmbedding(
          num_features=self.dim
      )
    else:
      self.fail(f"Unknown embedding type: {embedding_type}")
    inputs = jnp.arange(self.batch_size)
    variables = module.init({"params": self.rng}, inputs)
    output = module.apply(variables, inputs)
    self.assertEqual(output.shape, (self.batch_size, self.dim))

  @parameterized.named_parameters(_get_invalid_num_features_params())
  def test_sequence_embedding_raises_error_on_invalid_num_features(
      self, mode: str, num_features: int
  ):
    """Tests that an error is raised for invalid num_features."""
    if mode == "sinusoidal_embedding":
      module = sequence_embedders.SinusoidalSequenceEmbedding(
          num_features=num_features
      )
    elif mode == "random_fourier_embedding":
      module = sequence_embedders.RandomFourierSequenceEmbedding(
          num_features=num_features
      )
    else:
      self.fail(f"Unknown mode: {mode}")
    inputs = jnp.arange(self.batch_size)
    with self.assertRaisesRegex(
        ValueError, "Number of features must be positive."
    ):
      module.init({"params": self.rng}, inputs)

  @parameterized.named_parameters(
      ("sinusoidal_embedding", "sinusoidal_embedding"),
      ("random_fourier_embedding", "random_fourier_embedding"),
  )
  def test_sequence_embedding_raises_error_on_invalid_inputs(self, mode: str):
    """Tests that sequence embeddings raises an error on invalid inputs."""
    if mode == "sinusoidal_embedding":
      module = sequence_embedders.SinusoidalSequenceEmbedding(
          num_features=self.dim
      )
    elif mode == "random_fourier_embedding":
      module = sequence_embedders.RandomFourierSequenceEmbedding(
          num_features=self.dim
      )
    else:
      self.fail(f"Unknown mode: {mode}")
    inputs = jnp.ones((self.batch_size, self.seq_len_q, self.dim))
    with self.assertRaisesRegex(ValueError, ".* 1D array."):
      module.init({"params": self.rng}, inputs)

  @parameterized.named_parameters(
      ("sinusoidal_embedding", "sinusoidal_embedding"),
      ("random_fourier_embedding", "random_fourier_embedding"),
  )
  def test_sequence_embedding_output_shape(self, mode: str):
    """Tests the output shape of sequence embeddings."""
    if mode == "sinusoidal_embedding":
      module = sequence_embedders.SinusoidalSequenceEmbedding(
          num_features=self.dim
      )
    elif mode == "random_fourier_embedding":
      module = sequence_embedders.RandomFourierSequenceEmbedding(
          num_features=self.dim
      )
    else:
      self.fail(f"Unknown mode: {mode}")
    inputs = jnp.arange(self.batch_size)
    variables = module.init({"params": self.rng}, inputs)
    output = module.apply(variables, inputs)
    self.assertEqual(output.shape, (self.batch_size, self.dim))

  def test_sinusoidal_embedding_has_no_params(self):
    """Tests that sinusoidal embeddings has no parameters."""
    module = sequence_embedders.SinusoidalSequenceEmbedding(
        num_features=self.dim
    )
    inputs = jnp.arange(self.batch_size)
    variables = module.init({"params": self.rng}, inputs)
    self.assertEmpty(variables)

  def test_random_fourier_embedding_params_are_not_updated(self):
    """Tests that RandomFourierSequenceEmbedding params are not updated."""
    module = sequence_embedders.RandomFourierSequenceEmbedding(
        num_features=self.dim
    )
    inputs = jnp.arange(self.batch_size, dtype=jnp.float32)
    variables = module.init({"params": self.rng}, inputs)
    initial_params = variables["params"]

    def loss_fn(params):
      output = module.apply({"params": params}, inputs)
      return jnp.sum(output)

    grads = jax.grad(loss_fn)(initial_params)

    # Check that the gradients are all zero.
    zero_grads = jax.tree_util.tree_map(jnp.zeros_like, initial_params)
    tree_leaves_are_close = jax.tree_util.tree_map(
        jnp.allclose, grads, zero_grads
    )
    self.assertTrue(jax.tree_util.tree_all(tree_leaves_are_close))

  # MARK: RoPESequenceEmbedding tests

  def test_rope_embedding_square_raises_error(self):
    """Tests that square RoPE raises an error for non-square sequences."""
    x_rope = jnp.ones((self.batch_size, 17, self.dim))  # Not a perfect square
    module = sequence_embedders.RoPESequenceEmbedding(
        rope_position_type=RoPEPositionType.SQUARE
    )
    with self.assertRaisesRegex(
        ValueError, "Sequence length must be a perfect square."
    ):
      module.init(self.rng, x_rope)

  @parameterized.named_parameters(
      ("linear", "linear", 3),
      ("square", "square", 5),
  )
  def test_rope_embedding_dimension_not_divisible(
      self, rope_position_type: RoPEPositionType, dimension: int
  ):
    """Tests that square RoPE raises an error if the embedding dim is not divisible by the denominator.

    The denominator is 2 in the case of linear RoPE and 4 in the case of
    square RoPE.

    Args:
      rope_position_type: The type of RoPE to use.
      dimension: The dimension of the embedding.
    """
    x_rope = jnp.ones((self.batch_size, self.seq_len_kv, dimension))  # Not divisible by the denominator # pylint: disable=line-too-long
    module = sequence_embedders.RoPESequenceEmbedding(
        rope_position_type=rope_position_type
    )
    with self.assertRaisesRegex(
        ValueError,
        "Embedding dimension must be divisible *.",
    ):
      module.init(self.rng, x_rope)

  @parameterized.named_parameters(
      ("linear", "linear"),
      ("square", "square"),
  )
  def test_rope_embedding_output_shape(
      self, rope_position_type: RoPEPositionType
  ):
    """Tests the output shape of RoPESequenceEmbedding."""
    module = sequence_embedders.RoPESequenceEmbedding(
        rope_position_type=rope_position_type
    )
    x_rope = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x_rope)
    output = module.apply(variables, x_rope)
    self.assertEqual(output.shape, x_rope.shape)

  def test_rope_embedding_has_no_params(self):
    """Tests that RoPESequenceEmbedding has no parameters."""
    module = sequence_embedders.RoPESequenceEmbedding()
    x_rope = jnp.ones((self.batch_size, self.seq_len_kv, self.dim))
    variables = module.init(self.rng, x_rope)
    self.assertEmpty(variables)


if __name__ == "__main__":
  absltest.main()
