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

"""MLP blocks."""

from typing import Sequence
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

################################################################################
# MARK: MLP
################################################################################


class MLP(nn.Module):
  """A simple MLP."""

  hidden_sizes: Sequence[int]
  output_size: int
  activation: str
  activate_final: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32
  zero_init_output: bool = False

  @nn.compact
  @kt.typechecked
  def __call__(
      self, x: Float['batch *other_dims num_inputs'], *, is_training: bool
  ) -> Float['batch *other_dims num_features']:
    """Applies MLP blocks to the input tensor.

    Args:
      x: The input tensor.
      is_training: Whether the model is in training mode. Used only for dropout.

    Returns:
      The output tensor after applying the MLP blocks.
    """
    activation_fn = getattr(nn, self.activation)
    output = x
    for i, hidden_size in enumerate(self.hidden_sizes):
      output = nn.Dense(
          features=hidden_size, name=f'Dense_Hidden_{i}', dtype=self.dtype
      )(output)
      output = activation_fn(output)
      output = nn.Dropout(
          rate=self.dropout_rate, deterministic=not is_training
      )(output)

    if self.zero_init_output:
      output = nn.Dense(
          features=self.output_size,
          kernel_init=nn.initializers.zeros_init(),
          bias_init=nn.initializers.zeros_init(),
          dtype=self.dtype,
          name='Dense_Output',
      )(output)
    else:
      output = nn.Dense(
          features=self.output_size, name='Dense_Output', dtype=self.dtype
      )(output)

    if self.activate_final:
      output = activation_fn(output)

    return output


################################################################################
# MARK: SwiGLU
################################################################################


class SwiGLU(nn.Module):
  """SwiGLU feed-forward network.

  A gated feed-forward network using SiLU (Swish) activation for the gate,
  following "GLU Variants Improve Transformer" (Shazeer, 2020):
  https://arxiv.org/abs/2002.05202

  The forward pass is:

    gate_and_val = x @ W_up           # (*, hidden_size) -> (*, ff_size * 2)
    val, gate = split(gate_and_val)   # (*, ff_size) each
    x = val * SiLU(gate)              # (*, ff_size)
    x = dropout(x)
    x = x @ W_down                    # (*, ff_size) -> (*, hidden_size)

  Attributes:
    hidden_size: Output dimension (residual stream width).
    ff_size: Intermediate dimension (before gating).
    zero_init_output: If True, the down-projection kernel is initialized to
      zeros so the block starts as identity.
    dropout_rate: Dropout rate applied after gating.
    dtype: Data type for computation.
  """

  hidden_size: int
  ff_size: int
  zero_init_output: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  @nn.compact
  @kt.typechecked
  def __call__(
      self, x: Float['batch *other_dims hidden_size'], *, is_training: bool
  ) -> Float['batch *other_dims hidden_size']:
    # Up-projection: (*, hidden_size) -> (*, ff_size * 2).
    gate_and_val = nn.Dense(
        features=self.ff_size * 2,
        use_bias=False,
        dtype=self.dtype,
        name='Dense_Up',
    )(x)
    # Split into value and gate, apply SiLU gating.
    val, gate = jnp.split(gate_and_val, 2, axis=-1)
    x = val * nn.silu(gate)
    x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)
    # Down-projection: (*, ff_size) -> (*, hidden_size).
    down_kernel_init = (
        nn.initializers.zeros_init()
        if self.zero_init_output
        else nn.initializers.lecun_normal()
    )
    x = nn.Dense(
        features=self.hidden_size,
        use_bias=False,
        dtype=self.dtype,
        kernel_init=down_kernel_init,
        name='Dense_Down',
    )(x)
    return x
