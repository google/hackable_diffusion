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

from typing import Literal, Sequence
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
import jax.numpy as jnp
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
FFNType = Literal["dense", "swiglu"]

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
# MARK: GatingSwiGLU
################################################################################


class GatingSwiGLU(nn.Module):
  """A Dense layer variant that outputs SwiGLU gating directly.

  A gated feed-forward network using SiLU (Swish) activation for the gate,
  following "GLU Variants Improve Transformer" (Shazeer, 2020):
  https://arxiv.org/abs/2002.05202

  Projects the input dimension to features * 2, chunks the result across the
  last dimension, and gates the activation channel with SiLU.
  """

  features: int
  use_bias: bool = False
  dtype: DType = jnp.float32

  @nn.compact
  @kt.typechecked
  def __call__(self, x: Float["*batch d_in"]) -> Float["*batch features"]:
    # Project to double feature width
    gate_and_val = nn.Dense(
        features=self.features * 2,
        use_bias=self.use_bias,
        dtype=self.dtype,
        name="Dense_Gate_Val",
    )(x)

    # Split and apply SiLU gating (mirrors torch.chunk(2, dim=-1))
    val, gate = jnp.split(gate_and_val, 2, axis=-1)
    return val * nn.silu(gate)


################################################################################
# MARK: FeedForward Unified Block
################################################################################


class FeedForward(nn.Module):
  """A unified FeedForward block selecting between SwiGLU or traditional layers.

  Attributes:
    output_size: Output dimension (residual stream width).
    hidden_size: Intermediate bottleneck network dimension.
    ffn_type: Layout type toggle. - 'swiglu' uses a gated SwiGLU projection
      layer. - 'dense' uses a classic dense projection followed by an
      activation.
    activation: Name of the activation function to use when `ffn_type='dense'`.
      This parameter is explicitly ignored when `ffn_type='swiglu'` because the
      SwiGLU path uses its own mathematical gating mechanism (SiLU/Swish).
    use_bias: Whether to use bias in the up and down projection layers. Modern
      architectures (LLaMA, Flux) typically set this to False.
    zero_init_output: If True, the terminal linear projections are zeroed out
      ensuring the block satisfies identity-at-init behavior.
    dropout_rate: Activation state dropout regularization coefficient.
    dtype: Numerical precision layout representation format.
  """

  output_size: int
  hidden_size: int
  ffn_type: FFNType = "dense"
  activation: str = "gelu"
  use_bias: bool = False
  zero_init_output: bool = False
  dropout_rate: float = 0.0
  dtype: DType = jnp.float32

  def setup(self):
    # Regularization Dropout Layer
    self.dropout = nn.Dropout(rate=self.dropout_rate)

    # Down Projection Layer Config
    down_kernel_init = (
        nn.initializers.zeros_init()
        if self.zero_init_output
        else nn.initializers.lecun_normal()
    )

    self.down_proj = nn.Dense(
        features=self.output_size,
        use_bias=self.use_bias,
        kernel_init=down_kernel_init,
        dtype=self.dtype,
        name="Dense_Down",
    )

  @nn.compact
  @kt.typechecked
  def __call__(
      self, x: Float["batch *other_dims output_size"], *, is_training: bool
  ) -> Float["batch *other_dims output_size"]:
    # Up-projection step
    if self.ffn_type == "swiglu":
      # Project to double feature width
      gate_and_val = nn.Dense(
          features=self.hidden_size * 2,
          use_bias=self.use_bias,
          dtype=self.dtype,
          name="Dense_Up",
      )(x)
      # Split and apply SiLU gating
      val, gate = jnp.split(gate_and_val, 2, axis=-1)
      x = val * nn.silu(gate)
    elif self.ffn_type == "dense":
      x = nn.Dense(
          features=self.hidden_size,
          use_bias=self.use_bias,
          dtype=self.dtype,
          name="Dense_Up",
      )(x)
      # Apply the configured activation function
      activation_fn = getattr(nn, self.activation)
      x = activation_fn(x)
    else:
      raise ValueError(f"Unknown ffn_type mapping strategy: {self.ffn_type!r}")

    # Middle regularization step
    x = self.dropout(x, deterministic=not is_training)

    # Final down-projection step
    x = self.down_proj(x)

    return x
