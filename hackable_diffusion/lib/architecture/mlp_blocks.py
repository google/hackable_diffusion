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

"""MLP blocks."""

from typing import Sequence
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp

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
  @typechecked
  def __call__(
      self, x: Float['batch num_inputs'], is_training: bool
  ) -> Float['batch num_features']:
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
