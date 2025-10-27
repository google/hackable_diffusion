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

"""Sequence embeddings."""

from typing import Sequence

import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp
import numpy as np


################################################################################
# MARK: Type aliases
################################################################################

Float = hd_typing.Float
Int = hd_typing.Int
Num = hd_typing.Num

RoPEPositionType = arch_typing.RoPEPositionType

################################################################################
# MARK: Sequence embedding modules
################################################################################


class SinusoidalSequenceEmbedding(nn.Module):
  """Sequence (positional) embedding as in Transformers."""

  num_features: int
  base_frequency: float = 10000.0
  rescale_factor: float = 1000.0

  def setup(self):
    if self.num_features <= 0:
      raise ValueError("Number of features must be positive.")

  @nn.compact
  @typechecked
  def __call__(
      self, inputs: Num["batch *#data_shape"]
  ) -> Float["batch num_features"]:
    if inputs.size != inputs.shape[0]:
      raise ValueError("Inputs must be a (maybe broadcasted) 1D array.")
    inputs = inputs.reshape((inputs.shape[0],))
    half_dim = self.num_features // 2
    e = jnp.log(self.base_frequency) / (half_dim - 1)
    embedding = jnp.exp(-e * jnp.arange(half_dim))
    # Rescale inputs to cover a wider range of frequencies
    inputs_rescaled = inputs * self.rescale_factor
    embedding = inputs_rescaled[:, None] * embedding
    embedding = jnp.concatenate(
        [jnp.cos(embedding), jnp.sin(embedding)], axis=-1
    )
    if self.num_features % 2 == 1:
      embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
    return embedding


class RandomFourierSequenceEmbedding(nn.Module):
  """Random Fourier sequence embedding.

  This follows the implementation of https://arxiv.org/abs/2006.10739.
  """

  num_features: int
  fourier_scale: float = 16.0

  def setup(self):
    if self.num_features <= 0:
      raise ValueError("Number of features must be positive.")
    self.freqs = self.param(
        "FourierFrequencies",
        nn.initializers.normal(stddev=1.0),
        (1, self.num_features // 2),
    )

  @nn.compact
  @typechecked
  def __call__(
      self, inputs: Num["batch *#data_shape"]
  ) -> Float["batch num_features"]:
    if inputs.size != inputs.shape[0]:
      raise ValueError("Inputs must be a (maybe broadcasted) 1D array.")
    inputs = inputs.reshape((inputs.shape[0],))
    freqs = jax.lax.stop_gradient(self.freqs) * self.fourier_scale
    embedding = inputs[:, None] * (2 * jnp.pi * freqs)
    embedding = jnp.concatenate(
        [jnp.cos(embedding), jnp.sin(embedding)], axis=-1
    )
    if self.num_features % 2 == 1:
      embedding = jnp.pad(embedding, ((0, 0), (0, 1)))
    return embedding


class RoPESequenceEmbedding(nn.Module):
  """Sequence (positional) embedding as in https://arxiv.org/abs/2104.09864."""

  rope_position_type: RoPEPositionType = RoPEPositionType.SQUARE
  max_rotary_wavelength: int = 10_000

  @typechecked
  def _get_positions(
      self, x: Float["*batch sequence dim"]
  ) -> Sequence[Int["*batch sequence"]]:
    *b, t, _ = x.shape
    n_batch_dims = len(b)
    b = tuple(b)

    if self.rope_position_type == "linear":
      position = jnp.arange(start=0, stop=t)
      position = jnp.reshape(position, (1,) * n_batch_dims + (-1,))
      position = jnp.broadcast_to(position, b + (t,))
      return (position,)
    elif self.rope_position_type == "square":
      # get the square position grid
      t_sqrt = int(np.sqrt(t))
      if t != np.square(t_sqrt):
        raise ValueError("Sequence length must be a perfect square.")

      sq_arange = jnp.arange(start=0, stop=t_sqrt)
      position_x, position_y = jnp.meshgrid(sq_arange, sq_arange, indexing="ij")
      # [t_sqrt, t_sqrt]

      position_x = jnp.reshape(position_x, (1,) * n_batch_dims + (-1,))
      position_y = jnp.reshape(position_y, (1,) * n_batch_dims + (-1,))
      # [1, ..., t]

      position_x = jnp.broadcast_to(position_x, b + (t,))
      position_y = jnp.broadcast_to(position_y, b + (t,))
      # [*b, t]

      return (position_x, position_y)
    else:
      raise ValueError(f"Unknown RoPE position type: {self.rope_position_type}")

  @nn.compact
  @typechecked
  def __call__(
      self, x: Float["*batch sequence dim"]
  ) -> Float["*batch sequence dim"]:
    positions = self._get_positions(x)
    # list of elements of shape [*b, t]

    # dimension compatible with the positions
    if x.shape[-1] % (2 * len(positions)) != 0:
      raise ValueError(
          "Embedding dimension must be divisible by 2 * number of positions."
      )

    # compute number of features and (inverse) frequency exponents per position
    num_features = x.shape[-1] // len(positions)
    freq_exponents = (2.0 / num_features) * jnp.arange(num_features // 2)
    freq_exponents = jnp.power(self.max_rotary_wavelength, freq_exponents)
    inv_freq = jnp.pi / freq_exponents
    # [d / (2 * len(positions))]

    result = []
    x_parts = jnp.split(x, 2 * len(positions), axis=-1)

    for pos in positions:
      # pos is [*b, t]
      rescaled_pos = pos[..., None] * inv_freq
      # [*b, t, d / (2 * len(positions))] with broadcasting
      sin, cos = jnp.sin(rescaled_pos), jnp.cos(rescaled_pos)
      x1, x2, *x_parts = x_parts
      # x1 and x2 are [*b, t, d / (2 * len(positions))]
      # apply the rotation to the embedding
      result += [x1 * cos - x2 * sin, x2 * cos + x1 * sin]

    out = jnp.asarray(jnp.concatenate(result, axis=-1), x.dtype)
    return out
