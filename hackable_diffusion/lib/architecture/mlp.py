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

"""MLP backbones.

We only recommend using this backbone for very simple datasets.
"""

from typing import Literal, Sequence
from flax import linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import mlp_blocks
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax.numpy as jnp
import numpy as np

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float

DataArray = hd_typing.DataArray

ConditionalBackbone = arch_typing.ConditionalBackbone
ConditioningMechanism = arch_typing.ConditioningMechanism

################################################################################
# MARK: ConditionalMLP
################################################################################


class ConditionalMLP(ConditionalBackbone):
  """Conditional MLP backbone for diffusion models.

  Receives `x`, rocess them first separately using `hidden_sizes_preprocess`
  layers, producing `x_emb`. Then, it takes conditioning_embeddings and combines
  them with `x_emb`. After that, it feeds them into `hidden_sizes_postprocess`
  MLP blocks, and finally outputs the result.

  Attributes:
    hidden_sizes_preprocess: The number of layers in the preprocessing MLP.
    hidden_sizes_postprocess: The number of layers in the postprocessing MLP.
    activation: The activation function to use in the MLP.
    zero_init_output: Whether to initialize the output layer with zeros.
    dropout_rate: The dropout rate to use in the MLP.
    conditioning_mechanism: The conditioning mechanism to use.
    dtype: The dtype to use.
  """

  hidden_sizes_preprocess: Sequence[int]
  hidden_sizes_postprocess: Sequence[int]
  activation: str
  zero_init_output: bool
  dropout_rate: float
  conditioning_mechanism: Literal[
      ConditioningMechanism.SUM, ConditioningMechanism.CONCATENATE
  ]
  dtype: DType = jnp.float32

  @nn.compact
  @typechecked
  def __call__(
      self,
      x: DataArray,
      conditioning_embeddings: dict[ConditioningMechanism, Float['batch ...']],
      is_training: bool,
  ) -> DataArray:
    x_emb = jnp.reshape(x, shape=(x.shape[0], -1))
    # Input preprocessing.
    if self.hidden_sizes_preprocess:
      output_size = self.hidden_sizes_preprocess[-1]
      x_emb = mlp_blocks.MLP(
          hidden_sizes=self.hidden_sizes_preprocess[:-1],
          output_size=output_size,
          activation=self.activation,
          activate_final=True,
          dropout_rate=self.dropout_rate,
          dtype=self.dtype,
          name='PreprocessMLP',
      )(x_emb, is_training)

    # The conditioning was already processed by the `conditioning_encoder`, so
    # here we just need to concatenate it with the `x`.
    c_emb = conditioning_embeddings.get(self.conditioning_mechanism)
    if c_emb is None:
      raise ValueError('Conditioning embeddings are not provided.')
    if self.conditioning_mechanism == ConditioningMechanism.SUM:
      # Since the conditioning embedding may not have the same dimension as
      # `x_emb`, we project it to the same size as `x_emb`.
      c_emb = nn.Dense(
          features=x_emb.shape[-1],
          dtype=self.dtype,
          name='Dense_Projection_Conditioning',
      )(c_emb)
      emb = c_emb + x_emb
    elif self.conditioning_mechanism == ConditioningMechanism.CONCATENATE:
      emb = jnp.concatenate((c_emb, x_emb), axis=-1)
    else:
      raise ValueError(
          f'Unknown conditioning mechanism: {self.conditioning_mechanism}'
      )

    prod_shape = int(np.prod(x.shape[1:]))
    output = mlp_blocks.MLP(
        hidden_sizes=self.hidden_sizes_postprocess,
        output_size=prod_shape,
        activation=self.activation,
        activate_final=False,
        dropout_rate=self.dropout_rate,
        dtype=self.dtype,
        zero_init_output=self.zero_init_output,
        name='PostprocessMLP',
    )(emb, is_training)

    output = jnp.reshape(output, shape=x.shape)
    output = utils.optional_bf16_to_fp32(output)
    return output
