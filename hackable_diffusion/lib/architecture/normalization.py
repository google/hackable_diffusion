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

"""Normalization layers.

Implements the following methods:
- RMSNorm: https://arxiv.org/abs/1910.07467
- GroupNorm: https://arxiv.org/abs/1803.08494
- LayerNorm: https://arxiv.org/abs/1607.06450
"""

import einops
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import jax_helpers
from hackable_diffusion.lib.architecture import arch_typing
import jax.numpy as jnp
import kauldron.ktyping as kt


################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
Bool = hd_typing.Bool

NormalizationType = arch_typing.NormalizationType


################################################################################
# MARK: NormalizationLayer
################################################################################


class NormalizationLayer(nn.Module):
  """A generic normalization layer with optional conditioning.

  This layer applies a specified normalization method to the input tensor `x`.
  If `conditional` is True, it then applies a learned scale and shift
  transformation conditioned on an embedding `c`.

  The scale and shift are computed from conditioning `c` using a dense layer.

  Supported normalization methods:

  - RMSNorm: https://arxiv.org/abs/1910.07467 with `reduction_axes=-1`, meaning
  that normalization statistics are computed along the last dimension.

  - GroupNorm: https://arxiv.org/abs/1803.08494 with `reduction_axes=None`,
  meaning that normalization statistics are computed over all dimensions except
  the batch dimension.

  - LayerNorm: https://arxiv.org/abs/1607.06450 with `reduction_axes=-1`,
  meaning
  that normalization statistics are computed along the last dimension.

  Sharp bit: for the normalization statistics to be correct in the case of
  padded inputs, please provide a mask when calling this layer.

  Attributes:
    normalization_method: The normalization method to use.
    conditional: Whether to apply conditional scaling and shifting.
    num_groups: The number of groups to use for group normalization. If None,
      group normalization cannot be used and an error will be raised.
    epsilon: Epsilon value for numerical stability in normalization.
    dtype: The data type of the computation.
    use_bias: Whether to use bias in the normalization layer.
    use_scale: Whether to use scale in the normalization layer.
  """

  normalization_method: NormalizationType
  conditional: bool
  num_groups: int | None = None
  epsilon: float = 1e-5
  dtype: DType = jnp.float32
  use_bias: bool = True
  use_scale: bool = True

  def setup(self):
    if (
        self.normalization_method == NormalizationType.GROUP_NORM
        and self.num_groups is None
    ):
      raise ValueError("num_groups must be specified for Group normalization.")

  @nn.compact
  @kt.typechecked
  def __call__(
      self,
      x: Float["batch *other channels"],
      c: Float["batch cond_dim"] | None = None,
      mask: (
          Bool["batch *#other #channels"] | Bool["batch *#other"] | None
      ) = None,
  ) -> Float["batch *other channels"]:
    """Run the normalization layer.

    If `mask` is provided, it is expected to be broadcastable to the shape of
    `x`. This is in accordance with Flax conventions. The mask indicates at
    which positions the reduction features (like mean and variance in the case
    of GroupNorm) should be computed.

    Args:
      x: The input tensor.
      c: The conditioning tensor.
      mask: (Optional) The mask to use for normalization. The value of the mask
        is true when the element is valid and false when it is padding, i.e., we
        only compute the reduction over the valid values.

    Returns:
      The normalized tensor.
    """

    x_shape = x.shape
    ch = x_shape[-1]

    if self.normalization_method == NormalizationType.RMS_NORM:
      x = nn.RMSNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          reduction_axes=-1,  # For (B ... ch) results in (B ... ) RMS values.
          feature_axes=-1,  # Per channel scale.
          use_scale=self.use_scale,
      )(x=x, mask=mask)
    elif self.normalization_method == NormalizationType.GROUP_NORM:

      # If using GroupNorm the mask data must be such that the last dimension
      # corresponds to the channels.
      if mask is not None and mask.shape[-1] != x_shape[-1]:
        raise ValueError(
            "If using GroupNorm with a mask, the mask's last dimension must"
            " match the input's channel dimension. Otherwise, one cannot"
            " reshape the mask during the grouping operation."
        )

      x = nn.GroupNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          reduction_axes=None,  # Reduction over all non-batch axes.
          num_groups=self.num_groups,
          use_bias=self.use_bias,
          use_scale=self.use_scale,
      )(x=x, mask=mask)
    elif self.normalization_method == NormalizationType.LAYER_NORM:
      x = nn.LayerNorm(
          epsilon=self.epsilon,
          dtype=self.dtype,
          use_bias=self.use_bias,
          use_scale=self.use_scale,
          reduction_axes=-1,  # For (B ... ch) results in (B ... ) values.
          feature_axes=-1,  # Per channel scale.
      )(x=x, mask=mask)
    else:
      raise ValueError(
          "Unsupported normalization method: %s" % self.normalization_method
      )

    if self.conditional:

      scale_and_shift = nn.Dense(
          ch * 2,
          kernel_init=nn.zeros_init(),
          bias_init=nn.zeros_init(),
          dtype=self.dtype,
      )(c)
      scale, shift = jnp.split(scale_and_shift, 2, axis=-1)  # (B, ch) each.

      x = einops.rearrange(x, "b ... c -> b c ...")  # (B, ch, ...).
      scale = jax_helpers.bcast_right(scale, x.ndim)
      shift = jax_helpers.bcast_right(shift, x.ndim)
      x = (1.0 + scale) * x + shift
      x = einops.rearrange(x, "b c ... -> b ... c")

    return x


################################################################################
# MARK: NormalizationLayerFactory
################################################################################


class NormalizationLayerFactory:
  """A factory for creating normalization layers.

  This class provides a convenient way to configure and create
  `NormalizationLayer` instances. It separates the configuration of the
  normalization from its application, allowing for easy injection of different
  normalization strategies.

  It can create both conditional and unconditional normalization layers via
  the `conditional_norm_factory` and `unconditional_norm_factory` properties.

  Attributes:
    normalization_method: The normalization method to use (e.g., 'rms_norm').
    num_groups: The number of groups to use for group normalization. If None,
      group normalization cannot be used and an error will be raised.
    epsilon: A small float added to variance to avoid dividing by zero.
    dtype: The data type of the computation.
    use_bias: Whether to use bias in the normalization layer.
    use_scale: Whether to use scale in the normalization layer.
  """

  def __init__(
      self,
      normalization_method: NormalizationType,
      num_groups: int | None = None,
      epsilon: float = 1e-5,
      dtype: DType = jnp.float32,
      use_bias: bool = True,
      use_scale: bool = True,
  ):
    self.normalization_method = normalization_method
    self.epsilon = epsilon
    self.num_groups = num_groups
    self.dtype = dtype
    self.use_bias = use_bias
    self.use_scale = use_scale

  def unconditional_norm(
      self, norm_name: str = "UnconditionalNorm"
  ) -> NormalizationLayer:
    """Returns a factory for creating unconditional normalization layers."""
    return NormalizationLayer(
        normalization_method=self.normalization_method,
        conditional=False,
        num_groups=self.num_groups,
        epsilon=self.epsilon,
        name=norm_name,
        dtype=self.dtype,
        use_bias=self.use_bias,
        use_scale=self.use_scale,
    )

  def conditional_norm(
      self, norm_name: str = "ConditionalNorm"
  ) -> NormalizationLayer:
    """Returns a factory for creating conditional normalization layers."""
    return NormalizationLayer(
        normalization_method=self.normalization_method,
        conditional=True,
        num_groups=self.num_groups,
        epsilon=self.epsilon,
        name=norm_name,
        dtype=self.dtype,
        use_bias=self.use_bias,
        use_scale=self.use_scale,
    )
