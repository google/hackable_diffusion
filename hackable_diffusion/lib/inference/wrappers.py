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

"""Wrappers for inference functions for flax.linen and flax.nnx."""

import dataclasses
from typing import Protocol
from flax import nnx
import flax.linen as nn
from flax.nnx import bridge
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
import jax
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

PyTree = hd_typing.PyTree

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
TargetInfo = hd_typing.TargetInfo
TimeArray = hd_typing.TimeArray

InferenceFn = hd_api.InferenceFn


################################################################################
# MARK: FlaxLinenInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class FlaxLinenInferenceFn(InferenceFn):
  """Inference function protocol with a diffusion network given by nn.Module."""

  network: nn.Module
  params: PyTree  # pyrefly: ignore[not-a-type]

  @kt.typechecked
  def __call__(
      self,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfo:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""
    return self.network.apply(  # pyrefly: ignore[bad-return]
        {"params": self.params},
        time=time,
        xt=xt,
        conditioning=conditioning,
        is_training=False,
    )


################################################################################
# MARK: NNXDiffusionNetwork
################################################################################


class ConvertedNNXDiffusionNetwork(Protocol):
  """NNX diffusion network."""

  def __call__(
      self,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
      is_training: bool,
      rngs: nnx.Rngs,
  ) -> TargetInfo:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""
    ...


################################################################################
# MARK: NNXInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class FlaxNNXInferenceFn(InferenceFn):
  """Inference function protocol with a diffusion network given by nn.Module.

  Note: ``inference_seed`` is used for any stochastic layers (e.g., dropout)
  that remain active at inference time. Since ``is_training=False`` is always
  passed, dropout layers are typically disabled and this seed has no effect.
  If you need stochastic inference (e.g., MC dropout), provide different seeds.
  """

  nnx_network: ConvertedNNXDiffusionNetwork
  inference_seed: int = 0

  @kt.typechecked
  def __call__(
      self,
      time: TimeArray,  # pyrefly: ignore[not-a-type]
      xt: DataArray,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfo:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""
    return self.nnx_network(
        time=time,
        xt=xt,
        conditioning=conditioning,
        is_training=False,
        rngs=nnx.Rngs(self.inference_seed),
    )


################################################################################
# MARK: LinenToNNX
################################################################################


def convert_flax_linen_module_with_params_to_nnx(
    linen_module: nn.Module,
    restored_linen_params: PyTree,  # pyrefly: ignore[not-a-type]
    *init_args,
    **init_kwargs,
) -> nnx.Module:
  """Converts a Linen module and its parameters to a complete NNX module.

  This function bridges a classic Linen module to its NNX equivalent, loads the
  provided pre-trained weights, and returns a single, ready-to-use NNX module.

  Args:
      linen_module: The initialized Flax `linen.Module` instance.
      restored_linen_params: A dictionary (pytree) of the pre-trained parameters
        to be loaded into the NNX module.
      *init_args: Positional arguments required for the initial forward pass to
        materialize the NNX module structure (e.g., dummy input tensors).
      **init_kwargs: Keyword arguments for the initial forward pass.

  Returns:
      A complete `nnx.Module` with the pre-trained weights merged into its
      state.
  """
  # Convert the restored Linen parameter tree to NNX State format.
  params_nnx_state = nnx.State(
      jax.tree.map(
          lambda x: nnx.Param(value=x),
          restored_linen_params,
      )
  )
  # `lazy_init` performs a forward pass to determine the NNX parameter
  # structure.
  nnx_model_struct = bridge.lazy_init(
      bridge.ToNNX(linen_module, rngs=nnx.Rngs(0)), *init_args, **init_kwargs
  )

  # Split the NNX module to separate its structure from its parameters.
  graphdef, nnx_params_struct, other = nnx.split(
      nnx_model_struct, nnx.Param, ...
  )

  # Sanity check parameter structures.
  trees_match = jax.tree.all(
      jax.tree.map(
          lambda x, y: x.shape == y.shape,
          params_nnx_state,
          nnx_params_struct,
      )
  )
  if not trees_match:
    raise ValueError("Tree structure of restored and NNX parameters must match")

  # Merge the GraphDef with the restored parameters to create the final model.
  final_model = nnx.merge(graphdef, params_nnx_state, other)

  return final_model
