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

"""Inference class for diffusion models."""

import dataclasses

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib.inference import guidance
from hackable_diffusion.lib.inference import projection
import kauldron.ktyping as kt

################################################################################
# MARK: Type Aliases
################################################################################

Conditioning = hd_typing.Conditioning
DataTree = hd_typing.DataTree
TimeTree = hd_typing.TimeTree
TargetInfoTree = hd_typing.TargetInfoTree

InferenceFn = hd_api.InferenceFn

################################################################################
# MARK: IdentityInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class IdentityInferenceFn(InferenceFn):
  """Identity inference function that predicts x0 = xt."""

  process: hd_api.CorruptionProcess

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""
    del conditioning  # unused
    return self.process.convert_predictions(
        prediction={"x0": xt},
        xt=xt,
        time=time,
    )


################################################################################
# MARK: GuidedDiffusionInferenceFn
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class GuidedDiffusionInferenceFn(InferenceFn):
  """Inference function protocol."""

  base_inference_fn: InferenceFn
  guidance_fn: guidance.GuidanceFn = guidance.ScalarGuidanceFn()
  projection_fn: projection.ProjectionFn = projection.IdentityProjectionFn()

  @kt.typechecked
  def __call__(
      self,
      time: TimeTree,  # pyrefly: ignore[not-a-type]
      xt: DataTree,  # pyrefly: ignore[not-a-type]
      conditioning: Conditioning | None,
  ) -> TargetInfoTree:  # pyrefly: ignore[not-a-type]
    """Returns the model outputs."""

    cond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=conditioning,
    )
    uncond_outputs = self.base_inference_fn(
        time=time,
        xt=xt,
        conditioning=None,
    )

    guided_outputs = self.guidance_fn(
        xt=xt,
        conditioning=conditioning,  # pyrefly: ignore[bad-argument-type]
        time=time,
        cond_outputs=cond_outputs,
        uncond_outputs=uncond_outputs,
    )

    projected_outputs = self.projection_fn(
        xt=xt,
        conditioning=conditioning,  # pyrefly: ignore[bad-argument-type]
        time=time,
        outputs=guided_outputs,
    )
    return projected_outputs
