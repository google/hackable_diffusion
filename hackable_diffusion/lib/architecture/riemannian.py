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

"""Riemannian Flow Matching architectures."""

import flax.linen as nn
from hackable_diffusion.lib import diffusion_network
from hackable_diffusion.lib import manifolds

################################################################################
# MARK: Riemannian Conditional Backbone
################################################################################

ConditionalBackbone = diffusion_network.ConditionalBackbone


class RiemannianConditionalBackbone(nn.Module, ConditionalBackbone):
  """Velocity model for Riemannian Flow Matching.

  Projects the output of a backbone network to the tangent space of a manifold.
  """

  backbone: ConditionalBackbone
  manifold: manifolds.Manifold

  @nn.compact
  def __call__(self, x, conditioning_embeddings, is_training=True):

    v = self.backbone(x, conditioning_embeddings, is_training=is_training)

    # Project v to tangent space at xt.
    if isinstance(v, dict) and 'velocity' in v:
      v = v['velocity']

    v_proj = self.manifold.project(x, v)
    return v_proj
