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

"""Tests for Riemannian Flow Matching architectures."""

from absl.testing import absltest
import flax
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import mlp
from hackable_diffusion.lib.architecture import riemannian
import jax
import jax.numpy as jnp


class RiemannianArchitectureTest(absltest.TestCase):

  def test_riemannian_backbone_projection(self):
    manifold = manifolds.Sphere()
    backbone = mlp.ConditionalMLP(
        hidden_sizes_preprocess=(16,),
        hidden_sizes_postprocess=(16,),
        activation='relu',
        zero_init_output=True,
        dropout_rate=0.0,
        conditioning_mechanism='concatenate',
    )
    model = riemannian.RiemannianConditionalBackbone(
        backbone=backbone,
        manifold=manifold,
    )

    key = jax.random.PRNGKey(0)
    xt = manifold.random_uniform(key, (4, 3))
    time_emb = jnp.array([[0.5], [0.5], [0.5], [0.5]])

    conditioning_embeddings = {
        'concatenate': time_emb,
    }

    variables = model.init(key, xt, conditioning_embeddings, is_training=False)
    v = model.apply(variables, xt, conditioning_embeddings, is_training=False)

    self.assertEqual(v.shape, (4, 3))  # pyrefly: ignore[missing-attribute]

    # Check that v is tangent to xt
    inner_products = jnp.sum(xt * v, axis=-1)
    # Project should ensure dot(xt, v) = 0 for sphere
    self.assertAlmostEqual(jnp.max(jnp.abs(inner_products)), 0.0, places=5)  # pyrefly: ignore[no-matching-overload]

  def test_variable_names_and_shapes(self):
    """Check that model variables have expected names and shapes."""
    manifold = manifolds.Sphere()
    backbone = mlp.ConditionalMLP(
        hidden_sizes_preprocess=(16,),
        hidden_sizes_postprocess=(16,),
        activation='relu',
        zero_init_output=True,
        dropout_rate=0.0,
        conditioning_mechanism='concatenate',
    )
    model = riemannian.RiemannianConditionalBackbone(
        backbone=backbone,
        manifold=manifold,
    )

    key = jax.random.PRNGKey(0)
    xt = manifold.random_uniform(key, (4, 3))
    time_emb = jnp.array([[0.5], [0.5], [0.5], [0.5]])
    conditioning_embeddings = {
        'concatenate': time_emb,
    }

    variables = model.init(key, xt, conditioning_embeddings, is_training=False)
    shapes = jax.tree_util.tree_map(lambda x: x.shape, variables)
    flat_shapes = flax.traverse_util.flatten_dict(shapes, sep='/')

    expected = {
        # PreprocessMLP: Dense_Output maps input dim 3 -> 16.
        'params/backbone/PreprocessMLP/Dense_Output/kernel': (3, 16),
        'params/backbone/PreprocessMLP/Dense_Output/bias': (16,),
        # PostprocessMLP: Dense_Hidden_0 maps 16 (preprocess) + 1 (time) -> 16.
        'params/backbone/PostprocessMLP/Dense_Hidden_0/kernel': (17, 16),
        'params/backbone/PostprocessMLP/Dense_Hidden_0/bias': (16,),
        # PostprocessMLP: Dense_Output maps 16 -> 3 (sphere dim).
        'params/backbone/PostprocessMLP/Dense_Output/kernel': (16, 3),
        'params/backbone/PostprocessMLP/Dense_Output/bias': (3,),
    }

    self.assertEqual(flat_shapes, expected)


if __name__ == '__main__':
  absltest.main()
