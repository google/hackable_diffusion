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

"""Tests for wrappers."""

from flax import nnx
import flax.linen as nn
from hackable_diffusion.lib.inference import wrappers
import jax
import jax.numpy as jnp
import mock
from absl.testing import absltest


class WrappersTest(absltest.TestCase):

  def test_flax_linen_inference_fn(self):

    time = jnp.ones((1, 2, 3, 1))
    xt = jnp.ones((1, 2, 3, 4))
    conditioning = {'label': jnp.ones((1, 2, 3, 4))}
    params = {'params': jnp.ones((1, 2, 3, 4))}
    target_info = {'x0': jnp.ones((1, 2, 3, 4))}

    network = mock.MagicMock()
    network.apply.return_value = target_info
    target_info_out = wrappers.FlaxLinenInferenceFn(
        network=network, params=params
    )(time=time, xt=xt, conditioning=conditioning)
    self.assertTrue(jnp.allclose(target_info_out['x0'], target_info['x0']))

  def test_flax_nnx_inference_fn(self):

    time = jnp.ones((1, 2, 3, 1))
    xt = jnp.ones((1, 2, 3, 4))
    conditioning = {'label': jnp.ones((1, 2, 3, 4))}
    target_info = {'x0': jnp.ones((1, 2, 3, 4))}

    nnx_network = mock.MagicMock(return_value=target_info)
    target_info_out = wrappers.FlaxNNXInferenceFn(nnx_network=nnx_network)(
        time=time, xt=xt, conditioning=conditioning
    )
    self.assertTrue(jnp.allclose(target_info_out['x0'], target_info['x0']))

  def test_flax_nnx_inference_fn_with_rngs(self):
    mlp_linen = nn.Dense(10)
    x = jnp.ones((1, 2))
    params = mlp_linen.init(rngs={'params': jax.random.PRNGKey(0)}, inputs=x)
    self.assertEqual(params['params']['kernel'].shape, (2, 10))
    self.assertEqual(params['params']['bias'].shape, (10,))

    outputs_linen = mlp_linen.apply(params, inputs=x)

    mlp_nnx_converted = wrappers.convert_flax_linen_module_with_params_to_nnx(
        mlp_linen, params['params'], inputs=x
    )
    _, nnx_params, _ = nnx.split(mlp_nnx_converted, nnx.Param, ...)
    self.assertTrue(jnp.allclose(nnx_params['bias'], params['params']['bias']))
    self.assertTrue(
        jnp.allclose(nnx_params['kernel'], params['params']['kernel'])
    )
    self.assertIsInstance(mlp_nnx_converted.to_nnx__module, nn.Dense)

    outputs_nnx = mlp_nnx_converted(inputs=x)
    self.assertTrue(jnp.allclose(outputs_nnx, outputs_linen))


if __name__ == '__main__':
  absltest.main()
