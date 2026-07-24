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

"""Tests for evals."""

from unittest import mock
import flax.linen as nn
from hackable_diffusion import hd
from hackable_diffusion.kdiff import evals
import jax
import jax.numpy as jnp
from kauldron import kd

from absl.testing import absltest


class DummyNetwork(nn.Module):

  @nn.compact
  def __call__(self, time, xt, conditioning, is_training=False):
    return {"v": jnp.zeros_like(xt)}


class DummyModel(nn.Module):
  network: nn.Module


class KDiffInferenceFnTest(absltest.TestCase):

  def test_guidance_fn_not_specified(self):
    dummy_net = DummyNetwork()
    model = DummyModel(network=dummy_net)
    context = kd.train.Context(step=0, batch={}, params={"network": {}})

    factory = evals.KDiffInferenceFn(network_path="network")
    inference_fn = factory.from_model_and_context(model, context)

    self.assertIsInstance(inference_fn, hd.inference.FlaxLinenInferenceFn)
    self.assertNotIsInstance(
        inference_fn, hd.inference.GuidedDiffusionInferenceFn
    )

  def test_guidance_fn_specified(self):
    dummy_net = DummyNetwork()
    model = DummyModel(network=dummy_net)
    context = kd.train.Context(step=0, batch={}, params={"network": {}})

    mock_guidance_fn = mock.Mock()
    factory = evals.KDiffInferenceFn(
        network_path="network",
        guidance_fn=mock_guidance_fn,
    )
    inference_fn = factory.from_model_and_context(model, context)

    self.assertIsInstance(
        inference_fn, hd.inference.GuidedDiffusionInferenceFn
    )
    self.assertEqual(inference_fn.guidance_fn, mock_guidance_fn)

  def test_guidance_fn_called_during_inference(self):
    dummy_net = DummyNetwork()
    model = DummyModel(network=dummy_net)
    context = kd.train.Context(step=0, batch={}, params={"network": {}})

    mock_guidance_fn = mock.Mock(return_value={"v": jnp.ones((2, 4))})
    factory = evals.KDiffInferenceFn(
        network_path="network",
        guidance_fn=mock_guidance_fn,
    )
    inference_fn = factory.from_model_and_context(model, context)

    xt = jnp.zeros((2, 4))
    time = jnp.array([0.5, 0.5])
    out = inference_fn(xt=xt, conditioning=None, time=time)

    mock_guidance_fn.assert_called_once()
    self.assertEqual(out["v"].shape, (2, 4))


if __name__ == "__main__":
  absltest.main()
