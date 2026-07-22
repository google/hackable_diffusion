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

"""Tests for Riemannian Flow Matching sampler step."""

from absl.testing import absltest
from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib import manifolds
from hackable_diffusion.lib.corruption import riemannian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.sampling import riemannian_sampling
import jax
import jax.numpy as jnp
import numpy as np


def _make_sampler(manifold):
  process = riemannian.RiemannianProcess(
      manifold=manifold,
      schedule=schedules.LinearRiemannianSchedule(),
  )
  return riemannian_sampling.RiemannianFlowSamplerStep(
      corruption_process=process,
  )


class RiemannianFlowSamplerStepTest(absltest.TestCase):

  def test_update_sphere(self):
    """Euler step on S2 moves along geodesic."""
    manifold = manifolds.Sphere()
    sampler = _make_sampler(manifold)
    key = jax.random.PRNGKey(0)

    xt = jnp.array([[1.0, 0.0, 0.0]])
    v = jnp.array([[0.0, 1.0, 0.0]])  # Tangent vector

    current_step = hd_api.DiffusionStep(
        xt=xt,
        step_info=hd_api.StepInfo(step=0, time=jnp.array([0.0]), rng=key),
        aux={},
    )
    next_step_info = hd_api.StepInfo(step=1, time=jnp.array([0.1]), rng=key)

    prediction = {"velocity": v}
    next_step = sampler.update(prediction, current_step, next_step_info)

    # dt = 0.1, so next_xt = exp_xt(0.1 * v) = [cos(0.1), sin(0.1), 0].
    expected_xt = jnp.array([[jnp.cos(0.1), jnp.sin(0.1), 0.0]])
    np.testing.assert_allclose(next_step.xt, expected_xt, atol=1e-5)
    # Result stays on the sphere.
    np.testing.assert_allclose(jnp.linalg.norm(next_step.xt), 1.0, atol=1e-5)

  def test_update_so3(self):
    """Euler step on SO(3) produces a valid rotation matrix."""
    manifold = manifolds.SO3()
    sampler = _make_sampler(manifold)
    key = jax.random.PRNGKey(1)

    xt = jnp.eye(3)[None, ...]  # Identity rotation (1, 3, 3).
    v = jnp.array([[[0.0, -0.1, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.0]]])

    current_step = hd_api.DiffusionStep(
        xt=xt,
        step_info=hd_api.StepInfo(step=0, time=jnp.array([0.0]), rng=key),
        aux={},
    )
    next_step_info = hd_api.StepInfo(step=1, time=jnp.array([0.1]), rng=key)

    prediction = {"velocity": v}
    next_step = sampler.update(prediction, current_step, next_step_info)

    # Check result is a valid rotation: R^T R = I and det(R) = 1.
    rtrt = jnp.matmul(jnp.swapaxes(next_step.xt, -2, -1), next_step.xt)
    np.testing.assert_allclose(rtrt, jnp.eye(3)[None, ...], atol=1e-5)
    np.testing.assert_allclose(jnp.linalg.det(next_step.xt), 1.0, atol=1e-5)

  def test_update_torus(self):
    """Euler step on Torus wraps around [0, 1)."""
    manifold = manifolds.Torus()
    sampler = _make_sampler(manifold)
    key = jax.random.PRNGKey(2)

    xt = jnp.array([[0.9, 0.1, 0.5]])
    v = jnp.array([[0.5, -0.5, 0.0]])

    current_step = hd_api.DiffusionStep(
        xt=xt,
        step_info=hd_api.StepInfo(step=0, time=jnp.array([0.0]), rng=key),
        aux={},
    )
    next_step_info = hd_api.StepInfo(step=1, time=jnp.array([1.0]), rng=key)

    prediction = {"velocity": v}
    next_step = sampler.update(prediction, current_step, next_step_info)

    # dt = 1.0, so next_xt = exp(xt, v) = (xt + v) % 1.0.
    expected_xt = jnp.array([[(0.9 + 0.5) % 1.0, (0.1 - 0.5) % 1.0, 0.5]])
    np.testing.assert_allclose(next_step.xt, expected_xt, atol=1e-5)
    self.assertTrue(jnp.all(next_step.xt >= 0.0))
    self.assertTrue(jnp.all(next_step.xt < 1.0))

  def test_initialize(self):
    """Initialize returns a DiffusionStep with the given noise and step info."""
    manifold = manifolds.Sphere()
    sampler = _make_sampler(manifold)
    key = jax.random.PRNGKey(0)

    initial_noise = manifold.random_uniform(key, (4, 3))
    initial_step_info = hd_api.StepInfo(step=0, time=jnp.array([0.0]), rng=key)

    step = sampler.initialize(initial_noise, initial_step_info)

    np.testing.assert_array_equal(step.xt, initial_noise)
    self.assertEqual(step.step_info.step, 0)

  def test_finalize(self):
    """Finalize returns the current step unchanged."""
    manifold = manifolds.Sphere()
    sampler = _make_sampler(manifold)
    key = jax.random.PRNGKey(0)

    xt = jnp.array([[1.0, 0.0, 0.0]])
    current_step = hd_api.DiffusionStep(
        xt=xt,
        step_info=hd_api.StepInfo(step=5, time=jnp.array([1.0]), rng=key),
        aux={},
    )
    last_step_info = hd_api.StepInfo(step=6, time=jnp.array([1.0]), rng=key)
    prediction = {"velocity": jnp.zeros_like(xt)}

    result = sampler.finalize(prediction, current_step, last_step_info)
    np.testing.assert_array_equal(result.xt, xt)


if __name__ == "__main__":
  absltest.main()
