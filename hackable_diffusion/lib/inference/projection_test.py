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

"""Tests for projection."""

import chex
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.corruption import gaussian
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.inference import projection
import jax.numpy as jnp

from absl.testing import absltest
from absl.testing import parameterized


################################################################################
# MARK: Tests
################################################################################


class ProjectionTest(parameterized.TestCase):
  """Tests for projection functions."""

  def setUp(self):
    super().setUp()
    self.batch_size = 2
    self.data_shape = (4, 4, 3)
    self.other_data_shape = (4, 8, 9)
    self.xt = jnp.ones((self.batch_size, *self.data_shape)) * 0.5
    self.time = utils.bcast_right(jnp.array([0.5, 0.5]), self.xt.ndim)
    self.conditioning = {}  # Not currently used by projections.
    self.process = gaussian.GaussianProcess(schedule=schedules.RFSchedule())
    # create some fake outputs for testing
    x0 = jnp.arange(3) + 1.0  # [1.0, 2.0, 3.0]
    x0 *= jnp.ones_like(self.xt)
    self.x0_outputs = {'x0': x0}
    self.v_outputs = {'v': jnp.ones_like(self.xt) * 1.5}

    self.nested_xt = {
        'data_continuous_1': jnp.ones((self.batch_size, *self.data_shape)),
        'modality': {
            'data_continuous_2': jnp.ones(
                (self.batch_size, *self.other_data_shape)
            )
        },
    }
    self.nested_time = {
        'data_continuous_1': utils.bcast_right(
            jnp.array([0.5, 0.5]), self.nested_xt['data_continuous_1'].ndim
        ),
        'modality': {
            'data_continuous_2': utils.bcast_right(
                jnp.array([0.5, 0.5]),
                self.nested_xt['modality']['data_continuous_2'].ndim,
            )
        },
    }
    self.nested_outputs = {
        'data_continuous_1': {
            'x0': jnp.ones_like(self.nested_xt['data_continuous_1'])
        },
        'modality': {
            'data_continuous_2': {
                'x0': jnp.ones_like(
                    self.nested_xt['modality']['data_continuous_2']
                )
            }
        },
    }

  # MARK: Helper functions tests

  @parameterized.named_parameters(
      ('x0_prediction', 'x0'),
      ('v_prediction', 'v'),
  )
  def test_from_x0_is_inverse_of_to_x0(self, prediction_type: str):
    """Tests that _from_x0 is the inverse of _to_x0."""
    if prediction_type == 'x0':
      original_preds = self.x0_outputs
    elif prediction_type == 'v':
      original_preds = self.v_outputs
    else:
      raise ValueError(f'Unsupported prediction type: {prediction_type}')

    original_pred_array = original_preds[prediction_type]

    # Convert to x0.
    returned_prediction_type, x0_preds = projection._to_x0(
        preds=original_preds,
        xt=self.xt,
        time=self.time,
        process=self.process,
    )

    self.assertEqual(returned_prediction_type, prediction_type)

    # Convert back from x0.
    converted_back_pred_array = projection._from_x0(
        preds=x0_preds,
        xt=self.xt,
        time=self.time,
        process=self.process,
        prediction_type=returned_prediction_type,
    )

    # Check if the result is close to the original prediction.
    self.assertTrue(
        jnp.allclose(
            converted_back_pred_array[prediction_type],
            original_pred_array,
            rtol=1e-5,
        )
    )

  # MARK: IdentityProjectionFn tests

  @parameterized.named_parameters(
      ('x0_prediction', 'x0'),
      ('v_prediction', 'v'),
  )
  def test_identity_projection_fn(self, output_type: str):
    """Tests that the IdentityProjectionFn returns the outputs unchanged."""
    if output_type == 'x0':
      outputs = self.x0_outputs
    elif output_type == 'v':
      outputs = self.v_outputs
    else:
      raise ValueError(f'Unsupported output type: {output_type}')
    proj_fn = projection.IdentityProjectionFn()
    result = proj_fn(self.xt, self.conditioning, self.time, outputs)
    self.assertIs(result, outputs)

  # MARK: StaticThresholdProjectionFn tests

  def test_static_threshold_projection_fn(self):
    """Tests that StaticThresholdProjectionFn clips the x0 prediction."""
    proj_fn = projection.StaticThresholdProjectionFn(process=self.process)
    result = proj_fn(self.xt, self.conditioning, self.time, self.x0_outputs)
    expected_x0 = jnp.ones_like(self.xt)
    self.assertTrue(jnp.allclose(result['x0'], expected_x0))

  # MARK: DynamicThresholdProjectionFn tests

  @parameterized.named_parameters(
      ('negative_percentile', -0.1),
      ('too_large_percentile', 101.0),
  )
  def test_dynamic_threshold_invalid_percentile_raises_error(
      self, percentile: float
  ):
    """Tests that an error is raised for an invalid percentile."""
    with self.assertRaisesRegex(
        ValueError, 'Percentile must be between 0.0 and 100.0'
    ):
      projection.DynamicThresholdProjectionFn(
          process=self.process, percentile=percentile
      )

  def test_dynamic_threshold_projection_fn(self):
    """Tests that DynamicThresholdProjectionFn correctly rescales the x0 pred."""
    percentile = 50.0
    proj_fn = projection.DynamicThresholdProjectionFn(
        process=self.process, percentile=percentile
    )
    expected_x0 = jnp.array([0.5, 1.0, 1.0])
    expected_x0 *= jnp.ones_like(self.xt)
    result = proj_fn(self.xt, self.conditioning, self.time, self.x0_outputs)
    self.assertTrue(jnp.allclose(result['x0'], expected_x0))

  def test_nested_projection_fn(self):
    """Tests that NestedProjectionFn correctly applies projections."""
    proj_fn = projection.NestedProjectionFn(
        projection_fns={
            'data_continuous_1': projection.IdentityProjectionFn(),
            'modality': {
                'data_continuous_2': projection.StaticThresholdProjectionFn(
                    process=self.process
                )
            },
        }
    )
    result = proj_fn(
        self.nested_xt, self.conditioning, self.nested_time, self.nested_outputs
    )
    self.assertIsInstance(result, dict)
    self.assertEqual(
        result['data_continuous_1']['x0'].shape,
        self.nested_xt['data_continuous_1'].shape,
    )
    self.assertEqual(
        result['modality']['data_continuous_2']['x0'].shape,
        self.nested_xt['modality']['data_continuous_2'].shape,
    )
    chex.assert_trees_all_equal_structs(result, self.nested_outputs)


if __name__ == '__main__':
  absltest.main()
