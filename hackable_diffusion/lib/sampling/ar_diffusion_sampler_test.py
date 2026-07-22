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

"""Tests for ar_diffusion_sampler."""

import dataclasses
from typing import Any

from hackable_diffusion.lib import hd_api
from hackable_diffusion.lib.sampling import ar_diffusion_sampler
import jax
import jax.numpy as jnp
import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

################################################################################
# MARK: Mock objects
################################################################################

_BATCH_SIZE = 2
_CANVAS_LENGTH = 4
_VOCAB_SIZE = 10


def _make_state(
    batch_size: int = _BATCH_SIZE,
    canvas_length: int = _CANVAS_LENGTH,
    max_num_canvases: int = 3,
    done: bool = False,
) -> dict[str, Any]:
  """Creates a minimal SamplerState for testing."""
  total_len = max_num_canvases * canvas_length
  return {
      'done': jnp.full((batch_size,), done, dtype=jnp.bool_),
      'predicted_tokens': jnp.zeros((batch_size, total_len), dtype=jnp.int32),
      'step': jnp.int32(0),
  }


class MockStateHandler:
  """A mock ARStateHandler for testing with jax.lax.while_loop.

  stop_after_n_canvases: If set, marks all batch elements as done after
    this many canvases (based on sampler_state['step']).
  """

  def __init__(
      self,
      max_num_canvases: int = 3,
      stop_after_n_canvases: int | None = None,
  ):
    self._max_num_canvases = max_num_canvases
    self._stop_after_n_canvases = stop_after_n_canvases

  def init_ar_state(
      self, batch_size, conditioning, canvas_length, max_num_canvases
  ):
    del conditioning  # Unused.
    return _make_state(
        batch_size=batch_size,
        canvas_length=canvas_length,
        max_num_canvases=max_num_canvases,
    )

  def update_ar_state(self, canvas_last_step, sampler_state):
    # Squeeze trailing dimension (hackable_diffusion uses <B, L, 1>).
    canvas = canvas_last_step.xt[..., 0]

    step = sampler_state['step']
    canvas_length = canvas.shape[1]
    indices = jnp.arange(canvas_length) + step
    sampler_state['predicted_tokens'] = (
        sampler_state['predicted_tokens'].at[:, indices].set(canvas)
    )
    sampler_state['step'] = step + canvas_length

    if self._stop_after_n_canvases is not None:
      num_canvases_done = (step + canvas_length) // canvas_length
      sampler_state['done'] = jnp.where(
          num_canvases_done >= self._stop_after_n_canvases,
          jnp.ones_like(sampler_state['done']),
          sampler_state['done'],
      )

    return sampler_state

  def finalize_ar_state(self, sampler_state):
    return sampler_state['predicted_tokens']

  def create_conditioning_from_state(self, sampler_state):
    return sampler_state


class InitArStateInterceptStateHandler(MockStateHandler):
  """Records init_ar_state args for verification."""

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.received_args = {}

  def init_ar_state(
      self, batch_size, conditioning, canvas_length, max_num_canvases
  ):
    self.received_args['batch_size'] = batch_size
    self.received_args['conditioning'] = conditioning
    self.received_args['canvas_length'] = canvas_length
    self.received_args['max_num_canvases'] = max_num_canvases
    return _make_state(
        batch_size=batch_size,
        canvas_length=canvas_length,
        max_num_canvases=max_num_canvases,
        done=True,  # Stop immediately.
    )


class MockDiffusionProcess:
  """Returns a fixed canvas instead of sampling from the invariant."""

  def __init__(self, fill_value: int = 1):
    self._fill_value = fill_value

  def sample_from_invariant(self, key, data_spec):
    return jnp.full_like(data_spec, self._fill_value)


@dataclasses.dataclass(kw_only=True, frozen=True)
class MockCanvasSampler:
  """Returns the initial_noise as-is, wrapped in a DiffusionStep."""

  time_schedule: None = None
  stepper: None = None
  num_steps: int = 1
  store_trajectory: bool = False
  update_conditioning_fn: None = None

  def __call__(self, inference_fn, rng, initial_noise, conditioning=None):
    step_info = hd_api.StepInfo(
        step=jnp.int32(0),
        time=jnp.float32(0.0),
        rng=rng,
    )
    last_step = hd_api.DiffusionStep(
        xt=initial_noise,
        step_info=step_info,
        aux=None,
    )
    return last_step, None


def _mock_inference_fn(xt, time, conditioning=None):
  """No-op inference function."""
  del time, conditioning  # Unused.
  return xt


def _make_sampler(**overrides):
  """Helper to construct a sampler with fake dependencies."""
  defaults = dict(
      canvas_sampler=MockCanvasSampler(),
      diffusion_process=MockDiffusionProcess(fill_value=7),
      max_num_canvases=3,
      canvas_length=_CANVAS_LENGTH,
      state_handler=MockStateHandler(),
      data_dtype=jnp.int32,
      data_shape=(1,),
  )
  defaults.update(overrides)
  return ar_diffusion_sampler.AutoregressiveDiffusionSampler(**defaults)


def _sample(sampler, **kwargs):
  """Call sampler and return only the output tokens (discard trajectory)."""
  defaults = dict(
      diffusion_inference_fn=_mock_inference_fn,
      batch_size=_BATCH_SIZE,
      rng=jax.random.PRNGKey(0),
      conditioning={'prompts': ['hello'] * _BATCH_SIZE},
  )
  defaults.update(kwargs)
  output, _ = sampler(**defaults)
  return output


################################################################################
# MARK: Tests
################################################################################


class AutoregressiveDiffusionSamplerTest(parameterized.TestCase):

  def test_runs_all_canvases_when_no_early_stopping(self):
    """Without early stopping, the loop runs max_num_canvases times."""
    sampler = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=7),
        state_handler=MockStateHandler(),
        max_num_canvases=3,
        canvas_length=_CANVAS_LENGTH,
    )
    tokens = _sample(sampler)
    # All 3*4=12 positions should be filled with 7.
    expected = jnp.full((_BATCH_SIZE, 3 * _CANVAS_LENGTH), 7, dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  def test_early_stopping_breaks_loop(self):
    """while_loop terminates when update_ar_state sets done after 2 canvases."""
    handler = MockStateHandler(stop_after_n_canvases=2)
    sampler = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=7),
        state_handler=handler,
        max_num_canvases=5,
        canvas_length=_CANVAS_LENGTH,
    )
    tokens = _sample(sampler)
    # First 2*4=8 positions filled with 7, remaining 3*4=12 positions are 0.
    expected = jnp.zeros((_BATCH_SIZE, 5 * _CANVAS_LENGTH), dtype=jnp.int32)
    expected = expected.at[:, : 2 * _CANVAS_LENGTH].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_done_after_one_canvas(self):
    """The while_loop stops after 1 canvas when done is set immediately."""
    handler = MockStateHandler(stop_after_n_canvases=1)
    sampler = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=7),
        state_handler=handler,
        max_num_canvases=5,
        canvas_length=_CANVAS_LENGTH,
    )
    tokens = _sample(sampler)
    # First 1*4=4 positions filled with 7, rest are 0.
    total_len = 5 * _CANVAS_LENGTH
    expected = jnp.zeros((_BATCH_SIZE, total_len), dtype=jnp.int32)
    expected = expected.at[:, :_CANVAS_LENGTH].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_output_shape(self):
    """Output has shape [batch_size, max_num_canvases * canvas_length]."""
    sampler = _make_sampler(max_num_canvases=3, canvas_length=4)
    tokens = _sample(sampler)
    self.assertEqual(tokens.shape, (_BATCH_SIZE, 3 * 4))

  def test_canvas_tokens_written_to_output(self):
    """Canvas tokens are correctly written into the predicted_tokens buffer."""
    sampler = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=7),
        max_num_canvases=2,
        canvas_length=3,
    )
    tokens = _sample(sampler)
    # All positions should be filled with 7.
    expected = jnp.full((_BATCH_SIZE, 6), 7, dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  def test_init_ar_state_receives_correct_args(self):
    """Verify init_ar_state is called with the sampler's config."""
    handler = InitArStateInterceptStateHandler()

    sampler = _make_sampler(
        state_handler=handler,
        max_num_canvases=4,
        canvas_length=8,
    )
    _ = _sample(sampler)
    expected_args = {
        'batch_size': _BATCH_SIZE,
        'canvas_length': 8,
        'max_num_canvases': 4,
        'conditioning': {'prompts': ['hello'] * _BATCH_SIZE},
    }
    self.assertEqual(handler.received_args, expected_args)

  def test_different_fill_values_produce_different_output(self):
    """Different diffusion processes produce different output tokens."""
    sampler_a = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=3),
        max_num_canvases=1,
    )
    sampler_b = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=9),
        max_num_canvases=1,
    )
    tokens_a = _sample(sampler_a)
    tokens_b = _sample(sampler_b)
    self.assertFalse(jnp.array_equal(tokens_a, tokens_b))

  def test_already_done_produces_zeros(self):
    """When init returns done=True, no canvases are generated."""
    handler = InitArStateInterceptStateHandler()  # init returns done=True
    sampler = _make_sampler(
        diffusion_process=MockDiffusionProcess(fill_value=7),
        state_handler=handler,
        max_num_canvases=3,
    )
    tokens = _sample(sampler)
    # No canvases generated, so all zeros.
    expected = jnp.zeros((_BATCH_SIZE, 3 * _CANVAS_LENGTH), dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)


class PerElementDoneHandler:
  """Sets done independently per batch element at different canvas counts.

  done_at_canvas[i] is the canvas index (1-based) at which element i becomes
  done.  For example, done_at_canvas=[1, 3] means element 0 is done after
  canvas 1 and element 1 is done after canvas 3.  The while_loop (using
  jnp.all) should continue until *all* elements are done.
  """

  def __init__(
      self,
      done_at_canvas: list[int],
      max_num_canvases: int = 5,
  ):
    self._done_at_canvas = jnp.array(done_at_canvas, dtype=jnp.int32)
    self._max_num_canvases = max_num_canvases

  def init_ar_state(
      self, batch_size, conditioning, canvas_length, max_num_canvases
  ):
    del conditioning  # Unused.
    total_len = max_num_canvases * canvas_length
    state = {
        'done': jnp.zeros((batch_size,), dtype=jnp.bool_),
        'predicted_tokens': jnp.zeros((batch_size, total_len), dtype=jnp.int32),
        'step': jnp.int32(0),
    }
    return state

  def update_ar_state(self, canvas_last_step, sampler_state):
    canvas = canvas_last_step.xt[..., 0]
    step = sampler_state['step']
    canvas_length = canvas.shape[1]
    indices = jnp.arange(canvas_length) + step
    sampler_state['predicted_tokens'] = (
        sampler_state['predicted_tokens'].at[:, indices].set(canvas)
    )
    sampler_state['step'] = step + canvas_length

    num_canvases_done = (step + canvas_length) // canvas_length
    # Each element becomes done independently.
    sampler_state['done'] = sampler_state['done'] | (
        num_canvases_done >= self._done_at_canvas
    )
    return sampler_state

  def finalize_ar_state(self, sampler_state):
    return sampler_state['predicted_tokens']

  def create_conditioning_from_state(self, sampler_state):
    return sampler_state


class EarlyStoppingTest(parameterized.TestCase):
  """Tests for per-batch-element early stopping via jnp.all(state['done'])."""

  def test_done_fn_returns_true_when_all_done(self):
    """DoneEarlyStoppingFn returns True when all elements are done."""
    fn = ar_diffusion_sampler.DoneEarlyStoppingFn()
    state = {'done': jnp.array([True, True, True])}
    self.assertTrue(fn(state))

  def test_done_fn_returns_false_when_any_not_done(self):
    """DoneEarlyStoppingFn returns False when any element is not done."""
    fn = ar_diffusion_sampler.DoneEarlyStoppingFn()
    state = {'done': jnp.array([True, False, True])}
    self.assertFalse(fn(state))

  def test_done_fn_returns_false_when_none_done(self):
    """DoneEarlyStoppingFn returns False when no elements are done."""
    fn = ar_diffusion_sampler.DoneEarlyStoppingFn()
    state = {'done': jnp.array([False, False])}
    self.assertFalse(fn(state))

  def test_done_fn_scalar_done(self):
    """DoneEarlyStoppingFn works with a scalar done (batch_size=1)."""
    fn = ar_diffusion_sampler.DoneEarlyStoppingFn()
    self.assertTrue(fn({'done': jnp.array([True])}))
    self.assertFalse(fn({'done': jnp.array([False])}))

  def test_done_fn_raises_when_done_missing(self):
    """DoneEarlyStoppingFn raises ValueError when 'done' key is missing."""
    fn = ar_diffusion_sampler.DoneEarlyStoppingFn()
    with self.assertRaisesRegex(
        ValueError,
        r'DoneEarlyStoppingFn requires sampler_state\["done"\] to be set.',
    ):
      fn({'step': jnp.int32(0)})

  def test_loop_continues_until_all_elements_done(self):
    """Loop continues as long as any batch element has done=False.

    Element 0 is done after canvas 1, element 1 after canvas 3.
    The loop should run 3 canvases total (waiting for element 1).
    """
    handler = PerElementDoneHandler(done_at_canvas=[1, 3], max_num_canvases=5)
    sampler = _make_sampler(state_handler=handler, max_num_canvases=5)
    tokens = _sample(sampler, rng=jax.random.PRNGKey(42))
    total_len = 5 * _CANVAS_LENGTH
    # Both elements should have 3 canvases filled with 7, rest zeros.
    filled = 3 * _CANVAS_LENGTH
    expected = jnp.zeros((_BATCH_SIZE, total_len), dtype=jnp.int32)
    expected = expected.at[:, :filled].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_partial_done_does_not_stop_loop(self):
    """If only element 0 is done but element 1 is not, loop continues.

    Element 0 done at canvas 1, element 1 done at canvas 4.
    The loop must run 4 canvases despite element 0 being done early.
    """
    handler = PerElementDoneHandler(done_at_canvas=[1, 4], max_num_canvases=5)
    sampler = _make_sampler(state_handler=handler, max_num_canvases=5)
    tokens = _sample(sampler)
    # 4 canvases worth of tokens should be filled.
    filled = 4 * _CANVAS_LENGTH
    total_len = 5 * _CANVAS_LENGTH
    expected = jnp.zeros((_BATCH_SIZE, total_len), dtype=jnp.int32)
    expected = expected.at[:, :filled].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_all_elements_done_simultaneously(self):
    """When all elements finish at the same canvas, loop stops there."""
    handler = PerElementDoneHandler(done_at_canvas=[2, 2], max_num_canvases=5)
    sampler = _make_sampler(state_handler=handler, max_num_canvases=5)
    tokens = _sample(sampler)
    filled = 2 * _CANVAS_LENGTH
    total_len = 5 * _CANVAS_LENGTH
    expected = jnp.zeros((2, total_len), dtype=jnp.int32)
    expected = expected.at[:, :filled].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_early_stop_respects_max_canvases_budget(self):
    """Even if no element is done, the loop stops at max_num_canvases."""
    handler = PerElementDoneHandler(done_at_canvas=[10, 10], max_num_canvases=3)
    sampler = _make_sampler(state_handler=handler, max_num_canvases=3)
    tokens = _sample(sampler)
    # All 3 canvases should be filled (max budget reached before done).
    expected = jnp.full((2, 3 * _CANVAS_LENGTH), 7, dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  def test_per_element_done_at_canvas_zero_runs_one_canvas(self):
    """done_at_canvas=0: update_ar_state sets done after the first canvas.

    Since done is initialized to False, the loop body runs once before
    done is set to True.  One canvas worth of tokens should be filled.
    """
    handler = PerElementDoneHandler(done_at_canvas=[0, 0], max_num_canvases=5)
    sampler = _make_sampler(state_handler=handler, max_num_canvases=5)
    tokens = _sample(sampler)
    total_len = 5 * _CANVAS_LENGTH
    expected = jnp.zeros((2, total_len), dtype=jnp.int32)
    expected = expected.at[:, :_CANVAS_LENGTH].set(7)
    np.testing.assert_array_equal(tokens, expected)

  def test_init_done_true_stops_before_any_canvas(self):
    """When init_ar_state sets done=True for all elements, loop never runs."""
    handler = (
        InitArStateInterceptStateHandler()
    )  # init returns done=True for all elements
    sampler = _make_sampler(
        state_handler=handler,
        max_num_canvases=5,
    )
    tokens = _sample(
        sampler,
        conditioning={'prompts': ['hello'] * _BATCH_SIZE},
    )
    expected = jnp.zeros((_BATCH_SIZE, 5 * _CANVAS_LENGTH), dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  def test_custom_early_stopping_fn_overrides_default(self):
    """A custom early_stopping_fn is used instead of DoneEarlyStoppingFn."""

    def always_stop(sampler_state):
      """Always request stop, regardless of state['done']."""
      del sampler_state
      return jnp.bool_(True)

    handler = MockStateHandler()
    sampler = _make_sampler(
        state_handler=handler,
        max_num_canvases=5,
        early_stopping_fn=always_stop,
    )
    tokens = _sample(
        sampler,
        conditioning={'prompts': ['hello'] * _BATCH_SIZE},
    )
    # always_stop returns True before the first iteration, so zero tokens.
    expected = jnp.zeros((_BATCH_SIZE, 5 * _CANVAS_LENGTH), dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  def test_custom_fn_never_stop_runs_full_budget(self):
    """A custom early_stopping_fn that never stops runs all canvases."""

    def never_stop(sampler_state):
      del sampler_state
      return jnp.bool_(False)

    sampler = _make_sampler(
        max_num_canvases=3,
        early_stopping_fn=never_stop,
    )
    tokens = _sample(
        sampler,
        conditioning={'prompts': ['hello'] * _BATCH_SIZE},
    )
    expected = jnp.full((_BATCH_SIZE, 3 * _CANVAS_LENGTH), 7, dtype=jnp.int32)
    np.testing.assert_array_equal(tokens, expected)

  @parameterized.parameters(
      (1,),
      (3,),
      (5,),
  )
  def test_single_batch_element_done_at_various_canvases(self, stop_at):
    """Single-element batch: done after N canvases."""
    handler = PerElementDoneHandler(
        done_at_canvas=[stop_at], max_num_canvases=5
    )
    sampler = _make_sampler(state_handler=handler, max_num_canvases=5)
    tokens = _sample(sampler, batch_size=1)
    filled = stop_at * _CANVAS_LENGTH
    total_len = 5 * _CANVAS_LENGTH
    np.testing.assert_array_equal(
        tokens[0, :filled],
        jnp.full((filled,), 7, dtype=jnp.int32),
    )
    np.testing.assert_array_equal(
        tokens[0, filled:],
        jnp.zeros((total_len - filled,), dtype=jnp.int32),
    )


if __name__ == '__main__':
  absltest.main()
