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

"""Diffusion network."""

import abc
import dataclasses
from typing import Callable, Protocol
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import conditioning_encoder
from hackable_diffusion.lib.corruption import schedules
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
PRNGKey = hd_typing.PRNGKey
PyTree = hd_typing.PyTree

GaussianSchedule = schedules.GaussianSchedule

Conditioning = hd_typing.Conditioning
DataArray = hd_typing.DataArray
DataTree = hd_typing.DataTree
TargetInfo = hd_typing.TargetInfo
TargetInfoTree = hd_typing.TargetInfoTree
TimeArray = hd_typing.TimeArray
TimeTree = hd_typing.TimeTree

ConditioningShape = hd_typing.ConditioningShape
Shape = hd_typing.Shape
ShapeTree = hd_typing.ShapeTree

################################################################################
# MARK: Rescalers
################################################################################


class InputRescaler(Protocol):
  """Rescales the input in a schedule-dependent way."""

  def __call__(self, time: TimeArray, inputs: DataArray) -> DataArray:
    ...


class TimeRescaler(Protocol):
  """Rescales the time, optionally in a schedule-dependent way."""

  def __call__(self, time: TimeArray) -> TimeArray:
    ...


################################################################################
# MARK: Diffusion Network
################################################################################


class BaseDiffusionNetwork(nn.Module, abc.ABC):
  """Base diffusion network."""

  @abc.abstractmethod
  def __call__(
      self,
      time: TimeTree,
      xt: DataTree,
      conditioning: Conditioning | None,
      is_training: bool,
  ) -> TargetInfoTree:
    ...


class DiffusionNetwork(BaseDiffusionNetwork):
  """Diffusion network.

  This class is responsible for orchestrating the different components of the
  model (backbone and conditioning encoders in the case of diffusion models for
  instance). It wraps those modules in order to create a consistent interface
  for the model. The output of the __call__ method is a dictionary of model
  outputs. The keys of the dictionary are specified by the prediction function,
  for instance ['x0', 'epsilon', 'score', 'velocity', 'v'] in the case of a
  Gaussian diffusion model.

  The processing is done as follows. First, it optionally rescales the time and
  the input using the `time_rescaler` and `input_rescaler`, which are
  schedule-dependent. Then, it encodes the conditioning information and the
  rescaled time using the `conditioning_encoder`. After that, it passed the
  input and the processed conditioning embeddings to the `backbone_network`.

  Attributes:
    backbone_network: The backbone network to use for the diffusion model.
    conditioning_encoder: The conditioning encoder to use for the diffusion
      model.
    predition_type: the type of prediction used by the diffusion model. For
      example, in the Gaussian diffusion model, the prediction type can be 'x0',
      'epsilon', 'score', 'velocity', or 'v'.
    input_rescaler: The input rescaler to use for the diffusion model,
      optionally schedule-dependent. By default, we do not use rescaler.
    time_rescaler: The time rescaler to use for the diffusion model, optionally
      schedule-dependent. By default, we do not use rescaler.
  """

  backbone_network: arch_typing.ConditionalBackbone
  conditioning_encoder: conditioning_encoder.BaseConditioningEncoder
  prediction_type: str
  data_dtype: DType = jnp.float32
  input_rescaler: InputRescaler | None = None
  time_rescaler: TimeRescaler | None = None

  def initialize_variables(
      self,
      input_shape: Shape,
      conditioning_shape: ConditioningShape,
      key: PRNGKey,
      is_training: bool = False,
  ) -> PyTree:
    """Initializes the variables of the model from shapes."""
    dummy_xt = utils.get_dummy_batch_fixed_dtype(
        input_shape, dtype=self.data_dtype
    )
    dummy_conditioning = utils.get_dummy_batch_fixed_dtype(
        conditioning_shape, dtype=jnp.float32
    )
    dummy_time = utils.get_dummy_batch_fixed_dtype(
        input_shape, only_first_axis=True, dtype=jnp.float32
    )

    params_key, dropout_key = jax.random.split(key)
    return self.init(
        {'params': params_key, 'dropout': dropout_key},
        time=dummy_time,
        xt=dummy_xt,
        conditioning=dummy_conditioning,
        is_training=is_training,
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      time: TimeArray,
      xt: DataArray,
      conditioning: Conditioning | None,
      is_training: bool,
  ) -> TargetInfo:

    # Rescale time and input.

    time_rescaled = (
        self.time_rescaler(time) if self.time_rescaler is not None else time
    )

    xt_rescaled = (
        self.input_rescaler(time, xt) if self.input_rescaler is not None else xt
    )

    # Encode conditioning.

    conditioning_embeddings = self.conditioning_encoder.copy(
        name='ConditioningEncoder'
    )(
        time=time_rescaled,
        conditioning=conditioning,
        is_training=is_training,
    )
    # Run backbone.
    backbone_outputs = self.backbone_network.copy(name='Backbone')(
        x=xt_rescaled,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    return {self.prediction_type: backbone_outputs}


################################################################################
# MARK: Multi-modal Diffusion Network
################################################################################


class MultiModalDiffusionNetwork(BaseDiffusionNetwork):
  """Multi-modal diffusion network.

  This DiffusioNetwork is a generalization of the DiffusionNetwork to
  multi-modal data. It is able to handle different data types (e.g., continuous
  and discrete), different prediction types (e.g., x0, logits), and different
  conditioning encoders (e.g., time embedder, token embedder, etc.).

  The main assumption is that the PyTree structures of `prediction_type`,
  `data_dtype`, `input_rescaler`, and `time_rescaler` are the same as `xt` and
  `time`.

  Example usage:

    backbone = ...

    ConditioningMechanism = arch_typing.ConditioningMechanism

    conditioning_embedders = {
        'label': conditioning_encoder.LabelEmbedder(
            num_classes=10,
            num_features=256,
            conditioning_key='label',
        )
    }

    time_embedder_continuous = conditioning_encoder.SinusoidalTimeEmbedder(
            activation='gelu', embedding_dim=256, num_features=256
        )
    time_embedder_discrete = conditioning_encoder.SinusoidalTimeEmbedder(
            activation='gelu', embedding_dim=256, num_features=256
        )
    time_embedders = {
        'data_continuous': time_embedder_continuous,
        'data_discrete': time_embedder_discrete,
    }
    time_embedder = conditioning_encoder.NestedTimeEmbedder(time_embedders=)

    encoder = conditioning_encoder.ConditioningEncoder(
        time_embedder=time_embedder,
        conditioning_embedders=conditioning_embedders,
        embedding_merging_method=arch_typing.EmbeddingMergeMethod.SUM,
        conditioning_rules={
            'time': ConditioningMechanism.ADAPTIVE_NORM,
            'label': ConditioningMechanism.ADAPTIVE_NORM,
        },
    )

    network = hd.diffusion_network.MultiModalDiffusionNetwork(
        backbone_network=backbone,
        conditioning_encoder=encoder,
        prediction_type={'data_continuous': 'x0', 'data_discrete': 'logits'},
        data_dtype={'data_continuous': jnp.float32, 'data_discrete': jnp.int32},
        input_rescaler=None,
        time_rescaler=None,
        )

  Attributes:
    backbone_network: The backbone network to use for the diffusion model.
    conditioning_encoder: The conditioning encoder to use for the diffusion
      model.
    prediction_type: the type of prediction used by the diffusion model. For
      example, in the Gaussian diffusion model, the prediction type can be 'x0',
      'epsilon', 'score', 'velocity', or 'v'.
    data_dtype: the dtype of the data.
    input_rescaler: The input rescaler to use for the diffusion model,
      optionally schedule-dependent. By default, we do not use rescaler.
    time_rescaler: The time rescaler to use for the diffusion model, optionally
      schedule-dependent. By default, we do not use rescaler.
  """

  backbone_network: arch_typing.ConditionalBackbone
  conditioning_encoder: conditioning_encoder.BaseConditioningEncoder
  prediction_type: PyTree[str]
  data_dtype: PyTree[DType]
  input_rescaler: PyTree[InputRescaler | None] | None = None
  time_rescaler: PyTree[TimeRescaler | None] | None = None

  def initialize_variables(
      self,
      input_shape: ShapeTree,
      conditioning_shape: ConditioningShape,
      key: PRNGKey,
      is_training: bool = False,
  ) -> PyTree:
    dummy_xt = utils.get_dummy_batch(input_shape, dtype=self.data_dtype)
    dummy_conditioning = utils.get_dummy_batch_fixed_dtype(
        conditioning_shape, dtype=jnp.float32
    )
    dummy_time = utils.get_dummy_batch_fixed_dtype(
        input_shape, only_first_axis=True, dtype=jnp.float32
    )

    params_key, dropout_key = jax.random.split(key)
    return self.init(
        {'params': params_key, 'dropout': dropout_key},
        time=dummy_time,
        xt=dummy_xt,
        conditioning=dummy_conditioning,
        is_training=is_training,
    )

  @nn.compact
  @typechecked
  def __call__(
      self,
      xt: DataTree,
      time: TimeTree,
      conditioning: Conditioning | None,
      is_training: bool,
  ):
    # Rescale time and input.

    if self.time_rescaler is not None:
      # lenient alternative to jax.tree.map
      time_rescaled = utils.lenient_map(
          lambda time, time_rescaler: time_rescaler(time)
          if time_rescaler is not None
          else time,
          time,
          self.time_rescaler,
      )
    else:
      time_rescaled = time

    if self.input_rescaler is not None:
      # lenient alternative to jax.tree.map
      xt_rescaled = utils.lenient_map(
          lambda time, xt, input_rescaler: input_rescaler(time, xt)
          if input_rescaler is not None
          else xt,
          time,
          xt,
          self.input_rescaler,
      )
    else:
      xt_rescaled = xt

    # Encode conditioning.

    conditioning_embeddings = self.conditioning_encoder.copy(
        name='ConditioningEncoder'
    )(
        time=time_rescaled,
        conditioning=conditioning,
        is_training=is_training,
    )

    # Run backbone.
    backbone_outputs = self.backbone_network.copy(name='Backbone')(
        x=xt_rescaled,
        conditioning_embeddings=conditioning_embeddings,
        is_training=is_training,
    )

    # lenient alternative to jax.tree.map
    outputs = utils.lenient_map(
        lambda backbone_output, prediction_type: {
            prediction_type: backbone_output
        },
        backbone_outputs,
        self.prediction_type,
    )
    return outputs


################################################################################
# MARK: Time rescaling functions
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class LogSnrTimeRescaler(TimeRescaler):
  """Time rescaler that uses the logsnr of the process."""

  schedule: GaussianSchedule
  postprocess_fn: Callable[[TimeArray], TimeArray] | None = None

  @typechecked
  def __call__(self, time: TimeArray) -> TimeArray:
    """Returns the time rescaled by the logsnr of the process."""
    if self.postprocess_fn is None:
      postprocess_fn = lambda x: x
    else:
      postprocess_fn = self.postprocess_fn
    return postprocess_fn(self.schedule.logsnr(time))


################################################################################
# MARK: Input rescaling functions
################################################################################


@dataclasses.dataclass(kw_only=True, frozen=True)
class MagnitudeScheduleInputRescaler(InputRescaler):
  """Input rescaler that uses the magnitude of the schedule."""

  schedule: GaussianSchedule

  @typechecked
  def __call__(self, time: TimeArray, inputs: DataArray) -> DataArray:
    """Returns the inputs rescaled by the magnitude of the schedule."""
    alpha_t = self.schedule.alpha(time)
    sigma_t = self.schedule.sigma(time)
    alpha_t = utils.bcast_right(alpha_t, inputs.ndim)
    sigma_t = utils.bcast_right(sigma_t, inputs.ndim)
    magnitude = jnp.sqrt(jnp.square(alpha_t) + jnp.square(sigma_t))
    return inputs / magnitude
