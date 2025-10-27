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

"""Conditioning and time encoders.

This module implements various encoders for time and conditioning signals,
as well as a module to process and combine them.

These modules do not cover all possible usecases, but rather provide a reference
 implementation for new encoders.
"""

import abc
from typing import Sequence
import flax.linen as nn
from hackable_diffusion.lib import hd_typing
from hackable_diffusion.lib import utils
from hackable_diffusion.lib.architecture import arch_typing
from hackable_diffusion.lib.architecture import mlp_blocks
from hackable_diffusion.lib.architecture import sequence_embedders
from hackable_diffusion.lib.hd_typing import typechecked  # pylint: disable=g-multiple-import,g-importing-member
import jax
import jax.numpy as jnp

################################################################################
# MARK: Type Aliases
################################################################################

DType = hd_typing.DType
Float = hd_typing.Float
PyTree = hd_typing.PyTree

ConditioningMechanism = arch_typing.ConditioningMechanism
EmbeddingMergeMethod = arch_typing.EmbeddingMergeMethod
MLP_INPUT_SIGNATURE = arch_typing.MLP_INPUT_SIGNATURE
MLP_OUTPUT_SIGNATURE = arch_typing.MLP_OUTPUT_SIGNATURE

################################################################################
# MARK: Base classes
################################################################################


class BaseTimeEmbedder(nn.Module, abc.ABC):
  """Abstract base class for the time embedder."""

  @abc.abstractmethod
  def __call__(self, time: hd_typing.TimeTree) -> Float['batch ...']:
    ...


class BaseEmbedder(nn.Module, abc.ABC):
  """Abstract base class for conditioning embedders."""

  @property
  @abc.abstractmethod
  def output_shape(self) -> tuple[int, ...]:
    ...

  @abc.abstractmethod
  def __call__(
      self, conditioning: hd_typing.Conditioning
  ) -> Float['batch ...']:
    ...


class BaseConditioningEncoder(nn.Module, abc.ABC):
  """Abstract base class for the conditioning encoder."""

  @abc.abstractmethod
  def __call__(
      self,
      time: hd_typing.TimeArray,
      conditioning: hd_typing.Conditioning | None,
      is_training: bool,
  ) -> dict[ConditioningMechanism, Float['batch ...']]:
    ...


################################################################################
# MARK: Time embedders
################################################################################


class SinusoidalTimeEmbedder(BaseTimeEmbedder):
  """Sinusoidal time embedder.

  This module encodes the time step `t` into a dense embedding.
  It performs the following sequence of operations:
  > [SinusoidalSequenceEmbedding]
  > [Dense]
  > [Activation]
  > [Dense]

  Attributes:
    activation: The activation function to use.
    embedding_dim: The dimension of sinusoidal embeddings.
    num_features: Number of features in the dense layers.
    dropout_rate: The dropout rate to use.
    dtype: The dtype to use.
  """

  activation: str
  embedding_dim: int
  num_features: int
  dtype: DType = jnp.float32

  def setup(self):
    self.act = getattr(jax.nn, self.activation)
    self.init_input = nn.with_logical_partitioning(
        nn.linear.default_kernel_init, MLP_INPUT_SIGNATURE
    )
    self.init_output = nn.with_logical_partitioning(
        nn.linear.default_kernel_init, MLP_OUTPUT_SIGNATURE
    )

  @nn.compact
  @typechecked
  def __call__(self, time: hd_typing.TimeArray) -> Float['batch num_features']:
    t_emb = sequence_embedders.SinusoidalSequenceEmbedding(self.embedding_dim)(
        time
    )
    t_emb = nn.Dense(
        features=self.num_features,
        kernel_init=self.init_input,
        dtype=self.dtype,
        name='DenseInput',
    )(t_emb)
    t_emb = self.act(t_emb)
    t_emb = nn.Dense(
        features=self.num_features,
        kernel_init=self.init_output,
        dtype=self.dtype,
        name='DenseOutput',
    )(t_emb)

    return t_emb


class ZeroTimeEmbedder(BaseTimeEmbedder):
  """Time embedder that returns zeros.

  This allows to train models without time conditioning.

  Attributes:
    num_features: The output dimension of the embedder.
  """

  num_features: int

  @typechecked
  def __call__(self, time: hd_typing.TimeArray) -> Float['batch num_features']:
    return jnp.zeros((time.shape[0], self.num_features))


class IdentityTimeEmbedder(BaseTimeEmbedder):
  """Time embedder that returns time without any transformation."""

  @typechecked
  def __call__(self, time: hd_typing.TimeArray) -> hd_typing.TimeArray:
    return time


class NestedTimeEmbedder(BaseTimeEmbedder):
  """Wrapper for a pytree of time embedders mapped over the time tree."""

  time_embedders: PyTree[BaseTimeEmbedder]

  @nn.compact
  @typechecked
  def __call__(self, time: hd_typing.TimeTree) -> Float['batch ...']:
    # lenient alternative to jax.tree.map
    t_emb_tree = utils.lenient_map(
        lambda x, time_embedder: time_embedder.copy()(x),
        time,
        self.time_embedders,
    )
    # Add all the time embeddings together.
    leaves, _ = jax.tree_util.tree_flatten(t_emb_tree)
    t_emb = jnp.sum(jnp.stack(leaves), axis=0)
    return t_emb


################################################################################
# MARK: Conditioning embedders
################################################################################


class LabelEmbedder(BaseEmbedder):
  """Embedder for integer labels.

  Attributes:
    num_classes: The number of classes.
    num_features: The number of features in the embedding.
    conditioning_key: The field in the conditioning dictionary to which apply
      the Embedding layer.
    dtype: The dtype to use.
  """

  num_classes: int
  num_features: int
  conditioning_key: str = 'label'
  dtype: DType = jnp.float32

  @property
  def output_shape(self) -> tuple[int, ...]:
    """Returns the output shape of the embedder, excluding the batch dim."""
    return (self.num_features,)

  @nn.compact
  @typechecked
  def __call__(
      self,
      conditioning: hd_typing.Conditioning,
  ) -> Float['batch num_features']:
    if self.conditioning_key not in conditioning:
      raise ValueError(
          f'Conditioning key {self.conditioning_key} not found in conditioning.'
          f' Available keys: {sorted(list(conditioning.keys()))}'
      )
    inputs = conditioning[self.conditioning_key]
    integer_inputs = inputs.astype(jnp.int32)
    return nn.Embed(
        num_embeddings=self.num_classes,
        features=self.num_features,
        dtype=self.dtype,
    )(integer_inputs)


class LinearEmbedder(BaseEmbedder):
  """Linear embedding.

  This module encodes an input into a dense embedding by applying a linear
  transformation. It does not assume that the inputs are in any particular
  range. This is useful for vector conditioning.

  Attributes:
    num_features: The number of features in the embedding.
    dtype: The dtype to use.
  """

  num_features: int
  conditioning_key: str = 'label'
  use_bias: bool = False
  dtype: DType = jnp.float32

  @property
  def output_shape(self) -> tuple[int, ...]:
    """Returns the output shape of the embedder, excluding the batch dim."""
    return (self.num_features,)

  @nn.compact
  @typechecked
  def __call__(
      self,
      conditioning: hd_typing.Conditioning,
  ) -> Float['batch num_features']:
    if self.conditioning_key not in conditioning:
      raise ValueError(
          f'Conditioning key {self.conditioning_key} not found in conditioning.'
          f' Available keys: {sorted(list(conditioning.keys()))}'
      )

    inputs = conditioning[self.conditioning_key]
    # If the inputs have shape (batch,), then we reshape it to (batch, 1).
    if len(inputs.shape) == 1:
      inputs = jnp.reshape(inputs, (inputs.shape[0], -1))

    return nn.Dense(
        features=self.num_features,
        use_bias=self.use_bias,
        dtype=self.dtype,
    )(inputs)


class MLPEmbedder(BaseEmbedder):
  """MLP embedding.

  This module encodes an input into a dense embedding by applying a MLP.
  It does not assume that the inputs are in any particular range. This is useful
  for vector conditioning. For a given list of conditioning keys, the inputs are
  reshaped into (batch_size, -1) and are concatenated and fed into a MLP.

  Attributes:
    num_features: The number of features in the embedding.
    dtype: The dtype to use.
  """

  hidden_sizes: Sequence[int]
  num_features: int
  conditioning_keys: Sequence[str]
  activation: str = 'gelu'
  dtype: DType = jnp.float32

  @property
  def output_shape(self) -> tuple[int, ...]:
    """Returns the output shape of the embedder, excluding the batch dim."""
    return (self.num_features,)

  @nn.compact
  @typechecked
  def __call__(
      self,
      conditioning: hd_typing.Conditioning,
  ) -> Float['batch num_features']:
    for key in self.conditioning_keys:
      if key not in conditioning:
        raise ValueError(
            f'Conditioning key {key} not found in conditioning.'
            f' Available keys: {sorted(list(conditioning.keys()))}'
        )

    all_inputs = []
    for key in self.conditioning_keys:
      inputs = conditioning[key]
      batch_size = inputs.shape[0]
      inputs = jnp.reshape(inputs, (batch_size, -1))
      all_inputs.append(inputs)
    all_inputs = jnp.concatenate(all_inputs, axis=-1)

    mlp_module = mlp_blocks.MLP(
        hidden_sizes=self.hidden_sizes,
        output_size=self.num_features,
        activation=self.activation,
    )
    # We put `is_training=False` because we do not use dropout in the `MLP`.
    return mlp_module(all_inputs, is_training=False)


################################################################################
# MARK: Process and combine time and conditioning signals
################################################################################


class ConditioningEncoder(BaseConditioningEncoder):
  """Encodes and combines time and conditioning signals for a diffusion model.

  This module orchestrates the transformation of raw time and a dictionary
  of conditioning signals into a unified embedding. This final embedding
  is then supplied to the main model, typically through an adaptive
  normalization layer or a cross-attention mechanism.

  During training, the embeddings of conditioning signals can be dropped by
  setting `conditioning_dropout_rate` > 0. This would set the embeddings to
  zero. It is normally used for classifier-free guidance
  (https://arxiv.org/abs/2207.12598).

  Usage:
    ```python
    process_conditioning = ConditioningEncoder(
        time_embedder=SinusoidalTimeEmbedder(...),
        conditioning_embedders=dict(
            label_foo=LabelEmbedder(...),
            label_bar=LinearEmbedder(...),
        ),
        embedding_merging_method=EmbeddingMergeMethod.SUM,
        conditioning_rules=dict(
            label_foo =ConditioningMechanism.ADAPTIVE_NORM,
            label_bar=ConditioningMechanism.CROSS_ATTENTION,
            time=ConditioningMechanism.ADAPTIVE_NORM,
        ),
    )
    ```


  Attributes:
    time_embedder: A module (e.g., `SinusoidalTimeEmbedder`) that converts the
      time step into a dense embedding.
    conditioning_embedders: A dictionary of named embedders. The names can be
      chosen arbitrarily, but are used to identify the signals in the
      `conditioning_rules`.
    embedding_merging_method: The method used to combine embeddings, e.g. sum
      them or concatenate them.
    conditioning_rules: A dictionary specifying which conditioning mechanism
      (e.g., `adaptive_norm`) to use for which embedding. For a given
      conditioning mechanism, the embeddings are merged according to the
      specified `embedding_merging_method`.
    conditioning_dropout_rate: The rate at which to drop the conditioning
      signals.
  """

  time_embedder: BaseTimeEmbedder
  conditioning_embedders: dict[str, BaseEmbedder]
  embedding_merging_method: EmbeddingMergeMethod
  conditioning_rules: dict[str, ConditioningMechanism]
  conditioning_dropout_rate: float = 0.0

  def setup(self):

    if self.embedding_merging_method == EmbeddingMergeMethod.SUM:
      self.embedding_merging_fn = lambda x, y: x + y
    elif self.embedding_merging_method == EmbeddingMergeMethod.CONCAT:
      self.embedding_merging_fn = lambda x, y: jnp.concatenate([x, y], axis=-1)
    else:
      raise ValueError(
          'Unsupported embedding merging method:'
          f' {self.embedding_merging_method}'
      )

    self.embedders_names = set(self.conditioning_embedders.keys())
    embedders_names_with_time = self.embedders_names | {'time'}

    cond_rule_names = set(self.conditioning_rules.keys())

    if embedders_names_with_time != cond_rule_names:
      raise ValueError(
          'The keys in the `conditioning_rules` must exactly match the'
          ' keys in the `conditioning_encoders` (apart from the `time` key).\n'
          f'Provided keys: {sorted(list(cond_rule_names))}\n'
          f'Expected keys: {sorted(list(embedders_names_with_time))}'
      )

  @nn.compact
  @typechecked
  def __call__(
      self,
      time: hd_typing.TimeTree,
      conditioning: hd_typing.Conditioning | None,
      is_training: bool,
  ) -> dict[ConditioningMechanism, Float['batch ...']]:
    """Encodes and combines time and conditioning signals.

    The output is a dictionary where keys are the embedding mechanisms specified
    in the `conditioning_rules` (e.g., `adaptive_norm`, `cross_attention`), and
    the values are the corresponding merged conditioning embeddings.

    Args:
      time: The time step.
      conditioning: The dictionary of conditioning signals. If None, then zeros
        are used as embeddings for all conditioning signals, so the output
        embedding will only depend on time.
      is_training: Whether the model is in training mode.

    Returns:
      A dictionary of conditioning embeddings, keyed by mechanism type.
    """

    # Embed time.
    t_emb = self.time_embedder.copy(name='TimeEmbedder')(time)
    batch_size = t_emb.shape[0]

    # Encode other conditioning info.
    # If conditioning is None, we use zeros as embeddings.
    # This covers the case for unconditional generation or classifier-free
    # guidance (https://arxiv.org/abs/2207.12598).
    cond_embs = {}
    if conditioning is not None:
      for embedder_name in self.embedders_names:
        embedder = self.conditioning_embedders[embedder_name]
        cond_embs[embedder_name] = embedder.copy(
            name=f'Embedder_{embedder_name}'
        )(conditioning)
    else:
      for embedder_name in self.embedders_names:
        embedder = self.conditioning_embedders[embedder_name]
        cond_embs[embedder_name] = jnp.zeros(
            (batch_size,) + embedder.output_shape, dtype=t_emb.dtype
        )

    # Apply dropout to the conditioning (same mask for all).
    if is_training and self.conditioning_dropout_rate > 0.0:
      rng = self.make_rng('dropout')
      keep_prob = 1.0 - self.conditioning_dropout_rate
      mask = jax.random.bernoulli(
          rng,
          p=keep_prob,
          shape=(batch_size,),
      )
      for name, emb in cond_embs.items():
        emb_shape = emb.shape[1:]
        broadcast_mask = jnp.reshape(
            mask, (batch_size,) + (1,) * len(emb_shape)
        )
        cond_embs[name] = jnp.where(broadcast_mask, emb, jnp.zeros_like(emb))

    # Merge time and conditionings
    cond_embs['time'] = t_emb

    out = dict()
    for name, conditioning_mechanism in self.conditioning_rules.items():
      if conditioning_mechanism in out:
        out[conditioning_mechanism] = self.embedding_merging_fn(
            out[conditioning_mechanism], cond_embs[name]
        )
      else:
        out[conditioning_mechanism] = cond_embs[name]

    return out
