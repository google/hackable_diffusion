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

r"""Simple diffusion config for training a UNet on ImageNet 64x64.

Should reach about FID=4.1 within 500k steps, and FID=2.7 within 2M steps.
"""

from kauldron import konfig

# pylint: disable=g-import-not-at-top
with konfig.imports():
  from kauldron import kd
  import optax
  from jax.experimental import checkify
  from hackable_diffusion import hd
  from hackable_diffusion.kdiff import core
  from hackable_diffusion.kdiff import data
  from hackable_diffusion.kdiff import evals


# pylint: enable=g-import-not-at-top


def get_config():
  """Get the default hyperparameter configuration."""
  cfg = kd.train.Trainer()
  cfg._konfig_experimental_nofreeze = True  # pylint: disable=protected-access
  cfg.seed = 1337

  cfg.aux = konfig.ConfigDict()
  cfg.aux.cond_embedding_dim = 192

  # MARK: Corruption Process
  corruption_process = hd.corruption.GaussianProcess(
      schedule=hd.corruption.RFSchedule(),
  )

  # MARK: Conditioning
  time_encoder = hd.architecture.SinusoidalTimeEmbedder(
      activation="gelu",
      num_features=cfg.ref.aux.cond_embedding_dim,
      embedding_dim=cfg.ref.aux.cond_embedding_dim,
  )
  label_encoder = hd.architecture.LabelEmbedder(
      num_classes=1000,
      num_features=cfg.ref.aux.cond_embedding_dim,
      conditioning_key="label",
  )
  conditioning_encoder = hd.architecture.ConditioningEncoder(
      time_embedder=time_encoder,
      conditioning_embedders={
          "label": label_encoder,
      },
      embedding_merging_method=hd.architecture.EmbeddingMergeMethod.SUM,
      conditioning_rules={
          "label": hd.architecture.ConditioningMechanism.ADAPTIVE_NORM,
          "time": hd.architecture.ConditioningMechanism.ADAPTIVE_NORM,
      },
      conditioning_dropout_rate=0.1,
  )

  # MARK: Backbone
  backbone_network = hd.architecture.Unet(
      base_channels=192,
      channels_multiplier=(1, 2, 3, 4),
      num_residual_blocks=(3, 3, 3, 3),
      downsample_method=hd.architecture.DownsampleType.AVG_POOL,
      upsample_method=hd.architecture.UpsampleType.NEAREST,
      dropout_rate=(0.0, 0.1, 0.1, 0.1),
      bottleneck_dropout_rate=0.1,
      self_attention_bool=(False, False, True, True),
      cross_attention_bool=(False, False, False, False),
      attention_normalize_qk=True,
      attention_use_rope=True,
      attention_rope_position_type=hd.architecture.RoPEPositionType.SQUARE,
      attention_num_heads=-1,
      attention_head_dim=64,
      normalization_type=hd.architecture.NormalizationType.RMS_NORM,
      normalization_num_groups=None,
      activation="gelu",
      skip_connection_method=hd.architecture.SkipConnectionMethod.UNNORMALIZED_ADD,
  )

  # MARK: Model / Diffusion

  cfg.model = core.Diffusion(
      x0="batch.image",
      cond={"label": "batch.label[:,0]"},
      corruption_process=corruption_process,
      time_sampler=hd.time_sampling.UniformTimeSampler(safety_epsilon=1e-4),
      network=hd.diffusion_network.DiffusionNetwork(
          prediction_type="velocity",
          backbone_network=backbone_network,
          conditioning_encoder=conditioning_encoder,
      ),
  )

  # MARK: Training
  cfg.num_train_steps = 3_000_000

  cfg.train_ds = _make_ds(training=True, batch_size=512)

  # MARK: Losses
  cfg.train_losses = {
      "diffusion_loss": core.KauldronLossWrapper(
          loss=hd.loss.SiD2Loss(
              schedule=cfg.ref.model.corruption_process.schedule,
              prediction_type=cfg.ref.model.network.prediction_type,
              bias=2.0,
          ),
      ),
  }

  # MARK: Optimizer
  cfg.schedules = {
      "learning_rate": optax.warmup_constant_schedule(
          init_value=0.0,
          peak_value=3e-5,
          warmup_steps=10_000,
      )
  }

  cfg.optimizer = kd.optim.named_chain(**{
      "clip": optax.clip_by_global_norm(max_norm=1.0),
      "adam": optax.scale_by_adam(b1=0.9, b2=0.99, eps=1e-12),
      "lr": optax.scale_by_learning_rate(cfg.ref.schedules["learning_rate"]),
      "ema": kd.optim.ema_params(decay=0.9999),
  })

  # MARK: Metrics
  cfg.train_metrics = {
      "grad_norm": kd.metrics.SkipIfMissing(
          kd.metrics.TreeReduce(
              metric=kd.metrics.Norm(
                  tensor="grads", axis=None, aggregation_type="concat"
              )
          )
      ),
  }
  cfg.train_summaries = {
      "overview": kd.contrib.summaries.ImageGrid.simple(
          columns={
              "gt": "batch.image",
              "x0_pred": "preds.output.x0",
              "xt": "preds.xt",
          },
          in_vrange=(-1.0, 1.0),
      ),
  }

  # MARK: Evals
  cfg.eval_ds = _make_ds(training=False, batch_size=512)

  cfg.evals = {
      "sample_DDIM": evals.SamplingEvaluator(
          run=kd.evals.EveryNSteps(50_000, skip_first=True),
          init_transform=kd.optim.UseEmaParams(),
          num_batches=None,
          sampler=hd.sampling.DiffusionSampler(
              time_schedule=hd.sampling.UniformTimeSchedule(),
              stepper=hd.sampling.DDIMStep(
                  stoch_coeff=0.25,
                  corruption_process=cfg.ref.model.corruption_process,
              ),
              num_steps=250,
          ),
          metrics={
              # TODO(klausg): FID with stats from training set.
              # TODO(klausg): opensource FID metric.
              "fid": kd.metrics.Fid(
                  pred="samples.xt",
                  target="batch.image",
                  in_vrange=(-1.0, 1.0),
              ),
          },
          summaries={
              "overview": kd.contrib.summaries.ImageGrid.simple(
                  columns={
                      "gt": "batch.image",
                      "sample": "samples.xt",
                  },
                  in_vrange=(-1.0, 1.0),
                  num_images=10,
              )
          },
      ),
  }

  # MARK: Checkpointer
  cfg.checkpointer = kd.ckpts.Checkpointer(
      save_interval_steps=10_000,
      max_to_keep=3,
  )

  # MARK: Other
  # hackable diffusion requires checkify to be activated.
  cfg.checkify_error_categories = checkify.user_checks
  # Set up random streams.
  cfg.rng_streams = kd.train.RngStreams([
      # The SamplingEvaluator uses the "sampling" stream.
      kd.train.RngStream("default", train=True, eval=True),
      kd.train.RngStream("sampling", train=True, eval=True),
  ])

  return cfg


# MARK: _make_ds
def _make_ds(training: bool, batch_size: int, split: str | None = None):
  """Imagenet 64x64 dataset."""
  transforms = [
      kd.data.Elements(keep=["image", "label"]),
      kd.data.tf.Resize(key="image", height=64, width=64, method="area"),
      kd.data.ValueRange(key="image", in_vrange=(0, 255), vrange=(-1, 1)),
      kd.data.Rearrange(key="label", pattern="... -> ... 1"),
  ]
  if training:
    transforms.append(kd.data.tf.RandomFlipLeftRight(key="image"))

  if split is None:
    split = "train" if training else "validation"

  return kd.data.tf.Tfds(
      name="imagenet2012",
      decoders={"image": data.decode_and_central_square_crop()},
      split=split,
      shuffle=True if training else False,
      num_epochs=None if training else 1,
      transforms=transforms,
      batch_size=batch_size,
  )
