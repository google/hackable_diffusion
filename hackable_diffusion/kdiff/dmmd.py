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

def _dot_product_fn(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
  return jnp.sum(x * y, axis=-1)


class DMMDModel(AbstractModel):
  """Main diffusion model."""

  def __init__(
      self,
      model_cfg: ml_collections.ConfigDict,
      process: GaussianProcess,
      precondition_wrapper: PreconditioningWrapper,
  ):
    self.process = process
    self.precondition_wrapper = precondition_wrapper

    self.discriminator = discriminators.get_noise_conditional_discriminator(
        model_cfg.discriminator_cfg
    )
    self.grad_penalty_coeff = model_cfg.grad_penalty_coeff
    self.num_noise_levels = model_cfg.num_noise_levels
    self.l2_penalty_coeff = model_cfg.l2_penalty_coeff
    self.model_type = model_cfg.model_type
    self.discriminator_cfg = model_cfg.discriminator_cfg

  def forward_fn(
      self,
      x: chex.Array,
      t: chex.Array,
      key: chex.Array,
      params: VariableDict,
      is_training: bool,
  ):
    return self.discriminator.apply(
        {'params': params['discriminator']},
        x,
        t * jnp.ones((x.shape[0],)),
        rngs={'dropout': key},
        is_training=is_training,
    )

  def apply(
      self,
      params: VariableDict,
      key: chex.Array,
      t: chex.Array,
      xt: chex.Array,
      conditioning: Conditioning,
      is_training: bool,
  ) -> OutputParameterization:
    del conditioning  # unused
    out = self.forward_fn(xt, t, key, params, is_training)
    return interface.OutputParameterization(
        x0=out, score=out, epsilon=out, velocity=out, v=out
    )

  def apply_inference(
      self,
      params: VariableDict,
      t: chex.Array,
      xt: chex.Array,
      conditioning: Conditioning,
  ) -> OutputParameterization:
    del conditioning  # unused
    out = self.discriminator.apply(
        {'params': params['discriminator']}, xt, t, is_training=False
    )
    return interface.OutputParameterization(
        x0=out, score=out, epsilon=out, velocity=out, v=out
    )

  def init_train_state(
      self,
      input_shape: tuple[int, ...],
      conditioning_shape: Conditioning,
      optimizer: Optimizer,
      key: chex.Array,
  ) -> TrainState:
    """Create and initialize the train state."""
    del conditioning_shape  # unused
    # TODO(agalashov): See b/395854578.
    batch_size = input_shape[0]
    dummy_t = self.precondition_wrapper.time_sampler(
        key=key, batch_size=batch_size
    )
    dummy_xt = jnp.ones(input_shape)

    variables = self.discriminator.init(
        {'params': key, 'dropout': key}, dummy_xt, dummy_t, is_training=True
    )
    params = {'discriminator': variables['params']}
    parameter_overview.log_parameter_overview(params)

    return interface.TrainState(
        step=0,
        params=params,
        ema_params=utils.copy_params(params),
        opt_state=optimizer.init(params),
    )

  def witness_train(self, z, params, diff_phi_clean_minus_noisy, t, key):
    """Computes witness value for a given `z` at `time`."""
    # E_{X} [K_Xz] - E_{Z} [K_Zz]; X - clean; Z - noisy
    # <phi(X), phi(z)> - <phi(Z), phi(z)> = <phi(X) - phi(Z), phi(z)>
    phi_z = self.forward_fn(z, t, key, params, is_training=True)
    return jnp.squeeze(_dot_product_fn(diff_phi_clean_minus_noisy, phi_z))

  def loss(
      self,
      params: VariableDict,
      key: chex.Array,
      x0: chex.Array,
      conditioning: Conditioning,
      is_training: bool,
  ) -> tuple[chex.Array, dict[str, Any]]:
    """Loss function for the model."""
    num_samples = x0.shape[0]
    num_noise_levels = self.num_noise_levels

    def _replicate_across_noise_levels(x):
      data_shape = x.shape[1:]
      x = jnp.expand_dims(x, axis=0)
      x = jnp.reshape(x, (1, num_samples) + data_shape)
      x = jnp.tile(x, (num_noise_levels, 1) + (1,) * len(data_shape))
      x = jnp.reshape(x, (num_noise_levels * num_samples,) + data_shape)
      return x

    def _split_noise_levels(x):
      x = jnp.reshape(x, (num_noise_levels, num_samples) + x.shape[1:])
      return x

    # Replicate data across noise levels.
    x0 = _replicate_across_noise_levels(x0)

    # Sample times
    key, _ = jax.random.split(key)
    t = self.precondition_wrapper.time_sampler(
        key=key, batch_size=num_noise_levels
    )
    t = jnp.reshape(t, (num_noise_levels, 1))
    t = jnp.tile(t, (1, num_samples))
    t = jnp.reshape(t, (num_noise_levels * num_samples,))

    # Sample noise
    key, _ = jax.random.split(key)
    _, xt = self.process.add_noise(key, t, x0)

    # Compute features
    key, _ = jax.random.split(key)
    phi_clean = self.forward_fn(x0, t, key, params, is_training)
    phi_clean = _split_noise_levels(phi_clean)
    key, _ = jax.random.split(key)
    phi_noisy = self.forward_fn(xt, t, key, params, is_training)
    phi_noisy = _split_noise_levels(phi_noisy)

    # Precomputed quantities
    sum_phi_noisy = jnp.sum(phi_noisy, axis=1, keepdims=True)
    sum_phi_clean = jnp.sum(phi_clean, axis=1, keepdims=True)

    n_sq = num_samples * num_samples
    n_sq_m_1 = num_samples * (num_samples - 1)

    metrics = {}
    total_loss = 0.0
    ############################################################################
    # Compute MMD^2
    ############################################################################
    k_xx = (
        jnp.mean((
            _dot_product_fn(sum_phi_noisy, sum_phi_noisy)
            - jnp.sum(
                _dot_product_fn(phi_noisy, phi_noisy), axis=1, keepdims=True
            )
        ))
        / n_sq_m_1
    )
    k_yy = (
        jnp.mean((
            _dot_product_fn(sum_phi_clean, sum_phi_clean)
            - jnp.sum(
                _dot_product_fn(phi_clean, phi_clean), axis=1, keepdims=True
            )
        ))
        / n_sq_m_1
    )
    k_xy = jnp.mean(_dot_product_fn(sum_phi_noisy, sum_phi_clean)) / n_sq
    mmd_sq = k_xx + k_yy - 2 * k_xy
    mmd_loss = -mmd_sq
    total_loss += mmd_loss
    metrics['mmd_loss'] = mmd_loss

    ############################################################################
    # Compute L2 penalty
    ############################################################################
    l2_penalty_loss = jnp.mean(
        jnp.sum(jnp.square(phi_noisy) + jnp.square(phi_clean), axis=-1)
    )
    l2_penalty_loss = l2_penalty_loss * self.l2_penalty_coeff
    metrics['l2_penalty_loss'] = l2_penalty_loss
    total_loss += l2_penalty_loss

    ############################################################################
    # Compute gradient penalty
    ############################################################################
    key, _ = jax.random.split(key)
    alpha = jax.random.uniform(
        key=key,
        shape=(num_noise_levels * num_samples,) + (1,) * len(xt.shape[1:]),
        minval=0.0,
        maxval=1.0,
    )
    mixed_data = x0 * alpha + (1.0 - alpha) * xt
    diff = jnp.mean(phi_clean - phi_noisy, axis=1, keeP1+r4632=1B5B32347E\P0+r2531\P0+r2638\P1+r6B62=7F\P0+r6B49\P1+r6B44=1B5B337E\P1+r6B68=1B4F48\P1+r4037=1B4F46\P1+r6B50=1B5B357E\P1+r6B4E=1B5B367E\pdims=True)
    diff = jnp.reshape(diff, (num_noise_levels, 1, -1))
    diff = jnp.tile(diff, (1, num_samples, 1))
    diff = jnp.reshape(diff, (num_noise_levels * num_samples, -1))

    def _vgrad(f, x):
      y, vjp_fn = jax.vjp(f, x)
      return vjp_fn(jnp.ones(y.shape))[0]

    key, _ = jax.random.split(key)
    witness_fn = lambda x: self.witness_train(x, params, diff, t, key)
    witness_gradients = _vgrad(witness_fn, mixed_data)
    witness_gradients = jnp.reshape(
        witness_gradients, (num_noise_levels * num_samples, -1)
    )
    gradient_penalty = jnp.mean(
        jnp.square(
            jnp.sqrt(1e-8 + jnp.sum(jnp.square(witness_gradients), axis=-1))
            - 1.0
        )
    )
    gradient_penalty_loss = gradient_penalty * self.grad_penalty_coeff
    metrics['grad_penalty_loss'] = gradient_penalty_loss
    total_loss += gradient_penalty_loss

    metrics['total_loss'] = total_loss
    return total_loss, metrics

  def update_params(
      self,
      train_state: TrainState,
      optimizer: Optimizer,
      key: chex.Array,
      batch: Batch,
  ) -> tuple[tuple[VariableDict, optax.OptState], Any]:
    """Update parameters and optimizer state."""
    x0, conditioning = batch.data, batch.conditioning

    grad_fn = jax.value_and_grad(self.loss, has_aux=True)
    (_, metrics), grads = grad_fn(
        train_state.params,
        key=key,
        x0=x0,
        conditioning=conditioning,
        is_training=True,
    )

    grads = jax.lax.pmean(grads, axis_name='batch')
    updates, new_opt_state = optimizer.update(
        grads, train_state.opt_state, train_state.params
    )
    new_params = optax.apply_updates(train_state.params, updates)
    return (new_params, new_opt_state), metrics

  def witness_f_sampling(
      self, noisy_data_z, noisy_data_big_z, phi_clean_data, time, key, params
  ):
    # E_{Z}[K[Zz]} - E_{X}[ K[Xz] ]
    key, _ = jax.random.split(key)
    phi_z = self.forward_fn(noisy_data_z, time, key, params, is_training=False)
    key, _ = jax.random.split(key)
    phi_noisy = self.forward_fn(
        noisy_data_big_z, time, key, params, is_training=False
    )
    m_noisy = phi_noisy.shape[0]
    expected_phi_noisy = (
        jnp.sum(phi_noisy, axis=0, keepdims=True) - jax.lax.stop_gradient(phi_z)
    ) / (m_noisy - 1)
    expected_phi_clean = jnp.mean(phi_clean_data, axis=0, keepdims=True)
    out = jnp.squeeze(
        _dot_product_fn(expected_phi_noisy - expected_phi_clean, phi_z)
    )
    return out

