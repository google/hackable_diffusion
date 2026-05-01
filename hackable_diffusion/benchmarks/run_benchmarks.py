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

"""Benchmark suite for Hackable Diffusion optimizations."""

import time
import jax
import jax.numpy as jnp
from hackable_diffusion.lib.architecture import attention
from hackable_diffusion.lib.architecture import normalization
from hackable_diffusion.lib.architecture import dit_blocks

def benchmark_component(name, fn, *args, iters=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        fn(*args).block_until_ready()
    
    # Measure
    start = time.time()
    for _ in range(iters):
        fn(*args).block_until_ready()
    end = time.time()
    
    avg_ms = (end - start) / iters * 1000
    print(f"{name:.<30} {avg_ms:.4f} ms")
    return avg_ms

def run_all():
    print("Starting Hackable Diffusion Optimizations Benchmark...")
    print("-" * 50)
    
    key = jax.random.PRNGKey(0)
    
    # 1. Attention
    batch, seq, heads, hdim = 16, 1024, 16, 64
    x_attn = jax.random.normal(key, (batch, seq, heads * hdim))
    mha = attention.MultiHeadAttention(num_heads=heads, head_dim=hdim)
    params_attn = mha.init(key, x_attn, None)
    
    @jax.jit
    def attn_fn(p, x): return mha.apply(p, x, None)
    benchmark_component("MultiHeadAttention (Flash)", attn_fn, params_attn, x_attn)

    # 2. RMSNorm
    x_norm = jax.random.normal(key, (batch, 128, 128, 64))
    norm = normalization.NormalizationLayer(
        normalization_method=normalization.NormalizationType.RMS_NORM,
        conditional=False
    )
    params_norm = norm.init(key, x_norm)
    
    @jax.jit
    def norm_fn(p, x): return norm.apply(p, x)
    benchmark_component("RMSNorm (Fused)", norm_fn, params_norm, x_norm)

    # 3. DiT Block
    x_dit = jax.random.normal(key, (batch, 256, 512))
    cond = jax.random.normal(key, (batch, 512))
    dit = dit_blocks.DiTBlockAdaLNZero(hidden_size=512, num_heads=8)
    params_dit = dit.init(key, x_dit, cond, is_training=True)
    
    @jax.jit
    def dit_fn(p, x, c): return dit.apply(p, x, c, is_training=True)
    benchmark_component("DiT Block (Optimized)", dit_fn, params_dit, x_dit, cond)

    print("-" * 50)
    print("Benchmark Complete.")

if __name__ == "__main__":
    run_all()
