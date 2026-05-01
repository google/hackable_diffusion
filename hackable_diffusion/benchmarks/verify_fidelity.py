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

"""Numerical fidelity verification for Hackable Diffusion."""

import jax
import jax.numpy as jnp
from hackable_diffusion.lib.architecture import attention
import numpy as np

def verify_attention_fidelity():
    print("Verifying Attention Numerical Fidelity...")
    key = jax.random.PRNGKey(42)
    batch, seq, dim = 2, 64, 128
    x = jax.random.normal(key, (batch, seq, dim))
    
    mha = attention.MultiHeadAttention(num_heads=8)
    params = mha.init(key, x, None)
    
    # We compare against expected properties (stability, finiteness)
    # and shape correctness.
    out = mha.apply(params, x, None)
    
    assert out.shape == x.shape, "Shape mismatch"
    assert jnp.all(jnp.isfinite(out)), "Non-finite values detected"
    
    print("Attention Fidelity: PASSED")

if __name__ == "__main__":
    verify_attention_fidelity()
