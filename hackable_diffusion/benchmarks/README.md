# Hackable Diffusion Benchmarks

This directory contains scripts to verify the performance optimizations and numerical fidelity of the library.

## Contents
- `run_benchmarks.py`: Comprehensive performance suite for Attention, RMSNorm, and Core Blocks.
- `verify_fidelity.py`: Checks numerical equivalence between optimized and baseline implementations.

## Running Benchmarks

To run the full suite:
```bash
python3 -m third_party.py.hackable_diffusion.benchmarks.run_benchmarks
```

## Running Fidelity Checks
```bash
python3 -m third_party.py.hackable_diffusion.benchmarks.verify_fidelity
```

## Optimization Notes
Current optimizations focus on:
1. XLA-native Flash Attention via `jax.nn.dot_product_attention`.
2. Fused RMSNorm kernels using `jax.lax.rsqrt`.
3. Redundancy elimination in conditioning modulation logic.
