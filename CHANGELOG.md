# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.1.0] - 2025-06-27

### Added
- Mixed-precision fp32 GEMM: large matmuls automatically cast to fp16 for ~2× speedup (controlled by `JAX_METALLIB_GEMM_TRUE_F32`)
- Single-pass full softmax kernel for ≤ 8192 features (eliminates second device read, ~1.5× faster)
- Adaptive CompletionEvent spin-then-condvar strategy (3-phase: 5K short → 50K medium → condvar; reduces tail latency for sub-200 µs GPU fences)
- Barrier-free MPSGraph direct-run path for single-graph-step plans (eliminates empty blit barrier, ~150–300 µs dispatch win)
- Register-based bitonic sort kernel for multi-row 64–256 element sorts (no shared mem / barriers, ~3% faster)
- Native last-axis `reduce.sum` routing (simdgroup per-row reduction, parity with `reduce.mean`)
- Cholesky LAPACK CPU fallback for small batches (`JAX_METALLIB_CHOLESKY_LAPACK_MAX_N`, default 256)
- macOS 15+ compatibility: modern MPSGraph random, slice, and concat APIs
- `rng_bit_generator` UInt32 now emits a clean error instead of crashing
- `select_and_scatter` supports StableHLO OpaqueProperties and defaults missing strides to 1
- NumPyro import guard in test configs (tests run without NumPyro installed)

### Changed
- Updated MPSGraph API calls to macOS 15+ equivalents (`randomTensorWithShape:descriptor:seed:`, `sliceTensor:starts:ends:strides:`, `concatTensors:dimension:`)
- Downgraded softmax fusion log from INFO to DEBUG to reduce console noise
- Tightened `dot_general` multi-batch-dim reshape logic
- Rebuilt execution-plan dispatch for single-step graphs to use `encodeToCommandBuffer:` instead of `runWithMTLCommandQueue:` + manual barrier

### Fixed
- `dot_general` multi-batch-dim correctness bug (previously failing on all transformer/GPT-2 benchmarks)
- `rng_bit_generator` assertion crash on UInt32 output
- `select_and_scatter` missing window_strides default
- Mixed-path single-step softmax dispatch correctness (barrier synchronization gap)

## [0.10.0.0] - 2025-06

### Added
- Support for JAX / jaxlib 0.10.x
- Native Metal kernels for elementwise, reduce, GEMM, LayerNorm, gather/scatter operations
- Initial single-pass softmax and LayerNorm fusion paths
- Custom Metal kernel API (`metal.py`) and MX quantization (`quantized.py`)

### Changed
- Bumped Python dependency pins to `jax>=0.10.0,<0.11` and `jaxlib>=0.10.0,<0.11`
- Updated native dependency pins to the XLA `b6f37ab` / StableHLO `0dc0fd71` / LLVM `5e14916` triple matching jaxlib 0.10.0 bytecode
- Bumped `_BUILT_FOR_JAXLIB` runtime version sentinel to `(0, 10)`

### Fixed
- Hardened control-flow lowering and zero-size tensor handling in pjrt_plugin

## [0.9.4.10] - 2024

### Added
- Initial jax-metallib PJRT plugin for Silicon and M Chips ([01469a0](https://github.com/erfanzar/jax-metallib/commit/01469a0))
- Native Metal kernels for elementwise, reduce, GEMM, LayerNorm, gather/scatter operations ([4120e2f](https://github.com/erfanzar/jax-metallib/commit/4120e2f))

### Fixed
- Hardened control-flow lowering and zero-size tensor handling in pjrt_plugin ([38e63d3](https://github.com/erfanzar/jax-metallib/commit/38e63d3))

[Unreleased]: https://github.com/erfanzar/jax-metallib/compare/v0.10.1.0...HEAD
[0.10.1.0]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.10.1.0
[0.10.0.0]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.10.0.0
[0.9.4.10]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.9.4.10
