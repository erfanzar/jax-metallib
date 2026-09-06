# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.1.0] - 2026-09-04

### Added
- Support for JAX / jaxlib 0.11.x (requires Python ≥ 3.12; JAX 0.11 dropped 3.11)

### Changed
- Bumped project version to `0.11.1.0` tracking JAX 0.11.1
- Bumped Python dependency pins to `jax>=0.11.0,<0.12` and `jaxlib>=0.11.0,<0.12`
- Updated native dependency pins to the XLA `dcf304bc` / StableHLO `7b1b157` / LLVM `6d1ace5` triple matching jaxlib 0.11.1 (PJRT C API 0.114)
- `setup_deps.sh --force` now removes installed StableHLO/LLVM/MLIR headers; stale generated headers previously shadowed fresh ones and broke StableHLO builds
- Abseil/Protobuf release tarballs in `setup_deps.sh` are now SHA256-verified before extraction
- `autotune_cache.bin` now carries a magic + format-version header; unversioned or mismatched cache files are discarded on load instead of silently misreading stale kernel decisions
- Mixed graph+native plans are now the default: large last-axis sum/mean reductions (`>= 32768` elements) run on the native TG kernel, which is 13× faster than the MPSGraph reduce at 64K and ~270× at 1M. Opt out with `JAX_METALLIB_DISABLE_MIXED_INTRINSIC_NATIVE=1`
- Small f32 matmuls (`MNK < 4.5M`, tunable via `JAX_METALLIB_CPU_BLAS_MAX_MNK`) now bake to CPU BLAS: gemm 128³ drops from 7.8 ms to ~196 µs
- Abseil/Protobuf release tarballs in `setup_deps.sh` are now SHA256-verified before extraction
- `autotune_cache.bin` now carries a magic + format-version header; unversioned or mismatched cache files are discarded on load instead of silently misreading stale kernel decisions
- Native intrinsic kernels are **disabled by default** on JAX 0.11: eager dispatch exposed a fixed `CFRelease(nil)` abort plus fused-chain scheduling defects producing wrong gradient results (MultivariateNormal log-prob). Opt in with `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1`; `JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS=1` still wins

### Fixed
- Batched `cholesky` (and other CPU-reading native handlers): commit and wait for pending GPU work before host reads, fixing garbage input reads under JAX 0.11's dispatch timing
- Fixed a `CFRelease(nil)` abort (SIGTRAP) in `MpsExecutable::Execute` when an eliminated (constant-folded) output slot had no buffer — surfaced by JAX 0.11's eager dispatch on eager gradients (e.g. `grad(cos)`) and NumPyro workloads; eliminated output slots now return a zeroed placeholder buffer
- Fixed the intrinsic-native broadcast-ordering bug (`broadcast_in_dim` before `divide`/`multiply` producing wrong results): fused elementwise chains are now split at intermediates that are consumed outside the chain, so cross-chain readers no longer observe eliminated (never-written) slots
- Known issue (open, opt-in only): a slot-aliasing defect between graph-step targets and fused-chain outputs still yields wrong results for some MultivariateNormal log-prob gradients when intrinsic kernels are enabled via `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1`
- `collective_ops`: new StableHLO exposes `getReplicaGroups()` as generic `mlir::Attribute`; validators now cast internally
- Test harness: grad target signature is now bindable by flax 0.12.9's `resolve_kwargs`; int64 splat subprocess pins `JAX_PLATFORMS=cpu,mps` for explicit backend selection

## [0.10.1.0] - 2026-06-27

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

[Unreleased]: https://github.com/erfanzar/jax-metallib/compare/v0.11.1.0...HEAD
[0.11.1.0]: https://github.com/erfanzar/jax-metallib/compare/v0.10.1.0...v0.11.1.0
[0.10.1.0]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.10.1.0
[0.10.0.0]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.10.0.0
[0.9.4.10]: https://github.com/erfanzar/jax-metallib/releases/tag/v0.9.4.10
