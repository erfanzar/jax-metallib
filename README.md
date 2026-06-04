<!-- markdownlint-disable MD033 MD045 MD041 -->

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png">
    <img alt="JAX" src="https://raw.githubusercontent.com/jax-ml/jax/main/images/jax_logo_250px.png" width="120">
  </picture>
  <br>
  <strong>jax-metallib</strong>
  <br>
  <em>Run JAX on Apple Metal GPUs</em>
</p>

<p align="center">
  <a href="https://github.com/erfanzar/jax-metallib/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg"></a>
  <a href="https://pypi.org/project/jax-metallib/"><img alt="PyPI" src="https://img.shields.io/pypi/v/jax-metallib.svg"></a>
  <a href="https://pypi.org/project/jax-metallib/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/jax-metallib.svg"></a>
  <a href="https://github.com/erfanzar/jax-metallib"><img alt="Platform" src="https://img.shields.io/badge/platform-macOS%20(Apple%20Silicon)-lightgrey.svg"></a>
</p>

---

**jax-metallib** is a [PJRT](https://openxla.org/xla/pjrt_integration) plugin that enables [JAX](https://github.com/jax-ml/jax) to run on Apple Metal GPUs. It compiles [StableHLO](https://github.com/openxla/stablehlo) IR to Metal compute kernels via [MPSGraph](https://developer.apple.com/documentation/metalperformanceshadersgraph), giving JAX programs native GPU acceleration on M-series Macs — no code changes required.

## Highlights

- **120+ StableHLO ops** — from basic arithmetic through convolutions, FFTs, linear algebra (Cholesky, QR, SVD), sorting, and control flow
- **Drop-in acceleration** — set `JAX_PLATFORMS=mps` and existing JAX code runs on the Metal GPU
- **Full gradient support** — `jax.grad`, `jax.value_and_grad`, and higher-order derivatives work out of the box
- **JIT kernel fusion** — consecutive elementwise ops are fused into single Metal Shading Language kernels at runtime
- **Native MPS kernels** — performance-critical operations (Cholesky decomposition, triangular solve) use MPS native kernels directly, bypassing the graph compiler
- **Executable serialization** — compiled programs can be serialized and deserialized for AOT compilation and `jax.jit` caching
- **Framework integration** — tested with [Flax](https://github.com/google/flax) (NNX) and [NumPyro](https://github.com/pyro-ppl/numpyro)

## Quick Start

### Install

```bash
pip install jax-metallib
```

### Verify

```bash
JAX_PLATFORMS=mps python -c "import jax; print(jax.devices())"
```

### Run

```python
import jax
import jax.numpy as jnp

x = jnp.ones((1024, 1024))
y = x @ x

f = jax.jit(jax.grad(lambda x: jnp.sum(jnp.tanh(x))))
grads = f(x)
```

## Requirements

| Requirement | Version                                |
| ----------- | -------------------------------------- |
| macOS       | 13.0+ (Ventura or later)               |
| Hardware    | Apple Silicon (M1 / M2 / M3 / M4 / M5) |
| Python      | 3.11, 3.12, or 3.13                    |
| JAX         | 0.10.x                                 |
| jaxlib      | 0.10.x                                 |

## Build from Source

Building from source compiles the native Metal plugin (~6 MB shared library). The first build automatically bootstraps LLVM/MLIR and StableHLO dependencies, which takes about 30 minutes.

```bash
brew install cmake ninja

git clone https://github.com/erfanzar/jax-metallib.git
cd jax-metallib
uv sync --all-groups
uv pip install -e .
```

To skip the automatic dependency bootstrap (if you manage LLVM/MLIR yourself):

```bash
CMAKE_ARGS="-DJAX_METALLIB_AUTO_SETUP_DEPS=OFF" uv pip install -e .
```

### Native Dependencies

The bootstrap script (`scripts/setup_deps.sh`) fetches and builds the following into `~/.local/jax-metallib-deps`:

| Dependency  | Version           | Purpose                                          |
| ----------- | ----------------- | ------------------------------------------------ |
| LLVM + MLIR | XLA pin `b6f37ab` | MLIR infrastructure, StableHLO dialect support   |
| StableHLO   | `3a8886d`        | IR parsing, serialization, dialect definitions   |
| Abseil      | 20250127.0        | C++ utilities (strings, status, synchronization) |
| Protobuf    | 29.3              | Device assignment protocol buffer serialization  |

## Supported Operations

122 operations are registered across StableHLO, CHLO, and MHLO dialects:

<details>
<summary><strong>Unary operations</strong> (42 ops)</summary>

`abs` `cbrt` `ceil` `cosine` `count_leading_zeros` `exponential` `exponential_minus_one` `erf` `floor` `imag` `is_finite` `log` `log_plus_one` `logistic` `negate` `real` `round_nearest_even` `rsqrt` `sign` `sine` `sqrt` `tan` `tanh`

CHLO: `acos` `acosh` `asin` `asinh` `atanh` `bessel_i1e` `conj` `cosh` `digamma` `erf_inv` `erfc` `is_inf` `is_neg_inf` `is_pos_inf` `lgamma` `sinh` `square`

</details>

<details>
<summary><strong>Binary operations</strong> (15 ops)</summary>

`add` `atan2` `clamp` `compare` `divide` `dot` `dot_general` `maximum` `minimum` `multiply` `power` `remainder` `select` `subtract`

CHLO: `next_after`

</details>

<details>
<summary><strong>Shape & indexing</strong> (18 ops)</summary>

`bitcast_convert` `broadcast` `broadcast_in_dim` `concatenate` `convert` `dynamic_broadcast_in_dim` `dynamic_reshape` `dynamic_slice` `dynamic_update_slice` `gather` `get_dimension_size` `pad` `reshape` `reverse` `scatter` `set_dimension_size` `slice` `transpose`

</details>

<details>
<summary><strong>Reductions</strong> (7 ops)</summary>

`reduce` (sum, product, max, min, and, or, argmax, argmin) `reduce_window` `select_and_scatter` `batch_norm_inference` `batch_norm_training` `batch_norm_grad` `return`

</details>

<details>
<summary><strong>Convolution</strong></summary>

`convolution` — full `conv_general_dilated` with arbitrary padding, dilation, strides, feature grouping, and batch grouping

</details>

<details>
<summary><strong>Linear algebra</strong></summary>

`cholesky` (native MPS kernel) `triangular_solve` (native MPS kernel)

</details>

<details>
<summary><strong>Other categories</strong></summary>

| Category        | Operations                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------- |
| Bitwise         | `and` `or` `xor` `not` `shift_left` `shift_right_logical` `shift_right_arithmetic` `popcnt` |
| FFT             | `fft` (FFT, RFFT, IFFT, IRFFT)                                                              |
| Sort            | `sort` `top_k`                                                                              |
| Random          | `rng` `rng_bit_generator` (Threefry / Philox)                                               |
| Tensor creation | `constant` `iota`                                                                           |
| Control flow    | `while` `case` (if/else) `custom_call`                                                      |
| Collective      | `all_reduce` `all_gather` `reduce_scatter` `collective_permute`                             |
| Higher-order    | `map` `complex`                                                                             |

</details>

Encountering an unsupported op prints a diagnostic with a direct link to [file a feature request](https://github.com/erfanzar/jax-metallib/issues/new?template=missing-op.yml).

## Architecture

```md
JAX Python program
        │
        ▼
  StableHLO IR (MLIR bytecode)          ← jax.jit compiles Python to StableHLO
        │
        ▼
  PJRT C API layer                      ← pjrt_api.cc exposes client/device/buffer/exec
        │
        ▼
  StableHLO Parser                      ← deserializes + inlines MLIR modules
        │
        ▼
  Execution Plan Builder                ← walks ops, groups into MPSGraph segments
        │                                  + native MPS kernel steps
        ├──► MPSGraph segments           ← op handlers build compute graphs
        │         │
        │         ▼
        │    Metal command buffer        ← compiled & dispatched to GPU
        │
        └──► Native MPS kernels          ← direct kernel dispatch (Cholesky, etc.)
                  │
                  ▼
         Device memory (MTLBuffer)       ← results flow back to JAX as DeviceArrays
```

The plugin implements two execution models that are interleaved within a single program:

1. **Graph execution** — Consecutive ops are batched into an `MPSGraph`, compiled to a Metal compute pipeline, and dispatched as a single GPU command. This is the primary path for most operations.

2. **Native execution** — Performance-critical operations (e.g., Cholesky decomposition via `MPSMatrixDecompositionCholesky`) bypass the graph compiler and dispatch MPS native kernels directly on `MTLBuffer` objects.

3. **JIT fusion** — Chains of elementwise operations are detected and fused into custom Metal Shading Language (MSL) kernels at runtime, reducing kernel launch overhead and improving memory bandwidth utilization.

## Configuration

### Environment Variables

| Variable                    | Default   | Description                                                 |
| --------------------------- | --------- | ----------------------------------------------------------- |
| `JAX_PLATFORMS`             | —         | Set to `mps` to select the Metal backend                    |
| `JAX_METALLIB_LIBRARY_PATH` | auto      | Override path to `libpjrt_plugin_silicon.dylib`             |
| `MPS_LOG_LEVEL`             | `1`       | Logging verbosity: `0` error, `1` warn, `2` info, `3` debug |
| `JAX_TEST_MODE`             | `compare` | Test mode: `compare` (CPU vs MPS), `mps`, or `cpu`          |
| `JAX_METALLIB_GEMM_NO_MPP`        | —         | Disable MPP fallback for GEMM operations                      |
| `JAX_METALLIB_GEMM_AUTOTUNE`      | —         | Enable GEMM autotuning                                        |
| `JAX_METALLIB_AUTOTUNE_CACHE_DIR` | `~/.cache/jax-metallib` | Directory for autotuning cache              |
| `JAX_METALLIB_GEMM_VARIANT`       | —         | Select GEMM implementation variant                            |
| `JAX_METALLIB_ELEMWISE_AUTOTUNE`  | —         | Enable elementwise operation autotuning                       |
| `JAX_METALLIB_COMPLETION_TIMEOUT_MS` | `30000` | Timeout for completion events in milliseconds                |

### Library Discovery

The plugin searches for the native library in this order:

1. `JAX_METALLIB_LIBRARY_PATH` environment variable
2. Package directory (editable install)
3. `<package>/lib/` (wheel install)
4. `build/*/lib/` (CMake build directory)
5. `/usr/local/lib/`, `/opt/homebrew/lib/`

## Testing

The test suite validates numerical correctness by comparing CPU and MPS results with configurable tolerances.

```bash
uv run pytest

JAX_TEST_MODE=mps uv run pytest

uv run pytest -k "unary"
uv run pytest -k "linalg"
uv run pytest -k "flax"
```

### Test Coverage

| Category          | What's tested                                             |
| ----------------- | --------------------------------------------------------- |
| Value correctness | CPU vs MPS output comparison for all 122 ops              |
| Gradient accuracy | `jax.grad` / `jax.value_and_grad` for differentiable ops  |
| Edge cases        | float16 precision, int64 large values, complex numbers    |
| Regressions       | Catastrophic cancellation (log1p), erf_inv range accuracy |
| Integration       | Flax NNX (Linear, Conv, LayerNorm, MultiHeadAttention)    |
| Integration       | NumPyro probabilistic programming models                  |

### Quality Gate

Pre-commit hooks enforce the full quality gate on every commit (MPS is unavailable in GitHub Actions):

1. **clang-format** — C/C++/ObjC++ formatting (LLVM style, 110 col)
2. **ruff** — Python formatting and linting
3. **build** — full native library rebuild
4. **clang-tidy** — C++ static analysis (with caching via ctcache)
5. **pytest** — full test suite with op coverage enforcement

## Benchmarks

```bash
uv run python -m benchmarks.bench list

JAX_PLATFORMS=mps uv run python -m benchmarks.bench run --case 'anchor\..*' --platform mps

uv run python -m benchmarks.bench run --case '.*' --json-out results.jsonl
```

The benchmark suite includes per-op micro-benchmarks, representative anchor workloads, and competitive benchmarks against other backends.

## Repository Layout

```md
src/
  jax_plugins/silicon/          Python entrypoint — plugin registration & library discovery
  pjrt_plugin/
    api/                        PJRT C API layer (8 implementation files)
      pjrt_api.cc                 Function pointer table (main entry point)
      pjrt_client.cc              Client: compile, buffer creation, platform info
      pjrt_buffer.cc              Buffer: host↔device transfer, copy, clone
      pjrt_executable.cc          Executable: execute, serialize, output metadata
      pjrt_device.cc              Device: attributes, memory, description
      pjrt_event.cc               Event: async completion, create, set
      pjrt_memory.cc              Memory: kind, addressable devices
      pjrt_topology.cc            Topology: device descriptions, platform
    core/                       Backend core (Objective-C++ / Metal)
      mps_client.h/.mm             Metal device & command queue management
      mps_device.h/.mm             GPU device abstraction
      mps_buffer.h/.mm             MTLBuffer wrapper — host copy, clone, blit
      mps_executable.h/.mm         Execution plan builder & runner
      stablehlo_parser.h/.mm       MLIR StableHLO deserializer
      type_utils.h/.mm             MLIR ↔ MPS type conversions
      completion_event.h           Thread-safe async completion primitives
      pjrt_types.h                 PJRT opaque wrapper structs
      logging.h                    Leveled logging macros
    ops/                        StableHLO op handlers (122 registrations)
      registry.h                  Op registry, handler types, macros
      unary_ops.mm                42 unary operations
      binary_ops.mm               15 binary operations
      shape_ops.mm                18 shape & indexing operations
      reduction_ops.mm            7 reduction operations
      convolution_ops.mm          General dilated convolution
      linalg_ops.mm               Cholesky, triangular solve (native MPS)
      bitwise_ops.mm              8 bitwise operations
      sort_ops.mm                 Sort, top-k
      fft_ops.mm                  FFT / RFFT / IFFT / IRFFT
      control_flow_ops.mm         While, case, custom_call
      random_ops.mm               Threefry / Philox RNG
      tensor_creation_ops.mm      Constant, iota
      collective_ops.mm           Single-device collective no-ops
      higher_order_ops.mm         Map
    runtime/                    JIT Metal kernel engine
      metal_kernels.h/.mm          MSL kernel source cache & pipeline compilation
    proto/
      device_assignment.proto     XLA DeviceAssignmentProto definition
tests/
  test_ops.py                   Parametrized test suite (value + gradient)
  test_int64_constant_splats.py Regression: int64 precision above 2^53
  configs/                      Per-category test configurations (15 modules)
benchmarks/
  bench.py                      Benchmark harness (list/run, JSONL output)
  bottlenecks.py                Per-op micro-benchmarks
  vs_jax_mps.py                 Competitive benchmarks
  anchors.py                    Representative anchor workloads
scripts/
  setup_deps.sh                 One-time native dependency bootstrap
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. The short version:

```bash
brew install cmake ninja
./scripts/setup_deps.sh
uv sync --all-groups
uv pip install -e .
pre-commit install

uv pip install -e .
uv run pytest
```

## Acknowledgements

This project draws significant inspiration from [MLX](https://github.com/ml-explore/mlx) by Apple Machine Learning Research. MLX's approach to leveraging Metal and unified memory on Apple Silicon was a major influence on the design and direction of jax-metallib.

## License

Copyright 2026 Erfan Zare Chavoshi (@erfanzar). Apache-2.0 — see [LICENSE](LICENSE).
