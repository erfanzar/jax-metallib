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

- **120 StableHLO ops** — from basic arithmetic through convolutions, FFTs, linear algebra (Cholesky, QR, SVD), sorting, and control flow
- **Drop-in acceleration** — set `JAX_PLATFORMS=mps` and existing JAX code runs on the Metal GPU
- **Full gradient support** — `jax.grad`, `jax.value_and_grad`, and higher-order derivatives work out of the box
- **JIT kernel fusion** — consecutive elementwise ops are fused into single Metal Shading Language kernels at runtime
- **Native MPS kernels** — performance-critical operations (Cholesky decomposition, triangular solve, softmax, LayerNorm) use native kernels directly, bypassing the graph compiler
- **Mixed-precision GEMM** — large fp32 matmuls automatically run at fp16 speed via graph-level cast (optional, controlled by env var)
- **Single-pass softmax** — transformer-scale softmax (≤ 8192 features) uses a fused single-kernel path, eliminating an extra dispatch
- **Custom Metal kernel API** — write and execute raw Metal Shading Language kernels directly on JAX arrays
- **Quantization support** — MX microscaling formats (MXFP4, MXFP8, NVFP4) and affine quantization for inference optimization
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
| macOS       | 13.0+ (Ventura) or 15+ (recommended)   |
| Hardware    | Apple Silicon (M1 / M2 / M3 / M4 / M5) |
| Python      | 3.12 or 3.13                           |
| JAX         | 0.11.x                                 |
| jaxlib      | 0.11.x                                 |

> **Note:** macOS 15+ is recommended for the best compatibility. Some MPSGraph APIs used on macOS 13–14 have been deprecated and replaced with modern equivalents on macOS 15+.

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

| Dependency  | Version | Purpose                                          |
| ----------- | ------- | ------------------------------------------------ |
| LLVM + MLIR | `6d1ace5` (override) | MLIR infrastructure, StableHLO dialect support. The build uses the LLVM commit integrated by the pinned StableHLO revision so MLIR's C++ API matches what StableHLO expects. |
| StableHLO   | `7b1b157` | IR parsing, serialization, dialect definitions   |
| Abseil      | 20250127.0 | C++ utilities (strings, status, synchronization) |
| Protobuf    | 29.3 | Device assignment protocol buffer serialization  |

> **Note:** The build script auto-resolves the correct commits. The table above reflects the actual pins in `scripts/setup_deps.sh`.

## Supported Operations

120 operations are registered across StableHLO, CHLO, and MHLO dialects:

<details>
<summary><strong>Unary operations</strong> (50 ops)</summary>

`abs` `cbrt` `ceil` `cosine` `count_leading_zeros` `exponential` `exponential_minus_one` `erf` `floor` `imag` `is_finite` `log` `log_plus_one` `logistic` `negate` `real` `round_nearest_even` `rsqrt` `sign` `sine` `sqrt` `tan` `tanh` `complex`

CHLO: `acos` `acosh` `asin` `asinh` `atanh` `bessel_i1e` `conj` `cosh` `digamma` `erf_inv` `erfc` `is_inf` `is_neg_inf` `is_pos_inf` `lgamma` `sinh` `square`

MHLO: `tan` `atan2` `cbrt` `is_finite` `round_nearest_even` `rsqrt` `sign` `exponential_minus_one` `log_plus_one` (aliases)

</details>

<details>
<summary><strong>Binary operations</strong> (15 ops)</summary>

`add` `atan2` `clamp` `compare` `divide` `dot` `dot_general` `maximum` `minimum` `multiply` `power` `remainder` `select` `subtract`

CHLO: `next_after`

</details>

<details>
<summary><strong>Shape & indexing</strong> (19 ops)</strong></summary>

`bitcast_convert` `broadcast` `broadcast_in_dim` `concatenate` `convert` `dynamic_broadcast_in_dim` `dynamic_reshape` `dynamic_slice` `dynamic_update_slice` `gather` `get_dimension_size` `pad` `reshape` `reverse` `scatter` `set_dimension_size` `slice` `transpose` `custom_call` (Sharding)

</details>

<details>
<summary><strong>Reductions</strong> (6 ops)</summary>

`reduce` (sum, product, max, min, and, or, argmax, argmin) `reduce_window` `select_and_scatter` `batch_norm_inference` `batch_norm_training` `batch_norm_grad`

</details>

<details>
<summary><strong>Convolution</strong></summary>

`convolution` — full `conv_general_dilated` with arbitrary padding, dilation, strides, feature grouping, and batch grouping

</details>

<details>
<summary><strong>Linear algebra</strong></summary>

`cholesky` (native MPS kernel + LAPACK fallback for small batches) `triangular_solve` (native MPS kernel)

</details>

<details>
<summary><strong>Other categories</strong></summary>

| Category        | Operations                                                                                  |
| --------------- | ------------------------------------------------------------------------------------------- |
| Bitwise         | `and` `or` `xor` `not` `shift_left` `shift_right_logical` `shift_right_arithmetic` `popcnt` |
| FFT             | `fft` (FFT, RFFT, IFFT, IRFFT)                                                              |
| Sort            | `sort` `top_k` (MHLO: `topk`)                                                               |
| Random          | `rng` `rng_bit_generator` (Threefry / Philox)                                               |
| Tensor creation | `constant` `iota`                                                                           |
| Control flow    | `while` `case` (if/else) `custom_call` `return`                                               |
| Collective      | `all_reduce` `all_gather` `reduce_scatter` `collective_permute`                             |
| Higher-order    | `map`                                                                                       |

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
        │                                  + native MPS kernel steps + fused MSL kernels
        ├──► MPSGraph segments           ← op handlers build compute graphs
        │         │
        │         ▼
        │    Metal command buffer        ← compiled & dispatched to GPU
        │
        ├──► Native MPS kernels          ← direct kernel dispatch (Cholesky, Softmax, etc.)
        │         │
        │         ▼
        │    Device memory (MTLBuffer)   ← results flow back to JAX as DeviceArrays
        │
        └──► Fused MSL kernels           ← JIT-fused elementwise chains
                  │
                  ▼
         Device memory (MTLBuffer)
```

The plugin implements three execution models that are interleaved within a single program:

1. **Graph execution** — Consecutive ops are batched into an `MPSGraph`, compiled to a Metal compute pipeline, and dispatched as a single GPU command. This is the primary path for most operations.

2. **Native execution** — Performance-critical operations (e.g., Cholesky decomposition via `MPSMatrixDecompositionCholesky`, softmax via custom Metal kernels, LayerNorm via fused reductions) bypass the graph compiler and dispatch MPS native kernels or hand-tuned MSL directly on `MTLBuffer` objects. Small-batch Cholesky automatically falls back to CPU LAPACK for inputs below a configurable threshold.

3. **JIT fusion** — Chains of elementwise operations are detected and fused into custom Metal Shading Language (MSL) kernels at runtime, reducing kernel launch overhead and improving memory bandwidth utilization.

## Configuration

### Environment Variables (User-facing)

| Variable | Default | Description |
|----------|---------|-------------|
| `JAX_PLATFORMS` | — | Set to `mps` to select the Metal backend |
| `JAX_METALLIB_LIBRARY_PATH` | auto | Override path to `libpjrt_plugin_silicon.dylib` |
| `MPS_LOG_LEVEL` | `1` | Logging verbosity: `0` error, `1` warn, `2` info, `3` debug |
| `JAX_TEST_MODE` | `compare` | Test mode: `compare` (CPU vs MPS), `mps`, or `cpu` |
| `JAX_METALLIB_GEMM_TRUE_F32` | `0` | Set to `1` to disable mixed-precision fp32→fp16 GEMM and force true fp32 (precision-critical workloads) |
| `JAX_METALLIB_CHOLESKY_LAPACK_MAX_N` | `256` | Maximum total elements (`n × batchCount`) for CPU LAPACK Cholesky fallback. Set to `0` to disable |
| `JAX_METALLIB_COMPLETION_TIMEOUT_MS` | `30000` | Timeout for completion events in milliseconds |

> **Precision note:** The mixed-precision fp32 GEMM path (enabled by default) casts inputs to `float16` when both have ≥ 1,000,000 elements. This yields near-2× speedup on large matmuls with a small precision loss. Set `JAX_METALLIB_GEMM_TRUE_F32=1` for exact fp32.

### Environment Variables (Advanced / Tuning)

The plugin exposes many tuning flags for power users and debugging. These are stable but change runtime behavior:

| Variable | Default | Description |
|----------|---------|-------------|
| `JAX_METALLIB_GEMM_NO_MPP` | — | Disable MPP fallback for GEMM operations |
| `JAX_METALLIB_GEMM_AUTOTUNE` | — | Enable GEMM autotuning |
| `JAX_METALLIB_GEMM_VARIANT` | — | Select GEMM implementation variant |
| `JAX_METALLIB_GEMM_F16_THRESHOLD` | `5120` | Threshold for f16 MPP fallback |
| `JAX_METALLIB_GEMM_AUTOTUNE_WARMUP` | — | GEMM autotune warmup iterations |
| `JAX_METALLIB_GEMM_AUTOTUNE_ITERS` | — | GEMM autotune benchmark iterations |
| `JAX_METALLIB_AUTOTUNE_CACHE_DIR` | `~/.cache/jax-metallib` | Directory for autotuning cache |
| `JAX_METALLIB_ELEMWISE_AUTOTUNE` | — | Enable elementwise operation autotuning |
| `JAX_METALLIB_ELEMWISE_AUTOTUNE_WARMUP` | — | Elementwise autotune warmup iterations |
| `JAX_METALLIB_ELEMWISE_AUTOTUNE_ITERS` | — | Elementwise autotune benchmark iterations |
| `JAX_METALLIB_MAX_WHILE_DEPTH` | `100` | Max recursion depth for `stablehlo.while` |
| `JAX_METALLIB_LAYERNORM_FUSION_MIN_ELEMENTS` | `131072` | Min elements for LayerNorm fusion |
| `JAX_METALLIB_SOFTMAX_FUSION_MIN_ELEMENTS` | `1024` | Min elements for Softmax fusion |
| `JAX_METALLIB_STRIDE_MIN_NUMEL` | `1048576` | Min elements for strided kernels |
| `JAX_METALLIB_REDUCE_NATIVE_MIN_ELEMENTS` | `32768` | Min elements before last-axis sum/max/min reduces run on the native kernel (13× faster than MPSGraph at 64K+) |
| `JAX_METALLIB_DISABLE_MIXED_INTRINSIC_NATIVE` | — | Force mixed graph+native plans back to all-graph (mixed plans are default-on; large reduces run natively) |
| `JAX_METALLIB_MAX_OPS_PER_CMDBUF` | `48` | Max ops per command buffer |
| `JAX_METALLIB_MAX_MB_PER_CMDBUF` | `48` | Max MB per command buffer |
| `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS` | — | Opt into native intrinsic kernels (disabled by default — known ordering bug under JAX 0.11, see known limitations) |
| `JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS` | — | Disable all native intrinsic kernels (LayerNorm, Softmax, GEMM, etc.); wins over the enable flag |
| `JAX_METALLIB_DISABLE_NATIVE_GEMM` | — | Disable native GEMM interception |
| `JAX_METALLIB_ENABLE_NATIVE_GEMM` | — | Force-enable native GEMM |
| `JAX_METALLIB_DISABLE_LAYERNORM_FUSION` | — | Disable native LayerNorm fusion |
| `JAX_METALLIB_DISABLE_SOFTMAX_FUSION` | — | Disable native Softmax fusion |
| `JAX_METALLIB_DISABLE_ELEMENTWISE_FUSION` | — | Disable elementwise JIT fusion |
| `JAX_METALLIB_DISABLE_ASYNC_COMPLETION` | — | Disable async completion events |
| `JAX_METALLIB_DISABLE_MPSGRAPH_EXECUTABLE` | — | Disable MPSGraph executable caching |
| `JAX_METALLIB_DISABLE_OUTPUT_BUFFER_POOL` | — | Disable output buffer pooling |
| `JAX_METALLIB_DISABLE_INTERMEDIATE_CACHE` | — | Disable intermediate tensor caching |
| `JAX_METALLIB_TRACE_PLAN` | — | Trace execution plan to stderr |
| `JAX_METALLIB_FORCE_INTRINSIC_GRAPH` | — | Force all intrinsics through graph path |
| `JAX_METALLIB_FORCE_INTRINSIC_NATIVE` | — | Force all intrinsics through native path |
| `JAX_SILICON_LIBRARY_PATH` | — | **Legacy alias** for `JAX_METALLIB_LIBRARY_PATH` |
| `JAX_MPS_LIBRARY_PATH` | — | **Legacy alias** for `JAX_METALLIB_LIBRARY_PATH` |

### Library Discovery

The plugin searches for the native library in this order:

1. `JAX_METALLIB_LIBRARY_PATH` environment variable
2. Package directory (editable install)
3. `<package>/lib/` (wheel install)
4. `build/*/lib/` (CMake build directory)
5. `/usr/local/lib/`, `/opt/homebrew/lib/`

## Performance

jax-metallib is benchmarked against both the JAX CPU backend and [jax-mps](https://github.com/tillahoffmann/jax-mps). The benchmark suite covers 43 competitive workloads, 109+ per-op micro-benchmarks, and scaling sweeps across tensor sizes.

### Key results vs jax-mps 0.10.6 (MLX backend)

- **Geo-mean speedup: 1.17×** (23 wins / 20 losses)
- **Strong on**: Conv2D (up to **3.3×**), softmax/layernorm (up to **2.2×**), GPT-2 scan-heavy models (**1.6–3.8×**)
- **Matmul**: fp32 1024² **5.00 TFLOPS** (94% of MLX), fp16 1024² **6.67 TFLOPS** (beats MLX's 6.20), fp32 2048² **9.68 TFLOPS** (beats MLX's 9.33)
- **Compile latency**: ~3–8 ms (5–10× faster than CPU backend)

### MPS vs CPU

- **Large workloads**: up to **5×** faster (matmul, sort, conv), up to **22×** for sort at scale
- **Small workloads**: per-op dispatch overhead (~250–550 µs) makes CPU faster below the ~4K×5K matmul crossover
- **fp16 advantage**: transcendals (sin/cos/log/exp) show 10–30× better ratios in fp16 vs fp32 because CPU emulates fp16 via fp32 promotion, while MPS has native fp16 SIMD

### Running benchmarks

```bash
# vs jax-mps (subprocess A/B — both register as platform 'mps')
uv run python benchmarks/vs_jax_mps.py

# vs CPU
uv run python benchmarks/vs_cpu.py

# Per-op micro-benchmarks (all 109+ ops)
uv run python benchmarks/bottlenecks.py

# Scaling sweeps
uv run python benchmarks/bench_ops.py

# Representative anchor workloads
uv run python -m benchmarks.bench run --case 'anchor\..*'
```

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
| Value correctness | CPU vs MPS output comparison for all 120 ops                |
| Gradient accuracy | `jax.grad` / `jax.value_and_grad` for differentiable ops    |
| Edge cases        | float16 precision, int64 large values, complex numbers    |
| Regressions       | Catastrophic cancellation (log1p), erf_inv range accuracy |
| Integration       | Flax NNX (Linear, Conv, LayerNorm, MultiHeadAttention)      |
| Integration       | NumPyro probabilistic programming models (optional)         |

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

## macOS Version Compatibility

- **macOS 13+ (Ventura):** Base support. All core ops work.
- **macOS 15+ (Sequoia):** Recommended. The plugin uses modern MPSGraph APIs (`randomTensorWithShape:descriptor:seed:`, `sliceTensor:starts:ends:strides:`, `concatTensors:dimension:`) that are only available on macOS 15+. On older systems, deprecated API paths may still function but are not actively tested.

## Known Limitations

- **`rng_bit_generator` with `UInt32` output** is not supported by MPSGraph and will raise a clean error. Use `jax.random.bits` (which lowers to a function-form Threefry call) or a different dtype.
- **Dispatch overhead floor:** ~250–550 µs per op launch means very small tensors (below ~10K elements) are often faster on CPU. This is a fundamental GPU driver characteristic, not a plugin bug.
- **Cholesky LAPACK fallback** is only active for `float32` inputs and `n ≤ 2048`. For larger inputs or non-float32 dtypes, the native MPS GPU kernel is always used.
- **Native intrinsic kernels are disabled by default on JAX 0.11.** JAX 0.11's eager dispatch exposed a `CFRelease(nil)` abort on eliminated output slots (fixed in 0.11.1.0 via placeholder buffers) and two fused-chain scheduling defects — a broadcast-ordering bug (fixed in 0.11.1.0 by splitting fused chains at externally-consumed intermediates) and a remaining slot-aliasing defect that yields wrong results for some MultivariateNormal log-prob gradients. All ops run through MPSGraph by default, which is correct. Opt into the native path with `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1` only for experiments.

## Python API

### Custom Metal Kernels

```python
from jax_plugins.silicon import metal
import jax.numpy as jnp

source = """
#include <metal_stdlib>
using namespace metal;
kernel void scale_add(device float* out [[buffer(0)]],
                      device const float* a [[buffer(1)]],
                      device const float* b [[buffer(2)]],
                      constant float& scale [[buffer(3)]],
                      uint gid [[thread_position_in_grid]]) {
    out[gid] = a[gid] * scale + b[gid];
}
"""

a = jnp.ones(1024)
b = jnp.ones(1024)
result = metal.execute(source, "scale_add", inputs=[a, b, 2.0], output_shapes=[(1024,)])
```

See `src/jax_plugins/silicon/metal.py` for the full functional, OOP, and decorator APIs.

### Quantization

```python
from jax_plugins.silicon.quantized import mx_quantize, mx_dequantize, quantized_matmul
import numpy as np

w = np.random.randn(512, 512).astype(np.float32)
w_q, scale = mx_quantize(w, mode="mxfp4")
w_dq = mx_dequantize(w_q, scale, mode="mxfp4")
```

Supported formats: MXFP4, MXFP8, NVFP4, and affine (GPTQ/AWQ-style) quantization with fused Metal GEMM kernels.

## Repository Layout

```md
src/
  jax_plugins/silicon/          Python entrypoint — plugin registration & library discovery
    __init__.py                   Plugin initialization & library discovery
    metal.py                      Custom Metal kernel API (functional, OOP, decorator)
    quantized.py                  MX / affine quantization with fused Metal GEMM
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
    ops/                        StableHLO op handlers (120 registrations)
      registry.h                  Op registry, handler types, macros
      unary_ops.mm                50 unary operations
      binary_ops.mm               15 binary operations
      shape_ops.mm                19 shape & indexing operations
      reduction_ops.mm            6 reduction operations
      convolution_ops.mm          General dilated convolution
      linalg_ops.mm               Cholesky, triangular solve (native MPS + LAPACK fallback)
      bitwise_ops.mm              8 bitwise operations
      sort_ops.mm                 Sort, top-k
      fft_ops.mm                  FFT / RFFT / IFFT / IRFFT
      control_flow_ops.mm         While, case, custom_call, return
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
  test_metal_kernel.py          Custom kernel API tests
  test_quantized.py             MX quantization tests
  configs/                      Per-category test configurations (15 modules)
benchmarks/
  bench.py                      Benchmark harness (list/run, JSONL output)
  bottlenecks.py                Per-op micro-benchmarks
  vs_jax_mps.py                 Competitive benchmarks vs jax-mps
  vs_cpu.py                     Competitive benchmarks vs CPU
  anchors.py                    Representative anchor workloads
  bench_ops.py                  Scaling sweeps
  linalg.py                     Linear algebra benchmarks
scripts/
  setup_deps.sh                 One-time native dependency bootstrap
  memory_monitor.sh             RSS sampling utility
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
