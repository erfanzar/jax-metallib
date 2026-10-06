# jax-metallib

A [PJRT](https://openxla.org/xla/pjrt_integration) plugin for running
[JAX](https://github.com/jax-ml/jax) on Apple Silicon GPUs. The backend name is
`mps`. It accepts StableHLO from JAX and executes it through MPSGraph, Metal
compute kernels, and selected CPU fallbacks.

The package is experimental. Support depends on the operation, dtype, shape,
and JAX lowering; it is not a complete replacement for the CPU backend.
This README describes the current source tree.

## Requirements

| Requirement | Supported configuration |
| --- | --- |
| Hardware | Apple Silicon Mac |
| Runtime OS | macOS 15.0 or later |
| Python | 3.12 or later; configured wheel builds target CPython 3.12 and 3.13 |
| JAX and jaxlib | `>=0.11.0,<0.12` |
| Source-build tools | Xcode 26 or later with the macOS SDK, CMake 3.20+, Ninja, and a C++17 compiler |

The runtime minimum and build SDK are different requirements. CMake sets a
macOS 15.0 deployment target, but the source also references Metal declarations
that require the newer SDK. macOS 13 and 14 are not supported.

## Install and select the backend

```bash
python -m pip install jax-metallib
JAX_PLATFORMS=mps python -c 'import jax; print(jax.devices())'
```

When initialization succeeds and `JAX_PLATFORMS` is unset, the plugin's
registration priority normally makes MPS the default backend. Set
`JAX_PLATFORMS=mps` explicitly to require it rather than silently using CPU if
initialization fails. Set the variable before importing JAX.

For programs that need both backends:

```bash
JAX_PLATFORMS=cpu,mps python your_program.py
```

The first platform in that list is the default. Select MPS for particular work
with `jax.default_device(jax.devices("mps")[0])` or move arrays explicitly with
`jax.device_put`.

### Example

Run this with `JAX_PLATFORMS=mps`:

```python
import jax
import jax.numpy as jnp

print(jax.devices())
x = jnp.ones((128, 128), dtype=jnp.float32)
y = jax.jit(lambda a: a @ a)(x)
grad = jax.jit(jax.grad(lambda a: jnp.sum(jnp.tanh(a))))(x)
y.block_until_ready()
grad.block_until_ready()
```

JAX differentiation works by lowering derivative computations to supported
operations. That does not give arbitrary Metal source an automatic derivative
or batching rule.

## Build from source

Install [uv](https://docs.astral.sh/uv/) and Xcode, select the Xcode toolchain,
then run:

```bash
brew install cmake ninja
git clone https://github.com/erfanzar/jax-metallib.git
cd jax-metallib
./scripts/setup_deps.sh
uv sync --all-groups
uv pip install -e .
```

The setup script builds static LLVM/MLIR, StableHLO, Abseil, and Protobuf
dependencies under `~/.local/jax-metallib-deps` and installs the PJRT C API
header. Its default source pins are:

| Dependency | Revision or version |
| --- | --- |
| XLA / PJRT header | `dcf304bc5dca1932b99f740b911dbd73631a1a69` |
| LLVM / MLIR | `6d1ace547c9f9ecf7df9ab87e2db74b5bd3f199d` |
| StableHLO | `7b1b15781ccbd770f50c7eef4b0c3e03834649fd` |
| Abseil | `20250127.0` |
| Protobuf | `29.3` |

These pins are selected for jaxlib 0.11.1. The PJRT header must match the ABI
expected by jaxlib; the package's dependency range alone is not a compatibility
test for every permitted version.

CMake can invoke the setup script automatically when the PJRT header is
missing and `JAX_METALLIB_AUTO_SETUP_DEPS` is enabled. It does not check whether
all installed dependencies are current. The script skips existing MLIR and
StableHLO installations based on file presence. After changing dependency
pins, use a fresh prefix or `./scripts/setup_deps.sh --force`; `--force` removes
installed dependency markers and build directories before rebuilding.

To use dependencies you have installed yourself:

```bash
CMAKE_ARGS="-DJAX_METALLIB_AUTO_SETUP_DEPS=OFF" uv pip install -e .
```

Use `JAX_METALLIB_DEPS_PREFIX` as a CMake setting, or the setup script's
`--prefix` option, to change the dependency location.

## Operation support and limits

Handlers cover the following families. A registered name is not a guarantee
that every dtype, shape, or custom computation is supported.

| Family | Examples |
| --- | --- |
| Elementwise | Arithmetic, comparisons, trigonometric functions, exponentials, logs, rounding, special functions, bitwise operations |
| Shape and indexing | Broadcast, reshape, transpose, slice, dynamic slice/update, gather, scatter, padding, conversion, bitcast |
| Reductions | Sum, product, extrema, integer/bool and/or/xor, argmin/argmax, window reductions, batch normalization |
| Convolution | Floating-point 1D/2D/3D convolution; integer 1D/2D convolution |
| Linear algebra | Dot products, Cholesky, triangular solve, QR, eig/eigh, SVD, Schur, Hessenberg and tridiagonal reduction |
| FFT | FFT, IFFT, RFFT, IRFFT |
| Sorting | Sort, top-k, and an exact selection lowering for the `ApproxTopK` custom call |
| Random | Python ThreeFry/Philox lowerings and direct StableHLO random handlers with separate restrictions |
| Control flow | While/counting loops, case, function calls, elementwise map, optimization barriers |
| Collectives | Single-replica identity behavior; no inter-device communication |

Important restrictions:

- **One device, one process.** The client exposes the system-default Metal
  device. Multi-replica and multi-partition execution are not implemented.
- **Static shapes.** Runtime slice indices are supported, but general runtime
  shape polymorphism is not. Dynamic reshape and broadcast require constant
  output dimensions. `set_dimension_size` does not resize storage.
- **Dtypes vary by operation.** Do not assume `jax_enable_x64` enables full
  float64 or complex128 execution. Some integer operations support only
  8/16/32-bit values, even though buffers and constants can represent int64.
- **Scatter and reducers have recognized bodies.** Ordinary scatter supports
  selected set/arithmetic/bool update forms, not arbitrary `x.at[i].apply(f)`.
  General user-defined sort comparators and reduction regions are not fully
  interpreted.
- **Convolution grouping is limited.** Floating forward batch grouping is
  emulated with at most four convolution calls. Integer batch grouping,
  integer 3D convolution, and transposed batch grouping are rejected.
- **Linear algebra has size and algorithm limits.** Real QR is float32-only
  with at most 128 Householder steps. Real Cholesky/triangular-solve graph
  fallbacks inside control flow are limited to matrix order 256. The Python
  eig/eigh/SVD lowerings use float32 internal arithmetic and bounded iterative
  algorithms. Partial `eigh` subsets and Schur eigenvalue sorting are not
  supported.
- **Random lowerings are not interchangeable.** Eligible public
  `lax.rng_bit_generator` calls use Python ThreeFry/Philox expansion, including
  `uint32` output and state updates. This requires a `uint32[4]` key, static
  output shape, and fewer than `2**32` output elements. Direct/unexpanded
  StableHLO `rng_bit_generator` still rejects `uint32`, ignores the input
  state, and returns a fixed zero state; it is not a stateful algorithm
  replacement. Disabling the Python lowering exposes those restrictions.
- **Numerical differences remain.** Tests use per-operation tolerances and
  targeted bitwise checks, not a blanket guarantee of equality with CPU.
  Some special functions use approximations. Eligible float32 matmuls narrow
  inputs by default; see the precision setting below.

Unsupported operations and custom-call targets report diagnostics with the
operation name and, where available, shape/type information. Native custom
Metal kernels have additional restrictions listed below.

## Execution paths

JAX lowers a program to StableHLO. The PJRT layer passes it to the parser,
which deserializes the module and inlines functions. The execution planner
then creates an ordered sequence of:

1. **MPSGraph segments:** operation handlers build graphs that are compiled
   and encoded through MPSGraph.
2. **Native steps:** Metal or MPS kernels operate on `MTLBuffer` objects.
   Eligible reductions, sorts, and softmax patterns have native routes;
   `metal.compile` calls use a separate native custom-call path.
3. **CPU steps:** selected small GEMMs, sorts, and Cholesky decompositions use
   Accelerate BLAS/vDSP/LAPACK routines.

Eligible native elementwise chains can be combined into generated MSL kernels.
This does not fuse arbitrary user-written Metal kernels.

The optional intrinsic-kernel family, including LayerNorm fusion and additional
GEMM/elementwise routing, is disabled by default. Enabling it is experimental.
That switch does not disable every native route: standalone reductions,
sorting, softmax, registered native handlers, and custom Metal calls have
separate eligibility rules.

Outputs carry completion events. Use `block_until_ready()` when measuring
execution time or when the host needs completion; host readback also waits.
Buffer release waits for GPU completion before storage can be reused.

Executable serialization saves a StableHLO portable artifact. Loading it
parses and recompiles the program; it does not restore compiled Metal pipelines
or MPSGraph executable state.

## Custom Metal kernels

### Native JAX-array API

`metal.compile` returns a callable whose outputs are JAX arrays and whose calls
can appear inside `jax.jit`. Source compilation happens when the backend
compiles the program, not when the Python callable is created. Inputs and
outputs use the plugin's Metal buffers without the legacy NumPy/PyObjC bridge.
This API is independent of the optional intrinsic-kernel flag.

```python
import jax
import jax.numpy as jnp
import numpy as np
from jax_plugins.silicon import metal

source = """
#include <metal_stdlib>
using namespace metal;
kernel void scale_add(device const float* a [[buffer(0)]],
                      device const float* b [[buffer(1)]],
                      device float* out [[buffer(2)]],
                      constant float& scale [[buffer(3)]],
                      constant uint& n [[buffer(4)]],
                      uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = a[gid] * scale + b[gid];
}
"""
scale_add = metal.compile(source, entry_point="scale_add")

@jax.jit
def compute(a, b):
    return scale_add(
        a, b,
        out=jax.ShapeDtypeStruct(a.shape, a.dtype),
        threads=a.size,
        threadgroup=256,
        constants={3: np.float32(2), 4: np.uint32(a.size)},
    ) + 1

with jax.default_device(jax.devices("mps")[0]):
    a = jnp.ones(1024, dtype=jnp.float32)
    result = compute(a, a)
    result.block_until_ready()
    print(result[0])  # 4.0
```

Binding and launch rules:

- Inputs occupy buffer indices `0..n-1`; outputs follow them. Constants use
  explicitly chosen free indices below 31.
- `out` is one `jax.ShapeDtypeStruct` or a nonempty sequence of output specs.
  A single spec returns an array; a sequence returns a tuple.
- `threads` and `threadgroup` accept an integer or a three-integer tuple.
  `threads` counts individual threads, not threadgroups. Threadgroup dimensions
  must be positive; a zero thread dimension skips dispatch.
- Output shapes, dtypes, launch geometry, and constants are static compilation
  parameters. Python integers/floats are packed as uint32/float32. Use NumPy
  scalars for explicit supported types, or bytes for a custom layout. Each
  constant must contain 1–4096 bytes.
- Array bindings must be contiguous row-major and on MPS. Supported dtypes are
  float32, float16, bool, and signed/unsigned 8/16/32-bit integers. Bfloat16,
  float64, complex, and 64-bit integers are not supported by this API.
- There are no automatic differentiation or batching rules for arbitrary MSL.
  `grad`/`vmap` reject it unless you supply appropriate rules. Native custom
  calls nested inside while/scan/conditional regions are not supported.

A 128-entry Python preparation cache reuses validated metadata and eager JIT
wrappers, not arrays or results. Eager calls revalidate constants by their
current value; an enclosing JIT captures static values when tracing. Native
pipeline caching is in-process, keyed by device/source/entry point, with at
most 64 weak entries. It is not a persistent compiled-program cache and does
not remove first-use JAX/PJRT compilation or per-launch overhead.

**Metal source is trusted, unsandboxed GPU code.** Bounds-check accesses, keep
inputs read-only, and write every output element. Reflection validates binding
interfaces, not every indexed access or complete output initialization. Do not
extract or mutate raw JAX buffer pointers.

### Legacy host-copy API

`metal.execute`, `MetalKernel`, and `@metal.kernel` use PyObjC. They copy inputs
through NumPy, wait for the Metal command buffer, and copy outputs back into JAX
arrays. They are eager interfaces, not a path for dynamic inputs inside
`jax.jit`. Their `grid` counts **threadgroups**, unlike native `threads`.

The native API does not use PyObjC, but package metadata still includes PyObjC
for these legacy interfaces and quantized GEMM wrappers. See
[`metal.py`](src/jax_plugins/silicon/metal.py) and
[`_native_metal.py`](src/jax_plugins/silicon/_native_metal.py) for the interfaces.

## Quantization

[`quantized.py`](src/jax_plugins/silicon/quantized.py) provides host-side
quantization/dequantization helpers and quantized matrix multiplication.
Weights are stored as `W[N, K]`; multiplication computes `x @ W.T`.

| Format | Weight encoding | Scale encoding | Group size |
| --- | --- | --- | --- |
| MXFP4 | Packed E2M1, two values per byte | E8M0 | 32 |
| MXFP8 | E4M3, one value per byte | E8M0 | 32 |
| NVFP4 | Packed E2M1, two values per byte | E4M3 | 16 |
| Affine | 2/3/4/8-bit integers | Per-group scale and bias (`scale * q + bias`) | Configurable; default 64 |

`K` must be divisible by the group size. For affine GPU GEMM, use group sizes
divisible by 32, such as 32 or 64. The host helpers accept other sizes, but the
GPU dispatcher does not check them and the kernel can cross group boundaries,
producing incorrect results (for example, with multiple groups of size 16).
FP4 dequantization needs the original shape because packed bytes contain two
values each.

```python
import numpy as np
from jax_plugins.silicon.quantized import (
    mx_dequantize,
    mx_quantize,
    quantized_matmul,
)

rng = np.random.default_rng(0)
w = rng.standard_normal((128, 256)).astype(np.float32)
x = rng.standard_normal((8, 256)).astype(np.float32)
packed, scales = mx_quantize(w, mode="mxfp4")
restored = mx_dequantize(packed, scales, mode="mxfp4", shape=w.shape)
y = quantized_matmul(x, packed, scales, mode="mxfp4", K=w.shape[-1])
```

Quantization and dequantization convert through host NumPy; they are not
operations for differentiating or JIT-tracing dynamic weights. Quantized GPU
GEMM requires PyObjC Metal and float32/float16 activations. Affine GPU kernels
cover 4/8-bit weights; 2/3-bit weights use host dequantization followed by JAX
matmul. That fallback matmul follows JAX placement rather than being pinned to
CPU. GPU wrappers use the synchronous host-copy `metal.execute` interface,
not `metal.compile`. GPU output is first stored in the activation dtype;
requesting a wider `out_dtype` casts afterward.

## Configuration

Set runtime flags before JAX initializes or compiles the relevant program;
some values are cached for the process. This is a selection of controls, not a
stable specification of every internal tuning flag.

| Variable | Default | Effect |
| --- | --- | --- |
| `JAX_PLATFORMS` | JAX backend selection | `mps` requires MPS; `cpu,mps` makes both available with CPU first |
| `JAX_METALLIB_LIBRARY_PATH` | Automatic discovery | Path to `libpjrt_plugin_silicon.dylib` |
| `JAX_METALLIB_GEMM_FAST_F32` | `0` | `1` opts into float16 input narrowing for large float32 GEMMs at `DEFAULT` precision |
| `JAX_METALLIB_COMPLETION_TIMEOUT_MS` | `30000` | Positive timeout for caller-facing GPU waits; timeout does not cancel GPU work |
| `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS` | Off | Opt into the experimental intrinsic-kernel family |
| `JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS` | Off | Disable that family; overrides its enable flag, not every native route |
| `JAX_METALLIB_DISABLE_ELEMENTWISE_FUSION` | Off | Disable native elementwise-chain fusion |
| `JAX_METALLIB_JACOBI_EARLY_EXIT` | `1` | `0` runs the eigh/SVD Jacobi paths for a fixed 12 sweeps |
| `JAX_METALLIB_LOOP_UNROLL` | `8` | Counted-loop unroll limit, clamped to 1–32; `1` disables it |
| `JAX_METALLIB_AUTOTUNE_CACHE_DIR` | `~/.cache/jax-silicon` | Directory for autotune choices, not compiled custom Metal pipelines |
| `JAX_METALLIB_TRACE_PLAN` | Off | Print the execution plan to stderr |

**Float32 GEMM precision:** float32 matmuls keep float32 inputs by default.
With `JAX_METALLIB_GEMM_FAST_F32=1`, the graph dot handler casts both inputs to
float16 when each contains at least 500,000 elements and
`batch * M * N * K >= 100,000,000`, then casts the output to float32. That mode
has about 3e-4 relative error on a 1024×1024 matmul and returns inf when an
operand exceeds the float16 range (65504). It never applies to a dot whose
`precision` is above `DEFAULT` (for example `precision=HIGHEST`).
`JAX_METALLIB_GEMM_TRUE_F32` is no longer read.

`MPS_LOG_LEVEL` is a **build-time** setting: `0` error, `1` warning, `2` info,
`3` debug. For example, rebuild with `MPS_LOG_LEVEL=3 uv pip install -e .`;
setting it only when launching Python does not change compiled logging.

Library discovery uses the first set override in this order:
`JAX_METALLIB_LIBRARY_PATH`, `JAX_SILICON_LIBRARY_PATH`, `JAX_MPS_LIBRARY_PATH`.
If that value is invalid, it warns and proceeds to filesystem discovery rather
than trying the remaining aliases. Without a usable override, it searches the
package directory, its `lib/` subdirectory, `build/*/lib/`, `/usr/local/lib/`, and
`/opt/homebrew/lib/`.

## Tests and local checks

```bash
uv run pytest tests/test_ops.py
uv run pytest tests/test_native_metal_kernel.py tests/test_native_metal_cache.py
uv run pytest tests/test_quantized.py
uv run pytest
```

The parameterized operation suite checks selected values and gradients;
dedicated regression files cover integer and half-precision behavior, NaNs,
signed zeros, indexing, control flow, linear algebra, PJRT ownership, and
completion handling. References include CPU, NumPy, and decomposition/property
checks. Passing one category does not establish support for every JAX program.

`JAX_TEST_MODE=compare` is the default for `test_ops.py`; `mps` and `cpu` select
its operation-test mode. It is **not a whole-suite backend switch**. Framework
cases still select MPS, and many standalone regressions explicitly use both
backends or subprocess settings. Flax cases cover NNX Linear, Conv, Embed, and
BatchNorm. Optional NumPyro cases exercise distribution log-probabilities and
gradients, not end-to-end probabilistic inference.

Some native ABI/ownership tests require an explicit candidate library and
matching PJRT headers and otherwise skip; read their module setup before using
them as release checks.

```bash
uv run ruff check src tests benchmarks
uv run ruff format --check src tests benchmarks
```

These are manual checks. The repository has no pre-commit configuration.
Hosted CI runs changed-file/changed-line lint, arm64 wheel builds, and wheel
packaging checks. Its Metal-device probe is allowed to fail; it is not a GPU
numerical-test gate. The implementation of these checks is in
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) and the
[`scripts/`](scripts/) directory. Existing lint/format debt may fail a full-tree
check even when the changed-file gate passes.

## Benchmarks

```bash
uv run python -m benchmarks.bench list
uv run python -m benchmarks.bench run --case 'anchor\..*' --platform mps --json-out results.jsonl
uv run python -m benchmarks.bench run --case 'anchor\..*' --platform cpu
uv run python benchmarks/vs_cpu.py
uv run python benchmarks/bottlenecks.py
```

The main harness reports lowering, compilation, first execution, and
synchronized steady-state timing separately. Its defaults are 10 warmups,
50 timed iterations, and 3 repeats; steady-state results use the median of
repeat medians. JSONL output is appended. The bottleneck script generates
microbenchmarks for a subset of registered operations and records skips/errors.

A jax-mps comparison needs a separate environment because both plugins register
as `mps`:

```bash
uv run python benchmarks/vs_jax_mps.py --jax-mps-python /path/to/jax-mps-venv/bin/python
```

These are synthetic workloads, not an application-level speedup guarantee.
Report hardware, OS, JAX/jaxlib and plugin versions, comparator revision,
dtypes, shapes, flags, and timing settings with results. This README does not
quote benchmark numbers without a matching measurement artifact.

## Source layout

```text
src/jax_plugins/silicon/
  __init__.py              Plugin registration and library discovery
  _native_metal.py          Native custom-call lowering and launch preparation
  metal.py                  Public custom-kernel and legacy host-copy APIs
  quantized.py              MX/affine packing and quantized GEMM wrappers
  linalg_lowerings.py       JAX implementations of extra linear-algebra lowerings
  random_lowerings.py       ThreeFry/Philox expansion and ThreeFry round cuts
  sort_lowerings.py         Sort-axis movement before lowering
src/pjrt_plugin/
  api/                      PJRT C API callbacks and handle ownership
  core/                     Devices, buffers, parser, planner, execution, events
  ops/                      StableHLO/CHLO handlers and registered custom calls
  runtime/                  Built-in MSL sources and custom-kernel compilation
  metal_shaders/            Steel GEMM shader
  proto/                    Device-assignment protobuf
scripts/                    Native dependency setup and CI checks
benchmarks/                 Timing harnesses and comparison scripts
tests/                      Operation configurations and regression tests
```

## Contributing and license

Rebuild the native library after C++/Objective-C++ changes with
`uv pip install -e .`, then run the relevant tests on a Metal-capable Mac.
The development overview is in [CONTRIBUTING.md](CONTRIBUTING.md); its pre-commit
instructions do not correspond to a configuration in this source tree.

The GEMM shader uses the MLX Steel BlockLoader/BlockMMA design.

Copyright 2026 Erfan Zare Chavoshi (@erfanzar).
[Apache-2.0](LICENSE).
