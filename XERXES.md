# XERXES.md — jax-metallib

> Project root: `/Users/erfan/Documents/Projects/jax-metallib`
> Initialized: 2026-06-23
> Purpose: PJRT plugin enabling JAX on Apple Metal GPUs (M-series)

---

## What this project is

jax-metallib is a native PJRT (Pluggable Device Runtime) plugin that lets JAX execute on Apple Metal GPUs by compiling StableHLO IR to Metal compute kernels via MPSGraph. It is a dual-language project: Python (plugin registration, custom kernel API, quantization) + C++17/Obj-C++17 (PJRT C API, MPS backend, op lowering, Metal runtime).

- **120 StableHLO ops** registered across 15 op category files
- **3 execution paths**: MPSGraph compilation, native Metal kernels for perf-critical ops, JIT fusion of elementwise chains into custom MSL kernels
- **Platform**: macOS 13.0+ (15+ recommended), Apple Silicon only
- **JAX version**: 0.11.x

---

## Repository layout

```
├── src/jax_plugins/silicon/          # Python plugin layer
│   ├── __init__.py                   # Registers "mps" backend with JAX
│   ├── metal.py                      # Custom Metal kernel API (execute, MetalKernel, @kernel)
│   └── quantized.py                  # MX microscaling + affine quantization
├── src/pjrt_plugin/
│   ├── api/                          # PJRT C API wrappers (8 .cc files)
│   ├── core/                         # MPS backend core
│   │   ├── mps_client.h/.mm          # Device, command queue, compilation
│   │   ├── mps_device.h/.mm          # Thin device abstraction
│   │   ├── mps_buffer.h/.mm          # MTLBuffer wrapper with dtype/shape/strides
│   │   ├── mps_executable.h/.mm      # Execution engine (~9.7K lines, largest file)
│   │   ├── stablehlo_parser.h/.mm    # StableHLO → MLIR parsing
│   │   ├── type_utils.h/.mm          # MLIR ↔ MPS type conversions
│   │   ├── completion_event.h        # GPU sync fence (spin-then-condvar)
│   │   └── logging.h, issue_url.h, pjrt_types.h
│   ├── ops/                          # 15 op files + registry.h
│   │   ├── registry.h                # OpRegistry, registration macros
│   │   ├── unary_ops.mm (50 ops)     # REGISTER_MLIR_UNARY_OP macros
│   │   ├── binary_ops.mm (15 ops)    # REGISTER_MLIR_BINARY_OP macros
│   │   └── reduction, shape, conv, linalg, sort, fft, random, bitwise,
│   │       collective, control_flow, higher_order, tensor_creation
│   ├── runtime/                      # Native Metal kernels
│   │   └── metal_kernels.h/.mm       # Softmax, LayerNorm, GEMM, Cholesky, etc.
│   ├── metal_shaders/
│   │   └── steel_gemm.metal          # Custom GEMM shader
│   └── proto/
│       └── device_assignment.proto
├── tests/
│   ├── test_ops.py                   # Main parameterized op suite (value + grad)
│   ├── test_metal_kernel.py          # Custom kernel API tests
│   ├── test_quantized.py             # Quantization tests
│   ├── conftest.py                   # xfail-match validation hook
│   └── configs/                      # 15 test config modules
│       ├── util.py                   # OperationTestConfig class
│       └── unary.py, binary.py, conv.py, linalg.py, etc.
├── benchmarks/
│   ├── bench.py                      # CLI harness (list, run, JSONL)
│   ├── vs_jax_mps.py                 # Head-to-head vs jax-mps
│   ├── vs_cpu.py                     # Head-to-head vs CPU
│   ├── bottlenecks.py                # Per-op micro-benchmarks (109 ops)
│   ├── anchors.py, cases.py, linalg.py
│   └── util.py
├── scripts/
│   ├── setup_deps.sh                 # Bootstrap LLVM/MLIR/StableHLO (~30 min)
│   └── memory_monitor.sh
├── CMakeLists.txt                    # Native build (scikit-build-core backend)
├── pyproject.toml                    # Python package, deps, ruff config
├── .clang-format                     # LLVM style, 110 columns
├── .clang-tidy                       # Warnings-as-errors, src/pjrt_plugin/ scope
└── .claude/settings.local.json         # Bash permissions for this agent
```

---

## Build, test, and development commands

### One-time setup
```bash
brew install cmake ninja
./scripts/setup_deps.sh          # ~30 min; installs to ~/.local/jax-metallib-deps
```

### Install and build
```bash
uv sync --all-groups
uv pip install -e .              # Rebuilds libpjrt_plugin_silicon.dylib
```

### Test
```bash
uv run pytest                    # Full suite, default compare mode (CPU vs MPS)
JAX_TEST_MODE=mps uv run pytest  # MPS only (faster)
JAX_TEST_MODE=cpu uv run pytest  # CPU only
uv run pytest -k "unary"         # Filter by category
```

### Benchmark
```bash
uv run python -m benchmarks.bench list
uv run python -m benchmarks.bench run --case 'anchor\..*' --platform mps
uv run python benchmarks/vs_jax_mps.py      # vs jax-mps
uv run python benchmarks/vs_cpu.py         # vs CPU
uv run python benchmarks/bottlenecks.py    # Per-op profiling
```

### Format / lint (manual — no pre-commit config exists)
```bash
ruff check .                     # Python lint
ruff format .                    # Python format
clang-format -i src/pjrt_plugin/**/*.mm src/pjrt_plugin/**/*.cc src/pjrt_plugin/**/*.h
clang-tidy src/pjrt_plugin/**/*.mm src/pjrt_plugin/**/*.cc  # requires compile_commands.json
```

---

## Architecture & runtime flow

**4-tier stack:**
1. **Python** (`src/jax_plugins/silicon/`) — plugin registration, custom Metal API, quantization
2. **PJRT C API** (`src/pjrt_plugin/api/`) — standard XLA plugin interface vtable
3. **Core MPS Backend** (`src/pjrt_plugin/core/`) — `MpsClient`, `MpsDevice`, `MpsBuffer`, `MpsExecutable`
4. **Ops + Runtime** (`src/pjrt_plugin/ops/`, `runtime/`) — 120 registered handlers, native Metal kernels

**Execution flow:**
1. JAX compiles Python to StableHLO IR (MLIR bytecode)
2. `MpsExecutable` parses the module via `stablehlo_parser`
3. `BuildExecutionPlan()` walks ops, classifies into `GraphStep` (MPSGraph) or `NativeStep` (Metal kernel)
4. Fusion passes merge elementwise chains, LayerNorm, Softmax, ReduceMean into fused native kernels
5. `Execute()` binds input buffers, runs graph segments via `MPSGraphExecutable`, native segments via `MTLComputeCommandEncoder`, and returns `CompletionEvent`

**Dual-path dispatch:**
- **GRAPH** handlers build `MPSGraphTensor` DAGs via `registry.h` macros
- **NATIVE** handlers bypass MPSGraph and directly encode Metal compute kernels for performance-critical ops (Cholesky, triangular solve, softmax, LayerNorm, GEMM, sort)

---

## Coding conventions

### C++ / Obj-C++
- **Formatter**: `.clang-format` — LLVM style, 110 columns, indent 4, attach braces, left-aligned pointers
- **Linter**: `.clang-tidy` — bugprone, clang-analyzer, modernize, performance, readability; warnings-as-errors; scoped to `src/pjrt_plugin/`
- **Language**: Obj-C++ (`.mm`) for Metal/MPSGraph files; plain C++ (`.cc`) for PJRT API wrappers
- **Namespace**: `jax_metallib` everywhere
- **Copyright**: Apache 2.0, `Copyright 2026 Erfan Zare Chavoshi (@erfanzar)` on every file

### Python
- **Formatter**: Black, 121 columns, target `py311`, preview mode
- **Linter**: Ruff, 121 columns, target `py311`; rules `A,B,E,F,I,NPY,RUF,UP,W`
- **Per-file ignores**: `__init__.py` ignores `E402`, `F401`

### Op registration pattern
```objc
// Simple unary (one-liner in ops/unary_ops.mm)
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);

// Custom handler
static ProcessResult HandleMyOp(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    // ... build graph ...
    return Result(ctx, output, "my_op");
}
REGISTER_MPS_OP("stablehlo.my_op", HandleMyOp);
```

### Test config pattern
```python
# In tests/configs/unary.py
OperationTestConfig(jnp.cos, lambda rng: complex_standard_normal(rng, (17,), is_complex=False))
```

---

## Important environment variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `JAX_PLATFORMS` | — | Set to `mps` to select Metal backend |
| `JAX_METALLIB_LIBRARY_PATH` | auto | Override dylib search path |
| `JAX_TEST_MODE` | `compare` | `compare` / `mps` / `cpu` |
| `MPS_LOG_LEVEL` | `1` | `0`=error, `1`=warn, `2`=info, `3`=debug |
| `JAX_METALLIB_GEMM_TRUE_F32` | `0` | `1` disables fp32→fp16 mixed-precision GEMM |
| `JAX_METALLIB_TRACE_PLAN` | — | Trace execution plan to stderr |
| `JAX_METALLIB_COMPLETION_TIMEOUT_MS` | `30000` | Async completion timeout |
| `JAX_METALLIB_AUTOTUNE_CACHE_DIR` | `~/.cache/jax-metallib` | Autotune cache directory |

Full list: 40+ tuning flags in `src/pjrt_plugin/core/mps_executable.mm` and `README.md`.

---

## Agent workflow rules

1. **C++ changes require rebuild**: Any edit to `.mm`, `.cc`, or `.h` in `src/pjrt_plugin/` requires `uv pip install -e .` before testing.
2. **Clear caches after C++ changes**: Delete `~/.cache/jax-silicon/autotune_cache.bin` and `~/.cache/jax-metallib/autotune_cache.bin` after changes that affect GEMM or elementwise kernels. The autotune cache is unversioned binary and can pick stale variant decisions.
3. **Use `JAX_TEST_MODE=mps` for fast iteration**: The default `compare` mode runs every test on both CPU and MPS. For quick feedback, use `JAX_TEST_MODE=mps`.
4. **Avoid eager gradients in compare mode**: `test_op_grad` is skipped in `compare` mode because eager gradients deadlock the Metal command queue after many prior dispatches. JIT gradients are the tested path.
5. **No pre-commit hooks**: Despite `CONTRIBUTING.md` claiming pre-commit hooks, there is no `.pre-commit-config.yaml`. Run formatting and tests manually.
6. **Subprocess benchmarking**: `vs_jax_mps.py` and `vs_cpu.py` spawn separate Python processes because `jax-mps` and `jax-metallib` both register as `mps` platform and cannot coexist in one process.
7. **Platform guard**: This only works on macOS Apple Silicon. Non-Darwin platforms raise `SiliconPluginError`.
8. **Shardy disabled**: The plugin auto-sets `jax_use_shardy_partitioner = False` because the StableHLO parser does not support `sdy` dialect ops.

---

## Project-specific caveats

- **mps_executable.mm is ~9.7K lines**: This is the heart of the backend. Changes here affect execution plan building, dispatch, native kernel baking, and memory management. Read carefully before editing.
- **Per-op dispatch overhead ~250–550 µs**: Small tensors (<10K elements) are often faster on CPU. The GPU wins at scale (matmul >4K×5K, sort >100K, conv >1K channels).
- **macOS 15+ strongly recommended**: Some MPSGraph APIs used on 13–14 are deprecated and replaced with modern equivalents on 15+. The `randomTensorWithShape:descriptor:seed:` and `sliceTensor:starts:ends:strides:` APIs require macOS 15+.
- **Mixed-precision GEMM**: Large fp32 matmuls (≥500K elements, ≥100M FLOPs) auto-cast to fp16 for speed. Set `JAX_METALLIB_GEMM_TRUE_F32=1` for exact precision.
- **Native intrinsic kernels are disabled by default on JAX 0.11** (`JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1` to opt in): eager dispatch exposed a `CFRelease(nil)` abort on eliminated output slots (fixed in 0.11.1.0) and fused-chain scheduling defects — the broadcast-ordering wrong-results bug is fixed (chains now split at externally-consumed intermediates), but a slot-aliasing defect remains for some MultivariateNormal log-prob gradients under the opt-in flag. `test_native_intrinsic_ordering_bug` guards the ordering pattern.
- **Bumping JAX**: pins come from the JAX release's `MODULE.bazel` `archive_override` (XLA commit), XLA's `third_party/stablehlo/workspace.bzl` (StableHLO commit), and that StableHLO commit's `build_tools/llvm_version.txt` (LLVM commit). After changing pins, `setup_deps.sh --force` is mandatory — stale installed headers shadow fresh builds.
- **Metal has no `atomic_half`**: Scatter-add uses non-atomic read-add-write; duplicate indices may lose updates.
- **UInt32 `rng_bit_generator` unsupported**: MPSGraph asserts on UInt32 random dtype. Use `jax.random.bits` instead.
- **Build fragility**: LLVM/MLIR/StableHLO are pinned to exact commits. Bumping JAX versions requires matching StableHLO/XLA/LLVM commits.
- **No CI/CD**: No `.github/workflows/`. All quality gates are local-only. Do not break the build.
- **`.gitignore` gaps**: `.DS_Store`, `env.py`, and `.claude/` are in the working tree but not fully ignored. Be careful not to commit accidental files.

---

## Skills available

See `.agents/skills/` for repository-specific skills:
- `add-op` — Adding a new StableHLO operation
- `rebuild-test` — Rebuilding and testing after C++ changes
- `benchmark-vs-jax-mps` — Running competitive benchmarks
- `debug-trace` — Debugging with trace plan and cache clearing
- `bottleneck-profile` — Running the per-op bottleneck profiler

---

*Last updated: 2026-06-23 by Xerxes setup lead.*
