# Beat jax-mps 0.11.0 Without Losing Accuracy: Implementation Plan

> **for Xerxes:** Use subagent-driven-development to execute this plan phase by phase. GPU timing is serial: only one agent may benchmark at a time. Reviews are read-only and may run in parallel.

**Goal:** Make jax-metallib faster than jax-mps 0.11.0 on every workload in `benchmarks/vs_jax_mps.py`, with CPU-level accuracy and no regression on current wins.

**Architecture:** Fix the measurement first, then fix the one correctness defect that inflates our f32 wins, then attack shared costs (per-call overhead, constant-only graph segments, f32 GEMM) before workload-specific kernels (max, sort, softmax, attention, gather). Every change is gated behind exact or CPU-tolerance tests and an order-balanced benchmark.

**Tech stack:** Objective-C++17 / MPSGraph / Metal (`src/pjrt_plugin/`), Python 3.13 / JAX 0.11.1, pytest, `benchmarks/vs_jax_mps.py`.

**Comparator:** jax-mps 0.11.0 (JAX 0.11.2) in `.venv-jax-mps-0.11`. This is the latest release on PyPI and GitHub; `main` is one version-bump commit ahead with no code changes. Hardware: Apple M5, 10-core GPU, 16 GB.

---

## Evidence this plan is based on

All files are under `build/` unless noted. Speedup = jax-mps time / our time; below 1.00 is a loss.

### 1. Order-balanced rerun (`vs-mps-r21-abba.jsonl`, A-B-B-A x2, 4 pairs per workload)

The stock script always runs us first. Alternating the order removed several losses and exposed two hidden ones.

| Status | Workload | ABBA speedup | Pairwise wins |
|---|---|---:|---:|
| Real loss | `pattern.attention_f16` | 0.807 | 0/4 |
| Real loss | `sort.sort_f32` | 0.884 | 0/4 |
| Real loss | `reduce.max_f32` | 0.920 | 0/4 |
| Real loss | `elemwise.add_f16` | 0.931 | 0/4 |
| Real loss (hidden) | `reduce.mean_f32` | 0.939 | 0/4 |
| Real loss | `fusion.add_mul_f16` | 0.947 | 0/4 |
| Real loss | `memory.gather_f32` | 0.954 | 0/4 |
| Real loss | `model.gpt2_small_f32` | 0.954 | 0/4 |
| Real loss | `fusion.exp_mul_add_f32` | 0.957 | 0/4 |
| Real loss (hidden) | `reduce.sum_f32` | 0.970 | 0/4 |
| Small loss | `mul_f32`, `add_mul_f32`, `diamond_f32`, `softmax_numerator_f32`, `5op_chain_f16`, `composite.softmax_f32` | 0.97–0.98 | 0–1/4 |
| Parity | `add_f32`, `exp_f32`, `sqrt_f32`, `abs_f32`, `transformer_layer_f32` | 0.98–1.00 | 1–2/4 |
| Noise | `attention_f32` (1.15), `batched_matmul_f32` (1.05) | — | 1/4 |
| Win | `gpt2_small_f16` 1.03, `gpt2_medium_f32` 1.01, `scan_rnn_f32` 1.10 | — | 4/4 |

### 2. Our f32 GEMM wins are mostly f16 narrowing (`vs-mps-r21-true-f32.log`)

`ops/binary_ops.mm:656-708` casts both f32 operands to f16 when each has at least 500,000 elements and `batch*M*N*K >= 1e8`.

| Workload | Default | `JAX_METALLIB_GEMM_TRUE_F32=1` |
|---|---:|---:|
| `linalg.matmul_f32` (1024²) | 3.00x | **0.64x** (1.228 vs 0.791 ms) |
| `linalg.matmul_large_f32` | 2.77x | 1.16x |
| `pattern.mlp_block_f32` | 2.89x | **0.94x** |
| `model.gpt2_medium_f32` | 1.01x | **0.93x** |
| `linalg.batched_matmul_f32` | noise | 0.97–1.01x (one 0.40x cold outlier) |

### 3. Narrowing is a correctness bug, not only a precision trade-off (`r21_probe_narrow_range.py`)

1024x1024 f32 matmul against XLA:CPU:

| Input scale | Result |
|---|---|
| 1, 1e3 | finite, max relative error 3.0–3.3e-4 (f32 GEMM is ~1e-6) |
| 1e5 | **non-finite output**; CPU is finite |
| any, `precision=HIGHEST` | identical to default; `precision_config` is never read (no match for it in `src/pjrt_plugin`) |

### 4. Fixed per-call overhead (`r21_probe_overhead.log`, 200 calls, 2 rounds)

| f32 add size | Ours median / min (ms) | jax-mps median / min (ms) |
|---|---|---|
| 1, 1024, 65536 | 0.18–0.29 / 0.10–0.14 | 0.17–0.19 / 0.08–0.09 |
| 16M | 1.82–1.85 | 1.81–1.87 |

Large-array bandwidth is at parity; small calls cost about 0.03–0.10 ms more. The f16 elementwise gaps (~0.06–0.08 ms) are the same size. jax-mps also folds `x * 1.0` to a 1.5 µs no-op; we take 1.27 ms.

### 5. Execution plans (`r21_plans/*.log`, `JAX_METALLIB_TRACE_PLAN=1`)

- `reduce.sum`, `composite.softmax_f32`: a graph segment holding only a constant, then one native step.
- `reduce.mean`: constant graph segment, native reduce, then a graph segment for the divide.
- `sort.sort_f32`: two-constant graph segment, then native sort.
- `reduce.max_f32`: graph only; the native row-reduce route admits max/min only for full reductions (`core/mps_executable.mm:2257`, `:6209`).
- `memory.gather_f32`: one 24-op graph (gather plus bounds mask: 3 compare, 2 and, 2 select, 1 reduce).
- `attention_*`, `transformer_layer_f32`: single graph segments, no native steps.

### 6. Open anomaly

The first baseline (`vs-mps-r21-baseline.log`) reported `composite.softmax_f32` at a **median** of 35.5 ms over 50 iterations; three reruns gave 1.28–1.30 ms. A median that high means the whole process was slow, not one sample. Task 0.4 must explain it before softmax work.

### Source comparisons

Three read-only agents compared `.cache/jax-mps-src-v0.11.0` with our code. jax-mps lowers `dot_general` to MLX einsum, compiles the program with `mlx::compile`, rewrites softmax to MLX softmax with `precise=false`, clips gather indices, and sorts values directly. MLX kernel sources are not in that checkout, so their accumulation precision is unverified. Our accuracy target is XLA:CPU, not jax-mps.

---

## Rules for every task

1. **Accuracy first.** A change ships only if its exactness tests pass: bitwise where the repo already requires it (NaN propagation, signed zero, stable/first-NaN ordering, half rounding), CPU tolerance elsewhere.
2. **Measure order-balanced.** Use `--order abba --cycles 3` (Task 0.1). A workload counts as a win at speedup >= 1.02 with at least 5/6 pairwise wins.
3. **Protect current wins.** After each phase, rerun the win list below. Allowed drift: 0.03 below the Phase-0 honest baseline, and never below 1.02.
4. **One change per measurement.** Keep each optimization behind an env switch (`JAX_METALLIB_DISABLE_<NAME>=1` restores the old path), as the repo already does, and document it in `XERXES.md`.
5. **Run the suite one file per process.** The whole suite in one process exceeds the 8 GB memory guard.

**Wins to protect** (Phase-0 values replace these numbers): `tanh_f32` 1.11, `sin_f32` 1.07, `gelu_approx_f32` 1.13, `5op_chain_f32` 1.21, `matmul_f16` 2.41, `softmax_f16` 2.78, `conv.2d_f32` 1.49, `conv.2d_f16` 3.98, `decode_slice_update` 1.26, `kv_cache_write` 1.63, `scan_rnn` 1.10, `layernorm_f32` 1.72, `conv_relu_block` 1.29, `mlp_block_f16` 2.84, `transformer_layer_f16` 1.24, `gpt2_small_f16` 1.03, `gpt2_medium_f16` 1.78, `gpt2_large_f16` 2.31, `gpt2_batch_f16` 1.33, `matmul_large_f32` (true f32) 1.16.

**Before starting:** commit or stash the uncommitted native Metal API work in `core/mps_executable.mm`, `CMakeLists.txt`, `runtime/custom_metal_kernel.*`, and `_native_metal.py`. Most tasks below edit `mps_executable.mm`, and mixed diffs cannot be reviewed or reverted cleanly.

---

## Phase 0: Measurement and gates

### Task 0.1: Order-balanced comparison in `vs_jax_mps.py`

**Files:** Modify `benchmarks/vs_jax_mps.py` (`main`, ~lines 885-1032).

1. Add `--order {ab,abba}` (default `abba`) and `--cycles N` (default 3). For `abba`, run A B B A per cycle with `run_one_benchmark`, then report median-of-medians per side, the speedup, and `pairwise_wins` (as in `build/r21_abba.py`).
2. Print the installed plugin versions, not only JAX's:
   ```python
   VERSION_SNIPPET = (
       "import importlib.metadata as m, jax; "
       "names = ('jax-metallib', 'jax-mps'); "
       "print(jax.__version__, [n + ' ' + m.version(n) for n in names "
       "if any(d.metadata['Name'] == n for d in m.distributions())])"
   )
   ```
3. Write `pairwise_wins` and all per-run medians to the JSONL.
4. Verify: `uv run python benchmarks/vs_jax_mps.py --jax-mps-python .venv-jax-mps-0.11/bin/python --filter sort.sort_f32 --cycles 1` prints `jax-mps 0.11.0` and four timings.

### Task 0.2: Accuracy gate for every benchmark workload

**Files:** Create `tests/test_vs_jax_mps_accuracy.py`.

For each of the 46 workloads, build the same function and inputs with fixed seeds, run it on CPU and MPS, and compare. Reuse the workload sources: execute each `BENCHMARKS` entry with a replacement `bench` that returns the jitted output instead of timing it.

- Elementwise, reduction, sort, gather: bitwise for f32 integer-exact ops (sort, max, gather); `max |out-ref| / max |ref| <= 2e-6` (f32) or 1 f16 ULP elsewhere.
- GEMM-based workloads: `max |out-ref| / max |ref| <= 2e-5` for f32 and `5e-3` for f16.

Expected today: f32 GEMM workloads **fail** at 3e-4 (narrowing). That failure is the point; Phase 1 makes them pass.

### Task 0.3: Honest baseline

After Phase 1 lands (or with `JAX_METALLIB_GEMM_TRUE_F32=1` until then), run the full 46-workload ABBA comparison and save it as `build/vs-mps-r22-honest-baseline.jsonl`. This replaces every number in this plan's tables and win list.

### Task 0.4: Explain the 35.5 ms softmax run

1. Run `--filter composite.softmax_f32 --order ab --cycles 20` and log per-iteration times (`times_ms` is already in the JSON).
2. If it recurs, capture `JAX_METALLIB_TRACE_PLAN=1` output and the autotune cache state (`~/.cache/jax-silicon/autotune_cache.bin`) for the slow process, and compare with a fast one.
3. Hypotheses to test, in order: an autotune decision persisted from a cold run; memory pressure (16 GB host, 64 MiB slots, jax-mps run back to back); a softmax route chosen differently on first compile.
4. Done when the slow mode is reproduced and explained, or 40 clean runs show no recurrence (record that in `XERXES.md`).

---

## Phase 1: Make f32 GEMM correct

Fixes the overflow and `precision=HIGHEST`. Changes the scoreboard: `matmul_f32`, `mlp_block_f32`, and `gpt2_medium_f32` become losses until Phase 3.

### Task 1.1: Failing tests

**Files:** Create `tests/test_f32_gemm_precision.py`.

```python
import jax
import jax.numpy as jnp
import numpy as np
import pytest

CPU, MPS = jax.devices("cpu")[0], jax.devices("mps")[0]


def _run(fn, *args, device):
    return np.asarray(jax.jit(fn)(*(jax.device_put(a, device) for a in args)))


def _rel(out, ref):
    return float(np.max(np.abs(out - ref)) / np.max(np.abs(ref)))


@pytest.mark.parametrize("scale", [1.0, 1e3, 1e5])
@pytest.mark.parametrize("precision", [None, jax.lax.Precision.HIGHEST])
def test_large_f32_matmul_matches_cpu(scale, precision):
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((1024, 1024)) * scale).astype(np.float32)
    b = (rng.standard_normal((1024, 1024)) / scale).astype(np.float32)
    fn = lambda x, y: jnp.matmul(x, y, precision=precision)
    out, ref = _run(fn, a, b, device=MPS), _run(fn, a, b, device=CPU)
    assert np.isfinite(out).all()
    assert _rel(out, ref) <= 2e-5


def test_batched_nt_f32_matmul_matches_cpu():
    rng = np.random.default_rng(1)
    q = rng.standard_normal((4, 16, 256, 64)).astype(np.float32)
    k = rng.standard_normal((4, 16, 256, 64)).astype(np.float32)
    fn = lambda x, y: x @ jnp.swapaxes(y, -2, -1)
    assert _rel(_run(fn, q, k, device=MPS), _run(fn, q, k, device=CPU)) <= 2e-5
```

Run `JAX_PLATFORMS=cpu,mps uv run pytest tests/test_f32_gemm_precision.py -v`. Expected: the large-matmul cases fail (scale 1e5 is non-finite; others at ~3e-4).

### Task 1.2: Stop narrowing by default

**Files:** Modify `src/pjrt_plugin/ops/binary_ops.mm:656-708`.

1. Replace the default-on `JAX_METALLIB_GEMM_TRUE_F32` check with an opt-in `JAX_METALLIB_GEMM_FAST_F32=1`. Keep `GEMM_TRUE_F32=1` accepted as a no-op for old scripts.
2. Even with the opt-in, narrow only when every `precision_config` entry is `DEFAULT`:
   ```cpp
   static bool AllowsReducedPrecision(mlir::stablehlo::DotGeneralOp op) {
       auto cfg = op.getPrecisionConfig();
       if (!cfg) return true;
       for (mlir::Attribute attr : *cfg) {
           auto p = mlir::dyn_cast<mlir::stablehlo::PrecisionAttr>(attr);
           if (p && p.getValue() != mlir::stablehlo::Precision::DEFAULT) return false;
       }
       return true;
   }
   ```
   Apply the same check to `stablehlo.dot` if it shares the narrowing path.
3. Rebuild (`uv pip install -e .`) and rerun Task 1.1: all pass.

### Task 1.3 (optional, opt-in mode only): Overflow-safe fast path

If the opt-in fast mode is kept, it must not produce inf where f32 does not. Scale each operand by an exact power of two `2^-e`, with `e` derived from `max |x|` so the scaled value fits f16; multiply the f32 result by `2^(ea+eb)`. Measure the two extra reductions; drop the fast mode if it no longer beats true f32.

### Task 1.4: Docs and baseline

Update the README float32-GEMM paragraph and configuration row, and the `XERXES.md` env table. Then run Task 0.3.

---

## Phase 2: Per-call overhead (all small and elementwise workloads)

Targets the 0.03–0.10 ms fixed gap behind `add_f16`, `add_mul_f16`, `exp_mul_add_f32`, `mul_f32`, `diamond_f32`, `softmax_numerator_f32`, `5op_chain_f16`, `reduce.sum`, and parts of every small graph.

### Task 2.1: Measure where host time goes

**Files:** Modify `src/pjrt_plugin/core/mps_executable.mm` (direct-run path ~9105-9180, cache acquisition ~9026-9040, output pool ~6360-6430).

Add `JAX_METALLIB_TRACE_HOST_TIMING=1`: accumulate `mach_absolute_time` deltas for input wrapping, output acquisition, intermediate-cache acquire/release, `MPSGraphTensorData` construction, descriptor creation, encode/commit, and the wait in `completion_event.h`. Print totals per 1,000 calls. Run `build/r21_probe_overhead.py` (n=1 and n=1024) with it enabled and rank the phases. No behavior change.

### Task 2.2: Fix the largest measured phase first

Candidates from the source review, applied only where Task 2.1 shows cost:

1. Skip intermediate-cache acquisition for graphs whose only buffers are inputs and outputs.
2. Lease per-executable execution contexts (tensor-data wrappers, descriptors) and recycle them after the completion fence, instead of destroying the cache set after every call (`:11006`). Never reuse in-flight storage.
3. Check whether the direct-run execution descriptor is released (`:9166-9168`); confirm ARC is enabled for `.mm` files in `CMakeLists.txt` before treating it as a leak.
4. Compare direct-run against encoding the compiled executable into an owned command buffer (`:10186`) for single-graph plans.

**Gate:** `test_completion_timeout.py`, `test_pjrt_client_ownership.py` (with its env setup), `test_native_metal_cache.py`, plus a concurrency test calling one executable from four threads. Target: tiny-call median within 0.01 ms of jax-mps.

### Task 2.3 (only if exactly provable): Identity elimination

`x * 1.0` can be folded for finite values, signed zero, and infinities, but may quiet signaling NaNs. `x + 0.0` is **not** an identity (`-0 + +0 = +0`). Fold `multiply(x, splat 1.0)` only after a test shows MPSGraph's own multiply preserves the same NaN bits; return an aliased buffer that shares the allocation and producer event (copying the input today costs 1.27 ms). Not a benchmark workload; do it last.

---

## Phase 3: Fast true-f32 GEMM

Targets `matmul_f32` (0.64x honest), `mlp_block_f32` (0.94x), `gpt2_medium_f32` (0.93x), `batched_matmul_f32` (parity), and the GEMMs inside `attention_f32`, `transformer_layer_f32`, `gpt2_small_f32`.

### Task 3.1: Measure available f32 kernels

Native GEMM selection runs only when the intrinsic family is enabled (`core/mps_executable.mm:6198`, `:6250`), and its classifier is NN-only (`:2550`). Write `build/r22_gemm_f32_variants.py` that forces each variant with `JAX_METALLIB_GEMM_VARIANT` (0–3 tiled, -1 MPP, -2 MPSGraph) under `JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1`, for shapes 1024², 2048², 4096², the MLP block, and batched `[64,256,64]x[64,64,256]`. Record time and CPU relative error. Also check whether `metal_shaders/steel_gemm.metal` has a `float` instantiation; MLX uses Steel for its f32 GEMM.

### Task 3.2: Standalone native f32 GEMM route

Like standalone softmax, add a default-on route for f32 dots that does not need the intrinsic family:

1. NN and NT layouts (absorb the transpose instead of materializing it); batched with contiguous batch.
2. f32 inputs, f32 accumulation, f32 output.
3. Choose the kernel with the existing `AutotuneGemm`, compared against MPSGraph; keep MPSGraph when it wins.
4. Do not split graphs where a native GEMM sits between small graph ops: the softmax comments (`:1912`) show splits cost more than they save. Admit the route only for standalone dots or dots at segment boundaries, until measured otherwise.

**Gate:** Task 1.1 tests, Task 0.2, `test_int_dot.py`, `test_dot_k1_signed_zero.py`, `test_dot_preferred_dtype.py`, `test_f16_native_dot.py` (f16 routes must not change), and the full win list. Target: `matmul_f32` true f32 >= 1.02.

---

## Phase 4: Remove constant-only graph segments

Targets `reduce.sum` (0.970), `reduce.mean` (0.939), `composite.softmax_f32` (0.977), `sort.sort_f32` (0.884).

### Task 4.1: Bake consumer-only constants

**Files:** `core/mps_executable.mm` segmentation (~7213), baking (~5062), single-native-step fast path (~8566).

1. When a constant is used only as a neutral reduce init (already checked by `IsNeutralReduceInit`) or only by a recognized sort comparator, drop it from the plan.
2. For any other constant consumed by a native step, materialize it once into an immutable plan-owned buffer at build time, preserving exact bytes.
3. Result: these plans become one native step and qualify for the fast path.

**Gate:** `test_reduce_init*.py`, `test_argreduce_init.py`, `test_reduce_routing.py`, `test_native_sort.py`, `test_softmax_routing.py`, `test_native_softmax.py`. Verify with `JAX_METALLIB_TRACE_PLAN=1` that the four plans have one step.

### Task 4.2: Fold mean's divide into the native step

`jnp.mean` is `sum / n`, a division. Add a native reduce epilogue that divides by `n` (not multiplies by `1/n`; those round differently). Then `reduce.mean` is one native step. Gate bitwise against the current graph result on 1,000 random rows plus NaN/inf rows.

---

## Phase 5: Native last-axis max/min (`reduce.max_f32`, 0.920)

**Files:** `runtime/metal_kernels.mm` (reduction kernels ~836-950), `core/mps_executable.mm` (classification ~2252-2260, bypass ~6209-6230), reference semantics in `ops/reduction_ops.mm:735` (`FloatToOrderKey`).

1. Write the tests first: per-row NaN at every lane and in the tail, -0/+0 ties in both orders, subnormals, all -inf rows, row lengths 1023/1024/4096/4097. The reference is the current graph output, which must already match `test_reduce_maxmin_signed_zero.py` and `test_reduce_minmax_nan.py`.
2. Implement a vector-loaded row kernel on the integer order key (key, then inverse key on store). Do not route the existing Metal `max` kernel: it does not order -0 below +0 or normalize subnormals.
3. Admit last-axis max/min for f32 rows >= 1024 (keep `REDUCE_BYPASS_MIN_ROW`), standalone or terminal only.

**Gate:** the tests above bitwise, `test_max_reduce_nan.py`, `test_nan_extrema.py`, `test_large_full_reduce.py`, `layernorm` and `softmax_f16` wins.

---

## Phase 6: Short-row sort (`sort.sort_f32`, 0.884)

Phase 4 removes the constant segment first; measure what remains.

**Files:** `runtime/metal_kernels.mm` (keyed bitonic ~1141-1200; register sort ~1350), `core/mps_executable.mm` (sort routing ~4983-4991).

1. Tests first: stable ties, mixed -0/+0, subnormals, NaN payloads, non-power-of-two rows (100, 255, 256, 257).
2. For rows <= 256: a register/simd-shuffle bitonic network over `(canonical key, original index)` pairs, 8 elements per lane in 32 lanes, then a raw-bit gather. The existing register kernel uses plain float comparison and must not be enabled as is.
3. Keep the current shared-memory kernel for longer rows.

**Gate:** `test_sort_bitwise.py`, `test_small_sort_native.py`, `test_native_sort.py`, `test_complex_sort.py`, `test_topk_total_order.py`, `test_sort_grad.py`, `test_sort_move_axis.py`.

---

## Phase 7: f16 attention (`pattern.attention_f16`, 0.807, the largest real loss)

### Task 7.1: Find out why no softmax fusion applies

The f16 widened-softmax matcher accepts up to 8192 columns, and attention has 128. Dump the StableHLO for the benchmark (`jax.jit(fn).lower(...).as_text()`) and the plan (`build/r21_plans/pattern.attention_f16.log`), and identify which matcher condition fails: 4-D shape, scale multiply before the max, minimum element count, or the multi-row f32 short-row filter (`:1916-1920`).

### Task 7.2: Fix the cheapest cause, then measure

If the matcher misses on shape, admit it and measure end to end. Graph/native splits made f32 attention slower before (`:1912`), so accept only an ABBA gain.

### Task 7.3 (if 7.2 is not enough): Fused attention region

Recognize `QK^T -> scale -> softmax -> PV` and run it as one native kernel for small sequence lengths, reusing the half-rounding points of the XLA-matching f16 softmax kernel (`runtime/metal_kernels.mm:1727`, `:1776-1799`) with f32 accumulation. Gate against CPU at 1 f16 ULP on the probabilities and the same output tolerance as `test_f16_softmax_fusion.py`. Protect `transformer_layer_f16`, all `gpt2_*_f16`, `mlp_block_f16`.

---

## Phase 8: f32 softmax kernel (`composite.softmax_f32`, 0.977 after Phase 4)

Only if Phase 4 does not reach 1.02. The current kernel stages a 16 KiB row in shared memory, runs an online max/sum, and recomputes `exp` for the output (`runtime/metal_kernels.mm:1580-1642`). Add an f32 variant that keeps the row in registers (vector loads), computes the row max, caches `exp(x - max)`, sums, and normalizes, with its own threadgroup size. Gate: CPU tolerance plus NaN, +inf, all -inf rows, and extreme logits; `test_native_softmax.py`. Leave the f16 kernel untouched.

---

## Phase 9: Gather (`memory.gather_f32`, 0.954)

1. Read the benchmark source (`benchmarks/vs_jax_mps.py:462`, [65536,512] table, 8192 indices) and the 24-op wrapper in `build/r21_plans/memory.gather_f32.log` to pin the exact JAX mode (fill or clip).
2. Recognize that wrapper and run it as one native row-gather that writes the same fill value for out-of-range and negative indices and copies rows bit-exactly (float4 loads). The existing row-gather writes zero for invalid indices (`runtime/metal_kernels.mm:2079`), which is only correct when the fill value is zero.
3. Gate: NaN payloads, -0, negative and out-of-range indices, int32 extremes, every fill mode; `test_gather_int32.py`, `test_complex_gather_scatter.py`, `test_gather_clamp_scan_scatter_body.py`.

---

## Phase 10: Remaining model gaps

Re-measure `gpt2_small_f32` (0.954), `transformer_layer_f32`, and `attention_f32` after Phases 2–3, since they inherit those changes. If `gpt2_small_f32` is still behind, try full unrolling of its four-layer body (unrolling now stops at 64 body ops, `ops/control_flow_ops.mm:499`), behind a size cap, and measure compile time as well as steady state.

---

## Done criteria

- Task 0.2 accuracy test and the full suite (one file per process) pass with default settings.
- Every workload: ABBA speedup >= 1.02 with >= 5/6 pairwise wins, or a written explanation of why it cannot be won without breaking accuracy (with the measurement).
- No protected win below its Phase-0 value minus 0.03.
- Every new route has a `JAX_METALLIB_DISABLE_*` switch and an `XERXES.md` entry with before/after numbers.
