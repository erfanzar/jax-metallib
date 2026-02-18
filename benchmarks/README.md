# Benchmarks (Manual Harness)

This folder contains a small, repeatable microbenchmark harness for `jax-metallib`.

Goals:

- Measure compile/build time and steady-state execution time.
- Attribute graphs to dominant StableHLO ops via op histograms.
- Provide a few "anchor" workloads that reflect real patterns.

## Run

List cases:

```bash
uv run python -m benchmarks.bench list
```

Run all anchors on MPS:

```bash
JAX_PLATFORMS=mps uv run python -m benchmarks.bench run --case 'anchor\\..*' --platform mps
```

Run on CPU (baseline):

```bash
JAX_PLATFORMS=cpu uv run python -m benchmarks.bench run --case 'anchor\\..*' --platform cpu
```

Write JSONL output:

```bash
uv run python -m benchmarks.bench run --case 'anchor\\..*' --json-out /tmp/jax_silicon_bench.jsonl
```

## Suggested Budgets (Document-Only)

On the same machine:

- `compile_s`: keep within +15%
- `steady_state_s`: keep within +10%

## Add A Case

1. Add a `BenchCase` in `benchmarks/anchors.py` (or a new module imported by `benchmarks/cases.py`).
2. Ensure the `build(rng)` returns `(func, args, kwargs)` where args/kwargs are JAX arrays or Python scalars.
3. Prefer shapes that run quickly but are not toy-sized.
