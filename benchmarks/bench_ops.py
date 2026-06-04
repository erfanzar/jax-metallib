#!/usr/bin/env python3
"""Benchmark MPS vs CPU for key workloads (mirrors jax-mps benchmark suite).

Usage:
    uv run python benchmarks/bench_ops.py
    uv run python benchmarks/bench_ops.py --grad       # include gradient benchmarks
    uv run python benchmarks/bench_ops.py --filter matmul  # only matching names
"""

import argparse
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

# ---------------------------------------------------------------------------
# Benchmark definitions — same workloads as jax-mps tests/configs/benchmark.py
# ---------------------------------------------------------------------------


def _make_benchmarks():
    benchmarks = []

    def add(name, func, *arg_factories, grad=True):
        benchmarks.append((name, func, arg_factories, grad))

    # --- Elementwise ---
    for scale in [1, 10, 100, 1000]:
        n = scale * 10_000
        add(f"exp_{scale}", jnp.exp, lambda rng, n=n: rng.standard_normal(n).astype(np.float32))
        add(
            f"add_{scale}",
            jnp.add,
            lambda rng, n=n: rng.standard_normal(n).astype(np.float32),
            lambda rng, n=n: rng.standard_normal(n).astype(np.float32),
        )
        add(f"sum_{scale}", jnp.sum, lambda rng, n=n: rng.standard_normal(n).astype(np.float32))
        add(
            f"softmax_{scale}",
            lambda x: jax.nn.softmax(x, axis=-1),
            lambda rng, n=n: rng.standard_normal((max(1, n // 1000), 1000)).astype(np.float32),
        )

    # --- Matmul ---
    for scale in [1, 10, 100, 1000]:
        add(
            f"matmul_{scale}",
            jnp.matmul,
            lambda rng, s=scale: rng.standard_normal((s * 4, s * 5)).astype(np.float32),
            lambda rng, s=scale: rng.standard_normal((s * 5, s * 3)).astype(np.float32),
        )

    # --- Batched matmul ---
    for batch in [8, 32, 128]:
        add(
            f"matmul_batched_{batch}",
            jnp.matmul,
            lambda rng, b=batch: rng.standard_normal((b, 64, 64)).astype(np.float32),
            lambda rng, b=batch: rng.standard_normal((b, 64, 64)).astype(np.float32),
        )

    # --- Conv2D ---
    for channels in [32, 64, 128]:
        add(
            f"conv2d_{channels}ch",
            lambda x, w: lax.conv_general_dilated(
                x, w, window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
            ),
            lambda rng, c=channels: rng.standard_normal((8, 32, 32, c)).astype(np.float32),
            lambda rng, c=channels: rng.standard_normal((3, 3, c, c)).astype(np.float32),
        )

    # --- LayerNorm ---
    def layer_norm(x):
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.var(x, axis=-1, keepdims=True)
        return (x - mean) / jnp.sqrt(var + 1e-5)

    for hidden in [256, 512, 1024]:
        add(
            f"layernorm_{hidden}", layer_norm, lambda rng, h=hidden: rng.standard_normal((32, 128, h)).astype(np.float32)
        )

    # --- Reduction axis ---
    for scale in [10, 100, 1000]:
        n = scale * 10_000
        add(
            f"mean_axis0_{scale}",
            lambda x: jnp.mean(x, axis=0),
            lambda rng, n=n: rng.standard_normal((1000, n // 1000)).astype(np.float32),
        )

    # --- Cholesky ---
    for size in [32, 128, 512]:

        def make_posdef(rng, s=size):
            a = rng.standard_normal((s, s)).astype(np.float32)
            return a @ a.T + s * np.eye(s, dtype=np.float32)

        add(f"cholesky_{size}", jnp.linalg.cholesky, make_posdef, grad=False)

    # --- Triangular solve ---
    for size in [32, 128, 512]:

        def make_tri(rng, s=size):
            a = np.tril(rng.standard_normal((s, s)).astype(np.float32))
            np.fill_diagonal(a, np.abs(np.diag(a)) + 1.0)
            return a

        add(
            f"triangular_solve_{size}",
            lambda a, b: jax.scipy.linalg.solve_triangular(a, b, lower=True),
            make_tri,
            lambda rng, s=size: rng.standard_normal((s, s)).astype(np.float32),
            grad=False,
        )

    # --- Sort ---
    for n in [10_000, 100_000, 1_000_000]:
        add(f"sort_{n // 1000}k", jnp.sort, lambda rng, n=n: rng.standard_normal(n).astype(np.float32), grad=False)

    return benchmarks


BENCHMARKS = _make_benchmarks()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def time_fn(fn, warmup=3, repeats=10):
    """Time a function, returning median seconds."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return np.median(times)


def run_benchmark(name, func, arg_factories, device, *, grad_argnum=None):
    """Run a single benchmark on a device, return median time in seconds."""
    rng = np.random.default_rng(42)
    raw_args = [f(rng) for f in arg_factories]

    with jax.default_device(device):
        args = [jax.device_put(jnp.asarray(a), device) for a in raw_args]
        for a in args:
            a.block_until_ready()

        if grad_argnum is not None:

            def scalar_fn(*args):
                r = func(*args)
                if isinstance(r, (tuple, list)):
                    r = r[0]
                if r.shape != ():
                    r = r.mean()
                return r

            jitted = jax.jit(jax.grad(scalar_fn, argnums=grad_argnum))
        else:
            jitted = jax.jit(func)

        def run():
            r = jitted(*args)
            jax.tree.map(lambda x: x.block_until_ready(), r)
            return r

        try:
            t = time_fn(run)
        except Exception as e:
            return None, str(e)
        return t, None


def format_time(t):
    if t is None:
        return "  FAIL  "
    if t < 1e-3:
        return f"{t * 1e6:7.1f}µs"
    if t < 1:
        return f"{t * 1e3:7.2f}ms"
    return f"{t:7.3f} s"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grad", action="store_true", help="Include gradient benchmarks")
    parser.add_argument("--filter", type=str, default="", help="Only run benchmarks matching this string")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()

    cpu = jax.devices("cpu")[0]
    try:
        mps = jax.devices("mps")[0]
    except Exception:
        print("MPS device not available")
        sys.exit(1)

    filtered = [(n, f, af, g) for n, f, af, g in BENCHMARKS if args.filter in n]
    if not filtered:
        print(f"No benchmarks matching '{args.filter}'")
        sys.exit(1)

    # Header
    print(f"\n{'Benchmark':<35} {'CPU':>10} {'MPS':>10} {'Speedup':>9} {'Status'}")
    print("-" * 80)

    slow_ops = []

    for name, func, arg_factories, supports_grad in filtered:
        # Value benchmark
        cpu_t, _cpu_err = run_benchmark(name, func, arg_factories, cpu)
        mps_t, mps_err = run_benchmark(name, func, arg_factories, mps)

        if mps_err:
            speedup = 0
            status = f"MPS err: {mps_err[:40]}"
        elif cpu_t and mps_t:
            speedup = cpu_t / mps_t
            status = "ok" if speedup >= 0.8 else "SLOW"
            if status == "SLOW":
                slow_ops.append((name, cpu_t, mps_t, speedup))
        else:
            speedup = 0
            status = "err"

        print(f"  {name:<33} {format_time(cpu_t):>10} {format_time(mps_t):>10} {speedup:>8.2f}x  {status}")

        # Gradient benchmark
        if args.grad and supports_grad:
            cpu_gt, _cpu_gerr = run_benchmark(name, func, arg_factories, cpu, grad_argnum=0)
            mps_gt, mps_gerr = run_benchmark(name, func, arg_factories, mps, grad_argnum=0)

            if cpu_gt and mps_gt:
                gspeedup = cpu_gt / mps_gt
                gstatus = "ok" if gspeedup >= 0.8 else "SLOW"
                if gstatus == "SLOW":
                    slow_ops.append((f"{name}_grad", cpu_gt, mps_gt, gspeedup))
            elif mps_gerr:
                gspeedup = 0
                gstatus = f"MPS err: {mps_gerr[:40]}"
            else:
                gspeedup = 0
                gstatus = "err"

            print(
                f"  {name + '_grad':<33} {format_time(cpu_gt):>10} "
                f"{format_time(mps_gt):>10} {gspeedup:>8.2f}x  {gstatus}"
            )

    # Summary
    print("\n" + "=" * 80)
    if slow_ops:
        print(f"\n{len(slow_ops)} operations where MPS is slower than CPU:\n")
        print(f"  {'Operation':<35} {'CPU':>10} {'MPS':>10} {'Speedup':>9}")
        print(f"  {'-' * 35} {'-' * 10} {'-' * 10} {'-' * 9}")
        for name, ct, mt, sp in sorted(slow_ops, key=lambda x: x[3]):
            print(f"  {name:<35} {format_time(ct):>10} {format_time(mt):>10} {sp:>8.2f}x")
    else:
        print("\nAll operations are at least as fast on MPS as CPU.")
    print()


if __name__ == "__main__":
    main()
