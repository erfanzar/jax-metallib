"""Robust A/B benchmark for noisy ops with >=5 trials median-of-medians.

Re-runs targeted priority ops with multiple outer trials for statistical
confidence, then prints a summary table.

Usage:
    uv run python -m benchmarks.routing_ab
    uv run python -m benchmarks.routing_ab --compare before.jsonl after.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.bottlenecks import (
    _block_until_ready_pytree,
    _make_case,
)

PRIORITY_OPS = [
    "stablehlo.reduce",
    "stablehlo.maximum",
    "stablehlo.minimum",
    "chlo.square",
    "stablehlo.multiply",
    "stablehlo.select",
    "stablehlo.scatter",
]

SECONDARY_OPS = [
    "stablehlo.reduce_window",
    "stablehlo.pad",
    "stablehlo.convert",
    "stablehlo.convolution",
    "stablehlo.while",
    "stablehlo.case",
]


def run_op_trials(
    op: str,
    fp_dtype: Any,
    *,
    n_outer: int = 5,
    warmup: int = 10,
    inner_iters: int = 30,
) -> dict[str, Any]:
    """Run n_outer independent trials, each with inner_iters timed iterations."""
    rng = np.random.default_rng(42)
    case = _make_case(op, rng, fp_dtype)
    if case is None:
        return {"op": op, "dtype": str(fp_dtype), "error": "no generator"}

    fn, args, kwargs = case
    results: dict[str, dict[str, Any]] = {}

    for platform in ("cpu", "mps"):
        try:
            device = jax.devices(platform)[0]
        except Exception as e:
            results[platform] = {"error": str(e)}
            continue

        dev_args = jax.device_put(args, device)
        dev_kwargs = jax.device_put(kwargs, device)

        trial_medians: list[float] = []
        with jax.default_device(device):
            jit_fn = jax.jit(fn)
            lowered = jit_fn.lower(*dev_args, **dev_kwargs)
            compiled = lowered.compile()

            for _trial in range(n_outer):
                for _ in range(warmup):
                    out = compiled(*dev_args, **dev_kwargs)
                    _block_until_ready_pytree(out)

                times: list[float] = []
                for _ in range(inner_iters):
                    t0 = time.perf_counter()
                    out = compiled(*dev_args, **dev_kwargs)
                    _block_until_ready_pytree(out)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                trial_medians.append(statistics.median(times))

        median_of_medians = statistics.median(trial_medians)
        results[platform] = {
            "trial_medians_ms": [m * 1e3 for m in trial_medians],
            "median_of_medians_ms": median_of_medians * 1e3,
        }

    return {
        "op": op,
        "dtype": "fp16" if fp_dtype == jnp.float16 else "fp32",
        **results,
    }


def compare_jsonl(before_path: str, after_path: str) -> None:
    """Load before/after JSONL and compare steady-state for priority ops."""

    def load(path):
        records = {}
        for line in Path(path).read_text().strip().split("\n"):
            if not line.strip():
                continue
            r = json.loads(line)
            key = (r["op"], r.get("dtype", ""))
            records[key] = r
        return records

    before = load(before_path)
    after = load(after_path)

    all_ops = PRIORITY_OPS + SECONDARY_OPS

    print(
        f"\n{'Op':<30} {'dtype':<6} {'Before MPS':<12} {'After MPS':<12} "
        f"{'Delta%':<10} {'CPU (ms)':<10} {'A/CPU':<8} {'Set'}"
    )
    print("-" * 100)

    for op in all_ops:
        for dtype in ("fp32", "fp16"):
            key = (op, dtype)
            b = before.get(key, {})
            a = after.get(key, {})

            def get_ss(rec, plat):
                d = rec.get(plat, {})
                if d.get("ok"):
                    return d.get("steady_state_s")
                return None

            b_mps = get_ss(b, "mps")
            a_mps = get_ss(a, "mps")
            b_cpu = get_ss(b, "cpu")

            b_ms = f"{b_mps * 1e3:.3f}" if b_mps else "N/A"
            a_ms = f"{a_mps * 1e3:.3f}" if a_mps else "N/A"
            cpu_ms = f"{b_cpu * 1e3:.3f}" if b_cpu else "N/A"

            if b_mps and a_mps:
                delta = (a_mps - b_mps) / b_mps * 100
                delta_s = f"{delta:+.1f}%"
            else:
                delta_s = "N/A"

            if a_mps and b_cpu and b_cpu > 0:
                a_ratio = f"{a_mps / b_cpu:.2f}x"
            else:
                a_ratio = "N/A"

            s = "PRI" if op in PRIORITY_OPS else "SEC"
            print(f"{op:<30} {dtype:<6} {b_ms:<12} {a_ms:<12} {delta_s:<10} {cpu_ms:<10} {a_ratio:<8} {s}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), help="Compare two JSONL files instead of running benchmarks"
    )
    parser.add_argument("--n-outer", type=int, default=5)
    parser.add_argument("--inner-iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.compare:
        compare_jsonl(args.compare[0], args.compare[1])
        return

    print("=" * 90)
    print(f"Robust A/B benchmark ({args.n_outer} outer trials x {args.inner_iters} inner iters)")
    print("=" * 90)

    all_ops = PRIORITY_OPS + SECONDARY_OPS
    records = []

    for op in all_ops:
        for fp_dtype in (jnp.float32, jnp.float16):
            dtype_label = "fp16" if fp_dtype == jnp.float16 else "fp32"
            print(f"\n--- {op} ({dtype_label}) ---")
            result = run_op_trials(
                op,
                fp_dtype,
                n_outer=args.n_outer,
                warmup=args.warmup,
                inner_iters=args.inner_iters,
            )
            records.append(result)

            mps = result.get("mps", {})
            cpu = result.get("cpu", {})

            if "error" in mps or "error" in cpu:
                print(f"  ERROR: mps={mps.get('error', '')} cpu={cpu.get('error', '')}")
                continue

            mps_ms = mps["median_of_medians_ms"]
            cpu_ms = cpu["median_of_medians_ms"]
            ratio = mps_ms / cpu_ms if cpu_ms > 0 else float("inf")
            print(f"  CPU: {cpu_ms:.3f} ms  MPS: {mps_ms:.3f} ms  MPS/CPU: {ratio:.2f}x")
            print(f"  MPS trials: {[f'{m:.3f}' for m in mps['trial_medians_ms']]}")

    out_path = Path("tmp-files/routing_ab_results.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 90)
    print(f"{'Op':<30} {'dtype':<6} {'CPU (ms)':<10} {'MPS (ms)':<10} {'MPS/CPU':<10} {'set'}")
    print("-" * 90)
    for r in records:
        mps = r.get("mps", {})
        cpu = r.get("cpu", {})
        if "error" in mps or "error" in cpu:
            print(f"{r['op']:<30} {r['dtype']:<6} {'ERR':<10} {'ERR':<10} {'ERR':<10}")
            continue
        mps_ms = mps["median_of_medians_ms"]
        cpu_ms = cpu["median_of_medians_ms"]
        ratio = mps_ms / cpu_ms if cpu_ms > 0 else float("inf")
        s = "PRI" if r["op"] in PRIORITY_OPS else "SEC"
        print(f"{r['op']:<30} {r['dtype']:<6} {cpu_ms:<10.3f} {mps_ms:<10.3f} {ratio:<10.2f} {s}")


if __name__ == "__main__":
    main()
