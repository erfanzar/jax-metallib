from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np


def _ensure_repo_on_syspath() -> None:
    # Allow running as `python benchmarks/bench.py ...` without `-m`.
    if __package__:
        return
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root))


_ensure_repo_on_syspath()

from benchmarks.cases import all_cases, load_all  # noqa: E402
from benchmarks.util import (  # noqa: E402
    block_until_ready_pytree,
    json_dumps,
    stablehlo_op_counts_from_text,
    stablehlo_text,
    summarize_pytree_shapes_and_dtypes,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bench.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List benchmark cases")
    p_list.add_argument("--case", default=".*", help="Regex filter (default: .*)")

    p_run = sub.add_parser("run", help="Run benchmark cases")
    p_run.add_argument("--case", default="anchor\\..*", help="Regex filter (default: anchor\\..*)")
    p_run.add_argument("--platform", choices=["mps", "cpu"], default="mps")
    p_run.add_argument("--jit", dest="jit", action="store_true", default=True)
    p_run.add_argument("--no-jit", dest="jit", action="store_false")
    p_run.add_argument("--warmup-iters", type=int, default=10)
    p_run.add_argument("--iters", type=int, default=50)
    p_run.add_argument("--repeats", type=int, default=3)
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--dump-stablehlo", action="store_true", default=False)
    p_run.add_argument("--stablehlo-op-counts", action="store_true", default=True)
    p_run.add_argument("--no-stablehlo-op-counts", dest="stablehlo_op_counts", action="store_false")
    p_run.add_argument("--json-out", type=str, default="")

    return p.parse_args(argv)


def _run_one_case(
    *,
    case_name: str,
    build,
    platform: str,
    do_jit: bool,
    warmup_iters: int,
    iters: int,
    repeats: int,
    seed: int,
    dump_stablehlo: bool,
    stablehlo_op_counts: bool,
) -> dict[str, Any]:
    device = jax.devices(platform)[0]

    lower_s_list: list[float] = []
    compile_s_list: list[float] = []
    first_s_list: list[float] = []
    steady_s_list: list[float] = []
    stablehlo_counts: dict[str, int] | None = None

    for rep in range(repeats):
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()

        rng = np.random.default_rng(seed + rep)
        with jax.default_device(device):
            fn, args, kwargs = build(rng)

            shapes, dtypes = summarize_pytree_shapes_and_dtypes((args, kwargs))
            shape_summary = shapes
            dtype_summary = dtypes

            if do_jit:
                jit_fn = jax.jit(fn)

                t0 = time.perf_counter()
                lowered = jit_fn.lower(*args, **kwargs)
                t1 = time.perf_counter()
                lower_s = t1 - t0

                text = ""
                if dump_stablehlo or stablehlo_op_counts:
                    text = stablehlo_text(lowered)
                    if dump_stablehlo and rep == 0:
                        print(text)
                    if stablehlo_op_counts and stablehlo_counts is None:
                        stablehlo_counts = stablehlo_op_counts_from_text(text)

                t0 = time.perf_counter()
                compiled = lowered.compile()
                t1 = time.perf_counter()
                compile_s = t1 - t0

                t0 = time.perf_counter()
                out = compiled(*args, **kwargs)
                block_until_ready_pytree(out)
                t1 = time.perf_counter()
                first_run_s = t1 - t0

                for _ in range(warmup_iters):
                    out = compiled(*args, **kwargs)
                    block_until_ready_pytree(out)

                times: list[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = compiled(*args, **kwargs)
                    block_until_ready_pytree(out)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                steady_state_s = statistics.median(times) if times else 0.0

                lower_s_list.append(lower_s)
                compile_s_list.append(compile_s)
                first_s_list.append(first_run_s)
                steady_s_list.append(steady_state_s)
            else:
                t0 = time.perf_counter()
                out = fn(*args, **kwargs)
                block_until_ready_pytree(out)
                t1 = time.perf_counter()
                first_run_s = t1 - t0

                for _ in range(warmup_iters):
                    out = fn(*args, **kwargs)
                    block_until_ready_pytree(out)

                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = fn(*args, **kwargs)
                    block_until_ready_pytree(out)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                steady_state_s = statistics.median(times) if times else 0.0

                first_s_list.append(first_run_s)
                steady_s_list.append(steady_state_s)

    def _median(xs: list[float]) -> float:
        return statistics.median(xs) if xs else 0.0

    rec: dict[str, Any] = {
        "case": case_name,
        "platform": platform,
        "jit": bool(do_jit),
        "shape_summary": shape_summary,
        "dtype_summary": dtype_summary,
        "lower_s": _median(lower_s_list) if do_jit else None,
        "compile_s": _median(compile_s_list) if do_jit else None,
        "first_run_s": _median(first_s_list),
        "steady_state_s": _median(steady_s_list),
        "stablehlo_op_counts": stablehlo_counts if stablehlo_op_counts else None,
    }
    return rec


def main(argv: list[str]) -> int:
    args = _parse_args(argv)
    load_all()

    rx = re.compile(args.case)
    selected = [c for c in all_cases() if rx.search(c.name)]

    if args.cmd == "list":
        for c in selected:
            print(c.name)
        return 0

    if not selected:
        raise SystemExit(f"No cases matched regex: {args.case!r}")

    json_out = Path(args.json_out) if args.json_out else None
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)

    for c in selected:
        rec = _run_one_case(
            case_name=c.name,
            build=c.build,
            platform=args.platform,
            do_jit=args.jit,
            warmup_iters=args.warmup_iters,
            iters=args.iters,
            repeats=args.repeats,
            seed=args.seed,
            dump_stablehlo=args.dump_stablehlo,
            stablehlo_op_counts=args.stablehlo_op_counts,
        )
        print(json_dumps(rec))
        if json_out:
            with json_out.open("a") as fp:
                fp.write(json.dumps(rec, sort_keys=True) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
