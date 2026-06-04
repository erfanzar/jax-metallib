from __future__ import annotations

import dataclasses
import json
import re
import statistics
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp_special
import numpy as np
from jax._src.lax import windowed_reductions as wr

REPO_ROOT = Path(__file__).resolve().parent.parent
OPS_DIR = REPO_ROOT / "src" / "pjrt_plugin" / "ops"

# Extract ops from registration macros.
REGISTER_OP_RE = re.compile(r"REGISTER_[A-Z0-9_]+\(\"((?:stablehlo|chlo)\.[^\"]+)\"")


def _extract_supported_ops() -> list[str]:
    ops: set[str] = set()
    for p in sorted(OPS_DIR.glob("*.mm")):
        text = p.read_text(encoding="utf-8", errors="replace")
        for m in REGISTER_OP_RE.finditer(text):
            ops.add(m.group(1))
    return sorted(ops)


def _block_until_ready_pytree(x: Any) -> None:
    def _block(leaf):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
        return leaf

    jax.tree_util.tree_map(_block, x)


@dataclasses.dataclass(frozen=True)
class BenchResult:
    ok: bool
    error: str
    lower_s: float | None
    compile_s: float | None
    first_run_s: float | None
    steady_state_s: float | None
    stablehlo_op_counts: dict[str, int] | None


def _stablehlo_text(lowered) -> str:
    return str(lowered.compiler_ir(dialect="stablehlo"))


_STABLEHLO_OP_RE = re.compile(r"(?<![\#\!])(?:stablehlo|chlo)\.[\w\.]+")


def _stablehlo_op_counts(text: str) -> dict[str, int]:
    from collections import Counter

    return dict(Counter(_STABLEHLO_OP_RE.findall(text)))


def _time_one(
    *,
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    platform: str,
    warmup_iters: int,
    iters: int,
    do_lower_compile: bool = True,
) -> BenchResult:
    try:
        device = jax.devices(platform)[0]
    except Exception as e:
        return BenchResult(
            ok=False,
            error=f"no device for platform {platform}: {e}",
            lower_s=None,
            compile_s=None,
            first_run_s=None,
            steady_state_s=None,
            stablehlo_op_counts=None,
        )

    # Put inputs on the target device once.
    try:
        args = jax.device_put(args, device)
        kwargs = jax.device_put(kwargs, device)
    except Exception as e:
        return BenchResult(
            ok=False,
            error=f"device_put failed ({platform}): {e}",
            lower_s=None,
            compile_s=None,
            first_run_s=None,
            steady_state_s=None,
            stablehlo_op_counts=None,
        )

    try:
        if hasattr(jax, "clear_caches"):
            jax.clear_caches()
    except Exception:
        pass

    try:
        with jax.default_device(device):
            jit_fn = jax.jit(fn)

            lowered = None
            lower_s = None
            compile_s = None
            stablehlo_counts = None

            if do_lower_compile:
                t0 = time.perf_counter()
                lowered = jit_fn.lower(*args, **kwargs)
                t1 = time.perf_counter()
                lower_s = t1 - t0

                text = _stablehlo_text(lowered)
                stablehlo_counts = _stablehlo_op_counts(text)

                t0 = time.perf_counter()
                compiled = lowered.compile()
                t1 = time.perf_counter()
                compile_s = t1 - t0

                t0 = time.perf_counter()
                out = compiled(*args, **kwargs)
                _block_until_ready_pytree(out)
                t1 = time.perf_counter()
                first_run_s = t1 - t0

                for _ in range(warmup_iters):
                    out = compiled(*args, **kwargs)
                    _block_until_ready_pytree(out)

                times: list[float] = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = compiled(*args, **kwargs)
                    _block_until_ready_pytree(out)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                steady_state_s = statistics.median(times) if times else 0.0
            else:
                # Fallback path for transformations that don't expose .lower/.compile.
                t0 = time.perf_counter()
                out = jit_fn(*args, **kwargs)
                _block_until_ready_pytree(out)
                t1 = time.perf_counter()
                first_run_s = t1 - t0

                for _ in range(warmup_iters):
                    out = jit_fn(*args, **kwargs)
                    _block_until_ready_pytree(out)

                times = []
                for _ in range(iters):
                    t0 = time.perf_counter()
                    out = jit_fn(*args, **kwargs)
                    _block_until_ready_pytree(out)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)
                steady_state_s = statistics.median(times) if times else 0.0

            return BenchResult(
                ok=True,
                error="",
                lower_s=lower_s,
                compile_s=compile_s,
                first_run_s=first_run_s,
                steady_state_s=steady_state_s,
                stablehlo_op_counts=stablehlo_counts,
            )
    except Exception as e:
        return BenchResult(
            ok=False,
            error=str(e),
            lower_s=None,
            compile_s=None,
            first_run_s=None,
            steady_state_s=None,
            stablehlo_op_counts=None,
        )


def _top_ops(op_counts: dict[str, int] | None, n: int = 4) -> str:
    if not op_counts:
        return ""

    exclude = {"stablehlo.constant", "stablehlo.return"}
    items = [(k, v) for k, v in op_counts.items() if k not in exclude]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return ", ".join(f"{k}({v})" for k, v in items[:n])


ELEM_SHAPE = (4096, 4096)  # ~16M elements: large enough for GPU to amortize dispatch overhead.


def _rand(rng: np.random.Generator, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    # Generate host NumPy arrays and device_put() later. This avoids accidental cross-device
    # copies if the default JAX backend isn't the benchmark target.
    if dtype == jnp.float16:
        return rng.standard_normal(shape).astype(np.float32).astype(np.float16)
    if dtype == jnp.bfloat16:
        # NumPy doesn't have bfloat16; we keep float32 and let JAX convert on device.
        return rng.standard_normal(shape).astype(np.float32)
    if dtype == jnp.float32:
        return rng.standard_normal(shape).astype(np.float32)
    if dtype == jnp.float64:
        return rng.standard_normal(shape).astype(np.float64)
    if dtype == jnp.int32:
        return rng.integers(-1000, 1000, size=shape, dtype=np.int32)
    if dtype == jnp.uint32:
        return rng.integers(0, 2**32 - 1, size=shape, dtype=np.uint32)
    if dtype == jnp.bool_:
        return rng.integers(0, 2, size=shape, dtype=np.int32).astype(np.bool_)
    raise ValueError(f"unsupported rand dtype: {dtype}")


def _make_case(
    op: str, rng: np.random.Generator, fp_dtype: Any
) -> tuple[Callable[..., Any], tuple[Any, ...], dict[str, Any]] | None:
    # Returns None when we don't have a meaningful microbench for this op.

    # Elementwise unary (float)
    unary = {
        "stablehlo.abs": jnp.abs,
        "stablehlo.cbrt": jnp.cbrt,
        "stablehlo.ceil": jnp.ceil,
        "stablehlo.cosine": jnp.cos,
        "stablehlo.erf": jax.lax.erf,
        "stablehlo.exponential": jnp.exp,
        "stablehlo.exponential_minus_one": jnp.expm1,
        "stablehlo.floor": jnp.floor,
        "stablehlo.is_finite": jnp.isfinite,
        "stablehlo.log": jnp.log,
        "stablehlo.log_plus_one": jnp.log1p,
        "stablehlo.negate": lambda x: -x,
        "stablehlo.round_nearest_even": jnp.rint,
        "stablehlo.rsqrt": jax.lax.rsqrt,
        "stablehlo.sign": jnp.sign,
        "stablehlo.sine": jnp.sin,
        "stablehlo.sqrt": jnp.sqrt,
        "stablehlo.tan": jnp.tan,
        "stablehlo.tanh": jnp.tanh,
    }

    # Elementwise unary (chlo)
    chlo_unary = {
        "chlo.acos": jnp.arccos,
        "chlo.acosh": jnp.arccosh,
        "chlo.asin": jnp.arcsin,
        "chlo.asinh": jnp.arcsinh,
        "chlo.atanh": jnp.arctanh,
        "chlo.cosh": jnp.cosh,
        "chlo.erf": jax.lax.erf,
        "chlo.erf_inv": jsp_special.erfinv,
        "chlo.next_after": jnp.nextafter,
        "chlo.sinh": jnp.sinh,
        "chlo.square": jax.lax.square,
    }

    # Elementwise binary (float)
    binary = {
        "stablehlo.add": lambda x, y: x + y,
        "stablehlo.atan2": jnp.arctan2,
        "stablehlo.divide": lambda x, y: x / y,
        "stablehlo.maximum": jnp.maximum,
        "stablehlo.minimum": jnp.minimum,
        "stablehlo.multiply": lambda x, y: x * y,
        "stablehlo.power": lambda x, y: x**y,
        "stablehlo.remainder": jnp.remainder,
        "stablehlo.subtract": lambda x, y: x - y,
    }

    # Elementwise binary (int/bool)
    bitwise = {
        "stablehlo.and": lambda x, y: jnp.bitwise_and(x, y),
        "stablehlo.or": lambda x, y: jnp.bitwise_or(x, y),
        "stablehlo.xor": lambda x, y: jnp.bitwise_xor(x, y),
    }

    if op in unary:
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return unary[op], (x,), {}

    if op in chlo_unary:
        if op == "chlo.next_after":
            x = _rand(rng, ELEM_SHAPE, fp_dtype)
            y = _rand(rng, ELEM_SHAPE, fp_dtype)
            return chlo_unary[op], (x, y), {}
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return chlo_unary[op], (x,), {}

    if op in binary:
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        y = _rand(rng, ELEM_SHAPE, fp_dtype)
        return binary[op], (x, y), {}

    if op in bitwise:
        x = _rand(rng, ELEM_SHAPE, jnp.int32)
        y = _rand(rng, ELEM_SHAPE, jnp.int32)
        return bitwise[op], (x, y), {}

    if op == "stablehlo.not":
        x = _rand(rng, ELEM_SHAPE, jnp.int32)
        return jnp.bitwise_not, (x,), {}

    if op in {"stablehlo.shift_left", "stablehlo.shift_right_logical", "stablehlo.shift_right_arithmetic"}:
        x = _rand(rng, ELEM_SHAPE, jnp.int32)
        sh = np.asarray(3, dtype=np.int32)
        if op == "stablehlo.shift_left":
            return lambda a, s: jnp.left_shift(a, s), (x, sh), {}
        if op == "stablehlo.shift_right_logical":
            return lambda a, s: jnp.right_shift(a.astype(jnp.uint32), s), (x, sh), {}
        return lambda a, s: jnp.right_shift(a, s), (x, sh), {}

    if op == "stablehlo.popcnt":
        x = _rand(rng, ELEM_SHAPE, jnp.uint32)
        # JAX exposes this as lax.population_count.
        return jax.lax.population_count, (x,), {}

    if op == "stablehlo.compare":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        y = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a, b: a < b, (x, y), {}

    if op == "stablehlo.select":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        y = _rand(rng, ELEM_SHAPE, fp_dtype)
        pred = _rand(rng, ELEM_SHAPE, jnp.bool_)
        return jnp.where, (pred, x, y), {}

    if op == "stablehlo.clamp":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        np_dtype = np.float16 if fp_dtype == jnp.float16 else np.float32
        lo = np.asarray(-1.0, dtype=np_dtype)
        hi = np.asarray(1.0, dtype=np_dtype)
        return lambda a, lo, hi: jnp.clip(a, lo, hi), (x, lo, hi), {}

    if op == "stablehlo.convert":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        out_dtype = jnp.float32 if fp_dtype == jnp.float16 else jnp.float16
        return lambda a: a.astype(out_dtype), (x,), {}

    if op == "stablehlo.bitcast_convert":
        x = _rand(rng, ELEM_SHAPE, jnp.int32)
        return lambda a: jax.lax.bitcast_convert_type(a, jnp.float32), (x,), {}

    if op in {"stablehlo.broadcast_in_dim", "stablehlo.broadcast"}:
        x = _rand(rng, (ELEM_SHAPE[1],), fp_dtype)
        if op == "stablehlo.broadcast":
            # Prefix broadcast: [N] -> [M,N]
            return lambda a: jax.lax.broadcast(a, (ELEM_SHAPE[0],)), (x,), {}
        # General broadcast with explicit dims.
        return lambda a: jax.lax.broadcast_in_dim(a, ELEM_SHAPE, (1,)), (x,), {}

    if op == "stablehlo.reshape":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        new_shape = (ELEM_SHAPE[0] * 2, ELEM_SHAPE[1] // 2)
        return lambda a: jnp.reshape(a, new_shape), (x,), {}

    if op == "stablehlo.transpose":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a: jnp.transpose(a, (1, 0)), (x,), {}

    if op == "stablehlo.reverse":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a: jnp.flip(a, axis=1), (x,), {}

    if op == "stablehlo.slice":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a: jax.lax.slice(a, (128, 128), (896, 896), (1, 1)), (x,), {}

    if op == "stablehlo.dynamic_slice":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        i = np.asarray(100, dtype=np.int32)
        j = np.asarray(100, dtype=np.int32)
        return lambda a, i, j: jax.lax.dynamic_slice(a, (i, j), (256, 256)), (x, i, j), {}

    if op == "stablehlo.dynamic_update_slice":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        upd = _rand(rng, (256, 256), fp_dtype)
        i = np.asarray(100, dtype=np.int32)
        j = np.asarray(100, dtype=np.int32)
        return lambda a, u, i, j: jax.lax.dynamic_update_slice(a, u, (i, j)), (x, upd, i, j), {}

    if op == "stablehlo.pad":
        x = _rand(rng, (2048, 2048), fp_dtype)
        np_dtype = np.float16 if fp_dtype == jnp.float16 else np.float32
        pad_val = np.asarray(0.0, dtype=np_dtype)
        pad_config = ((1, 1, 0), (2, 2, 0))
        return lambda a, v: jax.lax.pad(a, v, pad_config), (x, pad_val), {}

    if op == "stablehlo.concatenate":
        x = _rand(rng, (2048, 2048), fp_dtype)
        y = _rand(rng, (2048, 2048), fp_dtype)
        return lambda a, b: jnp.concatenate([a, b], axis=0), (x, y), {}

    if op == "stablehlo.reduce":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a: jnp.sum(a, axis=-1), (x,), {}

    if op == "stablehlo.reduce_window":
        x = _rand(rng, (4, 128, 128, 64), fp_dtype)
        np_dtype = np.float16 if fp_dtype == jnp.float16 else np.float32
        init = np.asarray(-np.inf, dtype=np_dtype)
        return (
            lambda a, init: jax.lax.reduce_window(
                a,
                init,
                jax.lax.max,
                window_dimensions=(1, 3, 3, 1),
                window_strides=(1, 1, 1, 1),
                padding="SAME",
            ),
            (x, init),
            {},
        )

    if op == "stablehlo.select_and_scatter":
        operand = _rand(rng, (4, 128, 128, 64), fp_dtype)
        source = _rand(rng, (4, 128, 128, 64), fp_dtype)
        padding = ((0, 0), (1, 1), (1, 1), (0, 0))
        return (
            lambda src, opnd: wr._select_and_scatter_add(
                src,
                opnd,
                jax.lax.ge_p,
                window_dimensions=(1, 3, 3, 1),
                window_strides=(1, 1, 1, 1),
                padding=padding,
            ),
            (source, operand),
            {},
        )

    if op == "stablehlo.gather":
        emb = _rand(rng, (65536, 512), fp_dtype)
        idx = rng.integers(0, 65536, size=(8192,), dtype=np.int32)
        return lambda e, i: jnp.take(e, i, axis=0), (emb, idx), {}

    if op == "stablehlo.scatter":
        x = _rand(rng, (65536, 512), fp_dtype)
        idx = rng.integers(0, 65536, size=(8192,), dtype=np.int32)
        upd = _rand(rng, (8192, 512), fp_dtype)

        def scatter_add(a, i, u):
            return a.at[i].add(u)

        return scatter_add, (x, idx, upd), {}

    if op in {"stablehlo.dot_general", "stablehlo.dot"}:
        # Attention-shaped batched matmul: [B,H,T,Dh] x [B,H,Dh,T] -> [B,H,T,T]
        B, H, T, Dh = 4, 16, 256, 64
        q = _rand(rng, (B, H, T, Dh), fp_dtype)
        k = _rand(rng, (B, H, T, Dh), fp_dtype)
        return lambda a, b: jnp.matmul(a, jnp.swapaxes(b, -2, -1)), (q, k), {}

    if op == "stablehlo.convolution":
        x = _rand(rng, (8, 128, 128, 64), fp_dtype)
        w = _rand(rng, (3, 3, 64, 64), fp_dtype)
        return (
            lambda a, k: jax.lax.conv_general_dilated(
                a,
                k,
                window_strides=(1, 1),
                padding="SAME",
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            ),
            (x, w),
            {},
        )

    if op == "stablehlo.sort":
        x = _rand(rng, (4096, 256), fp_dtype)
        return lambda a: jax.lax.sort(a, dimension=-1), (x,), {}

    if op == "chlo.top_k" or op == "stablehlo.dynamic_top_k":
        x = _rand(rng, (1024, 1024), fp_dtype)
        return lambda a: jax.lax.top_k(a, 32, axis=-1), (x,), {}

    if op == "stablehlo.fft":
        # Complex64 FFT on 1D length 1024.
        x_re = _rand(rng, (1024,), jnp.float32)
        x_im = _rand(rng, (1024,), jnp.float32)
        x = (x_re + 1j * x_im).astype(np.complex64)
        return lambda a: jax.lax.fft(a, "FFT", (1024,)), (x,), {}

    if op == "stablehlo.complex":
        x = _rand(rng, ELEM_SHAPE, fp_dtype)
        y = _rand(rng, ELEM_SHAPE, fp_dtype)
        return lambda a, b: a + 1j * b, (x, y), {}

    if op == "stablehlo.real":
        x_re = _rand(rng, ELEM_SHAPE, fp_dtype)
        x_im = _rand(rng, ELEM_SHAPE, fp_dtype)
        z = x_re + 1j * x_im
        return jnp.real, (z,), {}

    if op == "stablehlo.imag":
        x_re = _rand(rng, ELEM_SHAPE, fp_dtype)
        x_im = _rand(rng, ELEM_SHAPE, fp_dtype)
        z = x_re + 1j * x_im
        return jnp.imag, (z,), {}

    if op == "stablehlo.iota":
        return lambda: jax.lax.iota(jnp.int32, 1024), (), {}

    if op == "stablehlo.constant":
        return lambda: jnp.asarray(1.0, dtype=fp_dtype), (), {}

    if op == "stablehlo.rng_bit_generator":
        # Key must be u32[4] (or u64[2] with x64 enabled). MPS intentionally errors.
        key = np.asarray([0, 1, 2, 3], dtype=np.uint32)
        return (
            lambda k: jax.lax.rng_bit_generator(k, (1024,), dtype=np.uint32)[1],
            (key,),
            {},
        )

    if op == "stablehlo.case":
        x = _rand(rng, (1024,), fp_dtype)
        pred = np.asarray(True, dtype=np.bool_)
        return (
            lambda p, a: jax.lax.cond(p, lambda t: t + 1, lambda t: t - 1, a),
            (pred, x),
            {},
        )

    if op == "stablehlo.while":
        x = _rand(rng, (1024,), fp_dtype)
        n = np.asarray(16, dtype=np.int32)

        def cond_fun(state):
            i, _, n = state
            return i < n

        def body_fun(state):
            i, a, n = state
            one = jnp.asarray(1.0, dtype=a.dtype)
            return i + 1, a + one, n

        return (
            lambda a, n: jax.lax.while_loop(cond_fun, body_fun, (jnp.asarray(0, jnp.int32), a, n))[1],
            (x, n),
            {},
        )

    if op == "stablehlo.map":
        x = _rand(rng, (1024, 256), fp_dtype)

        def f(v):
            return v * jnp.asarray(1.1, dtype=v.dtype) + jnp.asarray(0.3, dtype=v.dtype)

        return lambda a: jax.lax.map(f, a), (x,), {}

    if op == "stablehlo.triangular_solve":
        # Batched triangular_solve (float16/float32). Use small sizes to keep fast.
        # A: [B,N,N], B: [B,N,M]
        Bn, N, M = 8, 64, 32
        a = rng.standard_normal((Bn, N, N)).astype(np.float32)
        a += np.eye(N, dtype=np.float32)[None, :, :] * 5.0
        if fp_dtype == jnp.float16:
            a = a.astype(np.float16)
        b = _rand(rng, (Bn, N, M), fp_dtype)
        return (
            lambda A, B: jax.lax.linalg.triangular_solve(
                A, B, left_side=True, lower=True, transpose_a=False, conjugate_a=False, unit_diagonal=False
            ),
            (a, b),
            {},
        )

    if op == "stablehlo.cholesky":
        # Batched cholesky (float32 only in plugin).
        Bn, N = 8, 64
        x = rng.standard_normal((Bn, N, N)).astype(np.float32)
        a = x @ np.swapaxes(x, -2, -1) + np.eye(N, dtype=np.float32)[None, :, :] * 1e-1
        return (
            lambda A: jax.lax.linalg.cholesky(A, symmetrize_input=False),
            (a,),
            {},
        )

    # Collectives and shape-polymorphic/dynamic ops are skipped here.
    return None


def main() -> None:
    supported_ops = _extract_supported_ops()

    warmup_iters = 10
    iters = 50

    # For speed, we only run fp16+fp32 for ops where we can reasonably pass float inputs.
    float_dtypes = [jnp.float32, jnp.float16]

    rng = np.random.default_rng(0)

    records: list[dict[str, Any]] = []

    for op in supported_ops:
        for fp_dtype in float_dtypes:
            dtype_label = "fp16" if fp_dtype == jnp.float16 else "fp32"

            case = _make_case(op, rng, fp_dtype)
            if case is None:
                records.append(
                    {
                        "op": op,
                        "dtype": dtype_label,
                        "status": "skip",
                        "note": "no microbench generator for this op in this JAX version",
                    }
                )
                continue

            fn, args, kwargs = case

            # Some ops are not meaningful or not supported for fp16; we'll record errors.
            cpu_res = _time_one(
                fn=fn,
                args=args,
                kwargs=kwargs,
                platform="cpu",
                warmup_iters=warmup_iters,
                iters=iters,
            )
            mps_res = _time_one(
                fn=fn,
                args=args,
                kwargs=kwargs,
                platform="mps",
                warmup_iters=warmup_iters,
                iters=iters,
            )

            rec = {
                "op": op,
                "dtype": dtype_label,
                "cpu": dataclasses.asdict(cpu_res),
                "mps": dataclasses.asdict(mps_res),
            }
            records.append(rec)

    out_jsonl = Path("tmp-files/bottlenecks.jsonl")
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_jsonl.write_text("", encoding="utf-8")
    with out_jsonl.open("a", encoding="utf-8") as fp:
        for r in records:
            fp.write(json.dumps(r, sort_keys=True) + "\n")

    # Summarize into markdown.
    by_op: dict[str, dict[str, dict[str, Any]]] = {}
    for r in records:
        op = r["op"]
        dtype = r["dtype"]
        by_op.setdefault(op, {})[dtype] = r

    def fmt_ms(s: float | None) -> str:
        if s is None:
            return ""
        return f"{s * 1e3:.3f}"

    def ratio(mps_s: float | None, cpu_s: float | None) -> str:
        if mps_s is None or cpu_s is None:
            return ""
        if cpu_s == 0:
            return "inf"
        return f"{mps_s / cpu_s:.2f}x"

    rows: list[tuple[float, str]] = []
    for op, d in by_op.items():
        # Ranking key: worst available mps/cpu ratio among dtypes.
        worst = 0.0
        for dtype in ("fp32", "fp16"):
            rr = d.get(dtype)
            if not rr or rr.get("status") == "skip":
                continue
            cpu = rr["cpu"]
            mps = rr["mps"]
            if cpu.get("ok") and mps.get("ok") and cpu.get("steady_state_s") and cpu["steady_state_s"] > 0:
                worst = max(worst, mps["steady_state_s"] / cpu["steady_state_s"])
        rows.append((worst, op))

    rows.sort(key=lambda t: (-t[0], t[1]))

    out_md = Path("tmp-files/bottlenecks.md")
    lines: list[str] = []
    lines.append("# MPS vs CPU Bottlenecks (fp32/fp16)\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    try:
        lines.append(f"JAX: {jax.__version__}\n")
        import jaxlib  # type: ignore

        lines.append(f"jaxlib: {jaxlib.__version__}\n")
    except Exception:
        pass

    def dev_str(platform: str) -> str:
        try:
            d = jax.devices(platform)[0]
            return str(d)
        except Exception as e:
            return f"<no device: {e}>"

    lines.append(f"CPU device: `{dev_str('cpu')}`\n")
    lines.append(f"MPS device: `{dev_str('mps')}`\n")
    lines.append("\n")

    lines.append("## Summary Table\n")
    lines.append(
        "| op | fp32 cpu (ms) | fp32 mps (ms) | fp32 mps/cpu | fp16 cpu (ms) | fp16 mps (ms) | "
        "fp16 mps/cpu | top StableHLO ops (cpu fp32) | notes |\n"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|---|\n")

    for _, op in rows:
        fp32 = by_op.get(op, {}).get("fp32")
        fp16 = by_op.get(op, {}).get("fp16")

        def cell(rr: dict[str, Any] | None, side: str, field: str) -> str:
            if not rr or rr.get("status") == "skip":
                return ""
            res = rr.get(side, {})
            if not res.get("ok"):
                return "ERR"
            v = res.get(field)
            if v is None:
                return ""
            if field.endswith("_s"):
                return fmt_ms(v)
            return str(v)

        def ratio_cell(rr: dict[str, Any] | None) -> str:
            if not rr or rr.get("status") == "skip":
                return ""
            cpu = rr.get("cpu", {})
            mps = rr.get("mps", {})
            if not cpu.get("ok") or not mps.get("ok"):
                return "ERR"
            return ratio(mps.get("steady_state_s"), cpu.get("steady_state_s"))

        fp32_cpu = cell(fp32, "cpu", "steady_state_s")
        fp32_mps = cell(fp32, "mps", "steady_state_s")
        fp32_r = ratio_cell(fp32)

        fp16_cpu = cell(fp16, "cpu", "steady_state_s")
        fp16_mps = cell(fp16, "mps", "steady_state_s")
        fp16_r = ratio_cell(fp16)

        top = ""
        notes: list[str] = []

        def _short_err(s: str) -> str:
            # Keep markdown tables intact and readable.
            s = re.sub(r"\s+", " ", s).strip()
            s = s.replace("|", "\\|")
            if len(s) > 220:
                s = s[:217] + "..."
            return s

        if fp32 and fp32.get("status") != "skip":
            cpu = fp32.get("cpu", {})
            if cpu.get("ok"):
                top = _top_ops(cpu.get("stablehlo_op_counts"), n=5)

            if not fp32.get("cpu", {}).get("ok"):
                notes.append(f"fp32 cpu err: {_short_err(fp32['cpu']['error'])}")
            if not fp32.get("mps", {}).get("ok"):
                notes.append(f"fp32 mps err: {_short_err(fp32['mps']['error'])}")

        if fp16 and fp16.get("status") != "skip":
            if not fp16.get("cpu", {}).get("ok"):
                notes.append(f"fp16 cpu err: {_short_err(fp16['cpu']['error'])}")
            if not fp16.get("mps", {}).get("ok"):
                notes.append(f"fp16 mps err: {_short_err(fp16['mps']['error'])}")

        if op == "stablehlo.cholesky":
            notes.append("bench uses fp32 inputs (plugin supports fp32 only)")

        if fp32 and fp32.get("status") == "skip" and fp16 and fp16.get("status") == "skip":
            notes.append("no generator")

        lines.append(
            "| "
            + op
            + " | "
            + (fp32_cpu or "")
            + " | "
            + (fp32_mps or "")
            + " | "
            + (fp32_r or "")
            + " | "
            + (fp16_cpu or "")
            + " | "
            + (fp16_mps or "")
            + " | "
            + (fp16_r or "")
            + " | "
            + (top or "")
            + " | "
            + ("; ".join(notes) if notes else "")
            + " |\n"
        )

    lines.append("\n")
    lines.append("## Skipped Ops\n")
    lines.append(
        "Ops that are registered in the plugin but not currently generated by this script (either because they require"
        " multi-device collectives, shape-polymorphic/dynamic shape APIs not present in this JAX version, "
        "or a missing microbench generator).\n\n"
    )

    skipped = []
    for r in records:
        if r.get("status") == "skip":
            skipped.append((r["op"], r["dtype"], r.get("note", "")))
    if skipped:
        lines.append("| op | dtype | reason |\n")
        lines.append("|---|---|---|\n")
        for op, dtype, note in sorted(skipped):
            lines.append(f"| {op} | {dtype} | {note} |\n")

    out_md.write_text("".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
