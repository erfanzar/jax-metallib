# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer divide, right shifts and abs must match XLA CPU bit-for-bit.

On MPS: 8-bit divide returned garbage (int8 100/5 -> 0, -1/2 -> -1; uint8
254/3 -> 0); MPSGraph right shift followed the dtype's signedness instead of
the op, so shift_right_logical on signed ints sign-extended and
shift_right_arithmetic on unsigned ints zero-filled; int8 abs(-128) saturated
to 127 instead of wrapping to -128.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]


def _values(dtype):
    info = np.iinfo(dtype)
    raw = np.array(
        [0, 1, -1, 2, -2, 7, -7, 100, -100, info.max, info.min, info.max - 1, info.min + 1, 3, -3, 5],
        dtype=np.int64,
    )
    return np.clip(raw, info.min, info.max).astype(dtype)


def _divisors(dtype):
    info = np.iinfo(dtype)
    raw = np.array([3, -3, 2, -2, 7, 1, -1, 5, 3, 2, -1, 3, 7, 5, 2, -4], dtype=np.int64)
    out = np.clip(raw, info.min, info.max)
    return np.where(out == 0, 1, out).astype(dtype)


def _shifts(dtype, count):
    # Includes 0, the full width and an over-wide amount (overflow semantics).
    bits = np.iinfo(dtype).bits
    raw = np.array([0, 1, 2, 3, bits - 1, bits, bits + 3, 4], dtype=np.int64)
    return np.resize(raw, count).astype(dtype)


_OPS = {
    "div": (lambda a, b: lax.div(a, b), _divisors),
    "floordiv": (lambda a, b: a // b, _divisors),
    "shr_logical": (lambda a, b: lax.shift_right_logical(a, b), None),
    "shr_arith": (lambda a, b: lax.shift_right_arithmetic(a, b), None),
    "shl": (lambda a, b: lax.shift_left(a, b), None),
    "abs": (lambda a, b: jnp.abs(a), _divisors),
}


def _compare(fn, *args, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(*map(jnp.asarray, args)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(*map(jnp.asarray, args)))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_int_elementwise_matches_cpu(dtype, op, jit):
    fn, second = _OPS[op]
    a = _values(dtype)
    b = second(dtype) if second else _shifts(dtype, a.size)
    _compare(fn, a, b, jit=jit)


_DIV_OPS = {
    "div": lambda a, b: lax.div(a, b),
    "floordiv": lambda a, b: a // b,
    "rem": lambda a, b: lax.rem(a, b),
    "mod": lambda a, b: a % b,
}


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op", list(_DIV_OPS))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_int_divide_by_zero_matches_cpu(dtype, op, jit):
    # XLA: integer x / 0 is all ones (-1 signed, max unsigned), x % 0 is x, and
    # INT_MIN / -1 wraps. MPS returned sign-dependent values for x / 0.
    info = np.iinfo(dtype)
    a = np.clip(np.array([0, 1, -1, 7, -7, info.max, info.min, info.min, 5, -5]), info.min, info.max)
    b = np.clip(np.array([0, 0, 0, 0, 0, 0, 0, -1, -1, 3]), info.min, info.max)
    _compare(_DIV_OPS[op], a.astype(dtype), b.astype(dtype), jit=jit)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op", ["shr_logical", "shr_arith"])
def test_int_right_shift_scalar(dtype, op):
    fn, _ = _OPS[op]
    info = np.iinfo(dtype)
    a = np.asarray(info.min if info.min < 0 else info.max, dtype=dtype)
    _compare(fn, a, np.asarray(2, dtype=dtype), jit=True)
