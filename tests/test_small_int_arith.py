# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer arithmetic must wrap modulo 2**n like CPU.

On MPS: 8-bit add/subtract/multiply saturated instead of wrapping; negate of
8/16-bit integers returned all zeros; and x*x on integers (int16/int32/uint32,
JIT) came back saturated and float32-rounded.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_RAW = np.concatenate(
    [
        np.array([0, 1, -1, 127, -128, 255, 300, 32767, -32768, 65535, 46341, 7621], dtype=np.int64),
        np.random.default_rng(11).integers(-40000, 40000, size=52, dtype=np.int64),
    ]
)
_WIDE = np.array([7621, -7621, 46341, 1 << 20, 3, 2147483647, -2147483648, -1, 0], dtype=np.int64)

_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "square": lambda a, b: a * a,
    "square_via_copy": lambda a, b: a * (a + a.dtype.type(0)),
    "neg": lambda a, b: -a,
}


def _compare(fn, *args, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(*map(jnp.asarray, args)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(*map(jnp.asarray, args)))
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [np.int8, np.uint8, np.int16, np.uint16], ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_small_int_arith_wraps(dtype, op, jit):
    _compare(_OPS[op], _RAW.astype(dtype), _RAW[::-1].astype(dtype), jit=jit)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32], ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("op", ["mul", "square", "square_via_copy", "neg"])
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_32bit_int_mul_and_neg_wrap(dtype, op, jit):
    _compare(_OPS[op], _WIDE.astype(dtype), _WIDE[::-1].astype(dtype), jit=jit)
