# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Integer remainder by zero must return the dividend (XLA: x % 0 == x).

With a runtime zero divisor MPSGraph modulo was already correct, but with a
compile-time-constant zero divisor it returned junk ([1, 1, 0, 1, ...] for
[7, -7, 0, 5, ...]), which also broke JAX's floor_divide correction: a // 0
gave -1 everywhere instead of -2 for nonzero a.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = [np.int8, np.uint8, np.int16, np.int32, np.uint32]


def _data(dtype):
    info = np.iinfo(dtype)
    vals = [7, -7, 0, 5, -1, 3, int(info.min), int(info.max)]
    return np.clip(np.array(vals * 4, np.int64), info.min, info.max).astype(dtype).reshape(4, 8)


def _run(fn, *xs):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(x) for x in xs]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_rem_by_constant_zero(dtype):
    _run(lambda a: lax.rem(a, jnp.zeros(a.shape, a.dtype)), _data(dtype))


@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_rem_by_constant_mixed_divisor(dtype):
    div = np.array([3, 0, 2, 0, 5, 0, 1, 0] * 4, np.int64).reshape(4, 8).astype(dtype)
    _run(lambda a: lax.rem(a, jnp.asarray(div)), _data(dtype))


@pytest.mark.parametrize("dtype", [np.int32, np.int16, np.uint32], ids=lambda d: np.dtype(d).name)
def test_floor_divide_and_mod_by_constant_zero(dtype):
    zero = np.array(0, dtype)
    _run(lambda a: a // zero, _data(dtype))
    _run(lambda a: a % zero, _data(dtype))


def test_rem_by_runtime_zero():
    _run(lambda a, b: lax.rem(a, b), _data(np.int32), np.zeros((4, 8), np.int32))
