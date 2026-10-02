# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Binary ops with a broadcast scalar operand must match XLA:CPU bit for bit.

x + 1.0 lowers to add(x, broadcast_in_dim(constant)); the plugin now feeds the
one-element source to MPSGraph and lets it broadcast implicitly instead of
materialising a full-size tensor. These cases cover both operand orders, the
integer wraparound/divide-by-zero handlers, NaN-propagating max/min, runtime
scalars, broadcasts with other users, and a scan body.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_RNG = np.random.default_rng(21)


def _f32(shape=(64, 257)):
    x = _RNG.standard_normal(shape).astype(np.float32)
    x.reshape(-1)[::31] = np.nan
    x.reshape(-1)[7::37] = -0.0
    return x


def _i(dtype, shape=(64, 257)):
    info = np.iinfo(dtype)
    return _RNG.integers(info.min, int(info.max) + 1, size=shape).astype(dtype)


_CASES = {
    "f32_add_const": (lambda a: a + 1.5, _f32),
    "f32_const_minus": (lambda a: 2.0 - a, _f32),
    "f32_mul_const": (lambda a: a * -3.0, _f32),
    "f32_const_div": (lambda a: 1.0 / a, _f32),
    "f32_max_const": (lambda a: jnp.maximum(a, 0.0), _f32),
    "f32_min_nan_const": (lambda a: jnp.minimum(a, jnp.float32(np.nan)), _f32),
    "f16_add_const": (lambda a: a + jnp.float16(0.1), lambda: _f32().astype(np.float16)),
    "i8_add_wraps": (lambda a: a + jnp.int8(100), lambda: _i(np.int8)),
    "u8_mul_wraps": (lambda a: a * jnp.uint8(7), lambda: _i(np.uint8)),
    "i32_div_zero": (lambda a: a // jnp.int32(0), lambda: _i(np.int32)),
    "i32_const_div": (lambda a: jnp.int32(1000) // jnp.where(a == 0, 1, a), lambda: _i(np.int32)),
    "i32_mul_const": (lambda a: a * jnp.int32(-3), lambda: _i(np.int32)),
    "shared_broadcast": (lambda a: (a + 2.0) * (a - 2.0) + jnp.full(a.shape, 2.0), _f32),
    "rank3": (lambda a: a * 0.5 + 0.25, lambda: _f32((4, 16, 33))),
    "scan_body": (lambda a: lax.scan(lambda c, r: (c * 0.9 + r, c), a[0], a)[1], _f32),
}


def _run(fn, *xs):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(x) for x in xs]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    return actual, expected


def _bits_equal(actual, expected):
    view = {1: np.uint8, 2: np.uint16, 4: np.uint32}[actual.dtype.itemsize]
    np.testing.assert_array_equal(actual.view(view), expected.view(view))


@pytest.mark.parametrize("case", list(_CASES))
def test_scalar_operand_binary_matches_cpu(case):
    fn, make = _CASES[case]
    actual, expected = _run(fn, make())
    if case in ("f32_const_div", "scan_body", "shared_broadcast"):
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
    else:
        _bits_equal(actual, expected)


def test_both_operands_scalar_broadcast():
    # Both sides broadcast from one element: only one may be folded, or the
    # result collapses to a single element and a later reshape aborts.
    def fn(a, b, c):
        y = jnp.broadcast_to(a, (3, 1)) * jnp.broadcast_to(b, (3, 1))
        return jnp.reshape(y - jnp.broadcast_to(c, (3, 1)), (3, 1)) + jnp.ones((3, 1), jnp.float32)

    actual, expected = _run(fn, np.float32(1.5), np.float32(-2.0), np.float32(0.25))
    _bits_equal(actual, expected)


def test_runtime_scalar_operand_matches_cpu():
    # XLA:CPU contracts a * b + b into one FMA; MPS rounds the multiply first.
    # Near a == -1 the result cancels to ~0, so an ulp metric is meaningless
    # there: require MPS to equal unfused float32 bit for bit (NaN payloads
    # aside), and XLA:CPU to agree to one rounding of the product.
    x = _f32()
    s = np.float32(-1.25)
    actual, expected = _run(lambda a, b: a * b + b, x, s)
    with np.errstate(invalid="ignore"):
        unfused = (x * s).astype(np.float32) + s
    nan = np.isnan(unfused)
    np.testing.assert_array_equal(np.isnan(actual), nan)
    _bits_equal(actual[~nan], unfused[~nan])
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
