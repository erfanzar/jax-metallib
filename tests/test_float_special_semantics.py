# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Float special-value semantics must match XLA CPU.

On MPS: tanh(nan) returned 1; tanh and expm1 flushed tiny inputs (1e-8) to 0
and dropped the sign of -0; expm1(0.1) was ~5e-7 off relative (exp(x) - 1
cancellation); sign(nan) returned 0 and sign(-0) returned +0; and
stablehlo.round_nearest_afz / stablehlo.reduce_precision had no handler.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = [np.float32, np.float16, jnp.bfloat16]
_IDS = ["float32", "float16", "bfloat16"]

_VALS = np.array(
    [
        *(np.nan, -0.0, 0.0, 1e-8, -1e-8, 1e-30, -3e-5, 0.01, -0.04, 0.1, 0.5, -1.0, 3.0, -7.5, 20.0, -20.0),
        *(100.0, -100.0, np.inf, -np.inf),
    ],
    dtype=np.float32,
)
_ROUND_VALS = np.array(
    [
        *(0.5, -0.5, 1.5, 2.5, -2.5, 0.49999997, -0.49999997, -0.3, 0.3, -0.0, 0.0, 3.7, -3.7, 8388609.0, 1e30),
        *(np.nan, np.inf, -np.inf),
    ],
    dtype=np.float32,
)

_UNARY = {
    "tanh": (jnp.tanh, _VALS, True),
    "sign": (jnp.sign, _VALS, False),
    "expm1": (jnp.expm1, _VALS, True),
    "round_afz": (lambda a: lax.round(a, lax.RoundingMethod.AWAY_FROM_ZERO), _ROUND_VALS, False),
}


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return np.asarray(func(jnp.asarray(x)))


def _assert_same(expected, actual, rtol):
    assert actual.dtype == expected.dtype
    e, a = expected.astype(np.float64), actual.astype(np.float64)
    np.testing.assert_array_equal(np.isnan(a), np.isnan(e))
    keep = ~np.isnan(e)
    if rtol:
        np.testing.assert_allclose(a[keep], e[keep], rtol=rtol, atol=0)
    else:
        np.testing.assert_array_equal(a[keep], e[keep])
    zero = keep & (e == 0)
    np.testing.assert_array_equal(np.signbit(a[zero]), np.signbit(e[zero]))


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
@pytest.mark.parametrize("op", list(_UNARY))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_float_special_values_match_cpu(dtype, op, jit):
    fn, vals, approx = _UNARY[op]
    x = vals.astype(dtype)
    rtol = 4 * float(jnp.finfo(dtype).eps) if approx else 0
    _assert_same(_run(fn, x, "cpu", jit), _run(fn, x, "mps", jit), rtol)


_RP_BITS = [(5, 10), (8, 7), (4, 3), (3, 2), (5, 2), (8, 23)]


@pytest.mark.parametrize("dtype", _DTYPES, ids=_IDS)
@pytest.mark.parametrize("bits", _RP_BITS, ids=lambda b: f"e{b[0]}m{b[1]}")
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_reduce_precision_matches_cpu(dtype, bits, jit):
    exponent_bits, mantissa_bits = bits
    rng = np.random.default_rng(3)
    mags = 10.0 ** rng.integers(-9, 9, size=96)
    specials = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1e-40, -1e-40, 6e-8, 65504.0, 70000.0, 3.4e38])
    x = np.concatenate([rng.standard_normal(96) * mags, specials]).astype(np.float32).astype(dtype)

    def fn(a):
        return lax.reduce_precision(a, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits)

    _assert_same(_run(fn, x, "cpu", jit), _run(fn, x, "mps", jit), 0)
