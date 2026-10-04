# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Large float16 dots must keep XLA:CPU semantics when run as native f16 GEMMs.

f16 x f16 dots were widened to f32 because MPSGraph returned inf for results in
[60000, 65504]. A standalone Level0 probe showed native f16 matmul is exact
there, accumulates in f32 and is 1.2-2.7x faster at MLP/GPT-2 sizes, so large
dots skip the widening. These cases guard overflow, accumulation precision and
the error bound, including inside a scan body.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _near_max(m, k, n):
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, (m, k)).astype(np.float16)
    b = rng.uniform(-1, 1, (k, n)).astype(np.float16)
    a[0, :] = 1
    b[:, 0] = np.float16(62000.0 / k)
    a[1, :] = 0
    a[1, 0] = 60000
    b[0, 1] = 1
    a[2, :] = 1
    b[:, 2] = np.float16(66000.0 / k)
    a[3, :] = 0
    a[3, 0] = -60000
    b[0, 3] = 1
    return a, b


def _accum(k=4096):
    a = np.zeros((8, k), np.float16)
    b = np.zeros((k, 8), np.float16)
    a[0, :] = 1
    b[:, 0] = 1
    a[1, 0] = 1
    b[0, 1] = 1
    a[1, 1:1001] = np.float16(2.0**-12)
    b[1:1001, 1] = 1
    return a, b


_SHAPES = [(128, 256, 768), (512, 768, 3072), (1024, 1024, 1024), (4096, 2048, 512)]


def _run(fn, *args):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*args))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(a) for a in args]))
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    return actual, expected


def _check_bound(actual, expected, a, b):
    # f32 accumulation in any order stays within ~2^-11 of sum|a*b|, plus the
    # final f16 rounding of the output (another 2^-11 relative).
    bound = np.abs(a.astype(np.float32)) @ np.abs(b.astype(np.float32))
    finite = np.isfinite(expected)
    np.testing.assert_array_equal(np.isinf(actual), np.isinf(expected))
    err = np.abs(actual.astype(np.float64) - expected.astype(np.float64))[finite]
    assert np.all(err <= 2.0**-10 * bound[finite] + 1e-6), float(np.max(err / (bound[finite] + 1e-30)))


@pytest.mark.parametrize("shape", _SHAPES, ids=["x".join(map(str, s)) for s in _SHAPES])
def test_large_f16_dot_near_max_and_bound(shape):
    a, b = _near_max(*shape)
    actual, expected = _run(lambda x, y: x @ y, a, b)
    for i in range(4):
        np.testing.assert_array_equal(actual[i, i], expected[i, i])
    assert np.isinf(actual[2, 2]) and actual[1, 1] == 60000 and actual[3, 3] == -60000
    _check_bound(actual, expected, a, b)


def test_large_f16_dot_accumulates_in_f32():
    a, b = _accum()
    actual, expected = _run(lambda x, y: x @ y, a, b)
    assert actual[0, 0] == 4096 and expected[0, 0] == 4096
    np.testing.assert_array_equal(actual[1, 1], expected[1, 1])


def test_large_f16_batched_dot_bound():
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, (16, 256, 512)).astype(np.float16)
    w = rng.uniform(-1, 1, (512, 2048)).astype(np.float16)
    actual, expected = _run(lambda x, y: x @ y, a, w)
    _check_bound(actual.reshape(-1, 2048), expected.reshape(-1, 2048), a.reshape(-1, 512), w)


def test_large_f16_dot_in_scan_body():
    rng = np.random.default_rng(2)
    ws = rng.uniform(-0.1, 0.1, (4, 256, 256)).astype(np.float16)
    x = rng.uniform(-1, 1, (128, 256)).astype(np.float16)

    def fn(x, ws):
        def step(h, w):
            h = jnp.tanh(h @ w)
            return h, h.sum()

        return lax.scan(step, x, ws)

    h_cpu, s_cpu = jax.jit(fn, backend="cpu")(x, ws)
    with jax.default_device(jax.devices("mps")[0]):
        h_mps, s_mps = jax.jit(fn)(jnp.asarray(x), jnp.asarray(ws))
    np.testing.assert_allclose(np.asarray(h_mps, np.float32), np.asarray(h_cpu, np.float32), atol=4e-3, rtol=4e-3)
    np.testing.assert_allclose(np.asarray(s_mps, np.float32), np.asarray(s_cpu, np.float32), rtol=1e-2, atol=0.5)


def test_large_f16_dot_preferred_f32_is_unrounded():
    # preferred_element_type=f32 must return the f32 sum, not an f16-rounded one.
    rng = np.random.default_rng(3)
    a = rng.uniform(-1, 1, (512, 1024)).astype(np.float16)
    b = rng.uniform(-1, 1, (1024, 512)).astype(np.float16)
    actual, expected = _run(lambda x, y: lax.dot(x, y, preferred_element_type=jnp.float32), a, b)
    assert actual.dtype == np.float32
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)
