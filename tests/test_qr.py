# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""jnp.linalg.qr must run on MPS and match XLA CPU.

JAX lowers QR to two custom calls: Qr (LAPACK geqrf: packed Householder
vectors + tau) and ProductOfElementaryHouseholderReflectors (orgqr). Both were
unsupported, so every QR failed with "Unsupported custom call target 'Qr'".
The MPS lowering follows LAPACK's sign conventions, so Q and R match CPU
directly, not only up to column signs.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_SHAPES = {
    "tall_4x3": (4, 3),
    "square_5x5": (5, 5),
    "wide_3x4": (3, 4),
    "tall_9x4": (9, 4),
    "batched_2x6x4": (2, 6, 4),
    "col_vector_5x1": (5, 1),
}


def _matrix(shape):
    rng = np.random.default_rng(11)
    return rng.standard_normal(shape).astype(np.float32)


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return [np.asarray(r) for r in func(jnp.asarray(x))]


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_qr_matches_cpu(shape, mode, jit):
    x = _matrix(_SHAPES[shape])

    def fn(a):
        out = jnp.linalg.qr(a, mode=mode)
        return (out,) if mode == "r" else tuple(out)

    expected = _run(fn, x, "cpu", jit)
    actual = _run(fn, x, "mps", jit)
    for e, a in zip(expected, actual, strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        np.testing.assert_allclose(a, e, rtol=2e-4, atol=2e-5)
    if mode != "r":
        q, r = actual
        np.testing.assert_allclose(q @ r, x, rtol=2e-4, atol=2e-5)


_GRAD_LOSSES = {
    "r_sum": (lambda a: jnp.linalg.qr(a)[1].sum(), (5, 4)),
    "q_weighted": (lambda a: (jnp.linalg.qr(a)[0] ** 2 * jnp.arange(4.0)).sum(), (5, 4)),
    "r_diag_square": (lambda a: jnp.linalg.qr(a)[1].diagonal().sum(), (4, 4)),
}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("loss", list(_GRAD_LOSSES))
def test_qr_grad_matches_cpu(loss, jit):
    fn, shape = _GRAD_LOSSES[loss]
    x = _matrix(shape)
    grad = jax.grad(fn)
    func = jax.jit(grad) if jit else grad
    out = {}
    for p in ("cpu", "mps"):
        with jax.default_device(jax.devices(p)[0]):
            out[p] = np.asarray(func(jnp.asarray(x)))
    np.testing.assert_allclose(out["mps"], out["cpu"], rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_qr_rank_deficient_matches_cpu(jit):
    # A zero column below the diagonal exercises the tau = 0 (identity reflector) branch.
    x = np.array([[1.0, 2.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]], np.float32)
    expected = _run(jnp.linalg.qr, x, "cpu", jit)
    actual = _run(jnp.linalg.qr, x, "mps", jit)
    for e, a in zip(expected, actual, strict=True):
        np.testing.assert_allclose(a, e, rtol=2e-4, atol=2e-5)
