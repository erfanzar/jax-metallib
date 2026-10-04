# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex QR (geqrf + householder_product) must run on MPS and match XLA:CPU.

The native Qr / ProductOfElementaryHouseholderReflectors handlers are
float32-only, so complex jnp.linalg.qr failed. Complex inputs now lower to an
unblocked Householder QR on float32 (re, im) parts with LAPACK's convention
(real beta, v[0] = 1), so the packed factors, Q and R match CPU directly.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp
from jax._src.lax import linalg as lax_linalg  # geqrf is not in the public jax.lax.linalg

_RNG = np.random.default_rng(41)


def _c(shape):
    return (_RNG.standard_normal(shape) + 1j * _RNG.standard_normal(shape)).astype(np.complex64)


def _on(platform, fn, *xs, jit=True):
    f = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, f(*[jnp.asarray(x) for x in xs]))


def _check(fn, *xs, jit=True, tol=2e-4):
    expected = _on("cpu", fn, *xs)
    actual = _on("mps", fn, *xs, jit=jit)
    for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        assert a.shape == e.shape and a.dtype == e.dtype
        np.testing.assert_allclose(a, e, rtol=tol, atol=tol)


_SHAPES = {"5x5": (5, 5), "7x4": (7, 4), "4x7": (4, 7), "batched_3x6x5": (3, 6, 5), "1x1": (1, 1)}


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mode", ["reduced", "complete", "r"])
@pytest.mark.parametrize("shape", list(_SHAPES))
def test_complex_qr_matches_cpu(shape, mode, jit):
    _check(lambda a: jnp.linalg.qr(a, mode=mode), _c(_SHAPES[shape]), jit=jit)


@pytest.mark.parametrize("shape", list(_SHAPES))
def test_complex_geqrf_matches_cpu(shape):
    _check(lax_linalg.geqrf, _c(_SHAPES[shape]))


def test_complex_qr_rank_deficient_reconstructs():
    # A zero column gives a trivial reflector (tau = 0); Q must stay unitary.
    x = _c((6, 4))
    x[:, 1] = 0
    q, r = _on("mps", jnp.linalg.qr, x)
    np.testing.assert_allclose(np.conj(q.T) @ q, np.eye(4), atol=2e-4)
    np.testing.assert_allclose(q @ r, x, atol=2e-4)


def test_real_qr_unchanged():
    _check(jnp.linalg.qr, _RNG.standard_normal((6, 4)).astype(np.float32))
