# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""hessenberg and tridiagonal reductions must run on MPS and match XLA:CPU.

Neither primitive had an MPS lowering. Both now lower to a Householder
similarity reduction on float32 (re, im) parts with LAPACK's zlarfg convention,
so the packed outputs (reduced matrix plus reflector tails) and taus match
LAPACK gehrd / sytrd / hetrd elementwise.
"""

import jax
import jax.scipy.linalg as jsl
import numpy as np
import pytest
from jax import numpy as jnp
from jax._src.lax import linalg as lax_linalg  # hessenberg/tridiagonal primitives

_RNG = np.random.default_rng(53)


def _mat(shape, dtype):
    x = _RNG.standard_normal(shape)
    if dtype == "c64":
        return (x + 1j * _RNG.standard_normal(shape)).astype(np.complex64)
    return x.astype(np.float32)


def _on(platform, fn, *xs):
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, jax.jit(fn)(*[jnp.asarray(x) for x in xs]))


def _check(fn, *xs, tol=3e-4):
    expected = _on("cpu", fn, *xs)
    actual = _on("mps", fn, *xs)
    for a, e in zip(jax.tree_util.tree_leaves(actual), jax.tree_util.tree_leaves(expected), strict=True):
        assert a.shape == e.shape and a.dtype == e.dtype
        np.testing.assert_allclose(a, e, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", ["f32", "c64"])
@pytest.mark.parametrize("shape", [(1, 1), (2, 2), (5, 5), (8, 8), (3, 6, 6)], ids=lambda s: "x".join(map(str, s)))
def test_hessenberg_matches_cpu(shape, dtype):
    _check(lax_linalg.hessenberg, _mat(shape, dtype))


@pytest.mark.parametrize("dtype", ["f32", "c64"])
def test_scipy_hessenberg_with_q(dtype):
    _check(lambda a: jsl.hessenberg(a, calc_q=True), _mat((6, 6), dtype))


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
@pytest.mark.parametrize("dtype", ["f32", "c64"])
@pytest.mark.parametrize("shape", [(1, 1), (4, 4), (7, 7), (2, 5, 5)], ids=lambda s: "x".join(map(str, s)))
def test_tridiagonal_matches_cpu(shape, dtype, lower):
    # Not Hermitian: only the requested triangle may be read.
    _check(lambda a: lax_linalg.tridiagonal(a, lower=lower), _mat(shape, dtype))
