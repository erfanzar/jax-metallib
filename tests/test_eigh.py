# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""jnp.linalg.eigh / eigvalsh must run on MPS and match XLA CPU.

JAX has no eigh lowering rule for platform "mps", so every eigh failed at
lowering time. The plugin registers a pure-JAX Jacobi lowering. Eigenvalues are
compared with CPU; eigenvectors are only defined up to sign (and rotation within
degenerate eigenspaces), so they are checked through reconstruction and
orthogonality instead.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp


def _sym(n, batch=(), seed=1):
    m = np.random.default_rng(seed).standard_normal((*batch, n, n)).astype(np.float32)
    return (m + np.swapaxes(m, -1, -2)) / 2


_MATS = {
    "n2": _sym(2),
    "n5": _sym(5),
    "n8": _sym(8),
    "n33": _sym(33),
    "batched_3x6": _sym(6, (3,)),
    "degenerate": np.diag(np.array([2.0, 2.0, -1.0, 5.0], np.float32)),
}


def _on(platform, fn, *args, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, func(*map(jnp.asarray, args)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_eigvalsh_matches_cpu(mat, jit):
    x = _MATS[mat]
    expected = _on("cpu", jnp.linalg.eigvalsh, x, jit=jit)
    actual = _on("mps", jnp.linalg.eigvalsh, x, jit=jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_eigh_vectors_reconstruct(mat, jit):
    x = _MATS[mat]
    w, v = _on("mps", jnp.linalg.eigh, x, jit=jit)
    eye = np.broadcast_to(np.eye(x.shape[-1], dtype=np.float32), x.shape)
    np.testing.assert_allclose(np.swapaxes(v, -1, -2) @ v, eye, atol=5e-5)
    np.testing.assert_allclose((v * w[..., None, :]) @ np.swapaxes(v, -1, -2), x, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("lower", [True, False], ids=["lower", "upper"])
def test_eigh_reads_requested_triangle(lower):
    # Non-symmetric input: only the requested triangle may be used.
    x = np.random.default_rng(4).standard_normal((5, 5)).astype(np.float32)

    def fn(a):
        return jax.lax.linalg.eigh(a, lower=lower, symmetrize_input=False)[1]

    expected = _on("cpu", fn, x, jit=True)
    actual = _on("mps", fn, x, jit=True)
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_eigvalsh_grad_matches_cpu(jit):
    x = _sym(5)
    grad = jax.grad(lambda a: (jnp.linalg.eigvalsh(a) * jnp.arange(5.0)).sum())
    expected = _on("cpu", grad, x, jit=jit)
    actual = _on("mps", grad, x, jit=jit)
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-4)
