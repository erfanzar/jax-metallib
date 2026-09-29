# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""jnp.linalg.svd / pinv / lstsq must run on MPS and match XLA CPU.

JAX's generic svd lowering runs QDWH, which calls stablehlo.cholesky inside
while bodies; Cholesky is native-only on MPS, so every svd, pinv and lstsq
failed with "Native op 'stablehlo.cholesky' encountered during graph building".
The plugin registers a one-sided Jacobi SVD for platform "mps". Singular values
are compared with CPU; singular vectors are only defined up to sign, so they are
checked by reconstruction and orthogonality.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp


def _mat(shape, seed=2):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _rank_deficient():
    b = _mat((6, 2), seed=5)
    return (b @ _mat((2, 4), seed=6)).astype(np.float32)


_MATS = {
    "tall_6x4": _mat((6, 4)),
    "square_5x5": _mat((5, 5)),
    "wide_3x7": _mat((3, 7)),
    "batched_2x5x3": _mat((2, 5, 3)),
    "column_4x1": _mat((4, 1)),
    "rank_deficient_6x4": _rank_deficient(),
}


def _on(platform, fn, *args, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return jax.tree_util.tree_map(np.asarray, func(*map(jnp.asarray, args)))


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_singular_values_match_cpu(mat, jit):
    x = _MATS[mat]
    fn = lambda a: jnp.linalg.svd(a, compute_uv=False)  # noqa: E731
    expected = _on("cpu", fn, x, jit=jit)
    actual = _on("mps", fn, x, jit=jit)
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=5e-5)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("full", [False, True], ids=["reduced", "full"])
@pytest.mark.parametrize("mat", list(_MATS))
def test_svd_vectors_reconstruct(mat, full, jit):
    x = _MATS[mat]
    fn = lambda a: jnp.linalg.svd(a, full_matrices=full)  # noqa: E731
    u_cpu, s_cpu, vt_cpu = _on("cpu", fn, x, jit=jit)
    u, s, vt = _on("mps", fn, x, jit=jit)
    assert u.shape == u_cpu.shape and s.shape == s_cpu.shape and vt.shape == vt_cpu.shape
    k = s.shape[-1]
    recon = (u[..., :, :k] * s[..., None, :]) @ vt[..., :k, :]
    np.testing.assert_allclose(recon, x, rtol=1e-4, atol=1e-4)
    for mtx in (u, np.swapaxes(vt, -1, -2)):
        gram = np.swapaxes(mtx, -1, -2) @ mtx
        np.testing.assert_allclose(gram, np.broadcast_to(np.eye(gram.shape[-1]), gram.shape), atol=1e-4)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_pinv_and_lstsq_match_cpu(jit):
    a = _mat((6, 3), seed=9)
    b = _mat((6,), seed=10)
    for fn, args in (
        (jnp.linalg.pinv, (a,)),
        (lambda m, y: jnp.linalg.lstsq(m, y)[0], (a, b)),
    ):
        np.testing.assert_allclose(_on("mps", fn, *args, jit=jit), _on("cpu", fn, *args, jit=jit), rtol=1e-3, atol=1e-4)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_singular_value_grad_matches_cpu(jit):
    x = _mat((5, 4), seed=12)
    grad = jax.grad(lambda a: (jnp.linalg.svd(a, compute_uv=False) * jnp.arange(1.0, 5.0)).sum())
    np.testing.assert_allclose(_on("mps", grad, x, jit=jit), _on("cpu", grad, x, jit=jit), rtol=1e-3, atol=1e-4)
