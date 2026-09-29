# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""stablehlo.optimization_barrier must lower on MPS.

jax.checkpoint (remat) emits optimization_barrier in its gradients, so every
rematerialized gradient failed on MPS with "Unsupported operation(s):
stablehlo.optimization_barrier". The op is an identity over its operands.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _run(fn, args, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        out = func(*map(jnp.asarray, args))
        return jax.tree_util.tree_map(np.asarray, out)


def _compare(fn, *args, jit, rtol=1e-5):
    expected = _run(fn, args, "cpu", jit)
    actual = _run(fn, args, "mps", jit)
    for e, a in zip(jax.tree_util.tree_leaves(expected), jax.tree_util.tree_leaves(actual), strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        np.testing.assert_allclose(a, e, rtol=rtol, atol=1e-6)


_X = np.linspace(-1.5, 2.0, 24, dtype=np.float32).reshape(4, 6)


def _remat_loss(x):
    g = jax.checkpoint(lambda y: jnp.sin(y) * y + jnp.tanh(y @ y.T).sum())
    return g(x).sum()


def _remat_mlp(x):
    w = jnp.ones((6, 6), jnp.float32) * 0.1

    @jax.checkpoint
    def layer(h):
        return jax.nn.gelu(h @ w)

    return layer(layer(x)).sum()


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("loss", [_remat_loss, _remat_mlp], ids=["elementwise", "mlp"])
def test_remat_grad_matches_cpu(loss, jit):
    _compare(jax.grad(loss), _X, jit=jit, rtol=1e-4)


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_variadic_barrier_mixed_dtypes(jit):
    def fn(a, b, c):
        a2, b2, c2 = lax.optimization_barrier((a * 2, b + 1, c))
        return a2 + 1, b2 * 3, c2

    _compare(fn, _X, np.arange(5, dtype=np.int32), np.asarray(True), jit=jit)


def test_barrier_inside_scan():
    def fn(x):
        def body(carry, _):
            (carry,) = lax.optimization_barrier((carry * 1.5 - 0.25,))
            return carry, carry.sum()

        return lax.scan(body, x, None, length=4)

    _compare(fn, _X, jit=True)
