# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Softmax must match XLA:CPU on both sides of the native/graph routing gate.

Multi-row f32 softmax with rows shorter than 1024 stays in MPSGraph; longer
rows, single rows and f16 use the native kernel. Covers standalone softmax,
softmax between matmuls (attention), and masked (-inf) rows.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_RNG = np.random.default_rng(13)


def _run(fn, *xs, tol):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs)).astype(np.float32)
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(x) for x in xs])).astype(np.float32)
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("shape", [(64, 128), (32, 512), (16, 1023), (8, 1024), (4, 4096), (1, 300), (2, 8, 128)])
def test_softmax_matches_cpu(shape, dtype):
    x = (_RNG.standard_normal(shape) * 3).astype(dtype)
    x.reshape(-1)[::17] = -np.inf
    tol = 1e-5 if dtype == np.float32 else 2e-3
    _run(lambda a: jax.nn.softmax(a, axis=-1), x, tol=tol)


@pytest.mark.parametrize("dtype", [np.float32, np.float16], ids=["f32", "f16"])
@pytest.mark.parametrize("t", [128, 1024])
def test_attention_matches_cpu(t, dtype):
    q, k, v = ((_RNG.standard_normal((2, 4, t, 32)) * 0.5).astype(dtype) for _ in range(3))

    def attention(q, k, v):
        logits = jnp.matmul(q, jnp.swapaxes(k, -2, -1)) * dtype(32) ** dtype(-0.5)
        return jnp.matmul(jax.nn.softmax(logits, axis=-1), v)

    tol = 1e-4 if dtype == np.float32 else 5e-3
    _run(attention, q, k, v, tol=tol)
