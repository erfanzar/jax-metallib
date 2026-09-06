"""Regression tests for fused-pattern shapes that crashed or miscomputed.

The softmax fusion used to build a shape-invalid MPSGraph for 2-D inputs
(abort during executable verification), so these shapes are pinned explicitly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _softmax_ref(t: np.ndarray) -> np.ndarray:
    shifted = t - t.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


@pytest.mark.parametrize("shape", [(3,), (3, 7), (4096, 64), (128, 4096)])
def test_softmax_2d_matches_reference(shape: tuple[int, ...]) -> None:
    t = jnp.asarray(
        np.random.default_rng(7).normal(size=shape).astype(np.float32),
    )
    out = jax.jit(lambda x: jax.nn.softmax(x, axis=-1))(t)
    np.testing.assert_allclose(np.asarray(out), _softmax_ref(np.asarray(t)), atol=1e-5, rtol=1e-5)


def test_layernorm_2d_matches_reference() -> None:
    t = np.asarray(
        jnp.asarray(np.random.default_rng(11).normal(size=(4096, 64)).astype(np.float32)),
    )
    out = jax.jit(
        lambda x: (x - x.mean(axis=-1, keepdims=True))
        / jnp.sqrt(((x - x.mean(axis=-1, keepdims=True)) ** 2).mean(axis=-1, keepdims=True) + 1e-5),
    )(jnp.asarray(t))
    m = t.mean(axis=-1, keepdims=True)
    v = ((t - m) ** 2).mean(axis=-1, keepdims=True)
    np.testing.assert_allclose(np.asarray(out), (t - m) / np.sqrt(v + 1e-5), atol=1e-5, rtol=1e-5)
