# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Small f32 sorts run on the native bitonic kernel and must stay bit-exact.

Sorts under 100000 elements (unless rows > 1 and cols > 256) used to take the
exact MPSGraph argsort + gather path, 6-22x slower ([4096,16] 4.95 ms vs
0.22 ms native). XLA:CPU returns the original bits in stable order, so NaN
payloads and the sign of zero must survive; the flip-sort-flip variant forces
a graph/native/graph plan split.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_SPECIAL = np.array([np.nan, -np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], np.float32)
_SHAPES = [(4096, 16), (1024, 64), (256, 256), (64, 64), (3, 5), (5, 1), (1, 1), (4096,), (2000,), (7,), (1,)]


def _data(shape, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    special = rng.random(shape) < 0.4
    x[special] = rng.choice(_SPECIAL, size=int(special.sum()))
    nan = np.isnan(x)
    x.view(np.uint32)[nan] |= rng.integers(1, 1 << 22, size=int(nan.sum()), dtype=np.uint32)
    return x


def _run(fn, x):
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_small_sort_bit_exact(shape):
    _run(lambda a: lax.sort(a, dimension=-1), _data(shape))


@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_small_sort_between_graph_ops(shape):
    _run(lambda a: jnp.flip(lax.sort(jnp.flip(a, -1), dimension=-1), -1), _data(shape, seed=1))
