# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Shifts by a constant amount must match XLA:CPU bit for bit.

x >> k with a constant k lowers to shift(x, broadcast_in_dim(constant k)). An
in-range splat k is now fed to MPSGraph as a [1,...,1] constant without the
k >= bitWidth overflow select (that select over the full-size constant made
MPSGraph fold n elements one APInt at a time on the first run: ~11 s for 20
rotate rounds on 4M uint32). Out-of-range and negative amounts keep the
guarded path. These cases cover every in-range amount, the guard boundary,
rank 0, an implicitly broadcast lhs (fold not taken) and shifts inside scan.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
_OPS = {
    "shl": lax.shift_left,
    "shr_logical": lax.shift_right_logical,
    "shr_arith": lax.shift_right_arithmetic,
}


def _data(dtype, shape=(4, 8)):
    info = np.iinfo(dtype)
    vals = [0, 1, -1, 7, -7, 0x55, int(info.min), int(info.max)]
    n = int(np.prod(shape)) if shape else 1
    base = np.array((vals * (n // len(vals) + 1))[:n], np.int64)
    return np.clip(base, info.min, info.max).astype(dtype).reshape(shape)


def _amounts(dtype):
    bits = np.iinfo(dtype).bits
    ks = [*range(bits), bits, bits + 3]
    ks.append(-1 if np.issubdtype(dtype, np.signedinteger) else int(np.iinfo(dtype).max))
    return ks


def _run(fn, *xs):
    expected = np.asarray(jax.jit(fn, backend="cpu")(*xs))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(*[jnp.asarray(x) for x in xs]))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)


def _all_amounts(op, dtype):
    ks = _amounts(dtype)

    def f(x):
        return jnp.stack([op(x, jnp.full(x.shape, k, dtype)) for k in ks])

    return f


@pytest.mark.parametrize("op", list(_OPS), ids=str)
@pytest.mark.parametrize("dtype", _DTYPES, ids=lambda d: np.dtype(d).name)
def test_constant_shift_all_amounts(dtype, op):
    _run(_all_amounts(_OPS[op], dtype), _data(dtype))


@pytest.mark.parametrize("op", list(_OPS), ids=str)
@pytest.mark.parametrize("dtype", [np.int8, np.uint16, np.int32, np.uint32], ids=lambda d: np.dtype(d).name)
def test_constant_shift_rank0(dtype, op):
    f = _all_amounts(_OPS[op], dtype)
    for v in _data(dtype, (8,)):
        _run(f, np.asarray(v, dtype))


@pytest.mark.parametrize("dtype", [np.int32, np.uint32, np.int8], ids=lambda d: np.dtype(d).name)
def test_scalar_lhs_broadcast_against_amounts(dtype):
    # lhs is broadcast to the amount's shape: the input does not have the output
    # shape, so the fold must not apply.
    ks = np.array(_amounts(dtype), np.int64).astype(dtype)
    for v in _data(dtype, (8,)):
        x = np.asarray(v, dtype)
        _run(lambda a, x=x: jnp.left_shift(x, a), ks)
        _run(lambda a, x=x: jnp.right_shift(x, a), ks)


def test_rotate_chain_in_scan():
    def rot(v, r):
        return (v << jnp.uint32(r)) | (v >> jnp.uint32(32 - r))

    def f(x):
        def body(v, i):
            v = rot(v, 13) ^ (v + i.astype(jnp.uint32))
            v = rot(v, 7) + (v >> jnp.uint32(3))
            return v, v[0]

        carry, ys = lax.scan(body, x, jnp.arange(9))
        return jnp.concatenate([carry, ys])

    _run(f, np.arange(1024, dtype=np.uint32) * np.uint32(2654435761))


def test_jax_random_bits_matches_cpu():
    def f(seed):
        k = jax.random.key(seed)
        return jax.random.bits(k, (4096,), jnp.uint32), jax.random.bits(jax.random.split(k)[1], (3, 5), jnp.uint8)

    for seed in (0, 1, 12345):
        expected = jax.jit(f, backend="cpu")(seed)
        with jax.default_device(jax.devices("mps")[0]):
            actual = jax.jit(f)(seed)
        for a, e in zip(actual, expected, strict=True):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(e))
