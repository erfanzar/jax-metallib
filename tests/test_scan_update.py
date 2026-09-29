# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.scan carrying a buffer updated by dynamic_update_slice.

Carried buffers >= 64 KiB are promoted to MPSGraph variables inside counted
while loops; that path previously dropped every update and returned the
initial buffer.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _write_rows(x):
    t_len, dim = x.shape

    def step(buf, t):
        return lax.dynamic_update_slice(buf, x[t].reshape((1, dim)), (t, 0)), None

    buf, _ = lax.scan(step, jnp.zeros_like(x), jnp.arange(t_len, dtype=jnp.int32))
    return buf


def _read_modify_write(x):
    t_len, dim = x.shape

    def step(buf, t):
        row = lax.dynamic_slice(buf, (t, 0), (1, dim)).reshape((dim,))
        new_row = jnp.tanh(row + x[t])
        return lax.dynamic_update_slice(buf, new_row.reshape((1, dim)), (t, 0)), new_row.sum()

    buf, sums = lax.scan(step, jnp.zeros_like(x), jnp.arange(t_len, dtype=jnp.int32))
    return buf + sums.mean()


def test_scan_int32_row_update_preserves_large_values():
    # In-loop dynamic_update_slice used to round-trip int32 updates through
    # float32, corrupting values above 2**24.
    values = np.random.default_rng(0).integers(-1000, 1000, size=(64, 256), dtype=np.int32) + np.int32(1 << 25)

    def fn(x):
        def step(buf, t):
            return lax.dynamic_update_slice(buf, x[t].reshape((1, x.shape[1])), (t, 0)), None

        return lax.scan(step, jnp.zeros_like(x), jnp.arange(x.shape[0], dtype=jnp.int32))[0]

    func = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


def _scatter_set_rows(x):
    def step(buf, t):
        return buf.at[t].set(x[t]), None

    return lax.scan(step, jnp.zeros_like(x), jnp.arange(x.shape[0], dtype=jnp.int32))[0]


@pytest.mark.parametrize("case", ["int32_large", "inf_update", "nan_in_target"])
def test_scan_scatter_set_matches_cpu(case):
    # In-loop scatter-set was lowered as input*(1-mask) + update*mask in
    # float32: int32 values above 2**24 lost precision, a non-finite update
    # leaked NaN into every other row, and a non-finite target survived.
    rng = np.random.default_rng(0)
    if case == "int32_large":
        values = rng.integers(-1000, 1000, size=(16, 8), dtype=np.int32) + np.int32(1 << 25)
        fn = _scatter_set_rows
    elif case == "inf_update":
        values = rng.standard_normal((16, 8)).astype(np.float32)
        values[3, 2] = np.inf
        fn = _scatter_set_rows
    else:
        values = rng.standard_normal((16, 8)).astype(np.float32)

        def fn(x):
            def step(buf, t):
                return buf.at[t].set(x[t]), None

            init = jnp.full_like(x, jnp.nan)
            return lax.scan(step, init, jnp.arange(x.shape[0], dtype=jnp.int32))[0]

    func = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("case", ["int32_large", "inf_value"])
def test_scan_block_update_matches_cpu(case):
    # Multi-row in-loop updates used a float32 one-hot matmul expansion: int32
    # values above 2**24 lost precision and inf*0 leaked NaN into the block.
    if case == "int32_large":
        values = np.random.default_rng(1).integers(-1000, 1000, size=(64, 16), dtype=np.int32) + np.int32(1 << 25)
    else:
        values = np.random.default_rng(1).standard_normal((64, 16)).astype(np.float32)
        values[5, 3] = np.inf

    def fn(x):
        def step(buf, t):
            blk = lax.dynamic_slice(x, (t * 4, 0), (4, x.shape[1]))
            return lax.dynamic_update_slice(buf, blk, (t * 4, 0)), None

        return lax.scan(step, jnp.zeros_like(x), jnp.arange(x.shape[0] // 4, dtype=jnp.int32))[0]

    func = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("read", ["dynamic_slice_row", "dynamic_slice_block", "take_rows", "take_cols"])
def test_scan_int32_reads_preserve_large_values(read):
    # gatherAlongAxis measurably lost int32 precision above 2**24 inside loop
    # bodies; in-loop reads of int32 data must stay exact.
    values = np.random.default_rng(2).integers(-1000, 1000, size=(64, 16), dtype=np.int32) + np.int32(1 << 25)
    idx = np.random.default_rng(3).permutation(64).astype(np.int32)

    def fn(x, order):
        def step(acc, t):
            if read == "dynamic_slice_row":
                piece = lax.dynamic_slice(x, (t, 0), (1, x.shape[1]))[0]
            elif read == "dynamic_slice_block":
                piece = lax.dynamic_slice(x, (jnp.minimum(t, x.shape[0] - 4), 0), (4, x.shape[1]))[0]
            elif read == "take_rows":
                piece = jnp.take(x, order[t], axis=0)
            else:
                piece = jnp.take(x, order[t], axis=1)[:16]
            return acc, piece

        return lax.scan(step, jnp.int32(0), jnp.arange(x.shape[0], dtype=jnp.int32))[1]

    func = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values), jnp.asarray(idx % 16 if read == "take_cols" else idx)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values), jnp.asarray(idx % 16 if read == "take_cols" else idx)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("fn", [_write_rows, _read_modify_write], ids=["write", "read_modify_write"])
@pytest.mark.parametrize("rows", [32, 1024], ids=["below_threshold", "variable_carry"])
def test_scan_dynamic_update_slice_matches_cpu(fn, rows):
    values = np.random.default_rng(0).standard_normal((rows, 256)).astype(np.float32)
    func = jax.jit(fn)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func(jnp.asarray(values)))
    np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
