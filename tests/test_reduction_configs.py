# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Reduction registration and partial-window correctness regressions."""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

from .configs.reduction import make_reduction_op_configs


def test_standalone_reduction_configs_are_unique():
    names = [config.name for config in make_reduction_op_configs() if config.name.startswith("reduction.")]
    assert len(names) == len(set(names)), names


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_window_sum_does_not_include_values_outside_window(reverse):
    values = jnp.arange(1, 7, dtype=jnp.float32)
    padding = ((0, 2),) if reverse else ((2, 0),)
    reduce = jax.jit(lambda x: lax.reduce_window(x, 0.0, lax.add, (3,), (1,), padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(values))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(values))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("window", [3, 6])
def test_reduce_window_sum_respects_nonzero_init(window):
    padding = ((window - 1, 0),)
    reduce = jax.jit(lambda x: lax.reduce_window(x, 7.0, lax.add, (window,), (1,), padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "reducer,init,values",
    [
        (lax.mul, 1.0, [2.0, 3.0, 5.0, 7.0]),
        (lax.max, -np.inf, [3.0, 1.0, 2.0, 4.0]),
        (lax.min, np.inf, [1.0, 3.0, 2.0, 4.0]),
    ],
)
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_sliding_nonadd_window_drops_values_outside_window(reducer, init, values, reverse, dtype):
    padding = ((0, 1),) if reverse else ((1, 0),)
    reduce = jax.jit(lambda x: lax.reduce_window(x, jnp.asarray(init, dtype=dtype), reducer, (2,), (1,), padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "reducer,init,values",
    [
        (lax.mul, 2.0, [2.0, 0.0, 5.0, 7.0]),
        (lax.max, 7.0, [3.0, 1.0, 2.0, 4.0]),
        (lax.min, -7.0, [1.0, 3.0, 2.0, 4.0]),
    ],
)
def test_sliding_nonadd_window_respects_init_and_zero(reducer, init, values):
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, reducer, (2,), (1,), ((1, 0),)))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reducer,init,first", [(lax.max, -np.inf, 100.0), (lax.min, np.inf, -100.0)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("reverse", [False, True])
def test_wide_sliding_extrema_excludes_old_values(reducer, init, first, dtype, reverse):
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, reducer, (33,), (1,), ((0, 32) if reverse else (32, 0),)))
    values = np.arange(1, 65, dtype=np.float32)
    values[-1 if reverse else 0] = first
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reducer,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("reverse", [False, True])
def test_float16_sliding_extrema_gradient_matches_cpu(reducer, init, reverse):
    padding = ((0, 1),) if reverse else ((1, 0),)

    def loss(x):
        return jnp.sum(lax.reduce_window(x, jnp.asarray(init, x.dtype), reducer, (2,), (1,), padding))

    # JAX itself cannot JIT this max/min reduce_window reverse-mode gradient.
    gradient = jax.grad(loss)
    values = np.asarray([3.0, 1.0, 2.0, 4.0], dtype=np.float16)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reducer,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("tied", [False, True])
def test_wide_float16_sliding_extrema_gradient_matches_cpu(reducer, init, reverse, tied):
    padding = ((0, 32),) if reverse else ((32, 0),)

    def loss(x):
        return jnp.sum(lax.reduce_window(x, jnp.asarray(init, x.dtype), reducer, (33,), (1,), padding))

    values = np.arange(1, 65, dtype=np.float16)
    if tied:
        values[:] = 1
    gradient = jax.grad(loss)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reducer,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("axis", [0, 1])
def test_rank2_sliding_extrema_gradient_matches_cpu(reducer, init, axis):
    window = (2, 1) if axis == 0 else (1, 2)
    padding = ((1, 0), (0, 0)) if axis == 0 else ((0, 0), (0, 1))

    def loss(x):
        return jnp.sum(lax.reduce_window(x, jnp.asarray(init, x.dtype), reducer, window, (1, 1), padding))

    values = np.arange(1, 21, dtype=np.float16).reshape(4, 5)
    gradient = jax.grad(loss)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reducer,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("strides", [(1, 1), (2, 1)])
def test_rank2_spatial_extrema_gradient_matches_cpu(reducer, init, strides):
    window = (2, 3)
    padding = ((1, 0), (0, 2))

    def loss(x):
        return jnp.sum(lax.reduce_window(x, jnp.asarray(init, x.dtype), reducer, window, strides, padding))

    values = np.arange(1, 31, dtype=np.float16).reshape(5, 6)
    gradient = jax.grad(loss)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("reducer,init", [(lax.max, 7.0), (lax.min, -7.0)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_rank2_spatial_extrema_nonidentity_init_matches_cpu(reducer, init, dtype):
    reduce = jax.jit(
        lambda x: lax.reduce_window(x, jnp.asarray(init, x.dtype), reducer, (2, 3), (2, 1), ((1, 0), (0, 2)))
    )
    values = np.arange(1, 31, dtype=np.float32).reshape(5, 6)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_large_sliding_max_matches_cpu(dtype):
    reduce = jax.jit(lambda x: lax.reduce_window(x, jnp.asarray(-jnp.inf, dtype), lax.max, (33,), (1,), ((0, 32),)))
    values = np.random.default_rng(0).standard_normal(32768).astype(np.float32)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("reducer,init", [(lax.max, -np.inf), (lax.min, np.inf)])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
def test_sliding_extrema_rank2_matches_cpu(axis, reducer, init, reverse, dtype):
    window = (2, 1) if axis == 0 else (1, 2)
    padding = tuple(((0, 1) if reverse else (1, 0)) if dim == axis else (0, 0) for dim in range(2))
    reduce = jax.jit(lambda x: lax.reduce_window(x, jnp.asarray(init, dtype=dtype), reducer, window, (1, 1), padding))
    values = np.arange(1, 21, dtype=np.float32).reshape(4, 5)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rank,reducer,init,first", [(3, lax.max, -np.inf, 100.0), (4, lax.min, np.inf, -100.0)])
@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32])
@pytest.mark.parametrize("reverse", [False, True])
def test_wide_spatial_sliding_extrema_excludes_old_values(rank, reducer, init, first, dtype, reverse):
    window = (1, 33, 1) if rank == 3 else (1, 1, 33, 1)
    axis_padding = (0, 32) if reverse else (32, 0)
    padding = ((0, 0), axis_padding, (0, 0)) if rank == 3 else ((0, 0), (0, 0), axis_padding, (0, 0))
    shape = (1, 64, 1) if rank == 3 else (1, 1, 64, 1)
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, reducer, window, (1,) * rank, padding))
    values = np.arange(1, 65, dtype=np.float32).reshape(shape)
    values.flat[-1 if reverse else 0] = first
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.asarray(values, dtype=dtype)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reducer,init", [(lax.mul, 2.0), (lax.max, 7.0), (lax.min, -7.0)])
def test_cumulative_reduction_respects_nonidentity_init(reducer, init):
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, reducer, (6,), (1,), ((5, 0),)))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_nonidentity_partial_sum_avoids_prefix_cancellation(reverse):
    padding = ((0, 2),) if reverse else ((2, 0),)
    reduce = jax.jit(lambda x: lax.reduce_window(x, 7.0, lax.add, (3,), (1,), padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.full((1024,), 1e8, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.full((1024,), 1e8, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reducer,init", [(lax.add, 7.0), (lax.max, 7.0), (lax.min, -7.0)])
def test_pooling_reduction_respects_nonidentity_init(reducer, init):
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, reducer, (1, 3, 1), (1, 1, 1), ((0, 0), (1, 1), (0, 0))))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32).reshape(1, 6, 1)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.arange(1, 7, dtype=jnp.float32).reshape(1, 6, 1)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_window_sum_gradient_matches_cpu(reverse):
    padding = ((0, 2),) if reverse else ((2, 0),)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(lax.reduce_window(x, 0.0, lax.add, (3,), (1,), padding))))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.arange(1, 7, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.arange(1, 7, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("reverse", [False, True])
def test_partial_window_sum_avoids_prefix_cancellation(reverse):
    padding = ((0, 2),) if reverse else ((2, 0),)
    reduce = jax.jit(lambda x: lax.reduce_window(x, 0.0, lax.add, (3,), (1,), padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.full((1024,), 1e8, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.full((1024,), 1e8, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rank", [3, 4])
@pytest.mark.parametrize("init", [0.0, 7.0])
def test_partial_window_higher_rank_avoids_prefix_cancellation(rank, init):
    window = (1, 3, 1) if rank == 3 else (1, 1, 3, 1)
    padding = ((0, 0), (2, 0), (0, 0)) if rank == 3 else ((0, 0), (0, 0), (2, 0), (0, 0))
    shape = (1, 1024, 2) if rank == 3 else (1, 3, 1024, 2)
    reduce = jax.jit(lambda x: lax.reduce_window(x, init, lax.add, window, (1,) * rank, padding))
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(reduce(jnp.full(shape, 1e8, dtype=jnp.float32)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(reduce(jnp.full(shape, 1e8, dtype=jnp.float32)))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("axis", [0, 1])
def test_partial_window_valid_sum_rank2(axis):
    dimensions = (2, 1) if axis == 0 else (1, 2)

    def reduce(x):
        return lax.reduce_window(x, 0.0, lax.add, dimensions, (1, 1), "VALID")

    value = jax.jit(reduce)
    gradient = jax.jit(jax.grad(lambda x: jnp.sum(reduce(x))))
    with jax.default_device(jax.devices("cpu")[0]):
        values = jnp.arange(1, 13, dtype=jnp.float32).reshape(4, 3)
        expected_value = np.asarray(value(values))
        expected_gradient = np.asarray(gradient(values))
    with jax.default_device(jax.devices("mps")[0]):
        values = jnp.arange(1, 13, dtype=jnp.float32).reshape(4, 3)
        actual_value = np.asarray(value(values))
        actual_gradient = np.asarray(gradient(values))
    np.testing.assert_allclose(actual_value, expected_value, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(actual_gradient, expected_gradient, rtol=1e-5, atol=1e-5)
