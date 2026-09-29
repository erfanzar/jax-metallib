# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU/MPS reverse-mode comparisons for stable key/value sort."""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


@pytest.mark.parametrize("jit", [False, True])
def test_stable_sort_tied_keys_routes_nonuniform_value_cotangents(jit):
    weights = np.asarray([[1.0, 4.0, 2.0, 5.0, 3.0], [5.0, 2.0, 4.0, 1.0, 3.0]], dtype=np.float32)
    values = np.arange(10, dtype=np.float32).reshape(2, 5)

    def loss(x):
        _, sorted_values = lax.sort_key_val(jnp.zeros_like(x), x, dimension=1, is_stable=True)
        return jnp.sum(sorted_values * weights)

    gradient = jax.grad(loss)
    if jit:
        gradient = jax.jit(gradient)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_array_equal(expected, weights)
    np.testing.assert_array_equal(actual, expected)
