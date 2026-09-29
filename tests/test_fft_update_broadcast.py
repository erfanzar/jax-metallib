# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex broadcast update coverage."""

import jax
import numpy as np
from jax import lax
from jax import numpy as jnp


def test_scalar_broadcast_complex_update_matches_cpu():
    def update():
        operand = jnp.broadcast_to(jnp.asarray(2 + 0j, dtype=jnp.complex64), (5,))
        value = jnp.broadcast_to(jnp.asarray(1 + 0j, dtype=jnp.complex64), (1,))
        operand = lax.dynamic_update_slice(operand, value, (0,))
        return lax.dynamic_update_slice(operand, value, (4,))

    func = jax.jit(update)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func())
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func())
    np.testing.assert_array_equal(actual, expected)


def test_weighted_rfftn_complex_updates_match_cpu():
    def weighted_fft():
        scaled = jnp.asarray(1.0, dtype=jnp.float32) / jnp.asarray(64.0, dtype=jnp.float32)
        spectrum = jnp.fft.rfftn(jnp.broadcast_to(scaled, (2, 4, 8)), s=(4, 8), axes=(1, 2))
        weights = jnp.broadcast_to(jnp.asarray(2 + 0j, dtype=jnp.complex64), (5,))
        one = jnp.broadcast_to(jnp.asarray(1 + 0j, dtype=jnp.complex64), (1,))
        weights = lax.dynamic_update_slice(weights, one, (0,))
        weights = lax.dynamic_update_slice(weights, one, (4,))
        weights = jnp.broadcast_to(weights, (1, 1, 5))
        scale = jnp.broadcast_to(jnp.asarray(1.0 / 32, dtype=jnp.complex64), (1, 1, 5))
        return jnp.conj(spectrum * (scale * weights))

    func = jax.jit(weighted_fft)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func())
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func())
    assert expected[0, 0, 0] != 0
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
