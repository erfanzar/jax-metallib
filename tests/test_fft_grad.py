# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Track CPU/MPS complex inverse-FFT gradient parity."""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


@pytest.mark.parametrize("jit", [False, True])
def test_irfftn_complex_gradient_matches_cpu(jit):
    def loss(x):
        return jnp.fft.irfftn(x, s=(4, 8), axes=(1, 2)).mean()

    gradient = jax.grad(loss)
    if jit:
        gradient = jax.jit(gradient)
    values = (np.random.default_rng(42).standard_normal((2, 4, 5)) + 1j).astype(np.complex64)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(gradient(jnp.asarray(values)))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(gradient(jnp.asarray(values)))
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.complex64])
def test_identity_pad_of_fft_output_matches_cpu(dtype):
    # A zero-width pad of a computed tensor previously returned all zeros on
    # MPS (JIT), which broke the irfftn gradient.
    def padded():
        scale = jnp.asarray(1.0, dtype=jnp.float32) / jnp.asarray(64.0, dtype=jnp.float32)
        spectrum = lax.fft(jnp.broadcast_to(scale, (2, 4, 8)), lax.FftType.RFFT, (4, 8))
        values = spectrum if dtype == jnp.complex64 else jnp.real(spectrum)
        return lax.pad(values, jnp.asarray(0, dtype=values.dtype), ((0, 0, 0),) * 3)

    func = jax.jit(padded)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(func())
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(func())
    assert np.abs(expected).max() > 0
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
