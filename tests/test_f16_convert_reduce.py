# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""A large f16 -> f32 convert feeding a last-axis sum must match XLA:CPU.

With the cast-fold barrier on the f16 side of convert, sum(convert_f32(f(x16)))
over the last axis returned zero rows on every call once the tensor reached
~16M elements (4096x4095, 16384x1024), while 3072x4096 was correct. f16
jax.nn.softmax at [4096,4096] (which JAX lowers to exactly this pattern) then
returned all-zero or NaN rows. Each case is called three times because the
first call once differed from later ones.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_CASES = {
    "exp_sum_4096x4095": (lambda a: jnp.sum(jnp.exp(a).astype(jnp.float32), axis=-1), (4096, 4095)),
    "mul2_sum_16384x1024": (lambda a: jnp.sum((a * 2).astype(jnp.float32), axis=-1), (16384, 1024)),
    "softmax_4096x4096": (lambda a: jax.nn.softmax(a, axis=-1), (4096, 4096)),
    "exp_sum_3072x4096_control": (lambda a: jnp.sum(jnp.exp(a).astype(jnp.float32), axis=-1), (3072, 4096)),
}


@pytest.mark.parametrize("case", list(_CASES))
def test_large_f16_convert_last_axis_sum_matches_cpu(case):
    fn, shape = _CASES[case]
    x = np.random.default_rng(42).standard_normal(shape).astype(np.float16)
    expected = np.asarray(jax.jit(fn, backend="cpu")(x)).astype(np.float32)
    jitted = jax.jit(fn)
    with jax.default_device(jax.devices("mps")[0]):
        xa = jnp.asarray(x)
        for _ in range(3):
            actual = np.asarray(jitted(xa)).astype(np.float32)
            assert not np.isnan(actual).any()
            tol = 5e-3 if expected.dtype == np.float16 or case.startswith("softmax") else 1e-3
            np.testing.assert_allclose(actual, expected, rtol=tol, atol=1e-2 if "sum" in case else 5e-4)
