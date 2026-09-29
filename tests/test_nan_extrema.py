# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""NaN must propagate through max/min, argmax/argmin, clip, and float16 descending sort.

On MPS, float32/bfloat16 maximum/minimum, max/min reductions and clip dropped
NaN (returning a finite value), argmax/argmin ignored the NaN position, and
float16 descending sort returned wrong values for every element.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _data(nan):
    x = (np.random.default_rng(3).standard_normal((4, 32)) * 50).astype(np.float32)
    if nan:
        x[0, 2] = np.nan
        x[2, 17] = np.nan
        x[3, 0] = np.inf
        x[3, 1] = -np.inf
    return x


_OPS = {
    "maximum": lambda a: jnp.maximum(a, a[::-1]),
    "minimum": lambda a: jnp.minimum(a, a[::-1]),
    "max_red": lambda a: jnp.max(a, axis=-1),
    "min_red": lambda a: jnp.min(a, axis=0),
    "max_all": lambda a: jnp.max(a),
    "argmax": lambda a: jnp.argmax(a, axis=-1),
    "argmin": lambda a: jnp.argmin(a, axis=0),
    "clip": lambda a: jnp.clip(a, -10, 10),
    "clamp": lambda a: lax.clamp(a.dtype.type(-10), a, a.dtype.type(10)),
    "relu": lambda a: jax.nn.relu(a),
    "sort_desc": lambda a: jnp.sort(a, axis=-1, descending=True),
    "top_k": lambda a: lax.top_k(a, 5)[0],
}


@pytest.mark.parametrize("nan", [False, True], ids=["finite", "nan"])
@pytest.mark.parametrize("op", list(_OPS))
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16"])
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_nan_extrema_match_cpu(dtype, op, nan, jit):
    fn = jax.jit(_OPS[op]) if jit else _OPS[op]
    x = jnp.asarray(_data(nan)).astype(dtype)
    with jax.default_device(jax.devices("cpu")[0]):
        expected = np.asarray(fn(jax.device_put(x, jax.devices("cpu")[0]))).astype(np.float64)
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(fn(jax.device_put(x, jax.devices("mps")[0]))).astype(np.float64)
    np.testing.assert_array_equal(actual, expected)
