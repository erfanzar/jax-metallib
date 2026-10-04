# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""max/min reductions over slices containing NaN must return NaN.

StableHLO maximum/minimum propagate NaN, and NumPy agrees. XLA:CPU (jax 0.11)
returns the reduce init (-inf for max, +inf for min) for a NaN row of a
multi-row last-axis reduction from about [64,64] up, while tiny and single-row
inputs give NaN. These tests therefore compare MPS against NumPy, not XLA:CPU,
so the MPS lowering is never "fixed" to copy that upstream behaviour.
"""

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": jnp.bfloat16}
_FNS = {"max": (jnp.max, np.max), "min": (jnp.min, np.min)}


def _input(shape, axis, dtype, seed=0):
    rng = np.random.default_rng(seed)
    # Nonzero values only: -0/+0 ties are not what this file is about.
    x = (rng.standard_normal(shape) + 4.0).astype(np.float32)
    x = np.moveaxis(x, axis, -1)
    flat = x.reshape(-1, x.shape[-1])
    rows = flat.shape[0]
    flat[rows // 3, flat.shape[1] // 2] = np.nan  # one NaN mid-row
    flat[0, 0] = np.nan  # NaN first element
    flat[rows - 1, -1] = np.nan  # NaN last element
    if rows > 3:
        flat[rows // 2] = np.nan  # all-NaN row
        flat[rows // 2 + 1, :2] = [np.inf, np.nan]
    x = np.moveaxis(flat.reshape(x.shape), -1, axis)
    return np.ascontiguousarray(x).astype(_DTYPES[dtype])


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("op", list(_FNS))
@pytest.mark.parametrize(
    "shape,axis",
    [
        ((4, 8), -1),
        ((64, 64), -1),
        ((256, 1024), -1),
        ((16, 4096), -1),
        ((2048, 2048), -1),
        ((64, 300), 0),
        ((4, 33, 17), 1),
    ],
)
def test_reduce_max_min_propagates_nan(dtype, op, shape, axis):
    jfn, nfn = _FNS[op]
    x = _input(shape, axis, dtype)
    # max/min select an element, so reducing the widened copy and narrowing back is exact.
    expected = nfn(x.astype(np.float32), axis=axis).astype(_DTYPES[dtype])
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(lambda a: jfn(a, axis=axis))(jnp.asarray(x)))
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.astype(np.float32), expected.astype(np.float32))


def test_xla_cpu_multi_row_max_drops_nan_rows():
    """Pins the upstream XLA:CPU behaviour the NumPy reference above works around.

    If this starts failing, XLA:CPU now propagates NaN here and the NumPy
    reference in this file can go back to a plain CPU-vs-MPS comparison.
    """
    x = np.ones((64, 64), np.float32)
    x[21, 32] = np.nan
    xc = jax.device_put(x, jax.local_devices(backend="cpu")[0])
    got = np.asarray(jax.jit(lambda a: jnp.max(a, axis=-1))(xc))
    assert got[21] == -np.inf
