# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.approx_max_k / approx_min_k (ApproxTopK custom call) on MPS.

Off TPU, ApproxTopK is an exact top-k. The MPS lowering returns the stable
order of a canonical key: NaN is largest, -0 == +0, and ties keep the first
index. Small k takes the lax.top_k selection path (k max-reductions), larger k
a full argsort (JAX_METALLIB_APPROX_TOPK_SELECT=0 forces it). Both must give
the same bits.

The reference is NumPy, not XLA:CPU. The XLA:CPU fallback (jax 0.11) sorts with
a plain GT comparator: it returns wrong values for rows that contain NaN and
the last index on ties.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": jnp.bfloat16}
_SPECIAL = np.array([np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0, 2.5, 2.5], np.float32)


def _input(shape, dtype, kind, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    if kind == "special":
        m = rng.random(shape) < 0.5
        x[m] = rng.choice(_SPECIAL, size=int(m.sum()))
    elif kind == "ties":
        x = np.round(x * 2).astype(np.float32)
    elif kind == "allnan":
        x[...] = np.nan
    return x.astype(_DTYPES[dtype])


def _reference(x, k, axis, largest):
    key = np.moveaxis(x.astype(np.float64), axis, -1)
    key = np.where(np.isnan(key), np.inf, key)  # NaN sorts above +inf
    nan = np.isnan(np.moveaxis(x.astype(np.float64), axis, -1))
    # NaN ranks above every value; -0 == +0 because 0.0 == -0.0 in the key.
    rank = np.where(nan, 1.0, 0.0)
    order_key = np.stack([key, rank], axis=-1)
    flat = order_key.reshape(-1, key.shape[-1], 2)
    idx = np.empty(flat.shape[:2], np.int64)
    for r in range(flat.shape[0]):
        v, nr = flat[r, :, 0], flat[r, :, 1]
        if largest:
            idx[r] = np.lexsort((np.arange(v.size), -v, -nr))
        else:
            idx[r] = np.lexsort((np.arange(v.size), v, nr))
    idx = idx.reshape(key.shape)[..., :k]
    raw = np.moveaxis(x, axis, -1)
    vals = np.take_along_axis(raw, idx, axis=-1)
    return np.moveaxis(vals, -1, axis), np.moveaxis(idx, -1, axis).astype(np.int32)


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)


@pytest.mark.parametrize("dtype", list(_DTYPES))
@pytest.mark.parametrize("largest", [True, False], ids=["max", "min"])
@pytest.mark.parametrize("kind", ["normal", "special", "ties", "allnan"])
@pytest.mark.parametrize(
    "shape,axis,k",
    [
        ((64, 256), 1, 4),
        ((4096, 64), 1, 1),
        ((1024, 256), 1, 8),
        ((256, 1024), 1, 12),
        ((256, 512), 0, 4),
        ((3, 33, 40), 1, 3),
        ((16, 64), 1, 40),
    ],
)
def test_approx_topk_matches_stable_reference(dtype, largest, kind, shape, axis, k):
    x = _input(shape, dtype, kind, seed=len(shape) * 7 + k)
    fn = lax.approx_max_k if largest else lax.approx_min_k
    want_v, want_i = _reference(x, k, axis, largest)
    with jax.default_device(jax.devices("mps")[0]):
        got_v, got_i = jax.jit(lambda a: fn(a, k, reduction_dimension=axis, recall_target=1.0))(jnp.asarray(x))
    np.testing.assert_array_equal(np.asarray(got_i), want_i)
    np.testing.assert_array_equal(_bits(got_v), _bits(want_v))
