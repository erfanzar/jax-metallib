# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""lax.approx_max_k / approx_min_k must run on MPS.

JAX lowers them to the ApproxTopK custom call, which had no MPS handler. On
non-TPU backends XLA treats it as an exact top-k (is_fallback=true), so the MPS
lowering is exact and must match XLA CPU. Inputs have distinct values so the
result is unambiguous.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp


def _distinct(shape, seed=0):
    n = int(np.prod(shape))
    vals = np.random.default_rng(seed).permutation(n).astype(np.float32) - n / 2
    return (vals * 0.37).reshape(shape)


_CASES = {
    "max_last": (lambda a: lax.approx_max_k(a, 4), (3, 10)),
    "min_last": (lambda a: lax.approx_min_k(a, 4), (3, 10)),
    "max_axis0": (lambda a: lax.approx_max_k(a, 2, reduction_dimension=0), (5, 7)),
    "min_axis0": (lambda a: lax.approx_min_k(a, 3, reduction_dimension=0), (5, 7)),
    "max_1d": (lambda a: lax.approx_max_k(a, 5), (33,)),
    "max_3d_mid": (lambda a: lax.approx_max_k(a, 2, reduction_dimension=1), (2, 6, 3)),
    "max_k_equals_n": (lambda a: lax.approx_max_k(a, 6), (2, 6)),
}


def _run(fn, x, platform, jit):
    func = jax.jit(fn) if jit else fn
    with jax.default_device(jax.devices(platform)[0]):
        return [np.asarray(r) for r in func(jnp.asarray(x))]


@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
@pytest.mark.parametrize("case", list(_CASES))
def test_approx_topk_matches_cpu(case, jit):
    fn, shape = _CASES[case]
    x = _distinct(shape)
    expected = _run(fn, x, "cpu", jit)
    actual = _run(fn, x, "mps", jit)
    for e, a in zip(expected, actual, strict=True):
        assert a.dtype == e.dtype and a.shape == e.shape
        np.testing.assert_array_equal(a, e)
