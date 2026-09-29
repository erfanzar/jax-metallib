# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""float16 matmul must not overflow to inf for results representable in float16.

MPSGraph's float16 matmul returned +/-inf for products/sums between ~60000 and
the float16 max (65504), e.g. 60000 * 1.0 with K=1, where CPU is exact.
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_A = np.array([60000, -60000, 40000, 3.0, 2.0], dtype=np.float16)
_B = np.array([1.0, -1.0, 0.5, 2.0, 7.0, 1.5, 3.0], dtype=np.float16)

_CASES = {
    "outer_matmul": lambda x, y: x[:, None] @ y[None, :],
    "outer_dot_general": lambda x, y: lax.dot_general(x[:, None], y[None, :], (((1,), (0,)), ((), ()))),
    "k2": lambda x, y: jnp.stack([x, x], 1) @ jnp.stack([y, y], 0),
    "batched": lambda x, y: jnp.einsum("bik,bkj->bij", x[None, :, None], y[None, None, :]),
    "preferred_f32": lambda x, y: lax.dot(x[:, None], y[None, :], preferred_element_type=jnp.float32),
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_float16_matmul_near_max_matches_cpu(case, jit):
    fn = jax.jit(_CASES[case]) if jit else _CASES[case]
    res = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            res[p] = np.asarray(fn(jax.device_put(jnp.asarray(_A), d), jax.device_put(jnp.asarray(_B), d)))
    assert res["mps"].dtype == res["cpu"].dtype
    np.testing.assert_array_equal(res["mps"].astype(np.float64), res["cpu"].astype(np.float64))
