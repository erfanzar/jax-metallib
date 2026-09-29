# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""bool scatter must match CPU.

On MPS, bool scatter-set returned wrong values and scalar bool updates
(``x.at[i, j].set(True)``) failed with "scalar scatter is not supported yet".
"""

import jax
import numpy as np
import pytest
from jax import lax
from jax import numpy as jnp

_IDX = np.random.default_rng(0).permutation(6).astype(np.int32)


def _in_scan(a):
    def step(buf, t):
        return buf.at[t, t].set(True), None

    return lax.scan(step, a, jnp.arange(6, dtype=jnp.int32))[0]


_CASES = {
    "set_rows": lambda a: a.at[jnp.asarray(_IDX)].set(~a),
    "set_rows_dup": lambda a: a.at[jnp.asarray(_IDX) % 3].set(~a),
    "max_rows": lambda a: jnp.zeros_like(a).at[jnp.asarray(_IDX) % 3].max(a),
    "min_rows": lambda a: jnp.ones_like(a).at[jnp.asarray(_IDX) % 3].min(a),
    "scalar_set_true": lambda a: a.at[1, 2].set(True),
    "scalar_set_false": lambda a: a.at[4, 7].set(False),
    "scalar_max": lambda a: a.at[0, 0].max(True),
    "scalar_in_scan": _in_scan,
}


@pytest.mark.parametrize("case", list(_CASES))
@pytest.mark.parametrize("jit", [True, False], ids=["jit", "eager"])
def test_bool_scatter_matches_cpu(case, jit):
    if case == "set_rows_dup":
        pytest.skip("duplicate-index set order is implementation-defined")
    x = np.random.default_rng(1).random((6, 8)) > 0.5
    fn = jax.jit(_CASES[case]) if jit else _CASES[case]
    res = {}
    for p in ("cpu", "mps"):
        d = jax.devices(p)[0]
        with jax.default_device(d):
            res[p] = np.asarray(fn(jax.device_put(jnp.asarray(x), d)))
    assert res["mps"].dtype == res["cpu"].dtype == np.bool_
    np.testing.assert_array_equal(res["mps"], res["cpu"])
