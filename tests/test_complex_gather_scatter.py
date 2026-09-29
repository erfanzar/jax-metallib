# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""complex64 gather/scatter/sort-payload must match CPU instead of aborting.

MPSGraph gather, scatter and gather_along_axis reject complex operands with a
process-aborting assertion (exit 134), so each case runs in a child process.
"""

import os
import subprocess
import sys

import pytest

SCRIPT = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

case, mode = sys.argv[1], sys.argv[2]
rng = np.random.default_rng(0)
x = (rng.standard_normal((6, 8)) + 1j * rng.standard_normal((6, 8))).astype(np.complex64)
idx = rng.permutation(6).astype(np.int32)
keys = rng.standard_normal(8).astype(np.float32)

CASES = {
    "take_rows": lambda a: jnp.take(a, jnp.asarray(idx), axis=0),
    "take_cols": lambda a: jnp.take(a, jnp.asarray(idx) % 8, axis=1),
    "scatter_set": lambda a: a.at[jnp.asarray(idx)].set(a * 2),
    "scatter_add": lambda a: jnp.zeros_like(a).at[jnp.asarray(idx) % 3].add(a),
    "scalar_set": lambda a: a.at[1, 2].set(3 + 4j),
    "scalar_add": lambda a: a.at[4, 7].add(1 - 2j),
    "sort_key_val": lambda a: lax.sort_key_val(jnp.broadcast_to(jnp.asarray(keys), (6, 8)), a, dimension=1)[1],
}
fn = CASES[case]
if mode == "jit":
    fn = jax.jit(fn)
res = {}
for p in ("cpu", "mps"):
    d = jax.devices(p)[0]
    with jax.default_device(d):
        res[p] = np.asarray(fn(jax.device_put(jnp.asarray(x), d)))
assert res["cpu"].dtype == res["mps"].dtype, (res["cpu"].dtype, res["mps"].dtype)
np.testing.assert_allclose(res["mps"], res["cpu"], atol=1e-5, rtol=1e-5)
"""


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize(
    "case", ["take_rows", "take_cols", "scatter_set", "scatter_add", "scalar_set", "scalar_add", "sort_key_val"]
)
def test_complex_gather_scatter_matches_cpu(case, mode):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, case, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
