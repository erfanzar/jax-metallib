# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""stablehlo.compare with compare_type TOTALORDER matches CPU.

HandleCompare ignored compare_type, so TOTALORDER compares (jax lt_to/le_to/
eq_to, searchsorted, sort comparators) used IEEE semantics: lt_to(-0, 0),
lt_to(1, nan) and eq_to(nan, nan) were false, and searchsorted put NaN queries
at the end. Each case runs in a subprocess so a regression fails one case.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import sys
    import jax, jax.numpy as jnp, numpy as np
    from jax._src.lax import lax as li
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)
    sp = np.array([np.nan, -np.nan, 1.0, -0.0, 0.0, np.inf, -np.inf, -1.0, 0.0, -0.0], np.float32)
    a = np.concatenate([np.repeat(sp, len(sp)), rng.standard_normal(28).astype(np.float32)])
    b = np.concatenate([np.tile(sp, len(sp)), rng.standard_normal(28).astype(np.float32)])
    big_a = rng.standard_normal(1 << 20).astype(np.float32)
    big_b = rng.standard_normal(1 << 20).astype(np.float32)
    big_a[::97] = np.nan; big_b[::89] = -0.0; big_a[::53] = 0.0
    srt = np.sort(np.array([0.5, np.nan, -1.0, 2.0, np.nan, 0.0, -0.0, np.inf], np.float32))
    q = np.array([np.nan, 0.25, -5.0, 3.0, 2.0, -0.0, 0.0, np.inf, -np.inf], np.float32)
    kind, dt = sys.argv[1].split(":")
    dt = {"f32": np.float32, "f16": np.float16, "bf16": jnp.bfloat16}[dt]
    prims = {"lt": li.lt_to_p, "le": li.le_to_p, "eq": li.eq_to_p}
    if kind in prims:
        p = prims[kind]
        f, args = (lambda x, y: p.bind(x, y)), (a.astype(dt), b.astype(dt))
    elif kind.startswith("scalar_"):
        # Rank-0 operands: MPSGraph aborted compiling the total-order key of a scalar (jsl.expm hit it).
        p = prims[kind.split("_", 1)[1]]
        f = lambda x, y: jnp.stack([p.bind(x[i], y[i]) for i in range(100)])
        args = (a[:100].astype(dt), b[:100].astype(dt))
    elif kind == "searchsorted_scalar":
        f = lambda s, v: jnp.stack([jnp.searchsorted(s, v[i]) for i in range(len(q))])
        args = (srt.astype(dt), q.astype(dt))
    elif kind == "sort_lt":
        f, args = (lambda x, y: li._sort_lt_comparator(x, y)), (a.astype(dt), b.astype(dt))
    elif kind == "fused_chain":
        f = lambda x, y: jnp.where(li.lt_to_p.bind(x * 1.0, y), x * 2.0 + 1.0, y - 3.0)
        args = (big_a.astype(dt), big_b.astype(dt))
    elif kind.startswith("searchsorted_"):
        _, method, side = kind.split("_", 2)
        method = method.replace("-", "_")
        f = lambda s, v: jnp.searchsorted(s, v, side=side, method=method)
        args = (srt.astype(dt), q.astype(dt))
    want = np.asarray(jax.jit(f)(*[jax.device_put(x, cpu) for x in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(x, mps) for x in args]))
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, want.dtype)
    if got.dtype.kind == "f" or str(got.dtype) == "bfloat16":
        g32, w32 = got.astype(np.float32), want.astype(np.float32)
        np.testing.assert_array_equal(np.isnan(g32), np.isnan(w32))
        np.testing.assert_array_equal(np.nan_to_num(g32), np.nan_to_num(w32))
    else:
        np.testing.assert_array_equal(got, want)
    print("CASE_OK")
    """
)

_CASES = (
    [
        f"{k}:{d}"
        for k in ("lt", "le", "eq", "scalar_lt", "scalar_le", "scalar_eq", "sort_lt", "fused_chain")
        for d in ("f32", "f16", "bf16")
    ]
    + ["searchsorted_scalar:f32", "searchsorted_scalar:bf16"]
    + [
        f"searchsorted_{m}_{s}:{d}"
        for m in ("scan", "scan-unrolled", "compare-all", "sort")
        for s in ("left", "right")
        for d in ("f32", "bf16")
    ]
)


@pytest.mark.parametrize("name", _CASES)
def test_totalorder_compare_matches_cpu(name):
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD, name], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "CASE_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
