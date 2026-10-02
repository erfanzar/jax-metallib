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
"""Non-last-axis sorts: the MPS lowering moves the sort dimension to the end.

Only last-axis sorts reach the native bitonic kernel (f32 lax.sort(dimension=0)
[64,65536] 87.9 -> 2.4 ms). Results must stay bit-exact against XLA:CPU for every
sort form (values, descending, stable argsort, key/value, multi-key), including
NaN, -0 and infinities, and with the rule disabled.
"""

import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

_DTS = {
    "f32": np.float32,
    "f16": np.float16,
    "bf16": jnp.bfloat16,
    "i32": np.int32,
    "u8": np.uint8,
    "i16": np.int16,
    "bool": np.bool_,
}
_CASES = (((33, 7), 0), ((300, 5), 0), ((6, 37, 4), 1), ((5, 6, 300), 0), ((257, 3), 0), ((1, 9), 0))
_FNS = {
    "sort": lambda x, a: (lax.sort(x, dimension=a),),
    "sort_desc": lambda x, a: (jnp.sort(x, axis=a, descending=True),),
    "argsort": lambda x, a: (jnp.argsort(x, axis=a, stable=True),),
    "key_val": lambda x, a: lax.sort_key_val(x, jnp.cumsum(jnp.ones(x.shape, jnp.int32), axis=a), dimension=a),
    "lexsort": lambda x, a: lax.sort(
        (x, jnp.flip(x, a), jnp.arange(x.size, dtype=jnp.int32).reshape(x.shape)), dimension=a, num_keys=2
    ),
}


def _data(dt, shape, seed):
    rng = np.random.default_rng(seed)
    if dt == np.bool_:
        return jnp.asarray(rng.integers(0, 2, size=shape).astype(bool))
    if dt in (np.float32, np.float16, jnp.bfloat16):
        x = rng.integers(-6, 6, size=shape).astype(np.float32) * 0.5
        flat = x.reshape(-1)
        flat[::13] = np.nan
        flat[1::17] = -0.0
        flat[2::19] = np.inf
        flat[3::23] = -np.inf
        return jnp.asarray(x, dt)
    info = np.iinfo(dt)
    return jnp.asarray(rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=np.int64).astype(dt))


def _bits(a):
    a = np.asarray(a)
    if a.dtype.kind == "f" or a.dtype == jnp.bfloat16:
        return a.view({2: np.uint16, 4: np.uint32}[a.dtype.itemsize])
    return a


@pytest.mark.parametrize("dn", list(_DTS))
@pytest.mark.parametrize("case", range(len(_CASES)))
def test_sort_non_last_axis_bitwise(dn, case):
    mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
    shape, axis = _CASES[case]
    x = _data(_DTS[dn], shape, sum(shape))
    for name, fn in _FNS.items():
        if name == "sort_desc" and dn == "bool":
            continue
        f = jax.jit(lambda v, fn=fn, a=axis: fn(v, a))
        got = f(jax.device_put(x, mps))
        want = f(jax.device_put(x, cpu))
        for k, (g, w) in enumerate(zip(got, want, strict=True)):
            assert np.array_equal(_bits(g), _bits(w)), (name, k)


def test_sort_non_last_axis_lowers_to_last_axis():
    x = jax.device_put(jnp.ones((4, 8)), jax.devices("mps")[0])
    text = jax.jit(lambda v: lax.sort(v, dimension=0)).lower(x).as_text()
    assert "dimension = 1" in text and "transpose" in text


def test_sort_move_axis_optout_exact():
    code = textwrap.dedent(
        """
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        x = np.random.default_rng(0).standard_normal((300, 5)).astype(np.float32)
        x[::7, 1] = np.nan
        f = jax.jit(lambda v: lax.sort(v, dimension=0))
        assert "transpose" not in f.lower(jax.device_put(x, mps)).as_text()
        got = np.asarray(f(jax.device_put(x, mps))).view(np.uint32)
        want = np.asarray(f(jax.device_put(x, cpu))).view(np.uint32)
        assert np.array_equal(got, want)
        print("CASE_OK")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_DISABLE_SORT_MOVE_AXIS="1")
    # close_fds=False lets CPython use posix_spawn on macOS: fork() from a process running Metal
    # threads intermittently died in the child (libmalloc _malloc_fork_child assertion, SIGTRAP).
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300, close_fds=False
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]
