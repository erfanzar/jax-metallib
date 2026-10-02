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
"""Integer cummax/cummin (full-prefix reduce_window) routing.

The integer max/min window lowering folded one full-size slice per window
position, so a prefix scan of length L cost L passes (4M-element i32 L=256:
113 ms). Scans of length >= 12 (32-bit) or >= 24 (8/16-bit) now take the direct
cumulative lowering (~0.3-1 ms). Both routes must stay exact, including a
non-identity init (max(init, prefix)) and reverse scans.
"""

import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import lax

_DTS = (np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32)
_LS = (2, 11, 12, 23, 24, 64, 256, 257)


def _ref(x, axis, is_max, rev, init=None):
    acc = np.maximum if is_max else np.minimum
    src = np.flip(x, axis) if rev else x
    out = acc.accumulate(src, axis=axis)
    if rev:
        out = np.flip(out, axis)
    if init is not None:
        out = acc(out, init)
    return out


def _data(dt, shape, seed):
    info = np.iinfo(dt)
    rng = np.random.default_rng(seed)
    x = rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=np.int64).astype(dt)
    x.flat[0] = info.min
    x.flat[-1] = info.max
    return x


@pytest.mark.parametrize("dt", _DTS, ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("L", _LS)
@pytest.mark.parametrize("layout", ["1d", "last", "ax0", "mid"])
def test_int_cumminmax_exact(dt, L, layout):
    mps = jax.devices("mps")[0]
    shape, axis = {"1d": ((L,), 0), "last": ((19, L), 1), "ax0": ((L, 19), 0), "mid": ((3, L, 5), 1)}[layout]
    x = _data(dt, shape, L * 31 + np.dtype(dt).itemsize)
    xd = jax.device_put(x, mps)
    for is_max in (True, False):
        fn = lax.cummax if is_max else lax.cummin
        for rev in (False, True):
            got = np.asarray(jax.jit(lambda v, f=fn, r=rev: f(v, axis=axis, reverse=r))(xd))
            want = _ref(x, axis, is_max, rev)
            assert got.dtype == want.dtype and np.array_equal(got, want), (is_max, rev)


@pytest.mark.parametrize("dt", (np.int32, np.uint16, np.int8), ids=lambda d: np.dtype(d).name)
@pytest.mark.parametrize("L", (8, 64))
def test_full_scan_reduce_window_nonidentity_init(dt, L):
    mps = jax.devices("mps")[0]
    x = _data(dt, (7, L), L + 3)
    init = np.array(np.iinfo(dt).max // 3, dt)
    xd = jax.device_put(x, mps)
    for is_max in (True, False):
        op = lax.max if is_max else lax.min
        for pad in ((L - 1, 0), (0, L - 1)):
            f = jax.jit(lambda v, i, o=op, p=pad: lax.reduce_window(v, i, o, (1, L), (1, 1), [(0, 0), p]))
            got = np.asarray(f(xd, jax.device_put(init, mps)))
            want = _ref(x, 1, is_max, pad[0] == 0, init)
            assert np.array_equal(got, want), (is_max, pad)


def test_int_cumminmax_optout_exact():
    code = textwrap.dedent(
        """
        import jax, numpy as np
        from jax import lax
        mps = jax.devices("mps")[0]
        for dt in (np.int32, np.int16, np.uint8):
            info = np.iinfo(dt)
            for L in (12, 24, 64):
                x = np.random.default_rng(L).integers(info.min, info.max, size=(9, L), endpoint=True,
                                                      dtype=np.int64).astype(dt)
                got = np.asarray(jax.jit(lambda v: lax.cummax(v, axis=1))(jax.device_put(x, mps)))
                assert np.array_equal(got, np.maximum.accumulate(x, axis=1)), (dt, L)
        print("CASE_OK")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_DISABLE_INT_SCAN_DIRECT="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]
