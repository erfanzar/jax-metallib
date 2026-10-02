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
"""Float cummax/cummin: short-scan fold, infinity saturation and identity windows.

Three defects on the default route, all against XLA:CPU and NumPy:
* MPSGraph cumulative max/min saturate infinities: a prefix of only -inf gave
  cummax -FLT_MAX (+inf gave cummin +FLT_MAX). A four-state int32 flag scan now
  restores -inf, +inf and NaN.
* A length-1 scan (all-ones window) went to the pooling route: max(-inf) gave
  0.0, min(+inf) gave -0.0, and bf16 min at rank 3 raised "min-pooling only
  supported". It is now reducer(x, init).
* Short scans (f32 L <= 4, f16/bf16 L <= 8) fold columns with NaN-propagating
  max/min instead (4M f32 L=2 9.8 -> 2.1 ms); the opt-out
  JAX_METALLIB_DISABLE_SHORT_FLOAT_SCAN_FOLD=1 must stay exact too.
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

_DTS = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}
_LAYOUTS = {
    "1d": lambda L: ((L,), 0),
    "last": lambda L: ((37, L), 1),
    "ax0": lambda L: ((L, 37), 0),
    "mid": lambda L: ((3, L, 5), 1),
}


def _data(dt, shape, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    flat = x.reshape(-1)
    k = flat.size
    idx = rng.permutation(k)
    flat[idx[: k // 9]] = np.nan
    flat[idx[k // 9 : k // 6]] = np.inf
    flat[idx[k // 6 : k // 5]] = -np.inf
    flat[idx[k // 5 : k // 4]] = 0.0
    flat[idx[k // 4 : k // 3]] = -0.0
    return jnp.asarray(x, dtype=dt)


def _ref(x, axis, is_max, rev, init=None):
    x = np.asarray(x, np.float32)
    acc = np.fmax if is_max else np.fmin
    src = np.flip(x, axis) if rev else x
    vals = acc.accumulate(src, axis=axis)
    nan = np.logical_or.accumulate(np.isnan(src), axis=axis)
    out = np.where(nan, np.nan, vals)
    if rev:
        out = np.flip(out, axis)
    if init is not None:
        out = np.where(np.isnan(out), np.nan, (np.maximum if is_max else np.minimum)(out, np.float32(init)))
    return out


def _assert_same(got, want, msg):
    got = np.asarray(got, np.float32)
    assert got.shape == want.shape, msg
    eq = (got == want) | (np.isnan(got) & np.isnan(want))
    assert eq.all(), (msg, int((~eq).sum()))


@pytest.mark.parametrize("dn", list(_DTS))
@pytest.mark.parametrize("L", [*range(1, 11), 64])
@pytest.mark.parametrize("layout", list(_LAYOUTS))
def test_float_cumminmax_exact(dn, L, layout):
    mps = jax.devices("mps")[0]
    shape, axis = _LAYOUTS[layout](L)
    x = _data(_DTS[dn], shape, L * 7 + len(layout))
    xd = jax.device_put(x, mps)
    for is_max in (True, False):
        fn = lax.cummax if is_max else lax.cummin
        for rev in (False, True):
            got = jax.jit(lambda v, f=fn, r=rev: f(v, axis=axis, reverse=r))(xd)
            _assert_same(got, _ref(x, axis, is_max, rev), (is_max, rev))
    init = jnp.asarray(0.25, _DTS[dn])
    for is_max in (True, False):
        op = lax.max if is_max else lax.min
        for pad in ((L - 1, 0), (0, L - 1)):
            wd = tuple(L if d == axis else 1 for d in range(len(shape)))
            pads = [pad if d == axis else (0, 0) for d in range(len(shape))]
            g = jax.jit(lambda v, i, o=op, w=wd, p=pads: lax.reduce_window(v, i, o, w, (1,) * len(w), p))
            got = g(xd, jax.device_put(init, mps))
            _assert_same(got, _ref(x, axis, is_max, pad[0] == 0, 0.25), ("init", is_max, pad))


@pytest.mark.parametrize("dn", list(_DTS))
@pytest.mark.parametrize("L", (2, 6, 300))
def test_infinite_prefix_not_saturated(dn, L):
    mps = jax.devices("mps")[0]
    dt = _DTS[dn]
    x = np.full((4, L), -np.inf, np.float32)
    x[1, L // 2 :] = 1.5
    x[2, -1] = np.inf
    x[3, L // 2] = np.nan
    for is_max, sign in ((True, 1.0), (False, -1.0)):
        v = jnp.asarray(sign * x, dt)
        fn = lax.cummax if is_max else lax.cummin
        got = jax.jit(lambda a, f=fn: f(a, axis=1))(jax.device_put(v, mps))
        _assert_same(got, _ref(v, 1, is_max, False), (dn, is_max))


def test_short_float_scan_fold_optout_exact():
    code = textwrap.dedent(
        """
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        mps = jax.devices("mps")[0]
        for dt in (jnp.float32, jnp.float16, jnp.bfloat16):
            for L in (1, 2, 4, 8, 64):
                x = np.random.default_rng(L).standard_normal((9, L)).astype(np.float32)
                x[::3, 0] = -np.inf
                x[1::4, -1] = np.nan
                v = jnp.asarray(x, dt)
                got = np.asarray(jax.jit(lambda a: lax.cummax(a, axis=1))(jax.device_put(v, mps)), np.float32)
                xs = np.asarray(v, np.float32)
                want = np.where(np.logical_or.accumulate(np.isnan(xs), axis=1), np.nan,
                                np.fmax.accumulate(xs, axis=1))
                assert np.array_equal(got, want, equal_nan=True), (dt, L)
        print("CASE_OK")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_DISABLE_SHORT_FLOAT_SCAN_FOLD="1")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]
