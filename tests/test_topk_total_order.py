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
"""lax.top_k on floats must match XLA:CPU bit for bit.

JAX lowers top_k to a chlo.top_k composite whose decomposition is a stable
stablehlo.sort with `compare GT, %a, %b, TOTALORDER` on the raw arguments:
IEEE totalOrder (-NaN < -inf < -0 < +0 < subnormals < +inf < +NaN, no DAZ).
The plugin used to apply the jnp.sort canonical key (-0 == +0, all NaN = +NaN,
subnormals == 0), which reordered ties, zeros and NaNs (75/247 probe cases).
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


def _has_backends():
    try:
        return bool(jax.devices("mps")) and bool(jax.devices("cpu"))
    except RuntimeError:
        return False


pytestmark = pytest.mark.skipif(not _has_backends(), reason="needs both the mps and cpu backends")

_DTYPES = {"f32": np.float32, "f16": np.float16, "bf16": jnp.bfloat16}
_SHAPES = [(64, 16), (8, 256), (2, 4097), (3, 17), (5, 1)]


def _make(dt, shape, kind, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape)
    if kind == "ties":
        x = np.round(x * 2) / 2
    x = x.astype(np.float32)
    if kind == "special":
        flat = x.reshape(-1)
        idx = rng.permutation(flat.size)
        q = max(1, flat.size // 16)
        specials = [np.nan, -np.float32(np.nan), 0.0, -0.0, np.inf, -np.inf, 1e-40, -1e-40]
        for j, v in enumerate(specials):
            flat[idx[j * q : (j + 1) * q]] = v
    out = x.astype(_DTYPES[dt])
    if kind == "special" and dt != "f32":
        # Native 16-bit subnormals and NaN payloads (the f32 values above flush or canonicalise).
        bits = out.reshape(-1).view(np.uint16)
        n = bits.size
        pick = rng.permutation(n)[: max(1, n // 32)]
        payload = [0x0001, 0x8001, 0x0003] + ([0x7E01, 0xFE01] if dt == "f16" else [0x7FC1, 0xFFC1])
        bits[pick] = np.resize(np.array(payload, np.uint16), pick.size)
    return out


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint32 if a.dtype.itemsize == 4 else np.uint16)


_CASES = [
    (dt, shape, kind, k)
    for dt in _DTYPES
    for shape in _SHAPES
    for kind in ("special", "ties")
    for k in sorted({1, min(8, shape[-1]), shape[-1]})
]


@pytest.mark.parametrize(("dt", "shape", "kind", "k"), _CASES)
def test_top_k_matches_cpu_bitwise(dt, shape, kind, k):
    mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
    seed = _CASES.index((dt, shape, kind, k))
    x = _make(dt, shape, kind, seed)
    f = jax.jit(lambda v: lax.top_k(v, k))
    gv, gi = (np.asarray(t) for t in f(jax.device_put(x, mps)))
    wv, wi = (np.asarray(t) for t in f(jax.device_put(x, cpu)))
    np.testing.assert_array_equal(gi, wi)
    np.testing.assert_array_equal(_bits(gv), _bits(wv))


def test_top_k_zero_order():
    # +0 ranks above -0 and ties keep input order (XLA's stable TOTALORDER comparator).
    x = np.array([[1.0, 2.0, 2.0, -0.0, 0.0, 3.0]], np.float32)
    v, i = lax.top_k(jax.device_put(x, jax.devices("mps")[0]), 6)
    assert np.asarray(i).tolist() == [[5, 1, 2, 0, 4, 3]]
    assert _bits(v).tolist() == [[0x40400000, 0x40000000, 0x40000000, 0x3F800000, 0x0, 0x80000000]]


def test_top_k_select_path_matches_cpu_bitwise():
    # JAX_METALLIB_TOPK_SELECT_MAX_K=16 sends every k < L <= 16 through the k-round
    # selection path (the auto gate only takes small k on large inputs). Covers
    # -NaN 0xFFFFFFFF, whose total-order key equals the INT32_MIN taken-mask value,
    # all-equal rows, and a 3-D operand.
    code = textwrap.dedent(
        """
        import sys
        sys.path.insert(0, sys.argv[1])
        import jax, numpy as np
        from jax import lax
        import test_topk_total_order as t
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        bad, n = [], 0
        shapes = [(64, 16), (8, 256), (3, 17), (2, 4097), (4, 3, 40)]
        for dt in t._DTYPES:
            for shape in shapes:
                for kind in ("special", "ties"):
                    x = t._make(dt, shape, kind, 1000 + n)
                    for k in (2, 3, 8, 12, 16):
                        if k >= shape[-1]:
                            continue
                        f = jax.jit(lambda v, k=k: lax.top_k(v, k))
                        gv, gi = (np.asarray(a) for a in f(jax.device_put(x, mps)))
                        wv, wi = (np.asarray(a) for a in f(jax.device_put(x, cpu)))
                        n += 1
                        if not (np.array_equal(gi, wi) and np.array_equal(t._bits(gv), t._bits(wv))):
                            bad.append((dt, shape, kind, k))
        hard = np.array([0xFFFFFFFF, 0x3F800000, 0xFFFFFFFF, 0x7FC00000, 0xFFFFFFFF, 0x80000000] * 3, np.uint32)
        rows = [hard.view(np.float32), np.full(18, np.float32(2.5)), np.full(18, -np.float32(0.0))]
        x = np.stack(rows + [hard.view(np.float32)[::-1].copy()])
        for k in (1, 2, 5, 8, 16):
            f = jax.jit(lambda v, k=k: lax.top_k(v, k))
            gv, gi = (np.asarray(a) for a in f(jax.device_put(x, mps)))
            wv, wi = (np.asarray(a) for a in f(jax.device_put(x, cpu)))
            n += 1
            if not (np.array_equal(gi, wi) and np.array_equal(t._bits(gv), t._bits(wv))):
                bad.append(("hard", k))
        assert not bad, (len(bad), n, bad[:10])
        print(f"CASE_OK {n}")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_TOPK_SELECT_MAX_K="16")
    r = subprocess.run(
        [sys.executable, "-c", code, os.path.dirname(os.path.abspath(__file__))],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        close_fds=False,
    )
    assert r.returncode == 0 and "CASE_OK" in r.stdout, (r.stdout + r.stderr)[-1500:]


@pytest.mark.parametrize("forced", ["16", "auto"])
def test_integer_top_k_select_path_matches_cpu(forced):
    # Integer lax.top_k lowers to iota -> stable sort with a raw GT (SIGNED/UNSIGNED)
    # compare -> slice[..., 0:k]; HandleSort takes the k-round selection path for it.
    # "16" forces every k < L onto that path; "auto" uses the default gate on shapes it
    # admits. Covers ties, full range, INT_MIN / uint 0 / uint max (mask-colliding keys)
    # and 8-bit dtypes (refused by the gate, so they must keep the exact sort).
    code = textwrap.dedent(
        """
        import sys
        import jax, numpy as np
        from jax import lax
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        forced = sys.argv[1] == "16"
        rng = np.random.default_rng(7)
        dtypes = ["int32", "uint32", "int16", "uint16", "int8", "uint8"]
        shapes = [(64, 16), (8, 256), (3, 17), (2, 4097), (4, 3, 40)] if forced else [(16384, 64), (4096, 256)]
        bad, n = [], 0
        for dt in dtypes:
            info = np.iinfo(dt)
            pool = np.array([info.min, info.min + 1, info.max, info.max - 1, 0, 1], np.int64)
            for shape in shapes:
                for kind in ("ties", "full", "extreme"):
                    if kind == "ties":
                        x = np.clip(rng.integers(-3, 4, size=shape), info.min, info.max).astype(dt)
                    elif kind == "full":
                        x = rng.integers(info.min, int(info.max) + 1, size=shape, dtype=np.int64).astype(dt)
                    else:
                        x = rng.choice(pool, size=shape).astype(dt)
                    for k in ((1, 2, 3, 8, 12, 16) if forced else (1, 4, 8)):
                        if k >= shape[-1]:
                            continue
                        f = jax.jit(lambda v, k=k: lax.top_k(v, k))
                        gv, gi = (np.asarray(a) for a in f(jax.device_put(x, mps)))
                        wv, wi = (np.asarray(a) for a in f(jax.device_put(x, cpu)))
                        n += 1
                        if gv.dtype != wv.dtype or not (np.array_equal(gv, wv) and np.array_equal(gi, wi)):
                            bad.append((dt, shape, kind, k))
        assert not bad, (len(bad), n, bad[:10])
        print(f"CASE_OK {n}")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    env.pop("JAX_METALLIB_TOPK_SELECT_INT", None)
    if forced == "16":
        env["JAX_METALLIB_TOPK_SELECT_MAX_K"] = "16"
    else:
        env.pop("JAX_METALLIB_TOPK_SELECT_MAX_K", None)
    r = subprocess.run(
        [sys.executable, "-c", code, forced],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        close_fds=False,
    )
    assert r.returncode == 0 and "CASE_OK" in r.stdout, (r.stdout + r.stderr)[-1500:]


def test_top_k_legacy_float_optout_runs():
    code = textwrap.dedent(
        """
        import jax, numpy as np
        from jax import lax
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        x = np.random.default_rng(1).permutation(4096).reshape(16, 256).astype(np.float32)
        f = jax.jit(lambda v: lax.top_k(v, 8))
        gv, gi = (np.asarray(t) for t in f(jax.device_put(x, mps)))
        wv, wi = (np.asarray(t) for t in f(jax.device_put(x, cpu)))
        assert np.array_equal(gi, wi) and np.array_equal(gv, wv)
        print("CASE_OK")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_METALLIB_TOPK_LEGACY_FLOAT="1")
    # close_fds=False lets CPython use posix_spawn on macOS: fork() from a process running Metal
    # threads intermittently died in the child (libmalloc _malloc_fork_child assertion, SIGTRAP).
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300, close_fds=False
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]
