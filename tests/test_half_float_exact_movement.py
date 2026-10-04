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
"""Bit-exact data movement for 16-bit floats, sort keys and composite argsort.

MPSGraph gather/gatherAlongAxis rewrote every f16/bf16 NaN to +qNaN (-NaN 0xfe00
and payload NaNs became 0x7e00 / 0x7fc0) and flushed bf16 subnormals, f16 reverse
did the same through its f32 cast round trip, and the f32 sort keys ordered
subnormals by value while XLA:CPU compares with denormals-as-zero (ties keep input
order). XLA:CPU sorts bf16 through f32, so every bf16 NaN out of a sort is
sign|0x7fc0 (a length-1 sort axis is a no-op and keeps raw bits). Many short
last-axis rows use one flat composite-key argsort; it must stay bit-exact too.
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

_U = {2: np.uint16, 4: np.uint32}
_FLOATS = {"f32": np.float32, "f16": np.float16, "bf16": jnp.bfloat16}


def _special(dt, shape, seed=0):
    """Small half-integers with -NaN, payload NaN, subnormals of both signs and -0."""
    rng = np.random.default_rng(seed)
    x = (rng.integers(-6, 6, size=shape).astype(np.float32) * 0.5).astype(dt)
    b = x.reshape(-1).view(_U[np.dtype(dt).itemsize]).copy()
    sign = b.dtype.type(1 << (8 * b.itemsize - 1))
    nan = np.array(np.nan, dt).view(b.dtype)
    b[::5] = nan | 1
    b[1::7] = nan | sign
    b[2::11] = 1
    b[3::13] = 3 | sign
    b[4::17] = sign
    return b.view(dt).reshape(shape)


def _bits(a):
    a = np.asarray(a)
    if a.dtype.kind == "f" or a.dtype == jnp.bfloat16:
        return a.view(_U[a.dtype.itemsize])
    return a


def _check(fn, *args):
    mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
    f = jax.jit(fn)
    got = jax.tree.leaves(f(*(jax.device_put(a, mps) for a in args)))
    want = jax.tree.leaves(f(*(jax.device_put(a, cpu) for a in args)))
    for k, (g, w) in enumerate(zip(got, want, strict=True)):
        gb, wb = _bits(g).reshape(-1), _bits(w).reshape(-1)
        bad = np.nonzero(gb != wb)[0][:4]
        assert not len(bad), (k, [hex(int(v)) for v in gb[bad]], [hex(int(v)) for v in wb[bad]])


_GATHERS = {
    "take_clip": lambda x, i: jnp.take(x, i, axis=0, mode="clip"),
    "index_rows": lambda x, i: x[i],
    "index_elems": lambda x, i: x[i, i % x.shape[1]],
    "take_along_clip": lambda x, i: jnp.take_along_axis(
        x, jnp.broadcast_to(i[: x.shape[0], None] % x.shape[1], x.shape), axis=1, mode="clip"
    ),
}


@pytest.mark.parametrize("dn", list(_FLOATS))
@pytest.mark.parametrize("name", list(_GATHERS))
def test_gather_moves_raw_bits(dn, name):
    x = _special(_FLOATS[dn], (64, 33))
    i = np.random.default_rng(1).integers(0, 64, size=(64,)).astype(np.int32)
    _check(_GATHERS[name], x, i)


@pytest.mark.parametrize("dn", list(_FLOATS))
@pytest.mark.parametrize("shape", [(64, 1), (3, 17), (64, 64), (5, 2, 9)])
def test_reverse_moves_raw_bits(dn, shape):
    x = _special(_FLOATS[dn], shape)
    _check(lambda v: (lax.rev(v, (v.ndim - 1,)), lax.rev(v, (0,)), jnp.flip(v)), x)


_SORTS = {
    "sort": lambda x: lax.sort(x, dimension=x.ndim - 1),
    "sort_desc": lambda x: jnp.sort(x, axis=-1, descending=True),
    "argsort": lambda x: jnp.argsort(x, axis=-1, stable=True),
    "argsort_desc": lambda x: jnp.argsort(x, axis=-1, stable=True, descending=True),
    "key_val": lambda x: lax.sort_key_val(x, lax.broadcasted_iota(jnp.int32, x.shape, x.ndim - 1)),
    "lexsort": lambda x: lax.sort(
        (x, jnp.flip(x, -1), jnp.arange(x.size, dtype=jnp.int32).reshape(x.shape)), dimension=x.ndim - 1, num_keys=2
    ),
    "argsort_take": lambda x: jnp.take_along_axis(x, jnp.argsort(x, axis=-1, stable=True), axis=-1, mode="clip"),
}


@pytest.mark.parametrize("dn", list(_FLOATS))
@pytest.mark.parametrize("shape", [(8, 1), (5,), (3, 17), (64, 64), (4, 513), (2, 3, 129)])
@pytest.mark.parametrize("name", list(_SORTS))
def test_sort_special_values_bitwise(dn, shape, name):
    _check(_SORTS[name], _special(_FLOATS[dn], shape, seed=sum(shape)))


# Shapes past the composite auto gate: 32-bit keys L <= 256 and rows >= 1024,
# 16-bit keys L <= 64 and rows >= 4096.
_COMPOSITE = {
    "f32": (np.float32, (1024, 16)),
    "f16": (np.float16, (4096, 8)),
    "bf16": (jnp.bfloat16, (4096, 33)),
    "i32": (np.int32, (2048, 256)),
    "u8": (np.uint8, (4096, 64)),
}


def _composite_data(dn):
    dt, shape = _COMPOSITE[dn]
    if dn in _FLOATS:
        return _special(dt, shape, seed=7)
    info = np.iinfo(dt)
    rng = np.random.default_rng(7)
    return rng.integers(info.min, info.max, size=shape, endpoint=True, dtype=np.int64).astype(dt)


@pytest.mark.parametrize("dn", list(_COMPOSITE))
@pytest.mark.parametrize("name", ["argsort", "argsort_desc", "key_val", "lexsort"])
def test_composite_argsort_bitwise(dn, name):
    _check(_SORTS[name], _composite_data(dn))


def _run(code, **env_overrides):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", **env_overrides)
    # close_fds=False lets CPython use posix_spawn on macOS: fork() from a process running Metal
    # threads intermittently died in the child (libmalloc _malloc_fork_child assertion, SIGTRAP).
    r = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        close_fds=False,
    )
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]


_SUBPROCESS_SORT = """
    import jax, jax.numpy as jnp, numpy as np
    from jax import lax
    mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
    rng = np.random.default_rng(0)
    for shape in ((2, 3), (37, 5), (300, 40), (1024, 16)):
        x = (rng.integers(-6, 6, size=shape) * 0.5).astype(np.float32)
        x.reshape(-1)[::9] = np.nan
        x.reshape(-1)[1::11] = -0.0
        for fn in (lambda v: jnp.argsort(v, axis=-1, stable=True),
                   lambda v: jnp.argsort(v, axis=-1, stable=True, descending=True),
                   lambda v: lax.sort_key_val(v, lax.broadcasted_iota(jnp.int32, v.shape, 1))):
            f = jax.jit(fn)
            for g, w in zip(jax.tree.leaves(f(jax.device_put(x, mps))), jax.tree.leaves(f(jax.device_put(x, cpu)))):
                g, w = np.asarray(g), np.asarray(w)
                if g.dtype == np.float32:
                    g, w = g.view(np.uint32), w.view(np.uint32)
                assert np.array_equal(g, w), (shape, g.reshape(-1)[:8], w.reshape(-1)[:8])
    print("CASE_OK")
"""


@pytest.mark.parametrize("max_row", ["0", "4096"])
def test_composite_gate_override_exact(max_row):
    _run(_SUBPROCESS_SORT, JAX_METALLIB_SORT_COMPOSITE_MAX_ROW=max_row)


def test_half_bit_sign_ops_optin_exact():
    code = """
        import jax, jax.numpy as jnp, numpy as np
        from jax import lax
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        for dt in (np.float16, jnp.bfloat16):
            b = (np.arange(64) % 9 - 4).astype(np.float32).astype(dt).view(np.uint16).copy()
            nan = int(np.array(np.nan, dt).view(np.uint16))
            b[::5] = nan | 1
            b[1::7] = nan | 0x8000
            b[3::13] = 0x0001
            b[4::17] = 0x8003
            x = b.view(dt)
            y = x[::-1].copy()
            m = np.arange(64) % 3 != 0
            for fn in (lambda x, y, m: -x, lambda x, y, m: jnp.abs(x), lambda x, y, m: lax.select(m, x, y),
                       lambda x, y, m: jnp.where(m, x, jnp.array(0, x.dtype)), lambda x, y, m: jnp.copysign(x, y)):
                f = jax.jit(fn)
                g = np.asarray(f(*(jax.device_put(a, mps) for a in (x, y, m)))).view(np.uint16)
                w = np.asarray(f(*(jax.device_put(a, cpu) for a in (x, y, m)))).view(np.uint16)
                bad = g != w
                assert np.array_equal(g, w), (dt, [hex(int(v)) for v in g[bad][:4]], [hex(int(v)) for v in w[bad][:4]])
        print("CASE_OK")
    """
    _run(code, JAX_METALLIB_HALF_BIT_SIGN_OPS="1")


def test_gather16_raw_bits_optout_runs():
    code = """
        import jax, jax.numpy as jnp, numpy as np
        mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]
        x = np.random.default_rng(0).standard_normal((100, 16)).astype(np.float16)
        i = np.arange(0, 100, 3, dtype=np.int32)
        f = jax.jit(lambda t, j: t[j])
        g = np.asarray(f(jax.device_put(x, mps), jax.device_put(i, mps)))
        assert np.array_equal(g.view(np.uint16), x[i].view(np.uint16))
        print("CASE_OK")
    """
    _run(code, JAX_METALLIB_DISABLE_GATHER16_RAW_BITS="1")
