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
"""Narrowing convert (f16/bf16) of constant-derived values.

MPSGraph folded reinterpretCast(cast(constant)) into a reinterpret of the
constant's original f32 bytes, so the cast-fold barrier returned garbage:
jnp.clip(f16, -0.5, 0.5) returned all zeros (-0.5f is 0xBF000000) and
arange(f32) * c cast to bf16 returned reinterpreted bit patterns. A nested jit
keeps the constant as an HLO constant feeding the convert (JAX would fold a
concrete convert at trace time). XLA:CPU is the reference, compared bit-exactly.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest


def _devices():
    try:
        return jax.devices("cpu")[0], jax.devices("mps")[0]
    except RuntimeError:
        return None, None


CPU, MPS = _devices()
pytestmark = pytest.mark.skipif(CPU is None or MPS is None, reason="needs both cpu and mps backends")

DTYPES = [jnp.float16, jnp.bfloat16]
CONSTS = {
    "-0.5": np.float32(-0.5),
    "1.1": np.float32(1.1),
    "pi": np.float32(np.pi),
    "70000": np.float32(70000.0),
    "1e-8": np.float32(1e-8),
    "2/3": np.float32(2.0 / 3.0),
    "nan": np.float32(np.nan),
    "-inf": np.float32(-np.inf),
    "array": np.array([1.1, -0.5, 3.0, 70000.0, 1e-3], np.float32),
    "int32": np.int32(7),
    "int32_big": np.int32(2049),
}
SOURCES = {
    "const": lambda c: c,
    "bcast": lambda c: jnp.broadcast_to(c, (5,)),
    "neg": lambda c: -c,
    "mul2": lambda c: c * 2.0,
    "div3": lambda c: c / 3.0,
    "reshape": lambda c: jnp.reshape(c, (-1,)),
    "iota": lambda c: jnp.arange(5, dtype=jnp.float32) * c,
    "exp": jnp.exp,
}


def _check(fn, *args):
    got = np.asarray(jax.jit(fn)(*[jax.device_put(a, MPS) for a in args])).astype(np.float32)
    want = np.asarray(jax.jit(fn)(*[jax.device_put(a, CPU) for a in args])).astype(np.float32)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("cname", list(CONSTS))
@pytest.mark.parametrize("widen", [False, True], ids=["narrow", "widen"])
def test_constant_convert(dtype, cname, widen):
    c = CONSTS[cname]

    @jax.jit
    def to(c, v):
        return c.astype(v.dtype)

    def f(v):
        k = to(jnp.asarray(c), v)
        if widen:  # narrow then widen: must round (70000 -> inf in f16)
            return v.astype(jnp.float32) + k.astype(jnp.float32)
        return v + k

    _check(f, np.linspace(-2, 2, 5).astype(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("sname", list(SOURCES))
@pytest.mark.parametrize("cval", [1.1, -0.5, 70000.0])
def test_constant_derived_convert(dtype, sname, cval):
    src = SOURCES[sname]

    @jax.jit
    def conv(c, v):
        return src(c).astype(v.dtype)

    def f(v):
        k = conv(jnp.asarray(np.float32(cval)), v)
        return (v.astype(jnp.float32) + k.astype(jnp.float32)).astype(v.dtype)

    _check(f, np.linspace(-2, 2, 5).astype(dtype))


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize(
    "bounds", [(-0.5, 0.5), (-0.5, None), (None, 0.5), (1.1, 3.3)], ids=["both", "lo", "hi", "lo_hi_inexact"]
)
def test_clip_python_float_bounds(dtype, bounds):
    x = np.linspace(-4, 4, 33).astype(dtype)
    x[3] = np.nan
    _check(lambda v: jnp.clip(v, *bounds), x)


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
def test_rope_style_table(dtype):
    # f32 position table * inverse frequencies, cast to a half type inside a jitted helper.
    @jax.jit
    def table(v):
        pos = jnp.arange(64, dtype=jnp.float32)[:, None]
        inv = 1.0 / (10000.0 ** (jnp.arange(0, 16, 2, dtype=jnp.float32) / 16.0))
        return (pos * inv).astype(v.dtype)

    _check(lambda v: v + table(v), np.zeros((64, 8), dtype))


_BASE = np.array([1.1, -0.5, 70000.0, np.pi, 1e-3, -2.25, 3.3, 0.7], np.float32)
# Constant-derived ops outside the host fold: these take the integer
# round-to-nearest-even path (castTensor of a folded constant kept f32 bytes).
OTHER_SOURCES = {
    "slice": lambda c: c[1:6],
    "transpose": lambda c: jnp.reshape(c, (2, 4)).T.reshape(-1)[:5],
    "concat": lambda c: jnp.concatenate([c[:2], c[5:8]]),
    "select": lambda c: jnp.where(c[:5] > 0, c[:5], -c[:5]),
    "abs": lambda c: jnp.abs(c[:5]),
    "max": lambda c: jnp.maximum(c[:5], 0.25),
    "pad": lambda c: jnp.pad(c[:3], (1, 1)),
    "rev": lambda c: c[::-1][:5],
    "sqrt": lambda c: jnp.sqrt(jnp.abs(c[:5])),
    "dyn_slice": lambda c: jax.lax.dynamic_slice(c, (jnp.int32(2),), (5,)),
    "gather": lambda c: c[jnp.array([7, 0, 3, 3, 1])],
}


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
@pytest.mark.parametrize("sname", list(OTHER_SOURCES))
def test_constant_derived_other_ops(dtype, sname):
    src = OTHER_SOURCES[sname]

    @jax.jit
    def conv(c, v):
        return src(c).astype(v.dtype)

    _check(
        lambda v: v.astype(jnp.float32) + conv(jnp.asarray(_BASE), v).astype(jnp.float32),
        np.linspace(-2, 2, 5).astype(dtype),
    )


@pytest.mark.parametrize("dtype", DTYPES, ids=lambda d: jnp.dtype(d).name)
def test_large_closed_over_constant(dtype):
    # Above the host-fold element cap (2**22): f32 -> bf16 ties rounded wrongly (28 of 4.2M).
    n = (1 << 22) + 8
    w = (np.random.default_rng(n).standard_normal(n) * 3).astype(np.float32)
    _check(lambda v: v + jnp.asarray(w).astype(v.dtype), np.zeros(n, dtype))


def test_non_constant_convert_unchanged():
    # A runtime operand keeps the cast-fold barrier: f32 -> f16 -> f32 still rounds.
    x = np.array([70000.0, np.pi, 1e-8, -0.5, np.nan], np.float32)
    _check(lambda v: v.astype(jnp.float16).astype(jnp.float32), x)
    _check(lambda v: v.astype(jnp.bfloat16).astype(jnp.float32), x)
