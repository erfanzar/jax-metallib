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
"""Float reduce max/min match XLA:CPU bitwise on -0/+0 ties, NaN and subnormals.

XLA:CPU orders -0 < +0 inside max/min reductions and flushes f32 subnormals to a
signed zero; MPSGraph's reductions returned whichever zero they met. The graph
handler now reduces an order-preserving integer key
(JAX_METALLIB_DISABLE_REDUCE_MAXMIN_KEY=1 restores the old path).

Two XLA:CPU artefacts are not ours and are handled explicitly:
- XLA:CPU drops a NaN in some larger reductions (f32 max of [-1]*4095 + [NaN]
  returns -inf). Where NumPy says NaN and XLA:CPU does not, MPS must say NaN.
- XLA:CPU flushes bf16 subnormals in reductions of <= 2048 elements and keeps
  them from 4096 elements, so bf16 subnormals only appear in the large case.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

UINT = {2: np.uint16, 4: np.uint32}
DTYPES = {"f32": jnp.float32, "f16": jnp.float16, "bf16": jnp.bfloat16}


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _bits(a):
    a = np.asarray(a)
    return a.reshape(-1).view(UINT[a.dtype.itemsize])


def _pool(dtname, kind, with_sub):
    base = [-0.0, 0.0, -1.0, -2.5, -np.inf] if kind == "max" else [-0.0, 0.0, 1.0, 2.5, np.inf]
    if with_sub:
        sub = {"f32": 1e-40, "f16": 2.0**-24, "bf16": 2.0**-130}[dtname]
        base += [sub, -sub]
    return np.array(base, np.float32)


def _make(dtname, kind, shape, with_sub, with_nan, seed):
    rng = np.random.default_rng(seed)
    x = rng.choice(_pool(dtname, kind, with_sub), size=shape).astype(np.float32)
    # Guarantee a pure -0/+0 slice along every reduced axis, so a tie decides at
    # least one result even when a kept subnormal wins every random slice.
    if x.ndim == 3:
        x[0::2, 0, :] = -0.0
        x[1::2, 0, :] = 0.0
        x[-1, :, 0::2] = -0.0
        x[-1, :, 1::2] = 0.0
    flat = x.reshape(-1, shape[-1])
    flat[0::2, 0] = -0.0
    flat[1::2, 0] = 0.0
    flat[0] = -0.0
    flat[1] = 0.0
    flat[2, ::2] = -0.0
    flat[2, 1::2] = 0.0
    if with_nan:
        flat[3, shape[-1] // 2] = np.nan
    return np.asarray(jnp.asarray(x, DTYPES[dtname]))


def _check(fn, x, kind, axis, need_tie=True):
    """fn(x) on CPU vs MPS; `kind`/`axis` describe the reduction for the NumPy reference."""
    cpu, mps = _devices()
    want = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
    got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
    assert want.shape == got.shape and want.dtype == got.dtype
    xf = x.astype(np.float32)
    npred = np.max if kind == "max" else np.min
    with np.errstate(invalid="ignore"):
        ref = np.asarray(npred(xf, axis=axis)).reshape(-1)
    wf = want.astype(np.float32).reshape(-1)
    gf = got.astype(np.float32).reshape(-1)
    # Upstream XLA:CPU NaN drop: MPS must propagate the NaN there.
    upstream = np.isnan(ref) & ~np.isnan(wf)
    assert np.all(np.isnan(gf[upstream])), "MPS dropped a NaN that NumPy propagates"
    both_nan = np.isnan(wf) & np.isnan(gf)
    bad = (_bits(want) != _bits(got)) & ~both_nan & ~upstream
    assert not bad.any(), f"{int(bad.sum())} bitwise diffs, first cpu={wf[bad][0]!r} mps={gf[bad][0]!r}"
    if need_tie:
        # Non-vacuous: some slice holds both -0 and +0 and its result is a zero,
        # so the tie order decides the expected bits.
        with np.errstate(invalid="ignore"):
            has_neg = np.any((xf == 0) & np.signbit(xf), axis=axis)
            has_pos = np.any((xf == 0) & ~np.signbit(xf), axis=axis)
        tie = (np.asarray(has_neg) & np.asarray(has_pos)).reshape(-1) & (wf == 0)
        assert tie.any(), "test is vacuous: no slice where a -0/+0 tie decides the result"


SHAPES = [(64, 7), (64, 33), (64, 300), (16, 1000)]


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("axis", [0, 1])
def test_reduce_axis_tie_nan(dtname, kind, shape, axis):
    red = jnp.max if kind == "max" else jnp.min
    seed = sum(map(ord, f"{dtname}{kind}{shape}"))
    x = _make(dtname, kind, shape, with_sub=dtname != "bf16", with_nan=True, seed=seed)
    _check(lambda v: red(v, axis=axis), x, kind, axis)


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduce_large_subnormal(dtname, kind):
    # bf16 subnormals are stable vs XLA:CPU only from 4096 elements.
    red = jnp.max if kind == "max" else jnp.min
    x = _make(dtname, kind, (8, 4096), with_sub=True, with_nan=False, seed=7)
    _check(lambda v: red(v, axis=0), x, kind, 0)
    _check(lambda v: red(v, axis=1), x, kind, 1)


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduce_multi_axis_3d(dtname, kind):
    red = jnp.max if kind == "max" else jnp.min
    x = _make(dtname, kind, (6, 5, 33), with_sub=dtname != "bf16", with_nan=True, seed=11)
    _check(lambda v: red(v, axis=(0, 2)), x, kind, (0, 2))
    _check(lambda v: red(v, axis=(1, 2)), x, kind, (1, 2))


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduce_inside_scan(dtname, kind):
    red = jnp.max if kind == "max" else jnp.min
    x = _make(dtname, kind, (4, 64, 33), with_sub=dtname != "bf16", with_nan=True, seed=13)

    def body(c, xs):
        return c, red(xs, axis=1)

    _check(lambda v: lax.scan(body, 0, v)[1], x, kind, 2)


@pytest.mark.parametrize("dtname", list(DTYPES))
def test_reduce_full_small(dtname):
    # Full reduces of a single all-zero slice: the sign of the tie is the answer.
    for vals, kind in (([-0.0, 0.0, -0.0], "max"), ([0.0, -0.0, 0.0], "min")):
        red = jnp.max if kind == "max" else jnp.min
        x = np.asarray(jnp.asarray(np.array(vals, np.float32), DTYPES[dtname]))
        _check(lambda v, red=red: red(v), x, kind, None)


# Elementwise producers fused into the reduce. abs/exp (never negative) and
# add/subtract of a large-enough splat constant (no -0, no subnormal) skip the
# integer key; multiply and small constants keep it. Inputs include -1, -1/3
# neighbours (x*3+1 near-cancellation), -100 (exp underflows to an f32
# subnormal, which XLA:CPU flushes in the reduce) and forced -0/+0 slices.
PRODUCERS = {
    "abs": jnp.abs,
    "exp": jnp.exp,
    "add1": lambda v: v + 1.0,
    "affine": lambda v: v * 3.0 + 1.0,
    "sub_half": lambda v: v - 0.5,
    "neg_sub": lambda v: 2.0 - v,
    "mul2": lambda v: v * 2.0,
}


def _make_producer_input(dtname, shape, seed):
    third = np.float32(-1.0 / 3.0)
    pool = [-0.0, 0.0, -1.0, 1.0, 2.0, -2.5, -100.0, 0.5, -np.inf, np.inf]
    pool += [third, np.nextafter(third, np.float32(0)), np.nextafter(third, np.float32(-1))]
    pool += [np.nextafter(np.float32(-1), np.float32(0)), np.nextafter(np.float32(-1), np.float32(-2))]
    if dtname != "bf16":
        sub = {"f32": 1e-40, "f16": 2.0**-24}[dtname]
        pool += [sub, -sub]
    rng = np.random.default_rng(seed)
    x = rng.choice(np.array(pool, np.float32), size=shape)
    x[0::2, 0], x[1::2, 0] = -0.0, 0.0
    x[0], x[1] = -0.0, 0.0
    x[2, ::2], x[2, 1::2] = -0.0, 0.0
    x[4, :] = -100.0
    x[4, 0] = 1.0
    x[5, :] = -1.0
    x[5, 1::3] = 1.0
    x[3, shape[-1] // 2] = np.nan
    return np.asarray(jnp.asarray(x, DTYPES[dtname]))


def _xla_reduce_ref(y, kind, axis):
    """XLA:CPU max/min semantics in NumPy: -0 < +0, NaN propagates, f32 subnormals flush to a signed zero."""
    yf = y.astype(np.float64)
    if y.dtype == np.float32:
        yf = np.where(np.abs(yf) < np.finfo(np.float32).tiny, np.copysign(0.0, yf), yf)
    dmin = np.finfo(np.float64).smallest_subnormal
    yf = np.where(yf == 0, np.copysign(dmin, yf), yf)
    with np.errstate(invalid="ignore"):
        r = np.asarray((np.max if kind == "max" else np.min)(yf, axis=axis))
    r = np.where(np.abs(r) < 1e-300, np.copysign(0.0, r), r)
    return np.asarray(jnp.asarray(r.astype(np.float32), y.dtype))


@pytest.mark.parametrize("dtname", list(DTYPES))
@pytest.mark.parametrize("kind", ["max", "min"])
@pytest.mark.parametrize("prod", list(PRODUCERS))
@pytest.mark.parametrize("shape", [(64, 33), (16, 1000)])
def test_reduce_fused_producer(dtname, kind, prod, shape):
    cpu, mps = _devices()
    red = jnp.max if kind == "max" else jnp.min
    pf = PRODUCERS[prod]
    x = _make_producer_input(dtname, shape, seed=sum(map(ord, f"{dtname}{kind}{prod}{shape}")))
    if prod == "exp" and dtname == "f32":
        # MPS f32 exp is not bitwise with XLA:CPU (exp(1): 2.7182817 vs 2.718282), a separate
        # elementwise gap; check the reduce against XLA semantics applied to MPS's own exp output.
        y = np.asarray(jax.jit(pf)(jax.device_put(x, mps)))
        for axis in (0, 1):
            got = np.asarray(jax.jit(lambda v, axis=axis: red(pf(v), axis=axis))(jax.device_put(x, mps)))
            ref = _xla_reduce_ref(y, kind, axis)
            gf, rf = got.astype(np.float32).reshape(-1), ref.astype(np.float32).reshape(-1)
            bad = (_bits(got) != _bits(ref)) & ~(np.isnan(gf) & np.isnan(rf))
            assert not bad.any(), f"axis={axis}: {int(bad.sum())} diffs, first ref={rf[bad][0]!r} mps={gf[bad][0]!r}"
        return
    for axis in (0, 1):
        fn = jax.jit(lambda v, axis=axis: red(pf(v), axis=axis))
        want = np.asarray(fn(jax.device_put(x, cpu)))
        got = np.asarray(fn(jax.device_put(x, mps)))
        assert want.shape == got.shape and want.dtype == got.dtype
        y = np.asarray(jax.jit(pf)(jax.device_put(x, cpu))).astype(np.float32)
        npred = np.max if kind == "max" else np.min
        with np.errstate(invalid="ignore"):
            ref = np.asarray(npred(y, axis=axis)).reshape(-1)
        wf = want.astype(np.float32).reshape(-1)
        gf = got.astype(np.float32).reshape(-1)
        upstream = np.isnan(ref) & ~np.isnan(wf)
        assert np.all(np.isnan(gf[upstream])), "MPS dropped a NaN that NumPy propagates"
        bad = (_bits(want) != _bits(got)) & ~(np.isnan(wf) & np.isnan(gf)) & ~upstream
        assert not bad.any(), f"axis={axis}: {int(bad.sum())} bitwise diffs, first cpu={wf[bad][0]!r} mps={gf[bad][0]!r}"


@pytest.mark.parametrize("kind", ["max", "min"])
def test_reduce_exp_f32_subnormal_flush(kind):
    # exp(-100) is an f32 subnormal; XLA:CPU flushes the reduced result to +0.
    red = jnp.max if kind == "max" else jnp.min
    x = np.full((8, 64), -100.0, np.float32)
    x[:, 1] = -np.inf
    if kind == "min":
        x[:, 2] = 3.0
    cpu, mps = _devices()
    fn = jax.jit(lambda v: red(jnp.exp(v), axis=1))
    want = np.asarray(fn(jax.device_put(x, cpu)))
    got = np.asarray(fn(jax.device_put(x, mps)))
    assert np.array_equal(_bits(want), _bits(got)), f"cpu={want[:2]!r} mps={got[:2]!r}"


@pytest.mark.parametrize("dtname", list(DTYPES))
def test_reduce_nan_payload_canonical(dtname):
    # A NaN result is a NaN on both backends (payload not compared); row 1 is a tie.
    x = np.asarray(jnp.asarray(np.array([[1.0, np.nan, -0.0], [-0.0, 0.0, -0.0]], np.float32), DTYPES[dtname]))
    _check(lambda v: jnp.max(v, axis=1), x, "max", 1)
    _check(lambda v: jnp.min(v, axis=1), x, "min", 1)
