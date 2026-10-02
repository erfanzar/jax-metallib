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
"""Float dots with a one-element contraction (K=1, outer products) match XLA:CPU bitwise.

XLA:CPU folds a K=1 dot into a multiply, so 0 * -3 gives -0. The MPSGraph matmul
accumulated from +0 and returned +0 for every such element (f32, f16, bf16, batched,
outer product, matrix-vector). Dots with K >= 2 start from +0 on both backends and are
kept here as controls, as are integer K=1 dots.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

_UINT = {1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _arr(rng, shape, dtype):
    if jnp.issubdtype(dtype, jnp.integer):
        return rng.integers(-5, 6, shape).astype(dtype)
    # Signed zeros, small integers and random normals, so -0 products and rounding both occur.
    pool = np.array([0.0, -0.0, 1.0, -3.0, 0.5, -7.25], np.float32)
    vals = np.where(rng.random(shape) < 0.5, rng.choice(pool, shape), rng.standard_normal(shape) * 4.0)
    return vals.astype(np.float32).astype(dtype)


def _dg(lc, rc, lb=(), rb=(), acc=None):
    return lambda a, b: lax.dot_general(a, b, ((lc, rc), (lb, rb)), preferred_element_type=acc)


_CASES = {
    "matmul_3x1_1x4": (((3, 1), (1, 4)), lambda a, b: a @ b),
    "matmul_wide": (((257, 1), (1, 300)), lambda a, b: a @ b),
    "outer_einsum": (((5,), (6,)), lambda a, b: jnp.einsum("i,j->ij", a, b)),
    # jnp.outer lowers to a broadcast multiply, not dot_general: a control that passes with or
    # without the K=1 path (JAX_METALLIB_DISABLE_DOT_K1_MULTIPLY=1).
    "outer_np": (((5,), (6,)), jnp.outer),
    "dot_vv": (((1,), (1,)), jnp.dot),
    "dot_mv": (((4, 1), (1,)), jnp.dot),
    "dot_vm": (((1,), (1, 4)), jnp.dot),
    "batched": (((2, 3, 1), (2, 1, 4)), lambda a, b: a @ b),
    "lhs_contract_first": (((1, 3), (1, 4)), _dg((0,), (0,))),
    "rhs_contract_last": (((3, 1), (4, 1)), _dg((1,), (1,))),
    "batch_middle": (((3, 2, 1), (2, 1, 4)), _dg((2,), (1,), (1,), (0,))),
    "two_unit_contractions": (((3, 1, 1), (1, 1, 4)), _dg((1, 2), (0, 1))),
    "rank3_free": (((2, 1, 3), (1, 4, 5)), _dg((1,), (0,))),
}

_CONTROLS = {
    "k2": (((3, 2), (2, 4)), lambda a, b: a @ b),
    "k3_batched": (((2, 3, 3), (2, 3, 4)), lambda a, b: a @ b),
    "k64": (((8, 64), (64, 8)), lambda a, b: a @ b),
}


def _exact_arr(rng, shape, dtype):
    # Small integers and signed zeros: every K >= 2 sum is exact in any order, so a bitwise
    # check tests the sign of zero rather than the matmul's accumulation order (random f32
    # normals differ from XLA:CPU by 1 ulp at K >= 2 on the unchanged matmul path).
    pool = np.array([0.0, -0.0, 1.0, -1.0, 2.0, -3.0], np.float32)
    return rng.choice(pool, shape).astype(dtype)


def _compare(fn, shapes, dtype, seed=0, make=_arr):
    cpu, mps = _devices()
    rng = np.random.default_rng(seed)
    args = [make(rng, s, dtype) for s in shapes]
    jf = jax.jit(fn)
    want = np.asarray(jf(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jf(*[jax.device_put(a, mps) for a in args]))
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, got.dtype, want.shape, want.dtype)
    u = _UINT[want.dtype.itemsize]
    wb, gb = want.reshape(-1).view(u), got.reshape(-1).view(u)
    bad = np.nonzero(wb != gb)[0]
    assert bad.size == 0, f"{bad.size} bit mismatches, first at {bad[0]}: cpu={wb[bad[0]]:#x} mps={gb[bad[0]]:#x}"
    return want


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16, jnp.bfloat16])
@pytest.mark.parametrize("case", list(_CASES))
def test_k1_dot_bitwise(case, dtype):
    shapes, fn = _CASES[case]
    neg_zeros = 0
    for seed in range(4):
        want = _compare(fn, shapes, dtype, seed=seed).astype(np.float32)
        neg_zeros += int(np.count_nonzero((want == 0) & np.signbit(want)))
    # The data must actually produce a -0 product, or the case would not guard the fix.
    assert neg_zeros > 0, "no -0 product in the reference; the case does not exercise the K=1 sign of zero"


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16])
def test_k1_dot_preferred_f32(dtype):
    _compare(lambda a, b: lax.dot(a, b, preferred_element_type=jnp.float32), ((6, 1), (1, 5)), dtype)


def test_k1_signed_zero_explicit():
    cpu, mps = _devices()
    a = np.array([[0.0], [-0.0], [2.0]], np.float32)
    b = np.array([[-3.0, 3.0, 0.0, -0.0]], np.float32)
    want = np.asarray(jax.jit(jnp.matmul)(jax.device_put(a, cpu), jax.device_put(b, cpu)))
    got = np.asarray(jax.jit(jnp.matmul)(jax.device_put(a, mps), jax.device_put(b, mps)))
    np.testing.assert_array_equal(np.signbit(got), np.signbit(want))
    assert np.signbit(got[0, 0]) and not np.signbit(got[0, 1])


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int8])
def test_k1_dot_integer(dtype):
    _compare(lambda a, b: a @ b, ((3, 1), (1, 4)), dtype)


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
@pytest.mark.parametrize("case", list(_CONTROLS))
def test_k_ge2_controls(case, dtype):
    shapes, fn = _CONTROLS[case]
    _compare(fn, shapes, dtype, make=_exact_arr)
