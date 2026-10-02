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
"""Region/attribute semantics that used to miscompile silently.

* gather clamps start indices into [0, dim - slice_size] (StableHLO); MPS returned
  zeros for out-of-range/negative indices, so take(mode="clip") was wrong.
* reduce_window/reduce named the reducer after the FIRST op of the body, so
  lax.cumlogsumexp (logaddexp region starting with maximum) ran as a cummax.
* scatter defaulted unrecognised update bodies to "set", so x.at[i].apply(f)
  wrote the zero updates. It is now rejected with an error.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _devices():
    try:
        mps = jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")
    return jax.devices("cpu")[0], mps


def _run(fn, *args):
    cpu, mps = _devices()
    want = jax.jit(fn)(*[jax.device_put(a, cpu) for a in args])
    got = jax.jit(fn)(*[jax.device_put(a, mps) for a in args])
    return np.asarray(want), np.asarray(got)


_IDX = [[0, 7, -2, 4], [7], [-2], [3, -1], [0, 4, 4, 1]]


@pytest.mark.parametrize("idx", _IDX)
@pytest.mark.parametrize("axis", [0, 1])
def test_take_clip_clamps(idx, axis):
    x = np.arange(15, dtype=np.float32).reshape(3, 5)
    if axis == 0:
        x = x.T.copy()
    i = np.array(idx, np.int32)
    want, got = _run(lambda x, i: jnp.take(x, i, axis=axis, mode="clip"), x, i)
    np.testing.assert_array_equal(got, want)


def test_take_clip_int32_operand():
    x = np.arange(40, dtype=np.int32).reshape(8, 5) * 7 - 100
    i = np.array([9, -3, 2, 7, 100], np.int32)
    want, got = _run(lambda x, i: jnp.take(x, i, axis=0, mode="clip"), x, i)
    np.testing.assert_array_equal(got, want)


def test_two_component_index_clip():
    # Index vector of length 2 exercises the per-component upper bound.
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 6, 3)).astype(np.float32)
    i = np.array([0, 5, -1, 3, 9], np.int32)
    j = np.array([9, 2, -4, 5, 0], np.int32)
    want, got = _run(lambda x, i, j: x.at[i, j].get(mode="clip"), x, i, j)
    np.testing.assert_array_equal(got, want)


def test_gather_wide_slice_clamps_to_dim_minus_slice():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((10, 4)).astype(np.float32)
    starts = np.array([[0], [8], [9], [-5], [6]], np.int32)
    dn = lax.GatherDimensionNumbers(offset_dims=(1, 2), collapsed_slice_dims=(), start_index_map=(0,))

    def f(x, s):
        return lax.gather(x, s, dn, slice_sizes=(3, 4), mode=lax.GatherScatterMode.CLIP)

    want, got = _run(f, x, starts)
    np.testing.assert_array_equal(got, want)


def test_take_fill_still_fills():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 3)).astype(np.float32)
    i = np.array([0, 7, 4, -9, 2], np.int32)
    want, got = _run(lambda x, i: jnp.take(x, i, axis=0, mode="fill", fill_value=-1.0), x, i)
    np.testing.assert_array_equal(got, want)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cumlogsumexp(axis, reverse):
    rng = np.random.default_rng(3)
    x = (rng.standard_normal((5, 7, 6)) * 4).astype(np.float32)
    want, got = _run(lambda x: lax.cumlogsumexp(x, axis=axis, reverse=reverse), x)
    np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5)


def test_cumlogsumexp_special_values():
    x = np.array(
        [[-np.inf, -np.inf, 1.0, np.inf, 2.0, -np.inf], [0.0, np.nan, 3.0, -1.0, -np.inf, 5.0]],
        np.float32,
    )
    for reverse in (False, True):
        want, got = _run(lambda x, r=reverse: lax.cumlogsumexp(x, axis=1, reverse=r), x)
        np.testing.assert_allclose(got, want, rtol=2e-5, atol=2e-5, equal_nan=True)


def test_cumlogsumexp_long_axis_and_f16():
    rng = np.random.default_rng(4)
    x = rng.standard_normal((3, 1000)).astype(np.float32)
    want, got = _run(lambda x: lax.cumlogsumexp(x, axis=1), x)
    np.testing.assert_allclose(got, want, rtol=5e-5, atol=5e-5)
    h = rng.standard_normal((4, 33)).astype(np.float16)
    want, got = _run(lambda x: lax.cumlogsumexp(x, axis=1, reverse=True), h)
    assert got.dtype == np.float16
    np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=1e-2, atol=1e-2)


def test_cumlogsumexp_grad():
    rng = np.random.default_rng(5)
    x = rng.standard_normal((4, 9)).astype(np.float32)
    want, got = _run(jax.grad(lambda x: jnp.sum(jnp.sin(lax.cumlogsumexp(x, axis=1)))), x)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    "name",
    ["set", "add", "multiply", "min", "max", "divide", "sub"],
)
def test_supported_scatter_bodies(name):
    rng = np.random.default_rng(6)
    x = rng.standard_normal((6, 3)).astype(np.float32)
    i = np.array([1, 4, 1, 0], np.int32)
    u = (rng.standard_normal((4, 3)) + 2.0).astype(np.float32)
    if name == "sub":
        dn = lax.ScatterDimensionNumbers(
            update_window_dims=(1,), inserted_window_dims=(0,), scatter_dims_to_operand_dims=(0,)
        )

        def f(x, i, u):
            return lax.scatter_sub(x, i[:, None], u, dn)

    else:

        def f(x, i, u):
            return getattr(x.at[i], name)(u)

    want, got = _run(f, x, i, u)
    if name == "set":
        # Duplicate-index set order is unspecified; compare the untouched rows.
        keep = np.array([2, 3, 5])
        np.testing.assert_array_equal(got[keep], want[keep])
        np.testing.assert_array_equal(got[4], want[4])
    else:
        np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("f", [jnp.sin, jnp.negative, lambda v: v * v], ids=["sin", "neg", "square"])
def test_scatter_apply_is_rejected_not_zeroed(f):
    _, mps = _devices()
    x = jax.device_put(np.linspace(-1, 1, 6, dtype=np.float32), mps)
    i = jax.device_put(np.array([1, 4, 1], np.int32), mps)
    with pytest.raises(Exception, match="unsupported update computation"):
        np.asarray(jax.jit(lambda x, i: x.at[i].apply(f))(x, i))
