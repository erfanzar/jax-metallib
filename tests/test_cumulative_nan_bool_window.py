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
"""Cumulative/sliding max-min propagate NaN, and bool reduce_window works.

MPSGraph cumulativeMaximum/Minimum skip NaN, so lax.cummax([3, nan, 5])
returned [3, 3, 5] where XLA propagates NaN from its position onward; the
sliding fold and the final init fold also used the non-propagating max/min.
bool reduce_window with max/min/or/and raised "unsupported pattern".
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax


def _devices():
    try:
        return jax.devices("cpu")[0], jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("cpu and mps backends are both required")


def _run(fn, x):
    cpu, mps = _devices()
    want = np.asarray(jax.jit(fn)(jax.device_put(x, cpu)))
    got = np.asarray(jax.jit(fn)(jax.device_put(x, mps)))
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, got.shape, want.dtype, want.shape)
    return got, want


_NAN_FNS = {
    "cummin": lambda x: lax.cummin(x, axis=0),
    "cummax": lambda x: lax.cummax(x, axis=0),
    "cummin_rev": lambda x: lax.cummin(x, axis=0, reverse=True),
    "cummax_rev": lambda x: lax.cummax(x, axis=0, reverse=True),
    "cummax_2d": lambda x: lax.cummax(jnp.stack([x, x[::-1]]), axis=1),
    "slide_max_same": lambda x: lax.reduce_window(x, -jnp.inf, lax.max, (3,), (1,), "SAME"),
    "slide_min_pad": lambda x: lax.reduce_window(x, jnp.inf, lax.min, (2,), (1,), ((1, 0),)),
    "slide_max_2d": lambda x: lax.reduce_window(jnp.stack([x, x[::-1]]), -jnp.inf, lax.max, (1, 3), (1, 1), "VALID"),
}


@pytest.mark.parametrize("dtype", [np.float32, np.float16, jnp.bfloat16])
@pytest.mark.parametrize("name", sorted(_NAN_FNS))
def test_extrema_scans_propagate_nan(name, dtype):
    x = np.array([3, 1, np.nan, 0, -1, 2, np.nan, 5], np.float32).astype(dtype)
    got, want = _run(_NAN_FNS[name], x)
    np.testing.assert_array_equal(got.astype(np.float32), want.astype(np.float32))


def test_cummax_without_nan_and_int_unchanged():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((6, 33)).astype(np.float32)
    got, want = _run(lambda v: lax.cummax(v, axis=1), x)
    np.testing.assert_array_equal(got, want)
    xi = rng.integers(-(2**30), 2**30, (6, 33)).astype(np.int32)
    got, want = _run(lambda v: lax.cummin(v, axis=1, reverse=True), xi)
    np.testing.assert_array_equal(got, want)


_BOOL_FNS = {
    "max_valid": lambda x: lax.reduce_window(x, False, lax.max, (2, 2), (1, 1), "VALID"),
    "min_same": lambda x: lax.reduce_window(x, True, lax.min, (3, 2), (1, 1), "SAME"),
    "max_stride_pad": lambda x: lax.reduce_window(x, False, lax.max, (2, 3), (2, 2), ((1, 0), (1, 1))),
    "max_init_true": lambda x: lax.reduce_window(x, True, lax.max, (2, 2), (1, 1), "VALID"),
    "min_win_dil": lambda x: lax.reduce_window(x, True, lax.min, (2, 2), (1, 1), "VALID", window_dilation=(2, 1)),
    "or": lambda x: lax.reduce_window(x, False, lax.bitwise_or, (2, 2), (1, 1), "VALID"),
    "and": lambda x: lax.reduce_window(x, True, lax.bitwise_and, (2, 2), (1, 1), "VALID"),
}


@pytest.mark.parametrize("name", sorted(_BOOL_FNS))
def test_bool_reduce_window_matches_cpu(name):
    x = np.random.default_rng(3).random((5, 7)) > 0.6
    got, want = _run(_BOOL_FNS[name], x)
    np.testing.assert_array_equal(got, want)
