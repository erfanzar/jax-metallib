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
"""Last-axis float argmax/argmin on rows wider than the int32-key gate (2048).

These rows take the NaN-guarded `if` lowering in reduction_ops.mm (a NaN-propagating
row extremum decides whether the uint8 NaN-mask pass runs). The reference is NumPy's
first-NaN / first-extremum index on the float32 widening, which is exact for
f16/bf16 and matches XLA's argmax semantics (-0 == +0, NaN wins, first index on ties).
"""

import zlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

try:
    mps_devices = jax.devices("mps")
except RuntimeError:
    mps_devices = []
pytestmark = pytest.mark.skipif(not mps_devices, reason="requires the mps backend")

DTYPES = {"float32": np.float32, "float16": np.float16, "bfloat16": jnp.bfloat16}
SHAPES = [(8, 4096), (4, 16384), (3, 5000), (2, 3, 4100)]
PATTERNS = ["plain", "nan_some", "all_nan", "nan_first", "nan_last", "ties", "inf", "signed_zero"]


def _make(shape, pattern, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(shape).astype(np.float32)
    rows = x.reshape(-1, shape[-1])
    n, cols = rows.shape
    if pattern == "nan_some":
        for r in range(0, n, 2):
            rows[r, rng.integers(cols)] = np.nan
        rows[0, cols // 3] = np.nan
    elif pattern == "all_nan":
        rows[n - 1, :] = np.nan
    elif pattern == "nan_first":
        rows[::2, 0] = np.nan
    elif pattern == "nan_last":
        rows[:, -1] = np.nan
    elif pattern == "ties":
        rows[0, :] = 1.0
        rows[n - 1, 10:] = rows[n - 1].max()
        rows[n - 1, 7] = rows[n - 1, 10]
    elif pattern == "inf":
        rows[0, 5] = np.inf
        rows[0, 100] = np.inf
        rows[0, 200] = -np.inf
        rows[n - 1, :] = -np.inf
        if n > 2:
            rows[1, :] = np.inf
    elif pattern == "signed_zero":
        rows[0, :] = -0.0
        rows[0, cols // 2] = 0.0
        rows[n - 1, :] = 0.0
        rows[n - 1, 3] = -0.0
    return rows.reshape(shape)


def _mps(x):
    return jax.device_put(x, mps_devices[0])


@pytest.mark.parametrize("pattern", PATTERNS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
@pytest.mark.parametrize("dtype", list(DTYPES))
@pytest.mark.parametrize("op", ["argmax", "argmin"])
def test_wide_row_argreduce_matches_first_nan_reference(op, dtype, shape, pattern):
    seed = zlib.crc32(f"{op}-{dtype}-{shape}-{pattern}".encode())
    x = _make(shape, pattern, seed=seed).astype(DTYPES[dtype])
    ref = getattr(np, op)(np.asarray(x).astype(np.float32), axis=-1)
    fn = jax.jit(lambda v: getattr(jnp, op)(v, axis=-1))
    got = np.asarray(fn(_mps(x)))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_wide_row_argmax_inside_scan(dtype):
    # Inside a loop body the handler keeps the unconditional mask pass (no nested `if`).
    xs = np.stack([_make((4, 4096), p, seed=i) for i, p in enumerate(("plain", "nan_some", "all_nan"))])
    xs = xs.astype(DTYPES[dtype])
    ref = np.argmax(xs.astype(np.float32), axis=-1)

    def body(carry, x):
        return carry, jnp.argmax(x, axis=-1)

    fn = jax.jit(lambda v: jax.lax.scan(body, 0, v)[1])
    np.testing.assert_array_equal(np.asarray(fn(_mps(xs))), ref)


def test_wide_row_argmax_and_max_together():
    # Value and index from the same row in one program; the value is gathered at the index.
    x = _make((8, 4096), "nan_some", seed=7)
    fn = jax.jit(lambda v: (jnp.argmax(v, axis=-1), jnp.argmin(v, axis=-1), jnp.max(v, axis=-1)))
    i_max, i_min, v_max = (np.asarray(t) for t in fn(_mps(x)))
    np.testing.assert_array_equal(i_max, np.argmax(x, axis=-1))
    np.testing.assert_array_equal(i_min, np.argmin(x, axis=-1))
    np.testing.assert_array_equal(v_max, np.max(x, axis=-1))
