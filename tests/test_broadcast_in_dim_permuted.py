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
"""broadcast_in_dim with non-increasing broadcast_dimensions (axis permutation) matches CPU.

The MPS lowering used to ignore the order of broadcast_dimensions, so a
permuting broadcast returned untransposed data.
"""

import jax
import numpy as np
import pytest


def _mps():
    try:
        return jax.devices("mps")[0]
    except RuntimeError:
        pytest.skip("mps backend not available")


@pytest.mark.parametrize(
    ("in_shape", "out_shape", "dims"),
    [
        ((7, 6), (6, 7), (1, 0)),
        ((2, 3, 4), (4, 2, 3), (1, 2, 0)),
        ((3, 4), (4, 5, 3), (2, 0)),
        ((3, 1), (5, 4, 3), (2, 0)),
        ((3, 4), (3, 5, 4), (0, 2)),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.int32, np.bool_])
def test_permuted_broadcast_in_dim(in_shape, out_shape, dims, dtype):
    mps = _mps()
    cpu = jax.devices("cpu")[0]
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(in_shape) * 10).astype(dtype)

    def f(x):
        return jax.lax.broadcast_in_dim(x, out_shape, dims)

    want = jax.jit(f)(jax.device_put(x, cpu))
    got = jax.jit(f)(jax.device_put(x, mps))
    np.testing.assert_array_equal(np.asarray(got), np.asarray(want))
