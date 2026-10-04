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
"""vmapped scatters whose only scatter dims are batching dims match CPU.

vmap(lambda x, i, v: x.at[:, i].set(v)) lowers to a scatter with
input_batching_dims=[0], inserted_window_dims=[2] and updates [B, R]. The
updates lacked the inserted scatter axis, so MPSGraph scatterAlongAxis asserted
("indices rank and result rank must be equal") and aborted the process. Each
case runs in a subprocess so a regression fails one case instead of the suite.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_CHILD = textwrap.dedent(
    """
    import sys
    import jax, jax.numpy as jnp, numpy as np
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)
    B = 5
    x = rng.standard_normal((B, 8, 9)).astype(np.float32)
    xi = rng.integers(-9, 9, (B, 8, 9)).astype(np.int32)
    col = rng.integers(0, 9, B).astype(np.int32)
    row = rng.integers(0, 8, B).astype(np.int32)
    vc = rng.standard_normal((B, 8)).astype(np.float32)
    vr = rng.standard_normal((B, 9)).astype(np.float32)
    cases = {
        "set_col": (jax.vmap(lambda x, i, v: x.at[:, i].set(v)), (x, col, vc)),
        "add_col": (jax.vmap(lambda x, i, v: x.at[:, i].add(v)), (x, col, vc)),
        "max_col": (jax.vmap(lambda x, i, v: x.at[:, i].max(v)), (x, col, vc)),
        "set_row": (jax.vmap(lambda x, i, v: x.at[i].set(v)), (x, row, vr)),
        "set_col_int": (jax.vmap(lambda x, i, v: x.at[:, i].set(v)), (xi, col, vc.astype(np.int32))),
        "set_col_in_axes1": (
            jax.vmap(lambda x, i, v: x.at[:, i].set(v), in_axes=(1, 0, 0)),
            (np.ascontiguousarray(np.moveaxis(x, 0, 1)), col, vc),
        ),
    }
    f, args = cases[sys.argv[1]]
    want = np.asarray(jax.jit(f)(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, got.shape, want.dtype, want.shape)
    np.testing.assert_allclose(got, want, rtol=0, atol=0)
    print("CASE_OK")
    """
)

_CASES = ["set_col", "add_col", "max_col", "set_row", "set_col_int", "set_col_in_axes1"]


@pytest.mark.parametrize("name", _CASES)
def test_vmap_batched_scatter_matches_cpu(name):
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD, name], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "CASE_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
