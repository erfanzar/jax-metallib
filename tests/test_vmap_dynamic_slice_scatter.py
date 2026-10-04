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
"""vmapped dynamic_slice / dynamic_update_slice and drop-mode window scatters match CPU.

vmap(dynamic_slice) lowers to a gather with operand_batching_dims and a
multi-component index vector; vmap(dynamic_update_slice) to the matching
scatter. Both were rejected ("unsupported gather/scatter pattern"); they now
use an explicit-coordinate lowering. A windowed scatter in mode="drop" whose
window only partly fits was partially written (scatterND drops elements one at
a time) while XLA skips the whole window. Each case runs in a subprocess so a
regression fails one case instead of the suite.
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
    from jax import lax
    cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)
    B = 5
    x = rng.standard_normal((B, 8, 9)).astype(np.float32)
    x4 = rng.standard_normal((B, 3, 8, 9)).astype(np.float32)
    xi = rng.integers(-2**30, 2**30, (B, 8, 9)).astype(np.int32)
    xb = rng.random((B, 8, 9)) > 0.5
    u = rng.standard_normal((B, 2, 3)).astype(np.float32)
    s = rng.integers(0, 7, B).astype(np.int32)
    t = rng.integers(0, 7, B).astype(np.int32)
    so = np.array([-3, 0, 7, 12, 2], np.int32)
    to = np.array([9, -1, 8, 3, 100], np.int32)
    ds = lambda x, s, t: lax.dynamic_slice(x, (s, t), (2, 3))
    dus = lambda x, u, s, t: lax.dynamic_update_slice(x, u, (s, t))
    x2 = x[0]
    wu = rng.standard_normal((5, 2, 3)).astype(np.float32)
    dn = lax.ScatterDimensionNumbers(update_window_dims=(1, 2), inserted_window_dims=(),
                                     scatter_dims_to_operand_dims=(0, 1))
    oob = np.array([[7, 8], [9, 0], [-3, 2], [2, 7], [3, 3]], np.int32)
    neg = np.array([[-1, 0], [0, -1], [8, 0], [0, 9], [3, 3]], np.int32)
    far = np.array([[-1000, 2], [2, -1000], [2**30, 1], [1, 2**30], [6, 6], [7, 0], [0, 7], [-2, -2], [4, 4]],
                   np.int32)
    dn3 = lax.ScatterDimensionNumbers(update_window_dims=(1, 2, 3), inserted_window_dims=(),
                                      scatter_dims_to_operand_dims=(0, 1, 2))
    idx3 = np.array([[1, 0, 0], [2, 0, 0], [0, 5, 0], [-1, 2, 2], [0, 3, 3], [1, 4, 5]], np.int32)
    cases = {
        "ds_both": (jax.vmap(ds), (x, s, t)),
        "ds_shared_x": (jax.vmap(ds, in_axes=(None, 0, 0)), (x2, s, t)),
        "ds_shared_idx": (jax.vmap(ds, in_axes=(0, None, None)), (x, np.int32(3), np.int32(5))),
        "ds_clamped_starts": (jax.vmap(ds), (x, so, to)),
        "ds_int32": (jax.vmap(ds), (xi, s, t)),
        "ds_bool": (jax.vmap(ds), (xb, s, t)),
        "ds_f16": (jax.vmap(ds), (x.astype(np.float16), s, t)),
        "ds_inner_axes": (jax.vmap(lambda x, s, t: lax.dynamic_slice(x, (0, s, t), (3, 2, 3))), (x4, s, t)),
        "ds_in_axes1": (jax.vmap(ds, in_axes=(1, 0, 0)), (np.ascontiguousarray(np.moveaxis(x, 0, 1)), s, t)),
        "dus_both": (jax.vmap(dus), (x, u, s, t)),
        "dus_shared_x": (jax.vmap(dus, in_axes=(None, 0, 0, 0)), (x2, u, s, t)),
        "dus_clamped_starts": (jax.vmap(dus), (x, u, so, to)),
        "dus_int32": (jax.vmap(dus), (xi, u.astype(np.int32) * 7, s, t)),
        "grad_ds": (jax.grad(lambda x, s, t: (jax.vmap(ds)(x, s, t) * 2.0).sum()), (x, s, t)),
        "drop_add_partial_window": (lambda x, i, u: lax.scatter_add(x, i, u, dn, mode="drop"), (x2, oob, wu)),
        "drop_set_partial_window": (lambda x, i, u: lax.scatter(x, i, u, dn, mode="drop"), (x2, oob, wu)),
        "drop_add_unit_window_negative": (
            lambda x, i, u: lax.scatter_add(x, i, u[:, :1, :1], dn, mode="drop"), (x2, neg, wu)),
        "drop_set_rows": (
            lambda x, i, u: x.at[i[:, 0]].set(jnp.tile(u[:, 0, :], (1, 3)), mode="drop"), (x2, oob, wu)),
        "dus_bool": (jax.vmap(dus), (xb, rng.random((B, 2, 3)) > 0.5, so, to)),
        "dus_int8": (jax.vmap(dus), (xi.astype(np.int8), (u * 50).astype(np.int8), s, t)),
        "dus_c64": (jax.vmap(dus), ((x + 1j * x[::-1]).astype(np.complex64), (u - 2j).astype(np.complex64), so, to)),
        "dus_f16": (jax.vmap(dus), (x.astype(np.float16), u.astype(np.float16), s, t)),
        "drop_add_far_oob_starts": (
            lambda x, i, u: lax.scatter_add(x, i, u, dn, mode="drop"),
            (x2, far, rng.standard_normal((far.shape[0], 2, 3)).astype(np.float32))),
        "drop_add_3d_row_oob": (
            lambda x, i, u: lax.scatter_add(x, i, u, dn3, mode="drop"),
            (x4[0], idx3, rng.standard_normal((idx3.shape[0], 2, 4, 4)).astype(np.float32))),
    }
    f, args = cases[sys.argv[1]]
    want = np.asarray(jax.jit(f)(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, got.shape, want.dtype, want.shape)
    if sys.argv[1].startswith("drop_add") and np.issubdtype(got.dtype, np.floating):
        # Overlapping float scatter-add windows: XLA leaves the order of updates
        # to one element implementation-defined, and GPU accumulation order
        # varies run to run (1 ulp on ~1 in 10 runs of drop_add_3d_row_oob).
        np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-6)
    else:
        np.testing.assert_array_equal(got, want)
    print("CASE_OK")
    """
)

_CASES = [
    "ds_both",
    "ds_shared_x",
    "ds_shared_idx",
    "ds_clamped_starts",
    "ds_int32",
    "ds_bool",
    "ds_f16",
    "ds_inner_axes",
    "ds_in_axes1",
    "dus_both",
    "dus_shared_x",
    "dus_clamped_starts",
    "dus_int32",
    "grad_ds",
    "drop_add_partial_window",
    "drop_set_partial_window",
    "drop_add_unit_window_negative",
    "drop_set_rows",
    "dus_bool",
    "dus_int8",
    "dus_c64",
    "dus_f16",
    "drop_add_far_oob_starts",
    "drop_add_3d_row_oob",
]


@pytest.mark.parametrize("name", _CASES)
def test_vmap_dynamic_slice_and_window_scatter_match_cpu(name):
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD, name], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "CASE_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
