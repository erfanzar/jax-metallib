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
"""Scatters whose update window is narrower than the operand on a non-scattered
dim are scattered into the window and written back into the operand. The old
write-back used MPSGraph sliceUpdateDataTensor with constant starts, which
returned the operand unchanged whenever it was computed in the same graph
(every write lost, f32/f16/int32, set and add). Each case runs in its own
subprocess and must match XLA:CPU bit for bit."""

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
    layout, mode, dt, src = sys.argv[1:5]
    rng = np.random.default_rng(0)
    def mk(shape):
        if dt == "bool":
            return rng.random(shape) > 0.5
        if dt == "i32":
            return rng.integers(-2**31, 2**31 - 1, shape, dtype=np.int64).astype(np.int32)
        if dt == "u32":
            return rng.integers(0, 2**32, shape, dtype=np.uint64).astype(np.uint32)
        if dt == "c64":
            return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)
        return rng.standard_normal(shape).astype({"f32": np.float32, "f16": np.float16}[dt])
    def pre(x):
        if src == "arg":
            return x
        if src == "zeros":
            return jnp.zeros(x.shape, x.dtype)
        if dt == "bool":
            return jnp.logical_not(x)
        if dt in ("i32", "u32"):
            return x ^ x.dtype.type(5)
        return x * 2
    if layout == "rows2d":
        dn = lax.ScatterDimensionNumbers(update_window_dims=(1,), inserted_window_dims=(0,),
                                         scatter_dims_to_operand_dims=(0,))
        idx = np.array([[2], [5]], np.int32); x = mk((8, 9)); u = mk((2, 3))
    elif layout == "mid3d":
        dn = lax.ScatterDimensionNumbers(update_window_dims=(0, 2), inserted_window_dims=(1,),
                                         scatter_dims_to_operand_dims=(1,))
        idx = np.array([[1], [6], [3]], np.int32); x = mk((4, 8, 9)); u = mk((4, 3, 5))
    else:  # two narrowed dims
        dn = lax.ScatterDimensionNumbers(update_window_dims=(0, 2), inserted_window_dims=(1,),
                                         scatter_dims_to_operand_dims=(1,))
        idx = np.array([[0], [7]], np.int32); x = mk((4, 8, 9)); u = mk((2, 2, 5))
    def f(x, u):
        x = pre(x)
        return lax.scatter_add(x, idx, u, dn) if mode == "add" else lax.scatter(x, idx, u, dn)
    want = np.asarray(jax.jit(f)(jax.device_put(x, cpu), jax.device_put(u, cpu)))
    got = np.asarray(jax.jit(f)(jax.device_put(x, mps), jax.device_put(u, mps)))
    assert got.dtype == want.dtype and got.shape == want.shape, (got.dtype, got.shape, want.dtype, want.shape)
    np.testing.assert_array_equal(got.view(np.uint8), want.view(np.uint8))
    print("CASE_OK")
    """
)

_CASES = [
    (layout, mode, dt, src)
    for layout in ("rows2d", "mid3d", "two_narrow")
    for mode in ("set", "add")
    for dt in ("f32", "f16", "i32", "u32", "c64", "bool")
    for src in ("arg", "graph", "zeros")
    if not (mode == "add" and dt == "bool")
]


@pytest.mark.parametrize(("layout", "mode", "dt", "src"), _CASES, ids=["-".join(c) for c in _CASES])
def test_window_scatter_writeback_matches_cpu(layout, mode, dt, src):
    import jax

    try:
        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD, layout, mode, dt, src], capture_output=True, text=True, env=env)
    assert r.returncode == 0 and "CASE_OK" in r.stdout, f"rc={r.returncode}\n{r.stderr[-3000:]}"
