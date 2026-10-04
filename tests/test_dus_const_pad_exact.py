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
"""dynamic_update_slice and pad lowerings that silently lost data.

1. Top-level dynamic_update_slice with constant starts (literal, or folded from
   constants by jnp index normalisation) returned the operand unchanged when the
   operand or update was computed in the same graph, for every dtype. A full
   replacement (update shape == operand shape) was dropped with traced starts too.
2. The scatterND fallback rounded uint32 through f32, dropped bool writes and
   aborted on complex; interior pad used it too.
3. Edge pad rounded integer pad values beyond 2^24 through f32, and a traced
   float pad value used (x - p) -> pad 0 -> + p (inexact, NaN for p = +-inf).
Each case runs in a subprocess so a regression fails one case.
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
    name, dts = sys.argv[1].split(":")
    dt = {"f32": np.float32, "f16": np.float16, "bf16": jnp.bfloat16, "i32": np.int32, "u32": np.uint32,
          "i8": np.int8, "u8": np.uint8, "u16": np.uint16, "bool": np.bool_, "c64": np.complex64}[dts]
    rng = np.random.default_rng(11)
    def mk(shape):
        if dt == np.bool_:
            return rng.integers(0, 2, size=shape).astype(np.bool_)
        if dt == np.complex64:
            return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(dt)
        if np.issubdtype(np.dtype(dt), np.integer):
            info = np.iinfo(dt)
            return rng.integers(info.min, info.max, size=shape, dtype=np.int64).astype(dt)
        return (rng.standard_normal(shape) * 1e3).astype(dt)
    a, b = mk((4, 16, 16)), mk((4, 4, 4))
    op = (lambda x: ~x) if dt == np.bool_ else (lambda x: x + x)
    D = lax.dynamic_update_slice
    if np.issubdtype(np.dtype(dt), np.integer):
        pv = np.array(np.iinfo(dt).max - 2, dt)
    elif dt == np.bool_:
        pv = np.array(True)
    else:
        pv = np.array(-7, dt)
    s = np.array([5, 9], np.int32)
    cases = {
        "dus_slices": lambda x, y, p: D(x[0], y[0], (5, 9)),
        "dus_computed": lambda x, y, p: D(op(x[1]), op(y[2]), (5, 9)),
        "dus_clamp": lambda x, y, p: D(op(x[0]), op(y[0]), (-3, 40)),
        "dus_rank1": lambda x, y, p: D(op(x[0, 0]), op(y[0, 0]), (7,)),
        "dus_rank3": lambda x, y, p: D(op(x), op(y[:1]), (2, 5, 9)),
        "dus_folded_starts": lambda x, y, p: D(x[0], y[0], (jnp.int32(s[0]), jnp.int32(s[1]))),
        "dus_full_const": lambda x, y, p: D(op(x[0]), op(x[1]), (0, 0)),
        "dus_full_traced": lambda x, y, p: D(op(x[0]), op(x[1]), (jnp.argmax(y[0, 0] != y[0, 0]), 0)),
        "dus_twice": lambda x, y, p: D(D(op(x[0]), y[0], (5, 9)), y[1], (0, 0)),
        "dus_loop": lambda x, y, p: lax.fori_loop(0, jnp.int32(x.shape[0] - 2), lambda i, h: D(h, y[0], (5, 9)), x[0]),
        "pad_interior": lambda x, y, p: lax.pad(op(x[0]), p, [(2, -3, 1), (0, 1, 2)]),
        "pad_edge_const": lambda x, y, p: lax.pad(op(x[0]), pv, [(1, 2, 0), (3, 0, 0)]),
        "pad_edge_traced": lambda x, y, p: lax.pad(op(x[0]), p, [(1, 2, 0), (3, 0, 0)]),
        "pad_edge_crop": lambda x, y, p: lax.pad(op(x[0]), p, [(-1, 2, 0), (3, -2, 0)]),
    }
    f = cases[name]
    args = (a, b, pv)
    want = np.asarray(jax.jit(f)(*[jax.device_put(v, cpu) for v in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(v, mps) for v in args]))
    assert got.shape == want.shape and got.dtype == want.dtype, (got.shape, want.shape, got.dtype, want.dtype)
    if got.tobytes() != want.tobytes():
        bad = np.flatnonzero(got.ravel() != want.ravel())
        raise SystemExit(f"MISMATCH n={bad.size} got={got.ravel()[bad[:3]]} want={want.ravel()[bad[:3]]}")
    print("CASE_OK")
    """
)

_ALL = ["f32", "f16", "bf16", "i32", "u32", "i8", "u8", "u16", "bool", "c64"]
_CASES = [f"{n}:{d}" for n in ("dus_computed", "dus_clamp", "dus_full_const", "dus_rank3") for d in _ALL]
_CASES += [
    f"{n}:{d}"
    for n in ("dus_slices", "dus_rank1", "dus_folded_starts", "dus_full_traced", "dus_twice")
    for d in ("f32", "i32", "u32", "c64")
]
_CASES += [f"dus_loop:{d}" for d in ("f32", "i32")]
_CASES += [
    f"{n}:{d}"
    for n in ("pad_interior", "pad_edge_const", "pad_edge_traced", "pad_edge_crop")
    for d in ("f32", "i32", "u32", "u16", "bool", "c64")
]


def _run(case: str) -> subprocess.CompletedProcess:
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    return subprocess.run([sys.executable, "-c", _CHILD, case], capture_output=True, text=True, env=env, timeout=300)


@pytest.mark.parametrize("case", _CASES)
def test_dus_const_pad_exact(case):
    r = _run(case)
    tail = (r.stdout + r.stderr)[-1500:]
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), tail


def test_pad_edge_traced_nonfinite():
    code = textwrap.dedent(
        """
        import jax, numpy as np
        from jax import lax
        cpu = jax.devices("cpu")[0]; mps = jax.devices("mps")[0]
        x = np.array([[1, -0.0, np.nan], [3, 4, 9053559.0]], np.float32)
        f = jax.jit(lambda a, p: lax.pad(a * 1, p, [(1, 2, 0), (0, 1, 0)]))
        for p in (np.inf, -np.inf, np.nan, -0.0, 7.5):
            w = np.asarray(f(jax.device_put(x, cpu), jax.device_put(np.float32(p), cpu)))
            g = np.asarray(f(jax.device_put(x, mps), jax.device_put(np.float32(p), mps)))
            assert w.tobytes() == g.tobytes(), (p, w, g)
        print("CASE_OK")
        """
    )
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0 and r.stdout.strip().endswith("CASE_OK"), (r.stdout + r.stderr)[-1500:]
