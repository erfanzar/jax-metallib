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
"""lax.rng_bit_generator on MPS must match XLA:CPU bit for bit (bits and returned state).

Before the MPS lowering in ``jax_plugins/silicon/random_lowerings.py`` the native handler
ignored the input state, returned a zero state, rejected uint32 output and aborted the
process (SIGABRT) for uint8/uint16, so ``jax.random`` with ``impl='rbg'`` failed on MPS.
The vmap and sliced-key cases also guard an MPSGraph compile hang (> 60 s) that the key
gather in the lowering avoids. Each x64 mode runs in a child process so a native abort
fails one case, not the suite.
"""

import os
import subprocess
import sys

import pytest

CHILD = r"""
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

cpu, mps = jax.devices("cpu")[0], jax.devices("mps")[0]
ALGS = [lax.RandomAlgorithm.RNG_DEFAULT, lax.RandomAlgorithm.RNG_THREE_FRY, lax.RandomAlgorithm.RNG_PHILOX]
KEYS = [
    np.array([0x12345678, 0x9ABCDEF0, 0x0F1E2D3C, 0x4B5A6978], np.uint32),
    np.zeros(4, np.uint32),
    np.array([1, 2, 0xFFFFFFF0, 0xFFFFFFFF], np.uint32),
    np.array([0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 7], np.uint32),
]
x64 = jax.config.jax_enable_x64
DTYPES = [np.uint64] if x64 else [np.uint8, np.uint16, np.uint32]
SHAPES = [(), (0,), (1,), (3,), (3, 3), (3, 2), (4, 129), (5, 1, 2), (3, 5, 7), (2, 0, 3)]
bad = []
n = 0
t_start = time.perf_counter()


def run(jf, *args):
    return [
        [np.asarray(r) for r in jax.tree_util.tree_leaves(jf(*[jax.device_put(a, dev) for a in args]))]
        for dev in (cpu, mps)
    ]


def check(tag, jf, *args):
    global n
    n += 1
    c, m = run(jf, *args)
    ok = len(c) == len(m) and all(
        a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a, b) for a, b in zip(c, m)
    )
    if not ok:
        bad.append(tag)


for alg in ALGS:
    for dt in DTYPES:
        for shape in SHAPES:
            # One compile per configuration; every key reuses it.
            jf = jax.jit(lambda k, a=alg, d=dt, s=shape: lax.rng_bit_generator(k, s, dtype=d, algorithm=a))
            for i, key in enumerate(KEYS):
                check(f"{alg.name}/{np.dtype(dt).name}/{shape}/key{i}", jf, key)

        # Threading the state through a scan on device must give the same stream.
        def scan_fn(k, a=alg, d=dt):
            def body(state, _):
                return lax.rng_bit_generator(state, (5, 3), dtype=d, algorithm=a)

            return lax.scan(body, k, None, length=4)

        check(f"scan/{alg.name}/{np.dtype(dt).name}", jax.jit(scan_fn), KEYS[0])

    # vmap over a batch of keys (the batching rule slices keys[0]: a non-argument key).
    vf = jax.jit(jax.vmap(lambda k, a=alg: lax.rng_bit_generator(k, (6,), dtype=DTYPES[-1], algorithm=a)))
    check(f"vmap/{alg.name}", vf, np.stack(KEYS))
    # A key computed in the graph (sliced from a 2-D input).
    sf = jax.jit(lambda ks, a=alg: lax.rng_bit_generator(ks[1], (4, 6), dtype=DTYPES[-1], algorithm=a))
    check(f"slicekey/{alg.name}", sf, np.stack(KEYS))

if not x64:
    for impl in ("rbg", "unsafe_rbg"):
        kd = np.asarray(jax.random.key_data(jax.random.key(7, impl=impl)))
        for name, fn in (
            ("bits", lambda k: jax.random.bits(k, (1000,))),
            ("uniform", lambda k: jax.random.uniform(k, (64, 33))),
            ("split_bits", lambda k: jax.random.bits(jax.random.split(k, 3)[1], (17,), jnp.uint8)),
            ("split_vmap", lambda k: jax.vmap(lambda s: jax.random.bits(s, (9,)))(jax.random.split(k, 4))),
        ):
            jf = jax.jit(lambda d, f=fn, i=impl: f(jax.random.wrap_key_data(d, impl=i)))
            check(f"jax.random/{impl}/{name}", jf, kd)
        # normal goes through erf_inv, whose f32 MPS result is 1-2 ulp off XLA:CPU independently of
        # the generator (the default threefry PRNG shows the same gap); the bits feeding it are
        # checked bitwise above, so only bound the error here.
        n += 1
        nf = jax.jit(lambda d, i=impl: jax.random.normal(jax.random.wrap_key_data(d, impl=i), (257,)))
        c, m = run(nf, kd)
        tol = 4 * np.finfo(np.float32).eps * np.maximum(np.abs(c[0]), np.finfo(np.float32).tiny)
        if c[0].shape != m[0].shape or not np.all(np.abs(c[0] - m[0]) <= tol):
            bad.append(f"jax.random/{impl}/normal(4ulp)")

print(f"CHECKED={n} X64={x64} BAD={len(bad)} SECONDS={time.perf_counter() - t_start:.1f}")
for tag in bad[:20]:
    print("BAD", tag)
print("RBG_OK" if not bad else "RBG_FAIL")
sys.exit(1 if bad else 0)
"""


@pytest.mark.parametrize("x64", [False, True], ids=["x32", "x64"])
def test_rng_bit_generator_matches_cpu(x64):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps", JAX_ENABLE_X64="1" if x64 else "0")
    proc = subprocess.run([sys.executable, "-c", CHILD], env=env, capture_output=True, text=True, timeout=600)
    lines = [ln for ln in proc.stdout.splitlines() if ln.startswith(("CHECKED", "BAD", "RBG_"))]
    print("\n".join(lines))
    assert proc.returncode == 0, f"rc={proc.returncode}\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}"
    assert "RBG_OK" in proc.stdout
