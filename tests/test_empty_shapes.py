# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Zero-element arrays must behave like XLA:CPU, bit for bit.

Before the fix every rank >= 2 empty input aborted the process (MPSNDArray
"buffer is not large enough. Must be 16 bytes": the 4-byte placeholder was
smaller than MPSNDArray's binding size, which counts each zero dim as 1),
reducing a zero-length axis aborted inside MPSGraph's reduction encode, f16/bf16
prod over an empty axis returned f32 bytes read as halves ([0, 1.875, ...]), and
dot_general with K=0 or M=0 raised "zero-sized dimensions are not supported".

All cases run in one child process so a regressed abort fails this test instead
of killing the suite. Not covered (pre-existing, in the identity fast path of
mps_executable.mm): jit identity / 1-D reshape, 2-D where and cumsum on empty
inputs, which raise "Failed to allocate buffer for identity output".
"""

import os
import subprocess
import sys

import jax
import pytest

try:
    mps_devices = jax.devices("mps")
except RuntimeError:
    mps_devices = []
pytestmark = pytest.mark.skipif(not mps_devices, reason="requires the mps backend")

_CHILD = r"""
import jax
import jax.numpy as jnp
import numpy as np

mps, cpu = jax.devices("mps")[0], jax.devices("cpu")[0]


def fl(v):
    return jnp.issubdtype(v.dtype, jnp.floating)


FNS = {
    "exp": lambda v: jnp.exp(v) if fl(v) else v + 1,
    "add": lambda v: v + v,
    "reshape": lambda v: v.reshape(-1),
    "sum": lambda v: jnp.sum(v, axis=-1),
    "sum_all": lambda v: jnp.sum(v),
    "sum_ax0": lambda v: jnp.sum(v, axis=0),
    "prod": lambda v: jnp.prod(v, axis=-1),
    "max": lambda v: jnp.max(v, axis=-1, initial=-jnp.inf if fl(v) else -7),
    "min": lambda v: jnp.min(v, axis=-1, initial=jnp.inf if fl(v) else 7),
    "any": lambda v: jnp.any(v != 0, axis=-1),
    "all": lambda v: jnp.all(v != 0, axis=-1),
    "keepdims": lambda v: jnp.sum(v, axis=-1, keepdims=True),
    "dot": lambda v: v @ jnp.ones((v.shape[-1], 3), v.dtype),
    "concat": lambda v: jnp.concatenate([v, jnp.ones_like(v)], axis=0),
}
REDUCE = ("sum", "sum_all", "sum_ax0", "prod", "max", "min", "any", "all", "keepdims")

bad, n = [], 0
for dt in (np.float32, np.float16, jnp.bfloat16, np.int32):
    for name, f in FNS.items():
        shapes = [(4, 0), (0, 4), (0, 0)]
        if name in REDUCE:
            shapes.append((0,))
        if name in ("dot", "concat"):
            shapes = [(4, 0), (0, 4)]
        for shape in shapes:
            x = np.zeros(shape, np.float32).astype(dt)
            g = jax.jit(f)
            got = np.asarray(g(jax.device_put(x, mps)))
            want = np.asarray(g(jax.device_put(x, cpu)))
            n += 1
            if got.shape != want.shape or got.dtype != want.dtype or got.tobytes() != want.tobytes():
                bad.append(f"{np.dtype(dt).name}/{name}/{shape}: {got.shape} {got.ravel()[:3]} vs {want.ravel()[:3]}")

# device_put round trip and eager ops on empty arrays.
for shape in [(0,), (4, 0), (0, 4), (2, 0, 3)]:
    x = np.zeros(shape, np.float32)
    d = jax.device_put(x, mps)
    n += 1
    if np.asarray(d).shape != shape or np.asarray(jnp.exp(d)).shape != shape:
        bad.append(f"eager/{shape}")
print("BAD", len(bad), "of", n)
for b in bad:
    print("  ", b)
print("CHILD_DONE")
"""


def test_empty_shapes_match_cpu_bitwise():
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0 and "CHILD_DONE" in r.stdout, f"rc={r.returncode}\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}"
    assert "BAD 0 of" in r.stdout, r.stdout
