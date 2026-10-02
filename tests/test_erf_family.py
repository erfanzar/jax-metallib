# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""erf / erfc / exact gelu against XLA:CPU.

The test-config machinery only marks stablehlo.erf as exercised, so these ops had
no value test. Measured on this backend (86k inputs incl. +-0, tiny, tails, inf,
NaN):

* f16 and bf16 erf/erfc/gelu are bitwise equal to XLA:CPU.
* bf16 exact gelu matches only with --xla_allow_excess_precision=false on the CPU
  side. JAX lowers gelu as erfc(bf16 multiply(-x, 0.70703)); XLA:CPU by default
  drops that bf16 rounding and evaluates erfc on the unrounded product, while MPS
  honours the StableHLO bf16 type. In erfc's tail the difference is a factor of
  exp(t_rounded^2 - t^2) (x=-11.625: -1.81e-30 vs -1.09e-30, 98 ulp), so default
  XLA:CPU agrees on only ~67% of inputs. Without excess precision it is bitwise.
* f32 is not bitwise: the erf composite is inlined and evaluated without FMA
  (erf 7, erfc 6, gelu 9 ulp max). The bounds below catch regressions beyond that.

The XLA flag is read at backend start, and earlier suite tests may export
JAX_PLATFORMS=mps, so every comparison runs in a child with both overridden.
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
from jax.scipy import special as jsp

cpu = jax.devices("cpu")[0]
mps = jax.devices("mps")[0]
rng = np.random.default_rng(0)
special = [0.0, -0.0, 1e-30, -1e-30, 1e-8, 1e-4, 0.5, -0.5, 1.0, 2.0, 3.9, 4.0, 9.0, 10.0, 30.0, -30.0,
           -11.625, np.inf, -np.inf, np.nan]
base = np.concatenate([
    rng.standard_normal(16384) * 3,
    rng.uniform(-12, 12, 8192),
    np.logspace(-30, 1.5, 1024),
    -np.logspace(-30, 1.5, 1024),
    np.array(special),
]).astype(np.float32)

FUNCS = {
    "erf": jsp.erf,
    "erfc": jsp.erfc,
    "gelu_exact": lambda x: jax.nn.gelu(x, approximate=False),
}
# Max allowed ulp distance from XLA:CPU (f32 measured 7 / 6 / 9).
F32_BOUND = {"erf": 8, "erfc": 8, "gelu_exact": 12}
BITS = {"float32": np.int32, "float16": np.int16, "bfloat16": np.int16}


def key(v, dtype):
    bits = BITS[dtype]
    i = np.asarray(v).view(bits).astype(np.int64)
    sign = np.int64(1) << (np.dtype(bits).itemsize * 8 - 1)
    return np.where(i < 0, -(i + sign), i)


bad = []
for dtype in (jnp.float32, jnp.float16, jnp.bfloat16):
    name_t = np.dtype(dtype).name
    x = jnp.asarray(base).astype(dtype)
    for name, f in FUNCS.items():
        g = jax.jit(f)
        c = np.asarray(g(jax.device_put(x, cpu)))
        m = np.asarray(g(jax.device_put(x, mps)))
        cn = np.isnan(c.astype(np.float32))
        mn = np.isnan(m.astype(np.float32))
        if (cn != mn).any():
            bad.append(f"{name_t} {name}: NaN mismatch at {int((cn != mn).sum())} inputs")
            continue
        ok = ~cn
        d = np.abs(key(c, name_t) - key(m, name_t))[ok]
        bound = F32_BOUND[name] if name_t == "float32" else 0
        worst = int(d.max())
        print(f"{name_t} {name} max_ulp={worst} bound={bound}")
        if worst > bound:
            bad.append(f"{name_t} {name}: max {worst} ulp > {bound}")
if bad:
    raise SystemExit("; ".join(bad))
print("ERF_FAMILY_OK")
"""


def test_erf_family_matches_xla_cpu():
    flags = os.environ.get("XLA_FLAGS", "")
    env = dict(
        os.environ,
        JAX_PLATFORMS="cpu,mps",
        XLA_FLAGS=(flags + " --xla_allow_excess_precision=false").strip(),
    )
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD], env=env, capture_output=True, text=True, timeout=600, check=False
    )
    out = proc.stdout + proc.stderr
    print(proc.stdout)
    assert proc.returncode == 0 and "ERF_FAMILY_OK" in proc.stdout, out[-4000:]
