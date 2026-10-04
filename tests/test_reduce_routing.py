# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Short-row multi-row reductions must stay inside the MPSGraph segment.

With the intrinsic kernel family off, every last-axis sum of >= 32768 elements
was routed to the native reduce kernel. For short rows (LayerNorm, in-layer
softmax) that split the plan into graph/native/graph segments and made
LayerNorm [16,256,512] 3.4x slower (1.19 ms vs 0.35 ms). Long rows and
single-row reduces must still go native, and values must match XLA:CPU.
The plan is read from JAX_METALLIB_TRACE_PLAN output in a child process.
"""

import json
import os
import subprocess
import sys
import textwrap

import jax
import numpy as np
import pytest
from jax import numpy as jnp

_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np

    def layernorm(x):
        m = x.mean(-1, keepdims=True)
        return (x - m) * jax.lax.rsqrt(((x - m) ** 2).mean(-1, keepdims=True) + 1e-5)

    CASES = {
        "layernorm_2x128x256": (layernorm, (2, 128, 256)),
        "layernorm_16x256x512": (layernorm, (16, 256, 512)),
        "softmax_sum_2x8x128x128": (lambda a: jnp.exp(a).sum(-1), (2, 8, 128, 128)),
        "sum_rows_256x16384": (lambda a: a.sum(-1), (256, 16384)),
        "sum_single_row_65536": (lambda a: a.sum(), (65536,)),
    }
    fn, shape = CASES[sys.argv[1]]
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    with jax.default_device(jax.devices("mps")[0]):
        np.asarray(jax.jit(fn)(jnp.asarray(x)))
    print("OK")
    """
)


def _native_steps(case):
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps", "JAX_METALLIB_TRACE_PLAN": "1"}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, case], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
    counts = []
    for line in proc.stderr.splitlines():
        if line.startswith("JAX_METALLIB_PLAN_JSON:"):
            event = json.loads(line.split(":", 1)[1])
            if event.get("event") == "steps":
                counts.append(event["native_steps"])
    assert counts, "no plan trace found"
    return max(counts)


@pytest.mark.parametrize("case", ["layernorm_2x128x256", "layernorm_16x256x512", "softmax_sum_2x8x128x128"])
def test_short_row_reduce_stays_in_graph(case):
    assert _native_steps(case) == 0


@pytest.mark.parametrize("case", ["sum_rows_256x16384", "sum_single_row_65536"])
def test_long_row_and_single_row_reduce_stay_native(case):
    assert _native_steps(case) >= 1


def _special(shape):
    x = np.random.default_rng(1).standard_normal(shape).astype(np.float32)
    x[0, 3] = np.nan
    x[1, 5] = np.inf
    x[2, 7] = -np.inf
    x[3, :] = -0.0
    return x


@pytest.mark.parametrize("shape", [(256, 512), (64, 1024), (32, 4096)])
def test_row_sum_values_match_cpu(shape):
    x = _special(shape)
    fn = lambda a: a.sum(-1)  # noqa: E731
    expected = np.asarray(jax.jit(fn, backend="cpu")(x))
    with jax.default_device(jax.devices("mps")[0]):
        actual = np.asarray(jax.jit(fn)(jnp.asarray(x)))
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4, equal_nan=True)
