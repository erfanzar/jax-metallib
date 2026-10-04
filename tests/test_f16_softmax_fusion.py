# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""f16 jax.nn.softmax must fuse into one native kernel and match XLA:CPU rounding.

JAX lowers f16 softmax as max -> sub -> exp (f16) -> convert f32 -> sum ->
convert f16 -> divide (f16). The matcher only knew the same-dtype chain, so f16
ran as graph / native reduce / graph (0.55x vs jax-mps at [4096,4096]). The
fused kernel must keep XLA's rounding points (f16 sub, f16 exp, f32 sum rounded
to f16, f16 divide) and NaN rules; rows over 8192 columns stay unfused.
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


def _special(shape):
    x = (np.random.default_rng(5).standard_normal(shape) * 4).astype(np.float32)
    x[0, 3] = np.nan
    x[1, :] = -np.inf
    x[2, 5] = np.inf
    x[3, 0] = np.nan
    x[3, 1:] = -np.inf
    x[4, :] = 80.0
    x[5, 7] = -np.inf
    x[6, :] = -60000.0
    x[6, 0] = 60000.0
    x[7, :] = 0.0
    return x.astype(np.float16)


_CASES = {
    "f16_4096x4096": lambda: np.random.default_rng(0).standard_normal((4096, 4096)).astype(np.float16),
    "f16_special_64x1000": lambda: _special((64, 1000)),
    "f16_special_16x8192": lambda: _special((16, 8192)),
    "f16_rank3_4x128x300": lambda: (np.random.default_rng(1).standard_normal((4, 128, 300)) * 3).astype(np.float16),
    "f16_16x10000_unfused": lambda: np.random.default_rng(2).standard_normal((16, 10000)).astype(np.float16),
}


def _softmax(a):
    return jax.nn.softmax(a, axis=-1)


def _softmax_and_exp(a):
    # exp gets an extra external user, so the fusion must decline.
    return jax.nn.softmax(a, axis=-1), jnp.exp(a - a.max(-1, keepdims=True))


def _ordered(h):
    i = h.view(np.int16).astype(np.int32)
    return np.where(i < 0, -(i & 0x7FFF), i)


def _check_f16(actual, expected, max_ulp=1):
    assert actual.dtype == expected.dtype == np.float16 and actual.shape == expected.shape
    np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected))
    ok = ~np.isnan(expected)
    ulp = np.abs(_ordered(actual[ok]) - _ordered(expected[ok]))
    assert ulp.size == 0 or int(ulp.max()) <= max_ulp, f"max ulp {int(ulp.max())}, >1 ulp: {int((ulp > 1).sum())}"


def _run(fn, x):
    expected = jax.jit(fn, backend="cpu")(x)
    with jax.default_device(jax.devices("mps")[0]):
        actual = jax.jit(fn)(jnp.asarray(x))
    return jax.tree_util.tree_map(np.asarray, actual), jax.tree_util.tree_map(np.asarray, expected)


# Unfused f16 softmax runs on MPSGraph, whose fused sub->exp skips the f16
# rounding of x - max (measured up to 9 ulp vs XLA:CPU; pre-existing). The
# fused native kernel is bit-exact to within 1 ulp.
_GRAPH_PATH_MAX_ULP = 10


@pytest.mark.parametrize("case", list(_CASES))
def test_f16_softmax_matches_cpu(case):
    actual, expected = _run(_softmax, _CASES[case]())
    _check_f16(actual, expected, max_ulp=_GRAPH_PATH_MAX_ULP if case.endswith("_unfused") else 1)


def test_f16_softmax_with_extra_exp_user_matches_cpu():
    # exp has an external user, so the fusion declines and the graph path runs.
    x = _special((64, 1000))
    (sa, ea), (se, ee) = _run(_softmax_and_exp, x)
    _check_f16(sa, se, max_ulp=_GRAPH_PATH_MAX_ULP)
    _check_f16(ea, ee, max_ulp=_GRAPH_PATH_MAX_ULP)


_CHILD = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    shape = tuple(int(v) for v in sys.argv[1].split("x"))
    with jax.default_device(jax.devices("mps")[0]):
        jax.jit(lambda a: jax.nn.softmax(a, axis=-1))(jnp.ones(shape, jnp.float16)).block_until_ready()
    print("OK")
    """
)


def _steps(shape, extra_env=None):
    env = {**os.environ, "JAX_PLATFORMS": "cpu,mps", "JAX_METALLIB_TRACE_PLAN": "1", **(extra_env or {})}
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD, shape], capture_output=True, text=True, timeout=120, env=env, check=False
    )
    assert proc.returncode == 0 and "OK" in proc.stdout, proc.stderr[-2000:]
    steps = []
    for line in proc.stderr.splitlines():
        if line.startswith("JAX_METALLIB_PLAN_JSON:"):
            event = json.loads(line.split(":", 1)[1])
            if event.get("event") == "steps" and event["num_steps"] > 0:
                steps.append(event)
    assert steps, "no plan trace found"
    return steps[-1]


def test_f16_softmax_fuses_into_one_native_step():
    s = _steps("4096x4096")
    assert s["native_steps"] == 1 and s["graph_steps"] <= 1, s


def test_f16_softmax_fusion_switch_and_long_rows_stay_unfused():
    s = _steps("4096x4096", {"JAX_METALLIB_DISABLE_F16_WIDENED_SOFTMAX_FUSION": "1"})
    assert s["graph_steps"] == 2, s
    s = _steps("16x10000")
    assert s["graph_steps"] >= 2, s
