# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Complex stablehlo.pad CPU/MPS regressions, isolated from MPSGraph aborts."""

import os
import subprocess
import sys

import pytest

# Complex interior padding previously reached MPSGraph scatterND, which rejects
# complex tensors with a process-aborting assertion; positive edge pads of a
# computed complex tensor silently returned zeros. Run each case in a child so a
# regression reports a test failure instead of killing the suite.
SCRIPT = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

case, jit = sys.argv[1], sys.argv[2] == "jit"
CONFIGS = {
    "edge_hi": (((0, 0, 0), (0, 0, 0), (0, 1, 0)), 0j),
    "edge_lo": (((1, 0, 0), (0, 0, 0), (0, 0, 0)), 0j),
    "edge_neg": (((0, 0, 0), (0, 0, 0), (0, -1, 0)), 0j),
    "interior": (((0, 0, 0), (0, 0, 1), (0, 0, 0)), 0j),
    "mixed": (((1, 0, 0), (-1, 2, 2), (0, 1, 1)), 0j),
    "value": (((0, 1, 0), (1, 0, 1), (2, 0, 0)), 1.5 - 2j),
}


def spectrum(x):
    return lax.fft(x, lax.FftType.RFFT, (4, 8))


config, fill = CONFIGS[case.removeprefix("grad_")]
x = np.random.default_rng(0).standard_normal((2, 4, 8)).astype(np.float32)

if case.startswith("grad_"):
    weights = np.random.default_rng(1).standard_normal(64).astype(np.float32)

    def fn(v):
        y = lax.pad(spectrum(v), jnp.asarray(fill, jnp.complex64), config)
        return jnp.sum(jnp.abs(y).ravel() * jnp.resize(jnp.asarray(weights), y.size))

    fn = jax.grad(fn)
else:

    def fn(v):
        return lax.pad(spectrum(v), jnp.asarray(fill, jnp.complex64), config)

if jit:
    fn = jax.jit(fn)
with jax.default_device(jax.devices("cpu")[0]):
    expected = np.asarray(fn(jnp.asarray(x)))
with jax.default_device(jax.devices("mps")[0]):
    actual = np.asarray(fn(jnp.asarray(x)))
assert actual.shape == expected.shape, (actual.shape, expected.shape)
assert np.abs(expected).max() > 0
np.testing.assert_allclose(actual, expected, atol=1e-4, rtol=1e-4)
"""

VALUE_CASES = ["edge_hi", "edge_lo", "edge_neg", "interior", "mixed", "value"]
GRAD_CASES = ["grad_edge_hi", "grad_interior", "grad_mixed"]


@pytest.mark.parametrize("mode", ["jit", "eager"])
@pytest.mark.parametrize("case", VALUE_CASES + GRAD_CASES)
def test_complex_pad_matches_cpu(case, mode):
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", SCRIPT, case, mode],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
