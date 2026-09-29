# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Scalar bitcast regressions isolated from native MPSGraph compiler aborts."""

import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("target_dtype", ["uint16", "uint8", "uint32"])
@pytest.mark.parametrize("input_shape", [(), (2, 3)], ids=["scalar", "matrix"])
def test_bitcast_matches_cpu(target_dtype, input_shape):
    # A malformed MPSGraph reshape aborts the process rather than raising Python.
    # Keep each compile in a child so a regression reports a test failure.
    script = """
import sys
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

convert = jax.jit(lambda x: lax.bitcast_convert_type(x, getattr(jnp, sys.argv[1])))
shape = tuple(map(int, sys.argv[2:]))
source = np.asarray(1.25, dtype=np.float32) if not shape else (np.arange(6, dtype=np.float32) + 1.25).reshape(shape)
with jax.default_device(jax.devices('cpu')[0]):
    expected = np.asarray(convert(jnp.asarray(source)))
with jax.default_device(jax.devices('mps')[0]):
    actual = np.asarray(convert(jnp.asarray(source)))
assert actual.shape == expected.shape, (actual.shape, expected.shape)
np.testing.assert_array_equal(actual, expected)
"""
    env = os.environ.copy()
    env["JAX_PLATFORMS"] = "cpu,mps"
    result = subprocess.run(
        [sys.executable, "-c", script, target_dtype, *map(str, input_shape)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:] or result.stdout
