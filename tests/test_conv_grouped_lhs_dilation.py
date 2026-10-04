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
"""Grouped convolution with lhs_dilation (transposed conv / input gradient of a
strided grouped conv) matches CPU.

The data-gradient lowering swapped the kernel's O and I axes across all groups,
so MPSGraph asserted 'Source and weight input channels mismatch' and aborted
the process. Each case runs in a subprocess so an abort fails one test only.
"""

import os
import subprocess
import sys
import textwrap

import pytest

_SCRIPT = textwrap.dedent(
    """
    import sys
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax import lax

    case = sys.argv[1]
    cpu = jax.devices("cpu")[0]
    mps = jax.devices("mps")[0]
    rng = np.random.default_rng(0)

    def conv(lhs_dil, rhs_dil, groups, pad, dn=("NHWC", "HWIO", "NHWC")):
        def f(x, k):
            return lax.conv_general_dilated(
                x, k, (1, 1), pad, lhs_dilation=lhs_dil, rhs_dilation=rhs_dil,
                feature_group_count=groups, dimension_numbers=dn)
        return f

    if case == "g2_lhs_dil_pad":
        f = conv((2, 1), (1, 1), 2, ((1, 2), (0, 1)))
        args = (rng.standard_normal((2, 6, 7, 4)), rng.standard_normal((3, 2, 2, 6)))
    elif case == "g2_lhs_rhs_dil_pad":
        f = conv((2, 1), (1, 2), 2, ((1, 2), (0, 1)))
        args = (rng.standard_normal((2, 6, 7, 4)), rng.standard_normal((3, 2, 2, 6)))
    elif case == "g4_lhs_dil_2d":
        f = conv((2, 3), (1, 1), 4, ((2, 2), (1, 1)))
        args = (rng.standard_normal((1, 5, 4, 8)), rng.standard_normal((3, 3, 2, 12)))
    elif case == "depthwise_lhs_dil":
        f = conv((2, 2), (1, 1), 4, ((1, 1), (1, 1)))
        args = (rng.standard_normal((2, 5, 5, 4)), rng.standard_normal((3, 3, 1, 4)))
    elif case == "g2_nchw_oihw":
        f = conv((2, 2), (1, 1), 2, ((1, 1), (2, 0)), dn=("NCHW", "OIHW", "NCHW"))
        args = (rng.standard_normal((2, 6, 5, 4)), rng.standard_normal((4, 3, 2, 3)))
    elif case == "grad_input_strided_grouped":
        k = jnp.asarray(rng.standard_normal((3, 3, 2, 6)).astype(np.float32))
        def f(x, w):
            y = lax.conv_general_dilated(x, w, (2, 2), "SAME", feature_group_count=2,
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
            return jnp.sum(jnp.sin(y))
        f = jax.grad(f, argnums=(0, 1))
        args = (rng.standard_normal((2, 9, 8, 4)), np.asarray(k))
    elif case == "grad_input_strided_depthwise":
        def f(x, w):
            y = lax.conv_general_dilated(x, w, (2, 1), ((1, 1), (1, 1)), feature_group_count=4,
                                         dimension_numbers=("NHWC", "HWIO", "NHWC"))
            return jnp.sum(y * y)
        f = jax.grad(f, argnums=(0, 1))
        args = (rng.standard_normal((2, 8, 6, 4)), rng.standard_normal((3, 3, 1, 4)))
    elif case == "g1_lhs_dil_control":
        f = conv((2, 1), (1, 2), 1, ((1, 2), (0, 1)))
        args = (rng.standard_normal((2, 6, 7, 4)), rng.standard_normal((3, 2, 4, 6)))
    else:
        raise SystemExit("unknown case " + case)

    args = [np.asarray(a, np.float32) for a in args]
    want = jax.jit(f)(*[jax.device_put(a, cpu) for a in args])
    got = jax.jit(f)(*[jax.device_put(a, mps) for a in args])
    for w, g in zip(jax.tree.leaves(want), jax.tree.leaves(got)):
        np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=2e-4, atol=2e-4)
    print("CASE_OK")
    """
)

_CASES = [
    "g2_lhs_dil_pad",
    "g2_lhs_rhs_dil_pad",
    "g4_lhs_dil_2d",
    "depthwise_lhs_dil",
    "g2_nchw_oihw",
    "grad_input_strided_grouped",
    "grad_input_strided_depthwise",
    "g1_lhs_dil_control",
]


@pytest.mark.parametrize("case", _CASES)
def test_grouped_lhs_dilated_conv(case):
    try:
        import jax

        jax.devices("mps")
    except RuntimeError:
        pytest.skip("mps backend not available")
    # Override, not setdefault: earlier tests in the suite export JAX_PLATFORMS=mps.
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    proc = subprocess.run(
        [sys.executable, "-c", _SCRIPT, case], capture_output=True, text=True, env=env, timeout=300, check=False
    )
    tail = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode == 0 and "CASE_OK" in proc.stdout, f"rc={proc.returncode}\n{tail}"
