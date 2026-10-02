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
"""Integer and mixed-precision convolutions match CPU.

MPSGraph convolution accepts only FP32/FP16 and aborted the process on integer
operands; integer convs are now lowered exactly in int32 (wrapping like XLA).
f16/bf16 convs with preferred_element_type=f32 returned the f16 bytes read as
f32 (errors up to 2.9e6, no error raised). Each case runs in a subprocess so
an abort fails one test only.
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
    NHWC = ("NHWC", "HWIO", "NHWC")

    def ints(shape, lo, hi, dt):
        return rng.integers(lo, hi, shape).astype(dt)

    def conv(stride=(1, 1), pad="SAME", lhs=(1, 1), rhs=(1, 1), groups=1, dn=NHWC, pref=None):
        def f(x, k):
            return lax.conv_general_dilated(x, k, stride, pad, lhs_dilation=lhs, rhs_dilation=rhs,
                                            feature_group_count=groups, dimension_numbers=dn,
                                            preferred_element_type=pref)
        return f

    expect_error = False
    tol = 0.0
    if case == "int8_wraps":
        f = conv()
        args = (ints((2, 6, 5, 8), -128, 128, np.int8), ints((3, 3, 8, 4), -128, 128, np.int8))
    elif case == "int32_wraps":
        f = conv(pad="VALID")
        args = (ints((1, 5, 5, 3), -2**30, 2**30, np.int32), ints((2, 2, 3, 2), -2**30, 2**30, np.int32))
    elif case == "uint32":
        f = conv(stride=(2, 1))
        args = (ints((2, 7, 6, 3), 0, 2**32, np.uint32), ints((3, 2, 3, 5), 0, 2**32, np.uint32))
    elif case == "int_negative_padding":
        f = conv(pad=((-1, 2), (1, -2)))
        args = (ints((2, 8, 7, 4), -9, 10, np.int16), ints((3, 2, 4, 6), -5, 6, np.int16))
    elif case == "int_stride_dilations_groups":
        f = conv(stride=(2, 3), pad=((1, 0), (2, 1)), lhs=(2, 1), rhs=(1, 2), groups=2)
        args = (ints((2, 7, 9, 6), -9, 10, np.int32), ints((3, 2, 3, 8), -5, 6, np.int32))
    elif case == "int_nchw_oihw_1d":
        f = conv(stride=(2,), pad=((2, 1),), lhs=(1,), rhs=(2,), dn=("NCH", "OIH", "NCH"))
        args = (ints((3, 4, 11), -9, 10, np.int8), ints((5, 4, 3), -5, 6, np.int8))
    elif case == "int_channel_chunking":
        f = conv(pad="VALID", pref=jnp.int32)
        args = (ints((1, 66, 66, 128), -3, 4, np.int8), ints((3, 3, 128, 64), -3, 4, np.int8))
    elif case == "uint8_pref_int32":
        f = conv(pref=jnp.int32)
        args = (ints((2, 6, 6, 4), 0, 256, np.uint8), ints((2, 2, 4, 3), 0, 256, np.uint8))
    elif case == "f16_pref_f32":
        f = conv(stride=(2, 1), groups=2, pref=jnp.float32)
        args = (rng.standard_normal((2, 8, 7, 4)).astype(np.float16),
                rng.standard_normal((3, 3, 2, 6)).astype(np.float16))
        tol = 1e-4
    elif case == "bf16_pref_f32":
        f = conv(lhs=(2, 1), pad=((1, 1), (0, 2)), pref=jnp.float32)
        args = (rng.standard_normal((2, 6, 5, 4)).astype(jnp.bfloat16),
                rng.standard_normal((3, 2, 4, 6)).astype(jnp.bfloat16))
        tol = 1e-4
    elif case == "int_3d_errors":
        f = conv(stride=(1, 1, 1), pad="SAME", lhs=(1, 1, 1), rhs=(1, 1, 1), dn=("NDHWC", "DHWIO", "NDHWC"))
        args = (ints((1, 4, 4, 4, 2), -5, 6, np.int32), ints((2, 2, 2, 2, 3), -5, 6, np.int32))
        expect_error = True
    else:
        raise SystemExit("unknown case " + case)

    if expect_error:
        try:
            np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
        except Exception as e:
            assert "integer convolution" in str(e), str(e)
            print("CASE_OK")
            raise SystemExit(0)
        raise SystemExit("expected an error")

    want = np.asarray(jax.jit(f)(*[jax.device_put(a, cpu) for a in args]))
    got = np.asarray(jax.jit(f)(*[jax.device_put(a, mps) for a in args]))
    assert got.dtype == want.dtype, (got.dtype, want.dtype)
    if tol:
        np.testing.assert_allclose(got.astype(np.float32), want.astype(np.float32), rtol=tol, atol=tol)
    else:
        np.testing.assert_array_equal(got, want)
    print("CASE_OK")
    """
)

_CASES = [
    "int8_wraps",
    "int32_wraps",
    "uint32",
    "int_negative_padding",
    "int_stride_dilations_groups",
    "int_nchw_oihw_1d",
    "int_channel_chunking",
    "uint8_pref_int32",
    "f16_pref_f32",
    "bf16_pref_f32",
    "int_3d_errors",
]


@pytest.mark.parametrize("case", _CASES)
def test_conv_dtype(case):
    env = dict(os.environ, JAX_PLATFORMS="cpu,mps")
    r = subprocess.run([sys.executable, "-c", _SCRIPT, case], capture_output=True, text=True, env=env, timeout=300)
    if "mps" in r.stderr and "Unknown backend" in r.stderr:
        pytest.skip("mps backend not available")
    assert r.returncode == 0 and "CASE_OK" in r.stdout, (
        f"rc={r.returncode}\nstdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-3000:]}"
    )
