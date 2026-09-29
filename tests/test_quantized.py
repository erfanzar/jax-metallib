# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for quantization: MX (MXFP4, MXFP8, NVFP4) and affine (INT4, INT8)."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.silicon.quantized import (
    _fp4_e2m1_dequantize,
    _fp4_e2m1_quantize,
    _fp8_e4m3_dequantize,
    _fp8_e4m3_quantize,
    _fp8_e8m0_dequantize,
    _fp8_e8m0_quantize,
    affine_dequantize,
    affine_quantize,
    affine_quantized_matmul,
    mx_dequantize,
    mx_quantize,
    quantized_matmul,
)

# ──────────────────────────────────────────────────────────────────
# FP4 E2M1 type tests
# ──────────────────────────────────────────────────────────────────


class TestFP4E2M1:
    """Test FP4 E2M1 quantize/dequantize."""

    def test_representable_values(self):
        """FP4 E2M1 should roundtrip its 8 positive representable values."""
        values = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
        bits = _fp4_e2m1_quantize(values)
        recovered = _fp4_e2m1_dequantize(bits)
        np.testing.assert_array_equal(recovered, values)

    def test_negative_values(self):
        """FP4 E2M1 should handle negative values correctly."""
        values = np.array([-0.5, -1.0, -3.0, -6.0], dtype=np.float32)
        bits = _fp4_e2m1_quantize(values)
        recovered = _fp4_e2m1_dequantize(bits)
        np.testing.assert_array_equal(recovered, values)

    def test_signed_zero(self):
        bits = _fp4_e2m1_quantize(np.array([0.0, -0.0], dtype=np.float32))
        np.testing.assert_array_equal(bits, [0, 0x8])

    def test_rounding(self):
        """Values between representable points should round correctly."""
        # 0.3 → rounds to 0.5 (nearest to 0.25 boundary)
        bits = _fp4_e2m1_quantize(np.array([0.3], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 0.5

        # 1.6 → rounds to 1.5
        bits = _fp4_e2m1_quantize(np.array([1.6], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 1.5

    @pytest.mark.parametrize(
        "midpoint, even_code",
        [(0.25, 0), (0.75, 2), (1.25, 2), (1.75, 4), (2.5, 4), (3.5, 6), (5.0, 6)],
    )
    def test_midpoints_round_to_even(self, midpoint, even_code):
        bits = _fp4_e2m1_quantize(np.array([midpoint, -midpoint], dtype=np.float32))
        np.testing.assert_array_equal(bits, [even_code, even_code | 0x8])

    def test_saturation(self):
        """Values beyond 6.0 should saturate to 6.0."""
        bits = _fp4_e2m1_quantize(np.array([100.0], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 6.0


# ──────────────────────────────────────────────────────────────────
# FP8 E4M3 type tests
# ──────────────────────────────────────────────────────────────────


class TestFP8E4M3:
    """Test finite E4M3 encodings at the upper exponent boundary."""

    def test_signed_zero(self):
        bits = _fp8_e4m3_quantize(np.array([0.0, -0.0], dtype=np.float32))
        np.testing.assert_array_equal(bits, [0, 0x80])
        assert np.signbit(_fp8_e4m3_dequantize(bits)[1])

    def test_reserved_nan_codes(self):
        values = _fp8_e4m3_dequantize(np.array([0x7E, 0x7F, 0xFE, 0xFF], dtype=np.uint8))
        np.testing.assert_array_equal(values[[0, 2]], [448.0, -448.0])
        assert np.isnan(values[[1, 3]]).all()

    @pytest.mark.parametrize("sign", [1, -1])
    def test_upper_exponent_representable_values(self, sign):
        # Bias 7: exponent 15 represents 2**8 * (1 + mantissa / 8).
        # Mantissas 0..6 are finite; only mantissa 7 is reserved for NaN.
        values = sign * np.array([240, 256, 288, 320, 352, 384, 416, 448], dtype=np.float32)
        expected = np.arange(0x77, 0x7F, dtype=np.uint8) | (0x80 if sign < 0 else 0)

        bits = _fp8_e4m3_quantize(values)

        np.testing.assert_array_equal(bits, expected)
        np.testing.assert_array_equal(_fp8_e4m3_dequantize(bits), values)

    @pytest.mark.parametrize("sign", [1, -1])
    def test_saturation(self, sign):
        values = sign * np.array([448, 480, 512, np.finfo(np.float32).max, np.inf], dtype=np.float32)

        bits = _fp8_e4m3_quantize(values)

        np.testing.assert_array_equal(bits, 0xFE if sign < 0 else 0x7E)
        np.testing.assert_array_equal(_fp8_e4m3_dequantize(bits), sign * 448.0)

    @pytest.mark.parametrize("sign", [1, -1])
    def test_subnormals_and_nearest_even_rounding(self, sign):
        values = sign * np.array([1 / 512, 2 / 512, 3 / 512, 280, 304, 432], dtype=np.float32)
        # 280 rounds to 288; 304 is halfway between 288 and 320 (even code 0x7A).
        # 432 is halfway between 416 and 448 (even code 0x7E).
        expected = np.array([1, 2, 3, 0x79, 0x7A, 0x7E], dtype=np.uint8)
        expected |= 0x80 if sign < 0 else 0

        np.testing.assert_array_equal(_fp8_e4m3_quantize(values), expected)


# ──────────────────────────────────────────────────────────────────
# FP8 E8M0 scale tests
# ──────────────────────────────────────────────────────────────────


class TestFP8E8M0:
    """Test FP8 E8M0 (power-of-2 scale) quantize/dequantize."""

    def test_powers_of_two(self):
        """E8M0 should roundtrip exact powers of 2."""
        for exp in [-10, -1, 0, 1, 5, 10]:
            val = np.array([2.0**exp], dtype=np.float32)
            bits = _fp8_e8m0_quantize(val)
            recovered = _fp8_e8m0_dequantize(bits)
            np.testing.assert_allclose(recovered, val, rtol=0)

    def test_non_powers_round(self):
        """Non-power-of-2 values round to nearest power of 2."""
        val = np.array([3.0], dtype=np.float32)  # between 2 and 4
        bits = _fp8_e8m0_quantize(val)
        recovered = _fp8_e8m0_dequantize(bits)
        # log2(3) ≈ 1.58, rounds to 2, so 2^2 = 4.0
        assert recovered[0] == 4.0

    def test_reserved_nan_scale_code(self):
        """E8M0 0xFF denotes NaN, not positive infinity."""
        values = _fp8_e8m0_dequantize(np.array([0, 127, 254, 255], dtype=np.uint8))
        np.testing.assert_array_equal(values[:3], np.array([2.0**-127, 1.0, 2.0**127], dtype=np.float32))
        assert np.isnan(values[3])

    def test_nonfinite_scale_inputs_do_not_wrap_to_small_finite_codes(self):
        bits = _fp8_e8m0_quantize(np.array([np.nan, np.inf], dtype=np.float32))
        np.testing.assert_array_equal(bits, [0xFF, 0xFF])


# ──────────────────────────────────────────────────────────────────
# MX Quantize/Dequantize roundtrip tests
# ──────────────────────────────────────────────────────────────────


class TestMxQuantize:
    """Test full quantize → dequantize roundtrip for all modes."""

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_roundtrip_small(self, mode):
        """Quantize → dequantize should approximately preserve values."""
        rng = np.random.default_rng(42)
        K = 64  # must be divisible by group_size
        w = jnp.array(rng.standard_normal((4, K)).astype(np.float32) * 0.1)

        packed, scales = mx_quantize(w, mode=mode)
        w_restored = mx_dequantize(packed, scales, mode=mode, shape=w.shape)

        # FP4 has very coarse quantization; FP8 still has limited precision
        if "fp4" in mode:
            atol = 0.15
        else:
            atol = 0.1  # E4M3 has 3 mantissa bits → ~12.5% relative error

        np.testing.assert_allclose(np.array(w_restored), np.array(w), atol=atol, rtol=0.8)

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_shapes(self, mode):
        """Output shapes should be correct."""
        from jax_plugins.silicon.quantized import _MX_MODES

        cfg = _MX_MODES[mode]
        K = 128
        w = jnp.ones((16, K), dtype=jnp.float32)
        packed, scales = mx_quantize(w, mode=mode)

        pack_factor = 2 if cfg["bits"] == 4 else 1
        assert packed.shape == (16, K // pack_factor)
        assert scales.shape == (16, K // cfg["group_size"])
        assert packed.dtype == jnp.uint8
        assert scales.dtype == jnp.uint8

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
    def test_nan_weight_group_keeps_nan_scale(self, mode):
        w = np.ones((1, 32), dtype=np.float32)
        w[0, 0] = np.nan
        packed, scales = mx_quantize(w, mode=mode)
        np.testing.assert_array_equal(np.asarray(scales), [[0xFF]])
        restored = mx_dequantize(packed, scales, mode=mode, shape=w.shape)
        assert np.isnan(np.asarray(restored)).all()

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_zero_weights(self, mode):
        """Zero weights should quantize and dequantize to zero."""
        K = 64
        w = jnp.zeros((2, K), dtype=jnp.float32)
        packed, scales = mx_quantize(w, mode=mode)
        w_restored = mx_dequantize(packed, scales, mode=mode, shape=w.shape)
        np.testing.assert_array_equal(np.array(w_restored), 0.0)


# ──────────────────────────────────────────────────────────────────
# Quantized matmul tests
# ──────────────────────────────────────────────────────────────────


class TestQuantizedMatmul:
    """Test quantized matmul correctness."""

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_basic_matmul(self, mode):
        """quantized_matmul should approximate standard matmul."""
        rng = np.random.default_rng(42)
        M, N, K = 4, 8, 64
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        # Reference: standard matmul
        ref = x @ w.T

        # Quantized matmul
        w_packed, scales = mx_quantize(w, mode=mode)
        out = quantized_matmul(x, w_packed, scales, mode=mode, K=K)

        # FP4 has significant quantization error; FP8 still has limited precision
        if "fp4" in mode:
            rtol = 0.5
            atol = 0.1
        else:
            rtol = 0.5
            atol = 0.06

        np.testing.assert_allclose(np.array(out), np.array(ref), rtol=rtol, atol=atol)

    def test_matmul_shapes(self):
        """Output shape should be [M, N]."""
        rng = np.random.default_rng(42)
        M, N, K = 16, 32, 128
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32))
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32))

        w_packed, scales = mx_quantize(w, mode="mxfp4")
        out = quantized_matmul(x, w_packed, scales, mode="mxfp4", K=K)
        assert out.shape == (M, N)


class TestGPUPath:
    """Test that GPU Metal kernel dispatch works."""

    def test_gpu_available(self):
        """GPU path should be available when PyObjC Metal is installed."""
        from jax_plugins.silicon.quantized import _gpu_available

        assert _gpu_available(), "GPU path not available — install pyobjc-framework-Metal"

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_gpu_matmul_correctness(self, mode):
        """GPU quantized matmul should match CPU dequant + matmul."""
        from jax_plugins.silicon.quantized import _quantized_matmul_gpu

        rng = np.random.default_rng(42)
        M, N, K = 8, 16, 64
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        w_packed, scales = mx_quantize(w, mode=mode)

        # GPU path
        gpu_out = _quantized_matmul_gpu(x, w_packed, scales, mode=mode, K=K, out_dtype=jnp.float32)

        # CPU reference: dequantize + matmul
        w_dequant = mx_dequantize(w_packed, scales, mode=mode, shape=(N, K))
        cpu_ref = x @ np.array(w_dequant).T

        if "fp4" in mode:
            atol = 0.15
        else:
            atol = 0.1

        np.testing.assert_allclose(np.array(gpu_out), cpu_ref, atol=atol, rtol=0.5)

    def test_gpu_fp8_nan_weight_propagates(self):
        """Reserved E4M3FN NaN bits in an FP8 weight must not decode as 480."""
        from jax_plugins.silicon.quantized import _quantized_matmul_gpu

        packed = jnp.array([[0x7F, *([0] * 31)]], dtype=jnp.uint8)
        scales = jnp.array([[127]], dtype=jnp.uint8)  # E8M0 scale 1.0
        out = _quantized_matmul_gpu(
            jnp.ones((1, 32), dtype=jnp.float32), packed, scales, mode="mxfp8", K=32, out_dtype=jnp.float32
        )
        assert np.isnan(np.asarray(out)[0, 0])

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8"])
    def test_gpu_e8m0_nan_scale_propagates(self, mode):
        """The reserved E8M0 scale code must produce NaN in MX GEMM."""
        from jax_plugins.silicon.quantized import _quantized_matmul_gpu

        packed = jnp.array([[0x11] * (16 if mode == "mxfp4" else 32)], dtype=jnp.uint8)
        scales = jnp.array([[0xFF]], dtype=jnp.uint8)
        out = _quantized_matmul_gpu(
            jnp.ones((1, 32), dtype=jnp.float32), packed, scales, mode=mode, K=32, out_dtype=jnp.float32
        )
        assert np.isnan(np.asarray(out)[0, 0])

    def test_gpu_nvfp4_nan_scale_propagates(self):
        """Reserved E4M3FN NaN bits in an NVFP4 scale must propagate."""
        from jax_plugins.silicon.quantized import _quantized_matmul_gpu

        packed = jnp.array([[0x11] * 8], dtype=jnp.uint8)  # E2M1 weight 0.5
        scales = jnp.array([[0x7F]], dtype=jnp.uint8)
        out = _quantized_matmul_gpu(
            jnp.ones((1, 16), dtype=jnp.float32), packed, scales, mode="nvfp4", K=16, out_dtype=jnp.float32
        )
        assert np.isnan(np.asarray(out)[0, 0])

    def test_gpu_larger_matmul(self):
        """Larger matmul should work correctly on GPU."""
        rng = np.random.default_rng(42)
        M, N, K = 64, 128, 256
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        w_packed, scales = mx_quantize(w, mode="mxfp4")
        out = quantized_matmul(x, w_packed, scales, mode="mxfp4", K=K)
        assert out.shape == (M, N)

        # Basic sanity: result shouldn't be all zeros or NaN
        assert not jnp.any(jnp.isnan(out))
        assert jnp.any(out != 0)


class TestAllModesDtypes:
    """Test all 6 combinations: 3 modes × f32/f16 on GPU."""

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_gpu_qgemm_all_variants(self, mode, dtype):
        """Each (mode, dtype) variant should produce correct results on GPU."""
        from jax_plugins.silicon.quantized import _quantized_matmul_gpu

        rng = np.random.default_rng(42)
        M, N, K = 8, 16, 64
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1, dtype=dtype)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        w_packed, scales = mx_quantize(w, mode=mode)

        # GPU path
        gpu_out = _quantized_matmul_gpu(x, w_packed, scales, mode=mode, K=K, out_dtype=dtype)

        # CPU reference
        w_dequant = mx_dequantize(w_packed, scales, mode=mode, shape=(N, K))
        x_f32 = np.array(x, dtype=np.float32)
        cpu_ref = x_f32 @ np.array(w_dequant, dtype=np.float32).T

        assert gpu_out.shape == (M, N), f"Wrong shape: {gpu_out.shape}"
        assert gpu_out.dtype == dtype, f"Wrong dtype: {gpu_out.dtype}"
        assert not jnp.any(jnp.isnan(gpu_out)), "NaN in output"

        # Tolerance depends on format + precision
        if "fp4" in mode:
            atol = 0.2 if dtype == jnp.float16 else 0.15
        else:
            atol = 0.15 if dtype == jnp.float16 else 0.1

        np.testing.assert_allclose(
            np.array(gpu_out, dtype=np.float32),
            cpu_ref,
            atol=atol,
            rtol=0.5,
            err_msg=f"Failed for mode={mode}, dtype={dtype}",
        )

    @pytest.mark.parametrize("mode", ["mxfp4", "mxfp8", "nvfp4"])
    def test_gpu_qgemm_f16_output(self, mode):
        """F16 inputs should produce f16 outputs."""
        rng = np.random.default_rng(42)
        M, N, K = 4, 8, 64
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1, dtype=jnp.float16)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        w_packed, scales = mx_quantize(w, mode=mode)
        out = quantized_matmul(x, w_packed, scales, mode=mode, K=K)

        assert out.dtype == jnp.float16, f"Expected float16, got {out.dtype}"
        assert out.shape == (M, N)


# ──────────────────────────────────────────────────────────────────
# Affine quantization tests
# ──────────────────────────────────────────────────────────────────


class TestAffineQuantize:
    """Test affine (INT4/INT8) quantize → dequantize."""

    @pytest.mark.parametrize("bits", [4, 8])
    def test_roundtrip(self, bits):
        """Affine quantize → dequantize roundtrip."""
        rng = np.random.default_rng(42)
        K = 128
        w = jnp.array(rng.standard_normal((8, K)).astype(np.float32) * 0.1)

        packed, scales, biases = affine_quantize(w, bits=bits, group_size=64)
        w_restored = affine_dequantize(packed, scales, biases, bits=bits, group_size=64, shape=w.shape)

        # INT4 has 16 levels → coarser; INT8 has 256 levels → much finer
        atol = 0.02 if bits == 8 else 0.05
        np.testing.assert_allclose(np.array(w_restored), np.array(w), atol=atol, rtol=0.3)

    @pytest.mark.parametrize("bits", [4, 8])
    def test_shapes(self, bits):
        """Packed shapes should be correct."""
        K = 128
        w = jnp.ones((16, K), dtype=jnp.float32)
        packed, scales, biases = affine_quantize(w, bits=bits, group_size=64)

        if bits == 4:
            assert packed.shape == (16, K // 8)  # uint32
            assert packed.dtype == jnp.uint32
        else:
            assert packed.shape == (16, K)
            assert packed.dtype == jnp.uint8

        assert scales.shape == (16, 2)  # K/group_size = 128/64 = 2
        assert biases.shape == (16, 2)

    def test_zeros(self):
        """Zero weights roundtrip."""
        K = 64
        w = jnp.zeros((2, K), dtype=jnp.float32)
        packed, scales, biases = affine_quantize(w, bits=4, group_size=32)
        w_restored = affine_dequantize(packed, scales, biases, bits=4, group_size=32, shape=w.shape)
        np.testing.assert_allclose(np.array(w_restored), 0.0, atol=1e-6)


class TestAffineMatmul:
    """Test affine quantized matmul."""

    @pytest.mark.parametrize("bits", [4, 8])
    @pytest.mark.parametrize("dtype", [jnp.float32, jnp.float16])
    def test_correctness(self, bits, dtype):
        """Affine quantized matmul should approximate standard matmul."""
        rng = np.random.default_rng(42)
        M, N, K = 8, 16, 128
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1, dtype=dtype)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        ref = jnp.array(np.array(x, dtype=np.float32) @ np.array(w).T, dtype=dtype)

        packed, scales, biases = affine_quantize(w, bits=bits, group_size=64)
        out = affine_quantized_matmul(x, packed, scales, biases, bits=bits, group_size=64, K=K)

        assert out.shape == (M, N)
        assert out.dtype == dtype

        atol = 0.01 if bits == 8 else 0.05
        np.testing.assert_allclose(
            np.array(out, dtype=np.float32),
            np.array(ref, dtype=np.float32),
            atol=atol,
            rtol=0.3,
            err_msg=f"Failed for bits={bits}, dtype={dtype}",
        )

    def test_gpu_dispatch(self):
        """Affine matmul should dispatch to GPU when available."""
        from jax_plugins.silicon.quantized import _gpu_available

        if not _gpu_available():
            pytest.skip("GPU not available")

        rng = np.random.default_rng(42)
        M, N, K = 4, 8, 64
        x = jnp.array(rng.standard_normal((M, K)).astype(np.float32) * 0.1)
        w = jnp.array(rng.standard_normal((N, K)).astype(np.float32) * 0.1)

        packed, scales, biases = affine_quantize(w, bits=4, group_size=32)
        out = affine_quantized_matmul(x, packed, scales, biases, bits=4, group_size=32, K=K)

        assert out.shape == (M, N)
        assert not jnp.any(jnp.isnan(out))


class TestGPUOutputDtype:
    """GPU output buffers must match the kernel's write dtype before conversion."""

    def test_mx_matmul_widens_float16_output_to_float32(self):
        from jax_plugins.silicon.quantized import _gpu_available

        if not _gpu_available():
            pytest.skip("Metal GPU not available")

        x = jnp.ones((1, 64), dtype=jnp.float16)
        w = jnp.ones((2, 64), dtype=jnp.float32)
        packed, scales = mx_quantize(w, mode="mxfp8")
        expected = (
            np.asarray(x, dtype=np.float32)
            @ np.asarray(mx_dequantize(packed, scales, mode="mxfp8", shape=w.shape), dtype=np.float32).T
        )

        out = quantized_matmul(x, packed, scales, mode="mxfp8", K=64, out_dtype=jnp.float32)

        assert out.dtype == jnp.float32
        np.testing.assert_allclose(np.asarray(out), expected, atol=0.1, rtol=0.01)

    def test_affine_matmul_widens_float16_output_to_float32(self):
        from jax_plugins.silicon.quantized import _gpu_available

        if not _gpu_available():
            pytest.skip("Metal GPU not available")

        x = jnp.ones((1, 64), dtype=jnp.float16)
        w = jnp.ones((2, 64), dtype=jnp.float32)
        packed, scales, biases = affine_quantize(w, bits=4, group_size=64)

        out = affine_quantized_matmul(x, packed, scales, biases, bits=4, group_size=64, K=64, out_dtype=jnp.float32)

        assert out.dtype == jnp.float32
        np.testing.assert_array_equal(np.asarray(out), [[64.0, 64.0]])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
