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

    def test_rounding(self):
        """Values between representable points should round correctly."""
        # 0.3 → rounds to 0.5 (nearest to 0.25 boundary)
        bits = _fp4_e2m1_quantize(np.array([0.3], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 0.5

        # 1.6 → rounds to 1.5
        bits = _fp4_e2m1_quantize(np.array([1.6], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 1.5

    def test_saturation(self):
        """Values beyond 6.0 should saturate to 6.0."""
        bits = _fp4_e2m1_quantize(np.array([100.0], dtype=np.float32))
        assert _fp4_e2m1_dequantize(bits)[0] == 6.0


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
