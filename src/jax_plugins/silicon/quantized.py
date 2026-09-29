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

"""MX Microscaling format support for jax-metallib.

Supports MXFP4 (E2M1 elements, E8M0 scales, group=32),
MXFP8 (E4M3 elements, E8M0 scales, group=32), and
NVFP4 (E2M1 elements, E4M3 scales, group=16).

Usage:
    from jax_plugins.silicon.quantized import mx_quantize, mx_dequantize, quantized_matmul

    # Quantize weights
    w_packed, scales = mx_quantize(weights, mode="mxfp4")

    # Run quantized matmul (activations @ dequant(weights)^T)
    output = quantized_matmul(activations, w_packed, scales, mode="mxfp4", K=weights.shape[0])

    # Dequantize for inspection
    w_restored = mx_dequantize(w_packed, scales, mode="mxfp4", shape=weights.shape)
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

_MX_MODES = {
    "mxfp4": {"bits": 4, "group_size": 32, "scale_type": "e8m0"},
    "mxfp8": {"bits": 8, "group_size": 32, "scale_type": "e8m0"},
    "nvfp4": {"bits": 4, "group_size": 16, "scale_type": "e4m3"},
}

# Affine quantization: scale * (int_val - zero_point)
# Supported bit widths: 2, 4, 8. Group sizes: 32, 64, 128.
_AFFINE_DEFAULTS = {"bits": 4, "group_size": 64}

# E2M1 representable values (positive): 0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
_FP4_E2M1_TABLE = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def _fp4_e2m1_quantize(x: np.ndarray) -> np.ndarray:
    """Quantize float32 values to 4-bit E2M1 (returns uint8 with 4-bit values)."""
    sign = np.signbit(x).astype(np.uint8) << 3
    a = np.abs(x)
    mag = np.zeros_like(a, dtype=np.uint8)
    # At exact midpoints, choose the value with an even magnitude code.
    mag = np.where(a > 5.0, 7, mag)
    mag = np.where((a >= 3.5) & (a <= 5.0), 6, mag)
    mag = np.where((a > 2.5) & (a < 3.5), 5, mag)
    mag = np.where((a >= 1.75) & (a <= 2.5), 4, mag)
    mag = np.where((a > 1.25) & (a < 1.75), 3, mag)
    mag = np.where((a >= 0.75) & (a <= 1.25), 2, mag)
    mag = np.where((a > 0.25) & (a < 0.75), 1, mag)
    return sign | mag.astype(np.uint8)


def _fp4_e2m1_dequantize(bits: np.ndarray) -> np.ndarray:
    """Dequantize 4-bit E2M1 values to float32."""
    sign = np.where(bits & 0x8, -1.0, 1.0).astype(np.float32)
    mag_idx = (bits & 0x7).astype(np.int32)
    return sign * _FP4_E2M1_TABLE[mag_idx]


def _fp8_e4m3_quantize(x: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP8 E4M3 using nearest-even rounding (returns uint8)."""
    x = np.asarray(x, dtype=np.float32)
    sign = np.signbit(x).astype(np.uint8)
    a = np.abs(np.clip(x, -448.0, 448.0))
    # Preserve the existing saturating-NaN behavior; do not encode the reserved NaN code.
    a = np.where(np.isnan(a), 448.0, a)

    upper = np.searchsorted(_FP8_E4M3_POSITIVE_VALUES, a, side="left")
    upper = np.minimum(upper, 0x7E)
    lower = np.maximum(upper - 1, 0)
    lower_distance = a - _FP8_E4M3_POSITIVE_VALUES[lower]
    upper_distance = _FP8_E4M3_POSITIVE_VALUES[upper] - a
    choose_upper = (upper_distance < lower_distance) | ((upper_distance == lower_distance) & (upper % 2 == 0))
    mag = np.where(choose_upper, upper, lower).astype(np.uint8)
    return mag | (sign << 7)


def _fp8_e4m3_dequantize(bits: np.ndarray) -> np.ndarray:
    """Dequantize FP8 E4M3 → float32."""
    sign = np.where(bits & 0x80, -1.0, 1.0).astype(np.float32)
    unsigned = (bits & 0x7F).astype(np.uint32)
    # Reconstruct: shift to half format and scale
    half_bits = (unsigned << 7).astype(np.uint16)
    # Interpret as float16 and convert
    result = half_bits.view(np.float16).astype(np.float32) * 256.0
    return np.where(unsigned == 0x7F, np.nan, sign * result)


_FP8_E4M3_POSITIVE_VALUES = _fp8_e4m3_dequantize(np.arange(0x7F, dtype=np.uint8))


def _fp8_e8m0_quantize(x: np.ndarray) -> np.ndarray:
    """Quantize positive float → E8M0 (pure power-of-2 scale)."""
    nonfinite = ~np.isfinite(x)
    x = np.where(nonfinite, 1.0, np.maximum(x, np.finfo(np.float32).tiny))  # avoid log(0)
    log2_x = np.log2(x)
    n = np.round(log2_x).astype(np.int32)
    n = np.clip(n, -127, 127)
    return np.where(nonfinite, 0xFF, n + 127).astype(np.uint8)


def _fp8_e8m0_dequantize(bits: np.ndarray) -> np.ndarray:
    """Dequantize E8M0 → float32 (2^(bits - 127)); 0xFF is NaN."""
    exp = np.minimum(bits.astype(np.int32) - 127, 127)
    values = np.ldexp(np.ones_like(exp, dtype=np.float32), exp)
    return np.where(bits == 0xFF, np.nan, values)


def mx_quantize(w, *, mode: str = "mxfp4") -> tuple:
    """Quantize a float weight matrix to MX microscaling format.

    Args:
        w: Weight array of shape [..., K]. Must be float32 or float16.
        mode: One of "mxfp4", "mxfp8", "nvfp4".

    Returns:
        (packed_weights, scales) tuple:
        - packed_weights: uint8 array. For FP4: shape [..., K//2]. For FP8: shape [..., K].
        - scales: uint8 array of shape [..., K // group_size].
    """
    if mode not in _MX_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(_MX_MODES.keys())}")

    cfg = _MX_MODES[mode]
    bits = cfg["bits"]
    group_size = cfg["group_size"]

    # Convert to numpy for CPU quantization
    w_np = np.asarray(w, dtype=np.float32)
    orig_shape = w_np.shape
    K = orig_shape[-1]

    if K % group_size != 0:
        raise ValueError(f"Last dimension ({K}) must be divisible by group_size ({group_size})")

    # Reshape to [num_rows, K]
    w_flat = w_np.reshape(-1, K)
    num_rows = w_flat.shape[0]

    # Process each group
    num_groups = K // group_size
    pack_factor = 2 if bits == 4 else 1

    packed_list = []
    scales_list = []

    for row in range(num_rows):
        row_packed = []
        row_scales = []
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = w_flat[row, start:end]

            # Compute group scale
            group_max = np.max(np.abs(group))
            max_repr = 6.0 if bits == 4 else 448.0
            raw_scale = group_max / max_repr if group_max != 0 else 0.0

            # Quantize scale
            if cfg["scale_type"] == "e8m0":
                q_scale = _fp8_e8m0_quantize(np.array([raw_scale]))[0]
                scale = _fp8_e8m0_dequantize(np.array([q_scale]))[0]
            else:
                q_scale = _fp8_e4m3_quantize(np.array([raw_scale]))[0]
                scale = _fp8_e4m3_dequantize(np.array([q_scale]))[0]

            row_scales.append(q_scale)

            # Quantize elements
            normalized = group / scale if scale != 0 else np.zeros_like(group)
            if bits == 4:
                quantized = _fp4_e2m1_quantize(normalized)
                # Pack pairs into bytes
                for i in range(0, group_size, 2):
                    byte = quantized[i] | (quantized[i + 1] << 4)
                    row_packed.append(byte)
            else:
                quantized = _fp8_e4m3_quantize(normalized)
                row_packed.extend(quantized.tolist())

        packed_list.append(row_packed)
        scales_list.append(row_scales)

    packed = np.array(packed_list, dtype=np.uint8)
    scales = np.array(scales_list, dtype=np.uint8)

    # Reshape to match original batch dims
    packed_shape = [*list(orig_shape[:-1]), K // pack_factor]
    scales_shape = [*list(orig_shape[:-1]), num_groups]

    packed = packed.reshape(packed_shape)
    scales = scales.reshape(scales_shape)

    return jnp.array(packed), jnp.array(scales)


def mx_dequantize(packed, scales, *, mode: str = "mxfp4", shape=None, dtype=jnp.float32):
    """Dequantize MX-packed weights back to float.

    Args:
        packed: uint8 packed weights from mx_quantize.
        scales: uint8 scales from mx_quantize.
        mode: One of "mxfp4", "mxfp8", "nvfp4".
        shape: Original weight shape (required for FP4 to determine K).
        dtype: Output dtype (float32 or float16).

    Returns:
        Dequantized weight array.
    """
    if mode not in _MX_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(_MX_MODES.keys())}")

    cfg = _MX_MODES[mode]
    bits = cfg["bits"]
    group_size = cfg["group_size"]

    packed_np = np.asarray(packed, dtype=np.uint8)
    scales_np = np.asarray(scales, dtype=np.uint8)

    if bits == 4 and shape is None:
        raise ValueError("shape is required for FP4 dequantization")
    if shape is not None:
        K = shape[-1]
    else:
        K = packed_np.shape[-1]  # FP8: K == packed last dim

    # Flatten to 2D
    num_groups = K // group_size
    packed_flat = packed_np.reshape(-1, packed_np.shape[-1])
    scales_flat = scales_np.reshape(-1, num_groups)
    num_rows = packed_flat.shape[0]

    result = np.zeros((num_rows, K), dtype=np.float32)

    for row in range(num_rows):
        for g in range(num_groups):
            # Dequantize scale
            if cfg["scale_type"] == "e8m0":
                scale = _fp8_e8m0_dequantize(np.array([scales_flat[row, g]]))[0]
            else:
                scale = _fp8_e4m3_dequantize(np.array([scales_flat[row, g]]))[0]

            start = g * group_size
            for i in range(group_size):
                k = start + i
                if bits == 4:
                    byte_idx = k // 2
                    byte = packed_flat[row, byte_idx]
                    nibble = (byte & 0x0F) if (k % 2 == 0) else (byte >> 4)
                    val = _fp4_e2m1_dequantize(np.array([nibble]))[0]
                else:
                    val = _fp8_e4m3_dequantize(np.array([packed_flat[row, k]]))[0]
                result[row, k] = scale * val

    if shape is not None:
        result = result.reshape(shape)

    return jnp.array(result, dtype=dtype)


def affine_quantize(w, *, bits: int = 4, group_size: int = 64) -> tuple:
    """Quantize a float weight matrix using affine (asymmetric) quantization.

    Affine quantization: quantized = round((w - zero_point) / scale)
    where scale and zero_point are computed per group.

    This is the standard quantization used by GPTQ, AWQ, and most LLM
    quantization methods. Compatible with MLX's mode="affine".

    Args:
        w: Weight array of shape [..., K]. Must be float32 or float16.
        bits: Bit width (2, 3, 4, or 8). Default 4.
        group_size: Number of elements per quantization group. Default 64.

    Returns:
        (packed_weights, scales, biases) tuple:
        - packed_weights: uint8 (for 8-bit) or uint32 (for sub-byte) packed array.
          For 4-bit: shape [..., K // 8] as uint32 (8 x 4-bit values per uint32).
          For 8-bit: shape [..., K] as uint8.
        - scales: float32 array of shape [..., K // group_size].
        - biases: float32 array of shape [..., K // group_size].
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"bits must be 2, 3, 4, or 8; got {bits}")

    w_np = np.asarray(w, dtype=np.float32)
    orig_shape = w_np.shape
    K = orig_shape[-1]

    if K % group_size != 0:
        raise ValueError(f"Last dimension ({K}) must be divisible by group_size ({group_size})")

    w_flat = w_np.reshape(-1, K)
    num_rows = w_flat.shape[0]
    num_groups = K // group_size
    n_bins = (1 << bits) - 1  # e.g. 15 for 4-bit

    scales_list = []
    biases_list = []
    quantized_list = []

    for row in range(num_rows):
        row_scales = []
        row_biases = []
        row_quant = []
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            group = w_flat[row, start:end]

            w_min = float(np.min(group))
            w_max = float(np.max(group))

            # Compute scale and zero point
            scale = (w_max - w_min) / n_bins if w_max != w_min else 1.0
            bias = w_min

            row_scales.append(scale)
            row_biases.append(bias)

            # Quantize: round((w - bias) / scale), clamp to [0, n_bins]
            quantized = np.round((group - bias) / scale).astype(np.int32)
            quantized = np.clip(quantized, 0, n_bins)
            row_quant.extend(quantized.tolist())

        scales_list.append(row_scales)
        biases_list.append(row_biases)
        quantized_list.append(row_quant)

    quant_flat = np.array(quantized_list, dtype=np.uint32)
    scales_arr = np.array(scales_list, dtype=np.float32)
    biases_arr = np.array(biases_list, dtype=np.float32)

    # Pack sub-byte values
    if bits == 8:
        packed = quant_flat.astype(np.uint8)
        packed_shape = [*list(orig_shape[:-1]), K]
    elif bits == 4:
        # Pack 8 x 4-bit values into one uint32
        packed_rows = []
        for row in range(num_rows):
            row_packed = []
            for i in range(0, K, 8):
                val = np.uint32(0)
                for j in range(8):
                    val |= np.uint32(quant_flat[row, i + j] & 0xF) << np.uint32(j * 4)
                row_packed.append(val)
            packed_rows.append(row_packed)
        packed = np.array(packed_rows, dtype=np.uint32)
        packed_shape = [*list(orig_shape[:-1]), K // 8]
    elif bits == 3:
        # Pack 32 x 3-bit values into 3 uint32s → store as 3 uint32 per 32 elements
        packed_rows = []
        for row in range(num_rows):
            row_packed = []
            for i in range(0, K, 32):
                words = [np.uint32(0)] * 3
                for j in range(32):
                    val3 = quant_flat[row, i + j] & 0x7
                    bit_pos = j * 3
                    word_idx = bit_pos // 32
                    bit_off = bit_pos % 32
                    words[word_idx] |= np.uint32(val3) << np.uint32(bit_off)
                    # Handle overflow across word boundary
                    if bit_off > 29 and word_idx < 2:
                        words[word_idx + 1] |= np.uint32(val3) >> np.uint32(32 - bit_off)
                row_packed.extend(words)
            packed_rows.append(row_packed)
        packed = np.array(packed_rows, dtype=np.uint32)
        packed_shape = [*list(orig_shape[:-1]), K * 3 // 32]
    elif bits == 2:
        # Pack 16 x 2-bit values into one uint32
        packed_rows = []
        for row in range(num_rows):
            row_packed = []
            for i in range(0, K, 16):
                val = np.uint32(0)
                for j in range(16):
                    val |= np.uint32(quant_flat[row, i + j] & 0x3) << np.uint32(j * 2)
                row_packed.append(val)
            packed_rows.append(row_packed)
        packed = np.array(packed_rows, dtype=np.uint32)
        packed_shape = [*list(orig_shape[:-1]), K // 16]

    packed = packed.reshape(packed_shape)
    scales_shape = [*list(orig_shape[:-1]), num_groups]
    scales_arr = scales_arr.reshape(scales_shape)
    biases_arr = biases_arr.reshape(scales_shape)

    return jnp.array(packed), jnp.array(scales_arr), jnp.array(biases_arr)


def affine_dequantize(packed, scales, biases, *, bits: int = 4, group_size: int = 64, shape=None, dtype=jnp.float32):
    """Dequantize affine-packed weights back to float.

    Dequantization: w = scale * quantized_val + bias

    Args:
        packed: Packed weight array from affine_quantize.
        scales: Float32 scale array from affine_quantize.
        biases: Float32 bias array from affine_quantize.
        bits: Bit width (2, 3, 4, or 8).
        group_size: Group size used during quantization.
        shape: Original weight shape (required for sub-byte packing).
        dtype: Output dtype.

    Returns:
        Dequantized weight array.
    """
    if bits not in (2, 3, 4, 8):
        raise ValueError(f"bits must be 2, 3, 4, or 8; got {bits}")

    packed_np = np.asarray(packed)
    scales_np = np.asarray(scales, dtype=np.float32)
    biases_np = np.asarray(biases, dtype=np.float32)

    if bits != 8 and shape is None:
        raise ValueError("shape is required for sub-byte dequantization")
    K = shape[-1] if shape is not None else packed_np.shape[-1]

    num_groups = K // group_size
    packed_flat = packed_np.reshape(-1, packed_np.shape[-1])
    scales_flat = scales_np.reshape(-1, num_groups)
    biases_flat = biases_np.reshape(-1, num_groups)
    num_rows = scales_flat.shape[0]

    result = np.zeros((num_rows, K), dtype=np.float32)
    (1 << bits) - 1

    for row in range(num_rows):
        # Unpack all quantized values for this row
        quant_vals = np.zeros(K, dtype=np.int32)
        if bits == 8:
            quant_vals = packed_flat[row, :K].astype(np.int32)
        elif bits == 4:
            for i in range(0, K, 8):
                word = int(packed_flat[row, i // 8])
                for j in range(8):
                    quant_vals[i + j] = (word >> (j * 4)) & 0xF
        elif bits == 2:
            for i in range(0, K, 16):
                word = int(packed_flat[row, i // 16])
                for j in range(16):
                    quant_vals[i + j] = (word >> (j * 2)) & 0x3
        elif bits == 3:
            for i in range(0, K, 32):
                base_word = (i * 3) // 32
                for j in range(32):
                    bit_pos = j * 3
                    word_idx = bit_pos // 32
                    bit_off = bit_pos % 32
                    val = (int(packed_flat[row, base_word + word_idx]) >> bit_off) & 0x7
                    if bit_off > 29 and (base_word + word_idx + 1) < packed_flat.shape[1]:
                        val |= (int(packed_flat[row, base_word + word_idx + 1]) << (32 - bit_off)) & 0x7
                    quant_vals[i + j] = val

        # Dequantize with per-group scale + bias
        for g in range(num_groups):
            start = g * group_size
            end = start + group_size
            scale = scales_flat[row, g]
            bias = biases_flat[row, g]
            result[row, start:end] = scale * quant_vals[start:end].astype(np.float32) + bias

    if shape is not None:
        result = result.reshape(shape)
    return jnp.array(result, dtype=dtype)


_QGEMM_KERNEL_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ── FP4 E2M1 dequantize ──
inline float dequant_fp4(uint8_t nibble) {
    // E2M1: bit trick for fast dequantize
    constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return table[nibble & 0xF];
}

// ── FP8 E4M3 dequantize ──
inline float dequant_fp8_e4m3(uint8_t bits) {
    if ((bits & 127) == 127) return as_type<float>(0x7fc00000u);
    uint16_t v = (uint16_t)(bits & 127) << 7;
    half converted = as_type<half>(v);
    converted *= 256.0h;
    return (bits & 128) ? -(float)converted : (float)converted;
}

// ── FP8 E8M0 scale dequantize ──
inline float dequant_scale_e8m0(uint8_t bits) {
    if (bits == 0xFF) return as_type<float>(0x7fc00000u);
    uint32_t out = (bits == 0) ? 0x00400000u : ((uint32_t)bits << 23);
    return as_type<float>(out);
}

// ── FP8 E4M3 scale dequantize ──
inline float dequant_scale_e4m3(uint8_t bits) {
    return dequant_fp8_e4m3(bits);
}

// ═══════════════════════════════════════════════════════════════════
// Quantized GEMM: C[M,N] = A[M,K] @ dequant(B[N,K/pack], scales[N,K/gs])^T
// Each SIMD group (32 lanes) cooperatively reduces over one K-group.
// Threadgroup: 4 SIMD groups → 4 rows per threadgroup.
// Grid: (N, ceil(M/4), 1)
// ═══════════════════════════════════════════════════════════════════

kernel void mx_qgemm_mxfp4_f32(
    device const float* A          [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device float* C                [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 32;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e8m0(s_row[g]);
        uint k = g * GROUP_SIZE + lane;
        float a_val = a_row[k];
        uint8_t packed_byte = b_row[k / 2];
        uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        float w_val = scale * dequant_fp4(nibble);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_mxfp4_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 32;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e8m0(s_row[g]);
        uint k = g * GROUP_SIZE + lane;
        float a_val = (float)a_row[k];
        uint8_t packed_byte = b_row[k / 2];
        uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        float w_val = scale * dequant_fp4(nibble);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

kernel void mx_qgemm_mxfp8_f32(
    device const float* A          [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device float* C                [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 32;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e8m0(s_row[g]);
        uint k = g * GROUP_SIZE + lane;
        float a_val = a_row[k];
        float w_val = scale * dequant_fp8_e4m3(b_row[k]);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_mxfp8_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 32;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e8m0(s_row[g]);
        uint k = g * GROUP_SIZE + lane;
        float a_val = (float)a_row[k];
        float w_val = scale * dequant_fp8_e4m3(b_row[k]);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

kernel void mx_qgemm_nvfp4_f32(
    device const float* A          [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device float* C                [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 16;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e4m3(s_row[g]);
        // NV: group_size=16, only first 16 lanes active
        if (lane < 16) {
            uint k = g * GROUP_SIZE + lane;
            float a_val = a_row[k];
            uint8_t packed_byte = b_row[k / 2];
            uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            float w_val = scale * dequant_fp4(nibble);
            acc += simd_sum(a_val * w_val);
        } else {
            acc += simd_sum(0.0f);
        }
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_nvfp4_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint GROUP_SIZE = 16;
    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        float scale = dequant_scale_e4m3(s_row[g]);
        if (lane < 16) {
            uint k = g * GROUP_SIZE + lane;
            float a_val = (float)a_row[k];
            uint8_t packed_byte = b_row[k / 2];
            uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            float w_val = scale * dequant_fp4(nibble);
            acc += simd_sum(a_val * w_val);
        } else {
            acc += simd_sum(0.0f);
        }
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}
"""

# ──────────────────────────────────────────────────────────────────
# Affine quantized GEMM Metal kernel source
# ──────────────────────────────────────────────────────────────────

_AFFINE_QGEMM_KERNEL_SOURCE = r"""
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// Affine quantized GEMM: C[M,N] = A[M,K] @ dequant(B_packed[N,...], scales[N,ngrp], biases[N,ngrp])^T
// Supports 4-bit (packed as 8 x 4-bit per uint32) and 8-bit packing.
// Each SIMD group (32 lanes) reduces over one group (group_size elements).
// Grid: (N, ceil(M/4), 1), Threadgroup: (128, 1, 1) = 4 SIMD groups

// ── 4-bit affine: 8 values packed per uint32 ──

kernel void affine_qgemm_4bit_f32(
    device const float* A           [[buffer(0)]],
    device const uint32_t* B_packed [[buffer(1)]],   // [N, K/8]
    device const float* B_scales    [[buffer(2)]],   // [N, K/group_size]
    device const float* B_biases    [[buffer(3)]],   // [N, K/group_size]
    device float* C                 [[buffer(4)]],   // [M, N]
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    uint packed_K = K / 8;  // number of uint32 words per row
    device const uint32_t* b_row = B_packed + col * packed_K;
    uint num_groups = K / group_size;
    device const float* s_row = B_scales + col * num_groups;
    device const float* z_row = B_biases + col * num_groups;

    float acc = 0.0f;
    // Each lane processes group_size/32 elements per group iteration
    for (uint g = 0; g < num_groups; g++) {
        float scale = s_row[g];
        float bias = z_row[g];
        uint k_base = g * group_size;

        // Each of 32 lanes handles one element within the group
        // For group_size > 32, loop over sub-groups
        for (uint sub = 0; sub < group_size; sub += 32) {
            uint k = k_base + sub + lane;
            if (k < K) {
                float a_val = a_row[k];
                // Unpack 4-bit value from uint32 word
                uint word_idx = k / 8;
                uint bit_off = (k % 8) * 4;
                uint8_t q_val = (b_row[word_idx] >> bit_off) & 0xF;
                float w_val = scale * (float)q_val + bias;
                acc += simd_sum(a_val * w_val);
            }
        }
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void affine_qgemm_4bit_f16(
    device const half* A            [[buffer(0)]],
    device const uint32_t* B_packed [[buffer(1)]],
    device const float* B_scales    [[buffer(2)]],
    device const float* B_biases    [[buffer(3)]],
    device half* C                  [[buffer(4)]],
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    uint packed_K = K / 8;
    device const uint32_t* b_row = B_packed + col * packed_K;
    uint num_groups = K / group_size;
    device const float* s_row = B_scales + col * num_groups;
    device const float* z_row = B_biases + col * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = s_row[g];
        float bias = z_row[g];
        uint k_base = g * group_size;
        for (uint sub = 0; sub < group_size; sub += 32) {
            uint k = k_base + sub + lane;
            if (k < K) {
                float a_val = (float)a_row[k];
                uint word_idx = k / 8;
                uint bit_off = (k % 8) * 4;
                uint8_t q_val = (b_row[word_idx] >> bit_off) & 0xF;
                float w_val = scale * (float)q_val + bias;
                acc += simd_sum(a_val * w_val);
            }
        }
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

// ── 8-bit affine ──

kernel void affine_qgemm_8bit_f32(
    device const float* A           [[buffer(0)]],
    device const uint8_t* B_packed  [[buffer(1)]],   // [N, K]
    device const float* B_scales    [[buffer(2)]],
    device const float* B_biases    [[buffer(3)]],
    device float* C                 [[buffer(4)]],
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    uint num_groups = K / group_size;
    device const float* s_row = B_scales + col * num_groups;
    device const float* z_row = B_biases + col * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = s_row[g];
        float bias = z_row[g];
        uint k_base = g * group_size;
        for (uint sub = 0; sub < group_size; sub += 32) {
            uint k = k_base + sub + lane;
            if (k < K) {
                float a_val = a_row[k];
                float w_val = scale * (float)b_row[k] + bias;
                acc += simd_sum(a_val * w_val);
            }
        }
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void affine_qgemm_8bit_f16(
    device const half* A            [[buffer(0)]],
    device const uint8_t* B_packed  [[buffer(1)]],
    device const float* B_scales    [[buffer(2)]],
    device const float* B_biases    [[buffer(3)]],
    device half* C                  [[buffer(4)]],
    constant uint& M                [[buffer(5)]],
    constant uint& N                [[buffer(6)]],
    constant uint& K                [[buffer(7)]],
    constant uint& group_size       [[buffer(8)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    uint row = tgid.y * 4 + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    uint num_groups = K / group_size;
    device const float* s_row = B_scales + col * num_groups;
    device const float* z_row = B_biases + col * num_groups;

    float acc = 0.0f;
    for (uint g = 0; g < num_groups; g++) {
        float scale = s_row[g];
        float bias = z_row[g];
        uint k_base = g * group_size;
        for (uint sub = 0; sub < group_size; sub += 32) {
            uint k = k_base + sub + lane;
            if (k < K) {
                float a_val = (float)a_row[k];
                float w_val = scale * (float)b_row[k] + bias;
                acc += simd_sum(a_val * w_val);
            }
        }
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}
"""

# Kernel name mapping: (mode, dtype_str) → (source, kernel_name)
_QGEMM_KERNEL_NAMES = {
    ("mxfp4", "float32"): "mx_qgemm_mxfp4_f32",
    ("mxfp4", "float16"): "mx_qgemm_mxfp4_f16",
    ("mxfp8", "float32"): "mx_qgemm_mxfp8_f32",
    ("mxfp8", "float16"): "mx_qgemm_mxfp8_f16",
    ("nvfp4", "float32"): "mx_qgemm_nvfp4_f32",
    ("nvfp4", "float16"): "mx_qgemm_nvfp4_f16",
}

_AFFINE_KERNEL_NAMES = {
    (4, "float32"): "affine_qgemm_4bit_f32",
    (4, "float16"): "affine_qgemm_4bit_f16",
    (8, "float32"): "affine_qgemm_8bit_f32",
    (8, "float16"): "affine_qgemm_8bit_f16",
}

# Whether to use Metal GPU kernels (requires pyobjc-framework-Metal)
_USE_GPU_KERNEL = None


def _gpu_available() -> bool:
    """Check if Metal GPU kernel dispatch is available."""
    global _USE_GPU_KERNEL
    if _USE_GPU_KERNEL is None:
        try:
            from jax_plugins.silicon.metal import _ensure_metal

            _ensure_metal()
            _USE_GPU_KERNEL = True
        except Exception:
            _USE_GPU_KERNEL = False
    return _USE_GPU_KERNEL


def quantized_matmul(x, w_packed, scales, *, mode: str = "mxfp4", K: int, out_dtype=None):
    """Quantized matrix multiply: x @ dequant(w_packed, scales)^T.

    The weight matrix is stored transposed: w_packed[N, K//pack_factor],
    scales[N, K//group_size]. Output is x[..., M, K] @ W[N, K]^T = C[..., M, N].

    When PyObjC Metal bindings are available, dispatches a fused quantized GEMM
    kernel on GPU that dequantizes weights on-the-fly during matmul.
    Falls back to CPU dequantize + JAX matmul otherwise.

    Args:
        x: Activation array of shape [..., M, K]. float32 or float16.
        w_packed: Packed weight array of shape [N, K//pack_factor]. uint8.
        scales: Scale array of shape [N, K//group_size]. uint8.
        mode: One of "mxfp4", "mxfp8", "nvfp4".
        K: The inner dimension size (needed to reconstruct weight shape).
        out_dtype: Output dtype. Defaults to x.dtype.

    Returns:
        Output array of shape [..., M, N].
    """
    if mode not in _MX_MODES:
        raise ValueError(f"Unknown mode '{mode}'. Must be one of {list(_MX_MODES.keys())}")

    if out_dtype is None:
        out_dtype = x.dtype

    N = w_packed.shape[0]

    # Try GPU kernel path
    if _gpu_available():
        return _quantized_matmul_gpu(x, w_packed, scales, mode=mode, K=K, out_dtype=out_dtype)

    # CPU fallback: dequantize + standard matmul
    w_shape = (N, K)
    w_dequant = mx_dequantize(w_packed, scales, mode=mode, shape=w_shape, dtype=x.dtype)
    return jnp.matmul(x, w_dequant.T).astype(out_dtype)


def _quantized_matmul_gpu(x, w_packed, scales, *, mode: str, K: int, out_dtype):
    """GPU path: dispatch Metal quantized GEMM kernel."""
    from jax_plugins.silicon.metal import execute

    N = w_packed.shape[0]
    x_2d = x.reshape(-1, K) if x.ndim > 2 else x
    M = x_2d.shape[0]

    # Select kernel based on mode and dtype
    dtype_str = str(jnp.dtype(x.dtype))
    key = (mode, dtype_str)
    if key not in _QGEMM_KERNEL_NAMES:
        # Unsupported dtype for GPU — fall back to CPU
        w_shape = (N, K)
        w_dequant = mx_dequantize(w_packed, scales, mode=mode, shape=w_shape, dtype=x.dtype)
        return jnp.matmul(x, w_dequant.T).astype(out_dtype)

    kernel_name = _QGEMM_KERNEL_NAMES[key]

    # Grid: one threadgroup column per output column, ceil(M/4) rows
    # Threadgroup: 128 threads = 4 SIMD groups x 32 lanes
    grid = (N, (M + 3) // 4, 1)
    threadgroup = (128, 1, 1)

    result = execute(
        _QGEMM_KERNEL_SOURCE,
        kernel_name,
        inputs=[x_2d, w_packed, scales],
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
        constants={4: M, 5: N, 6: K},
        grid=grid,
        threadgroup=threadgroup,
    )

    # Reshape back to match original batch dims
    if x.ndim > 2:
        out_shape = [*list(x.shape[:-1]), N]
        result = result.reshape(out_shape)

    return result.astype(out_dtype)


def affine_quantized_matmul(x, w_packed, scales, biases, *, bits: int = 4, group_size: int = 64, K: int, out_dtype=None):
    """Affine quantized matmul: x @ dequant(w_packed, scales, biases)^T.

    Supports 4-bit and 8-bit affine quantization (the standard GPTQ/AWQ format).
    Weights are dequantized on-the-fly during matmul on GPU.

    Args:
        x: Activation array [..., M, K]. float32 or float16.
        w_packed: Packed weights [N, K//8] (4-bit uint32) or [N, K] (8-bit uint8).
        scales: Per-group scales [N, K//group_size]. float32.
        biases: Per-group biases [N, K//group_size]. float32.
        bits: Quantization bit width (4 or 8).
        group_size: Elements per group.
        K: Inner dimension size.
        out_dtype: Output dtype. Defaults to x.dtype.

    Returns:
        Output array [..., M, N].
    """
    if out_dtype is None:
        out_dtype = x.dtype

    N = w_packed.shape[0]

    # Try GPU
    if _gpu_available():
        return _affine_matmul_gpu(
            x, w_packed, scales, biases, bits=bits, group_size=group_size, K=K, out_dtype=out_dtype
        )

    # CPU fallback
    w_shape = (N, K)
    w_dequant = affine_dequantize(
        w_packed, scales, biases, bits=bits, group_size=group_size, shape=w_shape, dtype=x.dtype
    )
    return jnp.matmul(x, w_dequant.T).astype(out_dtype)


def _affine_matmul_gpu(x, w_packed, scales, biases, *, bits: int, group_size: int, K: int, out_dtype):
    """GPU path: dispatch Metal affine quantized GEMM kernel."""
    from jax_plugins.silicon.metal import execute

    N = w_packed.shape[0]
    x_2d = x.reshape(-1, K) if x.ndim > 2 else x
    M = x_2d.shape[0]

    dtype_str = str(jnp.dtype(x.dtype))
    key = (bits, dtype_str)
    if key not in _AFFINE_KERNEL_NAMES:
        # Unsupported — CPU fallback
        w_shape = (N, K)
        w_dequant = affine_dequantize(
            w_packed, scales, biases, bits=bits, group_size=group_size, shape=w_shape, dtype=x.dtype
        )
        return jnp.matmul(x, w_dequant.T).astype(out_dtype)

    kernel_name = _AFFINE_KERNEL_NAMES[key]
    grid = (N, (M + 3) // 4, 1)
    threadgroup = (128, 1, 1)

    result = execute(
        _AFFINE_QGEMM_KERNEL_SOURCE,
        kernel_name,
        inputs=[x_2d, w_packed, scales, biases],
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
        constants={5: M, 6: N, 7: K, 8: group_size},
        grid=grid,
        threadgroup=threadgroup,
    )

    if x.ndim > 2:
        out_shape = [*list(x.shape[:-1]), N]
        result = result.reshape(out_shape)

    return result.astype(out_dtype)
