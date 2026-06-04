# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Tests for custom Metal kernel API."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.silicon.metal import MetalKernel, device_info, execute, kernel

# ──────────────────────────────────────────────────────────────────
# Kernel sources
# ──────────────────────────────────────────────────────────────────

VEC_ADD_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void vec_add(device const float* a [[buffer(0)]],
                    device const float* b [[buffer(1)]],
                    device float* c [[buffer(2)]],
                    uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid] + b[gid];
}
"""

SCALE_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void scale(device const float* x [[buffer(0)]],
                  device float* y [[buffer(1)]],
                  constant float& factor [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    y[gid] = x[gid] * factor;
}
"""

MATMUL_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void matmul(device const float* A [[buffer(0)]],
                   device const float* B [[buffer(1)]],
                   device float* C [[buffer(2)]],
                   constant uint& M [[buffer(3)]],
                   constant uint& N [[buffer(4)]],
                   constant uint& K [[buffer(5)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }
    C[row * N + col] = acc;
}
"""

MULTI_OUTPUT_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void split(device const float* x [[buffer(0)]],
                  device float* pos [[buffer(1)]],
                  device float* neg [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    float v = x[gid];
    pos[gid] = max(0.0f, v);
    neg[gid] = min(0.0f, v);
}
"""

HALF_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void half_add(device const half* a [[buffer(0)]],
                     device const half* b [[buffer(1)]],
                     device half* c [[buffer(2)]],
                     uint gid [[thread_position_in_grid]]) {
    c[gid] = a[gid] + b[gid];
}
"""


# ──────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────


class TestDeviceInfo:
    def test_device_info(self):
        info = device_info()
        assert "name" in info
        assert info["has_unified_memory"] is True
        assert info["max_threads_per_threadgroup"] >= 256


class TestExecute:
    def test_vector_add(self):
        """Simple element-wise addition."""
        a = jnp.arange(256, dtype=jnp.float32)
        b = jnp.ones(256, dtype=jnp.float32) * 5
        c = execute(
            VEC_ADD_SOURCE,
            "vec_add",
            inputs=[a, b],
            output_shapes=[(256,)],
            output_dtypes=[jnp.float32],
            grid=(256, 1, 1),
            threadgroup=(256, 1, 1),
        )
        np.testing.assert_allclose(np.array(c), np.array(a + b), atol=1e-6)

    def test_scale_with_constant(self):
        """Kernel with constant buffer parameter."""
        x = jnp.arange(64, dtype=jnp.float32)
        y = execute(
            SCALE_SOURCE,
            "scale",
            inputs=[x],
            output_shapes=[(64,)],
            output_dtypes=[jnp.float32],
            constants={2: 2.5},
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
        )
        np.testing.assert_allclose(np.array(y), np.array(x * 2.5), atol=1e-5)

    def test_matmul(self):
        """2D matrix multiply with constant dimensions."""
        rng = np.random.default_rng(42)
        M, N, K = 8, 16, 32
        A = jnp.array(rng.standard_normal((M, K)).astype(np.float32))
        B = jnp.array(rng.standard_normal((K, N)).astype(np.float32))
        C = execute(
            MATMUL_SOURCE,
            "matmul",
            inputs=[A, B],
            output_shapes=[(M, N)],
            output_dtypes=[jnp.float32],
            constants={3: M, 4: N, 5: K},
            grid=(N, M, 1),
            threadgroup=(min(N, 16), min(M, 8), 1),
        )
        ref = A @ B
        np.testing.assert_allclose(np.array(C), np.array(ref), atol=1e-4)

    def test_multi_output(self):
        """Kernel producing multiple output buffers."""
        x = jnp.array([-3, -1, 0, 2, 5], dtype=jnp.float32)
        pos, neg = execute(
            MULTI_OUTPUT_SOURCE,
            "split",
            inputs=[x],
            output_shapes=[(5,), (5,)],
            output_dtypes=[jnp.float32, jnp.float32],
            grid=(5, 1, 1),
            threadgroup=(5, 1, 1),
        )
        np.testing.assert_array_equal(np.array(pos), [0, 0, 0, 2, 5])
        np.testing.assert_array_equal(np.array(neg), [-3, -1, 0, 0, 0])

    def test_half_precision(self):
        """float16 input/output."""
        a = jnp.arange(64, dtype=jnp.float16)
        b = jnp.ones(64, dtype=jnp.float16)
        c = execute(
            HALF_SOURCE,
            "half_add",
            inputs=[a, b],
            output_shapes=[(64,)],
            output_dtypes=[jnp.float16],
            grid=(64, 1, 1),
            threadgroup=(64, 1, 1),
        )
        np.testing.assert_allclose(np.array(c), np.array(a + b), atol=0.01)


class TestMetalKernel:
    def test_cached_kernel(self):
        """MetalKernel should cache and reuse compilation."""
        k = MetalKernel(VEC_ADD_SOURCE, "vec_add")
        a = jnp.ones(32, dtype=jnp.float32)
        b = jnp.ones(32, dtype=jnp.float32) * 2
        # Call twice — second call uses cached pipeline
        for _ in range(2):
            c = k(
                inputs=[a, b],
                output_shapes=[(32,)],
                output_dtypes=[jnp.float32],
                grid=(32, 1, 1),
                threadgroup=(32, 1, 1),
            )
            np.testing.assert_allclose(np.array(c), np.array(a + b))

    def test_repr(self):
        k = MetalKernel(VEC_ADD_SOURCE, "vec_add")
        r = repr(k)
        assert "vec_add" in r
        assert "max_tg=" in r


class TestDecoratorAPI:
    def test_decorator_kernel(self):
        """@kernel decorator should work with lambda grid."""

        @kernel("test_neg", grid=lambda n: ((n + 255) // 256, 1, 1), threadgroup=(256, 1, 1))
        def negate():
            return """
            #include <metal_stdlib>
            using namespace metal;
            kernel void test_neg(device const float* x [[buffer(0)]],
                                 device float* y [[buffer(1)]],
                                 constant uint& n [[buffer(2)]],
                                 uint gid [[thread_position_in_grid]]) {
                if (gid >= n) return;
                y[gid] = -x[gid];
            }
            """

        x = jnp.array([1, -2, 3, -4, 5], dtype=jnp.float32)
        y = negate(x, output_shapes=[(5,)], output_dtypes=[jnp.float32], constants={2: 5})
        np.testing.assert_array_equal(np.array(y), [-1, 2, -3, 4, -5])


class TestErrors:
    def test_invalid_kernel_name(self):
        with pytest.raises(ValueError, match="not found"):
            execute(
                VEC_ADD_SOURCE,
                "nonexistent",
                inputs=[],
                output_shapes=[],
                output_dtypes=[],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
            )

    def test_compile_error(self):
        with pytest.raises(RuntimeError, match="compilation failed"):
            execute(
                "kernel void bad( { invalid syntax",
                "bad",
                inputs=[],
                output_shapes=[],
                output_dtypes=[],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
            )

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="same length"):
            execute(
                VEC_ADD_SOURCE,
                "vec_add",
                inputs=[],
                output_shapes=[(10,)],
                output_dtypes=[],
                grid=(1, 1, 1),
                threadgroup=(1, 1, 1),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
