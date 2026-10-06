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

"""Device-resident custom Metal kernels, including composition inside jax.jit."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.silicon import _native_metal, metal

_ADD_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void add(device const float* x [[buffer(0)]],
                device const float* y [[buffer(1)]],
                device float* out [[buffer(2)]],
                constant uint& n [[buffer(3)]],
                uint gid [[thread_position_in_grid]]) {
    if (gid < n) out[gid] = x[gid] + y[gid];
}
"""


def test_native_kernel_composes_with_jitted_graph_without_eager_bridge(monkeypatch):
    add = metal.compile(_ADD_SOURCE, entry_point="add")

    def forbidden_bridge(*args, **kwargs):
        raise AssertionError("native kernels must not use the eager host-copy bridge")

    monkeypatch.setattr(metal, "_jax_to_numpy", forbidden_bridge)
    monkeypatch.setattr(metal, "_get_pipeline", forbidden_bridge)

    @jax.jit
    def composed(x, y):
        out = add(
            x + jnp.float32(2),
            y * jnp.float32(3),
            out=jax.ShapeDtypeStruct(x.shape, x.dtype),
            threads=x.size,
            threadgroup=256,
            constants={3: np.uint32(x.size)},
        )
        return out * jnp.float32(2)

    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.arange(257, dtype=jnp.float32)
        y = jnp.ones_like(x)
        result = composed(x, y)
        assert result.devices() == x.devices()
        np.testing.assert_array_equal(np.asarray(result), (np.arange(257, dtype=np.float32) + 5) * 2)


def test_compile_is_lazy_and_native_lowering_contains_static_metadata():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((257,), jnp.float32)

    def call(x, y):
        return add(x, y, out=spec, threads=257, constants={3: np.uint32(257)})

    with jax.default_device(jax.devices("mps")[0]):
        lowered = jax.jit(call).lower(spec, spec).as_text()
    assert "jax_metallib.metal_kernel" in lowered
    assert "pure_callback" not in lowered and "io_callback" not in lowered
    assert "entry_point" in lowered and "threadgroup" in lowered and "01010000" in lowered


@pytest.mark.parametrize("entry", ["", "bad-name", "1kernel"])
def test_entry_point_must_be_identifier(entry):
    with pytest.raises(ValueError, match=r"entry_point"):
        metal.compile(_ADD_SOURCE, entry_point=entry)


def test_source_must_be_nonempty_string():
    with pytest.raises(ValueError, match=r"source"):
        metal.compile("", entry_point="add")


@pytest.mark.parametrize("threads", [-1, (1, 2), (1, -1, 1), True, 1.5])
def test_launch_rejects_invalid_thread_counts(threads):
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises((TypeError, ValueError), match=r"threads"):
        add(out=spec, threads=threads)


@pytest.mark.parametrize("threadgroup", [0, -1, (1, 2), (1, 0, 1), True, 1.5])
def test_launch_rejects_invalid_threadgroup(threadgroup):
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises((TypeError, ValueError), match=r"threadgroup"):
        add(out=spec, threads=4, threadgroup=threadgroup)


def test_output_specs_cannot_be_empty():
    with pytest.raises(ValueError, match=r"out"):
        metal.compile(_ADD_SOURCE, entry_point="add")(out=(), threads=0)


def test_constants_cannot_overlap_output_buffers():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    with pytest.raises(ValueError, match=r"constant.*(overlap|buffer)"):
        add(out=jax.ShapeDtypeStruct((1,), jnp.float32), threads=1, constants={0: 1})


@pytest.mark.parametrize("slot", [-1, 31, True])
def test_constants_reject_invalid_buffer_slots(slot):
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    with pytest.raises((TypeError, ValueError), match=r"constant"):
        add(out=jax.ShapeDtypeStruct((1,), jnp.float32), threads=1, constants={slot: 1})


def test_constants_reject_oversized_inline_payload():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    with pytest.raises(ValueError, match=r"4096"):
        add(out=jax.ShapeDtypeStruct((1,), jnp.float32), threads=1, constants={1: b"x" * 4097})


def test_bfloat16_output_requires_explicit_unsupported_error():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    with pytest.raises((TypeError, NotImplementedError), match=r"dtype|bfloat16"):
        add(out=jax.ShapeDtypeStruct((1,), jnp.bfloat16), threads=1)


def test_cpu_lowering_has_clear_error():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with jax.default_device(jax.devices("cpu")[0]), pytest.raises(NotImplementedError, match=r"MPS|mps"):
        jax.jit(lambda x: add(x, out=spec, threads=4)).lower(spec)


def test_vmap_requires_explicit_batching_rule():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises(NotImplementedError, match=r"vmap|batch"):
        jax.make_jaxpr(jax.vmap(lambda x: add(x, out=spec, threads=4)))(np.ones((2, 4), np.float32))


def test_autodiff_requires_user_supplied_rule():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises(NotImplementedError, match=r"differenti|JVP|VJP"):
        jax.make_jaxpr(jax.grad(lambda x: jnp.sum(add(x, out=spec, threads=4))))(np.ones(4, np.float32))


@pytest.mark.parametrize(
    ("dtype", "msl_type"),
    [
        (jnp.float32, "float"),
        (jnp.float16, "half"),
        (jnp.int32, "int"),
        (jnp.uint32, "uint"),
        (jnp.int16, "short"),
        (jnp.uint16, "ushort"),
        (jnp.int8, "char"),
        (jnp.uint8, "uchar"),
        (jnp.bool_, "bool"),
    ],
)
def test_native_eager_copy_preserves_supported_storage(dtype, msl_type, monkeypatch):
    source = f"""
    #include <metal_stdlib>
    using namespace metal;
    kernel void copy_values(device const {msl_type}* x [[buffer(0)]],
                            device {msl_type}* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) {{ out[gid] = x[gid]; }}
    """
    copy = metal.compile(source, entry_point="copy_values")
    with jax.default_device(jax.devices("mps")[0]):
        expected = (np.arange(257) % 19).astype(np.dtype(dtype))
        x = jax.device_put(expected)

        def forbidden_bridge(*args, **kwargs):
            raise AssertionError("native eager execution cannot read JAX arrays through the host bridge")

        monkeypatch.setattr(metal, "_jax_to_numpy", forbidden_bridge)
        result = copy(x, out=jax.ShapeDtypeStruct(x.shape, x.dtype), threads=x.size)
        assert result.devices() == x.devices()
        np.testing.assert_array_equal(np.asarray(result), expected)


_SPLIT_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void split(device const float* x [[buffer(0)]],
                  device float* plus [[buffer(1)]],
                  device float* minus [[buffer(2)]],
                  uint gid [[thread_position_in_grid]]) {
    plus[gid] = x[gid] + 1.0f;
    minus[gid] = x[gid] - 1.0f;
}
"""


def test_native_multiple_outputs_feed_separate_graph_consumers():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")

    @jax.jit
    def call(x):
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        plus, minus = split(x, out=(spec, spec), threads=x.size)
        return plus * 2, minus * 3

    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.arange(33, dtype=jnp.float32)
        plus, minus = call(x)
        np.testing.assert_array_equal(np.asarray(plus), (np.arange(33, dtype=np.float32) + 1) * 2)
        np.testing.assert_array_equal(np.asarray(minus), (np.arange(33, dtype=np.float32) - 1) * 3)


def test_native_dead_output_still_has_distinct_writable_buffer():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")

    @jax.jit
    def call(x):
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        _, minus = split(x, out=(spec, spec), threads=x.size)
        return minus

    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.arange(17, dtype=jnp.float32)
        np.testing.assert_array_equal(np.asarray(call(x)), np.arange(17, dtype=np.float32) - 1)


def test_native_empty_output_with_zero_threads():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void empty(device float* out [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
        out[gid] = 1.0f;
    }
    """
    empty = metal.compile(source, entry_point="empty")
    with jax.default_device(jax.devices("mps")[0]):
        result = empty(out=jax.ShapeDtypeStruct((0, 4), jnp.float32), threads=0)
        assert result.shape == (0, 4)
        assert np.asarray(result).size == 0


def test_native_two_dimensional_thread_counts():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void index2d(device uint* out [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
        out[gid.y * 7 + gid.x] = gid.y * 100 + gid.x;
    }
    """
    indices = metal.compile(source, entry_point="index2d")
    with jax.default_device(jax.devices("mps")[0]):
        result = indices(out=jax.ShapeDtypeStruct((5, 7), jnp.uint32), threads=(7, 5, 1), threadgroup=(4, 2, 1))
        np.testing.assert_array_equal(np.asarray(result), np.arange(5, dtype=np.uint32)[:, None] * 100 + np.arange(7))


def test_native_typed_signed_and_float_constants():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void affine(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                       constant int& bias [[buffer(2)]], constant float& scale [[buffer(3)]],
                       uint gid [[thread_position_in_grid]]) {
        out[gid] = x[gid] * scale + float(bias);
    }
    """
    affine = metal.compile(source, entry_point="affine")
    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.arange(31, dtype=jnp.float32)
        out = affine(
            x,
            out=jax.ShapeDtypeStruct(x.shape, x.dtype),
            threads=x.size,
            constants={2: np.int32(-3), 3: np.float32(0.5)},
        )
        np.testing.assert_array_equal(np.asarray(out), np.arange(31, dtype=np.float32) * 0.5 - 3)


def test_native_repeated_async_calls_keep_inputs_and_dead_outputs_alive():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")
    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.arange(65537, dtype=jnp.float32)
        spec = jax.ShapeDtypeStruct(x.shape, x.dtype)
        for _ in range(16):
            previous = x
            x, dead = split(previous, out=(spec, spec), threads=x.size)
            del previous, dead
        np.testing.assert_array_equal(np.asarray(x), np.arange(65537, dtype=np.float32) + 16)


@pytest.mark.parametrize(
    ("source", "entry", "match"),
    [("not valid Metal source", "broken", "compil|Metal"), (_ADD_SOURCE, "absent", "absent|entry|function")],
)
def test_native_compilation_errors_are_normal_jax_errors(source, entry, match):
    compiled = metal.compile(source, entry_point=entry)
    with jax.default_device(jax.devices("mps")[0]), pytest.raises(jax.errors.JaxRuntimeError, match=match):
        compiled(out=jax.ShapeDtypeStruct((1,), jnp.float32), threads=1)


def test_native_threadgroup_exceeds_pipeline_limit_cleanly():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.ones(4, dtype=jnp.float32)
        with pytest.raises(jax.errors.JaxRuntimeError, match=r"threadgroup|limit|threads"):
            add(x, x, out=jax.ShapeDtypeStruct(x.shape, x.dtype), threads=4, threadgroup=4096, constants={3: 4})


def test_native_reshape_is_row_major_and_does_not_mutate_inputs():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")
    with jax.default_device(jax.devices("mps")[0]):
        original = jnp.arange(24, dtype=jnp.float32)
        x = original.reshape((4, 6))
        plus, minus = split(x, out=(jax.ShapeDtypeStruct(x.shape, x.dtype),) * 2, threads=x.size)
        np.testing.assert_array_equal(np.asarray(original), np.arange(24, dtype=np.float32))
        np.testing.assert_array_equal(np.asarray(plus), (np.arange(24, dtype=np.float32) + 1).reshape(4, 6))
        np.testing.assert_array_equal(np.asarray(minus), (np.arange(24, dtype=np.float32) - 1).reshape(4, 6))


def test_native_kernel_inside_while_has_explicit_error():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")

    @jax.jit
    def call(x):
        def body(value):
            out, _ = split(value, out=(jax.ShapeDtypeStruct(value.shape, value.dtype),) * 2, threads=value.size)
            return out

        return jax.lax.while_loop(lambda value: value[0] < 3, body, x)

    with (
        jax.default_device(jax.devices("mps")[0]),
        pytest.raises(jax.errors.JaxRuntimeError, match=r"nested|control.flow"),
    ):
        call(jnp.zeros(4, dtype=jnp.float32))


def test_dynamic_array_constants_must_be_passed_as_input_buffers():
    add = metal.compile(_ADD_SOURCE, entry_point="add")
    spec = jax.ShapeDtypeStruct((4,), jnp.float32)
    with pytest.raises(TypeError, match=r"static|input buffer"):
        jax.make_jaxpr(lambda value: add(value, value, out=spec, threads=4, constants={3: value[0]}))(
            np.ones(4, np.float32)
        )


def test_native_kernel_rejects_undeclared_shader_buffer_binding():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void extra(device float* out [[buffer(0)]], device const float* missing [[buffer(7)]],
                      uint gid [[thread_position_in_grid]]) { out[gid] = missing[gid]; }
    """
    extra = metal.compile(source, entry_point="extra")
    with jax.default_device(jax.devices("mps")[0]), pytest.raises(jax.errors.JaxRuntimeError, match=r"binding|buffer"):
        extra(out=jax.ShapeDtypeStruct((4,), jnp.float32), threads=4)


def test_native_kernel_cannot_write_input_buffers():
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void mutate(device float* x [[buffer(0)]], device float* out [[buffer(1)]],
                       uint gid [[thread_position_in_grid]]) { x[gid] += 1; out[gid] = x[gid]; }
    """
    mutate = metal.compile(source, entry_point="mutate")
    with jax.default_device(jax.devices("mps")[0]):
        x = jnp.ones(4, dtype=jnp.float32)
        with pytest.raises(jax.errors.JaxRuntimeError, match=r"read.only|input"):
            mutate(x, out=jax.ShapeDtypeStruct(x.shape, x.dtype), threads=x.size)
        np.testing.assert_array_equal(np.asarray(x), np.ones(4, np.float32))


@pytest.mark.parametrize("value", [np.float64(16777217), np.array(16777217, dtype=np.float64)])
def test_typed_float64_constants_are_rejected_without_silent_narrowing(value):
    with pytest.raises(TypeError, match=r"dtype|float64"):
        _native_metal._constants({1: value}, 1)


def test_zero_sized_output_dimensions_must_fit_int64():
    with pytest.raises(ValueError, match=r"dimension|int64"):
        _native_metal._outputs(jax.ShapeDtypeStruct((0, 2**63), np.float32))


@pytest.mark.parametrize(
    ("value", "expected_hex"),
    [
        (np.int16(-3), "fdff"),
        (np.float16(0.5), "0038"),
        (np.uint32(7), "07000000"),
        (2.0, "00000040"),
        (True, "01"),
        (np.array([1, -2], dtype=np.int16), "0100feff"),
        (b"\x01\x02", "0102"),
    ],
)
def test_inline_constants_have_exact_little_endian_bytes(value, expected_hex):
    assert _native_metal._constants({1: value}, 1) == [{"index": 1, "hex": expected_hex}]


def test_native_explicit_input_deletion_and_cache_clearing_keeps_pending_result_safe():
    split = metal.compile(_SPLIT_SOURCE, entry_point="split")
    with jax.default_device(jax.devices("mps")[0]):
        # Inputs from an earlier native launch belong to the output pool. Delete
        # them explicitly, then request same-sized buffers before observing output.
        base = jnp.arange(262144, dtype=jnp.float32)
        spec = jax.ShapeDtypeStruct(base.shape, base.dtype)
        x, discarded = split(base, out=(spec, spec), threads=base.size)
        discarded.delete()
        slow = metal.compile(
            """
            #include <metal_stdlib>
            using namespace metal;
            kernel void slow_copy(volatile device const float* input [[buffer(0)]],
                                  device float* out [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
                float value = 0;
                for (uint i = 0; i < 4096; ++i) value = input[gid];
                out[gid] = value + 1;
            }
            """,
            entry_point="slow_copy",
        )
        result = jax.jit(lambda value: slow(value, out=spec, threads=value.size))(x)
        x.delete()
        jax.clear_caches()
        for _ in range(8):
            reused, unused = split(base, out=(spec, spec), threads=base.size)
            reused.delete()
            unused.delete()
        np.testing.assert_array_equal(np.asarray(result), np.arange(base.size, dtype=np.float32) + 2)


def test_native_encode_failure_does_not_leak_large_output_buffers():
    Metal = pytest.importorskip("Metal")
    source = """
    #include <metal_stdlib>
    using namespace metal;
    kernel void short_input(device const float4& input [[buffer(0)]], device float* out [[buffer(1)]],
                            uint gid [[thread_position_in_grid]]) { out[gid] = input.x; }
    """
    call = metal.compile(source, entry_point="short_input")
    device = Metal.MTLCreateSystemDefaultDevice()
    with jax.default_device(jax.devices("mps")[0]):
        x = jax.device_put(np.ones(1, dtype=np.float32), jax.devices("mps")[0])  # Exactly four host-uploaded bytes.
        spec = jax.ShapeDtypeStruct((2097152,), jnp.float32)

        def invalid():
            with pytest.raises(jax.errors.JaxRuntimeError, match=r"insufficient bytes"):
                call(x, out=spec, threads=1)

        invalid()  # Warm the compiler/pipeline cache before measuring allocations.
        before = device.currentAllocatedSize()
        for _ in range(8):
            invalid()
        after = device.currentAllocatedSize()
        assert after - before < 4 * 1024 * 1024, f"Failed encoding retained {after - before} GPU bytes"
