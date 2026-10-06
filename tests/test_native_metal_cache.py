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

"""Warm-cache transitions must preserve native launch metadata and validation."""

import gc
import json
import weakref
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jax_plugins.silicon import _native_metal, metal

_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]], constant uint& n [[buffer(1)]],
                 uint gid [[thread_position_in_grid]]) { if (gid < n) out[gid] = n; }
"""


@pytest.fixture(autouse=True)
def _clear_preparation_cache():
    cached = getattr(_native_metal, "_prepare_launch", None)
    if cached is not None:
        cached.cache_clear()
    yield
    if cached is not None:
        cached.cache_clear()


def _trace(kernel, *, inputs=(), **kwargs):
    # Trace real primitive metadata without compiling or executing a GPU kernel.
    closed = jax.make_jaxpr(lambda *args: kernel(*args, **kwargs))(*inputs)
    calls = [eqn for eqn in closed.jaxpr.eqns if eqn.primitive is _native_metal._primitive]
    assert len(calls) == 1
    return calls[0].params, closed


def test_repeated_metadata_serializes_only_once(monkeypatch):
    kernel = metal.compile(_SOURCE, entry_point="fill")
    original = _native_metal.json.dumps
    encoded = []

    def record(value, *args, **kwargs):
        if isinstance(value, dict) and value.get("source") == _SOURCE:
            encoded.append(value)
        return original(value, *args, **kwargs)

    monkeypatch.setattr(_native_metal.json, "dumps", record)
    spec = jax.ShapeDtypeStruct((4,), np.uint32)
    first, _ = _trace(kernel, out=spec, threads=4, constants={1: np.uint32(4)})
    second, _ = _trace(kernel, out=spec, threads=4, constants={1: np.uint32(4)})
    assert first == second
    assert len(encoded) == 1


def test_preparation_cache_is_bounded_and_eviction_is_correct():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    first, _ = _trace(kernel, out=spec, threads=1, constants={1: 1})
    for n in range(2, 260):
        _trace(kernel, out=spec, threads=n, constants={1: n})
    cached = _native_metal._prepare_launch.cache_info()
    assert cached.maxsize == 128
    assert cached.currsize <= cached.maxsize
    again, _ = _trace(kernel, out=spec, threads=1, constants={1: 1})
    assert again == first


def test_replacement_spec_changes_shape_and_dtype():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    first, _ = _trace(kernel, out=jax.ShapeDtypeStruct((4,), np.uint32), threads=4)
    second, _ = _trace(kernel, out=jax.ShapeDtypeStruct((8,), np.int32), threads=4)
    assert first["outputs"] == (((4,), "<u4"),)
    assert second["outputs"] == (((8,), "<i4"),)


@pytest.mark.parametrize("shape", [(True,), (1.0,), (0, 2**63), (2**62,)])
def test_warm_cache_does_not_accept_invalid_replacement_shape(shape):
    kernel = metal.compile(_SOURCE, entry_point="fill")
    _trace(kernel, out=jax.ShapeDtypeStruct((1,), np.uint32), threads=1)
    with pytest.raises((TypeError, ValueError), match=r"shape|dimension|byte size|int64"):
        _trace(kernel, out=jax.ShapeDtypeStruct(shape, np.uint32), threads=1)


@pytest.mark.parametrize("dtype", [np.float64, np.int64, np.complex64, jnp.bfloat16])
def test_warm_cache_does_not_accept_unsupported_replacement_dtype(dtype):
    kernel = metal.compile(_SOURCE, entry_point="fill")
    _trace(kernel, out=jax.ShapeDtypeStruct((1,), np.uint32), threads=1)
    with pytest.raises(TypeError, match=r"dtype"):
        _trace(kernel, out=jax.ShapeDtypeStruct((1,), dtype), threads=1)


@pytest.mark.parametrize("value", [True, np.bool_(True), 1.0, np.float64(1)])
@pytest.mark.parametrize("field", ["threads", "threadgroup"])
def test_warm_cache_rejects_equal_but_invalid_geometry(field, value):
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    options = {"out": spec, "threads": 1, "threadgroup": 1}
    _trace(kernel, **options)
    options[field] = value
    with pytest.raises((TypeError, ValueError), match=field):
        _trace(kernel, **options)


def test_mutable_constant_dict_and_array_are_snapshotted_by_value():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((4,), np.uint32)
    value = np.array(4, dtype=np.uint32)
    constants = {1: value}
    first, _ = _trace(kernel, out=spec, threads=4, constants=constants)
    value[...] = 3
    second, _ = _trace(kernel, out=spec, threads=4, constants=constants)
    constants[1] = np.uint32(2)
    third, _ = _trace(kernel, out=spec, threads=4, constants=constants)
    assert [json.loads(result["backend_config"])["constants"][0]["hex"] for result in (first, second, third)] == [
        "04000000",
        "03000000",
        "02000000",
    ]


def test_same_constant_bytes_with_unsupported_mutated_dtype_are_rejected():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    value = np.array([0, 0], dtype=np.uint32)
    _trace(kernel, out=spec, threads=1, constants={1: value})
    original = value.tobytes()
    value.dtype = np.float64
    assert value.tobytes() == original
    with pytest.raises(TypeError, match=r"dtype|float64"):
        _trace(kernel, out=spec, threads=1, constants={1: value})


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (True, "01"),
        (1, "01000000"),
        (np.int16(1), "0100"),
        (np.uint32(1), "01000000"),
        (0.0, "00000000"),
        (-0.0, "00000080"),
    ],
)
def test_equal_values_preserve_typed_and_signed_zero_constant_bits(value, expected):
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    _trace(kernel, out=spec, threads=1, constants={1: 1})
    params, _ = _trace(kernel, out=spec, threads=1, constants={1: value})
    assert json.loads(params["backend_config"])["constants"] == [{"index": 1, "hex": expected}]


def test_valid_float_constant_does_not_cache_equal_invalid_float64():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    _trace(kernel, out=spec, threads=1, constants={1: 1.0})
    with pytest.raises(TypeError, match=r"dtype|float64"):
        _trace(kernel, out=spec, threads=1, constants={1: np.float64(1)})


def test_warm_cache_revalidates_buffer_count_overlap_and_slot_types():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    _trace(kernel, out=spec, threads=1, constants={1: 1})
    with pytest.raises(ValueError, match=r"overlap"):
        _trace(kernel, out=(spec, spec), threads=1, constants={1: 1})
    with pytest.raises(ValueError, match=r"overlap"):
        _trace(kernel, inputs=(np.ones(1, np.uint32),), out=spec, threads=1, constants={1: 1})
    with pytest.raises(TypeError, match=r"indices"):
        _trace(kernel, out=spec, threads=1, constants={True: 1})
    with pytest.raises(ValueError, match=r"31"):
        _trace(kernel, out=(spec,) * 32, threads=1)


def test_single_and_sequence_output_structure_is_not_cached_by_equality():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    single = jax.eval_shape(lambda: kernel(out=spec, threads=1))
    listed = jax.eval_shape(lambda: kernel(out=[spec], threads=1))
    tupled = jax.eval_shape(lambda: kernel(out=(spec,), threads=1))
    assert isinstance(single, jax.ShapeDtypeStruct)
    assert listed == (single,) and tupled == (single,)


def test_cached_metadata_is_isolated_by_kernel_source_and_entry_point():
    first = metal.compile(_SOURCE, entry_point="fill")
    second = metal.compile(_SOURCE.replace("out[gid] = n", "out[gid] = n + 1"), entry_point="fill")
    third = metal.compile(_SOURCE.replace("void fill", "void fill_other"), entry_point="fill_other")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    configs = [_trace(kernel, out=spec, threads=1)[0]["backend_config"] for kernel in (first, second, third)]
    assert len(set(configs)) == 3


def test_warm_cache_still_rejects_dynamic_constants():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    _trace(kernel, out=spec, threads=1, constants={1: 1})
    with pytest.raises(TypeError, match=r"static|input buffer"):
        jax.make_jaxpr(lambda n: kernel(out=spec, threads=1, constants={1: n}))(np.uint32(1))


def test_warm_cache_still_rejects_host_inputs_and_unsupported_input_dtype():
    kernel = metal.compile(_SOURCE, entry_point="fill")
    spec = jax.ShapeDtypeStruct((1,), np.uint32)
    _trace(kernel, out=spec, threads=1)
    with pytest.raises(TypeError, match=r"JAX arrays"):
        kernel(np.ones(1, np.uint32), out=spec, threads=1)
    with pytest.raises(TypeError, match=r"dtype"):
        _trace(kernel, inputs=(np.ones(1, np.complex64),), out=spec, threads=1)


_COPY_SOURCE = """
#include <metal_stdlib>
using namespace metal;
kernel void copy_values(device const float* x [[buffer(0)]], device float* out [[buffer(1)]],
                        uint gid [[thread_position_in_grid]]) { out[gid] = x[gid]; }
"""


def test_repeated_eager_launch_reuses_dispatch_not_results(monkeypatch):
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    original = _native_metal._primitive.bind
    binds = []

    def counted(*args, **kwargs):
        binds.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(_native_metal._primitive, "bind", counted)
    device = jax.devices("mps")[0]
    x = jax.device_put(np.arange(4, dtype=np.float32), device)
    first = kernel(x, out=spec, threads=4)
    first.block_until_ready()
    warm_binds = len(binds)
    y = jax.device_put(np.arange(4, dtype=np.float32) + 10, device)
    second = kernel(y, out=spec, threads=4)
    second.block_until_ready()
    assert len(binds) == warm_binds
    assert first is not second
    np.testing.assert_array_equal(np.asarray(first), np.arange(4, dtype=np.float32))
    np.testing.assert_array_equal(np.asarray(second), np.arange(4, dtype=np.float32) + 10)


@pytest.mark.parametrize("warm", [False, True])
def test_cached_eager_dispatch_works_with_disable_jit(warm):
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    x = jax.device_put(np.arange(4, dtype=np.float32), jax.devices("mps")[0])
    if warm:
        kernel(x, out=spec, threads=4).block_until_ready()
    with jax.disable_jit():
        result = kernel(x, out=spec, threads=4)
        np.testing.assert_array_equal(np.asarray(result), np.arange(4, dtype=np.float32))


def test_warmed_eager_dispatch_composes_with_outer_jit():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    x = jax.device_put(np.arange(4, dtype=np.float32), jax.devices("mps")[0])
    kernel(x, out=spec, threads=4).block_until_ready()
    compiled = jax.jit(lambda value: kernel(value + 1, out=spec, threads=4) * 2)
    np.testing.assert_array_equal(np.asarray(compiled(x)), (np.arange(4, dtype=np.float32) + 1) * 2)


def test_warmed_eager_dispatch_does_not_freeze_mps_placement():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    values = np.arange(4, dtype=np.float32)
    kernel(jax.device_put(values, jax.devices("mps")[0]), out=spec, threads=4).block_until_ready()
    with pytest.raises(NotImplementedError, match=r"MPS|mps"):
        kernel(jax.device_put(values, jax.devices("cpu")[0]), out=spec, threads=4)


@pytest.mark.parametrize("transform", [jax.grad, jax.vmap])
def test_warmed_eager_dispatch_preserves_transform_rejection(transform):
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    kernel(jax.device_put(np.ones(4, np.float32), jax.devices("mps")[0]), out=spec, threads=4).block_until_ready()

    def fn(value):
        return jnp.sum(kernel(value, out=spec, threads=4))

    values = np.ones((2, 4) if transform is jax.vmap else (4,), np.float32)
    with pytest.raises(NotImplementedError, match=r"differenti|JVP|VJP|vmap|batch"):
        jax.make_jaxpr(jax.jit(transform(fn)))(values)


def test_preparation_cache_does_not_retain_input_or_result_arrays():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    x = jax.device_put(np.arange(4, dtype=np.float32), jax.devices("mps")[0])
    result = kernel(x, out=spec, threads=4)
    result.block_until_ready()
    input_ref, output_ref = weakref.ref(x), weakref.ref(result)
    result.delete()
    x.delete()
    del result, x
    gc.collect()
    assert input_ref() is None and output_ref() is None
    assert _native_metal._prepare_launch.cache_info().currsize == 1


def test_prepared_dispatch_is_safe_for_concurrent_eager_calls():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    device = jax.devices("mps")[0]
    inputs = [jax.device_put(np.full(4, i, np.float32), device) for i in range(8)]
    kernel(inputs[0], out=spec, threads=4).block_until_ready()

    def call(value):
        result = kernel(value, out=spec, threads=4)
        return np.asarray(result)

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(call, inputs))
    for i, value in enumerate(results):
        np.testing.assert_array_equal(value, np.full(4, i, np.float32))


def test_clearing_preparation_and_jax_caches_preserves_returned_results():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((262144,), np.float32)
    expected = np.arange(262144, dtype=np.float32)
    x = jax.device_put(expected, jax.devices("mps")[0])
    result = kernel(x, out=spec, threads=expected.size)
    _native_metal._prepare_launch.cache_clear()
    jax.clear_caches()
    x.delete()
    np.testing.assert_array_equal(np.asarray(result), expected)


def test_warmed_eager_dispatch_keeps_nested_control_flow_rejection():
    kernel = metal.compile(_COPY_SOURCE, entry_point="copy_values")
    spec = jax.ShapeDtypeStruct((4,), np.float32)
    x = jax.device_put(np.ones(4, np.float32), jax.devices("mps")[0])
    kernel(x, out=spec, threads=4).block_until_ready()

    @jax.jit
    def call(value):
        def body(state):
            i, array = state
            return i + 1, kernel(array, out=spec, threads=4)

        return jax.lax.while_loop(lambda state: state[0] < 2, body, (0, value))[1]

    with pytest.raises(jax.errors.JaxRuntimeError, match=r"nested|control.flow"):
        call(x)
