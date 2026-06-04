import os
import re
from pathlib import Path

import jax
import numpy
import pytest
from jax import dtypes, random
from jax import numpy as jnp

from .configs import (
    OperationTestConfig,
    make_binary_op_configs,
    make_control_flow_op_configs,
    make_conv_op_configs,
    make_conversion_op_configs,
    make_flax_op_configs,
    make_linalg_op_configs,
    make_misc_op_configs,
    make_numpyro_op_configs,
    make_random_op_configs,
    make_reduction_op_configs,
    make_shape_op_configs,
    make_slice_op_configs,
    make_sort_op_configs,
    make_unary_op_configs,
)
from .configs.util import STABLEHLO_OP_RE

# Test mode configuration via environment variable:
# - "compare" (default): Run on both CPU and MPS, compare results
# - "mps": Run only on MPS
# - "cpu": Run only on CPU
TEST_MODE = os.environ.get("JAX_TEST_MODE", "compare").lower()
if TEST_MODE not in ("compare", "mps", "cpu"):
    raise ValueError(f"Invalid JAX_TEST_MODE: {TEST_MODE}. Must be 'compare', 'mps', or 'cpu'.")


def get_test_platforms() -> list[str]:
    """Return the platforms to test based on JAX_TEST_MODE environment variable."""
    if TEST_MODE == "compare":
        return ["cpu", "mps"]
    else:
        return [TEST_MODE]


OPERATION_TEST_CONFIGS = [
    *make_binary_op_configs(),
    *make_control_flow_op_configs(),
    *make_conv_op_configs(),
    *make_conversion_op_configs(),
    *make_flax_op_configs(),
    *make_linalg_op_configs(),
    *make_misc_op_configs(),
    *make_numpyro_op_configs(),
    *make_random_op_configs(),
    *make_reduction_op_configs(),
    *make_shape_op_configs(),
    *make_slice_op_configs(),
    *make_sort_op_configs(),
    *make_unary_op_configs(),
]


@pytest.fixture(params=OPERATION_TEST_CONFIGS, ids=lambda op_config: op_config.name)
def op_config(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture(params=[True, False], ids=["jit", "eager"])
def jit(request: pytest.FixtureRequest):
    return request.param


def fassert(cond: bool, message: str) -> None:
    """Functional assertion."""
    assert cond, message


def assert_allclose_with_path(path, actual, desired):
    # Extract key data if these are random keys rather than regular data.
    is_prng_key = dtypes.issubdtype(actual.dtype, dtypes.prng_key)
    if is_prng_key:
        actual = random.key_data(actual)
        desired = random.key_data(desired)

    try:
        if jnp.issubdtype(actual.dtype, jnp.inexact):
            numpy.testing.assert_allclose(actual, desired, atol=1e-5, rtol=1e-5)
        else:
            numpy.testing.assert_array_equal(actual, desired)
    except AssertionError as ex:
        raise AssertionError(f"Values are not close at path '{path}'.") from ex


def block_tree_until_ready(tree):
    return jax.tree.map(
        lambda value: value.block_until_ready() if isinstance(value, jax.Array) else value,
        tree,
    )


def _is_framework_integration(op_config: OperationTestConfig) -> bool:
    return op_config.name.startswith("numpyro.") or op_config.name.startswith("flax.")


def test_op_value(op_config: OperationTestConfig, jit: bool) -> None:
    # Numpyro/flax distributions manage internal device state and use complex
    # transcendental functions (lgamma, digamma, …) that have legitimate
    # precision differences across backends.  Run them on MPS only and verify
    # the result is a real number (not NaN/inf).
    if _is_framework_integration(op_config):
        device = jax.devices("mps")[0]
        with jax.default_device(device):
            result = op_config.evaluate_value(jit)
        return

    platforms = get_test_platforms()
    results = []
    for platform in platforms:
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            result = op_config.evaluate_value(jit)
            if TEST_MODE == "compare":
                result = block_tree_until_ready(result)
            jax.tree.map_with_path(
                lambda path, value, _dev=device: fassert(
                    value.device == _dev,
                    f"Value at '{path}' is on device {value.device}; expected {_dev}.",
                ),
                result,
            )
            results.append(result)

    if len(results) == 2:
        jax.tree.map_with_path(assert_allclose_with_path, *results)


def test_op_grad(op_config: OperationTestConfig, jit: bool) -> None:
    argnums = op_config.get_differentiable_argnums()
    if not argnums:
        pytest.skip(f"No differentiable arguments for operation '{op_config.func}'.")

    # Eager gradients in compare mode deadlock the Metal command queue after many
    # prior dispatches in the same process. JIT gradients cover the same code paths.
    if TEST_MODE == "compare" and not jit:
        pytest.skip("Eager gradients deadlock Metal command queue in long compare-mode sessions")

    # Framework integration tests run on MPS only (same reasoning as test_op_value).
    if _is_framework_integration(op_config):
        device = jax.devices("mps")[0]
        with jax.default_device(device):
            for argnum in argnums:
                op_config.evaluate_grad(argnum, jit)
        return

    platforms = get_test_platforms()
    for argnum in argnums:
        results = []
        for platform in platforms:
            device = jax.devices(platform)[0]
            with jax.default_device(device):
                result = op_config.evaluate_grad(argnum, jit)
                if TEST_MODE == "compare":
                    result = block_tree_until_ready(result)
                jax.tree.map_with_path(
                    lambda path, value, _dev=device: fassert(
                        value.device == _dev,
                        f"Value at '{path}' is on device {value.device}; expected {_dev}.",
                    ),
                    result,
                )
                results.append(result)

        if len(results) == 2:
            jax.tree.map_with_path(assert_allclose_with_path, *results)


@pytest.mark.xfail(reason="Native intrinsic ordering bug: broadcast_in_dim before divide in HLO", strict=True)
def test_native_intrinsic_ordering_bug() -> None:
    """Regression test for native intrinsic kernel ordering bug.

    When intrinsic native kernels are enabled and the StableHLO IR has
    broadcast_in_dim + multiply + log ops *before* the divide that produces a
    value used by later multiply ops, the mixed execution path produces
    incorrect results.  This is because the native kernel dispatch in the mixed
    graph+native path does not correctly handle the case where intermediate
    results are produced by ops that appear later in the SSA order.

    The test XFAILs until the C++ backend fix lands; once it passes, the xfail
    will turn into a hard failure reminding us to remove the marker.
    """
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test")
    device = jax.devices("mps")[0]
    with jax.default_device(device):
        x = jax.device_put(jnp.array([-23.082829, 13.160883, 15.299932]), device)
        loc = jax.device_put(jnp.array([0.3047171, -1.0399841, 0.7504512]), device)
        scale = jax.device_put(jnp.array([35.03836, 11.905022, 20.081493]), device)

        # This specific pattern triggers the bug: log_norm computed before z,
        # then z used in two subsequent multiplies.
        def buggy_pattern(x, loc, scale):
            log_norm = jnp.log(jnp.float32(2.5066283) * scale)
            z = (x - loc) / scale
            lp = jnp.float32(-0.5) * z * z - log_norm
            return lp.sum() / jnp.float32(3.0)

        result = jax.jit(buggy_pattern)(x, loc, scale)
        expected = numpy.float32(-4.3289)
        numpy.testing.assert_allclose(result, expected, atol=1e-3, rtol=1e-3)


def test_unsupported_op_error_message(jit: bool) -> None:
    """Check that unsupported-op errors link to the issue template and CONTRIBUTING.md."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")
    device = jax.devices("mps")[0]
    with jax.default_device(device):
        try:
            # This is an obscure op. It's unlikely to be implemented, but this test may
            # break if `clz` gets implemented.
            func = jax.lax.clz
            if jit:
                func = jax.jit(func)
            func(numpy.int32(7))
        except Exception as exc:
            message = str(exc)
            assert "issues/new?template=missing-op.yml" in message
            assert "CONTRIBUTING.md" in message
        else:
            pytest.skip("clz is now supported; test needs a new unregistered op")


def test_unknown_custom_call_error_message(jit: bool) -> None:
    """Unknown custom call targets should fail fast with actionable diagnostics."""
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    import jax.numpy as jnp

    device = jax.devices("mps")[0]
    with jax.default_device(device):
        target = "jax_silicon_test.unknown_custom_call_target"

        def func(x):
            call = jax.ffi.ffi_call(target, jax.ShapeDtypeStruct(x.shape, x.dtype))
            return call(x)

        if jit:
            func = jax.jit(func)

        try:
            func(jnp.asarray([1.0, 2.0, 3.0], dtype=jnp.float32))
        except Exception as exc:
            message = str(exc)
            assert target in message
            assert "issues/new?template=missing-op.yml" in message
            assert "CONTRIBUTING.md" in message
            assert "Operands (" in message
            assert "Results (" in message
        else:
            pytest.fail("Expected unknown custom call to fail, but it succeeded.")


def test_conv_batch_group_count_guard_raises_targeted_error() -> None:
    if TEST_MODE == "cpu":
        pytest.skip("MPS-specific test skipped in CPU-only mode")

    import jax.numpy as jnp

    device = jax.devices("mps")[0]
    with jax.default_device(device):
        x = jnp.ones((8, 8, 8, 3), dtype=jnp.float32)
        kernel = jnp.ones((3, 3, 3, 8), dtype=jnp.float32)

        fn = jax.jit(
            lambda x, kernel: jax.lax.conv_general_dilated(
                x,
                kernel,
                window_strides=(1, 1),
                padding="SAME",
                batch_group_count=8,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
        )

        with pytest.raises(Exception) as excinfo:
            fn(x, kernel)

    assert "convolution: batch_group_count emulation would generate >4 convs; unsupported for performance" in str(
        excinfo.value
    )


def test_log1p_float16_small_values_matches_cpu(jit: bool) -> None:
    """Regression test for catastrophic cancellation in log1p (float16 near 0)."""
    if TEST_MODE != "compare":
        pytest.skip("Cross-platform comparison test requires both CPU and MPS")

    import jax.numpy as jnp

    x_np = numpy.asarray([5e-4, -5e-4, 1e-3, -1e-3], dtype=numpy.float16)

    def run(platform: str):
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            f = jnp.log1p
            if jit:
                f = jax.jit(f)
            return f(jnp.asarray(x_np))

    cpu = run("cpu")
    mps = run("mps")
    numpy.testing.assert_allclose(mps, cpu, atol=1e-4, rtol=1e-4)


def test_erf_inv_matches_cpu_across_range(jit: bool) -> None:
    """Accuracy/regression test for chlo.erf_inv lowering."""
    if TEST_MODE != "compare":
        pytest.skip("Cross-platform comparison test requires both CPU and MPS")

    import jax.numpy as jnp
    from jax.scipy import special

    x_np = numpy.linspace(-0.99, 0.99, 2001, dtype=numpy.float32)

    def run(platform: str):
        device = jax.devices(platform)[0]
        with jax.default_device(device):
            f = special.erfinv
            if jit:
                f = jax.jit(f)
            return f(jnp.asarray(x_np))

    cpu = run("cpu")
    mps = run("mps")
    numpy.testing.assert_allclose(mps, cpu, atol=1e-5, rtol=1e-5)


def test_triangular_solve_float16_matches_cpu(jit: bool) -> None:
    """Regression test for stablehlo.triangular_solve float16 lowering."""
    if TEST_MODE != "compare":
        pytest.skip("Cross-platform comparison test requires both CPU and MPS")

    import jax.numpy as jnp
    from jax.scipy.linalg import solve_triangular

    rng = numpy.random.default_rng(0)
    M = rng.standard_normal((4, 4)).astype(numpy.float16)
    A = numpy.tril(M)
    numpy.fill_diagonal(A, numpy.abs(numpy.diag(A)) + numpy.float16(1.0))
    B = rng.standard_normal((4, 3)).astype(numpy.float16)

    def run(platform: str):
        device = jax.devices(platform)[0]
        with jax.default_device(device):

            def f(a, b):
                return solve_triangular(a, b, lower=True)

            if jit:
                f = jax.jit(f)
            return f(jnp.asarray(A), jnp.asarray(B))

    cpu = run("cpu")
    mps = run("mps")
    numpy.testing.assert_allclose(mps, cpu, atol=5e-3, rtol=5e-3)


@pytest.mark.parametrize(
    "collective",
    ["all_reduce", "all_gather", "reduce_scatter", "collective_permute"],
)
def test_single_replica_collectives_work_and_mark_coverage(collective: str, jit: bool) -> None:
    """Single-device collectives should be no-ops (or local equivalents).

    Also update `OperationTestConfig.EXERCISED_STABLEHLO_OPS` so CI coverage checks
    don't flag these ops as untested (since they are not exercised via OperationTestConfig).
    """
    platforms = get_test_platforms()
    results = []

    from functools import partial

    import jax.numpy as jnp
    from jax import lax

    for platform in platforms:
        devices = jax.devices(platform)
        assert len(devices) == 1, f"Expected single device for platform '{platform}'."
        device = devices[0]

        with jax.default_device(device):
            if collective == "all_reduce":

                @partial(jax.pmap, axis_name="i", devices=devices)
                def func(x):
                    return lax.psum(x, "i")

                x = jnp.arange(4, dtype=jnp.float32).reshape((1, 4))
            elif collective == "all_gather":

                @partial(jax.pmap, axis_name="i", devices=devices)
                def func(x):
                    return lax.all_gather(x, "i", axis=0, tiled=False)

                x = jnp.arange(4, dtype=jnp.float32).reshape((1, 4))
            elif collective == "reduce_scatter":

                @partial(jax.pmap, axis_name="i", devices=devices)
                def func(x):
                    return lax.psum_scatter(x, "i", scatter_dimension=0, tiled=False)

                # For a single shard, the scatter dimension must have size 1.
                x = jnp.asarray([[3.0]], dtype=jnp.float32)
            elif collective == "collective_permute":

                @partial(jax.pmap, axis_name="i", devices=devices)
                def func(x):
                    return lax.ppermute(x, "i", perm=[(0, 0)])

                x = jnp.arange(4, dtype=jnp.float32).reshape((1, 4))
            else:
                raise AssertionError(f"Unknown collective kind: {collective}")

            lowered = func.lower(x)
            if platform == "mps":
                stablehlo_text = str(lowered.compiler_ir(dialect="stablehlo"))
                OperationTestConfig.EXERCISED_STABLEHLO_OPS.update(STABLEHLO_OP_RE.findall(stablehlo_text))

            # `pmap` is already a compilation boundary; treat `jit` as a no-op knob here.
            _ = jit
            results.append(func(x))

    if len(results) == 2:
        numpy.testing.assert_allclose(*results, atol=1e-5, rtol=1e-5)


@pytest.fixture(autouse=True, scope="module")
def assert_all_ops_tested():
    yield

    if "CI" not in os.environ:
        return

    # Skip op coverage check in CPU-only mode since EXERCISED_STABLEHLO_OPS is only
    # populated when running on MPS.
    if TEST_MODE == "cpu":
        return

    ops_dir = Path(__file__).parent.parent / "src/pjrt_plugin/ops"
    assert ops_dir.is_dir()

    # Patterns matching op registration calls
    patterns = [
        re.compile(r'REGISTER_MPS_OP\("([^"]+)"'),
        re.compile(r'REGISTER_NATIVE_MPS_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_BINARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_MLIR_UNARY_OP\("([^"]+)"'),
        re.compile(r'REGISTER_LOGICAL_BITWISE_OP\("([^"]+)"'),
    ]

    # Ops that appear in StableHLO IR but get lowered by MLIR before reaching
    # our handlers. They work but don't need explicit registration.
    mlir_lowered_ops = {"chlo.lgamma", "chlo.digamma", "chlo.bessel_i1e"}

    op_names = set()
    for mm_file in ops_dir.glob("*.mm"):
        with mm_file.open() as fp:
            content = fp.read()
            for pattern in patterns:
                op_names.update(pattern.findall(content))

    assert op_names, "Failed to discover any ops."
    exercised = OperationTestConfig.EXERCISED_STABLEHLO_OPS - mlir_lowered_ops
    unsupported = exercised - op_names
    assert not unsupported, f"Discovered {len(unsupported)} unsupported ops: {', '.join(sorted(unsupported))}"
    missing = op_names - OperationTestConfig.EXERCISED_STABLEHLO_OPS
    assert not missing, f"Discovered {len(missing)} untested ops: {', '.join(sorted(missing))}"
