import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_reduction_op_configs():
    for is_complex in [True, False]:
        with OperationTestConfig.module_name("reduction-complex" if is_complex else "reduction-real"):
            for reduction in [jnp.sum, jnp.mean, jnp.var, jnp.std]:
                yield from [
                    OperationTestConfig(
                        reduction,
                        lambda rng, is_complex=is_complex: complex_standard_normal(rng, (4, 5), is_complex),
                    ),
                    # Explicit argument because capture doesn't work.
                    OperationTestConfig(
                        lambda x, reduction=reduction: reduction(x, axis=1),
                        lambda rng, is_complex=is_complex: complex_standard_normal(rng, (4, 5), is_complex),
                    ),
                ]

        with OperationTestConfig.module_name("reduction-real"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.max(x, axis=0),
                    lambda rng: rng.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.min(x, axis=-1),
                    lambda rng: rng.standard_normal((4, 5)),
                    differentiable_argnums=(),
                ),
            ]

        with OperationTestConfig.module_name("reduction"):
            yield from [
                OperationTestConfig(
                    lambda x: jnp.cumsum(x, axis=0),
                    lambda rng: rng.standard_normal((8,)).astype("float32"),
                    name="cumsum_1d_axis0",
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.cumsum(x, axis=0),
                    lambda rng: rng.standard_normal((3, 4)).astype("float32"),
                    name="cumsum_2d_axis0",
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: jnp.cumsum(x, axis=1),
                    lambda rng: rng.standard_normal((3, 4)).astype("float32"),
                    name="cumsum_2d_axis1",
                    differentiable_argnums=(),
                ),
                # Pooling-like reduce_window patterns (non-cumulative):
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        0.0,
                        lax.add,
                        window_dimensions=(1, 3, 3, 1),
                        window_strides=(1, 2, 2, 1),
                        padding=((0, 0), (1, 2), (2, 1), (0, 0)),
                    ),
                    lambda rng: rng.standard_normal((1, 7, 8, 2)).astype(numpy.float32),
                    name="reduce_window_sum_pool_2d_stride2_padding",
                    differentiable_argnums=(),
                ),
                # 1D pooling-like reduce_window on NWC.
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        0.0,
                        lax.add,
                        window_dimensions=(1, 3, 1),
                        window_strides=(1, 2, 1),
                        padding=((0, 0), (1, 2), (0, 0)),
                    ),
                    lambda rng: rng.standard_normal((1, 9, 2)).astype(numpy.float32),
                    name="reduce_window_sum_pool_1d_stride2_padding",
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 2, 1),
                        window_strides=(1, 2, 2, 1),
                        padding="SAME",
                    ),
                    lambda rng: rng.standard_normal((1, 8, 8, 1)).astype(numpy.float32),
                    name="reduce_window_max_pool_2d_same",
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 1),
                        window_strides=(1, 2, 1),
                        padding="SAME",
                    ),
                    lambda rng: rng.standard_normal((1, 9, 1)).astype(numpy.float32),
                    name="reduce_window_max_pool_1d_same",
                    differentiable_argnums=(),
                ),
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        jnp.inf,
                        lax.min,
                        window_dimensions=(1, 3, 1),
                        window_strides=(1, 2, 1),
                        padding=((0, 0), (1, 1), (0, 0)),
                    ),
                    lambda rng: rng.standard_normal((1, 8, 1)).astype(numpy.float32),
                    name="reduce_window_min_pool_1d_explicit",
                    differentiable_argnums=(),
                ),
                # Gradient of max pooling lowers to stablehlo.select_and_scatter.
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 2, 1),
                        window_strides=(1, 2, 2, 1),
                        padding="SAME",
                    ).sum(),
                    lambda rng: rng.standard_normal((1, 8, 8, 1)).astype(numpy.float32),
                    name="reduce_window_max_pool_2d_grad",
                ),
                # Rank-3 NWC max-pool gradient (select_and_scatter NWC lowering).
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 1),
                        window_strides=(1, 2, 1),
                        padding="SAME",
                    ).sum(),
                    lambda rng: rng.standard_normal((1, 9, 1)).astype(numpy.float32),
                    name="reduce_window_max_pool_1d_grad",
                ),
                # Window dilations for pooling-like reduce_window.
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        -jnp.inf,
                        lax.max,
                        window_dimensions=(1, 2, 2, 1),
                        window_strides=(1, 2, 2, 1),
                        padding="SAME",
                        window_dilation=(1, 2, 2, 1),
                    ),
                    lambda rng: rng.standard_normal((1, 8, 8, 1)).astype(numpy.float32),
                    name="reduce_window_max_pool_2d_same_window_dilation",
                    differentiable_argnums=(),
                ),
                # Min-pool gradient (select_and_scatter LT/LE lowering).
                OperationTestConfig(
                    lambda x: lax.reduce_window(
                        x,
                        jnp.inf,
                        lax.min,
                        window_dimensions=(1, 2, 2, 1),
                        window_strides=(1, 2, 2, 1),
                        padding="SAME",
                    ).sum(),
                    lambda rng: rng.standard_normal((1, 8, 8, 1)).astype(numpy.float32),
                    name="reduce_window_min_pool_2d_grad",
                ),
            ]
