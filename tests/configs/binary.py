import numpy
from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_binary_op_configs():
    for is_complex in [False, True]:
        with OperationTestConfig.module_name("binary-complex" if is_complex else "binary-real"):
            yield from [
                OperationTestConfig(
                    jnp.add,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 1), is_complex),
                    differentiable_argnums=(0, 1) if is_complex else None,
                ),
                OperationTestConfig(
                    jnp.subtract,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 1), is_complex),
                    differentiable_argnums=(0, 1) if is_complex else None,
                ),
                OperationTestConfig(
                    jnp.multiply,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 1), is_complex),
                    differentiable_argnums=(0, 1) if is_complex else None,
                ),
                OperationTestConfig(
                    jnp.divide,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 1), is_complex),
                    differentiable_argnums=(0, 1) if is_complex else None,
                ),
                OperationTestConfig(
                    jnp.dot,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (4, 5), is_complex),
                    differentiable_argnums=(0, 1),
                ),
                OperationTestConfig(
                    jnp.less,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                ),
                OperationTestConfig(
                    jnp.less_equal,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                ),
                OperationTestConfig(
                    jnp.equal,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                ),
                OperationTestConfig(
                    jnp.greater_equal,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                ),
                OperationTestConfig(
                    jnp.greater,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                ),
                OperationTestConfig(
                    jnp.broadcast_arrays,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 1), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    differentiable_argnums=(0, 1) if is_complex else None,
                ),
                OperationTestConfig(
                    jnp.minimum,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    differentiable_argnums=(0, 1),
                ),
                OperationTestConfig(
                    jnp.maximum,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (3, 4), is_complex),
                    differentiable_argnums=(0, 1),
                ),
                OperationTestConfig(
                    jnp.power,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (5,), is_complex),
                    lambda rng: rng.gamma(5, size=(7, 1)),
                ),
                OperationTestConfig(
                    jnp.power,
                    lambda rng: rng.gamma(5, size=(7, 1)),
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (5,), is_complex),
                ),
            ]

        # Ops that do not support complex arguments, typically because they require an
        # order to be defined.
        yield from [
            OperationTestConfig(
                lax.rem,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.gamma(5, size=(3, 4)),
            ),
            OperationTestConfig(
                jnp.nextafter,
                numpy.array([1.0, -1.0, 0.0, 2.0], dtype=numpy.float32),
                numpy.array([2.0, -2.0, 1.0, 1.0], dtype=numpy.float32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                differentiable_argnums=(0,),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                None,
                lambda rng: rng.standard_normal((3, 4)),
                differentiable_argnums=(0,),
            ),
            OperationTestConfig(
                jnp.clip,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                None,
                differentiable_argnums=(0,),
            ),
            OperationTestConfig(
                lax.clamp,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.arctan2,
                lambda rng: rng.standard_normal((3, 4)),
                lambda rng: rng.standard_normal((3, 4)),
            ),
            OperationTestConfig(
                jnp.bitwise_and,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_xor,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_or,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.int32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.bitwise_and,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_and-uint32",
            ),
            OperationTestConfig(
                jnp.bitwise_xor,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_xor-uint32",
            ),
            OperationTestConfig(
                jnp.bitwise_or,
                numpy.array([0, 1, 3, 7, 15], dtype=numpy.uint32),
                numpy.array([1, 3, 7, 15, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
                name="bitwise_or-uint32",
            ),
            OperationTestConfig(
                jnp.logical_and,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.logical_or,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.logical_xor,
                numpy.array([True, False, True, False]),
                numpy.array([True, True, False, False]),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_left,
                numpy.array([1, 2, 4, 8], dtype=numpy.int32),
                numpy.array([0, 1, 2, 3], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                jnp.right_shift,
                numpy.array([-8, -1, 8, 127], dtype=numpy.int32),
                numpy.array([1, 31, 2, 40], dtype=numpy.int32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_right_logical,
                numpy.array([1, 2, 4, 8, 0x80000000], dtype=numpy.uint32),
                numpy.array([0, 1, 2, 3, 31], dtype=numpy.uint32),
                differentiable_argnums=(),
            ),
            OperationTestConfig(
                lax.shift_left,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_left-negative-shift-count",
            ),
            OperationTestConfig(
                lax.shift_right_logical,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_right_logical-negative-shift-count",
            ),
            OperationTestConfig(
                lax.shift_right_arithmetic,
                numpy.array([0x40000000, -0x40000000, 5], dtype=numpy.int32),
                numpy.array([-2, -1, -31], dtype=numpy.int32),
                differentiable_argnums=(),
                name="lax.shift_right_arithmetic-negative-shift-count",
            ),
        ]

    with OperationTestConfig.module_name("dot_general"):
        yield from [
            # [M,K] x [K,N] -> [M,N]
            OperationTestConfig(
                lambda lhs, rhs: lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                ),
                lambda rng: rng.normal(size=(3, 4)).astype(numpy.float32),
                lambda rng: rng.normal(size=(4, 5)).astype(numpy.float32),
                name="matmul_rank2",
            ),
            # [B,M,K] x [B,K,N] -> [B,M,N]
            OperationTestConfig(
                lambda lhs, rhs: lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((2,), (1,)), ((0,), (0,))),
                ),
                lambda rng: rng.normal(size=(2, 3, 4)).astype(numpy.float32),
                lambda rng: rng.normal(size=(2, 4, 5)).astype(numpy.float32),
                name="bmm_rank3",
            ),
            # [K] x [K,N] -> [N]
            OperationTestConfig(
                lambda lhs, rhs: lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((0,), (0,)), ((), ())),
                ),
                lambda rng: rng.normal(size=(4,)).astype(numpy.float32),
                lambda rng: rng.normal(size=(4, 5)).astype(numpy.float32),
                name="vector_matrix",
            ),
            # [M,K] x [K] -> [M]
            OperationTestConfig(
                lambda lhs, rhs: lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((1,), (0,)), ((), ())),
                ),
                lambda rng: rng.normal(size=(3, 4)).astype(numpy.float32),
                lambda rng: rng.normal(size=(4,)).astype(numpy.float32),
                name="matrix_vector",
            ),
            # Already aligned: batch dims are a prefix; no transpose/reshape needed.
            OperationTestConfig(
                lambda lhs, rhs: lax.dot_general(
                    lhs,
                    rhs,
                    dimension_numbers=(((3,), (2,)), ((0, 1), (0, 1))),
                ),
                lambda rng: rng.normal(size=(2, 2, 3, 4)).astype(numpy.float32),
                lambda rng: rng.normal(size=(2, 2, 4, 5)).astype(numpy.float32),
                name="already_aligned_rank4",
            ),
        ]
