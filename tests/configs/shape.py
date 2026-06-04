from jax import lax
from jax import numpy as jnp

from .util import OperationTestConfig, complex_standard_normal


def make_shape_op_configs():
    # Flip and transpose ops for both real and complex inputs
    for is_complex in [False, True]:
        with OperationTestConfig.module_name("shape-complex" if is_complex else "shape-real"):
            yield from [
                OperationTestConfig(
                    jnp.flip,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (17,), is_complex),
                ),
                OperationTestConfig(
                    jnp.fliplr,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (17, 13), is_complex),
                ),
                OperationTestConfig(
                    jnp.flipud,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (17, 13), is_complex),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (17, 8, 9), is_complex),
                ),
                OperationTestConfig(
                    jnp.transpose,
                    lambda rng, is_complex=is_complex: complex_standard_normal(rng, (17, 8, 9), is_complex),
                    (1, 0, 2),
                    static_argnums=(1,),
                ),
            ]

    with OperationTestConfig.module_name("shape"):
        yield from [
            OperationTestConfig(
                lambda x, y: jnp.concatenate([x, y], axis=0),
                lambda rng: rng.normal(size=(3, 4)),
                lambda rng: rng.normal(size=(5, 4)),
            ),
            OperationTestConfig(
                lambda x: jnp.reshape(x, (20,)),
                lambda rng: rng.normal(size=(4, 5)),
            ),
            OperationTestConfig(
                lambda x: jnp.pad(x, ((1, 1), (2, 2))),
                lambda rng: rng.normal(size=(3, 3)),
                name="pad_positive_edge",
            ),
            OperationTestConfig(
                lambda x: lax.pad(
                    x,
                    jnp.array(0.0, dtype=x.dtype),
                    [(-1, -1, 0), (-2, -2, 0)],
                ),
                lambda rng: rng.normal(size=(5, 7)).astype("float32"),
                name="pad_negative_edge_crop",
            ),
            OperationTestConfig(
                lambda x: lax.pad(
                    x,
                    jnp.array(0.0, dtype=x.dtype),
                    [(1, 2, 1)],
                ),
                lambda rng: rng.normal(size=(4,)).astype("float32"),
                name="pad_interior_1d",
            ),
            OperationTestConfig(
                lambda x: lax.pad(
                    x,
                    jnp.array(0.0, dtype=x.dtype),
                    [(1, 0, 2), (0, 1, 1)],
                ),
                lambda rng: rng.normal(size=(3, 2)).astype("float32"),
                name="pad_interior_2d",
            ),
            OperationTestConfig(
                lambda x: lax.pad(
                    x,
                    jnp.array(0.0, dtype=x.dtype),
                    [(-1, 1, 1)],
                ),
                lambda rng: rng.normal(size=(5,)).astype("float32"),
                name="pad_interior_negative_edge",
            ),
            OperationTestConfig(
                lambda x, d: x.at[jnp.arange(x.shape[-1]), jnp.arange(x.shape[-1])].add(d),
                lambda rng: rng.normal(size=(4, 4)).astype("float32"),
                lambda rng: rng.normal(size=(4,)).astype("float32"),
                name="scatter_add_diag",
            ),
            OperationTestConfig(
                lambda x, d: x.at[:, jnp.arange(x.shape[-1]), jnp.arange(x.shape[-1])].add(d),
                lambda rng: rng.normal(size=(3, 4, 4)).astype("float32"),
                lambda rng: rng.normal(size=(3, 4)).astype("float32"),
                name="scatter_add_diag_batched",
            ),
            # Gather patterns that include smaller slice_sizes on non-indexed dims.
            # This exercises both gather and the corresponding scatter-add gradient.
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[0], [2], [4]], dtype=jnp.int32),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(1,),
                        collapsed_slice_dims=(0,),
                        start_index_map=(0,),
                        operand_batching_dims=(),
                        start_indices_batching_dims=(),
                    ),
                    slice_sizes=(1, 3),
                ),
                lambda rng: rng.normal(size=(5, 7)).astype("float32"),
                name="gather_slice_sizes_2d_window",
            ),
            # Gather window that truncates two non-scatter dims (exercises multi-window scatter grad).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[0], [2], [4]], dtype=jnp.int32),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(1, 2),
                        collapsed_slice_dims=(0,),
                        start_index_map=(0,),
                        operand_batching_dims=(),
                        start_indices_batching_dims=(),
                    ),
                    slice_sizes=(1, 3, 4),
                ),
                lambda rng: rng.normal(size=(5, 7, 9)).astype("float32"),
                name="gather_slice_sizes_3d_window_multi_trunc",
            ),
            # Batched gather with batching dims + slice window on a non-indexed dim.
            # This exercises gather with operand/start_indices batching dims and the
            # corresponding batched scatter gradient (including window scatter handling).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[[0], [2], [4]], [[1], [3], [0]]], dtype=jnp.int32),  # [B, N, 1]
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(2,),
                        collapsed_slice_dims=(1,),
                        start_index_map=(1,),
                        operand_batching_dims=(0,),
                        start_indices_batching_dims=(0,),
                    ),
                    slice_sizes=(1, 1, 3),
                ),
                lambda rng: rng.normal(size=(2, 5, 7)).astype("float32"),
                name="gather_batched_slice_sizes_3d_window",
            ),
            # Embedding gather with two batching dims (Tier A: prefix batching dims).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array(
                        [
                            [
                                [[0], [2], [4], [1]],
                                [[1], [3], [0], [2]],
                                [[2], [1], [3], [4]],
                            ],
                            [
                                [[4], [3], [2], [1]],
                                [[0], [1], [2], [3]],
                                [[1], [0], [4], [2]],
                            ],
                        ],
                        dtype=jnp.int32,
                    ),  # [B1, B2, N, 1]
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(3,),
                        collapsed_slice_dims=(2,),
                        start_index_map=(2,),
                        operand_batching_dims=(0, 1),
                        start_indices_batching_dims=(0, 1),
                    ),
                    slice_sizes=(1, 1, 1, 7),
                ),
                lambda rng: rng.normal(size=(2, 3, 5, 7)).astype("float32"),
                name="gather_batched_embedding_2batchdims",
            ),
            # Embedding gather batching dims that are not a prefix (Tier B: canonicalize + unpermute).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array(
                        [
                            [
                                [[0], [2], [4], [1]],
                                [[1], [3], [0], [2]],
                                [[2], [1], [3], [4]],
                            ],
                            [
                                [[4], [3], [2], [1]],
                                [[0], [1], [2], [3]],
                                [[1], [0], [4], [2]],
                            ],
                        ],
                        dtype=jnp.int32,
                    ).transpose(0, 2, 1, 3),  # [B1, N, B2, 1]
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(3,),
                        collapsed_slice_dims=(1,),
                        start_index_map=(1,),
                        operand_batching_dims=(0, 2),
                        start_indices_batching_dims=(0, 2),
                    ),
                    slice_sizes=(1, 1, 1, 7),
                ),
                lambda rng: rng.normal(size=(2, 5, 3, 7)).astype("float32"),
                name="gather_batched_embedding_nonprefix_batchdims",
            ),
            # Point gatherND pattern where the "remaining dims" are sliced to smaller windows.
            # This exercises gatherND + scatter window gradients (multi truncated dims).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[0, 0], [2, 1], [1, 3]], dtype=jnp.int32),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(1, 2),
                        collapsed_slice_dims=(0, 2),
                        start_index_map=(0, 2),
                        operand_batching_dims=(),
                        start_indices_batching_dims=(),
                    ),
                    slice_sizes=(1, 2, 1, 3),
                ),
                lambda rng: rng.normal(size=(3, 4, 5, 6)).astype("float32"),
                name="gather_nd_slice_sizes_multi_dim",
            ),
            # Point gatherND pattern where start_index_map dims are not collapsed (re-insert size-1
            # dims into the output via reshape + transpose).
            OperationTestConfig(
                lambda x: lax.gather(
                    x,
                    jnp.array([[0, 0], [2, 1], [1, 3]], dtype=jnp.int32),
                    dimension_numbers=lax.GatherDimensionNumbers(
                        offset_dims=(0, 2, 3, 4),
                        collapsed_slice_dims=(),
                        start_index_map=(0, 2),
                        operand_batching_dims=(),
                        start_indices_batching_dims=(),
                    ),
                    slice_sizes=(1, 2, 1, 3),
                ),
                lambda rng: rng.normal(size=(3, 4, 5, 6)).astype("float32"),
                name="gather_nd_offset_dims_nontrivial",
            ),
        ]
