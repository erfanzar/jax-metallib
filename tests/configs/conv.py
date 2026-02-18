import numpy
from jax import lax

from .util import OperationTestConfig


def make_conv_op_configs():
    with OperationTestConfig.module_name("conv"):
        return [
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
            ),
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1),
                    padding="SAME",
                    batch_group_count=2,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 4)).astype(numpy.float32),
                name="lax.conv_general_dilated-batch_group_count2",
            ),
            # Grouped convolution (feature_group_count > 1)
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1),
                    padding="SAME",
                    feature_group_count=2,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(1, 8, 8, 4)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 2, 6)).astype(numpy.float32),
                name="lax.conv_general_dilated-grouped",
            ),
            # Depthwise-ish convolution (feature_group_count == in_channels)
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1),
                    padding="SAME",
                    feature_group_count=3,
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(1, 8, 8, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 1, 6)).astype(numpy.float32),
                name="lax.conv_general_dilated-depthwise",
            ),
            OperationTestConfig(
                lambda lhs, rhs: lax.conv(lhs, rhs, (1, 1), "SAME"),
                lambda rng: rng.normal(size=(1, 3, 8, 8)),
                lambda rng: rng.normal(size=(16, 3, 3, 3)),
                name="lax.conv-SAME",
            ),
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2,),
                    padding="SAME",
                    dimension_numbers=("NWC", "WIO", "NWC"),
                ),
                lambda rng: rng.normal(size=(2, 16, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 4)).astype(numpy.float32),
                name="lax.conv_general_dilated-1d-NWC",
            ),
            # Strided 2D convolutions with SAME padding
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
                name="lax.conv_general_dilated-stride2-SAME",
            ),
            # Asymmetric padding test
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding=((1, 0), (1, 0)),
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 8, 8, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
                name="lax.conv_general_dilated-stride2-asymmetric",
            ),
            # Larger kernel with stride
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 16, 16, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(5, 5, 3, 8)).astype(numpy.float32),
                name="lax.conv_general_dilated-5x5-stride2",
            ),
            # Stride + dilation combined
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="SAME",
                    rhs_dilation=(2, 2),
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 16, 16, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
                name="lax.conv_general_dilated-stride2-dilated",
            ),
            # VALID padding with stride
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2, 2),
                    padding="VALID",
                    dimension_numbers=("NHWC", "HWIO", "NHWC"),
                ),
                lambda rng: rng.normal(size=(2, 16, 16, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 8)).astype(numpy.float32),
                name="lax.conv_general_dilated-stride2-VALID",
            ),
            # 1D strided + dilated
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(2,),
                    padding="SAME",
                    rhs_dilation=(2,),
                    dimension_numbers=("NWC", "WIO", "NWC"),
                ),
                lambda rng: rng.normal(size=(2, 32, 3)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 4)).astype(numpy.float32),
                name="lax.conv_general_dilated-1d-stride2-dilated",
            ),
            # 3D convolution (NDHWC / DHWIO)
            OperationTestConfig(
                lambda x, kernel: lax.conv_general_dilated(
                    x,
                    kernel,
                    window_strides=(1, 1, 1),
                    padding="SAME",
                    dimension_numbers=("NDHWC", "DHWIO", "NDHWC"),
                ),
                lambda rng: rng.normal(size=(1, 5, 6, 7, 2)).astype(numpy.float32),
                lambda rng: rng.normal(size=(3, 3, 3, 2, 4)).astype(numpy.float32),
                name="lax.conv_general_dilated-3d-NDHWC",
            ),
        ]
