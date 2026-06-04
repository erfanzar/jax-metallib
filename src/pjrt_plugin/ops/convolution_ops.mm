// Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "pjrt_plugin/ops/registry.h"

#include <sstream>

namespace jax_metallib {

namespace {

static std::string ShapeToString(NSArray<NSNumber*>* shape) {
    if (!shape) {
        return "<null>";
    }
    std::ostringstream oss;
    oss << "[";
    for (NSUInteger i = 0; i < shape.count; ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << [shape[i] longLongValue];
    }
    oss << "]";
    return oss.str();
}

} // namespace

static NSArray<NSNumber*>* computePermutation(const std::vector<int64_t>& srcLayout,
                                              const std::vector<int64_t>& dstLayout) {
    size_t rank = srcLayout.size();
    if (rank != dstLayout.size()) {
        return nil;
    }

    if (srcLayout == dstLayout) {
        return nil;
    }

    std::vector<int64_t> dstPosToDim(rank, -1);
    for (size_t d = 0; d < rank; ++d) {
        int64_t dstPos = dstLayout[d];
        if (dstPos < 0 || dstPos >= (int64_t)rank || dstPosToDim[(size_t)dstPos] != -1) {
            return nil;
        }
        dstPosToDim[(size_t)dstPos] = (int64_t)d;
    }

    NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rank];
    bool isIdentity = true;
    for (size_t dstPos = 0; dstPos < rank; ++dstPos) {
        int64_t dim = dstPosToDim[dstPos];
        if (dim < 0) {
            return nil;
        }

        int64_t srcPos = srcLayout[(size_t)dim];
        [perm addObject:@(srcPos)];
        if (srcPos != (int64_t)dstPos)
            isIdentity = false;
    }
    return isIdentity ? nil : perm;
}

static const std::vector<int64_t> LAYOUT_NHWC = {0, 3, 1, 2};

static const std::vector<int64_t> LAYOUT_NDHWC = {0, 4, 1, 2, 3};

struct ConvParams {
    int64_t strideD = 1, strideH = 1, strideW = 1;
    int64_t padFront = 0, padBack = 0, padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;
    int64_t dilationD = 1, dilationH = 1, dilationW = 1;
    int64_t inputDilationD = 1, inputDilationH = 1, inputDilationW = 1;
    int64_t featureGroupCount = 1;
    bool isTransposed = false;
};

static ConvParams extractConvParams(mlir::stablehlo::ConvolutionOp& convOp, size_t spatialRank) {
    ConvParams p;
    p.featureGroupCount = static_cast<int64_t>(convOp.getFeatureGroupCount());

    auto windowStrides = convOp.getWindowStrides();
    if (windowStrides) {
        auto v = windowStrides.value();
        if (spatialRank == 1 && v.size() >= 1) {
            p.strideW = v[0];
        } else if (spatialRank == 2 && v.size() >= 2) {
            p.strideH = v[0];
            p.strideW = v[1];
        } else if (spatialRank == 3 && v.size() >= 3) {
            p.strideD = v[0];
            p.strideH = v[1];
            p.strideW = v[2];
        }
    }

    auto padding = convOp.getPadding();
    if (padding) {
        auto paddingAttr = padding.value();
        if (spatialRank == 1 && paddingAttr.getNumElements() >= 2) {
            auto pv = paddingAttr.getValues<int64_t>();
            p.padLeft = pv[{0, 0}];
            p.padRight = pv[{0, 1}];
        } else if (spatialRank == 2 && paddingAttr.getNumElements() >= 4) {
            auto pv = paddingAttr.getValues<int64_t>();
            p.padTop = pv[{0, 0}];
            p.padBottom = pv[{0, 1}];
            p.padLeft = pv[{1, 0}];
            p.padRight = pv[{1, 1}];
        } else if (spatialRank == 3 && paddingAttr.getNumElements() >= 6) {
            auto pv = paddingAttr.getValues<int64_t>();
            p.padFront = pv[{0, 0}];
            p.padBack = pv[{0, 1}];
            p.padTop = pv[{1, 0}];
            p.padBottom = pv[{1, 1}];
            p.padLeft = pv[{2, 0}];
            p.padRight = pv[{2, 1}];
        }
    }

    auto rhsDilation = convOp.getRhsDilation();
    if (rhsDilation) {
        auto v = rhsDilation.value();
        if (spatialRank == 1 && v.size() >= 1) {
            p.dilationW = v[0];
        } else if (spatialRank == 2 && v.size() >= 2) {
            p.dilationH = v[0];
            p.dilationW = v[1];
        } else if (spatialRank == 3 && v.size() >= 3) {
            p.dilationD = v[0];
            p.dilationH = v[1];
            p.dilationW = v[2];
        }
    }

    auto lhsDilation = convOp.getLhsDilation();
    if (lhsDilation) {
        auto v = lhsDilation.value();
        if (spatialRank == 1 && v.size() >= 1) {
            p.inputDilationW = v[0];
            p.isTransposed = (p.inputDilationW != 1);
        } else if (spatialRank == 2 && v.size() >= 2) {
            p.inputDilationH = v[0];
            p.inputDilationW = v[1];
            p.isTransposed = (p.inputDilationH != 1 || p.inputDilationW != 1);
        } else if (spatialRank == 3 && v.size() >= 3) {
            p.inputDilationD = v[0];
            p.inputDilationH = v[1];
            p.inputDilationW = v[2];
            p.isTransposed = (p.inputDilationD != 1 || p.inputDilationH != 1 || p.inputDilationW != 1);
        }
    }

    return p;
}

static int64_t forwardPadding(int64_t kernelSize, int64_t dilation, int64_t transposedPad) {
    int64_t effectiveK = 1 + (kernelSize - 1) * dilation;
    return std::max(0LL, effectiveK - 1 - transposedPad);
}

static MPSGraphTensor* prepareKernelForDataGradient(MPSGraph* g, MPSGraphTensor* kernel,
                                                    NSArray<NSNumber*>* spatialAxes) {

    MPSGraphTensor* unreversed = [g reverseTensor:kernel axes:spatialAxes name:nil];

    NSUInteger rank = unreversed.shape.count;
    if (rank < 2) {
        return unreversed;
    }
    NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rank];
    [perm addObject:@1];
    [perm addObject:@0];
    for (NSUInteger i = 2; i < rank; ++i) {
        [perm addObject:@(i)];
    }
    return [g transposeTensor:unreversed permutation:perm name:nil];
}

static MPSGraphTensor* lift1DTo2D(MPSGraph* g, MPSGraphTensor* tensor, int64_t dim0, int64_t dim1,
                                  int64_t dim2) {
    MPSGraphTensor* reordered = tensor;
    if (!(dim0 == 0 && dim1 == 1 && dim2 == 2)) {
        reordered = [g transposeTensor:tensor permutation:@[@(dim0), @(dim1), @(dim2)] name:nil];
    }
    NSArray<NSNumber*>* shape = reordered.shape;
    if (!shape || shape.count != 3)
        return nullptr;
    return [g reshapeTensor:reordered withShape:@[shape[0], shape[1], @1, shape[2]] name:nil];
}

static MPSGraphTensor* convert2DOutputTo1D(MPSGraph* g, MPSGraphTensor* result, int64_t batchDim,
                                           int64_t featureDim, int64_t spatialDim) {

    result = [g transposeTensor:result permutation:@[@0, @3, @1, @2] name:nil];

    NSArray<NSNumber*>* shape4D = result.shape;
    result = [g reshapeTensor:result withShape:@[shape4D[0], shape4D[1], shape4D[3]] name:nil];

    std::vector<int64_t> srcLayout = {0, 1, 2};
    std::vector<int64_t> dstLayout = {batchDim, featureDim, spatialDim};
    NSArray<NSNumber*>* perm = computePermutation(srcLayout, dstLayout);
    if (perm) {
        result = [g transposeTensor:result permutation:perm name:nil];
    }
    return result;
}

static ProcessResult HandleConvolution(HandlerContext& ctx) {
    auto convOp = mlir::dyn_cast<mlir::stablehlo::ConvolutionOp>(ctx.op);
    if (!convOp) {
        return ProcessResult::Error("convolution: expected ConvolutionOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* kernel = GetInputTensor(ctx, 1);
    if (!input || !kernel)
        return ProcessResult::Error("convolution: missing input tensor");

    MPSGraph* g = ctx.graph;
    mlir::Operation* op = ctx.op;

    auto dimNumbers = convOp.getDimensionNumbers();
    int64_t inputBatchDim = dimNumbers.getInputBatchDimension();
    int64_t inputFeatureDim = dimNumbers.getInputFeatureDimension();
    auto inputSpatialDims = dimNumbers.getInputSpatialDimensions();

    int64_t kernelInputFeatureDim = dimNumbers.getKernelInputFeatureDimension();
    int64_t kernelOutputFeatureDim = dimNumbers.getKernelOutputFeatureDimension();
    auto kernelSpatialDims = dimNumbers.getKernelSpatialDimensions();

    int64_t outputBatchDim = dimNumbers.getOutputBatchDimension();
    int64_t outputFeatureDim = dimNumbers.getOutputFeatureDimension();
    auto outputSpatialDims = dimNumbers.getOutputSpatialDimensions();

    int64_t batchGroupCount = static_cast<int64_t>(convOp.getBatchGroupCount());
    if (batchGroupCount < 1) {
        return ProcessResult::Error("convolution: batch_group_count must be >= 1");
    }

    size_t spatialRank = inputSpatialDims.size();

    bool is1D = (spatialRank == 1);
    bool is3D = (spatialRank == 3);
    if (spatialRank > 3) {
        return ProcessResult::Error("convolution: only 1D/2D/3D convolution is currently supported");
    }

    ConvParams p = extractConvParams(convOp, spatialRank);
    if (batchGroupCount != 1 && p.isTransposed) {
        return ProcessResult::Error(
            "convolution: batch_group_count > 1 for transposed convolution is not supported");
    }

    if (is3D) {
        if (!input.shape || input.shape.count != 5 || !kernel.shape || kernel.shape.count != 5) {
            return ProcessResult::Error("convolution: 3D convolution expects rank-5 input and kernel");
        }
        if (@available(macOS 13.2, *)) {

            std::vector<int64_t> inputLayout = {inputBatchDim, inputFeatureDim, inputSpatialDims[0],
                                                inputSpatialDims[1], inputSpatialDims[2]};
            std::vector<int64_t> kernelLayout = {kernelOutputFeatureDim, kernelInputFeatureDim,
                                                 kernelSpatialDims[0], kernelSpatialDims[1],
                                                 kernelSpatialDims[2]};

            // Check if input is already in NDHWC layout
            NSArray<NSNumber*>* inputPerm = computePermutation(inputLayout, LAYOUT_NDHWC);
            MPSGraphTensor* convInput = input;
            bool inputAlreadyNDHWC = (inputPerm == nil);

            if (!inputAlreadyNDHWC) {
                convInput = [g transposeTensor:input permutation:inputPerm name:nil];
            }

            // Check if kernel is already in OIDHW layout
            NSArray<NSNumber*>* kernelPerm = computePermutation(kernelLayout, {0, 1, 2, 3, 4});
            MPSGraphTensor* mpsKernel = kernel;
            bool kernelAlreadyOIDHW = (kernelPerm == nil);

            if (!kernelAlreadyOIDHW) {
                mpsKernel = [g transposeTensor:kernel permutation:kernelPerm name:nil];
            }

            MPSGraphTensor* result = nil;
            if (p.isTransposed) {

                auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
                if (!resultType) {
                    return ProcessResult::Error(
                        "convolution: could not get result type for transposed convolution");
                }
                auto resultShape = resultType.getShape();
                int64_t outN = resultShape[outputBatchDim];
                int64_t outD = resultShape[outputSpatialDims[0]];
                int64_t outH = resultShape[outputSpatialDims[1]];
                int64_t outW = resultShape[outputSpatialDims[2]];
                int64_t outC = resultShape[outputFeatureDim];
                NSArray<NSNumber*>* outputShape = @[@(outN), @(outD), @(outH), @(outW), @(outC)];

                int64_t kD = [mpsKernel.shape[2] longLongValue];
                int64_t kH = [mpsKernel.shape[3] longLongValue];
                int64_t kW = [mpsKernel.shape[4] longLongValue];

                MPSGraphTensor* preparedKernel = prepareKernelForDataGradient(g, mpsKernel, @[@2, @3, @4]);

                MPSGraphConvolution3DOpDescriptor* desc = [MPSGraphConvolution3DOpDescriptor
                    descriptorWithStrideInX:(NSUInteger)p.inputDilationW
                                  strideInY:(NSUInteger)p.inputDilationH
                                  strideInZ:(NSUInteger)p.inputDilationD
                            dilationRateInX:(NSUInteger)p.dilationW
                            dilationRateInY:(NSUInteger)p.dilationH
                            dilationRateInZ:(NSUInteger)p.dilationD
                                     groups:(NSUInteger)p.featureGroupCount
                               paddingStyle:MPSGraphPaddingStyleExplicit
                                 dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
                              weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
                desc.paddingFront = (NSUInteger)forwardPadding(kD, p.dilationD, p.padFront);
                desc.paddingBack = (NSUInteger)forwardPadding(kD, p.dilationD, p.padBack);
                desc.paddingTop = (NSUInteger)forwardPadding(kH, p.dilationH, p.padTop);
                desc.paddingBottom = (NSUInteger)forwardPadding(kH, p.dilationH, p.padBottom);
                desc.paddingLeft = (NSUInteger)forwardPadding(kW, p.dilationW, p.padLeft);
                desc.paddingRight = (NSUInteger)forwardPadding(kW, p.dilationW, p.padRight);

                result = [g convolution3DDataGradientWithIncomingGradientTensor:convInput
                                                                  weightsTensor:preparedKernel
                                                                    outputShape:outputShape
                                                   forwardConvolutionDescriptor:desc
                                                                           name:nil];
            } else {

                auto makeDescriptor = [&]() {
                    MPSGraphConvolution3DOpDescriptor* desc = [MPSGraphConvolution3DOpDescriptor
                        descriptorWithStrideInX:(NSUInteger)p.strideW
                                      strideInY:(NSUInteger)p.strideH
                                      strideInZ:(NSUInteger)p.strideD
                                dilationRateInX:(NSUInteger)p.dilationW
                                dilationRateInY:(NSUInteger)p.dilationH
                                dilationRateInZ:(NSUInteger)p.dilationD
                                         groups:(NSUInteger)p.featureGroupCount
                                   paddingStyle:MPSGraphPaddingStyleExplicit
                                     dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
                                  weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
                    desc.paddingLeft = (NSUInteger)p.padLeft;
                    desc.paddingRight = (NSUInteger)p.padRight;
                    desc.paddingTop = (NSUInteger)p.padTop;
                    desc.paddingBottom = (NSUInteger)p.padBottom;
                    desc.paddingFront = (NSUInteger)p.padFront;
                    desc.paddingBack = (NSUInteger)p.padBack;
                    return desc;
                };

                if (batchGroupCount == 1) {
                    result = [g convolution3DWithSourceTensor:convInput
                                                weightsTensor:mpsKernel
                                                   descriptor:makeDescriptor()
                                                         name:nil];
                } else {
                    if (batchGroupCount > 4) {
                        return ProcessResult::Error(
                            "convolution: batch_group_count emulation would generate >4 convs; "
                            "unsupported for performance (batch_group_count=" +
                            std::to_string(batchGroupCount) +
                            ", input_shape=" + ShapeToString(convInput.shape) +
                            ", kernel_shape=" + ShapeToString(mpsKernel.shape) + ")");
                    }
                    if (!convInput.shape || convInput.shape.count != 5 || !mpsKernel.shape ||
                        mpsKernel.shape.count != 5) {
                        return ProcessResult::Error("convolution: batch_group_count > 1 requires "
                                                    "rank-5 NDHWC/OIDHW tensors");
                    }

                    int64_t lhsBatch = [convInput.shape[0] longLongValue];
                    if (lhsBatch <= 0) {
                        return ProcessResult::Error(
                            "convolution: batch_group_count > 1 requires static batch size");
                    }
                    if (lhsBatch % batchGroupCount != 0) {
                        return ProcessResult::Error("convolution: lhs batch dimension size (" +
                                                    std::to_string(lhsBatch) +
                                                    ") must be divisible by batch_group_count (" +
                                                    std::to_string(batchGroupCount) + ")");
                    }
                    int64_t outBatch = lhsBatch / batchGroupCount;

                    int64_t rhsOutChannels = [mpsKernel.shape[0] longLongValue];
                    if (rhsOutChannels <= 0) {
                        return ProcessResult::Error("convolution: batch_group_count > 1 requires "
                                                    "static output channel size");
                    }
                    if (rhsOutChannels % batchGroupCount != 0) {
                        return ProcessResult::Error("convolution: rhs output feature dimension size (" +
                                                    std::to_string(rhsOutChannels) +
                                                    ") must be a multiple of batch_group_count (" +
                                                    std::to_string(batchGroupCount) + ")");
                    }
                    int64_t outChannelsPerGroup = rhsOutChannels / batchGroupCount;

                    if (p.featureGroupCount <= 0) {
                        return ProcessResult::Error("convolution: feature_group_count must be >= 1");
                    }
                    if (outChannelsPerGroup % p.featureGroupCount != 0) {
                        return ProcessResult::Error(
                            "convolution: rhs output feature dimension per batch group (" +
                            std::to_string(outChannelsPerGroup) +
                            ") must be divisible by feature_group_count (" +
                            std::to_string(p.featureGroupCount) + ")");
                    }

                    NSMutableArray<MPSGraphTensor*>* groupResults =
                        [NSMutableArray arrayWithCapacity:(NSUInteger)batchGroupCount];
                    for (int64_t group = 0; group < batchGroupCount; ++group) {
                        int64_t batchStart = group * outBatch;
                        int64_t batchEnd = batchStart + outBatch;
                        MPSGraphTensor* lhsSlice =
                            [g sliceTensor:convInput
                                    starts:@[@(batchStart), @0, @0, @0, @0]
                                      ends:@[
                                          @(batchEnd), convInput.shape[1], convInput.shape[2],
                                          convInput.shape[3], convInput.shape[4]
                                      ]
                                   strides:@[@1, @1, @1, @1, @1]
                                      name:nil];

                        int64_t outStart = group * outChannelsPerGroup;
                        int64_t outEnd = outStart + outChannelsPerGroup;
                        MPSGraphTensor* rhsSlice =
                            [g sliceTensor:mpsKernel
                                    starts:@[@(outStart), @0, @0, @0, @0]
                                      ends:@[
                                          @(outEnd), mpsKernel.shape[1], mpsKernel.shape[2],
                                          mpsKernel.shape[3], mpsKernel.shape[4]
                                      ]
                                   strides:@[@1, @1, @1, @1, @1]
                                      name:nil];

                        MPSGraphTensor* groupResult = [g convolution3DWithSourceTensor:lhsSlice
                                                                         weightsTensor:rhsSlice
                                                                            descriptor:makeDescriptor()
                                                                                  name:nil];
                        [groupResults addObject:groupResult];
                    }

                    result = [g concatTensors:groupResults dimension:4 name:nil];
                }
            }

            std::vector<int64_t> outputLayout = {outputBatchDim, outputFeatureDim, outputSpatialDims[0],
                                                 outputSpatialDims[1], outputSpatialDims[2]};
            NSArray<NSNumber*>* outputPerm = computePermutation(LAYOUT_NDHWC, outputLayout);
            if (outputPerm) {
                result = [g transposeTensor:result permutation:outputPerm name:nil];
            }
            return Result(ctx, result, "convolution");
        }

        return ProcessResult::Error(
            "convolution: 3D convolution requires macOS 13.2+ (MPSGraph convolution3D)");
    }

    int64_t output1DSpatialDim = is1D ? outputSpatialDims[0] : 0;

    if (is1D) {

        input = lift1DTo2D(g, input, inputBatchDim, inputFeatureDim, inputSpatialDims[0]);
        kernel = lift1DTo2D(g, kernel, kernelOutputFeatureDim, kernelInputFeatureDim, kernelSpatialDims[0]);
        if (!input || !kernel) {
            return ProcessResult::Error("convolution: 1D convolution expects rank-3 input and kernel");
        }
    }

    int64_t inBatch = is1D ? 0 : inputBatchDim;
    int64_t inFeature = is1D ? 1 : inputFeatureDim;
    int64_t inSpatial0 = is1D ? 2 : inputSpatialDims[0];
    int64_t inSpatial1 = is1D ? 3 : inputSpatialDims[1];

    int64_t kOutput = is1D ? 0 : kernelOutputFeatureDim;
    int64_t kInput = is1D ? 1 : kernelInputFeatureDim;
    int64_t kSpatial0 = is1D ? 2 : kernelSpatialDims[0];
    int64_t kSpatial1 = is1D ? 3 : kernelSpatialDims[1];

    int64_t outSpatial0 = is1D ? 2 : outputSpatialDims[0];
    int64_t outSpatial1 = is1D ? 3 : outputSpatialDims[1];

    std::vector<int64_t> inputLayout = {inBatch, inFeature, inSpatial0, inSpatial1};
    std::vector<int64_t> kernelLayout = {kOutput, kInput, kSpatial0, kSpatial1};

    // Check if input is already in NHWC layout (N=batch, H=spatial0, W=spatial1, C=feature)
    // LAYOUT_NHWC = {0, 3, 1, 2} means target layout is [batch, H, W, channels]
    // where batch at pos 0, feature at pos 3, spatial0 at pos 1, spatial1 at pos 2
    NSArray<NSNumber*>* inputPerm = computePermutation(inputLayout, LAYOUT_NHWC);
    MPSGraphTensor* convInput = input;
    bool inputAlreadyNHWC = (inputPerm == nil);

    if (!inputAlreadyNHWC) {
        convInput = [g transposeTensor:input permutation:inputPerm name:nil];
    }

    // Check if kernel is already in OIHW layout (O=output_channels, I=input_channels, H, W)
    NSArray<NSNumber*>* kernelPerm = computePermutation(kernelLayout, {0, 1, 2, 3});
    MPSGraphTensor* mpsKernel = kernel;
    bool kernelAlreadyOIHW = (kernelPerm == nil);

    if (!kernelAlreadyOIHW) {
        mpsKernel = [g transposeTensor:kernel permutation:kernelPerm name:nil];
    }

    MPSGraphTensor* result;
    if (p.isTransposed) {

        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
        if (!resultType) {
            return ProcessResult::Error("convolution: could not get result type for transposed convolution");
        }

        auto resultShape = resultType.getShape();
        int64_t outN = resultShape[outputBatchDim];
        int64_t outH = is1D ? 1 : resultShape[outputSpatialDims[0]];
        int64_t outW = resultShape[is1D ? output1DSpatialDim : outputSpatialDims[1]];
        int64_t outC = resultShape[outputFeatureDim];
        NSArray<NSNumber*>* outputShape = @[@(outN), @(outH), @(outW), @(outC)];

        int64_t kH = [mpsKernel.shape[2] longLongValue];
        int64_t kW = [mpsKernel.shape[3] longLongValue];

        MPSGraphTensor* preparedKernel = prepareKernelForDataGradient(g, mpsKernel, @[@2, @3]);

        MPSGraphConvolution2DOpDescriptor* desc =
            [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)p.inputDilationW
                                                             strideInY:(NSUInteger)p.inputDilationH
                                                       dilationRateInX:(NSUInteger)p.dilationW
                                                       dilationRateInY:(NSUInteger)p.dilationH
                                                                groups:(NSUInteger)p.featureGroupCount
                                                          paddingStyle:MPSGraphPaddingStyleExplicit
                                                            dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                         weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        desc.paddingTop = (NSUInteger)forwardPadding(kH, p.dilationH, p.padTop);
        desc.paddingBottom = (NSUInteger)forwardPadding(kH, p.dilationH, p.padBottom);
        desc.paddingLeft = (NSUInteger)forwardPadding(kW, p.dilationW, p.padLeft);
        desc.paddingRight = (NSUInteger)forwardPadding(kW, p.dilationW, p.padRight);

        result = [g convolution2DDataGradientWithIncomingGradientTensor:convInput
                                                          weightsTensor:preparedKernel
                                                            outputShape:outputShape
                                           forwardConvolutionDescriptor:desc
                                                                   name:nil];
    } else {

        auto makeDescriptor = [&]() {
            MPSGraphConvolution2DOpDescriptor* desc =
                [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:(NSUInteger)p.strideW
                                                                 strideInY:(NSUInteger)p.strideH
                                                           dilationRateInX:(NSUInteger)p.dilationW
                                                           dilationRateInY:(NSUInteger)p.dilationH
                                                                    groups:(NSUInteger)p.featureGroupCount
                                                              paddingStyle:MPSGraphPaddingStyleExplicit
                                                                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                                                             weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
            desc.paddingLeft = (NSUInteger)p.padLeft;
            desc.paddingRight = (NSUInteger)p.padRight;
            desc.paddingTop = (NSUInteger)p.padTop;
            desc.paddingBottom = (NSUInteger)p.padBottom;
            return desc;
        };

        if (batchGroupCount == 1) {
            result = [g convolution2DWithSourceTensor:convInput
                                        weightsTensor:mpsKernel
                                           descriptor:makeDescriptor()
                                                 name:nil];
        } else {
            if (batchGroupCount > 4) {
                return ProcessResult::Error(
                    "convolution: batch_group_count emulation would generate >4 convs; "
                    "unsupported for performance (batch_group_count=" +
                    std::to_string(batchGroupCount) + ", input_shape=" + ShapeToString(convInput.shape) +
                    ", kernel_shape=" + ShapeToString(mpsKernel.shape) + ")");
            }
            if (!convInput.shape || convInput.shape.count != 4 || !mpsKernel.shape ||
                mpsKernel.shape.count != 4) {
                return ProcessResult::Error(
                    "convolution: batch_group_count > 1 requires rank-4 NHWC/OIHW tensors");
            }

            int64_t lhsBatch = [convInput.shape[0] longLongValue];
            if (lhsBatch <= 0) {
                return ProcessResult::Error("convolution: batch_group_count > 1 requires static batch size");
            }
            if (lhsBatch % batchGroupCount != 0) {
                return ProcessResult::Error(
                    "convolution: lhs batch dimension size (" + std::to_string(lhsBatch) +
                    ") must be divisible by batch_group_count (" + std::to_string(batchGroupCount) + ")");
            }
            int64_t outBatch = lhsBatch / batchGroupCount;

            int64_t rhsOutChannels = [mpsKernel.shape[0] longLongValue];
            if (rhsOutChannels <= 0) {
                return ProcessResult::Error(
                    "convolution: batch_group_count > 1 requires static output channel size");
            }
            if (rhsOutChannels % batchGroupCount != 0) {
                return ProcessResult::Error(
                    "convolution: rhs output feature dimension size (" + std::to_string(rhsOutChannels) +
                    ") must be a multiple of batch_group_count (" + std::to_string(batchGroupCount) + ")");
            }
            int64_t outChannelsPerGroup = rhsOutChannels / batchGroupCount;

            if (p.featureGroupCount <= 0) {
                return ProcessResult::Error("convolution: feature_group_count must be >= 1");
            }
            if (outChannelsPerGroup % p.featureGroupCount != 0) {
                return ProcessResult::Error("convolution: rhs output feature dimension per batch group (" +
                                            std::to_string(outChannelsPerGroup) +
                                            ") must be divisible by feature_group_count (" +
                                            std::to_string(p.featureGroupCount) + ")");
            }

            NSMutableArray<MPSGraphTensor*>* groupResults =
                [NSMutableArray arrayWithCapacity:(NSUInteger)batchGroupCount];
            for (int64_t group = 0; group < batchGroupCount; ++group) {
                int64_t batchStart = group * outBatch;
                int64_t batchEnd = batchStart + outBatch;

                MPSGraphTensor* lhsSlice =
                    [g sliceTensor:convInput
                            starts:@[@(batchStart), @0, @0, @0]
                              ends:@[@(batchEnd), convInput.shape[1], convInput.shape[2], convInput.shape[3]]
                           strides:@[@1, @1, @1, @1]
                              name:nil];

                int64_t outStart = group * outChannelsPerGroup;
                int64_t outEnd = outStart + outChannelsPerGroup;
                MPSGraphTensor* rhsSlice =
                    [g sliceTensor:mpsKernel
                            starts:@[@(outStart), @0, @0, @0]
                              ends:@[@(outEnd), mpsKernel.shape[1], mpsKernel.shape[2], mpsKernel.shape[3]]
                           strides:@[@1, @1, @1, @1]
                              name:nil];

                MPSGraphTensor* groupResult = [g convolution2DWithSourceTensor:lhsSlice
                                                                 weightsTensor:rhsSlice
                                                                    descriptor:makeDescriptor()
                                                                          name:nil];
                [groupResults addObject:groupResult];
            }

            result = [g concatTensors:groupResults dimension:3 name:nil];
        }
    }

    if (is1D) {
        result = convert2DOutputTo1D(g, result, outputBatchDim, outputFeatureDim, output1DSpatialDim);
    } else {
        std::vector<int64_t> outputLayout = {outputBatchDim, outputFeatureDim, outSpatial0, outSpatial1};
        NSArray<NSNumber*>* outputPerm = computePermutation(LAYOUT_NHWC, outputLayout);
        if (outputPerm) {
            result = [g transposeTensor:result permutation:outputPerm name:nil];
        }
    }

    return Result(ctx, result, "convolution");
}
REGISTER_MPS_OP("stablehlo.convolution", HandleConvolution);

} // namespace jax_metallib
