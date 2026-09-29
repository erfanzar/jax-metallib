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

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <sstream>

namespace jax_metallib {

namespace {

bool ReadConstantIntVector(mlir::Value value, std::vector<int64_t>& out) {
    out.clear();
    if (!value) {
        return false;
    }

    auto* defOp = value.getDefiningOp();
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
    if (!constOp) {
        return false;
    }

    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!denseAttr) {
        return false;
    }

    auto elemType = denseAttr.getElementType();
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType);
    if (!intType) {
        return false;
    }

    out.reserve(denseAttr.getNumElements());
    for (llvm::APInt v : denseAttr.getValues<llvm::APInt>()) {

        out.push_back(intType.isUnsigned() ? (int64_t)v.getZExtValue() : v.getSExtValue());
    }
    return true;
}

bool ReadConstantIntScalar(mlir::Value value, int64_t& out) {
    std::vector<int64_t> values;
    if (!ReadConstantIntVector(value, values) || values.size() != 1) {
        return false;
    }
    out = values[0];
    return true;
}

} // namespace

static ProcessResult HandleBroadcast(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("broadcast: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && input.shape && [outputShape isEqualToArray:input.shape]) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    MPSGraphTensor* result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    return Result(ctx, result, "broadcast");
}
REGISTER_MPS_OP("stablehlo.broadcast", HandleBroadcast);

static ProcessResult HandleBroadcastInDim(HandlerContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(ctx.op);
    if (!broadcastOp) {
        return ProcessResult::Error("broadcast_in_dim: expected BroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("broadcast_in_dim: input tensor not found");
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = broadcastOp.getBroadcastDimensions();

    MPSGraphTensor* result = nil;

    if (broadcastDims.empty() || inputRank == outputRank) {
        result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    } else {

        NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
        for (NSUInteger i = 0; i < outputRank; i++) {
            [intermediateShape addObject:@1];
        }

        for (size_t i = 0; i < broadcastDims.size() && i < inputRank; i++) {
            int64_t outDim = broadcastDims[i];
            if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
                intermediateShape[outDim] = inputShape[i];
            }
        }

        MPSGraphTensor* reshaped = [ctx.graph reshapeTensor:input withShape:intermediateShape name:nil];

        result = [ctx.graph broadcastTensor:reshaped toShape:outputShape name:nil];
    }

    return Result(ctx, result, "broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", HandleBroadcastInDim);

static ProcessResult HandleDynamicBroadcastInDim(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::DynamicBroadcastInDimOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("dynamic_broadcast_in_dim: expected DynamicBroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("dynamic_broadcast_in_dim: missing input tensor");
    }

    std::vector<int64_t> outDims;
    if (!ReadConstantIntVector(op.getOutputDimensions(), outDims)) {
        return ProcessResult::Error(
            "dynamic_broadcast_in_dim: dynamic output_dimensions is not supported (must be a "
            "stablehlo.constant)");
    }

    NSMutableArray<NSNumber*>* outputShape =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(outDims.size())];
    for (int64_t d : outDims) {
        if (d < 0) {
            return ProcessResult::Error(
                "dynamic_broadcast_in_dim: dynamic (unknown) output dimension is not supported");
        }
        [outputShape addObject:@(d)];
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;
    NSUInteger outputRank = outputShape.count;

    auto broadcastDims = op.getBroadcastDimensions();

    MPSGraphTensor* result = nil;
    if (broadcastDims.empty() || inputRank == outputRank) {
        result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    } else {
        NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
        for (NSUInteger i = 0; i < outputRank; i++) {
            [intermediateShape addObject:@1];
        }

        for (size_t i = 0; i < broadcastDims.size() && i < inputRank; i++) {
            int64_t outDim = broadcastDims[i];
            if (outDim >= 0 && (NSUInteger)outDim < outputRank) {
                intermediateShape[outDim] = inputShape[i];
            }
        }

        MPSGraphTensor* reshaped = [ctx.graph reshapeTensor:input withShape:intermediateShape name:nil];
        result = [ctx.graph broadcastTensor:reshaped toShape:outputShape name:nil];
    }

    return Result(ctx, result, "dynamic_broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.dynamic_broadcast_in_dim", HandleDynamicBroadcastInDim);

static ProcessResult HandleReshape(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reshape: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && input.shape && [outputShape isEqualToArray:input.shape]) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    // uint16 reshape under JIT corrupted values near 65535 (int16 and other
    // dtypes were exact); reshape the exact int32 widening and cast back.
    if (input.dataType == MPSDataTypeUInt16) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
        wide = [ctx.graph reshapeTensor:wide withShape:outputShape name:nil];
        return Result(ctx, [ctx.graph castTensor:wide toType:MPSDataTypeUInt16 name:nil], "reshape");
    }
    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "reshape");
}
REGISTER_MPS_OP("stablehlo.reshape", HandleReshape);

static ProcessResult HandleDynamicReshape(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::DynamicReshapeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("dynamic_reshape: expected DynamicReshapeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("dynamic_reshape: missing input tensor");
    }

    std::vector<int64_t> outDims;
    if (!ReadConstantIntVector(op.getOutputShape(), outDims)) {
        return ProcessResult::Error("dynamic_reshape: dynamic output_shape is not supported (must "
                                    "be a stablehlo.constant)");
    }

    NSMutableArray<NSNumber*>* outputShape =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(outDims.size())];
    for (int64_t d : outDims) {
        if (d < 0) {
            return ProcessResult::Error(
                "dynamic_reshape: dynamic (unknown) output dimension is not supported");
        }
        [outputShape addObject:@(d)];
    }

    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "dynamic_reshape");
}
REGISTER_MPS_OP("stablehlo.dynamic_reshape", HandleDynamicReshape);

static ProcessResult HandleGetDimensionSize(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::GetDimensionSizeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("get_dimension_size: expected GetDimensionSizeOp");
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    if (!inputType) {
        return ProcessResult::Error("get_dimension_size: expected ranked tensor operand");
    }

    int64_t dim = op.getDimension();
    auto shape = inputType.getShape();
    if (dim < 0 || dim >= (int64_t)shape.size()) {
        return ProcessResult::Error("get_dimension_size: dimension out of bounds");
    }
    if (shape[dim] < 0) {
        return ProcessResult::Error("get_dimension_size: dynamic dimension size is not supported");
    }

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("get_dimension_size: invalid result dtype");
    }

    MPSGraphTensor* result = [ctx.graph constantWithScalar:(double)shape[dim] dataType:dtype];
    return Result(ctx, result, "get_dimension_size");
}
REGISTER_MPS_OP("stablehlo.get_dimension_size", HandleGetDimensionSize);

static ProcessResult HandleSetDimensionSize(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::SetDimensionSizeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("set_dimension_size: expected SetDimensionSizeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("set_dimension_size: missing input tensor");
    }

    int64_t dim = op.getDimension();
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
    if (!inputType) {
        return ProcessResult::Error("set_dimension_size: expected ranked tensor operand");
    }

    auto shape = inputType.getShape();
    if (dim < 0 || dim >= (int64_t)shape.size()) {
        return ProcessResult::Error("set_dimension_size: dimension out of bounds");
    }

    int64_t requested = 0;
    if (shape[dim] >= 0 && ReadConstantIntScalar(op.getSize(), requested) && requested != shape[dim]) {
        return ProcessResult::Error("set_dimension_size: requested size " + std::to_string(requested) +
                                    " does not match static dim size " + std::to_string(shape[dim]));
    }

    SetOutputTensor(ctx.values, ctx.op, input);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.set_dimension_size", HandleSetDimensionSize);

static ProcessResult HandleTranspose(HandlerContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(ctx.op);
    if (!transposeOp) {
        return ProcessResult::Error("transpose: expected TransposeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("transpose: missing input tensor");

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    bool isIdentity = true;
    int64_t i = 0;
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
        if (d != i) {
            isIdentity = false;
        }
        ++i;
    }

    if (isIdentity) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    MPSGraphTensor* result = [ctx.graph transposeTensor:input permutation:perm name:nil];
    return Result(ctx, result, "transpose");
}
REGISTER_MPS_OP("stablehlo.transpose", HandleTranspose);

static bool IsHalfWidthFloat(MPSDataType t) {
    return t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
}

// Under jit MPSGraph folded narrowing cast chains: x.astype(f16).astype(f32)
// became the identity (70000 stayed 70000 instead of inf, pi kept its f32
// bits), and f16 arithmetic between the casts ran in float32. A reinterpret
// round trip through int16 on the 16-bit side is not folded (verified with a
// standalone MPSGraph program at Level0). Rank-0 is reshaped to [1] because
// reinterpret casts abort on rank 0. JAX_METALLIB_DISABLE_CAST_FOLD_BARRIER=1
// restores the plain cast.
static MPSGraphTensor* CastFoldBarrier(MPSGraph* g, MPSGraphTensor* t) {
    NSArray<NSNumber*>* shape = t.shape;
    if (shape == nil) {
        return t;
    }
    const bool scalar = shape.count == 0;
    MPSGraphTensor* x = scalar ? [g reshapeTensor:t withShape:@[@1] name:nil] : t;
    MPSGraphTensor* bits = [g reinterpretCastTensor:x toType:MPSDataTypeInt16 name:nil];
    x = [g reinterpretCastTensor:bits toType:t.dataType name:nil];
    return scalar ? [g reshapeTensor:x withShape:@[] name:nil] : x;
}

static ProcessResult HandleConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("convert: invalid dtype for convert operation");
    }
    if (input.dataType == dtype) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    static const bool barrierDisabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_CAST_FOLD_BARRIER");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    MPSGraph* g = ctx.graph;
    if (!barrierDisabled && IsHalfWidthFloat(input.dataType)) {
        input = CastFoldBarrier(g, input);
    }
    MPSGraphTensor* result = [g castTensor:input toType:dtype name:nil];
    if (!barrierDisabled && IsHalfWidthFloat(dtype)) {
        result = CastFoldBarrier(g, result);
    }
    return Result(ctx, result, "convert");
}
REGISTER_MPS_OP("stablehlo.convert", HandleConvert);

static ProcessResult HandleSlice(HandlerContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(ctx.op);
    if (!sliceOp) {
        return ProcessResult::Error("slice: expected SliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("slice: missing input tensor");

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceOp.getStartIndices()) {
        [starts addObject:@(s)];
    }
    for (int64_t l : sliceOp.getLimitIndices()) {
        [ends addObject:@(l)];
    }
    for (int64_t s : sliceOp.getStrides()) {
        [strides addObject:@(s)];
    }

    // uint16 slice under JIT corrupted values near 65535 (int16 and other
    // dtypes were exact); slice the exact int32 widening and cast back.
    if (input.dataType == MPSDataTypeUInt16) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
        wide = [ctx.graph sliceTensor:wide starts:starts ends:ends strides:strides name:nil];
        return Result(ctx, [ctx.graph castTensor:wide toType:MPSDataTypeUInt16 name:nil], "slice");
    }
    MPSGraphTensor* result = [ctx.graph sliceTensor:input starts:starts ends:ends strides:strides name:nil];
    return Result(ctx, result, "slice");
}
REGISTER_MPS_OP("stablehlo.slice", HandleSlice);

static ProcessResult HandleDynamicSliceImpl(HandlerContext& ctx) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(ctx.op);
    if (!dynSliceOp) {
        return ProcessResult::Error("dynamic_slice: expected DynamicSliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("dynamic_slice: missing input tensor");

    auto sliceSizes = dynSliceOp.getSliceSizes();
    NSUInteger rank = sliceSizes.size();
    NSArray<NSNumber*>* inputShape = input.shape;
    if (inputShape.count != rank) {
        return ProcessResult::Error("dynamic_slice: input rank does not match slice_sizes rank");
    }

    NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
    for (int64_t s : sliceSizes) {
        [outputShape addObject:@(s)];
    }

    std::vector<int64_t> startConst(rank, 0);
    std::vector<bool> startIsConst(rank, false);
    bool allConst = true;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        int64_t v = 0;
        if (ReadConstantIntScalar(ctx.op->getOperand(dim + 1), v)) {
            startConst[dim] = v;
            startIsConst[dim] = true;
        } else {
            startIsConst[dim] = false;
            allConst = false;
        }
    }
    if (allConst) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            int64_t inputDim = [inputShape[dim] longLongValue];
            int64_t sliceSize = sliceSizes[dim];
            if (inputDim < 0) {
                return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
            }
            if (sliceSize < 0) {
                return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
            }
            int64_t maxStart = inputDim - sliceSize;
            if (maxStart < 0) {
                return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
            }
            int64_t s = std::min<int64_t>(std::max<int64_t>(startConst[dim], 0), maxStart);
            [starts addObject:@(s)];
            [ends addObject:@(s + sliceSize)];
            [strides addObject:@1];
        }
        MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                                 starts:starts
                                                   ends:ends
                                                strides:strides
                                                   name:nil];
        return Result(ctx, result, "dynamic_slice");
    }

    if (rank > 0) {
        if (@available(macOS 15.2, *)) {
            NSMutableArray<MPSGraphTensor*>* startsScalars = [NSMutableArray arrayWithCapacity:rank];
            std::vector<int32_t> sizes32;
            sizes32.reserve(rank);

            for (NSUInteger dim = 0; dim < rank; ++dim) {
                MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
                if (!startIdx) {
                    return ProcessResult::Error("dynamic_slice: missing start index for dimension");
                }

                startIdx = EnsureInt32(ctx.graph, startIdx);
                int64_t inputDim = [inputShape[dim] longLongValue];
                int64_t sliceSize = sliceSizes[dim];
                if (inputDim < 0) {
                    return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
                }
                if (sliceSize < 0) {
                    return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
                }
                int64_t maxStart = inputDim - sliceSize;
                if (maxStart < 0) {
                    return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
                }

                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:startIdx.dataType];
                startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                               secondaryTensor:maxStartTensor
                                                          name:nil];
                [startsScalars addObject:startIdx];

                sizes32.push_back((int32_t)sliceSize);
            }

            MPSGraphTensor* startsTensor = [ctx.graph stackTensors:startsScalars axis:0 name:nil];
            startsTensor = EnsureInt32(ctx.graph, startsTensor);

            NSData* sizeData = [NSData dataWithBytes:sizes32.data()
                                              length:(sizes32.size() * sizeof(int32_t))];
            MPSGraphTensor* sizeTensor = [ctx.graph constantWithData:sizeData
                                                               shape:@[@(rank)]
                                                            dataType:MPSDataTypeInt32];

            MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                                startTensor:startsTensor
                                                 sizeTensor:sizeTensor
                                                squeezeMask:0
                                                       name:nil];
            return Result(ctx, result, "dynamic_slice");
        }
    }

    int64_t dynamicDim = -1;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        if (!startIsConst[dim]) {
            if (dynamicDim != -1) {
                dynamicDim = -2;
                break;
            }
            dynamicDim = (int64_t)dim;
        }
    }
    if (dynamicDim >= 0) {
        bool fullOtherDims = true;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            if ((int64_t)dim == dynamicDim) {
                continue;
            }
            int64_t inputDim = [inputShape[dim] longLongValue];
            int64_t sliceSize = sliceSizes[dim];
            if (inputDim < 0 || sliceSize < 0 || sliceSize != inputDim) {
                fullOtherDims = false;
                break;
            }
        }
        if (fullOtherDims) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dynamicDim + 1));
            if (!startIdx) {
                return ProcessResult::Error("dynamic_slice: missing start index for dimension");
            }

            startIdx = EnsureInt32(ctx.graph, startIdx);
            int64_t inputDim = [inputShape[(NSUInteger)dynamicDim] longLongValue];
            int64_t sliceSize = sliceSizes[(NSUInteger)dynamicDim];
            if (inputDim < 0) {
                return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
            }
            if (sliceSize < 0) {
                return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
            }
            int64_t maxStart = inputDim - sliceSize;
            if (maxStart < 0) {
                return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
            MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                  dataType:startIdx.dataType];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

            MPSGraphTensor* indices = nil;
            if (sliceSize == 1) {
                indices = [ctx.graph broadcastTensor:startIdx toShape:outputShape name:nil];
            } else {
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dynamicDim
                                                              withShape:outputShape
                                                                   name:nil];
                coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];
                indices = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:startIdx name:nil];
            }
            indices = EnsureInt32(ctx.graph, indices);

            MPSGraphTensor* result = [ctx.graph gatherAlongAxis:(NSInteger)dynamicDim
                                              withUpdatesTensor:input
                                                  indicesTensor:indices
                                                           name:nil];
            return Result(ctx, result, "dynamic_slice");
        }
    }

    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {

        MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_slice: missing start index for dimension");
        }

        startIdx = EnsureInt32(ctx.graph, startIdx);
        int64_t inputDim = [inputShape[dim] longLongValue];
        int64_t sliceSize = sliceSizes[dim];
        if (sliceSize < 0) {
            return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
        }
        int64_t maxStart = inputDim - sliceSize;
        if (maxStart < 0) {
            return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
        }
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
        MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart dataType:startIdx.dataType];
        startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
        startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:outputShape
                                                           name:nil];
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];
    indices = EnsureInt32(ctx.graph, indices);

    MPSGraphTensor* result = [ctx.graph gatherNDWithUpdatesTensor:input
                                                    indicesTensor:indices
                                                  batchDimensions:0
                                                             name:nil];
    return Result(ctx, result, "dynamic_slice");
}
// Eager dynamic_slice with runtime start indices on float16/int8/uint8
// operands segfaulted the process (exit 139, alongside ANE compile errors),
// while float32 operands and every JIT form were correct. Widen 8/16-bit
// operands exactly (f16 -> f32, small ints -> int32), slice, and cast back.
static ProcessResult HandleDynamicSlice(HandlerContext& ctx) {
    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    if (!operand || ctx.op->getNumResults() != 1) {
        return HandleDynamicSliceImpl(ctx);
    }
    MPSDataType dtype = operand.dataType;
    MPSDataType wideType = MPSDataTypeInvalid;
    if (dtype == MPSDataTypeFloat16) {
        wideType = MPSDataTypeFloat32;
    } else if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeInt16 ||
               dtype == MPSDataTypeUInt16) {
        wideType = MPSDataTypeInt32;
    }
    if (wideType == MPSDataTypeInvalid) {
        return HandleDynamicSliceImpl(ctx);
    }
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    if (ctx.variables && ctx.variables->find(operandKey) != ctx.variables->end()) {
        return HandleDynamicSliceImpl(ctx);
    }
    for (unsigned i = 1; i < ctx.op->getNumOperands(); ++i) {
        if (ctx.op->getOperand(i) == ctx.op->getOperand(0)) {
            return HandleDynamicSliceImpl(ctx);
        }
    }

    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;
    ctx.values[operandKey] = [ctx.graph castTensor:operand toType:wideType name:nil];
    ProcessResult r = HandleDynamicSliceImpl(ctx);
    if (original) {
        ctx.values[operandKey] = original;
    } else {
        ctx.values.erase(operandKey);
    }
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* sliced = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!sliced || sliced.dataType != wideType) {
        ctx.values.erase(ctx.op->getResult(0).getAsOpaquePointer());
        return HandleDynamicSliceImpl(ctx);
    }
    return Result(ctx, [ctx.graph castTensor:sliced toType:dtype name:nil], "dynamic_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", HandleDynamicSlice);

static ProcessResult HandleBitcastConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("bitcast_convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("bitcast_convert: invalid dtype");
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {

        input = [ctx.graph reshapeTensor:input withShape:@[@1] name:nil];
    }

    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:input toType:dtype name:nil];

    // A scalar source can bitcast to a vector when the destination element is
    // narrower (e.g. f32 -> 2xu16); only restore the scalar shape if requested.
    if (isScalar && GetOutputShape(ctx.op).count == 0) {
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(ctx, result, "bitcast_convert");
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", HandleBitcastConvert);

static ProcessResult HandleConcatenate(HandlerContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(ctx.op);
    if (!concatOp) {
        return ProcessResult::Error("concatenate: expected ConcatenateOp");
    }

    unsigned numOperands = ctx.op->getNumOperands();
    if (numOperands == 0) {
        return ProcessResult::Error("concatenate: no operands");
    }
    if (numOperands == 1) {
        MPSGraphTensor* only = GetInputTensor(ctx, 0);
        if (!only) {
            return ProcessResult::Error("concatenate: missing input tensor at operand 0");
        }
        SetOutputTensor(ctx.values, ctx.op, only);
        return ProcessResult{};
    }

    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    // Under jit, MPSGraph concatTensors with a zero-size operand in a trailing
    // or middle position shifted the result (rows offset by one element),
    // which broke square jnp.linalg.svd(full_matrices=True). Zero-size operands
    // contribute nothing, so drop them; if all are empty, keep the first (its
    // shape equals the result shape).
    NSMutableArray<MPSGraphTensor*>* input_tensors =
        [NSMutableArray arrayWithCapacity:(NSUInteger)numOperands];
    MPSGraphTensor* firstTensor = nil;
    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* tensor = GetInputTensor(ctx, i);
        if (!tensor) {
            return ProcessResult::Error("concatenate: missing input tensor at operand " + std::to_string(i));
        }
        if (!firstTensor) {
            firstTensor = tensor;
        }
        auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(i).getType());
        if (operandType && dimension >= 0 && dimension < (NSInteger)operandType.getRank() &&
            operandType.getDimSize(dimension) == 0) {
            continue;
        }
        [input_tensors addObject:tensor];
    }

    if (input_tensors.count == 0) {
        SetOutputTensor(ctx.values, ctx.op, firstTensor);
        return ProcessResult{};
    }
    if (input_tensors.count == 1) {
        SetOutputTensor(ctx.values, ctx.op, input_tensors[0]);
        return ProcessResult{};
    }

    MPSGraphTensor* result = [ctx.graph concatTensors:input_tensors dimension:dimension name:nil];
    return Result(ctx, result, "concatenate");
}
REGISTER_MPS_OP("stablehlo.concatenate", HandleConcatenate);

static ProcessResult HandleSharding(HandlerContext& ctx) {
    unsigned numOperands = ctx.op->getNumOperands();
    unsigned numResults = ctx.op->getNumResults();

    if (numOperands != numResults) {
        return ProcessResult::Error(
            "sharding: operand/result arity mismatch (operands=" + std::to_string(numOperands) +
            ", results=" + std::to_string(numResults) + ")");
    }

    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* input = GetInputTensor(ctx, i);
        if (!input) {
            return ProcessResult::Error("sharding: missing input tensor at operand " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, input, i);
    }
    return ProcessResult{};
}
REGISTER_CUSTOM_CALL("Sharding", HandleSharding, sharding);

static ProcessResult HandlePad(HandlerContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(ctx.op);
    if (!padOp) {
        return ProcessResult::Error("pad: expected PadOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(ctx, 1);
    if (!input || !paddingValue)
        return ProcessResult::Error("pad: missing input tensor");

    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();
    if (edgePaddingLow.size() != edgePaddingHigh.size()) {
        return ProcessResult::Error("pad: edge padding rank mismatch");
    }
    if (edgePaddingLow.size() != interiorPadding.size()) {
        return ProcessResult::Error("pad: interior padding rank mismatch");
    }

    bool hasInteriorPadding = false;
    for (int64_t p : interiorPadding) {
        if (p < 0) {
            return ProcessResult::Error("pad: negative interior padding is not supported");
        }
        if (p != 0) {
            hasInteriorPadding = true;
            break;
        }
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    if (inputShape.count != edgePaddingLow.size()) {
        return ProcessResult::Error("pad: input rank does not match padding rank");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);

    // An all-zero pad is an identity, regardless of the padding value. Avoid
    // MPSGraph's slice-update of a broadcast tensor, which can discard complex
    // values produced by a preceding FFT in JIT gradients.
    bool noPadding = !hasInteriorPadding;
    for (size_t i = 0; i < edgePaddingLow.size(); ++i) {
        noPadding &= edgePaddingLow[i] == 0 && edgePaddingHigh[i] == 0;
    }
    if (noPadding && [inputShape isEqualToArray:outputShape]) {
        return Result(ctx, input, "pad");
    }

    // MPSGraph scatterND rejects complex tensors (aborting the process), and
    // slice-update into a broadcast complex base can drop the written values.
    // Build complex pads one axis at a time from reshape/slice/concat instead.
    bool complexInput =
        input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16;
    bool complexNonEmpty = complexInput && outputShape.count == inputShape.count;
    for (NSNumber* d : inputShape) {
        complexNonEmpty &= [d longLongValue] > 0;
    }
    for (NSNumber* d : outputShape) {
        complexNonEmpty &= [d longLongValue] > 0;
    }
    if (complexNonEmpty) {
        auto sliceAxis = [&](MPSGraphTensor* t, NSUInteger axis, int64_t start, int64_t end) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:t.shape.count];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:t.shape.count];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:t.shape.count];
            for (NSUInteger d = 0; d < t.shape.count; ++d) {
                [starts addObject:d == axis ? @(start) : @0];
                [ends addObject:d == axis ? @(end) : t.shape[d]];
                [strides addObject:@1];
            }
            return [ctx.graph sliceTensor:t starts:starts ends:ends strides:strides name:nil];
        };
        auto fill = [&](MPSGraphTensor* t, NSUInteger axis, int64_t count) {
            NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:t.shape];
            shape[axis] = @(count);
            return [ctx.graph broadcastTensor:paddingValue toShape:shape name:nil];
        };

        MPSGraphTensor* current = input;
        for (NSUInteger axis = 0; axis < inputShape.count; ++axis) {
            int64_t size = [current.shape[axis] longLongValue];
            int64_t interior = interiorPadding[axis];
            if (interior > 0 && size > 1) {
                NSMutableArray<NSNumber*>* expanded = [NSMutableArray arrayWithArray:current.shape];
                [expanded insertObject:@1 atIndex:axis + 1];
                MPSGraphTensor* values = [ctx.graph reshapeTensor:current withShape:expanded name:nil];
                MPSGraphTensor* gaps = fill(values, axis + 1, interior);
                MPSGraphTensor* joined = [ctx.graph concatTensors:@[values, gaps]
                                                        dimension:(NSInteger)(axis + 1)
                                                             name:nil];
                NSMutableArray<NSNumber*>* flat = [NSMutableArray arrayWithArray:current.shape];
                flat[axis] = @(size * (interior + 1));
                current = [ctx.graph reshapeTensor:joined withShape:flat name:nil];
                size = size + (size - 1) * interior;
                current = sliceAxis(current, axis, 0, size);
            }
            int64_t low = edgePaddingLow[axis];
            int64_t high = edgePaddingHigh[axis];
            if (low < 0 || high < 0) {
                current =
                    sliceAxis(current, axis, std::max<int64_t>(-low, 0), size - std::max<int64_t>(-high, 0));
            }
            NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
            if (low > 0) {
                [parts addObject:fill(current, axis, low)];
            }
            [parts addObject:current];
            if (high > 0) {
                [parts addObject:fill(current, axis, high)];
            }
            if (parts.count > 1) {
                current = [ctx.graph concatTensors:parts dimension:(NSInteger)axis name:nil];
            }
        }
        if (![current.shape isEqualToArray:outputShape]) {
            return ProcessResult::Error("pad: complex pad produced an unexpected output shape");
        }
        return Result(ctx, current, "pad");
    }

    auto getScalarPaddingValueDouble = [&]() -> std::optional<double> {
        auto* defOp = padOp.getPaddingValue().getDefiningOp();
        auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
        if (!constOp) {
            return std::nullopt;
        }
        auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
        if (!denseAttr || denseAttr.getNumElements() != 1) {
            return std::nullopt;
        }
        auto elemType = denseAttr.getElementType();
        if (elemType.isF16() || elemType.isF32() || elemType.isF64()) {
            llvm::APFloat ap = *denseAttr.getValues<llvm::APFloat>().begin();
            return ap.convertToDouble();
        }
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
            llvm::APInt ap = *denseAttr.getValues<llvm::APInt>().begin();

            if (intType.isUnsigned()) {
                return static_cast<double>(ap.getZExtValue());
            }
            return static_cast<double>(ap.getSExtValue());
        }
        return std::nullopt;
    };

    auto canUseNativeEdgePad = [&](bool needsInputSlice, bool hasNonEmptyUpdate) -> bool {
        if (needsInputSlice || !hasNonEmptyUpdate) {
            return false;
        }

        auto padConst = getScalarPaddingValueDouble();
        if (!padConst.has_value()) {
            return false;
        }

        for (size_t i = 0; i < edgePaddingLow.size(); ++i) {
            if (edgePaddingLow[i] < 0 || edgePaddingHigh[i] < 0) {
                return false;
            }
        }

        for (NSNumber* d : inputShape) {
            if ([d longLongValue] == 0) {
                return false;
            }
        }
        for (NSNumber* d : outputShape) {
            if ([d longLongValue] == 0) {
                return false;
            }
        }
        return true;
    };

    MPSGraphTensor* padded = nil;
    auto getPadded = [&]() -> MPSGraphTensor* {
        if (!padded) {
            padded = [ctx.graph broadcastTensor:paddingValue toShape:outputShape name:nil];
        }
        return padded;
    };

    if (hasInteriorPadding) {
        NSUInteger rank = inputShape.count;
        NSMutableArray<NSNumber*>* preCropShape = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropStarts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropEnds = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropStrides = [NSMutableArray arrayWithCapacity:rank];

        bool hasElements = true;
        bool needsCrop = false;
        for (NSUInteger i = 0; i < rank; ++i) {
            int64_t low = edgePaddingLow[i];
            int64_t high = edgePaddingHigh[i];
            int64_t interior = interiorPadding[i];
            int64_t inputDim = [inputShape[i] longLongValue];
            int64_t outputDim = [outputShape[i] longLongValue];
            if (inputDim < 0 || outputDim < 0) {
                return ProcessResult::Error("pad: negative input/output dimension");
            }

            int64_t baseDim = 0;
            if (inputDim > 0) {
                baseDim = inputDim + (inputDim - 1) * interior;
            }
            int64_t preDim = std::max<int64_t>(low, 0) + baseDim + std::max<int64_t>(high, 0);
            int64_t cropLow = std::max<int64_t>(-low, 0);
            int64_t cropHigh = std::max<int64_t>(-high, 0);
            int64_t expectedOut = preDim - cropLow - cropHigh;
            if (expectedOut != outputDim) {
                return ProcessResult::Error("pad: inconsistent output shape for padding config");
            }

            if (low < 0 || high < 0) {
                needsCrop = true;
            }
            if (inputDim == 0 || outputDim == 0) {
                hasElements = false;
            }
            [preCropShape addObject:@(preDim)];
            [cropStarts addObject:@(cropLow)];
            [cropEnds addObject:@(cropLow + outputDim)];
            [cropStrides addObject:@1];
        }

        // MPSGraph's slice-update can silently discard interior-pad input values.
        // For a float32 row-only dilation, concatenate each row with its padding
        // instead of allocating and scattering a rank-vector index per element.
        if (!ctx.isInControlFlow() && rank == 2 && hasElements && !needsCrop &&
            input.dataType == MPSDataTypeFloat32 && [inputShape[0] longLongValue] > 1 &&
            interiorPadding[0] > 0 && interiorPadding[1] == 0 && edgePaddingLow[0] == 0 &&
            edgePaddingHigh[0] == 0 && edgePaddingLow[1] == 0 && edgePaddingHigh[1] == 0) {
            int64_t rows = [inputShape[0] longLongValue];
            int64_t cols = [inputShape[1] longLongValue];
            int64_t step = interiorPadding[0] + 1;
            MPSGraphTensor* rowValues = [ctx.graph reshapeTensor:input
                                                       withShape:@[@(rows), @1, @(cols)]
                                                            name:nil];
            MPSGraphTensor* gaps = [ctx.graph broadcastTensor:paddingValue
                                                      toShape:@[@(rows), @(interiorPadding[0]), @(cols)]
                                                         name:nil];
            MPSGraphTensor* interleaved = [ctx.graph concatTensors:@[rowValues, gaps] dimension:1 name:nil];
            MPSGraphTensor* flat = [ctx.graph reshapeTensor:interleaved
                                                  withShape:@[@(rows * step), @(cols)]
                                                       name:nil];
            MPSGraphTensor* result = [ctx.graph sliceTensor:flat
                                                     starts:@[@0, @0]
                                                       ends:outputShape
                                                    strides:@[@1, @1]
                                                       name:nil];
            return Result(ctx, result, "pad");
        }

        MPSGraphTensor* prePadded = [ctx.graph broadcastTensor:paddingValue toShape:preCropShape name:nil];

        if (!hasElements) {
            MPSGraphTensor* result = prePadded;
            if (needsCrop) {
                result = [ctx.graph sliceTensor:result
                                         starts:cropStarts
                                           ends:cropEnds
                                        strides:cropStrides
                                           name:nil];
            }
            if (outputShape && result) {
                result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
            }
            return Result(ctx, result, "pad");
        }

        NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                          withShape:inputShape
                                                               name:nil];
            int64_t interior = interiorPadding[dim];
            int64_t step = interior + 1;
            if (step != 1) {
                MPSGraphTensor* stepTensor = [ctx.graph constantWithScalar:step dataType:coords.dataType];
                coords = [ctx.graph multiplicationWithPrimaryTensor:coords
                                                    secondaryTensor:stepTensor
                                                               name:nil];
            }
            int64_t low = edgePaddingLow[dim];
            int64_t shift = std::max<int64_t>(low, 0);
            if (shift != 0) {
                MPSGraphTensor* shiftTensor = [ctx.graph constantWithScalar:shift dataType:coords.dataType];
                coords = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:shiftTensor name:nil];
            }
            [indexTensors addObject:coords];
        }

        MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];
        indices = EnsureInt32(ctx.graph, indices);
        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:prePadded
                                                      updatesTensor:input
                                                      indicesTensor:indices
                                                    batchDimensions:0
                                                               mode:MPSGraphScatterModeSet
                                                               name:nil];
        if (needsCrop) {
            result = [ctx.graph sliceTensor:result
                                     starts:cropStarts
                                       ends:cropEnds
                                    strides:cropStrides
                                       name:nil];
        }
        if (outputShape && result) {
            result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
        }
        return Result(ctx, result, "pad");
    }

    NSMutableArray<NSNumber*>* inputSliceStarts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* inputSliceEnds = [NSMutableArray array];
    NSMutableArray<NSNumber*>* inputSliceStrides = [NSMutableArray array];

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    bool needsInputSlice = false;
    bool hasNonEmptyUpdate = true;
    for (NSUInteger i = 0; i < edgePaddingLow.size(); i++) {
        int64_t low = edgePaddingLow[i];
        int64_t high = edgePaddingHigh[i];
        int64_t inputDim = [inputShape[i] longLongValue];
        int64_t cropLow = std::max<int64_t>(-low, 0);
        int64_t cropHigh = std::max<int64_t>(-high, 0);
        int64_t sliceStart = cropLow;
        int64_t sliceEnd = inputDim - cropHigh;
        if (sliceEnd < sliceStart) {
            sliceEnd = sliceStart;
        }
        int64_t slicedDim = sliceEnd - sliceStart;
        if (slicedDim == 0) {
            hasNonEmptyUpdate = false;
        }
        if (sliceStart != 0 || sliceEnd != inputDim) {
            needsInputSlice = true;
        }
        [inputSliceStarts addObject:@(sliceStart)];
        [inputSliceEnds addObject:@(sliceEnd)];
        [inputSliceStrides addObject:@1];

        int64_t outputStart = std::max<int64_t>(low, 0);
        [starts addObject:@(outputStart)];
        [ends addObject:@(outputStart + slicedDim)];
        [strides addObject:@1];
    }

    if (!hasNonEmptyUpdate) {
        return Result(ctx, getPadded(), "pad");
    }

    if (canUseNativeEdgePad(needsInputSlice, hasNonEmptyUpdate)) {
        auto padConst = getScalarPaddingValueDouble();
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:inputShape.count];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:inputShape.count];
        for (NSUInteger i = 0; i < edgePaddingLow.size(); ++i) {
            [left addObject:@(edgePaddingLow[i])];
            [right addObject:@(edgePaddingHigh[i])];
        }
        MPSGraphTensor* out = [ctx.graph padTensor:input
                                   withPaddingMode:MPSGraphPaddingModeConstant
                                       leftPadding:left
                                      rightPadding:right
                                     constantValue:*padConst
                                              name:nil];
        return Result(ctx, out, "pad");
    }

    auto isScalarLike = [](NSArray<NSNumber*>* shape) -> bool {
        if (!shape || shape.count == 0) {
            return true;
        }
        for (NSNumber* d : shape) {
            if ([d longLongValue] != 1) {
                return false;
            }
        }
        return true;
    };
    auto isFloatDType = [](MPSDataType t) -> bool {
        return t == MPSDataTypeFloat16 || t == MPSDataTypeFloat32 || t == MPSDataTypeBFloat16;
    };
    if (!needsInputSlice && hasNonEmptyUpdate && isFloatDType(input.dataType) &&
        isScalarLike(paddingValue.shape)) {
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:inputShape.count];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:inputShape.count];
        for (NSUInteger i = 0; i < edgePaddingLow.size(); ++i) {
            int64_t low = edgePaddingLow[i];
            int64_t high = edgePaddingHigh[i];
            if (low < 0 || high < 0) {
                left = nil;
                right = nil;
                break;
            }
            [left addObject:@(low)];
            [right addObject:@(high)];
        }
        if (left && right) {
            MPSGraphTensor* padScalar = paddingValue;
            if (padScalar.dataType != input.dataType) {
                padScalar = [ctx.graph castTensor:padScalar toType:input.dataType name:nil];
            }
            if (padScalar.shape && padScalar.shape.count != 0) {
                padScalar = [ctx.graph reshapeTensor:padScalar withShape:@[] name:nil];
            }

            MPSGraphTensor* shifted = [ctx.graph subtractionWithPrimaryTensor:input
                                                              secondaryTensor:padScalar
                                                                         name:nil];
            MPSGraphTensor* shiftedPadded = [ctx.graph padTensor:shifted
                                                 withPaddingMode:MPSGraphPaddingModeConstant
                                                     leftPadding:left
                                                    rightPadding:right
                                                   constantValue:0.0
                                                            name:nil];
            MPSGraphTensor* out = [ctx.graph additionWithPrimaryTensor:shiftedPadded
                                                       secondaryTensor:padScalar
                                                                  name:nil];
            return Result(ctx, out, "pad");
        }
    }

    MPSGraphTensor* updateTensor = input;
    if (needsInputSlice) {
        updateTensor = [ctx.graph sliceTensor:input
                                       starts:inputSliceStarts
                                         ends:inputSliceEnds
                                      strides:inputSliceStrides
                                         name:nil];
    }

    MPSGraphTensor* result = [ctx.graph sliceUpdateDataTensor:getPadded()
                                                 updateTensor:updateTensor
                                                       starts:starts
                                                         ends:ends
                                                      strides:strides
                                                         name:nil];
    return Result(ctx, result, "pad");
}
REGISTER_MPS_OP("stablehlo.pad", HandlePad);

static ProcessResult HandleDynamicUpdateSlice(HandlerContext& ctx) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(ctx.op);
    if (!updateSliceOp) {
        return ProcessResult::Error("dynamic_update_slice: expected DynamicUpdateSliceOp");
    }

    MPSGraphTensor* baseVariable = nil;
    if (ctx.variables) {
        auto it = ctx.variables->find(ctx.op->getOperand(0).getAsOpaquePointer());
        if (it != ctx.variables->end()) {
            baseVariable = it->second;
        }
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* update = GetInputTensor(ctx, 1);
    if (!operand || !update)
        return ProcessResult::Error("dynamic_update_slice: missing input tensor");

    NSArray<NSNumber*>* operandShape = operand.shape;
    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;
    if (operandShape.count != rank) {
        return ProcessResult::Error("dynamic_update_slice: operand rank does not match update rank");
    }

    MPSGraphTensor* outTensor = nil;
    // MPSGraph's slice-update may silently ignore the update when its data
    // tensor is a scalar broadcast (notably in JIT gradients). ScatterND
    // handles that input while preserving the same clamped index semantics.
    // Only route float32 here: MPSGraph's scatterND rejects complex tensors.
    bool scatterBroadcastOperand =
        !ctx.isInControlFlow() && operand.dataType == MPSDataTypeFloat32 &&
        static_cast<bool>(
            mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(ctx.op->getOperand(0).getDefiningOp()));

    std::vector<int64_t> startConst(rank, 0);
    std::vector<bool> startIsConst(rank, false);
    bool allConst = true;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        int64_t v = 0;
        if (ReadConstantIntScalar(ctx.op->getOperand(dim + 2), v)) {
            startConst[dim] = v;
            startIsConst[dim] = true;
        } else {
            startIsConst[dim] = false;
            allConst = false;
        }
    }
    if (allConst && !scatterBroadcastOperand) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            if (operandDim < 0 || updateDim < 0) {
                return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
            }
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            int64_t s = std::min<int64_t>(std::max<int64_t>(startConst[dim], 0), maxStart);
            [starts addObject:@(s)];
            [ends addObject:@(s + updateDim)];
            [strides addObject:@1];
        }
        outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                        updateTensor:update
                                              starts:starts
                                                ends:ends
                                             strides:strides
                                                name:nil];
    }

    // Path A: sliceUpdateDataTensor with tensor starts (macOS 14.4+).
    // Only used outside control flow; inside while_loop bodies, MPSGraph internally
    // lowers this to a scatter op that fails shape verification.
    if (!outTensor && !ctx.isInControlFlow() && !scatterBroadcastOperand) {

        if (rank > 0) {
            if (@available(macOS 14.4, *)) {
                NSMutableArray<MPSGraphTensor*>* startsScalars = [NSMutableArray arrayWithCapacity:rank];
                std::vector<int32_t> sizes32;
                sizes32.reserve(rank);

                for (NSUInteger dim = 0; dim < rank; ++dim) {
                    MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 2);
                    if (!startIdx) {
                        return ProcessResult::Error("dynamic_update_slice: missing start index");
                    }
                    startIdx = EnsureInt32(ctx.graph, startIdx);

                    int64_t operandDim = [operandShape[dim] longLongValue];
                    int64_t updateDim = [updateShape[dim] longLongValue];
                    if (operandDim < 0 || updateDim < 0) {
                        return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
                    }
                    int64_t maxStart = operandDim - updateDim;
                    if (maxStart < 0) {
                        return ProcessResult::Error(
                            "dynamic_update_slice: update dimension exceeds operand dimension");
                    }

                    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                    MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                          dataType:startIdx.dataType];
                    startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                    startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                                   secondaryTensor:maxStartTensor
                                                              name:nil];
                    [startsScalars addObject:startIdx];

                    sizes32.push_back((int32_t)updateDim);
                }

                MPSGraphTensor* startsTensor = [ctx.graph stackTensors:startsScalars axis:0 name:nil];
                startsTensor = EnsureInt32(ctx.graph, startsTensor);

                NSData* sizeData = [NSData dataWithBytes:sizes32.data()
                                                  length:(sizes32.size() * sizeof(int32_t))];
                MPSGraphTensor* sizeTensor = [ctx.graph constantWithData:sizeData
                                                                   shape:@[@(rank)]
                                                                dataType:MPSDataTypeInt32];

                MPSGraphTensor* endsTensor = [ctx.graph additionWithPrimaryTensor:startsTensor
                                                                  secondaryTensor:sizeTensor
                                                                             name:nil];
                MPSGraphTensor* stridesTensor = [ctx.graph constantWithScalar:1
                                                                        shape:@[@(rank)]
                                                                     dataType:MPSDataTypeInt32];

                if (@available(macOS 15.0, *)) {

                    outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                                    updateTensor:update
                                                    startsTensor:startsTensor
                                                      endsTensor:endsTensor
                                                   stridesTensor:stridesTensor
                                                            name:nil];
                } else {
                    outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                                    updateTensor:update
                                                    startsTensor:startsTensor
                                                      endsTensor:endsTensor
                                                   stridesTensor:stridesTensor
                                                       startMask:0
                                                         endMask:0
                                                     squeezeMask:0
                                                            name:nil];
                }
            }
        }
    }

    // Path B: scatterAlongAxis. Inside loop bodies it was verified exact (float32
    // and int32 > 2**24, NaN/inf) and ~2-3x faster than the Path C mask for
    // multi-row updates, but slower for single-row updates, so in control flow
    // it is used only for float32/int32 updates spanning more than one row.
    if (!outTensor) {

        int64_t dynamicDim = -1;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            if (!startIsConst[dim]) {
                if (dynamicDim != -1) {
                    dynamicDim = -2;
                    break;
                }
                dynamicDim = (int64_t)dim;
            }
        }
        if (dynamicDim >= 0) {
            bool fullOtherDims = true;
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                if ((int64_t)dim == dynamicDim) {
                    continue;
                }
                int64_t operandDim = [operandShape[dim] longLongValue];
                int64_t updateDim = [updateShape[dim] longLongValue];
                if (operandDim < 0 || updateDim < 0 || operandDim != updateDim) {
                    fullOtherDims = false;
                    break;
                }
            }
            if (fullOtherDims && ctx.isInControlFlow()) {
                const bool loopDtypeOk =
                    operand.dataType == MPSDataTypeFloat32 || operand.dataType == MPSDataTypeInt32;
                if (!loopDtypeOk || [updateShape[(NSUInteger)dynamicDim] longLongValue] <= 1) {
                    fullOtherDims = false;
                }
            }
            if (fullOtherDims) {
                MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dynamicDim + 2));
                if (!startIdx) {
                    return ProcessResult::Error("dynamic_update_slice: missing start index");
                }
                startIdx = EnsureInt32(ctx.graph, startIdx);

                int64_t operandDim = [operandShape[(NSUInteger)dynamicDim] longLongValue];
                int64_t updateDim = [updateShape[(NSUInteger)dynamicDim] longLongValue];
                if (operandDim < 0 || updateDim < 0) {
                    return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
                }
                int64_t maxStart = operandDim - updateDim;
                if (maxStart < 0) {
                    return ProcessResult::Error(
                        "dynamic_update_slice: update dimension exceeds operand dimension");
                }
                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:startIdx.dataType];
                startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                               secondaryTensor:maxStartTensor
                                                          name:nil];

                MPSGraphTensor* indices = nil;
                if (updateDim == 1) {
                    indices = [ctx.graph broadcastTensor:startIdx toShape:updateShape name:nil];
                } else {
                    MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dynamicDim
                                                                  withShape:updateShape
                                                                       name:nil];
                    coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];
                    indices = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:startIdx name:nil];
                }
                indices = EnsureInt32(ctx.graph, indices);

                outTensor = [ctx.graph scatterAlongAxis:(NSInteger)dynamicDim
                                         withDataTensor:operand
                                          updatesTensor:update
                                          indicesTensor:indices
                                                   mode:MPSGraphScatterModeSet
                                                   name:nil];
            }
        }
    }

    // Path C: Masking-based fallback for control flow contexts.
    // Inside while_loop bodies, MPSGraph's scatter operations have shape verification bugs.
    // We implement dynamic_update_slice using coordinate comparisons and selectWithPredicateTensor
    // which only uses elementwise ops that work reliably inside control flow.
    if (!outTensor && ctx.isInControlFlow() && rank > 0) {
        // Clamp each start index to [0, operandDim - updateDim].
        NSMutableArray<MPSGraphTensor*>* clampedStarts = [NSMutableArray arrayWithCapacity:rank];
        bool dimsValid = true;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dim + 2));
            if (!startIdx) {
                return ProcessResult::Error("dynamic_update_slice: missing start index");
            }
            startIdx = EnsureInt32(ctx.graph, startIdx);

            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            if (operandDim < 0 || updateDim < 0) {
                dimsValid = false;
                break;
            }
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
            MPSGraphTensor* maxStartT = [ctx.graph constantWithScalar:maxStart dataType:MPSDataTypeInt32];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartT name:nil];
            [clampedStarts addObject:startIdx];
        }

        if (dimsValid) {
            // Build a boolean mask over operandShape: true where the update region lies.
            // For each dimension, check: start[d] <= coord < start[d] + updateSize[d]
            // The combined mask is the AND of all per-dimension masks.
            MPSGraphTensor* combinedMask = nil;
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                int64_t updateDim = [updateShape[dim] longLongValue];
                MPSGraphTensor* startIdx = clampedStarts[dim];

                // Create coordinate tensor along this axis with operand shape.
                // Shape: broadcast-compatible — size operandDim along dim, 1 elsewhere.
                NSMutableArray<NSNumber*>* coordShape = [NSMutableArray arrayWithCapacity:rank];
                for (NSUInteger d = 0; d < rank; ++d) {
                    [coordShape addObject:(d == dim ? operandShape[d] : @1)];
                }
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                              withShape:coordShape
                                                                   name:nil];
                coords = [ctx.graph castTensor:coords toType:MPSDataTypeInt32 name:nil];

                MPSGraphTensor* endIdx =
                    [ctx.graph additionWithPrimaryTensor:startIdx
                                         secondaryTensor:[ctx.graph constantWithScalar:updateDim
                                                                              dataType:MPSDataTypeInt32]
                                                    name:nil];

                // mask_dim = (coords >= start) & (coords < end)
                MPSGraphTensor* geStart = [ctx.graph greaterThanOrEqualToWithPrimaryTensor:coords
                                                                           secondaryTensor:startIdx
                                                                                      name:nil];
                MPSGraphTensor* ltEnd = [ctx.graph lessThanWithPrimaryTensor:coords
                                                             secondaryTensor:endIdx
                                                                        name:nil];
                MPSGraphTensor* dimMask = [ctx.graph logicalANDWithPrimaryTensor:geStart
                                                                 secondaryTensor:ltEnd
                                                                            name:nil];

                if (!combinedMask) {
                    combinedMask = dimMask;
                } else {
                    combinedMask = [ctx.graph logicalANDWithPrimaryTensor:combinedMask
                                                          secondaryTensor:dimMask
                                                                     name:nil];
                }
            }

            // Build paddedUpdate (operand-shaped) using one_hot + matmul per dimension.
            // For each dim where updateDim < operandDim, expand the update using:
            //   indices = arange(updateDim) + start[d]
            //   selector = one_hot(indices, operandDim)  -> [updateDim, operandDim]
            //   selectorT = transpose(selector)          -> [operandDim, updateDim]
            //   Contract update along dim with selectorT to expand it.
            // This only uses one_hot, matmul, reshape - NO scatter/gather.

            MPSGraphTensor* paddedUpdate = update;

            // Expand each partial update axis to the operand extent without a
            // float32 one-hot matmul (which lost int32 precision above 2**24 and
            // turned inf*0 into NaN). Rows outside the update region are
            // discarded by the final select, so any in-range row may fill them.
            // - Real floating types: gather update row clamp(j - start) for each
            //   operand position j (exact, operand-sized intermediate).
            // - Integer/bool/complex: in-loop gatherAlongAxis measurably lost
            //   int32 precision above 2**24, so broadcast against a boolean
            //   one-hot selector and sum over the update axis instead. Each
            //   output receives exactly one value, so the sum is exact; the
            //   intermediate is updateDim x operand-sized.
            // A size-1 update axis needs no expansion: the final select's mask
            // picks exactly one position there and the update broadcasts.
            const MPSDataType updateDtype = paddedUpdate.dataType;
            const bool gatherExact = updateDtype == MPSDataTypeFloat32 || updateDtype == MPSDataTypeFloat16 ||
                                     updateDtype == MPSDataTypeBFloat16;
            const bool boolDtype = updateDtype == MPSDataTypeBool;
            if (boolDtype) {
                paddedUpdate = [ctx.graph castTensor:paddedUpdate toType:MPSDataTypeInt8 name:nil];
            }
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                int64_t opDim = [operandShape[dim] longLongValue];
                int64_t upDim = [updateShape[dim] longLongValue];

                if (upDim == opDim || upDim == 1) {
                    continue; // No expansion needed on this dimension.
                }

                NSMutableArray<NSNumber*>* outShape = [NSMutableArray arrayWithArray:paddedUpdate.shape];
                outShape[dim] = @(opDim);

                if (gatherExact) {
                    MPSGraphTensor* positions = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                                     withShape:outShape
                                                                          name:nil];
                    positions = [ctx.graph castTensor:positions toType:MPSDataTypeInt32 name:nil];
                    MPSGraphTensor* rows = [ctx.graph subtractionWithPrimaryTensor:positions
                                                                   secondaryTensor:clampedStarts[dim]
                                                                              name:nil];
                    MPSGraphTensor* zeroIdx = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
                    MPSGraphTensor* lastIdx = [ctx.graph constantWithScalar:upDim - 1
                                                                   dataType:MPSDataTypeInt32];
                    rows = [ctx.graph maximumWithPrimaryTensor:rows secondaryTensor:zeroIdx name:nil];
                    rows = [ctx.graph minimumWithPrimaryTensor:rows secondaryTensor:lastIdx name:nil];
                    paddedUpdate = [ctx.graph gatherAlongAxis:(NSInteger)dim
                                            withUpdatesTensor:paddedUpdate
                                                indicesTensor:rows
                                                         name:nil];
                    continue;
                }

                // selector[k, j] = (start + k == j) -> [updateDim, operandDim]
                MPSGraphTensor* localCoords = [ctx.graph coordinateAlongAxis:0
                                                                   withShape:@[@(upDim), @1]
                                                                        name:nil];
                localCoords = [ctx.graph castTensor:localCoords toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* globalIndices = [ctx.graph additionWithPrimaryTensor:localCoords
                                                                     secondaryTensor:clampedStarts[dim]
                                                                                name:nil];
                MPSGraphTensor* targets = [ctx.graph coordinateAlongAxis:1
                                                               withShape:@[@1, @(opDim)]
                                                                    name:nil];
                targets = [ctx.graph castTensor:targets toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* selector = [ctx.graph equalWithPrimaryTensor:globalIndices
                                                             secondaryTensor:targets
                                                                        name:nil];
                NSUInteger curRank = paddedUpdate.shape.count;
                NSMutableArray<NSNumber*>* selectorShape = [NSMutableArray arrayWithCapacity:curRank + 1];
                for (NSUInteger d = 0; d <= curRank; ++d) {
                    [selectorShape addObject:d == dim ? @(upDim) : (d == dim + 1 ? @(opDim) : @1)];
                }
                selector = [ctx.graph reshapeTensor:selector withShape:selectorShape name:nil];

                NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithArray:paddedUpdate.shape];
                [expandedShape insertObject:@1 atIndex:dim + 1];
                MPSGraphTensor* expanded = [ctx.graph reshapeTensor:paddedUpdate
                                                          withShape:expandedShape
                                                               name:nil];
                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:paddedUpdate.dataType];
                MPSGraphTensor* picked = [ctx.graph selectWithPredicateTensor:selector
                                                          truePredicateTensor:expanded
                                                         falsePredicateTensor:zero
                                                                         name:nil];
                MPSGraphTensor* summed = [ctx.graph reductionSumWithTensor:picked
                                                                      axis:(NSInteger)dim
                                                                      name:nil];
                paddedUpdate = [ctx.graph reshapeTensor:summed withShape:outShape name:nil];
            }
            if (boolDtype) {
                paddedUpdate = [ctx.graph castTensor:paddedUpdate toType:MPSDataTypeBool name:nil];
            }

            // result = where(mask, paddedUpdate, operand)
            outTensor = [ctx.graph selectWithPredicateTensor:combinedMask
                                         truePredicateTensor:paddedUpdate
                                        falsePredicateTensor:operand
                                                        name:nil];
        }
    }

    // Path D: scatterND fallback (outside control flow only)
    if (!outTensor && !ctx.isInControlFlow()) {

        NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
        for (NSUInteger i = 0; i < rank; i++) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, i + 2);
            if (!startIdx) {
                return ProcessResult::Error("dynamic_update_slice: missing start index");
            }
            [startIndices addObject:startIdx];
        }

        NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
        for (NSUInteger dim = 0; dim < rank; dim++) {
            MPSGraphTensor* startIdx = EnsureInt32(ctx.graph, startIndices[dim]);

            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
            MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                  dataType:startIdx.dataType];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

            MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                          withShape:updateShape
                                                               name:nil];
            coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

            MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                                  secondaryTensor:startIdx
                                                                             name:nil];

            [indexTensors addObject:adjustedCoords];
        }

        MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

        indices = EnsureInt32(ctx.graph, indices);

        outTensor = [ctx.graph scatterNDWithDataTensor:operand
                                         updatesTensor:update
                                         indicesTensor:indices
                                       batchDimensions:0
                                                  mode:MPSGraphScatterModeSet
                                                  name:nil];
    }

    if (!outTensor) {
        return ProcessResult::Error("dynamic_update_slice: handler returned null");
    }

    if (baseVariable) {
        MPSGraphOperation* assign = [ctx.graph assignVariable:baseVariable
                                            withValueOfTensor:outTensor
                                                         name:nil];
        if (!assign) {
            return ProcessResult::Error("dynamic_update_slice: assignVariable returned null");
        }
        NSArray<MPSGraphTensor*>* forwarded =
            [ctx.graph controlDependencyWithOperations:@[assign]
                                        dependentBlock:^NSArray<MPSGraphTensor*>*() {
                                          return @[baseVariable];
                                        }
                                                  name:nil];
        if (!forwarded || forwarded.count != 1 || !forwarded[0]) {
            return ProcessResult::Error("dynamic_update_slice: controlDependency returned null");
        }
        MPSGraphTensor* varOut = forwarded[0];

        (*ctx.variables)[ctx.op->getResult(0).getAsOpaquePointer()] = varOut;
        SetOutputTensor(ctx.values, ctx.op, varOut);
        return ProcessResult{};
    }

    return Result(ctx, outTensor, "dynamic_update_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice);

static std::string JoinInts(llvm::ArrayRef<int64_t> values) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < values.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << values[i];
    }
    oss << "]";
    return oss.str();
}

static std::string ShapeToString(NSArray<NSNumber*>* shape) {
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

static bool ContainsDim(llvm::ArrayRef<int64_t> dims, int64_t dim) {
    for (int64_t d : dims) {
        if (d == dim)
            return true;
    }
    return false;
}

static bool IsIdentityPermutation(NSArray<NSNumber*>* permutation) {
    for (NSUInteger i = 0; i < permutation.count; ++i) {
        if ([permutation[i] integerValue] != (NSInteger)i) {
            return false;
        }
    }
    return true;
}

static std::string GatherDebugString(mlir::stablehlo::GatherOp gatherOp, MPSGraphTensor* operand,
                                     MPSGraphTensor* startIndices) {
    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNumbers.getStartIndicesBatchingDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    auto sliceSizes = gatherOp.getSliceSizes();
    std::ostringstream oss;
    oss << "operand_shape=" << ShapeToString(operand.shape)
        << ", indices_shape=" << ShapeToString(startIndices.shape) << ", offset_dims=" << JoinInts(offsetDims)
        << ", collapsed_slice_dims=" << JoinInts(collapsedSliceDims)
        << ", operand_batching_dims=" << JoinInts(operandBatchingDims)
        << ", start_indices_batching_dims=" << JoinInts(startIndicesBatchingDims)
        << ", start_index_map=" << JoinInts(startIndexMap)
        << ", index_vector_dim=" << dimNumbers.getIndexVectorDim()
        << ", slice_sizes=" << JoinInts(sliceSizes);
    return oss.str();
}

static ProcessResult HandleGatherImpl(HandlerContext& ctx) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(ctx.op);
    if (!gatherOp) {
        return ProcessResult::Error("gather: expected GatherOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* startIndices = GetInputTensor(ctx, 1);
    if (!operand || !startIndices)
        return ProcessResult::Error("gather: missing input tensor");

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNumbers.getStartIndicesBatchingDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto sliceSizes = gatherOp.getSliceSizes();

    NSArray<NSNumber*>* indicesShape = startIndices.shape;
    NSArray<NSNumber*>* operandShape = operand.shape;
    NSUInteger indicesRank = indicesShape.count;
    NSUInteger operandRank = operandShape.count;

    bool collapsedGatherDim = startIndexMap.size() == 1 && collapsedSliceDims.size() == 1 &&
                              collapsedSliceDims[0] == startIndexMap[0];
    bool nonCollapsedGatherDim = startIndexMap.size() == 1 && collapsedSliceDims.empty();

    bool embeddingPattern = indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 &&
                            [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
                            (collapsedGatherDim || nonCollapsedGatherDim) && startIndexMap[0] >= 0 &&
                            startIndexMap[0] < (int64_t)sliceSizes.size() &&
                            sliceSizes[startIndexMap[0]] == 1;
    if (embeddingPattern) {
        int64_t gatherAxis = startIndexMap[0];
        if (gatherAxis < 0 || gatherAxis >= (int64_t)operandRank) {
            return ProcessResult::Error("gather: gather axis out of bounds (" +
                                        GatherDebugString(gatherOp, operand, startIndices) + ")");
        }

        NSUInteger batchDimensions = 0;
        bool canonicalizeBatching = false;
        int64_t gatherAxisForMps = gatherAxis;
        std::vector<int64_t> indicesCanonNonVector;
        std::vector<int64_t> indicesCanonNonVectorInverse;

        if (!operandBatchingDims.empty() || !startIndicesBatchingDims.empty()) {
            if (operandBatchingDims.size() != startIndicesBatchingDims.size() ||
                operandBatchingDims.empty()) {
                return ProcessResult::Error("gather: unsupported operand/start_indices batching dims (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }

            size_t B = operandBatchingDims.size();

            auto isPrefixBatchDims = [](llvm::ArrayRef<int64_t> dims) -> bool {
                for (size_t i = 0; i < dims.size(); ++i) {
                    if (dims[i] != (int64_t)i) {
                        return false;
                    }
                }
                return true;
            };

            bool tierA =
                isPrefixBatchDims(operandBatchingDims) && isPrefixBatchDims(startIndicesBatchingDims);

            if (tierA && operandBatchingDims.size() == startIndicesBatchingDims.size()) {
                for (size_t i = 0; i < B; ++i) {
                    if (operandBatchingDims[i] != startIndicesBatchingDims[i]) {
                        tierA = false;
                        break;
                    }
                }
            }

            batchDimensions = (NSUInteger)B;

            if (!tierA) {
                canonicalizeBatching = true;

                NSMutableArray<NSNumber*>* operandPerm = [NSMutableArray arrayWithCapacity:operandRank];
                for (int64_t d : operandBatchingDims) {
                    if (d < 0 || d >= (int64_t)operandRank || d == gatherAxis) {
                        return ProcessResult::Error("gather: invalid operand_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    [operandPerm addObject:@(d)];
                }
                [operandPerm addObject:@(gatherAxis)];
                for (NSUInteger d = 0; d < operandRank; ++d) {
                    if ((int64_t)d == gatherAxis || ContainsDim(operandBatchingDims, (int64_t)d)) {
                        continue;
                    }
                    [operandPerm addObject:@((NSInteger)d)];
                }
                if (operandPerm.count != operandRank) {
                    return ProcessResult::Error("gather: invalid canonical operand permutation (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }

                if (!IsIdentityPermutation(operandPerm)) {
                    operand = [ctx.graph transposeTensor:operand permutation:operandPerm name:nil];
                    operandShape = operand.shape;
                }

                NSMutableArray<NSNumber*>* indicesPerm = [NSMutableArray arrayWithCapacity:indicesRank];
                std::vector<bool> used((size_t)indicesRank, false);
                indicesCanonNonVector.clear();
                indicesCanonNonVector.reserve(indicesRank > 0 ? indicesRank - 1 : 0);

                for (int64_t d : startIndicesBatchingDims) {
                    if (d < 0 || d >= (int64_t)indicesRank - 1) {
                        return ProcessResult::Error("gather: invalid start_indices_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    if (used[(size_t)d]) {
                        return ProcessResult::Error("gather: duplicate start_indices_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    used[(size_t)d] = true;
                    indicesCanonNonVector.push_back(d);
                    [indicesPerm addObject:@(d)];
                }
                for (NSUInteger d = 0; d < indicesRank - 1; ++d) {
                    if (used[(size_t)d]) {
                        continue;
                    }
                    used[(size_t)d] = true;
                    indicesCanonNonVector.push_back((int64_t)d);
                    [indicesPerm addObject:@((NSInteger)d)];
                }
                [indicesPerm addObject:@((NSInteger)indexVectorDim)];
                if (indicesPerm.count != indicesRank) {
                    return ProcessResult::Error("gather: invalid canonical indices permutation (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }

                if (!IsIdentityPermutation(indicesPerm)) {
                    startIndices = [ctx.graph transposeTensor:startIndices permutation:indicesPerm name:nil];
                    indicesShape = startIndices.shape;
                }

                gatherAxisForMps = (int64_t)B;

                size_t nonVectorRank = indicesCanonNonVector.size();
                indicesCanonNonVectorInverse.assign(nonVectorRank, -1);
                for (size_t i = 0; i < nonVectorRank; ++i) {
                    int64_t originalAxis = indicesCanonNonVector[i];
                    if (originalAxis < 0 || (size_t)originalAxis >= nonVectorRank) {
                        return ProcessResult::Error("gather: invalid indices non-vector canonicalization (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    indicesCanonNonVectorInverse[(size_t)originalAxis] = (int64_t)i;
                }
                for (size_t i = 0; i < nonVectorRank; ++i) {
                    if (indicesCanonNonVectorInverse[i] < 0) {
                        return ProcessResult::Error(
                            "gather: invalid indices non-vector inverse permutation (" +
                            GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                }
            }
        }

        if (sliceSizes.size() == operandRank) {
            bool needsSlice = false;
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:operandRank];
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                [starts addObject:@0];
                [strides addObject:@1];
                int64_t operandDim = [operand.shape[dim] longLongValue];
                int64_t end = operandDim;
                if ((int64_t)dim != gatherAxis && !ContainsDim(operandBatchingDims, (int64_t)dim)) {
                    int64_t slice = sliceSizes[dim];
                    if (slice < 0 || operandDim <= 0 || slice > operandDim) {
                        return ProcessResult::Error("gather: invalid slice_sizes for operand dim " +
                                                    std::to_string(dim) + " (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    end = slice;
                    if (slice != operandDim) {
                        needsSlice = true;
                    }
                }
                [ends addObject:@(end)];
            }
            if (needsSlice) {
                operand = [ctx.graph sliceTensor:operand starts:starts ends:ends strides:strides name:nil];
                operandShape = operand.shape;
            }
        }

        MPSGraphTensor* gatherIndices = startIndices;
        if (collapsedGatherDim) {

            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesShape.count - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }
            gatherIndices = [ctx.graph reshapeTensor:startIndices withShape:squeezedShape name:nil];
        }

        gatherIndices = EnsureInt32(ctx.graph, gatherIndices);

        MPSGraphTensor* result = [ctx.graph gatherWithUpdatesTensor:operand
                                                      indicesTensor:gatherIndices
                                                               axis:(NSUInteger)gatherAxisForMps
                                                    batchDimensions:batchDimensions
                                                               name:nil];

        if (canonicalizeBatching && !indicesCanonNonVectorInverse.empty() && result) {
            NSUInteger prefixRank = indicesCanonNonVectorInverse.size();
            NSMutableArray<NSNumber*>* resultPerm = [NSMutableArray arrayWithCapacity:result.shape.count];
            for (NSUInteger i = 0; i < prefixRank; ++i) {
                [resultPerm addObject:@(indicesCanonNonVectorInverse[i])];
            }

            for (NSUInteger i = prefixRank; i < result.shape.count; ++i) {
                [resultPerm addObject:@((NSInteger)i)];
            }
            if (resultPerm.count == result.shape.count && !IsIdentityPermutation(resultPerm)) {
                result = [ctx.graph transposeTensor:result permutation:resultPerm name:nil];
            }
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && result && ![expectedShape isEqualToArray:result.shape]) {
            result = [ctx.graph reshapeTensor:result withShape:expectedShape name:nil];
        }

        return Result(ctx, result, "gather");
    }

    bool pointGatherNdPattern =
        indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 && startIndexMap.size() > 1 &&
        operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        (NSInteger)startIndexMap.size() == [indicesShape[indicesRank - 1] integerValue] &&
        (collapsedSliceDims.size() == startIndexMap.size() || collapsedSliceDims.empty()) &&
        sliceSizes.size() == operandRank;
    if (pointGatherNdPattern) {
        bool mappingValid = true;
        bool collapsedMappedDims = !collapsedSliceDims.empty();
        for (int64_t mappedDim : startIndexMap) {
            if (mappedDim < 0 || mappedDim >= (int64_t)operandRank ||
                (collapsedMappedDims && !ContainsDim(collapsedSliceDims, mappedDim)) ||
                sliceSizes[mappedDim] != 1) {
                mappingValid = false;
                break;
            }
        }
        if (!mappingValid) {
            pointGatherNdPattern = false;
        }
    }

    if (pointGatherNdPattern) {

        if (sliceSizes.size() == operandRank) {
            bool needsSlice = false;
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:operandRank];
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                [starts addObject:@0];
                [strides addObject:@1];
                int64_t operandDim = [operand.shape[dim] longLongValue];
                int64_t end = operandDim;
                if (!ContainsDim(startIndexMap, (int64_t)dim)) {
                    int64_t slice = sliceSizes[dim];
                    if (slice < 0 || operandDim <= 0 || slice > operandDim) {
                        return ProcessResult::Error("gather: invalid slice_sizes for operand dim " +
                                                    std::to_string(dim) + " (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    end = slice;
                    if (slice != operandDim) {
                        needsSlice = true;
                    }
                }
                [ends addObject:@(end)];
            }
            if (needsSlice) {
                operand = [ctx.graph sliceTensor:operand starts:starts ends:ends strides:strides name:nil];
                operandShape = operand.shape;
            }
        }

        bool collapsedMappedDims = !collapsedSliceDims.empty();
        NSUInteger indexPrefixRank = indicesRank - 1;
        NSUInteger mappedRank = startIndexMap.size();
        NSUInteger remainingRank = operandRank - mappedRank;
        NSUInteger sliceDimsCount = collapsedMappedDims ? remainingRank : operandRank;
        NSUInteger outputRank = indexPrefixRank + sliceDimsCount;

        if (offsetDims.size() != sliceDimsCount) {
            return ProcessResult::Error("gather: unsupported gatherND offset dims (" +
                                        GatherDebugString(gatherOp, operand, startIndices) + ")");
        }

        NSMutableArray<NSNumber*>* operandPerm = [NSMutableArray arrayWithCapacity:operandRank];
        std::vector<bool> isMapped(operandRank, false);
        for (int64_t mappedDim : startIndexMap) {
            if (mappedDim < 0 || mappedDim >= (int64_t)operandRank || isMapped[mappedDim]) {
                return ProcessResult::Error("gather: invalid start_index_map for gatherND (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            isMapped[mappedDim] = true;
            [operandPerm addObject:@(mappedDim)];
        }
        for (NSUInteger dim = 0; dim < operandRank; ++dim) {
            if (!isMapped[dim]) {
                [operandPerm addObject:@((NSInteger)dim)];
            }
        }

        MPSGraphTensor* transposedOperand = operand;
        if (!IsIdentityPermutation(operandPerm)) {
            transposedOperand = [ctx.graph transposeTensor:operand permutation:operandPerm name:nil];
        }

        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, startIndices);
        MPSGraphTensor* gathered = [ctx.graph gatherNDWithUpdatesTensor:transposedOperand
                                                          indicesTensor:int32Indices
                                                        batchDimensions:0
                                                                   name:nil];

        MPSGraphTensor* expanded = gathered;
        if (!collapsedMappedDims) {
            NSMutableArray<NSNumber*>* expandedShape =
                [NSMutableArray arrayWithCapacity:indexPrefixRank + operandRank];
            for (NSUInteger i = 0; i < indexPrefixRank; ++i) {
                [expandedShape addObject:gathered.shape[i]];
            }
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                int64_t slice = sliceSizes[dim];
                if (slice < 0) {
                    return ProcessResult::Error("gather: negative slice_sizes not supported (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }
                [expandedShape addObject:@(slice)];
            }
            expanded = [ctx.graph reshapeTensor:gathered withShape:expandedShape name:nil];
        }

        std::vector<int64_t> sliceDimToOutputAxis(sliceDimsCount, -1);
        std::vector<bool> outputAxisUsed(outputRank, false);
        for (NSUInteger s = 0; s < sliceDimsCount; ++s) {
            int64_t axis = offsetDims[s];
            if (axis < 0 || axis >= (int64_t)outputRank || outputAxisUsed[(size_t)axis]) {
                return ProcessResult::Error("gather: invalid offset_dims for gatherND (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            outputAxisUsed[(size_t)axis] = true;
            sliceDimToOutputAxis[s] = axis;
        }

        NSMutableArray<NSNumber*>* outputPermutation = [NSMutableArray arrayWithCapacity:outputRank];
        NSUInteger nextPrefixDim = 0;
        for (NSUInteger outAxis = 0; outAxis < outputRank; ++outAxis) {
            bool isOffsetAxis = false;
            NSUInteger sliceSourceIndex = 0;
            for (; sliceSourceIndex < sliceDimsCount; ++sliceSourceIndex) {
                if (sliceDimToOutputAxis[sliceSourceIndex] == (int64_t)outAxis) {
                    isOffsetAxis = true;
                    break;
                }
            }

            NSInteger sourceAxis = 0;
            if (isOffsetAxis) {
                sourceAxis = (NSInteger)(indexPrefixRank + sliceSourceIndex);
            } else {
                if (nextPrefixDim >= indexPrefixRank) {
                    return ProcessResult::Error("gather: invalid prefix/offset mapping (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }
                sourceAxis = (NSInteger)nextPrefixDim;
                ++nextPrefixDim;
            }
            [outputPermutation addObject:@(sourceAxis)];
        }

        MPSGraphTensor* result = expanded;
        if (outputRank > 1 && !IsIdentityPermutation(outputPermutation)) {
            result = [ctx.graph transposeTensor:expanded permutation:outputPermutation name:nil];
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && result && ![expectedShape isEqualToArray:result.shape]) {
            result = [ctx.graph reshapeTensor:result withShape:expectedShape name:nil];
        }
        return Result(ctx, result, "gather");
    }

    // Dynamic slice gather pattern: dynamic_slice (possibly via vmap) lowered to gather.
    // Handles both identity and non-identity start_index_map, and partial indexing
    // where only some operand dims are mapped.
    //
    // Pattern requirements:
    //   - index_vector_dim == indicesRank - 1 (last dim of indices is the index vector)
    //   - collapsed_slice_dims = [] (no dims collapsed)
    //   - offset_dims = [batchRank, batchRank+1, ..., batchRank+operandRank-1]
    //   - start_index_map entries are unique and valid operand dims
    //   - slice_sizes[mapped_dim] >= 1 (may be > 1, unlike embedding gather)
    //
    // Strategy: expand start indices to cover the full slice by adding range offsets,
    //           build full operandRank-dimensional indices, then use gatherND.
    bool dynamicSliceGatherPattern = indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 &&
                                     operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
                                     collapsedSliceDims.empty() && startIndexMap.size() > 0 &&
                                     sliceSizes.size() == operandRank;

    if (dynamicSliceGatherPattern) {
        // Validate index vector dim size matches start_index_map size
        // For indicesRank==1: the whole tensor is the index vector (no batch prefix)
        // indexVectorDim is the last axis, and its size should equal startIndexMap.size()
        int64_t indexVectorSize = [indicesShape[indexVectorDim] longLongValue];
        if ((int64_t)startIndexMap.size() != indexVectorSize) {
            dynamicSliceGatherPattern = false;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Validate start_index_map entries are valid and unique
        std::vector<bool> mappedDims(operandRank, false);
        for (int64_t d : startIndexMap) {
            if (d < 0 || d >= (int64_t)operandRank || mappedDims[(size_t)d]) {
                dynamicSliceGatherPattern = false;
                break;
            }
            mappedDims[(size_t)d] = true;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Validate offset_dims has operandRank entries with valid, strictly increasing values.
        // They may be interleaved with batch dims (not necessarily [batchRank, batchRank+1, ...]).
        NSUInteger batchRank = indicesRank - 1;
        NSUInteger outputRankDS = batchRank + operandRank;
        bool validOffsets = offsetDims.size() == operandRank;
        for (size_t i = 0; i < offsetDims.size() && validOffsets; ++i) {
            if (offsetDims[i] < 0 || offsetDims[i] >= (int64_t)outputRankDS) {
                validOffsets = false;
            }
            if (i > 0 && offsetDims[i] <= offsetDims[i - 1]) {
                validOffsets = false;
            }
        }
        if (!validOffsets) {
            dynamicSliceGatherPattern = false;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Build expanded indices for gatherND.
        // For each operand dimension, we create a range covering slice_sizes[dim].
        // For mapped dims, the range is offset by the start index.
        // For unmapped dims, the range starts at 0.
        //
        // Expanded indices shape: [batch..., S0, S1, ..., S_{R-1}, R]
        // where R = operandRank and Si = sliceSizes[i]

        NSUInteger batchRank = indicesRank - 1;
        NSUInteger R = operandRank; // all operand dims in the expanded index

        // Build a mapping: for each operand dim, which index in startIndexMap provides it?
        // mappedIndexPos[dim] = position in startIndexMap, or -1 if unmapped
        std::vector<int64_t> mappedIndexPos(R, -1);
        for (size_t i = 0; i < startIndexMap.size(); ++i) {
            mappedIndexPos[(size_t)startIndexMap[i]] = (int64_t)i;
        }

        // Validate total slice size
        int64_t totalSlice = 1;
        for (NSUInteger d = 0; d < R; ++d) {
            if (sliceSizes[d] <= 0) {
                return ProcessResult::Error("gather: dynamic_slice pattern has non-positive slice_sizes (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            totalSlice *= sliceSizes[d];
            if (totalSlice > 16 * 1024 * 1024) {
                return ProcessResult::Error("gather: dynamic_slice pattern slice too large (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
        }

        // expanded shape: [batch..., S0, S1, ..., S_{R-1}, R]
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < batchRank; ++i) {
            [expandedShape addObject:indicesShape[i]];
        }
        for (NSUInteger d = 0; d < R; ++d) {
            [expandedShape addObject:@(sliceSizes[d])];
        }
        [expandedShape addObject:@(R)];

        NSUInteger expandedRank = expandedShape.count;

        // Build per-dim index tensors and concatenate along last dim.
        // For each operand dim d:
        //   If mapped (startIndexMap has it): index = start[..., pos] + range(sliceSizes[d])
        //   If unmapped: index = range(sliceSizes[d])  (no start offset)
        //
        // Each produces shape [batch..., S0, ..., S_{R-1}, 1], concat → [..., R]

        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, startIndices);

        NSMutableArray<MPSGraphTensor*>* dimTensors = [NSMutableArray arrayWithCapacity:R];
        for (NSUInteger d = 0; d < R; ++d) {
            int64_t S = sliceSizes[d];

            // Create range [0, 1, ..., S-1] as int32
            int32_t* rangeData = (int32_t*)calloc((size_t)S, sizeof(int32_t));
            for (int32_t s = 0; s < (int32_t)S; ++s) {
                rangeData[s] = s;
            }
            NSData* rangeNSData = [NSData dataWithBytesNoCopy:rangeData
                                                       length:(NSUInteger)S * sizeof(int32_t)
                                                 freeWhenDone:YES];
            MPSGraphTensor* rangeTensor = [ctx.graph constantWithData:rangeNSData
                                                                shape:@[@(S)]
                                                             dataType:MPSDataTypeInt32];

            // Reshape range to [1, ..., 1, S, 1, ..., 1] with S at axis (batchRank + d)
            NSMutableArray<NSNumber*>* rangeShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < batchRank; ++i) {
                [rangeShape addObject:@1];
            }
            for (NSUInteger dd = 0; dd < R; ++dd) {
                [rangeShape addObject:(dd == d) ? @(S) : @1];
            }
            MPSGraphTensor* rangeReshaped = [ctx.graph reshapeTensor:rangeTensor
                                                           withShape:rangeShape
                                                                name:nil];

            // Broadcast to [batch..., S0, S1, ..., S_{R-1}]
            NSMutableArray<NSNumber*>* broadShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < batchRank; ++i) {
                [broadShape addObject:indicesShape[i]];
            }
            for (NSUInteger dd = 0; dd < R; ++dd) {
                [broadShape addObject:@(sliceSizes[dd])];
            }
            MPSGraphTensor* rangeBroadcasted = [ctx.graph broadcastTensor:rangeReshaped
                                                                  toShape:broadShape
                                                                     name:nil];

            MPSGraphTensor* dimIndex = rangeBroadcasted;

            if (mappedIndexPos[d] >= 0) {
                // This dim is mapped: add the start index
                int64_t pos = mappedIndexPos[d];

                // Extract the start value for this dim: startIndices[..., pos]
                // Slice along the index vector dim to get just this component
                NSMutableArray<NSNumber*>* sliceStarts = [NSMutableArray array];
                NSMutableArray<NSNumber*>* sliceEnds = [NSMutableArray array];
                NSMutableArray<NSNumber*>* sliceStrides = [NSMutableArray array];
                for (NSUInteger i = 0; i < indicesRank; ++i) {
                    if (i == (NSUInteger)indexVectorDim) {
                        [sliceStarts addObject:@(pos)];
                        [sliceEnds addObject:@(pos + 1)];
                    } else {
                        [sliceStarts addObject:@0];
                        [sliceEnds addObject:indicesShape[i]];
                    }
                    [sliceStrides addObject:@1];
                }
                MPSGraphTensor* startSlice = [ctx.graph sliceTensor:int32Indices
                                                             starts:sliceStarts
                                                               ends:sliceEnds
                                                            strides:sliceStrides
                                                               name:nil];

                // Reshape to [batch..., 1, 1, ..., 1] for broadcasting
                NSMutableArray<NSNumber*>* startBroadShape = [NSMutableArray array];
                for (NSUInteger i = 0; i < batchRank; ++i) {
                    [startBroadShape addObject:indicesShape[i]];
                }
                for (NSUInteger dd = 0; dd < R; ++dd) {
                    [startBroadShape addObject:@1];
                }
                MPSGraphTensor* startReshaped = [ctx.graph reshapeTensor:startSlice
                                                               withShape:startBroadShape
                                                                    name:nil];

                // Broadcast to [batch..., S0, ..., S_{R-1}]
                MPSGraphTensor* startBroadcasted = [ctx.graph broadcastTensor:startReshaped
                                                                      toShape:broadShape
                                                                         name:nil];

                // Add start + range
                dimIndex = [ctx.graph additionWithPrimaryTensor:startBroadcasted
                                                secondaryTensor:rangeBroadcasted
                                                           name:nil];
            }

            // Expand last dim: [batch..., S0, ..., S_{R-1}, 1]
            NSMutableArray<NSNumber*>* expandedDimShape = [NSMutableArray arrayWithArray:broadShape];
            [expandedDimShape addObject:@1];
            MPSGraphTensor* dimExpanded = [ctx.graph reshapeTensor:dimIndex
                                                         withShape:expandedDimShape
                                                              name:nil];
            [dimTensors addObject:dimExpanded];
        }

        // Concatenate along last dim → [batch..., S0, ..., S_{R-1}, R]
        MPSGraphTensor* expandedIndices = [ctx.graph concatTensors:dimTensors
                                                         dimension:(NSInteger)(expandedRank - 1)
                                                              name:nil];

        // gatherND: operand[expandedIndices]
        // Result shape: [batch..., S0, S1, ..., S_{R-1}]
        MPSGraphTensor* gathered = [ctx.graph gatherNDWithUpdatesTensor:operand
                                                          indicesTensor:expandedIndices
                                                        batchDimensions:0
                                                                   name:nil];

        // If offset_dims are not in the standard [batchRank, batchRank+1, ...] positions,
        // we need to permute the gathered result to place offset dims at the correct output
        // positions. The gathered result has layout [batch0, ..., batch_{B-1}, S0, ..., S_{R-1}].
        // We need to permute so that offset dim i maps to output position offsetDims[i].
        NSUInteger outputRankDS = batchRank + R;
        bool needsPermute = false;
        for (size_t i = 0; i < R; ++i) {
            if (offsetDims[i] != (int64_t)(batchRank + i)) {
                needsPermute = true;
                break;
            }
        }
        if (needsPermute) {
            // Build the output permutation.
            // Gathered layout: positions [0..batchRank-1] are batch dims, [batchRank..batchRank+R-1] are
            // slice dims. We need output[offsetDims[i]] = slice dim i, and the remaining positions filled by
            // batch dims in order.
            std::vector<bool> isOffsetPos(outputRankDS, false);
            for (size_t i = 0; i < R; ++i) {
                isOffsetPos[(size_t)offsetDims[i]] = true;
            }

            // Build inverse: for each output position, find which gathered position maps to it
            NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:outputRankDS];
            for (NSUInteger i = 0; i < outputRankDS; ++i) {
                [perm addObject:@0]; // placeholder
            }

            // Place offset dims: output position offsetDims[i] ← gathered position (batchRank + i)
            for (size_t i = 0; i < R; ++i) {
                perm[(NSUInteger)offsetDims[i]] = @((NSInteger)(batchRank + i));
            }

            // Place batch dims: remaining output positions ← gathered positions 0..batchRank-1
            NSUInteger batchIdx = 0;
            for (NSUInteger outPos = 0; outPos < outputRankDS; ++outPos) {
                if (!isOffsetPos[outPos]) {
                    perm[outPos] = @((NSInteger)batchIdx);
                    ++batchIdx;
                }
            }

            gathered = [ctx.graph transposeTensor:gathered permutation:perm name:nil];
        }

        // Reshape to expected output shape
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && gathered && ![expectedShape isEqualToArray:gathered.shape]) {
            gathered = [ctx.graph reshapeTensor:gathered withShape:expectedShape name:nil];
        }
        return Result(ctx, gathered, "gather");
    }

    return ProcessResult::Error("gather: unsupported gather pattern (" +
                                GatherDebugString(gatherOp, operand, startIndices) + ")");
}
// MPSGraph gathers 32-bit integers inexactly: int32/uint32 values above 2**24
// came back off by 1-2 (inside and outside loops) while float32 gathers were
// exact, consistent with an internal float32 path. Gather the low and high
// 16-bit halves separately (each exact) and recombine them bitwise.
// mps.gather rejects complex operands with a process-aborting assertion.
// Gather the real and imaginary parts separately (exact) and recombine.
static ProcessResult HandleComplexGather(HandlerContext& ctx, MPSGraphTensor* operand) {
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;
    auto gatherPart = [&](MPSGraphTensor* part, MPSGraphTensor** out) -> ProcessResult {
        ctx.values[operandKey] = part;
        ProcessResult r = HandleGatherImpl(ctx);
        if (r.ok()) {
            *out = GetTensor(ctx.values, ctx.op->getResult(0));
        }
        return r;
    };
    MPSGraphTensor* gRe = nil;
    MPSGraphTensor* gIm = nil;
    ProcessResult r = gatherPart([ctx.graph realPartOfTensor:operand name:nil], &gRe);
    if (r.ok()) {
        r = gatherPart([ctx.graph imaginaryPartOfTensor:operand name:nil], &gIm);
    }
    if (original) {
        ctx.values[operandKey] = original;
    } else {
        ctx.values.erase(operandKey);
    }
    if (!r.ok()) {
        return r;
    }
    if (!gRe || !gIm) {
        return ProcessResult::Error("gather: complex split lowering failed");
    }
    return Result(ctx, [ctx.graph complexTensorWithRealTensor:gRe imaginaryTensor:gIm name:nil], "gather");
}

static ProcessResult HandleGather(HandlerContext& ctx) {
    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    if (operand &&
        (operand.dataType == MPSDataTypeComplexFloat32 || operand.dataType == MPSDataTypeComplexFloat16) &&
        ctx.op->getNumOperands() >= 2 && ctx.op->getNumResults() == 1 &&
        ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        !(ctx.variables &&
          ctx.variables->find(ctx.op->getOperand(0).getAsOpaquePointer()) != ctx.variables->end())) {
        return HandleComplexGather(ctx, operand);
    }
    if (!operand || (operand.dataType != MPSDataTypeInt32 && operand.dataType != MPSDataTypeUInt32) ||
        ctx.op->getNumOperands() < 2 || ctx.op->getNumResults() != 1 ||
        ctx.op->getOperand(0) == ctx.op->getOperand(1)) {
        return HandleGatherImpl(ctx);
    }
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    if (ctx.variables && ctx.variables->find(operandKey) != ctx.variables->end()) {
        return HandleGatherImpl(ctx);
    }
    void* resultKey = ctx.op->getResult(0).getAsOpaquePointer();
    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;

    MPSDataType dtype = operand.dataType;
    MPSGraphTensor* mask = [ctx.graph constantWithScalar:0xFFFF dataType:dtype];
    MPSGraphTensor* sixteen = [ctx.graph constantWithScalar:16 dataType:dtype];
    MPSGraphTensor* lo = [ctx.graph bitwiseANDWithPrimaryTensor:operand secondaryTensor:mask name:nil];
    MPSGraphTensor* hi = [ctx.graph bitwiseRightShiftWithPrimaryTensor:operand
                                                       secondaryTensor:sixteen
                                                                  name:nil];
    hi = [ctx.graph bitwiseANDWithPrimaryTensor:hi secondaryTensor:mask name:nil];

    auto restore = [&]() {
        if (original) {
            ctx.values[operandKey] = original;
        } else {
            ctx.values.erase(operandKey);
        }
    };
    auto gatherHalf = [&](MPSGraphTensor* half, MPSGraphTensor** out) -> ProcessResult {
        ctx.values[operandKey] = half;
        ProcessResult r = HandleGatherImpl(ctx);
        if (r.ok()) {
            *out = GetTensor(ctx.values, ctx.op->getResult(0));
        }
        return r;
    };

    MPSGraphTensor* gatheredLo = nil;
    MPSGraphTensor* gatheredHi = nil;
    ProcessResult r = gatherHalf(lo, &gatheredLo);
    if (r.ok()) {
        r = gatherHalf(hi, &gatheredHi);
    }
    restore();
    if (!r.ok()) {
        return r;
    }
    if (!gatheredLo || !gatheredHi || gatheredLo.dataType != dtype || gatheredHi.dataType != dtype) {
        ctx.values.erase(resultKey);
        return HandleGatherImpl(ctx);
    }
    MPSGraphTensor* shifted = [ctx.graph bitwiseLeftShiftWithPrimaryTensor:gatheredHi
                                                           secondaryTensor:sixteen
                                                                      name:nil];
    MPSGraphTensor* combined = [ctx.graph bitwiseORWithPrimaryTensor:shifted
                                                     secondaryTensor:gatheredLo
                                                                name:nil];
    return Result(ctx, combined, "gather");
}
REGISTER_MPS_OP("stablehlo.gather", HandleGather);

static MPSGraphScatterMode GetScatterMode(mlir::stablehlo::ScatterOp scatterOp) {
    MPSGraphScatterMode mode = MPSGraphScatterModeSet;
    auto& updateRegion = scatterOp.getUpdateComputation();
    if (!updateRegion.empty()) {
        auto& block = updateRegion.front();
        for (auto& innerOp : block) {
            if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                return MPSGraphScatterModeAdd;
            } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                return MPSGraphScatterModeSub;
            } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                return MPSGraphScatterModeMul;
            } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                return MPSGraphScatterModeDiv;
            } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                return MPSGraphScatterModeMax;
            } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                return MPSGraphScatterModeMin;
            }
        }
    }
    return mode;
}

static std::string ScatterDebugString(mlir::stablehlo::ScatterOp scatterOp, MPSGraphTensor* input,
                                      MPSGraphTensor* scatterIndices, MPSGraphTensor* updates) {
    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();
    std::ostringstream oss;
    oss << "input_shape=" << ShapeToString(input.shape)
        << ", indices_shape=" << ShapeToString(scatterIndices.shape)
        << ", updates_shape=" << ShapeToString(updates.shape)
        << ", update_window_dims=" << JoinInts(updateWindowDims)
        << ", inserted_window_dims=" << JoinInts(insertedWindowDims)
        << ", scatter_dims_to_operand_dims=" << JoinInts(scatterDimsToOperandDims)
        << ", input_batching_dims=" << JoinInts(inputBatchingDims)
        << ", scatter_indices_batching_dims=" << JoinInts(scatterIndicesBatchingDims)
        << ", index_vector_dim=" << dimNumbers.getIndexVectorDim();
    return oss.str();
}

// Parallel scatter optimization helpers

static bool CanUseParallelScatter(NSArray<NSNumber*>* indices, int64_t indexVectorLen,
                                  NSUInteger updatesRank, bool isAddMode) {
    if (updatesRank != 1) return false;
    if (indexVectorLen != 1) return false;
    if (!isAddMode) return true;  // Set mode is naturally parallel-safe
    return true;
}

static bool IndicesAreUnique(NSArray<NSNumber*>* indices, int64_t indexVectorLen) {
    if (indexVectorLen != 1) return false;
    NSUInteger n = indices.count;
    if (n == 0) return true;
    NSMutableSet<NSNumber*>* seen = [NSMutableSet setWithCapacity:n];
    for (NSUInteger i = 0; i < n; ++i) {
        NSNumber* idx = indices[i];
        if ([seen containsObject:idx]) {
            return false;
        }
        [seen addObject:idx];
    }
    return true;
}

static NSArray<NSNumber*>* ComputeFlattenedIndices(NSArray<NSNumber*>* indices,
                                                   NSArray<NSNumber*>* inputShape,
                                                   int64_t indexVectorLen,
                                                   NSArray<NSNumber*>* updateWindowDims,
                                                   NSArray<NSNumber*>* insertedWindowDims,
                                                   NSArray<NSNumber*>* scatterDimsToOperandDims) {
    if (indexVectorLen != 1) return nil;
    if (updateWindowDims.count != 0) return nil;
    if (insertedWindowDims.count != 1) return nil;
    if (scatterDimsToOperandDims.count != 1) return nil;
    if (insertedWindowDims[0].integerValue != 0) return nil;
    if (scatterDimsToOperandDims[0].integerValue != 0) return nil;

    NSUInteger n = indices.count;
    int64_t totalSize = inputShape[0].longLongValue;

    NSMutableArray<NSNumber*>* flatIndices = [NSMutableArray arrayWithCapacity:n];
    for (NSUInteger i = 0; i < n; ++i) {
        [flatIndices addObject:indices[i]];
    }
    return flatIndices;
}

// Sort indices and updates together by index value for cache-friendly parallel scatter
static std::tuple<MPSGraphTensor*, MPSGraphTensor*>
SortIndicesAndUpdatesForParallelScatter(HandlerContext& ctx,
                                        MPSGraphTensor* indices,
                                        MPSGraphTensor* updates,
                                        MPSGraphTensor* input) {
    (void)input;  // input is only used for graph-side context

    // argSort returns the permutation of indices that would sort them.
    // We then use that permutation to gather the corresponding rows of
    // `updates`, yielding a (sortedIndices, sortedUpdates) pair without
    // needing a fabricated range tensor (the SDK on this platform does not
    // expose a `rangeWithStart:limit:rate:` API).
    MPSGraphTensor* sortedIndices;
    if (indices.dataType == MPSDataTypeInt64) {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, indices);
        sortedIndices = [ctx.graph argSortWithTensor:int32Indices
                                                axis:0
                                          descending:NO
                                                name:nil];
    } else {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, indices);
        sortedIndices = [ctx.graph argSortWithTensor:int32Indices
                                                axis:0
                                          descending:NO
                                                name:nil];
    }

    MPSGraphTensor* sortedUpdates = [ctx.graph gatherAlongAxis:0
                                              withUpdatesTensor:updates
                                                  indicesTensor:sortedIndices
                                                           name:@"scatter_gather_sorted_updates"];

    return {sortedIndices, sortedUpdates};
}

// Parallel scatter using batched atomic updates for additive mode
static MPSGraphTensor* ParallelAddScatter(HandlerContext& ctx,
                                          MPSGraphTensor* input,
                                          MPSGraphTensor* indices,
                                          MPSGraphTensor* updates,
                                          int64_t indexVectorLen,
                                          NSArray<NSNumber*>* inputShape) {
    NSUInteger n = (NSUInteger)[updates.shape[0] unsignedIntegerValue];
    int64_t elemCount = 1;
    for (NSUInteger i = 1; i < inputShape.count; ++i) {
        elemCount *= inputShape[i].longLongValue;
    }

    MPSGraphTensor* result = input;

    if (elemCount > 0 && n > 0) {
        NSUInteger maxBatches = 256;
        NSUInteger batchSize = (n + maxBatches - 1) / maxBatches;
        if (batchSize < 8) batchSize = 8;
        if (batchSize > 256) batchSize = 256;

        NSUInteger numBatches = (n + batchSize - 1) / batchSize;

        NSMutableArray<MPSGraphTensor*>* partialResults = [NSMutableArray arrayWithCapacity:numBatches];

        for (NSUInteger b = 0; b < numBatches; ++b) {
            NSUInteger start = b * batchSize;
            NSUInteger end = start + batchSize;
            if (end > n) end = n;

            NSArray<NSNumber*>* sliceStart = @[@(start)];
            NSArray<NSNumber*>* sliceEnd = @[@(end)];

            MPSGraphTensor* batchIndices = [ctx.graph sliceTensor:indices
                                                           starts:sliceStart
                                                             ends:sliceEnd
                                                          strides:@[@1]
                                                             name:nil];
            MPSGraphTensor* batchUpdates = [ctx.graph sliceTensor:updates
                                                           starts:sliceStart
                                                             ends:sliceEnd
                                                          strides:@[@1]
                                                             name:nil];

            batchIndices = EnsureInt32(ctx.graph, batchIndices);

            MPSGraphTensor* batchResult = [ctx.graph scatterWithDataTensor:input
                                                            updatesTensor:batchUpdates
                                                            indicesTensor:batchIndices
                                                                     axis:0
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:nil];
            [partialResults addObject:batchResult];
        }

        if (partialResults.count == 1) {
            result = partialResults[0];
        } else {
            result = partialResults[0];
            for (NSUInteger i = 1; i < partialResults.count; ++i) {
                result = [ctx.graph additionWithPrimaryTensor:result
                                             secondaryTensor:partialResults[i]
                                                        name:nil];
            }
        }
    }

    return result;
}

// sliceUpdateDataTensor can drop writes to a scalar-broadcast base tensor.
// Reinsert an already-scattered window with explicit in-bounds coordinates instead.
static MPSGraphTensor* ReinsertScatterWindow(MPSGraph* graph, MPSGraphTensor* input, MPSGraphTensor* window) {
    NSUInteger rank = window.shape.count;
    NSMutableArray<MPSGraphTensor*>* coordinates = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        MPSGraphTensor* coord = [graph coordinateAlongAxis:(NSInteger)dim withShape:window.shape name:nil];
        [coordinates addObject:EnsureInt32(graph, coord)];
    }
    MPSGraphTensor* indices = [graph stackTensors:coordinates axis:(NSInteger)rank name:nil];
    return [graph scatterNDWithDataTensor:input
                            updatesTensor:window
                            indicesTensor:indices
                          batchDimensions:0
                                     mode:MPSGraphScatterModeSet
                                     name:nil];
}

static ProcessResult HandleScatterImpl(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp) {
        return ProcessResult::Error("scatter: expected ScatterOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(ctx, 1);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    if (!input || !scatterIndices || !updates)
        return ProcessResult::Error("scatter: missing input tensor");
    bool scatterBroadcastBase = input.dataType == MPSDataTypeFloat32 &&
                                static_cast<bool>(mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(
                                    ctx.op->getOperand(0).getDefiningOp()));

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;
    MPSGraphScatterMode mode = GetScatterMode(scatterOp);

    // Parallel scatter optimization for independent updates
    // Use sorted scatter + gather approach for additive scatter with unique indices
    bool isParallelOptimizable =
        inputBatchingDims.empty() &&
        scatterIndicesBatchingDims.empty() &&
        updateWindowDims.empty() &&
        insertedWindowDims.size() == 1 &&
        insertedWindowDims[0] == 0 &&
        scatterDimsToOperandDims.size() == 1 &&
        scatterDimsToOperandDims[0] == 0 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 &&
        input.shape.count == 2 &&
        updates.shape.count == 1 &&
        scatterOp.getNumOperands() == 3;

    if (isParallelOptimizable) {
        NSUInteger n = (NSUInteger)[scatterIndices.shape[0] unsignedIntegerValue];
        if (n > 16) {
            MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, scatterIndices);

            if (mode == MPSGraphScatterModeSet) {
                // For set mode with unique indices: single parallel scatter
                // Sort indices for cache-friendly writes
                auto [sortedIndices, sortedUpdates] =
                    SortIndicesAndUpdatesForParallelScatter(ctx, int32Indices, updates, input);

                MPSGraphTensor* result = [ctx.graph scatterWithDataTensor:input
                                                            updatesTensor:sortedUpdates
                                                            indicesTensor:sortedIndices
                                                                     axis:0
                                                                     mode:mode
                                                                     name:nil];
                return Result(ctx, result, "scatter");
            }

            if (mode == MPSGraphScatterModeAdd) {
                // For add mode: use batched parallel scatter to avoid read-modify-write
                MPSGraphTensor* result = ParallelAddScatter(ctx, input, int32Indices, updates, 1, input.shape);
                return Result(ctx, result, "scatter");
            }
        }
    }

    // Original specialized paths
    if (mode == MPSGraphScatterModeAdd && input.shape.count == 2 && updates.shape.count == 1 &&
        inputBatchingDims.empty() && scatterIndicesBatchingDims.empty() && updateWindowDims.empty() &&
        insertedWindowDims.size() == 2 && scatterDimsToOperandDims.size() == 2 &&
        insertedWindowDims[0] == 0 && insertedWindowDims[1] == 1 && scatterDimsToOperandDims[0] == 0 &&
        scatterDimsToOperandDims[1] == 1 && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 2) {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, scatterIndices);
        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:int32Indices
                                                    batchDimensions:0
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    if (mode == MPSGraphScatterModeAdd && input.shape.count == 3 && updates.shape.count == 2 &&
        inputBatchingDims.empty() && scatterIndicesBatchingDims.empty() && updateWindowDims.size() == 1 &&
        updateWindowDims[0] == 0 && insertedWindowDims.size() == 2 && insertedWindowDims[0] == 1 &&
        insertedWindowDims[1] == 2 && scatterDimsToOperandDims.size() == 2 &&
        scatterDimsToOperandDims[0] == 1 && scatterDimsToOperandDims[1] == 2 &&
        indexVectorDim == (int64_t)indicesRank - 1 && [indicesShape[indicesRank - 1] integerValue] == 2) {
        NSMutableArray<NSNumber*>* expandedIndicesShape = [NSMutableArray arrayWithCapacity:indicesRank + 1];
        [expandedIndicesShape addObject:@1];
        for (NSUInteger i = 0; i < indicesRank; ++i) {
            [expandedIndicesShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* expandedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:expandedIndicesShape
                                                              name:nil];

        NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray arrayWithCapacity:indicesRank + 1];
        [broadcastShape addObject:input.shape[0]];
        for (NSUInteger i = 0; i < indicesRank; ++i) {
            [broadcastShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* batchedIndices = [ctx.graph broadcastTensor:expandedIndices
                                                            toShape:broadcastShape
                                                               name:nil];
        batchedIndices = EnsureInt32(ctx.graph, batchedIndices);

        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:batchedIndices
                                                    batchDimensions:1
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    if (!inputBatchingDims.empty() && !scatterIndicesBatchingDims.empty() &&
        inputBatchingDims.size() == scatterIndicesBatchingDims.size() &&
        scatterDimsToOperandDims.size() == 1 && insertedWindowDims.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 && [indicesShape[indicesRank - 1] integerValue] == 1) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        if (insertedWindowDims[0] != scatterAxis) {

        } else {

            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesRank - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }
            MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                             withShape:squeezedShape
                                                                  name:nil];

            MPSGraphTensor* scatterInput = input;
            bool needsWindowUpdate = false;
            NSMutableArray<NSNumber*>* windowStarts = [NSMutableArray arrayWithCapacity:input.shape.count];
            NSMutableArray<NSNumber*>* windowEnds = [NSMutableArray arrayWithCapacity:input.shape.count];
            NSMutableArray<NSNumber*>* windowStrides = [NSMutableArray arrayWithCapacity:input.shape.count];
            if (input.shape.count == updates.shape.count) {
                for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                    [windowStarts addObject:@0];
                    [windowStrides addObject:@1];
                    int64_t inDim = [input.shape[dim] longLongValue];
                    int64_t upDim = [updates.shape[dim] longLongValue];
                    int64_t end = inDim;

                    if ((int64_t)dim != scatterAxis) {
                        if (inDim <= 0 || upDim < 0) {
                            return ProcessResult::Error(
                                "scatter: dynamic/unknown shapes not supported for window scatter");
                        }
                        if (upDim > inDim) {
                            return ProcessResult::Error(
                                "scatter: updates window exceeds input dim (" +
                                ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                        }
                        if (upDim < inDim) {
                            end = upDim;
                            needsWindowUpdate = true;
                        }
                    }
                    [windowEnds addObject:@(end)];
                }

                if (needsWindowUpdate) {
                    scatterInput = [ctx.graph sliceTensor:input
                                                   starts:windowStarts
                                                     ends:windowEnds
                                                  strides:windowStrides
                                                     name:nil];
                }
            }

            MPSGraphTensor* indicesForScatter = squeezedIndices;
            NSUInteger idxRank = indicesForScatter.shape.count;
            NSUInteger updRank = updates.shape.count;
            if (idxRank > updRank) {
                return ProcessResult::Error("scatter: indices rank exceeds updates rank (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            if (idxRank < updRank) {
                NSMutableArray<NSNumber*>* expandedShape =
                    [NSMutableArray arrayWithCapacity:idxRank + (updRank - idxRank)];
                for (NSUInteger i = 0; i < idxRank; ++i) {
                    [expandedShape addObject:indicesForScatter.shape[i]];
                }
                for (NSUInteger i = idxRank; i < updRank; ++i) {
                    [expandedShape addObject:@1];
                }
                indicesForScatter = [ctx.graph reshapeTensor:indicesForScatter
                                                   withShape:expandedShape
                                                        name:nil];
            }
            if (updates.shape && indicesForScatter) {
                indicesForScatter = [ctx.graph broadcastTensor:indicesForScatter
                                                       toShape:updates.shape
                                                          name:nil];
            }
            indicesForScatter = EnsureInt32(ctx.graph, indicesForScatter);

            MPSGraphTensor* resultWindow = [ctx.graph scatterAlongAxis:static_cast<NSInteger>(scatterAxis)
                                                        withDataTensor:scatterInput
                                                         updatesTensor:updates
                                                         indicesTensor:indicesForScatter
                                                                  mode:mode
                                                                  name:nil];

            if (!needsWindowUpdate) {
                return Result(ctx, resultWindow, "scatter");
            }
            if (scatterBroadcastBase) {
                return Result(ctx, ReinsertScatterWindow(ctx.graph, input, resultWindow), "scatter");
            }

            MPSGraphTensor* result = [ctx.graph sliceUpdateDataTensor:input
                                                         updateTensor:resultWindow
                                                               starts:windowStarts
                                                                 ends:windowEnds
                                                              strides:windowStrides
                                                                 name:nil];
            return Result(ctx, result, "scatter");
        }
    }

    if (inputBatchingDims.empty() && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        insertedWindowDims.size() == 1 && insertedWindowDims[0] == scatterDimsToOperandDims[0]) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        if (squeezedShape.count == 0)
            [squeezedShape addObject:@1];

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        if (updates.shape.count == 0)
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];

        // Inside control flow, MPSGraph's scatter ops have shape verification bugs.
        // Use one_hot + arithmetic approach instead.
        if (ctx.isInControlFlow() && mode == MPSGraphScatterModeSet) {
            NSUInteger inputRank = input.shape.count;
            int64_t axisDim = [input.shape[(NSUInteger)scatterAxis] longLongValue];
            int64_t numIdx = 1;
            for (NSNumber* d in squeezedShape) {
                numIdx *= [d longLongValue];
            }

            // Sets one slice along scatterAxis: flatIdx is a [1] index, upd has
            // the input's dims except scatterAxis. Elementwise ops only.
            auto setSlice = [&](MPSGraphTensor* base, MPSGraphTensor* flatIdx,
                                MPSGraphTensor* upd) -> MPSGraphTensor* {
                MPSGraphTensor* updates = upd;
                MPSGraphTensor* input = base;
                MPSGraphTensor* oneHot = [ctx.graph oneHotWithIndicesTensor:flatIdx
                                                                      depth:(NSUInteger)axisDim
                                                                       axis:1
                                                                   dataType:MPSDataTypeFloat32
                                                                    onValue:1.0
                                                                   offValue:0.0
                                                                       name:nil];
                // oneHot shape: [1, axisDim]
                // Squeeze to [axisDim]
                oneHot = [ctx.graph reshapeTensor:oneHot withShape:@[@(axisDim)] name:nil];

                // Build a boolean mask over the input shape: true at the scatter position(s).
                // Reshape oneHot to be broadcastable: size axisDim on scatterAxis, 1 elsewhere.
                NSMutableArray<NSNumber*>* maskShape = [NSMutableArray arrayWithCapacity:inputRank];
                for (NSUInteger d = 0; d < inputRank; ++d) {
                    [maskShape addObject:((int64_t)d == scatterAxis ? @(axisDim) : @1)];
                }
                MPSGraphTensor* maskFloat = [ctx.graph reshapeTensor:oneHot withShape:maskShape name:nil];
                // mask is 1.0 at scatter position, 0.0 elsewhere.

                // Expand updates to match input shape.
                // updates has all dims of input EXCEPT the scatterAxis dim is missing
                // (it was "inserted" — i.e., the update covers a single slice along scatterAxis).
                // We need to unsqueeze updates along scatterAxis to make it broadcastable.
                NSMutableArray<NSNumber*>* expandedUpdatesShape =
                    [NSMutableArray arrayWithCapacity:inputRank];
                NSUInteger updateDimIdx = 0;
                for (NSUInteger d = 0; d < inputRank; ++d) {
                    if ((int64_t)d == scatterAxis) {
                        [expandedUpdatesShape addObject:@1];
                    } else {
                        if (updateDimIdx < updates.shape.count) {
                            [expandedUpdatesShape addObject:updates.shape[updateDimIdx]];
                            ++updateDimIdx;
                        } else {
                            [expandedUpdatesShape addObject:@1];
                        }
                    }
                }
                MPSGraphTensor* expandedUpdates = [ctx.graph reshapeTensor:updates
                                                                 withShape:expandedUpdatesShape
                                                                      name:nil];
                // expandedUpdates: [1, 4] for our 10x4 case (1 along scatterAxis, 4 along other dims)

                // result = where(mask, updates, input), in the input dtype. An
                // arithmetic float32 blend (input*(1-mask) + updates*mask) lost
                // int32 precision above 2**24, spread NaN from a non-finite update
                // into every other slice, and let a non-finite target survive.
                MPSGraphTensor* zeroF = [ctx.graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
                MPSGraphTensor* maskBool = [ctx.graph notEqualWithPrimaryTensor:maskFloat
                                                                secondaryTensor:zeroF
                                                                           name:nil];
                if (expandedUpdates.dataType != input.dataType) {
                    expandedUpdates = [ctx.graph castTensor:expandedUpdates toType:input.dataType name:nil];
                }
                return [ctx.graph selectWithPredicateTensor:maskBool
                                        truePredicateTensor:expandedUpdates
                                       falsePredicateTensor:input
                                                       name:nil];
            };

            // A single index whose updates keep a size-1 batch dim (full input
            // rank) goes through the per-index path, which drops that dim.
            if (numIdx == 1 && updates.shape.count != inputRank) {
                MPSGraphTensor* flatIdx = [ctx.graph reshapeTensor:squeezedIndices withShape:@[@1] name:nil];
                return Result(ctx, setSlice(input, flatIdx, updates), "scatter");
            }
            // Multi-row window set (e.g. LU row swaps, a.at[idx].set(rows)):
            // the single-index lowering reshaped K updates into one slice and
            // aborted on a broadcast-incompatible select. Apply it per index,
            // in order (later indices win; duplicate order is implementation-
            // defined). Bounded K keeps the graph small; larger K falls through.
            if (indicesRank == 2 && updates.shape.count == inputRank && numIdx <= 64) {
                NSInteger batchPos = -1;
                for (NSUInteger d = 0; d < updates.shape.count && batchPos < 0; ++d) {
                    bool isWindow = false;
                    for (auto w : updateWindowDims) {
                        if ((NSUInteger)w == d) {
                            isWindow = true;
                        }
                    }
                    if (!isWindow) {
                        batchPos = (NSInteger)d;
                    }
                }
                if (batchPos >= 0 && [updates.shape[(NSUInteger)batchPos] longLongValue] == numIdx) {
                    NSMutableArray<NSNumber*>* sliceShape = [NSMutableArray array];
                    for (NSUInteger d = 0; d < updates.shape.count; ++d) {
                        if ((NSInteger)d != batchPos) {
                            [sliceShape addObject:updates.shape[d]];
                        }
                    }
                    MPSGraphTensor* acc = input;
                    for (int64_t k = 0; k < numIdx; ++k) {
                        MPSGraphTensor* idxK = [ctx.graph sliceTensor:squeezedIndices
                                                            dimension:0
                                                                start:(NSInteger)k
                                                               length:1
                                                                 name:nil];
                        MPSGraphTensor* updK = [ctx.graph sliceTensor:updates
                                                            dimension:batchPos
                                                                start:(NSInteger)k
                                                               length:1
                                                                 name:nil];
                        updK = [ctx.graph reshapeTensor:updK withShape:sliceShape name:nil];
                        acc = setSlice(acc, idxK, updK);
                    }
                    return Result(ctx, acc, "scatter");
                }
            }
        }

        // When insertedWindowDims removes the scatter axis from updates, unsqueeze it back.
        // E.g., input=[10,4] scatter_axis=0 updates=[4] → need updates=[1,4]
        if (updates.shape.count + insertedWindowDims.size() == input.shape.count) {
            NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
            NSUInteger updateIdx = 0;
            for (NSUInteger d = 0; d < input.shape.count; ++d) {
                bool isInserted = false;
                for (auto iwd : insertedWindowDims) {
                    if ((NSUInteger)iwd == d) {
                        isInserted = true;
                        break;
                    }
                }
                if (isInserted) {
                    [expandedShape addObject:@1];
                } else {
                    if (updateIdx < updates.shape.count) {
                        [expandedShape addObject:updates.shape[updateIdx++]];
                    } else {
                        [expandedShape addObject:@1];
                    }
                }
            }
            updates = [ctx.graph reshapeTensor:updates withShape:expandedShape name:nil];
        }

        MPSGraphTensor* scatterInput = input;
        bool needsWindowUpdate = false;
        NSMutableArray<NSNumber*>* windowStarts = [NSMutableArray arrayWithCapacity:input.shape.count];
        NSMutableArray<NSNumber*>* windowEnds = [NSMutableArray arrayWithCapacity:input.shape.count];
        NSMutableArray<NSNumber*>* windowStrides = [NSMutableArray arrayWithCapacity:input.shape.count];
        if (input.shape.count == updates.shape.count) {
            for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                [windowStarts addObject:@0];
                [windowStrides addObject:@1];
                int64_t inDim = [input.shape[dim] longLongValue];
                int64_t upDim = [updates.shape[dim] longLongValue];
                int64_t end = inDim;
                if ((int64_t)dim != scatterAxis) {
                    if (inDim <= 0 || upDim < 0) {
                        return ProcessResult::Error(
                            "scatter: dynamic/unknown shapes not supported for window scatter");
                    }
                    if (upDim > inDim) {
                        return ProcessResult::Error(
                            "scatter: updates window exceeds input dim (" +
                            ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                    }
                    if (upDim < inDim) {
                        end = upDim;
                        needsWindowUpdate = true;
                    }
                }
                [windowEnds addObject:@(end)];
            }
            if (needsWindowUpdate) {
                scatterInput = [ctx.graph sliceTensor:input
                                               starts:windowStarts
                                                 ends:windowEnds
                                              strides:windowStrides
                                                 name:nil];
            }
        }

        MPSGraphTensor* resultWindow = [ctx.graph scatterWithDataTensor:scatterInput
                                                          updatesTensor:updates
                                                          indicesTensor:squeezedIndices
                                                                   axis:static_cast<NSInteger>(scatterAxis)
                                                                   mode:mode
                                                                   name:nil];

        if (!needsWindowUpdate) {
            return Result(ctx, resultWindow, "scatter");
        }
        if (scatterBroadcastBase) {
            return Result(ctx, ReinsertScatterWindow(ctx.graph, input, resultWindow), "scatter");
        }

        MPSGraphTensor* result = [ctx.graph sliceUpdateDataTensor:input
                                                     updateTensor:resultWindow
                                                           starts:windowStarts
                                                             ends:windowEnds
                                                          strides:windowStrides
                                                             name:nil];
        return Result(ctx, result, "scatter");
    }

    if (inputBatchingDims.empty() && scatterIndicesBatchingDims.empty()) {
        NSUInteger operandRank = input.shape.count;
        NSUInteger updatesRank = updates.shape.count;
        // Scalar update into a rank >= 1 operand (x.at[i, j].set(v)): the index
        // vector addresses one element. Build a mask where every coordinate equals
        // its index component (an out-of-bounds index matches nothing, matching
        // StableHLO's skip semantics), apply the reducer there, and select. Only
        // elementwise ops, so it is also safe inside loop bodies.
        if (operandRank > 0 && updatesRank == 0 && updateWindowDims.empty() &&
            insertedWindowDims.size() == operandRank && scatterDimsToOperandDims.size() == operandRank) {
            int64_t idxCount = 1;
            for (NSNumber* d in scatterIndices.shape) {
                idxCount *= [d longLongValue];
            }
            MPSDataType dt = input.dataType;
            const bool isFloat =
                dt == MPSDataTypeFloat32 || dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
            const bool isNarrowInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8 ||
                                     dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16;
            const bool isU32 = dt == MPSDataTypeUInt32;
            const bool isInt = isNarrowInt || isU32 || dt == MPSDataTypeInt32;
            if (idxCount == (int64_t)operandRank && (isFloat || isInt) &&
                !(isInt && mode == MPSGraphScatterModeDiv)) {
                MPSGraph* g = ctx.graph;
                MPSGraphTensor* flatIdx = [g reshapeTensor:scatterIndices withShape:@[@(idxCount)] name:nil];
                flatIdx = EnsureInt32(g, flatIdx);
                MPSGraphTensor* mask = nil;
                for (NSUInteger k = 0; k < operandRank; ++k) {
                    NSInteger dim = (NSInteger)scatterDimsToOperandDims[k];
                    MPSGraphTensor* coord = [g coordinateAlongAxis:dim withShape:input.shape name:nil];
                    coord = [g castTensor:coord toType:MPSDataTypeInt32 name:nil];
                    MPSGraphTensor* comp = [g sliceTensor:flatIdx
                                                dimension:0
                                                    start:(NSInteger)k
                                                   length:1
                                                     name:nil];
                    MPSGraphTensor* eq = [g equalWithPrimaryTensor:coord secondaryTensor:comp name:nil];
                    mask = mask ? [g logicalANDWithPrimaryTensor:mask secondaryTensor:eq name:nil] : eq;
                }
                MPSGraphTensor* upd =
                    updates.dataType == dt ? updates : [g castTensor:updates toType:dt name:nil];
                // Lift the rank-0 update to rank >= 1: MPSGraph reinterpret_cast (used
                // for uint32 below) aborts the process on rank-0 tensors.
                {
                    NSMutableArray<NSNumber*>* ones = [NSMutableArray arrayWithCapacity:operandRank];
                    for (NSUInteger d = 0; d < operandRank; ++d) {
                        [ones addObject:@1];
                    }
                    upd = [g reshapeTensor:upd withShape:ones name:nil];
                }
                // Integer arithmetic in int32 (8/16-bit MPSGraph arithmetic saturated;
                // uint32 via bit reinterpretation), converted back with wraparound.
                auto toWork = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
                    if (isNarrowInt)
                        return [g castTensor:t toType:MPSDataTypeInt32 name:nil];
                    if (isU32)
                        return [g reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
                    return t;
                };
                auto fromWork = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
                    if (isNarrowInt)
                        return [g castTensor:t toType:dt name:nil];
                    if (isU32)
                        return [g reinterpretCastTensor:t toType:dt name:nil];
                    return t;
                };
                MPSGraphTensor* combined = nil;
                if (mode == MPSGraphScatterModeSet) {
                    combined = upd;
                } else if (mode == MPSGraphScatterModeMax || mode == MPSGraphScatterModeMin) {
                    const bool isMax = mode == MPSGraphScatterModeMax;
                    if (isFloat) {
                        combined = isMax ? [g maximumWithNaNPropagationWithPrimaryTensor:input
                                                                         secondaryTensor:upd
                                                                                    name:nil]
                                         : [g minimumWithNaNPropagationWithPrimaryTensor:input
                                                                         secondaryTensor:upd
                                                                                    name:nil];
                    } else if (isU32) {
                        // Sign-bit XOR maps uint32 order onto int32 order.
                        MPSGraphTensor* sb = [g constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
                        MPSGraphTensor* a = [g bitwiseXORWithPrimaryTensor:toWork(input)
                                                           secondaryTensor:sb
                                                                      name:nil];
                        MPSGraphTensor* b = [g bitwiseXORWithPrimaryTensor:toWork(upd)
                                                           secondaryTensor:sb
                                                                      name:nil];
                        MPSGraphTensor* r = isMax ? [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil]
                                                  : [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
                        combined = fromWork([g bitwiseXORWithPrimaryTensor:r secondaryTensor:sb name:nil]);
                    } else {
                        combined = isMax ? [g maximumWithPrimaryTensor:input secondaryTensor:upd name:nil]
                                         : [g minimumWithPrimaryTensor:input secondaryTensor:upd name:nil];
                    }
                } else {
                    MPSGraphTensor* a = isInt ? toWork(input) : input;
                    MPSGraphTensor* b = isInt ? toWork(upd) : upd;
                    MPSGraphTensor* r = nil;
                    if (mode == MPSGraphScatterModeAdd)
                        r = [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeSub)
                        r = [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeMul)
                        r = [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeDiv)
                        r = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    combined = (r && isInt) ? fromWork(r) : r;
                }
                if (combined && mask) {
                    MPSGraphTensor* result = [g selectWithPredicateTensor:mask
                                                      truePredicateTensor:combined
                                                     falsePredicateTensor:input
                                                                     name:nil];
                    return Result(ctx, result, "scatter");
                }
            }
        }
        if (operandRank == 0 || updatesRank == 0) {
            return ProcessResult::Error("scatter: scalar scatter is not supported yet (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }

        MPSGraphTensor* indicesTensor = scatterIndices;
        NSArray<NSNumber*>* indicesTensorShape = indicesTensor.shape;
        NSUInteger indicesTensorRank = indicesTensorShape.count;
        if (indicesTensorRank == 0) {
            return ProcessResult::Error("scatter: invalid scatter_indices rank (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        if (indexVectorDim < 0 || indexVectorDim >= (int64_t)indicesTensorRank) {
            return ProcessResult::Error("scatter: invalid index_vector_dim (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        if (indexVectorDim != (int64_t)indicesTensorRank - 1) {
            NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:indicesTensorRank];
            for (NSUInteger d = 0; d < indicesTensorRank; ++d) {
                if ((int64_t)d == indexVectorDim) {
                    continue;
                }
                [perm addObject:@((NSInteger)d)];
            }
            [perm addObject:@((NSInteger)indexVectorDim)];
            if (perm.count != indicesTensorRank) {
                return ProcessResult::Error("scatter: invalid scatter_indices permutation (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            indicesTensor = [ctx.graph transposeTensor:indicesTensor permutation:perm name:nil];
            indicesTensorShape = indicesTensor.shape;
            indicesTensorRank = indicesTensorShape.count;
        }
        indicesTensor = EnsureInt32(ctx.graph, indicesTensor);

        std::vector<int64_t> operandDimsWithoutInserted;
        operandDimsWithoutInserted.reserve(operandRank);
        for (NSUInteger d = 0; d < operandRank; ++d) {
            if (ContainsDim(insertedWindowDims, (int64_t)d)) {
                continue;
            }
            operandDimsWithoutInserted.push_back((int64_t)d);
        }
        if (operandDimsWithoutInserted.size() != updateWindowDims.size()) {
            return ProcessResult::Error("scatter: unsupported update_window_dims mapping (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        std::vector<int64_t> operandToUpdateDim(operandRank, -1);
        for (size_t i = 0; i < operandDimsWithoutInserted.size(); ++i) {
            int64_t updateDim = updateWindowDims[i];
            if (updateDim < 0 || updateDim >= (int64_t)updatesRank) {
                return ProcessResult::Error("scatter: invalid update_window_dims (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            operandToUpdateDim[(size_t)operandDimsWithoutInserted[i]] = updateDim;
        }

        std::vector<int64_t> updateScatterDims;
        updateScatterDims.reserve(updatesRank);
        for (NSUInteger d = 0; d < updatesRank; ++d) {
            if (!ContainsDim(updateWindowDims, (int64_t)d)) {
                updateScatterDims.push_back((int64_t)d);
            }
        }

        if (indicesTensorRank < 1) {
            return ProcessResult::Error("scatter: invalid scatter_indices rank (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        int64_t indexVectorLen = [indicesTensor.shape[indicesTensorRank - 1] longLongValue];
        if (indexVectorLen <= 0) {
            return ProcessResult::Error("scatter: invalid index vector length (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        NSUInteger scatterPrefixRank = indicesTensorRank - 1;
        if (scatterPrefixRank != updateScatterDims.size()) {
            return ProcessResult::Error("scatter: unsupported scatter_indices/update dims (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }

        NSMutableArray<NSNumber*>* reshapedIndicesShape = [NSMutableArray arrayWithCapacity:updatesRank + 1];

        for (NSUInteger d = 0; d < updatesRank; ++d) {
            NSInteger size = 1;
            for (size_t i = 0; i < updateScatterDims.size(); ++i) {
                if (updateScatterDims[i] == (int64_t)d) {
                    size = [indicesTensor.shape[i] integerValue];
                    break;
                }
            }
            [reshapedIndicesShape addObject:@(size)];
        }
        [reshapedIndicesShape addObject:@(indexVectorLen)];

        MPSGraphTensor* expandedIndices = [ctx.graph reshapeTensor:indicesTensor
                                                         withShape:reshapedIndicesShape
                                                              name:nil];
        NSMutableArray<NSNumber*>* broadcastIndicesShape = [NSMutableArray arrayWithCapacity:updatesRank + 1];
        for (NSUInteger d = 0; d < updatesRank; ++d) {
            [broadcastIndicesShape addObject:updates.shape[d]];
        }
        [broadcastIndicesShape addObject:@(indexVectorLen)];
        MPSGraphTensor* broadcastIndices = [ctx.graph broadcastTensor:expandedIndices
                                                              toShape:broadcastIndicesShape
                                                                 name:nil];

        std::vector<MPSGraphTensor*> indexComponents;
        indexComponents.reserve((size_t)indexVectorLen);
        for (int64_t i = 0; i < indexVectorLen; ++i) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            for (NSUInteger d = 0; d < updatesRank; ++d) {
                [starts addObject:@0];
                [ends addObject:updates.shape[d]];
                [strides addObject:@1];
            }
            [starts addObject:@(i)];
            [ends addObject:@(i + 1)];
            [strides addObject:@1];
            MPSGraphTensor* slice = [ctx.graph sliceTensor:broadcastIndices
                                                    starts:starts
                                                      ends:ends
                                                   strides:strides
                                                      name:nil];

            slice = [ctx.graph reshapeTensor:slice withShape:updates.shape name:nil];
            indexComponents.push_back(slice);
        }

        MPSGraphTensor* zeroScalar = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* zero = [ctx.graph broadcastTensor:zeroScalar toShape:updates.shape name:nil];

        NSMutableArray<MPSGraphTensor*>* coordTensors = [NSMutableArray arrayWithCapacity:operandRank];
        for (NSUInteger operandDim = 0; operandDim < operandRank; ++operandDim) {
            MPSGraphTensor* coord = zero;
            int64_t updateDim = operandToUpdateDim[operandDim];
            if (updateDim >= 0) {
                coord = [ctx.graph coordinateAlongAxis:(NSInteger)updateDim withShape:updates.shape name:nil];
                coord = EnsureInt32(ctx.graph, coord);
            }

            for (size_t i = 0; i < (size_t)scatterDimsToOperandDims.size(); ++i) {
                if (scatterDimsToOperandDims[i] == (int64_t)operandDim) {
                    if (i >= indexComponents.size()) {
                        return ProcessResult::Error(
                            "scatter: scatter_dims_to_operand_dims out of range (" +
                            ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                    }
                    coord = [ctx.graph additionWithPrimaryTensor:coord
                                                 secondaryTensor:indexComponents[i]
                                                            name:nil];
                    break;
                }
            }
            [coordTensors addObject:coord];
        }

        MPSGraphTensor* ndIndices = [ctx.graph stackTensors:coordTensors
                                                       axis:(NSInteger)updatesRank
                                                       name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:ndIndices
                                                    batchDimensions:0
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    return ProcessResult::Error("scatter: unsupported scatter pattern (" +
                                ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
}
// MPSGraph scatter moved uint32 values inexactly (off by 1 above 2**24, and
// add saturated to INT32_MAX above 2**31) while int32 scatter was exact. For
// set and add (mod 2**32), scatter the int32 reinterpretation and reinterpret
// the result back; for min/max, additionally flip the sign bit so int32
// ordering matches uint32 ordering.
static ProcessResult HandleScatter(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp || ctx.op->getNumOperands() != 3 || ctx.op->getNumResults() != 1) {
        return HandleScatterImpl(ctx);
    }
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    MPSGraphScatterMode mode = GetScatterMode(scatterOp);

    // bool scatter returned wrong values (set) or was rejected (scalar
    // updates). Scatter the exact 0/1 int32 widening and map back with != 0,
    // which also yields bool semantics per reducer: add->OR, mul->AND,
    // max->OR, min->AND.
    if (input && updates && input.dataType == MPSDataTypeBool && updates.dataType == MPSDataTypeBool &&
        ctx.op->getOperand(0) != ctx.op->getOperand(2) && ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        ctx.op->getOperand(1) != ctx.op->getOperand(2)) {
        void* inKey = ctx.op->getOperand(0).getAsOpaquePointer();
        void* upKey = ctx.op->getOperand(2).getAsOpaquePointer();
        if (!(ctx.variables && (ctx.variables->find(inKey) != ctx.variables->end() ||
                                ctx.variables->find(upKey) != ctx.variables->end()))) {
            auto savedIn = ctx.values.find(inKey);
            MPSGraphTensor* origIn = savedIn != ctx.values.end() ? savedIn->second : nil;
            auto savedUp = ctx.values.find(upKey);
            MPSGraphTensor* origUp = savedUp != ctx.values.end() ? savedUp->second : nil;
            ctx.values[inKey] = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
            ctx.values[upKey] = [ctx.graph castTensor:updates toType:MPSDataTypeInt32 name:nil];
            ProcessResult r = HandleScatterImpl(ctx);
            auto restoreKey = [&](void* key, MPSGraphTensor* orig) {
                if (orig) {
                    ctx.values[key] = orig;
                } else {
                    ctx.values.erase(key);
                }
            };
            restoreKey(inKey, origIn);
            restoreKey(upKey, origUp);
            if (!r.ok()) {
                return r;
            }
            MPSGraphTensor* scattered = GetTensor(ctx.values, ctx.op->getResult(0));
            if (!scattered) {
                return ProcessResult::Error("scatter: bool widening lowering failed");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:scattered.dataType];
            MPSGraphTensor* result = [ctx.graph notEqualWithPrimaryTensor:scattered
                                                          secondaryTensor:zero
                                                                     name:nil];
            return Result(ctx, result, "scatter");
        }
    }

    // mps.scatter rejects complex operands with a process-aborting assertion.
    // set and add act independently on real and imaginary parts, so scatter
    // each part through the float path and recombine (exact).
    auto isComplex = [](MPSGraphTensor* t) {
        return t && (t.dataType == MPSDataTypeComplexFloat32 || t.dataType == MPSDataTypeComplexFloat16);
    };
    if (isComplex(input) && isComplex(updates) &&
        (mode == MPSGraphScatterModeSet || mode == MPSGraphScatterModeAdd) &&
        ctx.op->getOperand(0) != ctx.op->getOperand(2) && ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        ctx.op->getOperand(1) != ctx.op->getOperand(2)) {
        void* inKey = ctx.op->getOperand(0).getAsOpaquePointer();
        void* upKey = ctx.op->getOperand(2).getAsOpaquePointer();
        if (!(ctx.variables && (ctx.variables->find(inKey) != ctx.variables->end() ||
                                ctx.variables->find(upKey) != ctx.variables->end()))) {
            auto savedIn = ctx.values.find(inKey);
            MPSGraphTensor* origIn = savedIn != ctx.values.end() ? savedIn->second : nil;
            auto savedUp = ctx.values.find(upKey);
            MPSGraphTensor* origUp = savedUp != ctx.values.end() ? savedUp->second : nil;
            auto scatterPart = [&](bool real, MPSGraphTensor** out) -> ProcessResult {
                ctx.values[inKey] = real ? [ctx.graph realPartOfTensor:input name:nil]
                                         : [ctx.graph imaginaryPartOfTensor:input name:nil];
                ctx.values[upKey] = real ? [ctx.graph realPartOfTensor:updates name:nil]
                                         : [ctx.graph imaginaryPartOfTensor:updates name:nil];
                ProcessResult r = HandleScatterImpl(ctx);
                if (r.ok()) {
                    *out = GetTensor(ctx.values, ctx.op->getResult(0));
                }
                return r;
            };
            MPSGraphTensor* sRe = nil;
            MPSGraphTensor* sIm = nil;
            ProcessResult r = scatterPart(true, &sRe);
            if (r.ok()) {
                r = scatterPart(false, &sIm);
            }
            auto restoreKey = [&](void* key, MPSGraphTensor* orig) {
                if (orig) {
                    ctx.values[key] = orig;
                } else {
                    ctx.values.erase(key);
                }
            };
            restoreKey(inKey, origIn);
            restoreKey(upKey, origUp);
            if (!r.ok()) {
                return r;
            }
            if (!sRe || !sIm) {
                return ProcessResult::Error("scatter: complex split lowering failed");
            }
            return Result(ctx, [ctx.graph complexTensorWithRealTensor:sRe imaginaryTensor:sIm name:nil],
                          "scatter");
        }
    }

    // min/max: XOR with the sign bit maps uint32 order onto int32 order.
    const bool orderMode = mode == MPSGraphScatterModeMin || mode == MPSGraphScatterModeMax;
    // MPSGraph reinterpret_cast aborts the process on rank-0 tensors, so scalar
    // updates take HandleScatterImpl's scalar-scatter branch instead.
    if (!input || !updates || updates.shape.count == 0 || input.dataType != MPSDataTypeUInt32 ||
        updates.dataType != MPSDataTypeUInt32 ||
        (mode != MPSGraphScatterModeSet && mode != MPSGraphScatterModeAdd && !orderMode) ||
        ctx.op->getOperand(0) == ctx.op->getOperand(2) || ctx.op->getOperand(0) == ctx.op->getOperand(1) ||
        ctx.op->getOperand(1) == ctx.op->getOperand(2)) {
        return HandleScatterImpl(ctx);
    }
    void* inputKey = ctx.op->getOperand(0).getAsOpaquePointer();
    void* updatesKey = ctx.op->getOperand(2).getAsOpaquePointer();
    if (ctx.variables && (ctx.variables->find(inputKey) != ctx.variables->end() ||
                          ctx.variables->find(updatesKey) != ctx.variables->end())) {
        return HandleScatterImpl(ctx);
    }

    auto savedInput = ctx.values.find(inputKey);
    MPSGraphTensor* origInput = savedInput != ctx.values.end() ? savedInput->second : nil;
    auto savedUpdates = ctx.values.find(updatesKey);
    MPSGraphTensor* origUpdates = savedUpdates != ctx.values.end() ? savedUpdates->second : nil;
    auto restore = [&](void* key, MPSGraphTensor* orig) {
        if (orig) {
            ctx.values[key] = orig;
        } else {
            ctx.values.erase(key);
        }
    };

    MPSGraphTensor* signBit = [ctx.graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
    auto toInt32 = [&](MPSGraphTensor* t) {
        MPSGraphTensor* bits = [ctx.graph reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
        return orderMode ? [ctx.graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:signBit name:nil]
                         : bits;
    };
    ctx.values[inputKey] = toInt32(input);
    ctx.values[updatesKey] = toInt32(updates);
    ProcessResult r = HandleScatterImpl(ctx);
    restore(inputKey, origInput);
    restore(updatesKey, origUpdates);
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* scattered = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!scattered || scattered.dataType != MPSDataTypeInt32) {
        ctx.values.erase(ctx.op->getResult(0).getAsOpaquePointer());
        return HandleScatterImpl(ctx);
    }
    if (orderMode) {
        scattered = [ctx.graph bitwiseXORWithPrimaryTensor:scattered secondaryTensor:signBit name:nil];
    }
    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:scattered toType:MPSDataTypeUInt32 name:nil];
    return Result(ctx, result, "scatter");
}
REGISTER_MPS_OP("stablehlo.scatter", HandleScatter);

static ProcessResult HandleReverse(HandlerContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(ctx.op);
    if (!reverseOp) {
        return ProcessResult::Error("reverse: expected ReverseOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reverse: missing input tensor");

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    // reverseTensor on 8/16-bit tensors segfaulted (1-D float16/int8/uint8,
    // eager), returned wrong values (1-D int16, jit), and float16
    // reverse(sort(x)) under JIT came back unreversed. 32-bit reverses were
    // correct, so widen exactly (f16 -> f32, small ints -> int32), reverse,
    // and cast back.
    MPSDataType dtype = input.dataType;
    MPSDataType wideType = MPSDataTypeInvalid;
    if (dtype == MPSDataTypeFloat16) {
        wideType = MPSDataTypeFloat32;
    } else if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeInt16 ||
               dtype == MPSDataTypeUInt16) {
        wideType = MPSDataTypeInt32;
    }
    if (wideType != MPSDataTypeInvalid) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:wideType name:nil];
        MPSGraphTensor* reversed = [ctx.graph reverseTensor:wide axes:axes name:nil];
        return Result(ctx, [ctx.graph castTensor:reversed toType:dtype name:nil], "reverse");
    }

    MPSGraphTensor* result = [ctx.graph reverseTensor:input axes:axes name:nil];
    return Result(ctx, result, "reverse");
}
REGISTER_MPS_OP("stablehlo.reverse", HandleReverse);

} // namespace jax_metallib
