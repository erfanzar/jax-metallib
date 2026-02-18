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
#include "stablehlo/dialect/StablehloOps.h"

#include <cmath>
#include <sstream>

namespace jax_metallib {

namespace {

std::string GetReductionOpType(mlir::Region& body) {
    if (body.empty())
        return "";

    mlir::Block& block = body.front();

    for (mlir::Operation& op : block) {
        std::string opName = op.getName().getStringRef().str();

        if (opName == "stablehlo.return")
            continue;

        return opName;
    }

    return "";
}

enum class ArgReduceKind { kUnknown, kMax, kMin };

bool IsBlockArg(mlir::Value value, mlir::Block& block, unsigned index) {
    auto arg = mlir::dyn_cast<mlir::BlockArgument>(value);
    return arg && arg.getOwner() == &block && arg.getArgNumber() == index;
}

ArgReduceKind detectArgReduceKind(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return ArgReduceKind::kUnknown;
    }
    mlir::Block& body = reduceOp.getBody().front();
    if (body.getNumArguments() < 4) {
        return ArgReduceKind::kUnknown;
    }

    for (mlir::Operation& nestedOp : body) {
        auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
        if (!compareOp) {
            continue;
        }

        bool forwardValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 0) && IsBlockArg(compareOp.getRhs(), body, 2);
        bool reversedValueCompare =
            IsBlockArg(compareOp.getLhs(), body, 2) && IsBlockArg(compareOp.getRhs(), body, 0);
        if (!forwardValueCompare && !reversedValueCompare) {
            continue;
        }

        auto dir = compareOp.getComparisonDirection();
        bool lhsWins = false;
        if (dir == mlir::stablehlo::ComparisonDirection::GT ||
            dir == mlir::stablehlo::ComparisonDirection::GE) {
            lhsWins = true;
        } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                   dir == mlir::stablehlo::ComparisonDirection::LE) {
            lhsWins = false;
        } else {
            continue;
        }

        if (reversedValueCompare) {
            lhsWins = !lhsWins;
        }
        return lhsWins ? ArgReduceKind::kMax : ArgReduceKind::kMin;
    }
    return ArgReduceKind::kUnknown;
}

std::string JoinInts(llvm::ArrayRef<int64_t> values) {
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

std::string JoinPadding(const std::vector<std::pair<int64_t, int64_t>>& padding) {
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < padding.size(); ++i) {
        if (i > 0) {
            oss << ", ";
        }
        oss << "[" << padding[i].first << ", " << padding[i].second << "]";
    }
    oss << "]";
    return oss.str();
}

bool ReadArrayOrDefaultOnes(std::optional<llvm::ArrayRef<int64_t>> attr, size_t rank,
                            std::vector<int64_t>& out) {
    out.assign(rank, 1);
    if (!attr.has_value()) {
        return true;
    }
    if (attr->size() != rank) {
        return false;
    }
    out.assign(attr->begin(), attr->end());
    return true;
}

bool ReadPaddingOrDefaultZeros(std::optional<mlir::DenseIntElementsAttr> paddingAttr, size_t rank,
                               std::vector<std::pair<int64_t, int64_t>>& out) {
    out.clear();
    out.resize(rank, {0, 0});
    if (!paddingAttr.has_value()) {
        return true;
    }

    std::vector<int64_t> flatPadding;
    flatPadding.reserve(rank * 2);
    for (int64_t value : paddingAttr->getValues<int64_t>()) {
        flatPadding.push_back(value);
    }
    if (flatPadding.size() != rank * 2) {
        return false;
    }

    for (size_t i = 0; i < rank; ++i) {
        out[i] = {flatPadding[2 * i], flatPadding[2 * i + 1]};
    }
    return true;
}

std::string DescribeReduceWindowPattern(mlir::stablehlo::ReduceWindowOp op,
                                        llvm::ArrayRef<int64_t> windowDims,
                                        llvm::ArrayRef<int64_t> windowStrides,
                                        llvm::ArrayRef<int64_t> baseDilations,
                                        llvm::ArrayRef<int64_t> windowDilations,
                                        const std::vector<std::pair<int64_t, int64_t>>& padding,
                                        const std::string& reducerType) {
    std::ostringstream oss;
    oss << "reducer=" << reducerType << ", window_dimensions=" << JoinInts(windowDims)
        << ", window_strides=" << JoinInts(windowStrides) << ", base_dilations=" << JoinInts(baseDilations)
        << ", window_dilations=" << JoinInts(windowDilations) << ", padding=" << JoinPadding(padding)
        << ", input_shape=";
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getInputs().front().getType());
    if (!inputType) {
        oss << "<unranked>";
    } else {
        oss << JoinInts(inputType.getShape());
    }
    return oss.str();
}

bool GetScalarConstantAttr(mlir::Value value, mlir::DenseElementsAttr& outDense) {
    outDense = {};
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
    if (denseAttr.getNumElements() != 1) {
        return false;
    }
    outDense = denseAttr;
    return true;
}

bool IsScalarZero(mlir::Value value) {
    mlir::DenseElementsAttr dense;
    if (!GetScalarConstantAttr(value, dense)) {
        return false;
    }
    auto elemType = dense.getElementType();
    if (mlir::isa<mlir::FloatType>(elemType)) {
        llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
        return ap.isZero();
    }
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        llvm::APInt ap = *dense.getValues<llvm::APInt>().begin();
        return ap.isZero();
    }
    return false;
}

} // namespace

static ProcessResult HandleSingleResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("reduce: input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    MPSGraphTensor* result = nullptr;
    if (reductionType == "stablehlo.add") {
        result = [ctx.graph reductionSumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        result = [ctx.graph reductionProductWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        result = [ctx.graph reductionMaximumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        result = [ctx.graph reductionMinimumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.and") {
        result = [ctx.graph reductionAndWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.or") {
        result = [ctx.graph reductionOrWithTensor:input axes:axes name:nil];
    } else {
        return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && result && result.shape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }

    return Result(ctx, result, "reduce");
}

static ProcessResult HandleMultiResultReduce(HandlerContext& ctx) {
    auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(ctx.op);
    if (!reduceOp) {
        return ProcessResult::Error("reduce: expected ReduceOp");
    }
    if (ctx.op->getNumResults() != 2 || ctx.op->getNumOperands() < 2) {
        return ProcessResult::Error("reduce: unsupported multi-result shape");
    }

    MPSGraphTensor* valueInput = GetInputTensor(ctx, 0);
    if (!valueInput) {
        return ProcessResult::Error("reduce: value input tensor not found");
    }

    auto dimensions = reduceOp.getDimensions();
    if (dimensions.size() != 1) {
        return ProcessResult::Error("reduce: only single-axis multi-result reduce is supported");
    }
    NSInteger axis = (NSInteger)dimensions[0];

    ArgReduceKind kind = detectArgReduceKind(reduceOp);
    if (kind == ArgReduceKind::kUnknown) {
        return ProcessResult::Error("reduce: unsupported multi-result reduce body");
    }

    MPSGraphTensor* valueOut = nullptr;
    MPSGraphTensor* indexOut = nullptr;
    if (kind == ArgReduceKind::kMax) {
        valueOut = [ctx.graph reductionMaximumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMaximumWithTensor:valueInput axis:axis name:nil];
    } else {
        valueOut = [ctx.graph reductionMinimumWithTensor:valueInput axis:axis name:nil];
        indexOut = [ctx.graph reductionArgMinimumWithTensor:valueInput axis:axis name:nil];
    }
    if (!valueOut || !indexOut) {
        return ProcessResult::Error("reduce: failed to lower multi-result reduce");
    }

    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [ctx.graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [ctx.graph castTensor:indexOut toType:indexType name:nil];
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(ctx.op, 0);
    if (valueShape && valueOut) {
        valueOut = [ctx.graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(ctx.op, 1);
    if (indexShape && indexOut) {
        indexOut = [ctx.graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = valueOut;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

static ProcessResult HandleReduce(HandlerContext& ctx) {
    if (ctx.op->getNumResults() > 1) {
        return HandleMultiResultReduce(ctx);
    }
    return HandleSingleResultReduce(ctx);
}
REGISTER_MPS_OP("stablehlo.reduce", HandleReduce);

static ProcessResult HandleReduceWindow(HandlerContext& ctx) {
    auto reduceWindowOp = mlir::dyn_cast<mlir::stablehlo::ReduceWindowOp>(ctx.op);
    if (!reduceWindowOp) {
        return ProcessResult::Error("reduce_window: expected ReduceWindowOp");
    }
    if (ctx.op->getNumOperands() != 2 || ctx.op->getNumResults() != 1) {
        return ProcessResult::Error(
            "reduce_window: only single-input single-result reduce_window is supported");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("reduce_window: input tensor not found");
    }

    auto windowDimensions = reduceWindowOp.getWindowDimensions();
    size_t rank = windowDimensions.size();
    if (rank == 0) {
        return ProcessResult::Error("reduce_window: rank-0 reduce_window is not supported");
    }

    std::vector<int64_t> windowStrides;
    std::vector<int64_t> baseDilations;
    std::vector<int64_t> windowDilations;
    std::vector<std::pair<int64_t, int64_t>> padding;
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getWindowStrides(), rank, windowStrides)) {
        return ProcessResult::Error("reduce_window: invalid window_strides rank mismatch");
    }
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getBaseDilations(), rank, baseDilations)) {
        return ProcessResult::Error("reduce_window: invalid base_dilations rank mismatch");
    }
    if (!ReadArrayOrDefaultOnes(reduceWindowOp.getWindowDilations(), rank, windowDilations)) {
        return ProcessResult::Error("reduce_window: invalid window_dilations rank mismatch");
    }
    if (!ReadPaddingOrDefaultZeros(reduceWindowOp.getPadding(), rank, padding)) {
        return ProcessResult::Error("reduce_window: invalid padding shape");
    }

    for (int64_t d : baseDilations) {
        if (d != 1) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error(
                "reduce_window: base_dilations != 1 is unsupported (" +
                DescribeReduceWindowPattern(reduceWindowOp, windowDimensions, windowStrides, baseDilations,
                                            windowDilations, padding,
                                            GetReductionOpType(reduceWindowOp.getBody())) +
                ")");
#else
            return ProcessResult::Error("reduce_window: base_dilations != 1 is unsupported");
#endif
        }
    }

    std::string reducerType = GetReductionOpType(reduceWindowOp.getBody());
    NSInteger cumulativeAxis = -1;
    bool patternMatches = true;
    for (size_t dim = 0; dim < rank; ++dim) {
        if (windowStrides[dim] != 1 || baseDilations[dim] != 1 || windowDilations[dim] != 1) {
            patternMatches = false;
            break;
        }

        int64_t window = windowDimensions[dim];
        int64_t padLow = padding[dim].first;
        int64_t padHigh = padding[dim].second;
        if (window > 1) {
            if (cumulativeAxis != -1 || padLow != window - 1 || padHigh != 0) {
                patternMatches = false;
                break;
            }
            cumulativeAxis = (NSInteger)dim;
        } else if (window == 1) {
            if (padLow != 0 || padHigh != 0) {
                patternMatches = false;
                break;
            }
        } else {
            patternMatches = false;
            break;
        }
    }

    if (cumulativeAxis < 0 || !patternMatches) {

        auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
        bool is2DPooling =
            (rank == 4 && inputType && inputType.getRank() == 4 && windowDimensions.size() == 4 &&
             windowStrides.size() == 4 && baseDilations.size() == 4 && windowDilations.size() == 4 &&
             windowDimensions[0] == 1 && windowDimensions[3] == 1 && windowStrides[0] == 1 &&
             windowStrides[3] == 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             baseDilations[2] == 1 && baseDilations[3] == 1 && windowDilations[0] == 1 &&
             windowDilations[3] == 1 && windowDilations[1] >= 1 && windowDilations[2] >= 1 &&
             padding[0].first == 0 && padding[0].second == 0 && padding[3].first == 0 &&
             padding[3].second == 0 && padding[1].first >= 0 && padding[1].second >= 0 &&
             padding[2].first >= 0 && padding[2].second >= 0 && windowDimensions[1] >= 1 &&
             windowDimensions[2] >= 1 && windowStrides[1] >= 1 && windowStrides[2] >= 1);

        bool is1DPooling =
            (rank == 3 && inputType && inputType.getRank() == 3 && windowDimensions.size() == 3 &&
             windowStrides.size() == 3 && baseDilations.size() == 3 && windowDilations.size() == 3 &&
             windowDimensions[0] == 1 && windowDimensions[2] == 1 && windowStrides[0] == 1 &&
             windowStrides[2] == 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             baseDilations[2] == 1 && windowDilations[0] == 1 && windowDilations[2] == 1 &&
             windowDilations[1] >= 1 && padding[0].first == 0 && padding[0].second == 0 &&
             padding[2].first == 0 && padding[2].second == 0 && padding[1].first >= 0 &&
             padding[1].second >= 0 && windowDimensions[1] >= 1 && windowStrides[1] >= 1);

        if (is2DPooling || is1DPooling) {
            int64_t kH = is2DPooling ? windowDimensions[1] : 1;
            int64_t kW = is2DPooling ? windowDimensions[2] : windowDimensions[1];
            int64_t sH = is2DPooling ? windowStrides[1] : 1;
            int64_t sW = is2DPooling ? windowStrides[2] : windowStrides[1];
            int64_t dH = is2DPooling ? windowDilations[1] : 1;
            int64_t dW = is2DPooling ? windowDilations[2] : windowDilations[1];
            int64_t padTop = is2DPooling ? padding[1].first : 0;
            int64_t padBottom = is2DPooling ? padding[1].second : 0;
            int64_t padLeft = is2DPooling ? padding[2].first : padding[1].first;
            int64_t padRight = is2DPooling ? padding[2].second : padding[1].second;

            if (is1DPooling) {
                if (!input.shape || input.shape.count != 3) {
                    return ProcessResult::Error("reduce_window: expected rank-3 NWC tensor for 1D pooling");
                }
                input = [ctx.graph reshapeTensor:input
                                       withShape:@[input.shape[0], @1, input.shape[1], input.shape[2]]
                                            name:nil];
            }

            MPSGraphPooling2DOpDescriptor* desc =
                [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kW
                                                            kernelHeight:(NSUInteger)kH
                                                               strideInX:(NSUInteger)sW
                                                               strideInY:(NSUInteger)sH
                                                         dilationRateInX:(NSUInteger)dW
                                                         dilationRateInY:(NSUInteger)dH
                                                             paddingLeft:(NSUInteger)padLeft
                                                            paddingRight:(NSUInteger)padRight
                                                              paddingTop:(NSUInteger)padTop
                                                           paddingBottom:(NSUInteger)padBottom
                                                            paddingStyle:MPSGraphPaddingStyleExplicit
                                                              dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

            if (reducerType == "stablehlo.add") {

                if (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32) {
                    return ProcessResult::Error(
                        "reduce_window: sum-pooling only supported for float16/float32 (" +
                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions, windowStrides,
                                                    baseDilations, windowDilations, padding, reducerType) +
                        ")");
                }
                desc.includeZeroPadToAverage = YES;
                MPSGraphTensor* avg = [ctx.graph avgPooling2DWithSourceTensor:input descriptor:desc name:nil];
                MPSGraphTensor* scale = [ctx.graph constantWithScalar:(double)(kH * kW)
                                                             dataType:input.dataType];
                MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:avg
                                                                    secondaryTensor:scale
                                                                               name:nil];
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }

            if (reducerType == "stablehlo.maximum") {
                MPSGraphTensor* result = [ctx.graph maxPooling2DWithSourceTensor:input
                                                                      descriptor:desc
                                                                            name:nil];
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }

            if (reducerType == "stablehlo.minimum") {

                if (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32) {
                    return ProcessResult::Error(
                        "reduce_window: min-pooling only supported for float16/float32 (" +
                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions, windowStrides,
                                                    baseDilations, windowDilations, padding, reducerType) +
                        ")");
                }
                MPSGraphTensor* neg = [ctx.graph negativeWithTensor:input name:nil];
                MPSGraphTensor* pooled = [ctx.graph maxPooling2DWithSourceTensor:neg
                                                                      descriptor:desc
                                                                            name:nil];
                MPSGraphTensor* result = [ctx.graph negativeWithTensor:pooled name:nil];
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }
        }

        return ProcessResult::Error("reduce_window: unsupported pattern (" +
                                    DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                windowStrides, baseDilations, windowDilations,
                                                                padding, reducerType) +
                                    ")");
    }

    MPSGraphTensor* result = nullptr;
    if (reducerType == "stablehlo.add") {
        result = [ctx.graph cumulativeSumWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.multiply") {
        result = [ctx.graph cumulativeProductWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.maximum") {
        result = [ctx.graph cumulativeMaximumWithTensor:input axis:cumulativeAxis name:nil];
    } else if (reducerType == "stablehlo.minimum") {
        result = [ctx.graph cumulativeMinimumWithTensor:input axis:cumulativeAxis name:nil];
    } else {
        return ProcessResult::Error("reduce_window: unsupported reducer for cumulative lowering (" +
                                    DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                windowStrides, baseDilations, windowDilations,
                                                                padding, reducerType) +
                                    ")");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }
    return Result(ctx, result, "reduce_window");
}
REGISTER_MPS_OP("stablehlo.reduce_window", HandleReduceWindow);

static ProcessResult HandleSelectAndScatter(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::SelectAndScatterOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("select_and_scatter: expected SelectAndScatterOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* source = GetInputTensor(ctx, 1);
    if (!operand || !source) {
        return ProcessResult::Error("select_and_scatter: missing input tensors");
    }

    enum class SelectMode { kUnknown, kMax, kMin };
    SelectMode selectMode = SelectMode::kUnknown;
    if (!op.getSelect().empty()) {
        for (mlir::Operation& nestedOp : op.getSelect().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            auto dir = compareOp.getComparisonDirection();
            if (dir == mlir::stablehlo::ComparisonDirection::GT ||
                dir == mlir::stablehlo::ComparisonDirection::GE) {
                selectMode = SelectMode::kMax;
            } else if (dir == mlir::stablehlo::ComparisonDirection::LT ||
                       dir == mlir::stablehlo::ComparisonDirection::LE) {
                selectMode = SelectMode::kMin;
            }
            break;
        }
    }

    std::string scatterType = GetReductionOpType(op.getScatter());
    mlir::Value initValue = ctx.op->getOperand(2);

    if (selectMode == SelectMode::kUnknown || scatterType != "stablehlo.add" || !IsScalarZero(initValue)) {
#if MPS_LOG_LEVEL >= 3
        return ProcessResult::Error(
            "select_and_scatter: unsupported region bodies (expected select=GT/GE or LT/LE, "
            "scatter=add, init=0)");
#else
        return ProcessResult::Error("select_and_scatter: unsupported region bodies");
#endif
    }

    auto windowDimensionsOpt = op.getWindowDimensions();
    auto windowStridesOpt = op.getWindowStrides();
    if (!windowDimensionsOpt || !windowStridesOpt) {
        return ProcessResult::Error("select_and_scatter: missing window_dimensions/window_strides");
    }
    auto windowDimensions = *windowDimensionsOpt;
    auto windowStrides = *windowStridesOpt;
    size_t rank = windowDimensions.size();
    bool isNHWC = (rank == 4 && windowStrides.size() == 4);
    bool isNWC = (rank == 3 && windowStrides.size() == 3);
    if (!isNHWC && !isNWC) {
        return ProcessResult::Error("select_and_scatter: only rank-4 NHWC or rank-3 NWC is supported");
    }

    std::vector<std::pair<int64_t, int64_t>> padding;
    std::optional<mlir::DenseIntElementsAttr> paddingAttr;
    if (auto attr = ctx.op->getAttrOfType<mlir::DenseIntElementsAttr>("padding")) {
        paddingAttr = attr;
    }
    if (!ReadPaddingOrDefaultZeros(paddingAttr, rank, padding)) {
        return ProcessResult::Error("select_and_scatter: invalid padding attribute");
    }

    int64_t kH = 1, kW = 1;
    int64_t sH = 1, sW = 1;
    int64_t padTop = 0, padBottom = 0, padLeft = 0, padRight = 0;

    if (isNHWC) {

        if (windowDimensions[0] != 1 || windowDimensions[3] != 1 || windowStrides[0] != 1 ||
            windowStrides[3] != 1 || padding[0].first != 0 || padding[0].second != 0 ||
            padding[3].first != 0 || padding[3].second != 0 || padding[1].first < 0 ||
            padding[1].second < 0 || padding[2].first < 0 || padding[2].second < 0) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("select_and_scatter: unsupported NHWC pooling parameters");
#else
            return ProcessResult::Error("select_and_scatter: unsupported pooling parameters");
#endif
        }

        kH = windowDimensions[1];
        kW = windowDimensions[2];
        sH = windowStrides[1];
        sW = windowStrides[2];
        padTop = padding[1].first;
        padBottom = padding[1].second;
        padLeft = padding[2].first;
        padRight = padding[2].second;
    } else {

        if (windowDimensions[0] != 1 || windowDimensions[2] != 1 || windowStrides[0] != 1 ||
            windowStrides[2] != 1 || padding[0].first != 0 || padding[0].second != 0 ||
            padding[2].first != 0 || padding[2].second != 0 || padding[1].first < 0 ||
            padding[1].second < 0) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("select_and_scatter: unsupported NWC pooling parameters");
#else
            return ProcessResult::Error("select_and_scatter: unsupported pooling parameters");
#endif
        }

        if (!operand.shape || operand.shape.count != 3 || !source.shape || source.shape.count != 3) {
            return ProcessResult::Error("select_and_scatter: expected rank-3 NWC tensors");
        }
        operand = [ctx.graph reshapeTensor:operand
                                 withShape:@[operand.shape[0], @1, operand.shape[1], operand.shape[2]]
                                      name:nil];
        source = [ctx.graph reshapeTensor:source
                                withShape:@[source.shape[0], @1, source.shape[1], source.shape[2]]
                                     name:nil];

        kH = 1;
        kW = windowDimensions[1];
        sH = 1;
        sW = windowStrides[1];
        padTop = 0;
        padBottom = 0;
        padLeft = padding[1].first;
        padRight = padding[1].second;
    }

    MPSGraphPooling2DOpDescriptor* desc =
        [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:(NSUInteger)kW
                                                    kernelHeight:(NSUInteger)kH
                                                       strideInX:(NSUInteger)sW
                                                       strideInY:(NSUInteger)sH
                                                 dilationRateInX:1
                                                 dilationRateInY:1
                                                     paddingLeft:(NSUInteger)padLeft
                                                    paddingRight:(NSUInteger)padRight
                                                      paddingTop:(NSUInteger)padTop
                                                   paddingBottom:(NSUInteger)padBottom
                                                    paddingStyle:MPSGraphPaddingStyleExplicit
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNHWC];

    MPSGraphTensor* pooledOperand = operand;
    if (selectMode == SelectMode::kMin) {
        pooledOperand = [ctx.graph negativeWithTensor:operand name:nil];
    }

    MPSGraphTensor* result = [ctx.graph maxPooling2DGradientWithGradientTensor:source
                                                                  sourceTensor:pooledOperand
                                                                    descriptor:desc
                                                                          name:nil];
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
    }
    return Result(ctx, result, "select_and_scatter");
}
REGISTER_MPS_OP("stablehlo.select_and_scatter", HandleSelectAndScatter);

static ProcessResult HandleReturn(HandlerContext& ctx) {

    MPS_LOG_DEBUG("stablehlo.return terminator encountered; skipping execution\n");
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.return", HandleReturn);

static ProcessResult HandleBatchNormInference(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormInferenceOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_inference: expected BatchNormInferenceOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* offset = GetInputTensor(ctx, 2);
    MPSGraphTensor* mean = GetInputTensor(ctx, 3);
    MPSGraphTensor* variance = GetInputTensor(ctx, 4);
    if (!operand || !scale || !offset || !mean || !variance)
        return ProcessResult::Error("batch_norm_inference: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;
    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    scale = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];
    offset = [ctx.graph reshapeTensor:offset withShape:bcast_shape name:nil];
    mean = [ctx.graph reshapeTensor:mean withShape:bcast_shape name:nil];
    variance = [ctx.graph reshapeTensor:variance withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:variance secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand secondaryTensor:mean name:nil];
    MPSGraphTensor* normalized = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                            secondaryTensor:inv_std
                                                                       name:nil];
    MPSGraphTensor* scaled = [ctx.graph multiplicationWithPrimaryTensor:normalized
                                                        secondaryTensor:scale
                                                                   name:nil];
    MPSGraphTensor* result = [ctx.graph additionWithPrimaryTensor:scaled secondaryTensor:offset name:nil];

    return Result(ctx, result, "batch_norm_inference");
}
REGISTER_MPS_OP("stablehlo.batch_norm_inference", HandleBatchNormInference);

static ProcessResult HandleBatchNormTraining(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormTrainingOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_training: expected BatchNormTrainingOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* offset = GetInputTensor(ctx, 2);
    if (!operand || !scale || !offset)
        return ProcessResult::Error("batch_norm_training: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;

    NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            [reduce_axes addObject:@(i)];
        }
    }

    MPSGraphTensor* batch_mean = [ctx.graph meanOfTensor:operand axes:reduce_axes name:nil];
    MPSGraphTensor* batch_var = [ctx.graph varianceOfTensor:operand axes:reduce_axes name:nil];

    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    MPSGraphTensor* mean_bcast = [ctx.graph reshapeTensor:batch_mean withShape:bcast_shape name:nil];
    MPSGraphTensor* var_bcast = [ctx.graph reshapeTensor:batch_var withShape:bcast_shape name:nil];
    MPSGraphTensor* scale_bcast = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];
    MPSGraphTensor* offset_bcast = [ctx.graph reshapeTensor:offset withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:var_bcast secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand
                                                       secondaryTensor:mean_bcast
                                                                  name:nil];
    MPSGraphTensor* normalized = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                            secondaryTensor:inv_std
                                                                       name:nil];
    MPSGraphTensor* scaled = [ctx.graph multiplicationWithPrimaryTensor:normalized
                                                        secondaryTensor:scale_bcast
                                                                   name:nil];
    MPSGraphTensor* output = [ctx.graph additionWithPrimaryTensor:scaled
                                                  secondaryTensor:offset_bcast
                                                             name:nil];

    batch_mean = [ctx.graph reshapeTensor:batch_mean
                                withShape:@[operand.shape[(NSUInteger)feature_index]]
                                     name:nil];
    batch_var = [ctx.graph reshapeTensor:batch_var
                               withShape:@[operand.shape[(NSUInteger)feature_index]]
                                    name:nil];

    SetOutputTensor(ctx.values, ctx.op, output, 0);
    SetOutputTensor(ctx.values, ctx.op, batch_mean, 1);
    SetOutputTensor(ctx.values, ctx.op, batch_var, 2);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.batch_norm_training", HandleBatchNormTraining);

static ProcessResult HandleBatchNormGrad(HandlerContext& ctx) {
    auto bnOp = mlir::dyn_cast<mlir::stablehlo::BatchNormGradOp>(ctx.op);
    if (!bnOp)
        return ProcessResult::Error("batch_norm_grad: expected BatchNormGradOp");

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* scale = GetInputTensor(ctx, 1);
    MPSGraphTensor* mean = GetInputTensor(ctx, 2);
    MPSGraphTensor* variance = GetInputTensor(ctx, 3);
    MPSGraphTensor* grad_output = GetInputTensor(ctx, 4);
    if (!operand || !scale || !mean || !variance || !grad_output)
        return ProcessResult::Error("batch_norm_grad: missing input tensor");

    float epsilon = bnOp.getEpsilon().convertToFloat();
    int64_t feature_index = bnOp.getFeatureIndex();

    NSUInteger rank = operand.shape.count;

    NSMutableArray<NSNumber*>* reduce_axes = [NSMutableArray array];
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            [reduce_axes addObject:@(i)];
        }
    }

    double N = 1.0;
    for (NSUInteger i = 0; i < rank; i++) {
        if (i != (NSUInteger)feature_index) {
            N *= [operand.shape[i] longLongValue];
        }
    }

    NSMutableArray<NSNumber*>* bcast_shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger i = 0; i < rank; i++) {
        [bcast_shape addObject:(i == (NSUInteger)feature_index ? operand.shape[i] : @1)];
    }

    MPSGraphTensor* mean_bcast = [ctx.graph reshapeTensor:mean withShape:bcast_shape name:nil];
    MPSGraphTensor* var_bcast = [ctx.graph reshapeTensor:variance withShape:bcast_shape name:nil];
    MPSGraphTensor* scale_bcast = [ctx.graph reshapeTensor:scale withShape:bcast_shape name:nil];

    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsilon dataType:operand.dataType];
    MPSGraphTensor* var_eps = [ctx.graph additionWithPrimaryTensor:var_bcast secondaryTensor:eps name:nil];
    MPSGraphTensor* inv_std = [ctx.graph reciprocalSquareRootWithTensor:var_eps name:nil];
    MPSGraphTensor* centered = [ctx.graph subtractionWithPrimaryTensor:operand
                                                       secondaryTensor:mean_bcast
                                                                  name:nil];
    MPSGraphTensor* x_hat = [ctx.graph multiplicationWithPrimaryTensor:centered
                                                       secondaryTensor:inv_std
                                                                  name:nil];

    MPSGraphTensor* grad_offset = [ctx.graph reductionSumWithTensor:grad_output axes:reduce_axes name:nil];
    grad_offset = [ctx.graph reshapeTensor:grad_offset
                                 withShape:@[operand.shape[(NSUInteger)feature_index]]
                                      name:nil];

    MPSGraphTensor* grad_x_hat = [ctx.graph multiplicationWithPrimaryTensor:grad_output
                                                            secondaryTensor:x_hat
                                                                       name:nil];
    MPSGraphTensor* grad_scale = [ctx.graph reductionSumWithTensor:grad_x_hat axes:reduce_axes name:nil];
    grad_scale = [ctx.graph reshapeTensor:grad_scale
                                withShape:@[operand.shape[(NSUInteger)feature_index]]
                                     name:nil];

    MPSGraphTensor* n_const = [ctx.graph constantWithScalar:N dataType:operand.dataType];
    MPSGraphTensor* grad_scale_bcast = [ctx.graph reshapeTensor:grad_scale withShape:bcast_shape name:nil];
    MPSGraphTensor* grad_offset_bcast = [ctx.graph reshapeTensor:grad_offset withShape:bcast_shape name:nil];

    MPSGraphTensor* term1 = [ctx.graph multiplicationWithPrimaryTensor:x_hat
                                                       secondaryTensor:grad_scale_bcast
                                                                  name:nil];
    MPSGraphTensor* term2 = [ctx.graph additionWithPrimaryTensor:term1
                                                 secondaryTensor:grad_offset_bcast
                                                            name:nil];
    MPSGraphTensor* term3 = [ctx.graph divisionWithPrimaryTensor:term2 secondaryTensor:n_const name:nil];
    MPSGraphTensor* grad_normed = [ctx.graph subtractionWithPrimaryTensor:grad_output
                                                          secondaryTensor:term3
                                                                     name:nil];
    MPSGraphTensor* grad_operand = [ctx.graph multiplicationWithPrimaryTensor:grad_normed
                                                              secondaryTensor:scale_bcast
                                                                         name:nil];
    grad_operand = [ctx.graph multiplicationWithPrimaryTensor:grad_operand secondaryTensor:inv_std name:nil];

    SetOutputTensor(ctx.values, ctx.op, grad_operand, 0);
    SetOutputTensor(ctx.values, ctx.op, grad_scale, 1);
    SetOutputTensor(ctx.values, ctx.op, grad_offset, 2);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.batch_norm_grad", HandleBatchNormGrad);

} // namespace jax_metallib
