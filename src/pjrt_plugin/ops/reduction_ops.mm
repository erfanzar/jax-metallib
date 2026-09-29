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

#include <algorithm>
#include <cmath>
#include <cstdlib>
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
    // JAX often wraps a scalar init in a rank-0 broadcast_in_dim.
    if (auto broadcast = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(defOp)) {
        auto resultType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
        if (!resultType || resultType.getRank() != 0) {
            return false;
        }
        defOp = broadcast->getOperand(0).getDefiningOp();
    }
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

constexpr int64_t kTreeReduceThreshold = 1024;
constexpr int64_t kTreeReduceWindowSize = 256;

int64_t GetTotalElements(NSArray<NSNumber*>* shape) {
    if (!shape) return 0;
    int64_t total = 1;
    for (NSNumber* dim in shape) {
        total *= [dim longLongValue];
    }
    return total;
}

static bool IsFloatForNaN(MPSDataType t) {
    return t == MPSDataTypeFloat32 || t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
}

MPSGraphTensor* CreateReduction(MPSGraph* graph, MPSGraphTensor* input, NSArray<NSNumber*>* axes,
                                const std::string& reductionType) {
    if (reductionType == "stablehlo.add") {
        return [graph reductionSumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.multiply") {
        return [graph reductionProductWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.maximum") {
        // StableHLO max/min reductions propagate NaN; the plain MPSGraph
        // reductions returned the largest finite value instead.
        if (IsFloatForNaN(input.dataType)) {
            return [graph reductionMaximumPropagateNaNWithTensor:input axes:axes name:nil];
        }
        return [graph reductionMaximumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.minimum") {
        if (IsFloatForNaN(input.dataType)) {
            return [graph reductionMinimumPropagateNaNWithTensor:input axes:axes name:nil];
        }
        return [graph reductionMinimumWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.and") {
        return [graph reductionAndWithTensor:input axes:axes name:nil];
    } else if (reductionType == "stablehlo.or") {
        return [graph reductionOrWithTensor:input axes:axes name:nil];
    }
    return nil;
}

MPSGraphTensor* TreeReduce(HandlerContext& ctx, MPSGraphTensor* input,
                           NSArray<NSNumber*>* reduceAxes,
                           const std::string& reductionType) {
    int64_t totalElements = GetTotalElements(input.shape);
    if (totalElements <= kTreeReduceThreshold) {
        return nil;
    }

    NSArray<NSNumber*>* originalShape = input.shape;
    NSUInteger rank = originalShape.count;
    int64_t numReduceDims = (int64_t)reduceAxes.count;

    if (numReduceDims == 0) {
        return nil;
    }

    bool fullReduce = (numReduceDims == (int64_t)rank);

    if (fullReduce) {
        int64_t totalFlat = totalElements;
        int64_t numWindowElements = kTreeReduceWindowSize;
        int64_t numChunks = (totalFlat + numWindowElements - 1) / numWindowElements;

        NSMutableArray<NSNumber*>* flatShape = [NSMutableArray arrayWithObject:@(totalFlat)];
        MPSGraphTensor* flat = [ctx.graph reshapeTensor:input withShape:flatShape name:nil];

        NSMutableArray<MPSGraphTensor*>* levelResults = [NSMutableArray array];

        for (int64_t i = 0; i < numChunks; i++) {
            int64_t start = i * numWindowElements;
            int64_t end = start + numWindowElements;
            if (end > totalFlat) end = totalFlat;

            MPSGraphTensor* startTensor = [ctx.graph constantWithScalar:start dataType:MPSDataTypeInt64];
            startTensor = [ctx.graph reshapeTensor:startTensor withShape:@[ @1 ] name:nil];
            MPSGraphTensor* endTensor = [ctx.graph constantWithScalar:end dataType:MPSDataTypeInt64];
            endTensor = [ctx.graph reshapeTensor:endTensor withShape:@[ @1 ] name:nil];

            // The modern MPSGraph slice API is sliceTensor:tensor starts:ends:strides:name:
            // with NSArray<NSNumber*> args (the older 3-arg sliceTensor:start:end: selector
            // no longer exists on macOS 15+). The input `flat` is rank-1 here.
            MPSGraphTensor* sliced = [ctx.graph sliceTensor:flat
                                                     starts:@[ @(start) ]
                                                       ends:@[ @(end) ]
                                                    strides:@[ @1 ]
                                                       name:nil];

            MPSGraphTensor* reduced = CreateReduction(ctx.graph, sliced, @[ @0 ], reductionType);
            if (!reduced) return nil;

            [levelResults addObject:reduced];
        }

        MPSGraphTensor* current = nil;
        for (MPSGraphTensor* chunkResult in levelResults) {
            if (!current) {
                current = chunkResult;
            } else {
                // Modern API: concatTensors:dimension:name: (the second keyword is 'dimension', not 'axis').
                current = CreateReduction(ctx.graph,
                                          [ctx.graph concatTensors:@[ current, chunkResult ] dimension:0 name:nil],
                                          @[ @0 ], reductionType);
            }
        }

        if (!current) return nil;

        while (current.shape.count > 0 && [current.shape[0] longLongValue] > 1) {
            int64_t currentSize = [current.shape[0] longLongValue];
            int64_t nextSize = (currentSize + 1) / 2;

            // Modern API: sliceTensor:tensor starts:ends:strides:name: with NSArray args.
            MPSGraphTensor* left = [ctx.graph sliceTensor:current
                                                   starts:@[ @0 ]
                                                     ends:@[ @(nextSize) ]
                                                  strides:@[ @1 ]
                                                     name:nil];
            MPSGraphTensor* right = [ctx.graph sliceTensor:current
                                                    starts:@[ @(nextSize) ]
                                                      ends:@[ @(currentSize) ]
                                                   strides:@[ @1 ]
                                                      name:nil];

            if ([right.shape[0] longLongValue] < nextSize) {
                int64_t padding = nextSize - [right.shape[0] longLongValue];
                MPSGraphTensor* paddingTensor = [ctx.graph constantWithScalar:0 dataType:current.dataType];
                paddingTensor = [ctx.graph tileTensor:paddingTensor withValues:@[@(padding)] name:nil];
                // Modern API: concatTensors:dimension:name:
                right = [ctx.graph concatTensors:@[ right, paddingTensor ] dimension:0 name:nil];
            }

            // Modern API: concatTensors:dimension:name:
            current = CreateReduction(ctx.graph,
                                      [ctx.graph concatTensors:@[ left, right ] dimension:0 name:nil],
                                      @[ @0 ], reductionType);
        }

        if ([current.shape[0] longLongValue] == 1) {
            current = [ctx.graph reshapeTensor:current withShape:@[] name:nil];
        }

        return current;
    }

    return nil;
}

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
    std::string reductionType = GetReductionOpType(reduceOp.getBody());

    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    int64_t numReduceDims = (int64_t)dimensions.size();
    MPSGraphTensor* treeResult = TreeReduce(ctx, input, axes, reductionType);

    MPSGraphTensor* result = nullptr;
    if (treeResult) {
        result = treeResult;
    } else {
        if (reductionType == "stablehlo.add") {
            result = [ctx.graph reductionSumWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.multiply") {
            result = [ctx.graph reductionProductWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.maximum" || reductionType == "stablehlo.minimum") {
            result = CreateReduction(ctx.graph, input, axes, reductionType);
        } else if (reductionType == "stablehlo.and") {
            result = [ctx.graph reductionAndWithTensor:input axes:axes name:nil];
        } else if (reductionType == "stablehlo.or") {
            result = [ctx.graph reductionOrWithTensor:input axes:axes name:nil];
        } else {
            return ProcessResult::Error("reduce: unsupported reduction type: " + reductionType);
        }
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

    // MPSNDArrayReduction asserts (aborting the process) on uint32 arg-reductions.
    // XOR with the sign bit maps uint32 order onto int32 order; reduce there and
    // map the value back.
    const bool uint32Input = valueInput.dataType == MPSDataTypeUInt32;
    MPSGraphTensor* signBit = nil;
    if (uint32Input) {
        signBit = [ctx.graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
        valueInput = [ctx.graph reinterpretCastTensor:valueInput toType:MPSDataTypeInt32 name:nil];
        valueInput = [ctx.graph bitwiseXORWithPrimaryTensor:valueInput secondaryTensor:signBit name:nil];
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
    if (uint32Input) {
        valueOut = [ctx.graph bitwiseXORWithPrimaryTensor:valueOut secondaryTensor:signBit name:nil];
        valueOut = [ctx.graph reinterpretCastTensor:valueOut toType:MPSDataTypeUInt32 name:nil];
    }
    // MPSGraph arg-reductions returned -1 when every element of the slice equals
    // the reduction sentinel (e.g. argmin over an all-INT32_MAX column). That only
    // happens when all elements are equal, where the first index (0) is correct.
    MPSGraphTensor* zeroIndex = [ctx.graph constantWithScalar:0 dataType:indexOut.dataType];
    indexOut = [ctx.graph maximumWithPrimaryTensor:indexOut secondaryTensor:zeroIndex name:nil];

    // JAX argmax/argmin return the first NaN's index and a NaN value when the
    // slice contains NaN; MPSGraph arg-reductions ignored NaN entirely.
    if (IsFloatForNaN(valueInput.dataType)) {
        MPSGraphTensor* nanMask = [ctx.graph isNaNWithTensor:valueInput name:nil];
        MPSGraphTensor* nanInt = [ctx.graph castTensor:nanMask toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* nanIndex = [ctx.graph reductionArgMaximumWithTensor:nanInt axis:axis name:nil];
        MPSGraphTensor* anyNaN = [ctx.graph reductionMaximumWithTensor:nanInt axis:axis name:nil];
        MPSGraphTensor* zeroI = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
        anyNaN = [ctx.graph greaterThanWithPrimaryTensor:anyNaN secondaryTensor:zeroI name:nil];
        if (nanIndex.dataType != indexOut.dataType) {
            nanIndex = [ctx.graph castTensor:nanIndex toType:indexOut.dataType name:nil];
        }
        indexOut = [ctx.graph selectWithPredicateTensor:anyNaN
                                    truePredicateTensor:nanIndex
                                   falsePredicateTensor:indexOut
                                                   name:nil];
        MPSGraphTensor* nanValue = [ctx.graph constantWithScalar:NAN dataType:valueOut.dataType];
        valueOut = [ctx.graph selectWithPredicateTensor:anyNaN
                                    truePredicateTensor:nanValue
                                   falsePredicateTensor:valueOut
                                                   name:nil];
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

// MPSGraph max pooling returns the largest non-NaN value, so a NaN inside a
// window was dropped where XLA propagates it. Max-pool a NaN indicator with the
// same descriptor and put NaN back wherever any window element was NaN.
// isNaN(-x) == isNaN(x), so this also works on the negated min-pooling source.
static MPSGraphTensor* PropagatePoolingNaN(MPSGraph* g, MPSGraphTensor* pooledSource,
                                           MPSGraphPooling2DOpDescriptor* desc, MPSGraphTensor* pooled) {
    static const bool disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_POOLING_NAN_PROPAGATION");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    MPSDataType st = pooledSource.dataType;
    if (disabled || !pooled ||
        (st != MPSDataTypeFloat32 && st != MPSDataTypeFloat16 && st != MPSDataTypeBFloat16)) {
        return pooled;
    }
    // Costs one extra pooling pass (~0.39 ms on the sliding_max benches). Gating
    // it behind an MPSGraph `if` on any(isNaN) was measured at ~3.7 ms per call,
    // far worse, so the pass is unconditional.
    // JAX_METALLIB_DISABLE_POOLING_NAN_PROPAGATION=1 opts out.
    MPSGraphTensor* flag = [g castTensor:[g isNaNWithTensor:pooledSource name:nil]
                                  toType:MPSDataTypeFloat32
                                    name:nil];
    MPSGraphTensor* any = [g maxPooling2DWithSourceTensor:flag descriptor:desc name:nil];
    MPSGraphTensor* hasNaN = [g greaterThanWithPrimaryTensor:any
                                             secondaryTensor:[g constantWithScalar:0.0
                                                                          dataType:MPSDataTypeFloat32]
                                                        name:nil];
    MPSGraphTensor* nanT = [g constantWithScalar:NAN dataType:pooled.dataType];
    return [g selectWithPredicateTensor:hasNaN truePredicateTensor:nanT falsePredicateTensor:pooled name:nil];
}

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

    std::string reducerType = GetReductionOpType(reduceWindowOp.getBody());

    // base_dilation > 1 was rejected before any lowering ran. Materialise the
    // dilated operand instead (per axis: reshape to [n,1], concat k-1 filler
    // slots, reshape to n*k, trim to (n-1)*k+1) and continue with dilation 1.
    // Holes hold the reducer identity for add/multiply and init for max/min,
    // matching XLA:CPU (holes do not contribute; init is folded once).
    bool hasBaseDilation = false;
    for (int64_t d : baseDilations) {
        hasBaseDilation = hasBaseDilation || d != 1;
    }
    if (hasBaseDilation) {
        const bool isAdd = reducerType == "stablehlo.add";
        const bool isMul = reducerType == "stablehlo.multiply";
        const bool isExtrema = reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum";
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        bool supported = (isAdd || isMul || isExtrema) && input.shape && input.shape.count == rank &&
                         input.dataType != MPSDataTypeBool && (!isExtrema || init);
        for (size_t d = 0; supported && d < rank; ++d) {
            supported = baseDilations[d] >= 1 && input.shape[d].longLongValue >= 1;
        }
        if (!supported) {
#if MPS_LOG_LEVEL >= 3
            return ProcessResult::Error("reduce_window: base_dilations != 1 is unsupported (" +
                                        DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                    windowStrides, baseDilations,
                                                                    windowDilations, padding, reducerType) +
                                        ")");
#else
            return ProcessResult::Error("reduce_window: base_dilations != 1 is unsupported");
#endif
        }
        MPSGraphTensor* fill = nil;
        if (isExtrema) {
            fill = init.dataType == input.dataType
                       ? init
                       : [ctx.graph castTensor:init toType:input.dataType name:nil];
        } else {
            fill = [ctx.graph constantWithScalar:(isMul ? 1.0 : 0.0) dataType:input.dataType];
        }
        for (size_t d = 0; d < rank; ++d) {
            const int64_t k = baseDilations[d];
            if (k == 1) {
                continue;
            }
            NSArray<NSNumber*>* shape = input.shape;
            const int64_t n = shape[d].longLongValue;
            NSMutableArray<NSNumber*>* column = [shape mutableCopy];
            [column insertObject:@1 atIndex:d + 1];
            NSMutableArray<NSNumber*>* holes = [shape mutableCopy];
            [holes insertObject:@(k - 1) atIndex:d + 1];
            NSMutableArray<NSNumber*>* spread = [shape mutableCopy];
            spread[d] = @(n * k);
            MPSGraphTensor* col = [ctx.graph reshapeTensor:input withShape:column name:nil];
            MPSGraphTensor* gap = [ctx.graph broadcastTensor:fill toShape:holes name:nil];
            MPSGraphTensor* joined = [ctx.graph concatTensors:@[col, gap]
                                                    dimension:static_cast<NSInteger>(d + 1)
                                                         name:nil];
            MPSGraphTensor* flat = [ctx.graph reshapeTensor:joined withShape:spread name:nil];
            input = [ctx.graph sliceTensor:flat
                                 dimension:static_cast<NSUInteger>(d)
                                     start:0
                                    length:(n - 1) * k + 1
                                      name:nil];
        }
        std::fill(baseDilations.begin(), baseDilations.end(), 1);
    }

    // Integer max/min windows: the pooling and sliding paths below are float-only
    // (they pad with +/-inf), so integer extrema raised "unsupported pattern".
    // Lower exactly instead: pad with the init value (StableHLO pads with init),
    // take one strided/dilated slice per window position, and fold with max/min.
    auto integerExtremaWindow = [&]() -> MPSGraphTensor* {
        MPSDataType dt = input.dataType;
        const bool isInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8 || dt == MPSDataTypeInt16 ||
                           dt == MPSDataTypeUInt16 || dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
        const bool isMax = reducerType == "stablehlo.maximum";
        if (!isInt || (!isMax && reducerType != "stablehlo.minimum") || !input.shape ||
            input.shape.count != rank) {
            return nil;
        }
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return nil;
        }
        if (init.dataType != dt) {
            init = [ctx.graph castTensor:init toType:dt name:nil];
        }
        int64_t positions = 1;
        std::vector<int64_t> outDims(rank, 0);
        MPSGraphTensor* padded = input;
        for (size_t d = 0; d < rank; ++d) {
            if (windowDimensions[d] < 1 || windowStrides[d] < 1 || windowDilations[d] < 1 ||
                padding[d].first < 0 || padding[d].second < 0) {
                return nil;
            }
            positions *= windowDimensions[d];
            if (positions > 256) {
                return nil;
            }
            int64_t size = [input.shape[d] longLongValue] + padding[d].first + padding[d].second;
            int64_t effective = (windowDimensions[d] - 1) * windowDilations[d] + 1;
            if (size < effective) {
                return nil;
            }
            outDims[d] = (size - effective) / windowStrides[d] + 1;
            NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
            auto fill = [&](int64_t count) {
                NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:padded.shape];
                shape[d] = @(count);
                return [ctx.graph broadcastTensor:init toShape:shape name:nil];
            };
            if (padding[d].first > 0) {
                [parts addObject:fill(padding[d].first)];
            }
            [parts addObject:padded];
            if (padding[d].second > 0) {
                [parts addObject:fill(padding[d].second)];
            }
            if (parts.count > 1) {
                padded = [ctx.graph concatTensors:parts dimension:(NSInteger)d name:nil];
            }
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape) {
            if (expectedShape.count != rank) {
                return nil;
            }
            for (size_t d = 0; d < rank; ++d) {
                if ([expectedShape[d] longLongValue] != outDims[d]) {
                    return nil;
                }
            }
        }
        MPSGraphTensor* result = nil;
        for (int64_t p = 0; p < positions; ++p) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            std::vector<int64_t> offset(rank, 0);
            int64_t rem = p;
            for (size_t d = rank; d-- > 0;) {
                offset[d] = rem % windowDimensions[d];
                rem /= windowDimensions[d];
            }
            for (size_t d = 0; d < rank; ++d) {
                int64_t start = offset[d] * windowDilations[d];
                [starts addObject:@(start)];
                [ends addObject:@(start + (outDims[d] - 1) * windowStrides[d] + 1)];
                [strides addObject:@(windowStrides[d])];
            }
            MPSGraphTensor* slice = [ctx.graph sliceTensor:padded
                                                    starts:starts
                                                      ends:ends
                                                   strides:strides
                                                      name:nil];
            if (!result) {
                result = slice;
            } else if (isMax) {
                result = [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:slice name:nil];
            } else {
                result = [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:slice name:nil];
            }
        }
        if (!result) {
            return nil;
        }
        return isMax ? [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:init name:nil]
                     : [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:init name:nil];
    };
    if (MPSGraphTensor* extrema = integerExtremaWindow()) {
        return Result(ctx, extrema, "reduce_window");
    }

    // Pooling accumulates each window independently, avoiding float32 prefix
    // cancellation for sums and unbounded cumulative max/min for extrema.
    auto poolSmallRankWindow = [&](NSInteger axis, int64_t window, int64_t padLow,
                                   int64_t padHigh) -> MPSGraphTensor* {
        if (rank > 2 || !input.shape || input.shape.count != rank ||
            (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32)) {
            return nil;
        }
        NSArray<NSNumber*>* shape =
            rank == 1 ? @[@1, @1, input.shape[0], @1] : @[@1, input.shape[0], input.shape[1], @1];
        MPSGraphTensor* source = [ctx.graph reshapeTensor:input withShape:shape name:nil];
        bool height = rank == 2 && axis == 0;
        NSUInteger kWidth = (NSUInteger)(height ? 1 : window);
        NSUInteger kHeight = (NSUInteger)(height ? window : 1);
        NSUInteger left = (NSUInteger)(height ? 0 : padLow);
        NSUInteger right = (NSUInteger)(height ? 0 : padHigh);
        NSUInteger top = (NSUInteger)(height ? padLow : 0);
        NSUInteger bottom = (NSUInteger)(height ? padHigh : 0);
        if (input.dataType == MPSDataTypeFloat16 &&
            (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum")) {
            // Half-precision max pooling with right padding can shift windows
            // to the left on MPSGraph. Extrema are exact after widening.
            source = [ctx.graph castTensor:source toType:MPSDataTypeFloat32 name:nil];
        }
        MPSGraphPooling2DOpDescriptor* desc =
            [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:kWidth
                                                        kernelHeight:kHeight
                                                           strideInX:1
                                                           strideInY:1
                                                     dilationRateInX:1
                                                     dilationRateInY:1
                                                         paddingLeft:left
                                                        paddingRight:right
                                                          paddingTop:top
                                                       paddingBottom:bottom
                                                        paddingStyle:MPSGraphPaddingStyleExplicit
                                                          dataLayout:MPSGraphTensorNamedDataLayoutNHWC];
        if (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") {
            if (reducerType == "stablehlo.minimum") {
                source = [ctx.graph negativeWithTensor:source name:nil];
            }
            MPSGraphTensor* pooled = [ctx.graph maxPooling2DWithSourceTensor:source descriptor:desc name:nil];
            pooled = PropagatePoolingNaN(ctx.graph, source, desc, pooled);
            if (reducerType == "stablehlo.minimum") {
                pooled = [ctx.graph negativeWithTensor:pooled name:nil];
            }
            if (input.dataType == MPSDataTypeFloat16) {
                pooled = [ctx.graph castTensor:pooled toType:input.dataType name:nil];
            }
            MPSGraphTensor* result = [ctx.graph reshapeTensor:pooled
                                                    withShape:GetOutputShape(ctx.op)
                                                         name:nil];
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            return reducerType == "stablehlo.maximum"
                       ? [ctx.graph maximumWithNaNPropagationWithPrimaryTensor:result
                                                               secondaryTensor:init
                                                                          name:nil]
                       : [ctx.graph minimumWithNaNPropagationWithPrimaryTensor:result
                                                               secondaryTensor:init
                                                                          name:nil];
        }
        desc.includeZeroPadToAverage = YES;
        MPSGraphTensor* avg = [ctx.graph avgPooling2DWithSourceTensor:source descriptor:desc name:nil];
        MPSGraphTensor* scale = [ctx.graph constantWithScalar:(double)window dataType:input.dataType];
        MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:avg
                                                            secondaryTensor:scale
                                                                       name:nil];
        result = [ctx.graph reshapeTensor:result withShape:GetOutputShape(ctx.op) name:nil];
        if (!IsScalarZero(ctx.op->getOperand(1))) {
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
        }
        return result;
    };
    NSInteger cumulativeAxis = -1;
    bool reverseCumulative = false;
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
            bool forward = padLow == window - 1 && padHigh == 0;
            bool reverse = padLow == 0 && padHigh == window - 1;
            if (cumulativeAxis != -1 || (!forward && !reverse)) {
                patternMatches = false;
                break;
            }
            cumulativeAxis = (NSInteger)dim;
            reverseCumulative = reverse;
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

    // Sliding sums and extrema are not full-prefix reductions. Reuse the
    // NHWC/NWC pooling route for supported high-rank spatial axes.
    if (patternMatches && cumulativeAxis >= 0 &&
        (reducerType == "stablehlo.add" || reducerType == "stablehlo.maximum" ||
         reducerType == "stablehlo.minimum") &&
        rank >= 3 && input.shape && input.shape.count == rank &&
        ((rank == 3 && cumulativeAxis == 1) || (rank == 4 && (cumulativeAxis == 1 || cumulativeAxis == 2))) &&
        windowDimensions[(size_t)cumulativeAxis] < [input.shape[(NSUInteger)cumulativeAxis] longLongValue]) {
        patternMatches = false;
    }

    if (cumulativeAxis < 0 || !patternMatches) {
        // The transpose of a partial-window sum uses an unpadded VALID
        // reduce_window. Pool single-axis rank-1/2 sums directly; other
        // pool patterns are handled below.
        NSInteger validSumAxis = -1;
        bool validSum = rank <= 2 && reducerType == "stablehlo.add" && input.shape.count == rank &&
                        IsScalarZero(ctx.op->getOperand(1));
        for (size_t dim = 0; validSum && dim < rank; ++dim) {
            int64_t window = windowDimensions[dim];
            int64_t axisSize = [input.shape[dim] longLongValue];
            if (window < 1 || window > axisSize || windowStrides[dim] != 1 || baseDilations[dim] != 1 ||
                windowDilations[dim] != 1 || padding[dim].first != 0 || padding[dim].second != 0) {
                validSum = false;
            } else if (window > 1) {
                if (validSumAxis >= 0) {
                    validSum = false;
                } else {
                    validSumAxis = (NSInteger)dim;
                }
            }
        }
        if (validSum && validSumAxis >= 0) {
            int64_t window = windowDimensions[(size_t)validSumAxis];
            if (MPSGraphTensor* pooled = poolSmallRankWindow(validSumAxis, window, 0, 0)) {
                return Result(ctx, pooled, "reduce_window");
            }
            int64_t axisSize = [input.shape[(NSUInteger)validSumAxis] longLongValue];
            MPSGraphTensor* prefix = [ctx.graph cumulativeSumWithTensor:input axis:validSumAxis name:nil];
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* headEnds = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                bool axis = dim == (size_t)validSumAxis;
                [starts addObject:axis ? @(window - 1) : @0];
                [ends addObject:input.shape[dim]];
                [headEnds addObject:axis ? @(axisSize - window) : input.shape[dim]];
                [strides addObject:@1];
                [left addObject:axis ? @1 : @0];
                [right addObject:@0];
            }
            MPSGraphTensor* result = [ctx.graph sliceTensor:prefix
                                                     starts:starts
                                                       ends:ends
                                                    strides:strides
                                                       name:nil];
            if (axisSize > window) {
                for (size_t dim = 0; dim < rank; ++dim) {
                    starts[dim] = @0;
                }
                MPSGraphTensor* head = [ctx.graph sliceTensor:prefix
                                                       starts:starts
                                                         ends:headEnds
                                                      strides:strides
                                                         name:nil];
                MPSGraphTensor* shifted = [ctx.graph padTensor:head
                                               withPaddingMode:MPSGraphPaddingModeConstant
                                                   leftPadding:left
                                                  rightPadding:right
                                                 constantValue:0.0
                                                          name:nil];
                result = [ctx.graph subtractionWithPrimaryTensor:result secondaryTensor:shifted name:nil];
            }
            return Result(ctx, result, "reduce_window");
        }

        auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
        bool isSpatial2DPooling =
            (rank == 2 && inputType && inputType.getRank() == 2 && input.shape && input.shape.count == 2 &&
             (input.dataType == MPSDataTypeFloat16 || input.dataType == MPSDataTypeFloat32) &&
             (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum") &&
             windowDimensions[0] >= 1 && windowDimensions[1] >= 1 && windowStrides[0] >= 1 &&
             windowStrides[1] >= 1 && baseDilations[0] == 1 && baseDilations[1] == 1 &&
             windowDilations[0] == 1 && windowDilations[1] == 1 && padding[0].first >= 0 &&
             padding[0].second >= 0 && padding[1].first >= 0 && padding[1].second >= 0);
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

        if (is2DPooling || is1DPooling || isSpatial2DPooling) {
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return ProcessResult::Error("reduce_window: missing pooling init tensor");
            }
            int64_t kH = is2DPooling ? windowDimensions[1] : isSpatial2DPooling ? windowDimensions[0] : 1;
            int64_t kW = is2DPooling ? windowDimensions[2] : windowDimensions[1];
            int64_t sH = is2DPooling ? windowStrides[1] : isSpatial2DPooling ? windowStrides[0] : 1;
            int64_t sW = is2DPooling ? windowStrides[2] : windowStrides[1];
            int64_t dH = is2DPooling ? windowDilations[1] : 1;
            int64_t dW = is2DPooling ? windowDilations[2] : windowDilations[1];
            int64_t padTop = is2DPooling ? padding[1].first : isSpatial2DPooling ? padding[0].first : 0;
            int64_t padBottom = is2DPooling ? padding[1].second : isSpatial2DPooling ? padding[0].second : 0;
            int64_t padLeft = is2DPooling ? padding[2].first : padding[1].first;
            int64_t padRight = is2DPooling ? padding[2].second : padding[1].second;

            if (isSpatial2DPooling) {
                input = [ctx.graph reshapeTensor:input
                                       withShape:@[@1, input.shape[0], input.shape[1], @1]
                                            name:nil];
            }
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
                if (!IsScalarZero(ctx.op->getOperand(1))) {
                    result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
                }
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }

            if (isSpatial2DPooling && input.dataType == MPSDataTypeFloat16) {
                // Right-padded float16 max pooling can select shifted windows.
                input = [ctx.graph castTensor:input toType:MPSDataTypeFloat32 name:nil];
            }
            if (reducerType == "stablehlo.maximum") {
                MPSGraphTensor* result = [ctx.graph maxPooling2DWithSourceTensor:input
                                                                      descriptor:desc
                                                                            name:nil];
                result = PropagatePoolingNaN(ctx.graph, input, desc, result);
                if (isSpatial2DPooling && result.dataType != init.dataType) {
                    result = [ctx.graph castTensor:result toType:init.dataType name:nil];
                }
                result = [ctx.graph maximumWithNaNPropagationWithPrimaryTensor:result
                                                               secondaryTensor:init
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
                pooled = PropagatePoolingNaN(ctx.graph, neg, desc, pooled);
                MPSGraphTensor* result = [ctx.graph negativeWithTensor:pooled name:nil];
                if (isSpatial2DPooling && result.dataType != init.dataType) {
                    result = [ctx.graph castTensor:result toType:init.dataType name:nil];
                }
                result = [ctx.graph minimumWithNaNPropagationWithPrimaryTensor:result
                                                               secondaryTensor:init
                                                                          name:nil];
                NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
                if (result && outputShape && ![outputShape isEqualToArray:result.shape]) {
                    result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
                }
                return Result(ctx, result, "reduce_window");
            }
        }

        // General fallback for strided / dilated / multi-axis windows that no
        // path above handles (strided moving sums and products at any rank,
        // rank-1 strided max/min were rejected as "unsupported pattern").
        // StableHLO semantics exactly: pad with the init value, take one
        // strided/dilated slice per window position, fold with the reducer,
        // then fold in init once. 8-bit add/multiply widen to int32 (MPSGraph
        // saturates 8-bit arithmetic); f16/bf16 add/multiply accumulate in
        // f32; float max/min propagate NaN. Bounded to 256 window positions.
        auto generalWindowFold = [&]() -> MPSGraphTensor* {
            const bool isAdd = reducerType == "stablehlo.add";
            const bool isMul = reducerType == "stablehlo.multiply";
            const bool isMax = reducerType == "stablehlo.maximum";
            const bool isMin = reducerType == "stablehlo.minimum";
            if (!(isAdd || isMul || isMax || isMin) || !input.shape || input.shape.count != rank) {
                return nil;
            }
            const MPSDataType dt = input.dataType;
            const bool isHalf = dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
            const bool isFloat = dt == MPSDataTypeFloat32 || isHalf;
            const bool isNarrowInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8;
            const bool isInt = isNarrowInt || dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16 ||
                               dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
            if (!isFloat && !isInt) {
                return nil;
            }
            MPSGraphTensor* init = GetInputTensor(ctx, 1);
            if (!init) {
                return nil;
            }
            if (init.dataType != dt) {
                init = [ctx.graph castTensor:init toType:dt name:nil];
            }
            MPSDataType work = dt;
            if ((isAdd || isMul) && isHalf) {
                work = MPSDataTypeFloat32;
            } else if ((isAdd || isMul) && isNarrowInt) {
                work = MPSDataTypeInt32;
            }
            MPSGraphTensor* src = work == dt ? input : [ctx.graph castTensor:input toType:work name:nil];
            MPSGraphTensor* initW = work == dt ? init : [ctx.graph castTensor:init toType:work name:nil];

            // XLA:CPU skips padded positions (identity), then folds init once;
            // for max/min padding with init is equivalent after that fold.
            MPSGraphTensor* padValue =
                (isAdd || isMul) ? [ctx.graph constantWithScalar:(isAdd ? 0.0 : 1.0) dataType:work] : initW;
            int64_t positions = 1;
            std::vector<int64_t> outDims(rank, 0);
            MPSGraphTensor* padded = src;
            for (size_t d = 0; d < rank; ++d) {
                if (windowDimensions[d] < 1 || windowStrides[d] < 1 || windowDilations[d] < 1 ||
                    padding[d].first < 0 || padding[d].second < 0) {
                    return nil;
                }
                positions *= windowDimensions[d];
                if (positions > 256) {
                    return nil;
                }
                int64_t size = [src.shape[d] longLongValue] + padding[d].first + padding[d].second;
                int64_t effective = (windowDimensions[d] - 1) * windowDilations[d] + 1;
                if (size < effective) {
                    return nil;
                }
                outDims[d] = (size - effective) / windowStrides[d] + 1;
                NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
                auto fill = [&](int64_t count) {
                    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:padded.shape];
                    shape[d] = @(count);
                    return [ctx.graph broadcastTensor:padValue toShape:shape name:nil];
                };
                if (padding[d].first > 0) {
                    [parts addObject:fill(padding[d].first)];
                }
                [parts addObject:padded];
                if (padding[d].second > 0) {
                    [parts addObject:fill(padding[d].second)];
                }
                if (parts.count > 1) {
                    padded = [ctx.graph concatTensors:parts dimension:(NSInteger)d name:nil];
                }
            }
            NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
            if (expectedShape) {
                if (expectedShape.count != rank) {
                    return nil;
                }
                for (size_t d = 0; d < rank; ++d) {
                    if ([expectedShape[d] longLongValue] != outDims[d]) {
                        return nil;
                    }
                }
            }
            auto combine = [&](MPSGraphTensor* a, MPSGraphTensor* b) -> MPSGraphTensor* {
                if (isAdd) {
                    return [ctx.graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
                }
                if (isMul) {
                    return [ctx.graph multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
                }
                if (isMax) {
                    return isFloat ? [ctx.graph maximumWithNaNPropagationWithPrimaryTensor:a
                                                                           secondaryTensor:b
                                                                                      name:nil]
                                   : [ctx.graph maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
                }
                return isFloat ? [ctx.graph minimumWithNaNPropagationWithPrimaryTensor:a
                                                                       secondaryTensor:b
                                                                                  name:nil]
                               : [ctx.graph minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
            };
            MPSGraphTensor* result = nil;
            for (int64_t p = 0; p < positions; ++p) {
                NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
                NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
                std::vector<int64_t> offset(rank, 0);
                int64_t rem = p;
                for (size_t d = rank; d-- > 0;) {
                    offset[d] = rem % windowDimensions[d];
                    rem /= windowDimensions[d];
                }
                for (size_t d = 0; d < rank; ++d) {
                    int64_t start = offset[d] * windowDilations[d];
                    [starts addObject:@(start)];
                    [ends addObject:@(start + (outDims[d] - 1) * windowStrides[d] + 1)];
                    [strides addObject:@(windowStrides[d])];
                }
                MPSGraphTensor* slice = [ctx.graph sliceTensor:padded
                                                        starts:starts
                                                          ends:ends
                                                       strides:strides
                                                          name:nil];
                result = result ? combine(result, slice) : slice;
            }
            if (!result) {
                return nil;
            }
            result = combine(result, initW);
            if (work != dt) {
                result = [ctx.graph castTensor:result toType:dt name:nil];
            }
            return result;
        };
        if (MPSGraphTensor* folded = generalWindowFold()) {
            return Result(ctx, folded, "reduce_window");
        }

        return ProcessResult::Error("reduce_window: unsupported pattern (" +
                                    DescribeReduceWindowPattern(reduceWindowOp, windowDimensions,
                                                                windowStrides, baseDilations, windowDilations,
                                                                padding, reducerType) +
                                    ")");
    }

    int64_t window = windowDimensions[(size_t)cumulativeAxis];
    int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
    if (window < axisSize && (reducerType == "stablehlo.maximum" || reducerType == "stablehlo.minimum")) {
        if (MPSGraphTensor* pooled =
                poolSmallRankWindow(cumulativeAxis, window, padding[(size_t)cumulativeAxis].first,
                                    padding[(size_t)cumulativeAxis].second)) {
            return Result(ctx, pooled, "reduce_window");
        }
    }
    if (window < axisSize && (reducerType == "stablehlo.multiply" || reducerType == "stablehlo.maximum" ||
                              reducerType == "stablehlo.minimum")) {
        // A bounded window is not a cumulative reduction. Materialize the small
        // windows directly; max/min cannot be undone from a cumulative prefix,
        // and dividing cumulative products is incorrect when an input is zero.
        if (window > 32 || (input.dataType != MPSDataTypeFloat16 && input.dataType != MPSDataTypeFloat32)) {
            return ProcessResult::Error(
                "reduce_window: sliding multiply/max/min requires a float window <= 32");
        }
        double padValue = reducerType == "stablehlo.multiply"  ? 1.0
                          : reducerType == "stablehlo.maximum" ? -INFINITY
                                                               : INFINITY;
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
        for (size_t dim = 0; dim < rank; ++dim) {
            [left addObject:dim == (size_t)cumulativeAxis ? @(padding[dim].first) : @0];
            [right addObject:dim == (size_t)cumulativeAxis ? @(padding[dim].second) : @0];
        }
        MPSGraphTensor* padded = [ctx.graph padTensor:input
                                      withPaddingMode:MPSGraphPaddingModeConstant
                                          leftPadding:left
                                         rightPadding:right
                                        constantValue:padValue
                                                 name:nil];
        MPSGraphTensor* result = nil;
        for (int64_t offset = 0; offset < window; ++offset) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                [starts addObject:dim == (size_t)cumulativeAxis ? @(offset) : @0];
                [ends addObject:dim == (size_t)cumulativeAxis ? @(offset + axisSize) : input.shape[dim]];
                [strides addObject:@1];
            }
            MPSGraphTensor* next = [ctx.graph sliceTensor:padded
                                                   starts:starts
                                                     ends:ends
                                                  strides:strides
                                                     name:nil];
            if (!result) {
                result = next;
            } else if (reducerType == "stablehlo.multiply") {
                result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:next name:nil];
            } else if (reducerType == "stablehlo.maximum") {
                result = [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:next name:nil];
            } else {
                result = [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:next name:nil];
            }
        }
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return ProcessResult::Error("reduce_window: missing sliding init tensor");
        }
        if (reducerType == "stablehlo.multiply") {
            result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.maximum") {
            result = [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else {
            result = [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:init name:nil];
        }
        return Result(ctx, result, "reduce_window");
    }
    if (reducerType == "stablehlo.add") {
        int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
        if (window < axisSize) {
            if (MPSGraphTensor* pooled =
                    poolSmallRankWindow(cumulativeAxis, window, padding[(size_t)cumulativeAxis].first,
                                     padding[(size_t)cumulativeAxis].second)) {
                return Result(ctx, pooled, "reduce_window");
            }
        }
    }
    NSArray<NSNumber*>* axes = @[@(cumulativeAxis)];
    if (reverseCumulative) {
        input = [ctx.graph reverseTensor:input axes:axes name:nil];
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

    if (result && reducerType == "stablehlo.add") {
        // A window smaller than the axis is a sliding sum, not a full
        // cumulative sum: subtract the prefix that has left the window.
        int64_t window = windowDimensions[static_cast<size_t>(cumulativeAxis)];
        int64_t axisSize = [input.shape[(NSUInteger)cumulativeAxis] longLongValue];
        if (window < axisSize) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:rank];
            NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:rank];
            for (size_t dim = 0; dim < rank; ++dim) {
                [starts addObject:@0];
                [ends addObject:dim == (size_t)cumulativeAxis ? @(axisSize - window) : input.shape[dim]];
                [strides addObject:@1];
                [left addObject:dim == (size_t)cumulativeAxis ? @(window) : @0];
                [right addObject:@0];
            }
            MPSGraphTensor* prefix = [ctx.graph sliceTensor:result
                                                     starts:starts
                                                       ends:ends
                                                    strides:strides
                                                       name:nil];
            MPSGraphTensor* shifted = [ctx.graph padTensor:prefix
                                           withPaddingMode:MPSGraphPaddingModeConstant
                                               leftPadding:left
                                              rightPadding:right
                                             constantValue:0.0
                                                      name:nil];
            result = [ctx.graph subtractionWithPrimaryTensor:result secondaryTensor:shifted name:nil];
        }
    }
    if (result && reverseCumulative) {
        result = [ctx.graph reverseTensor:result axes:axes name:nil];
    }
    if (result && (reducerType != "stablehlo.add" || !IsScalarZero(ctx.op->getOperand(1)))) {
        MPSGraphTensor* init = GetInputTensor(ctx, 1);
        if (!init) {
            return ProcessResult::Error("reduce_window: missing init tensor");
        }
        if (reducerType == "stablehlo.add") {
            result = [ctx.graph additionWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.multiply") {
            result = [ctx.graph multiplicationWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.maximum") {
            result = [ctx.graph maximumWithPrimaryTensor:result secondaryTensor:init name:nil];
        } else if (reducerType == "stablehlo.minimum") {
            result = [ctx.graph minimumWithPrimaryTensor:result secondaryTensor:init name:nil];
        }
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

    // Read window_dimensions / window_strides from the Properties struct first
    // (the OpaqueProperties path in newer StableHLO pins stores attrs as
    // typed property fields, not in the NamedAttrList). Fall back to the
    // NamedAttrList if Properties didn't yield values — the textual-IR path
    // emits attrs in the dictionary form which we then need to read from
    // getAttrOfType.
    std::vector<int64_t> windowDimensions;
    std::vector<int64_t> windowStrides;
    // DenseI64ArrayAttr exposes its data via asArrayRef() (no STL iterators).
    if (auto attr = op.getWindowDimensionsAttr()) {
        auto ar = attr.asArrayRef();
        windowDimensions.assign(ar.begin(), ar.end());
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseI64ArrayAttr>("window_dimensions")) {
        auto ar = attr.asArrayRef();
        windowDimensions.assign(ar.begin(), ar.end());
    }
    if (auto attr = op.getWindowStridesAttr()) {
        auto ar = attr.asArrayRef();
        windowStrides.assign(ar.begin(), ar.end());
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseI64ArrayAttr>("window_strides")) {
        auto ar = attr.asArrayRef();
        windowStrides.assign(ar.begin(), ar.end());
    }
    if (windowDimensions.empty()) {
        return ProcessResult::Error("select_and_scatter: missing window_dimensions");
    }
    if (windowStrides.empty()) {
        // Default to all-ones strides (matches the StableHLO spec default).
        windowStrides.assign(windowDimensions.size(), 1);
    }
    size_t rank = windowDimensions.size();
    bool isNHWC = (rank == 4 && windowStrides.size() == 4);
    bool isNWC = (rank == 3 && windowStrides.size() == 3);
    bool isSpatialOnly = (rank == 1 || rank == 2) && windowStrides.size() == rank;
    if (!isNHWC && !isNWC && !isSpatialOnly) {
        return ProcessResult::Error(
            "select_and_scatter: expected rank-1/2 spatial, rank-3 NWC, or rank-4 NHWC");
    }

    std::vector<std::pair<int64_t, int64_t>> padding;
    std::optional<mlir::DenseIntElementsAttr> paddingAttr;
    // Same OpaqueProperties fallback as for window_dimensions above.
    if (auto attr = op.getPaddingAttr()) {
        paddingAttr = attr;
    } else if (auto attr = ctx.op->getAttrOfType<mlir::DenseIntElementsAttr>("padding")) {
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
    } else if (isNWC) {

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
    } else {
        if (!operand.shape || operand.shape.count != rank || !source.shape || source.shape.count != rank) {
            return ProcessResult::Error("select_and_scatter: rank-1/2 tensor shapes unavailable");
        }
        for (size_t dim = 0; dim < rank; ++dim) {
            if (windowDimensions[dim] <= 0 || windowStrides[dim] <= 0 || padding[dim].first < 0 ||
                padding[dim].second < 0) {
                return ProcessResult::Error("select_and_scatter: unsupported rank-1/2 pooling parameters");
            }
        }
        NSArray<NSNumber*>* operandShape =
            rank == 1 ? @[@1, @1, operand.shape[0], @1] : @[@1, operand.shape[0], operand.shape[1], @1];
        NSArray<NSNumber*>* sourceShape =
            rank == 1 ? @[@1, @1, source.shape[0], @1] : @[@1, source.shape[0], source.shape[1], @1];
        operand = [ctx.graph reshapeTensor:operand withShape:operandShape name:nil];
        source = [ctx.graph reshapeTensor:source withShape:sourceShape name:nil];
        size_t widthDim = rank - 1;
        kW = windowDimensions[widthDim];
        sW = windowStrides[widthDim];
        padLeft = padding[widthDim].first;
        padRight = padding[widthDim].second;
        if (rank == 2) {
            kH = windowDimensions[0];
            sH = windowStrides[0];
            padTop = padding[0].first;
            padBottom = padding[0].second;
        }
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
