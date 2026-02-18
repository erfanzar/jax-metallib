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

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include <algorithm>
#include <optional>

namespace jax_metallib {

namespace {

bool IsOrderingCompare(mlir::stablehlo::CompareOp compareOp) {
    auto dir = compareOp.getComparisonDirection();
    return dir == mlir::stablehlo::ComparisonDirection::LT ||
           dir == mlir::stablehlo::ComparisonDirection::LE ||
           dir == mlir::stablehlo::ComparisonDirection::GT || dir == mlir::stablehlo::ComparisonDirection::GE;
}

static void CollectRootBlockArgs(mlir::Value v, llvm::SmallSet<unsigned, 4>& out,
                                 llvm::SmallPtrSet<void*, 16>& visited, int depth = 0) {
    if (!v || depth > 16) {
        return;
    }

    void* key = v.getAsOpaquePointer();
    if (visited.count(key)) {
        return;
    }
    visited.insert(key);

    if (auto barg = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        out.insert(barg.getArgNumber());
        return;
    }

    mlir::Operation* def = v.getDefiningOp();
    if (!def) {
        return;
    }

    if (auto sel = mlir::dyn_cast<mlir::stablehlo::SelectOp>(def)) {
        CollectRootBlockArgs(sel.getOnTrue(), out, visited, depth + 1);
        CollectRootBlockArgs(sel.getOnFalse(), out, visited, depth + 1);
        return;
    }

    if (def->getNumOperands() == 1 && def->getNumResults() == 1) {
        CollectRootBlockArgs(def->getOperand(0), out, visited, depth + 1);
        return;
    }

    for (mlir::Value opnd : def->getOperands()) {
        CollectRootBlockArgs(opnd, out, visited, depth + 1);
    }
}

static std::optional<unsigned> InferOperandIndexFromComparatorCompare(mlir::stablehlo::CompareOp compareOp,
                                                                      unsigned numOperands) {
    llvm::SmallSet<unsigned, 4> lhsArgs;
    llvm::SmallSet<unsigned, 4> rhsArgs;
    llvm::SmallPtrSet<void*, 16> visited;
    CollectRootBlockArgs(compareOp.getLhs(), lhsArgs, visited);
    CollectRootBlockArgs(compareOp.getRhs(), rhsArgs, visited);

    for (unsigned i = 0; i < numOperands; ++i) {
        unsigned lhsIdx = 2 * i;
        unsigned rhsIdx = 2 * i + 1;
        if ((lhsArgs.count(lhsIdx) && rhsArgs.count(rhsIdx)) ||
            (lhsArgs.count(rhsIdx) && rhsArgs.count(lhsIdx))) {
            return i;
        }
    }
    return std::nullopt;
}

int64_t inferTopKFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count || outputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return outDim;
        }
    }
    return [outputShape.lastObject longLongValue];
}

NSInteger inferTopKAxisFromShapes(NSArray<NSNumber*>* inputShape, NSArray<NSNumber*>* outputShape) {
    if (!inputShape || !outputShape || inputShape.count != outputShape.count || inputShape.count == 0) {
        return -1;
    }
    for (NSUInteger i = 0; i < outputShape.count; ++i) {
        int64_t inDim = [inputShape[i] longLongValue];
        int64_t outDim = [outputShape[i] longLongValue];
        if (inDim != outDim) {
            return (NSInteger)i;
        }
    }
    return (NSInteger)inputShape.count - 1;
}

} // namespace

static ProcessResult HandleSort(HandlerContext& ctx) {
    auto sortOp = mlir::dyn_cast<mlir::stablehlo::SortOp>(ctx.op);
    if (!sortOp) {
        return ProcessResult::Error("Expected stablehlo.sort");
    }

    auto dimAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("dimension");
    if (!dimAttr) {
        return ProcessResult::Error("stablehlo.sort missing dimension attribute");
    }
    NSInteger axis = (NSInteger)dimAttr.getInt();

    bool descending = false;
    if (!sortOp.getComparator().empty()) {
        for (mlir::Operation& nestedOp : sortOp.getComparator().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            if (!IsOrderingCompare(compareOp)) {
                continue;
            }
            auto direction = compareOp.getComparisonDirection();
            descending = direction == mlir::stablehlo::ComparisonDirection::GT ||
                         direction == mlir::stablehlo::ComparisonDirection::GE;
            break;
        }
    }

    if (ctx.op->getNumOperands() == 1 && ctx.op->getNumResults() == 1) {
        MPSGraphTensor* input = GetInputTensor(ctx, 0);
        if (!input) {
            return ProcessResult::Error("stablehlo.sort input tensor not found");
        }
        MPSGraphTensor* sorted = [ctx.graph sortWithTensor:input axis:axis descending:descending name:nil];
        if (!sorted) {
            return ProcessResult::Error("stablehlo.sort lowering failed");
        }
        ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = sorted;
        return ProcessResult{};
    }

    if (ctx.op->getNumOperands() >= 2 && ctx.op->getNumOperands() == ctx.op->getNumResults()) {

        unsigned numOperands = ctx.op->getNumOperands();
        std::vector<unsigned> keyOperandIndices;
        if (!sortOp.getComparator().empty()) {
            for (mlir::Operation& nestedOp : sortOp.getComparator().front()) {
                auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
                if (!compareOp || !IsOrderingCompare(compareOp)) {
                    continue;
                }
                std::optional<unsigned> keyIdx =
                    InferOperandIndexFromComparatorCompare(compareOp, numOperands);
                if (!keyIdx.has_value()) {
                    continue;
                }
                if (std::find(keyOperandIndices.begin(), keyOperandIndices.end(), *keyIdx) ==
                    keyOperandIndices.end()) {
                    keyOperandIndices.push_back(*keyIdx);
                }
            }
        }

        if (keyOperandIndices.empty()) {

            if (auto numKeysAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("num_keys")) {
                int64_t numKeys = numKeysAttr.getInt();
                if (numKeys <= 0 || (uint64_t)numKeys > numOperands) {
                    return ProcessResult::Error("stablehlo.sort: invalid num_keys attribute");
                }
                for (int64_t i = 0; i < numKeys; ++i) {
                    keyOperandIndices.push_back((unsigned)i);
                }
            } else {
                keyOperandIndices.push_back(0);
            }
        }

        MPSGraphTensor* perm = nil;
        if (keyOperandIndices.size() == 1) {
            MPSGraphTensor* keyTensor = GetInputTensor(ctx, keyOperandIndices[0]);
            if (!keyTensor) {
                return ProcessResult::Error("stablehlo.sort key tensor missing");
            }
            perm = [ctx.graph argSortWithTensor:keyTensor axis:axis descending:descending name:nil];
            if (!perm) {
                return ProcessResult::Error("stablehlo.sort argSort lowering failed");
            }
            perm = EnsureInt32(ctx.graph, perm);
        } else {
            MPSGraphTensor* base = GetInputTensor(ctx, keyOperandIndices[0]);
            if (!base) {
                return ProcessResult::Error("stablehlo.sort key tensor not found");
            }

            perm = [ctx.graph coordinateAlongAxis:axis withShape:base.shape name:nil];
            perm = EnsureInt32(ctx.graph, perm);

            for (unsigned keyIdx : keyOperandIndices) {
                MPSGraphTensor* keyTensor = GetInputTensor(ctx, keyIdx);
                if (!keyTensor) {
                    return ProcessResult::Error("stablehlo.sort key tensor missing");
                }
                MPSGraphTensor* keyAtPerm = [ctx.graph gatherAlongAxis:axis
                                                     withUpdatesTensor:keyTensor
                                                         indicesTensor:perm
                                                                  name:nil];
                MPSGraphTensor* localOrder = [ctx.graph argSortWithTensor:keyAtPerm
                                                                     axis:axis
                                                               descending:descending
                                                                     name:nil];
                if (!localOrder) {
                    return ProcessResult::Error("stablehlo.sort argSort lowering failed");
                }
                localOrder = EnsureInt32(ctx.graph, localOrder);
                perm = [ctx.graph gatherAlongAxis:axis
                                withUpdatesTensor:perm
                                    indicesTensor:localOrder
                                             name:nil];
            }
        }

        for (unsigned i = 0; i < ctx.op->getNumResults(); ++i) {
            MPSGraphTensor* operandTensor = GetInputTensor(ctx, i);
            if (!operandTensor) {
                return ProcessResult::Error("stablehlo.sort operand tensor missing");
            }
            MPSGraphTensor* sorted = [ctx.graph gatherAlongAxis:axis
                                              withUpdatesTensor:operandTensor
                                                  indicesTensor:perm
                                                           name:nil];
            if (!sorted) {
                return ProcessResult::Error("stablehlo.sort gather lowering failed");
            }
            ctx.values[ctx.op->getResult(i).getAsOpaquePointer()] = sorted;
        }
        return ProcessResult{};
    }

    return ProcessResult::Error("Unsupported stablehlo.sort operand/result shape");
}

static ProcessResult HandleTopK(HandlerContext& ctx) {
    if (ctx.op->getNumOperands() < 1 || ctx.op->getNumResults() != 2) {
        return ProcessResult::Error("Unsupported top_k operand/result shape");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("top_k input tensor not found");
    }

    int64_t k = -1;
    if (auto kAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("k")) {
        k = kAttr.getInt();
    }

    NSInteger axis = -1;
    if (auto axisAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("axis")) {
        axis = (NSInteger)axisAttr.getInt();
    }

    NSArray<NSNumber*>* valueShape = GetOutputShape(ctx.op, 0);
    if (k < 0) {
        k = inferTopKFromShapes(input.shape, valueShape);
    }
    if (axis < 0) {
        axis = inferTopKAxisFromShapes(input.shape, valueShape);
    }

    if (k <= 0) {
        return ProcessResult::Error("Failed to infer top_k parameter k");
    }
    if (axis < 0 || !input.shape || axis >= (NSInteger)input.shape.count) {
        return ProcessResult::Error("Failed to infer top_k axis");
    }

    NSArray<MPSGraphTensor*>* topk = nil;
    if (axis == (NSInteger)input.shape.count - 1) {
        topk = [ctx.graph topKWithSourceTensor:input k:(NSUInteger)k name:nil];
    } else {
        topk = [ctx.graph topKWithSourceTensor:input axis:axis k:(NSUInteger)k name:nil];
    }
    if (!topk || topk.count != 2) {
        return ProcessResult::Error("top_k lowering failed");
    }

    MPSGraphTensor* valueOut = topk[0];
    MPSGraphTensor* indexOut = topk[1];
    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && valueOut.dataType != valueType) {
        valueOut = [ctx.graph castTensor:valueOut toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indexOut.dataType != indexType) {
        indexOut = [ctx.graph castTensor:indexOut toType:indexType name:nil];
    }

    if (valueShape && ![valueShape isEqualToArray:valueOut.shape]) {
        valueOut = [ctx.graph reshapeTensor:valueOut withShape:valueShape name:nil];
    }
    NSArray<NSNumber*>* indexShape = GetOutputShape(ctx.op, 1);
    if (indexShape && ![indexShape isEqualToArray:indexOut.shape]) {
        indexOut = [ctx.graph reshapeTensor:indexOut withShape:indexShape name:nil];
    }

    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = valueOut;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indexOut;
    return ProcessResult{};
}

REGISTER_MPS_OP("stablehlo.sort", HandleSort);
REGISTER_MPS_OP("chlo.top_k", HandleTopK);

REGISTER_CUSTOM_CALL("stablehlo.dynamic_top_k", HandleTopK, stablehlo_dynamic_top_k);
REGISTER_CUSTOM_CALL("mhlo.topk", HandleTopK, mhlo_topk);
REGISTER_CUSTOM_CALL("mhlo.top_k", HandleTopK, mhlo_top_k);

} // namespace jax_metallib
