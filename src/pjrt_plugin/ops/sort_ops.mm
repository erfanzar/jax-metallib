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

#include "mlir/Dialect/Func/IR/FuncOps.h"
#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"

#include <algorithm>
#include <optional>

namespace jax_metallib {

namespace {

// MPSGraph gatherAlongAxis moved int32/uint32 values above 2**24 inexactly
// (off by 1-2) while float32 was exact. Gather 32-bit integers as two 16-bit
// halves (each exact) and recombine them bitwise; other dtypes pass through.
MPSGraphTensor* ExactGatherAlongAxis(MPSGraph* graph, NSInteger axis, MPSGraphTensor* updates,
                                     MPSGraphTensor* indices) {
    MPSDataType dtype = updates.dataType;
    // mps.gather_along_axis rejects complex operands with a process-aborting
    // assertion; gather the real and imaginary parts separately (exact).
    if (dtype == MPSDataTypeComplexFloat32 || dtype == MPSDataTypeComplexFloat16) {
        MPSGraphTensor* re = [graph realPartOfTensor:updates name:nil];
        MPSGraphTensor* im = [graph imaginaryPartOfTensor:updates name:nil];
        MPSGraphTensor* gRe = [graph gatherAlongAxis:axis
                                   withUpdatesTensor:re
                                       indicesTensor:indices
                                                name:nil];
        MPSGraphTensor* gIm = [graph gatherAlongAxis:axis
                                   withUpdatesTensor:im
                                       indicesTensor:indices
                                                name:nil];
        if (!gRe || !gIm) {
            return nil;
        }
        return [graph complexTensorWithRealTensor:gRe imaginaryTensor:gIm name:nil];
    }
    if (dtype != MPSDataTypeInt32 && dtype != MPSDataTypeUInt32) {
        return [graph gatherAlongAxis:axis withUpdatesTensor:updates indicesTensor:indices name:nil];
    }
    MPSGraphTensor* mask = [graph constantWithScalar:0xFFFF dataType:dtype];
    MPSGraphTensor* sixteen = [graph constantWithScalar:16 dataType:dtype];
    MPSGraphTensor* lo = [graph bitwiseANDWithPrimaryTensor:updates secondaryTensor:mask name:nil];
    MPSGraphTensor* hi = [graph bitwiseRightShiftWithPrimaryTensor:updates secondaryTensor:sixteen name:nil];
    hi = [graph bitwiseANDWithPrimaryTensor:hi secondaryTensor:mask name:nil];
    MPSGraphTensor* gLo = [graph gatherAlongAxis:axis withUpdatesTensor:lo indicesTensor:indices name:nil];
    MPSGraphTensor* gHi = [graph gatherAlongAxis:axis withUpdatesTensor:hi indicesTensor:indices name:nil];
    if (!gLo || !gHi) {
        return nil;
    }
    gHi = [graph bitwiseLeftShiftWithPrimaryTensor:gHi secondaryTensor:sixteen name:nil];
    return [graph bitwiseORWithPrimaryTensor:gHi secondaryTensor:gLo name:nil];
}

// MPSGraph sortWithTensor returned inexact int32/uint32 values above 2**24
// (observed for large negative int32 in an ascending sort), while argSort was
// exact. For 32-bit integers, compute the permutation with argSort and move
// the values with the exact 16-bit-split gather. Equal values are
// indistinguishable, so argSort stability does not affect the result.
bool IsComplexDtype(MPSDataType t) {
    return t == MPSDataTypeComplexFloat32 || t == MPSDataTypeComplexFloat16;
}

// Sort permutation (int32) for one key. mps.sort/argSort reject complex
// operands with a process-aborting assertion, and JAX orders complex values
// lexicographically by (real, imag): stable-argsort the imaginary parts, then
// stable-argsort the real parts taken in that order, and compose the two.
MPSGraphTensor* ArgSortKey(MPSGraph* graph, MPSGraphTensor* key, NSInteger axis, bool descending) {
    if (!IsComplexDtype(key.dataType)) {
        MPSGraphTensor* perm = [graph argSortWithTensor:key axis:axis descending:descending name:nil];
        return perm ? EnsureInt32(graph, perm) : nil;
    }
    MPSGraphTensor* re = [graph realPartOfTensor:key name:nil];
    MPSGraphTensor* im = [graph imaginaryPartOfTensor:key name:nil];
    MPSGraphTensor* order = [graph argSortWithTensor:im axis:axis descending:descending name:nil];
    if (!order) {
        return nil;
    }
    order = EnsureInt32(graph, order);
    MPSGraphTensor* reInOrder = [graph gatherAlongAxis:axis
                                     withUpdatesTensor:re
                                         indicesTensor:order
                                                  name:nil];
    MPSGraphTensor* local = [graph argSortWithTensor:reInOrder axis:axis descending:descending name:nil];
    if (!local) {
        return nil;
    }
    local = EnsureInt32(graph, local);
    return [graph gatherAlongAxis:axis withUpdatesTensor:order indicesTensor:local name:nil];
}

MPSGraphTensor* ExactSortValues(MPSGraph* graph, MPSGraphTensor* input, NSInteger axis, bool descending) {
    const bool int32Like = input.dataType == MPSDataTypeInt32 || input.dataType == MPSDataTypeUInt32;
    if (!int32Like && !IsComplexDtype(input.dataType)) {
        return [graph sortWithTensor:input axis:axis descending:descending name:nil];
    }
    MPSGraphTensor* perm = ArgSortKey(graph, input, axis, descending);
    if (!perm) {
        return nil;
    }
    return ExactGatherAlongAxis(graph, axis, input, perm);
}

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
        MPSGraphTensor* sorted = ExactSortValues(ctx.graph, input, axis, descending);
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
            // The combined `sortWithTensor:axis:descending:withValueTensor:andIndexTensor:`
            // API does not exist on the on-disk MPSGraph SDK. Build the equivalent
            // from the two single-purpose ops we do have:
            //   - argSortWithTensor  -> permutation
            //   - sortWithTensor     -> sorted values
            //   - gatherAlongAxis    -> apply the permutation to other operands
            MPSGraphTensor* keyTensor = GetInputTensor(ctx, keyOperandIndices[0]);
            if (!keyTensor) {
                return ProcessResult::Error("stablehlo.sort key tensor missing");
            }
            MPSGraphTensor* sortedKey = ExactSortValues(ctx.graph, keyTensor, axis, descending);
            perm = ArgSortKey(ctx.graph, keyTensor, axis, descending);
            if (!sortedKey || !perm) {
                return ProcessResult::Error("stablehlo.sort lowering failed");
            }
            perm = EnsureInt32(ctx.graph, perm);
            ctx.values[ctx.op->getResult(keyOperandIndices[0]).getAsOpaquePointer()] = sortedKey;

            // Apply same permutation to all other operands in parallel
            for (unsigned i = 0; i < ctx.op->getNumResults(); ++i) {
                if (i == keyOperandIndices[0]) {
                    continue;
                }
                MPSGraphTensor* operand = GetInputTensor(ctx, i);
                if (!operand) {
                    return ProcessResult::Error("stablehlo.sort operand tensor missing");
                }
                MPSGraphTensor* sorted = ExactGatherAlongAxis(ctx.graph, axis, operand, perm);
                if (!sorted) {
                    return ProcessResult::Error("stablehlo.sort gather lowering failed");
                }
                ctx.values[ctx.op->getResult(i).getAsOpaquePointer()] = sorted;
            }
            return ProcessResult{};
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
                MPSGraphTensor* keyAtPerm = ExactGatherAlongAxis(ctx.graph, axis, keyTensor, perm);
                MPSGraphTensor* localOrder = ArgSortKey(ctx.graph, keyAtPerm, axis, descending);
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
            MPSGraphTensor* sorted = ExactGatherAlongAxis(ctx.graph, axis, operandTensor, perm);
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

    MPSGraphTensor* valueOut = nil;
    MPSGraphTensor* indexOut = nil;
    if (input.dataType == MPSDataTypeInt32 || input.dataType == MPSDataTypeUInt32) {
        // MPSGraph topK returned wrong int32/uint32 values (consistent with a
        // float32 comparison). Build an order-preserving int32 key (sign-bit
        // flip for uint32; bitwise NOT so ascending means largest first),
        // stable-argsort it, keep the first k and gather the values exactly.
        MPSGraphTensor* key = input.dataType == MPSDataTypeInt32
                                  ? input
                                  : [ctx.graph reinterpretCastTensor:input toType:MPSDataTypeInt32 name:nil];
        if (input.dataType == MPSDataTypeUInt32) {
            MPSGraphTensor* signBit = [ctx.graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
            key = [ctx.graph bitwiseXORWithPrimaryTensor:key secondaryTensor:signBit name:nil];
        }
        key = [ctx.graph bitwiseNOTWithTensor:key name:nil];
        MPSGraphTensor* order = [ctx.graph argSortWithTensor:key axis:axis descending:NO name:nil];
        if (!order) {
            return ProcessResult::Error("top_k argSort lowering failed");
        }
        order = EnsureInt32(ctx.graph, order);
        order = [ctx.graph sliceTensor:order dimension:axis start:0 length:(NSInteger)k name:nil];
        valueOut = ExactGatherAlongAxis(ctx.graph, axis, input, order);
        indexOut = order;
        if (!valueOut || !indexOut) {
            return ProcessResult::Error("top_k integer lowering failed");
        }
    } else {
        NSArray<MPSGraphTensor*>* topk = nil;
        if (axis == (NSInteger)input.shape.count - 1) {
            topk = [ctx.graph topKWithSourceTensor:input k:(NSUInteger)k name:nil];
        } else {
            topk = [ctx.graph topKWithSourceTensor:input axis:axis k:(NSUInteger)k name:nil];
        }
        if (!topk || topk.count != 2) {
            return ProcessResult::Error("top_k lowering failed");
        }
        valueOut = topk[0];
        indexOut = topk[1];
    }
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

// lax.approx_max_k / approx_min_k lower to the ApproxTopK custom call, which had
// no MPS handler. Off TPU, XLA treats it as an exact top-k (is_fallback=true),
// so this lowers it exactly: argsort along reduction_dim, keep the first k,
// and gather values and the supplied index operand with the exact gathers.
// Direction comes from the comparator in called_computations (GT/GE = max,
// LT/LE = min), falling back to the init value (-inf = max, +inf = min).
static int ApproxTopKDirection(HandlerContext& ctx) {
    if (auto calls = ctx.op->getAttrOfType<mlir::ArrayAttr>("called_computations")) {
        for (mlir::Attribute attr : calls) {
            auto ref = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr);
            if (!ref || !ctx.module) {
                continue;
            }
            auto fn = ctx.module.lookupSymbol<mlir::func::FuncOp>(ref.getValue());
            if (!fn || fn.empty()) {
                continue;
            }
            int dir = 0;
            fn.walk([&](mlir::stablehlo::CompareOp cmp) {
                if (dir != 0 || !IsOrderingCompare(cmp)) {
                    return;
                }
                auto d = cmp.getComparisonDirection();
                dir = (d == mlir::stablehlo::ComparisonDirection::GT ||
                       d == mlir::stablehlo::ComparisonDirection::GE)
                          ? 1
                          : -1;
            });
            if (dir != 0) {
                return dir;
            }
        }
    }
    if (ctx.op->getNumOperands() >= 3) {
        if (auto cst = ctx.op->getOperand(2).getDefiningOp<mlir::stablehlo::ConstantOp>()) {
            auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(cst.getValue());
            if (dense && dense.isSplat()) {
                llvm::APFloat v = dense.getSplatValue<llvm::APFloat>();
                if (v.isInfinity()) {
                    return v.isNegative() ? 1 : -1;
                }
            }
        }
    }
    return 0;
}

static ProcessResult HandleApproxTopK(HandlerContext& ctx) {
    if (ctx.op->getNumOperands() < 1 || ctx.op->getNumResults() != 2) {
        return ProcessResult::Error("ApproxTopK: expected operands and two results");
    }
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* indexSource = ctx.op->getNumOperands() >= 2 ? GetInputTensor(ctx, 1) : nil;
    if (!input) {
        return ProcessResult::Error("ApproxTopK: missing input tensor");
    }
    if (IsComplexDtype(input.dataType)) {
        return ProcessResult::Error("ApproxTopK: complex inputs are not supported on MPS");
    }
    NSArray<NSNumber*>* inShape = input.shape;
    NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op, 0);
    if (!inShape || !outShape || inShape.count != outShape.count || inShape.count == 0) {
        return ProcessResult::Error("ApproxTopK: expected static, rank-matching shapes");
    }
    int64_t axis = -1;
    if (auto cfg = ctx.op->getAttrOfType<mlir::DictionaryAttr>("mhlo.backend_config")) {
        if (auto rd = cfg.getAs<mlir::IntegerAttr>("reduction_dim")) {
            axis = rd.getInt();
        }
    }
    if (axis < 0) {
        axis += (int64_t)inShape.count;
    }
    if (axis < 0 || axis >= (int64_t)inShape.count) {
        axis = inferTopKAxisFromShapes(inShape, outShape);
    }
    if (axis < 0 || axis >= (int64_t)inShape.count) {
        return ProcessResult::Error("ApproxTopK: invalid reduction_dim");
    }
    const int dir = ApproxTopKDirection(ctx);
    if (dir == 0) {
        return ProcessResult::Error("ApproxTopK: could not determine max/min comparator");
    }
    const int64_t n = [inShape[(NSUInteger)axis] longLongValue];
    const int64_t k = [outShape[(NSUInteger)axis] longLongValue];
    if (k < 0 || k > n) {
        return ProcessResult::Error("ApproxTopK: invalid k");
    }

    MPSGraph* g = ctx.graph;
    MPSGraphTensor* perm = ArgSortKey(g, input, (NSInteger)axis, dir > 0);
    if (!perm) {
        return ProcessResult::Error("ApproxTopK: argsort lowering failed");
    }
    if (k < n) {
        perm = [g sliceTensor:perm dimension:(NSInteger)axis start:0 length:(NSInteger)k name:nil];
    }
    MPSGraphTensor* values = ExactGatherAlongAxis(g, (NSInteger)axis, input, perm);
    MPSGraphTensor* indices =
        indexSource ? ExactGatherAlongAxis(g, (NSInteger)axis, indexSource, perm) : perm;
    if (!values || !indices) {
        return ProcessResult::Error("ApproxTopK: gather lowering failed");
    }
    MPSDataType valueType = GetResultMpsType(ctx.op, 0);
    if (valueType != MPSDataTypeInvalid && values.dataType != valueType) {
        values = [g castTensor:values toType:valueType name:nil];
    }
    MPSDataType indexType = GetResultMpsType(ctx.op, 1);
    if (indexType != MPSDataTypeInvalid && indices.dataType != indexType) {
        indices = [g castTensor:indices toType:indexType name:nil];
    }
    ctx.values[ctx.op->getResult(0).getAsOpaquePointer()] = values;
    ctx.values[ctx.op->getResult(1).getAsOpaquePointer()] = indices;
    return ProcessResult{};
}

REGISTER_CUSTOM_CALL("ApproxTopK", HandleApproxTopK, approx_top_k);
REGISTER_CUSTOM_CALL("stablehlo.dynamic_top_k", HandleTopK, stablehlo_dynamic_top_k);
REGISTER_CUSTOM_CALL("mhlo.topk", HandleTopK, mhlo_topk);
REGISTER_CUSTOM_CALL("mhlo.top_k", HandleTopK, mhlo_top_k);

} // namespace jax_metallib
