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
#include <cstdlib>
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
        MPSGraphTensor* gRe = ExactGatherAlongAxis(graph, axis, re, indices);
        MPSGraphTensor* gIm = ExactGatherAlongAxis(graph, axis, im, indices);
        if (!gRe || !gIm) {
            return nil;
        }
        return [graph complexTensorWithRealTensor:gRe imaginaryTensor:gIm name:nil];
    }
    // gatherAlongAxis rewrote 16-bit float NaNs to +qNaN (-NaN 0xfe00 and payload
    // NaNs became 0x7e00) and flushed bf16 subnormals to zero; move the raw bits.
    // XLA:CPU sorts bf16 through f32 and narrows back, so every bf16 NaN it
    // emits is sign|0x7fc0 (keys and values, both directions); a length-1 axis
    // is a no-op there and keeps the raw bits. f16 keeps raw bits throughout.
    if (dtype == MPSDataTypeFloat16 || dtype == MPSDataTypeBFloat16) {
        MPSGraphTensor* raw = [graph reinterpretCastTensor:updates toType:MPSDataTypeInt16 name:nil];
        MPSGraphTensor* moved = [graph gatherAlongAxis:axis
                                     withUpdatesTensor:raw
                                         indicesTensor:indices
                                                  name:nil];
        if (!moved) {
            return nil;
        }
        NSArray<NSNumber*>* shape = updates.shape;
        const bool trivialAxis =
            shape && axis >= 0 && axis < (NSInteger)shape.count && shape[axis].longLongValue == 1;
        if (dtype == MPSDataTypeBFloat16 && !trivialAxis) {
            MPSGraphTensor* magMask = [graph constantWithScalar:32767.0 dataType:MPSDataTypeInt16];
            MPSGraphTensor* infBits = [graph constantWithScalar:32640.0 dataType:MPSDataTypeInt16];
            MPSGraphTensor* signMask = [graph constantWithScalar:-32768.0 dataType:MPSDataTypeInt16];
            MPSGraphTensor* qnan = [graph constantWithScalar:32704.0 dataType:MPSDataTypeInt16];
            MPSGraphTensor* mag = [graph bitwiseANDWithPrimaryTensor:moved secondaryTensor:magMask name:nil];
            MPSGraphTensor* isNan = [graph greaterThanWithPrimaryTensor:mag secondaryTensor:infBits name:nil];
            MPSGraphTensor* sign = [graph bitwiseANDWithPrimaryTensor:moved
                                                      secondaryTensor:signMask
                                                                 name:nil];
            MPSGraphTensor* canon = [graph bitwiseORWithPrimaryTensor:sign secondaryTensor:qnan name:nil];
            moved = [graph selectWithPredicateTensor:isNan
                                 truePredicateTensor:canon
                                falsePredicateTensor:moved
                                                name:nil];
        }
        return [graph reinterpretCastTensor:moved toType:dtype name:nil];
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
// XLA's float sort comparator maps every NaN to +NaN (sorted after +inf) and
// -0.0 to +0.0 (equal, so a stable sort keeps their input order). MPSGraph
// sort/argSort order by sign instead: -NaN sorted first and -0 before +0.
// JAX_METALLIB_SORT_LEGACY_VALUES=1 restores the old sortWithTensor values path.
static bool EnvFlagEnabledSort(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

static bool IsRealFloatDtype(MPSDataType dt) {
    return dt == MPSDataTypeFloat32 || dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
}

static MPSGraphTensor* CanonicalizeSortNaN(MPSGraph* graph, MPSGraphTensor* t) {
    if (!IsRealFloatDtype(t.dataType)) {
        return t;
    }
    MPSGraphTensor* nan = [graph constantWithScalar:NAN dataType:t.dataType];
    return [graph selectWithPredicateTensor:[graph isNaNWithTensor:t name:nil]
                        truePredicateTensor:nan
                       falsePredicateTensor:t
                                       name:nil];
}

static MPSGraphTensor* CanonicalSortKey(MPSGraph* graph, MPSGraphTensor* t) {
    if (!IsRealFloatDtype(t.dataType)) {
        return t;
    }
    MPSGraphTensor* zero = [graph constantWithScalar:0.0 dataType:t.dataType];
    MPSGraphTensor* isZero = [graph equalWithPrimaryTensor:t secondaryTensor:zero name:nil];
    MPSGraphTensor* noNegZero = [graph selectWithPredicateTensor:isZero
                                             truePredicateTensor:zero
                                            falsePredicateTensor:t
                                                            name:nil];
    return CanonicalizeSortNaN(graph, noNegZero);
}

// MPSGraph argSort costs a roughly fixed ~1.3-2 us per row, so many short rows
// are slow (f32 [65536,64] ~79 ms, [16384,256] ~24 ms for 4M elements) while one
// long row is not. Rows of at most JAX_METALLIB_SORT_COMPOSITE_MAX_ROW elements
// on the last axis are sorted as ONE flat stable argSort of an int64 composite
// key row * 2^32 + (s + 2^31), where s is an order-preserving int32 key
// (canonical -0/NaN like XLA's comparator; ~s for descending so ties keep
// input order). The composite costs ~5.3 ms per 1M elements whatever the row
// length, so it is gated on short AND numerous rows: 32-bit keys L <= 256 and
// rows >= 1024 (f32 4M L=64 88 -> 32 ms, 1M L=256 8.3 -> 5.3 ms); 16-bit and
// narrower keys (argSort is ~2x cheaper per row) L <= 64 and rows >= 4096.
// JAX_METALLIB_SORT_COMPOSITE_MAX_ROW overrides: 0 disables, N > 0 uses it for
// every rows >= 2 with L <= N. Returns -1 when unset (auto gate).
static int64_t SortCompositeMaxRow() {
    const char* v = std::getenv("JAX_METALLIB_SORT_COMPOSITE_MAX_ROW");
    if (!v || v[0] == '\0') {
        return -1;
    }
    return std::max<int64_t>(0, std::strtoll(v, nullptr, 10));
}

static MPSGraphTensor* OrderedInt32SortKey(MPSGraph* graph, MPSGraphTensor* t) {
    MPSDataType dt = t.dataType;
    if (IsRealFloatDtype(dt)) {
        if (dt != MPSDataTypeFloat32) {
            t = [graph castTensor:t toType:MPSDataTypeFloat32 name:nil];
        }
        t = CanonicalizeSortNaN(graph, t);
        MPSGraphTensor* bits = [graph reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* zero = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* magMask = [graph constantWithScalar:2147483647.0 dataType:MPSDataTypeInt32];
        // -0, +0 and subnormals share key 0 (XLA:CPU compares with
        // denormals-as-zero); decided on the bits, not on a float compare.
        MPSGraphTensor* minNormal = [graph constantWithScalar:8388608.0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* mag = [graph bitwiseANDWithPrimaryTensor:bits secondaryTensor:magMask name:nil];
        MPSGraphTensor* tiny = [graph lessThanWithPrimaryTensor:mag secondaryTensor:minNormal name:nil];
        bits = [graph selectWithPredicateTensor:tiny
                            truePredicateTensor:zero
                           falsePredicateTensor:bits
                                           name:nil];
        MPSGraphTensor* flipped = [graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:magMask name:nil];
        MPSGraphTensor* neg = [graph lessThanWithPrimaryTensor:bits secondaryTensor:zero name:nil];
        return [graph selectWithPredicateTensor:neg
                            truePredicateTensor:flipped
                           falsePredicateTensor:bits
                                           name:nil];
    }
    if (dt == MPSDataTypeUInt32) {
        MPSGraphTensor* bits = [graph reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* signBit = [graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
        return [graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:signBit name:nil];
    }
    if (dt == MPSDataTypeInt32) {
        return t;
    }
    if (dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16 || dt == MPSDataTypeInt8 ||
        dt == MPSDataTypeUInt8 || dt == MPSDataTypeBool) {
        return [graph castTensor:t toType:MPSDataTypeInt32 name:nil];
    }
    return nil;
}

static MPSGraphTensor* CompositeRowArgSort(MPSGraph* graph, MPSGraphTensor* key, NSInteger axis,
                                           bool descending) {
    const int64_t forcedMaxRow = SortCompositeMaxRow();
    const MPSDataType kdt = key.dataType;
    const bool wideKey = kdt == MPSDataTypeFloat32 || kdt == MPSDataTypeInt32 || kdt == MPSDataTypeUInt32;
    int64_t maxRow = wideKey ? 256 : 64;
    int64_t minRows = wideKey ? 1024 : 4096;
    if (forcedMaxRow >= 0) {
        maxRow = forcedMaxRow;
        minRows = 2;
    }
    NSArray<NSNumber*>* shape = key.shape;
    if (maxRow <= 0 || !shape || shape.count < 2 || axis != (NSInteger)shape.count - 1) {
        return nil;
    }
    int64_t total = 1;
    for (NSNumber* d : shape) {
        if (d.longLongValue <= 0) {
            return nil;
        }
        total *= d.longLongValue;
    }
    const int64_t cols = shape.lastObject.longLongValue;
    const int64_t rows = total / cols;
    if (rows < minRows || cols < 2 || cols > maxRow || total >= (int64_t{1} << 31)) {
        return nil;
    }
    MPSGraphTensor* s = OrderedInt32SortKey(graph, key);
    if (!s) {
        return nil;
    }
    if (descending) {
        s = [graph bitwiseNOTWithTensor:s name:nil];
    }
    NSArray<NSNumber*>* shape2d = @[@(rows), @(cols)];
    s = [graph reshapeTensor:s withShape:shape2d name:nil];
    MPSGraphTensor* row = EnsureInt32(graph, [graph coordinateAlongAxis:0 withShape:shape2d name:nil]);
    MPSGraphTensor* row64 = [graph castTensor:row toType:MPSDataTypeInt64 name:nil];
    MPSGraphTensor* twoPow32 = [graph constantWithScalar:4294967296.0 dataType:MPSDataTypeInt64];
    MPSGraphTensor* twoPow31 = [graph constantWithScalar:2147483648.0 dataType:MPSDataTypeInt64];
    MPSGraphTensor* low = [graph additionWithPrimaryTensor:[graph castTensor:s
                                                                      toType:MPSDataTypeInt64
                                                                        name:nil]
                                           secondaryTensor:twoPow31
                                                      name:nil];
    MPSGraphTensor* composite =
        [graph additionWithPrimaryTensor:[graph multiplicationWithPrimaryTensor:row64
                                                                secondaryTensor:twoPow32
                                                                           name:nil]
                         secondaryTensor:low
                                    name:nil];
    composite = [graph reshapeTensor:composite withShape:@[@(total)] name:nil];
    MPSGraphTensor* flat = [graph argSortWithTensor:composite axis:0 descending:NO name:nil];
    if (!flat) {
        return nil;
    }
    flat = [graph reshapeTensor:EnsureInt32(graph, flat) withShape:shape2d name:nil];
    MPSGraphTensor* colsT = [graph constantWithScalar:(double)cols dataType:MPSDataTypeInt32];
    MPSGraphTensor* rowStart = [graph multiplicationWithPrimaryTensor:row secondaryTensor:colsT name:nil];
    MPSGraphTensor* local = [graph subtractionWithPrimaryTensor:flat secondaryTensor:rowStart name:nil];
    return [graph reshapeTensor:local withShape:shape name:nil];
}

MPSGraphTensor* ArgSortKey(MPSGraph* graph, MPSGraphTensor* key, NSInteger axis, bool descending) {
    if (!IsComplexDtype(key.dataType)) {
        if (MPSGraphTensor* composite = CompositeRowArgSort(graph, key, axis, descending)) {
            return composite;
        }
        MPSGraphTensor* perm = [graph argSortWithTensor:CanonicalSortKey(graph, key)
                                                   axis:axis
                                             descending:descending
                                                   name:nil];
        return perm ? EnsureInt32(graph, perm) : nil;
    }
    MPSGraphTensor* re = CanonicalSortKey(graph, [graph realPartOfTensor:key name:nil]);
    MPSGraphTensor* im = CanonicalSortKey(graph, [graph imaginaryPartOfTensor:key name:nil]);
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
    if (!int32Like && !IsComplexDtype(input.dataType) && !IsRealFloatDtype(input.dataType)) {
        return [graph sortWithTensor:input axis:axis descending:descending name:nil];
    }
    // Real floats go through argsort + gather: XLA canonicalizes -0 and NaN only
    // inside the comparator and returns the original bits in stable order.
    // Sorting canonicalized values directly rewrote every NaN to +NaN, and
    // sortWithTensor is not stable on -0/+0 ties.
    if (IsRealFloatDtype(input.dataType) && !EnvFlagEnabledSort("JAX_METALLIB_SORT_LEGACY_VALUES")) {
        MPSGraphTensor* perm = ArgSortKey(graph, input, axis, descending);
        return perm ? ExactGatherAlongAxis(graph, axis, input, perm) : nil;
    }
    if (IsRealFloatDtype(input.dataType)) {
        return [graph sortWithTensor:CanonicalizeSortNaN(graph, input)
                                axis:axis
                          descending:descending
                                name:nil];
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

static MPSGraphTensor* TopKOrderKey(MPSGraph* graph, MPSGraphTensor* input);

// lax.top_k arrives as iota -> stable sort (GT TOTALORDER) -> slice[..., 0:k]
// on both results. Returns the widest leading-slice limit along `axis` when
// every user of every sort result is such a slice (start 0, stride 1), else -1.
static int64_t LeadingSliceWidth(mlir::Operation* op, NSInteger axis) {
    int64_t width = 0;
    for (mlir::Value result : op->getResults()) {
        for (mlir::Operation* user : result.getUsers()) {
            auto slice = mlir::dyn_cast<mlir::stablehlo::SliceOp>(user);
            if (!slice) {
                return -1;
            }
            auto starts = slice.getStartIndices();
            auto limits = slice.getLimitIndices();
            auto strides = slice.getStrides();
            if (axis < 0 || (size_t)axis >= starts.size() || starts[axis] != 0 || strides[axis] != 1) {
                return -1;
            }
            width = std::max<int64_t>(width, limits[axis]);
        }
    }
    return width;
}

// A full argsort costs a flat ~5.5 ms per 1M f32 elements on rows <= 256 (3.6
// ms at L=1024, 1.7 at 4096, 1.3 at 16384; ~1.1 ms at 256K), while top_k keeps
// only k. Selecting the permutation costs ~0.04 ms + ~0.2 ms per 1M elements
// per round (~0.55 per 1M at L=16), so the auto cap depends on the row length
// L and the total size N (measured f32, 64K-1M elements): 16 for 64 <= L <= 512
// and N >= 2^20; 12 for 64 <= L <= 1024 and N >= 2^18; 8 for L < 64 and N >=
// 2^20; 4 with >= 32 rows; 2 with >= 8 rows; else 1 (64K L=16384, 4 rows: k=2
// 0.32 vs 0.31 ms, k=1 0.27 vs 0.32). f32 1M k=1:
// 5.4 -> 0.40-0.56 ms. JAX_METALLIB_TOPK_SELECT_MAX_K=N > 0 forces a flat
// cap, 0 disables.
static int64_t TopKSelectMaxK(int64_t len, int64_t total) {
    const char* v = std::getenv("JAX_METALLIB_TOPK_SELECT_MAX_K");
    if (v && v[0] != '\0') {
        return std::max<int64_t>(0, std::strtoll(v, nullptr, 10));
    }
    if (len >= 64 && len <= 512 && total >= (int64_t{1} << 20)) {
        return 16;
    }
    if (len >= 64 && len <= 1024 && total >= (int64_t{1} << 18)) {
        return 12;
    }
    if (len < 64 && total >= (int64_t{1} << 20)) {
        return 8;
    }
    const int64_t rows = len > 0 ? total / len : 0;
    if (rows >= 32) {
        return 4;
    }
    return rows >= 8 ? 2 : 1;
}

// int32 key whose signed order is the integer order of `t` (raw GT compare of
// integer top_k), or nil for dtypes the selection path does not take.
static MPSGraphTensor* IntSelectKey(MPSGraph* g, MPSGraphTensor* t) {
    const MPSDataType dt = t.dataType;
    if (dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32) {
        return TopKOrderKey(g, t);
    }
    if (dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16 || dt == MPSDataTypeInt8 ||
        dt == MPSDataTypeUInt8) {
        // Sign-/zero-extension keeps the order.
        return [g castTensor:t toType:MPSDataTypeInt32 name:nil];
    }
    return nil;
}

// First k positions of a stable descending sort of the int32 `key` along
// `axis`: each round takes the max over not-yet-taken elements and the first
// untaken index holding it. Taken elements are masked to INT32_MIN, which a
// real key can also equal (f32 -NaN 0xFFFFFFFF), so the hit test excludes
// taken positions explicitly instead of relying on the mask value.
static MPSGraphTensor* TopKSelectPerm(MPSGraph* g, MPSGraphTensor* key, NSInteger axis, int64_t k) {
    MPSGraphTensor* coord = EnsureInt32(g, [g coordinateAlongAxis:axis withShape:key.shape name:nil]);
    MPSGraphTensor* minKey = [g constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* taken = nil;
    NSMutableArray<MPSGraphTensor*>* picks = [NSMutableArray arrayWithCapacity:(NSUInteger)k];
    for (int64_t r = 0; r < k; ++r) {
        MPSGraphTensor* masked = taken ? [g selectWithPredicateTensor:taken
                                                  truePredicateTensor:minKey
                                                 falsePredicateTensor:key
                                                                 name:nil]
                                       : key;
        MPSGraphTensor* best = [g reductionMaximumWithTensor:masked axis:axis name:nil];
        MPSGraphTensor* hit = [g equalWithPrimaryTensor:key secondaryTensor:best name:nil];
        if (taken) {
            hit = [g logicalANDWithPrimaryTensor:hit
                                 secondaryTensor:[g notWithTensor:taken name:nil]
                                            name:nil];
        }
        MPSGraphTensor* hitU8 = [g castTensor:hit toType:MPSDataTypeUInt8 name:nil];
        MPSGraphTensor* idx = [g reductionArgMaximumWithTensor:hitU8 axis:axis name:nil];
        if (!best || !idx) {
            return nil;
        }
        // Arg-reductions can return -1 when every element equals the sentinel;
        // a hit always exists, so that case means all hit and index 0 is first.
        idx = [g maximumWithPrimaryTensor:EnsureInt32(g, idx) secondaryTensor:zero name:nil];
        [picks addObject:idx];
        MPSGraphTensor* pick = [g equalWithPrimaryTensor:coord secondaryTensor:idx name:nil];
        taken = taken ? [g logicalORWithPrimaryTensor:taken secondaryTensor:pick name:nil] : pick;
    }
    return picks.count == 1 ? picks[0] : [g concatTensors:picks dimension:axis name:nil];
}

static bool PadExactSupported(MPSDataType dt) {
    return dt == MPSDataTypeFloat32 || dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32 ||
           dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16 || dt == MPSDataTypeInt16 ||
           dt == MPSDataTypeUInt16;
}

// Zero-pads `t` along `axis` up to `len` without moving values through a float
// path: 32-bit dtypes pad as int32 bits, 16-bit dtypes as their sign-extended
// int32 bits, cast back afterwards. Returns nil for other dtypes.
static MPSGraphTensor* PadAlongAxisExact(MPSGraph* g, MPSGraphTensor* t, NSInteger axis, int64_t len) {
    const MPSDataType dt = t.dataType;
    NSArray<NSNumber*>* shape = t.shape;
    if (!PadExactSupported(dt) || !shape || axis < 0 || axis >= (NSInteger)shape.count) {
        return nil;
    }
    const int64_t cur = shape[(NSUInteger)axis].longLongValue;
    if (cur >= len) {
        return t;
    }
    const bool wide = dt == MPSDataTypeFloat32 || dt == MPSDataTypeInt32 || dt == MPSDataTypeUInt32;
    const MPSDataType bitsType = wide ? MPSDataTypeInt32 : MPSDataTypeInt16;
    MPSGraphTensor* bits = dt == bitsType ? t : [g reinterpretCastTensor:t toType:bitsType name:nil];
    if (!wide) {
        bits = [g castTensor:bits toType:MPSDataTypeInt32 name:nil];
    }
    NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:shape.count];
    NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:shape.count];
    for (NSUInteger d = 0; d < shape.count; ++d) {
        [left addObject:@0];
        [right addObject:@(d == (NSUInteger)axis ? len - cur : 0)];
    }
    bits = [g padTensor:bits
        withPaddingMode:MPSGraphPaddingModeConstant
            leftPadding:left
           rightPadding:right
          constantValue:0.0
                   name:nil];
    if (!wide) {
        bits = [g castTensor:bits toType:MPSDataTypeInt16 name:nil];
    }
    return dt == bitsType ? bits : [g reinterpretCastTensor:bits toType:dt name:nil];
}

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
    // jnp.sort/argsort/lexsort canonicalise -0 and NaN with selects before a
    // TOTALORDER compare; lax.top_k (the chlo.top_k composite decomposition)
    // compares the raw block arguments with TOTALORDER, i.e. IEEE totalOrder
    // on the bits (-NaN < -0 < +0 < subnormals < +NaN, no DAZ). Only the latter
    // takes the raw total-order key below; every other comparator is unchanged.
    bool rawTotalOrder = false;
    bool rawIntGT = false;
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
            auto compareType = compareOp.getCompareType();
            const bool rawArgs = mlir::isa<mlir::BlockArgument>(compareOp.getLhs()) &&
                                 mlir::isa<mlir::BlockArgument>(compareOp.getRhs());
            rawTotalOrder =
                rawArgs && compareType && *compareType == mlir::stablehlo::ComparisonType::TOTALORDER;
            // Integer lax.top_k: raw GT with the element type's own signedness.
            if (rawArgs && direction == mlir::stablehlo::ComparisonDirection::GT) {
                auto shaped = mlir::dyn_cast<mlir::ShapedType>(compareOp.getLhs().getType());
                auto intTy =
                    shaped ? mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType()) : mlir::IntegerType();
                if (intTy && intTy.getWidth() > 1 && intTy.getWidth() <= 32) {
                    const auto want = intTy.isUnsigned() ? mlir::stablehlo::ComparisonType::UNSIGNED
                                                         : mlir::stablehlo::ComparisonType::SIGNED;
                    rawIntGT = !compareType || *compareType == want;
                }
            }
            break;
        }
    }
    // Stable permutation in raw total order for a real-float key, or nil to use
    // the canonical sort path (JAX_METALLIB_TOPK_LEGACY_FLOAT=1 forces nil).
    // Integer keys only take the small-k selection path (integer sorts are
    // already exact); JAX_METALLIB_TOPK_SELECT_INT=0 keeps them on the full sort.
    if (const char* v = std::getenv("JAX_METALLIB_TOPK_SELECT_INT"); v && v[0] == '0' && v[1] == '\0') {
        rawIntGT = false;
    }
    const int64_t sliceWidth =
        (rawTotalOrder || rawIntGT) && descending ? LeadingSliceWidth(ctx.op, axis) : -1;
    // Set when totalOrderPerm returned only the leading sliceWidth positions.
    int64_t selectPadTo = 0;
    auto totalOrderPerm = [&](MPSGraphTensor* key) -> MPSGraphTensor* {
        if (!key) {
            return nil;
        }
        const bool intKey = rawIntGT && !IsRealFloatDtype(key.dataType);
        if (!intKey && (!rawTotalOrder || !IsRealFloatDtype(key.dataType))) {
            return nil;
        }
        MPSGraphTensor* k = intKey ? IntSelectKey(ctx.graph, key) : TopKOrderKey(ctx.graph, key);
        if (!k) {
            return nil;
        }
        NSArray<NSNumber*>* shape = k.shape;
        const int64_t len =
            shape && axis >= 0 && axis < (NSInteger)shape.count ? shape[(NSUInteger)axis].longLongValue : -1;
        int64_t total = 1;
        for (NSNumber* d in shape) {
            total *= std::max<int64_t>(0, d.longLongValue);
        }
        bool padOk = true;
        for (unsigned i = 0; i < ctx.op->getNumOperands(); ++i) {
            MPSGraphTensor* operand = GetInputTensor(ctx, i);
            padOk = padOk && operand && PadExactSupported(operand.dataType);
        }
        if (padOk && sliceWidth > 0 && len > 0 && sliceWidth < len &&
            sliceWidth <= TopKSelectMaxK(len, total)) {
            if (MPSGraphTensor* sel = TopKSelectPerm(ctx.graph, k, axis, sliceWidth)) {
                selectPadTo = len;
                return sel;
            }
        }
        if (intKey) {
            return nil; // integer sorts outside the selection gate keep the exact sort path
        }
        if (descending) {
            k = [ctx.graph bitwiseNOTWithTensor:k name:nil];
        }
        return ArgSortKey(ctx.graph, k, axis, false);
    };

    if (ctx.op->getNumOperands() == 1 && ctx.op->getNumResults() == 1) {
        MPSGraphTensor* input = GetInputTensor(ctx, 0);
        if (!input) {
            return ProcessResult::Error("stablehlo.sort input tensor not found");
        }
        MPSGraphTensor* rawPerm = totalOrderPerm(input);
        MPSGraphTensor* sorted = rawPerm ? ExactGatherAlongAxis(ctx.graph, axis, input, rawPerm)
                                         : ExactSortValues(ctx.graph, input, axis, descending);
        if (sorted && selectPadTo > 0) {
            sorted = PadAlongAxisExact(ctx.graph, sorted, axis, selectPadTo);
        }
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
            MPSGraphTensor* sortedKey = nil;
            if ((perm = totalOrderPerm(keyTensor))) {
                sortedKey = ExactGatherAlongAxis(ctx.graph, axis, keyTensor, perm);
                if (sortedKey && selectPadTo > 0) {
                    sortedKey = PadAlongAxisExact(ctx.graph, sortedKey, axis, selectPadTo);
                }
            } else {
                sortedKey = ExactSortValues(ctx.graph, keyTensor, axis, descending);
                perm = ArgSortKey(ctx.graph, keyTensor, axis, descending);
            }
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
                if (sorted && selectPadTo > 0) {
                    sorted = PadAlongAxisExact(ctx.graph, sorted, axis, selectPadTo);
                }
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

// XLA's TopK comparator (is_stable = true) is IEEE totalOrder on the raw bits:
// key = bits < 0 ? bits ^ 0x7FF..F : bits, compared as signed integers, so
// -NaN < -inf < ... < -0 < +0 < subnormals < ... < +inf < +NaN with no NaN/-0
// canonicalisation and no denormal flush; f16/bf16 compare their 16-bit
// patterns. MPSGraph topK compares floats (-0 == +0, NaN unordered) and is not
// stable on ties (75/247 float probe cases differed from XLA:CPU, plain ties
// included); int32/uint32 topK compared through float32. Returns an int32 key
// whose ascending order is XLA's, or nil for dtypes that keep MPSGraph topK.
// JAX_METALLIB_TOPK_LEGACY_FLOAT=1 sends floats to MPSGraph topK again.
static MPSGraphTensor* TopKOrderKey(MPSGraph* graph, MPSGraphTensor* input) {
    const MPSDataType dt = input.dataType;
    if (dt == MPSDataTypeInt32) {
        return input;
    }
    if (dt == MPSDataTypeUInt32) {
        MPSGraphTensor* bits = [graph reinterpretCastTensor:input toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* signBit = [graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
        return [graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:signBit name:nil];
    }
    if (!IsRealFloatDtype(dt) || EnvFlagEnabledSort("JAX_METALLIB_TOPK_LEGACY_FLOAT")) {
        return nil;
    }
    MPSGraphTensor* bits = nil;
    MPSGraphTensor* magMask = nil;
    if (dt == MPSDataTypeFloat32) {
        bits = [graph reinterpretCastTensor:input toType:MPSDataTypeInt32 name:nil];
        magMask = [graph constantWithScalar:2147483647.0 dataType:MPSDataTypeInt32];
    } else {
        // Sign-extending int16 -> int32 keeps the 16-bit total order.
        MPSGraphTensor* raw = [graph reinterpretCastTensor:input toType:MPSDataTypeInt16 name:nil];
        bits = [graph castTensor:raw toType:MPSDataTypeInt32 name:nil];
        magMask = [graph constantWithScalar:32767.0 dataType:MPSDataTypeInt32];
    }
    if (!bits) {
        return nil;
    }
    MPSGraphTensor* zero = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg = [graph lessThanWithPrimaryTensor:bits secondaryTensor:zero name:nil];
    MPSGraphTensor* flipped = [graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:magMask name:nil];
    return [graph selectWithPredicateTensor:neg
                        truePredicateTensor:flipped
                       falsePredicateTensor:bits
                                       name:nil];
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
    if (MPSGraphTensor* key = TopKOrderKey(ctx.graph, input)) {
        // Bitwise NOT so an ascending stable argsort puts the largest first and
        // equal keys keep input order; keep the first k and gather the values
        // exactly (raw bits for f16, XLA's sign|0x7fc0 NaN for bf16).
        key = [ctx.graph bitwiseNOTWithTensor:key name:nil];
        MPSGraphTensor* order = ArgSortKey(ctx.graph, key, axis, false);
        if (!order) {
            return ProcessResult::Error("top_k argSort lowering failed");
        }
        order = [ctx.graph sliceTensor:order dimension:axis start:0 length:(NSInteger)k name:nil];
        valueOut = ExactGatherAlongAxis(ctx.graph, axis, input, order);
        indexOut = order;
        if (!valueOut || !indexOut) {
            return ProcessResult::Error("top_k exact gather lowering failed");
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
    // Small k reuses the lax.top_k selection (k max-reductions over an int32
    // order key) instead of a full argsort. OrderedInt32SortKey is the key the
    // composite argsort already uses (NaN largest, -0 == +0 == denormals), ~key
    // reverses it for min, and TopKSelectPerm takes the first index on ties like
    // the stable argsort. JAX_METALLIB_APPROX_TOPK_SELECT=0 keeps the argsort.
    MPSGraphTensor* perm = nil;
    const char* selectEnv = std::getenv("JAX_METALLIB_APPROX_TOPK_SELECT");
    const bool selectOff = selectEnv && selectEnv[0] == '0' && selectEnv[1] == '\0';
    if (!selectOff && k > 0 && k < n) {
        int64_t total = 1;
        for (NSNumber* d in inShape) {
            total *= std::max<int64_t>(0, d.longLongValue);
        }
        // 16-bit inputs fall back to a 16-bit argsort, about half the cost of the
        // 32-bit one, so selection wins only for smaller k (f16/bf16 1M elements:
        // L <= 256 k <= 8, L <= 1024 k <= 4, longer rows k = 1). An explicit
        // JAX_METALLIB_TOPK_SELECT_MAX_K still forces its own cap.
        int64_t cap = TopKSelectMaxK(n, total);
        const char* forcedCap = std::getenv("JAX_METALLIB_TOPK_SELECT_MAX_K");
        const bool half = input.dataType == MPSDataTypeFloat16 || input.dataType == MPSDataTypeBFloat16;
        if (half && !(forcedCap && forcedCap[0] != '\0')) {
            cap = std::min<int64_t>(cap, n <= 256 ? 8 : (n <= 1024 ? 4 : 1));
        }
        if (k <= cap) {
            if (MPSGraphTensor* key = OrderedInt32SortKey(g, input)) {
                if (dir < 0) {
                    key = [g bitwiseNOTWithTensor:key name:nil];
                }
                perm = TopKSelectPerm(g, key, (NSInteger)axis, k);
            }
        }
    }
    if (!perm) {
        perm = ArgSortKey(g, input, (NSInteger)axis, dir > 0);
        if (!perm) {
            return ProcessResult::Error("ApproxTopK: argsort lowering failed");
        }
        if (k < n) {
            perm = [g sliceTensor:perm dimension:(NSInteger)axis start:0 length:(NSInteger)k name:nil];
        }
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
