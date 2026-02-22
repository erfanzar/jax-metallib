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

#include <limits>
#include <sstream>

namespace jax_metallib {

REGISTER_MLIR_BINARY_OP("stablehlo.add", addition, add);
REGISTER_MLIR_BINARY_OP("stablehlo.subtract", subtraction, subtract);
REGISTER_MLIR_BINARY_OP("stablehlo.multiply", multiplication, multiply);
REGISTER_MLIR_BINARY_OP("stablehlo.divide", division, divide);
REGISTER_MLIR_BINARY_OP("stablehlo.maximum", maximum, maximum);
REGISTER_MLIR_BINARY_OP("stablehlo.minimum", minimum, minimum);
REGISTER_MLIR_BINARY_OP("stablehlo.remainder", modulo, remainder);
REGISTER_MLIR_BINARY_OP("stablehlo.power", power, power);
REGISTER_MLIR_BINARY_OP("stablehlo.atan2", atan2, atan2);

static ProcessResult HandleDot(HandlerContext& ctx) {
    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot: missing input tensor");
    MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                              secondaryTensor:rhs
                                                                         name:nil];
    return Result(ctx, result, "dot");
}
REGISTER_MPS_OP("stablehlo.dot", HandleDot);

namespace {

static bool IsIdentityPermutation(NSArray<NSNumber*>* permutation) {
    for (NSUInteger i = 0; i < permutation.count; ++i) {
        if ([permutation[i] integerValue] != (NSInteger)i) {
            return false;
        }
    }
    return true;
}

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

enum class ProductDimsStatus { kOk, kZero, kDynamic, kOverflow };

static ProductDimsStatus ProductDimsDetailed(const std::vector<int64_t>& dims, int64_t* out) {
    int64_t prod = 1;
    for (int64_t d : dims) {
        if (d == 0) {
            return ProductDimsStatus::kZero;
        }
        if (d < 0) {
            return ProductDimsStatus::kDynamic;
        }
        if (prod > (std::numeric_limits<int64_t>::max() / d)) {
            return ProductDimsStatus::kOverflow;
        }
        prod *= d;
    }
    *out = prod;
    return ProductDimsStatus::kOk;
}

static void ShapeHasZeroOrDynamic(NSArray<NSNumber*>* shape, bool* hasZero, bool* hasDynamic) {
    if (!shape) {
        return;
    }
    for (NSNumber* dim : shape) {
        int64_t d = [dim longLongValue];
        if (d == 0 && hasZero) {
            *hasZero = true;
        }
        if (d < 0 && hasDynamic) {
            *hasDynamic = true;
        }
    }
}

static std::string DotGeneralDebugString(mlir::stablehlo::DotGeneralOp dotOp, NSArray<NSNumber*>* lhsShape,
                                         NSArray<NSNumber*>* rhsShape, NSArray<NSNumber*>* outShape) {
    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();
    std::ostringstream oss;
    oss << "lhs_shape=" << ShapeToString(lhsShape) << ", rhs_shape=" << ShapeToString(rhsShape)
        << ", out_shape=" << ShapeToString(outShape) << ", lhs_batch=" << JoinInts(lhsBatchDims)
        << ", rhs_batch=" << JoinInts(rhsBatchDims) << ", lhs_contract=" << JoinInts(lhsContractingDims)
        << ", rhs_contract=" << JoinInts(rhsContractingDims);
    return oss.str();
}

static std::vector<int64_t> ComplementDims(NSUInteger rank, const std::vector<bool>& used) {
    std::vector<int64_t> dims;
    for (NSUInteger d = 0; d < rank; ++d) {
        if (!used[d]) {
            dims.push_back((int64_t)d);
        }
    }
    return dims;
}

static std::vector<int64_t> GetDimSizes(NSArray<NSNumber*>* shape, llvm::ArrayRef<int64_t> dims) {
    std::vector<int64_t> sizes;
    sizes.reserve(dims.size());
    for (int64_t d : dims) {
        sizes.push_back([shape[(NSUInteger)d] longLongValue]);
    }
    return sizes;
}

static NSArray<NSNumber*>* ToNSArray(const std::vector<int64_t>& values) {
    NSMutableArray<NSNumber*>* out = [NSMutableArray arrayWithCapacity:values.size()];
    for (int64_t v : values) {
        [out addObject:@(v)];
    }
    return out;
}

} // namespace

static ProcessResult HandleDotGeneral(HandlerContext& ctx) {
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(ctx.op);
    if (!dotOp) {
        return ProcessResult::Error("dot_general: expected DotGeneralOp");
    }

    MPSGraphTensor* lhsInput = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhsInput = GetInputTensor(ctx, 1);
    if (!lhsInput || !rhsInput)
        return ProcessResult::Error("dot_general: missing input tensor");

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    NSArray<NSNumber*>* lhsShape = lhsInput.shape;
    NSArray<NSNumber*>* rhsShape = rhsInput.shape;
    NSUInteger lhsRank = lhsShape.count;
    NSUInteger rhsRank = rhsShape.count;

    NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
    bool hasZero = false;
    bool hasDynamic = false;
    ShapeHasZeroOrDynamic(lhsShape, &hasZero, &hasDynamic);
    ShapeHasZeroOrDynamic(rhsShape, &hasZero, &hasDynamic);
    ShapeHasZeroOrDynamic(outShape, &hasZero, &hasDynamic);
    if (hasDynamic) {
        return ProcessResult::Error("dot_general: dynamic dimensions are not supported (" +
                                    DotGeneralDebugString(dotOp, lhsShape, rhsShape, outShape) + ")");
    }
    if (hasZero) {
        return ProcessResult::Error("dot_general: zero-sized dimensions are not supported yet (" +
                                    DotGeneralDebugString(dotOp, lhsShape, rhsShape, outShape) + ")");
    }

    if (lhsBatchDims.size() != rhsBatchDims.size()) {
        return ProcessResult::Error("dot_general: lhs/rhs batching dims size mismatch (lhs=" +
                                    JoinInts(lhsBatchDims) + ", rhs=" + JoinInts(rhsBatchDims) + ")");
    }
    if (lhsContractingDims.size() != rhsContractingDims.size()) {
        return ProcessResult::Error(
            "dot_general: lhs/rhs contracting dims size mismatch (lhs=" + JoinInts(lhsContractingDims) +
            ", rhs=" + JoinInts(rhsContractingDims) + ")");
    }

    auto validateDims = [](llvm::ArrayRef<int64_t> dims, NSUInteger rank, const char* kind,
                           std::vector<bool>& seen, std::string& error) -> bool {
        for (int64_t d : dims) {
            if (d < 0 || d >= (int64_t)rank) {
                error = std::string("dot_general: ") + kind + " dimension out of range: " + std::to_string(d);
                return false;
            }
            if (seen[(NSUInteger)d]) {
                error = std::string("dot_general: duplicate dimension in ") + kind + ": " + std::to_string(d);
                return false;
            }
            seen[(NSUInteger)d] = true;
        }
        return true;
    };

    std::vector<bool> lhsUsed(lhsRank, false);
    std::vector<bool> rhsUsed(rhsRank, false);
    std::string dimError;
    if (!validateDims(lhsBatchDims, lhsRank, "lhs_batching", lhsUsed, dimError) ||
        !validateDims(lhsContractingDims, lhsRank, "lhs_contracting", lhsUsed, dimError) ||
        !validateDims(rhsBatchDims, rhsRank, "rhs_batching", rhsUsed, dimError) ||
        !validateDims(rhsContractingDims, rhsRank, "rhs_contracting", rhsUsed, dimError)) {
        return ProcessResult::Error(dimError);
    }

    for (size_t i = 0; i < lhsBatchDims.size(); ++i) {
        int64_t lhsDimSize = [lhsShape[(NSUInteger)lhsBatchDims[i]] longLongValue];
        int64_t rhsDimSize = [rhsShape[(NSUInteger)rhsBatchDims[i]] longLongValue];
        if (lhsDimSize != rhsDimSize) {
            return ProcessResult::Error(
                "dot_general: incompatible batch dimensions (lhs_shape=" + ShapeToString(lhsShape) +
                ", rhs_shape=" + ShapeToString(rhsShape) + ", lhs_batch=" + JoinInts(lhsBatchDims) +
                ", rhs_batch=" + JoinInts(rhsBatchDims) + ")");
        }
    }
    for (size_t i = 0; i < lhsContractingDims.size(); ++i) {
        int64_t lhsDimSize = [lhsShape[(NSUInteger)lhsContractingDims[i]] longLongValue];
        int64_t rhsDimSize = [rhsShape[(NSUInteger)rhsContractingDims[i]] longLongValue];
        if (lhsDimSize != rhsDimSize) {
            return ProcessResult::Error("dot_general: incompatible contracting dimensions (lhs_shape=" +
                                        ShapeToString(lhsShape) + ", rhs_shape=" + ShapeToString(rhsShape) +
                                        ", lhs_contracting=" + JoinInts(lhsContractingDims) +
                                        ", rhs_contracting=" + JoinInts(rhsContractingDims) + ")");
        }
    }

    if (lhsBatchDims.empty() && rhsBatchDims.empty() && lhsContractingDims.size() == 1 &&
        rhsContractingDims.size() == 1) {
        int64_t lc = lhsContractingDims[0];
        int64_t rc = rhsContractingDims[0];

        if (lhsRank == 2 && rhsRank == 2 && lc == 1 && rc == 0) {
            MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsInput
                                                                      secondaryTensor:rhsInput
                                                                                 name:nil];
            if (outShape && result && ![outShape isEqualToArray:result.shape]) {
                result = [ctx.graph reshapeTensor:result withShape:outShape name:nil];
            }
            return Result(ctx, result, "dot_general");
        }

        if (lhsRank == 1 && rhsRank == 2 && lc == 0 && rc == 0) {
            MPSGraphTensor* lhs2d = [ctx.graph reshapeTensor:lhsInput withShape:@[@1, lhsShape[0]] name:nil];
            MPSGraphTensor* result2d = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs2d
                                                                        secondaryTensor:rhsInput
                                                                                   name:nil];
            MPSGraphTensor* result = result2d;
            if (outShape && result && ![outShape isEqualToArray:result.shape]) {
                result = [ctx.graph reshapeTensor:result2d withShape:outShape name:nil];
            }
            return Result(ctx, result, "dot_general");
        }

        if (lhsRank == 2 && rhsRank == 1 && lc == 1 && rc == 0) {
            MPSGraphTensor* rhs2d = [ctx.graph reshapeTensor:rhsInput withShape:@[rhsShape[0], @1] name:nil];
            MPSGraphTensor* result2d = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsInput
                                                                        secondaryTensor:rhs2d
                                                                                   name:nil];
            MPSGraphTensor* result = result2d;
            if (outShape && result && ![outShape isEqualToArray:result.shape]) {
                result = [ctx.graph reshapeTensor:result2d withShape:outShape name:nil];
            }
            return Result(ctx, result, "dot_general");
        }
    }

    if (lhsBatchDims.size() == 1 && rhsBatchDims.size() == 1 && lhsContractingDims.size() == 1 &&
        rhsContractingDims.size() == 1) {
        int64_t lb = lhsBatchDims[0];
        int64_t rb = rhsBatchDims[0];
        int64_t lc = lhsContractingDims[0];
        int64_t rc = rhsContractingDims[0];

        if (lhsRank == 3 && rhsRank == 3 && lb == 0 && rb == 0 && lc == 2 && rc == 1) {
            MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsInput
                                                                      secondaryTensor:rhsInput
                                                                                 name:nil];
            if (outShape && result && ![outShape isEqualToArray:result.shape]) {
                result = [ctx.graph reshapeTensor:result withShape:outShape name:nil];
            }
            return Result(ctx, result, "dot_general");
        }

        if (lhsRank == 2 && rhsRank == 3 && lb == 0 && rb == 0 && lc == 1 && rc == 1) {
            MPSGraphTensor* lhs3d = [ctx.graph reshapeTensor:lhsInput
                                                   withShape:@[lhsShape[0], @1, lhsShape[1]]
                                                        name:nil];
            MPSGraphTensor* result3d = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs3d
                                                                        secondaryTensor:rhsInput
                                                                                   name:nil];
            if (outShape && result3d && ![outShape isEqualToArray:result3d.shape]) {
                result3d = [ctx.graph reshapeTensor:result3d withShape:outShape name:nil];
            }
            return Result(ctx, result3d, "dot_general");
        }

        if (lhsRank == 3 && rhsRank == 2 && lb == 0 && rb == 0 && lc == 2 && rc == 1) {
            MPSGraphTensor* rhs3d = [ctx.graph reshapeTensor:rhsInput
                                                   withShape:@[rhsShape[0], rhsShape[1], @1]
                                                        name:nil];
            MPSGraphTensor* result3d = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhsInput
                                                                        secondaryTensor:rhs3d
                                                                                   name:nil];
            if (outShape && result3d && ![outShape isEqualToArray:result3d.shape]) {
                result3d = [ctx.graph reshapeTensor:result3d withShape:outShape name:nil];
            }
            return Result(ctx, result3d, "dot_general");
        }
    }

    std::vector<int64_t> lhsBatch(lhsBatchDims.begin(), lhsBatchDims.end());
    std::vector<int64_t> rhsBatch(rhsBatchDims.begin(), rhsBatchDims.end());
    std::vector<int64_t> lhsContract(lhsContractingDims.begin(), lhsContractingDims.end());
    std::vector<int64_t> rhsContract(rhsContractingDims.begin(), rhsContractingDims.end());
    std::vector<int64_t> lhsFree = ComplementDims(lhsRank, lhsUsed);
    std::vector<int64_t> rhsFree = ComplementDims(rhsRank, rhsUsed);

    std::vector<int64_t> lhsPerm = lhsBatch;
    lhsPerm.insert(lhsPerm.end(), lhsFree.begin(), lhsFree.end());
    lhsPerm.insert(lhsPerm.end(), lhsContract.begin(), lhsContract.end());
    std::vector<int64_t> rhsPerm = rhsBatch;
    rhsPerm.insert(rhsPerm.end(), rhsContract.begin(), rhsContract.end());
    rhsPerm.insert(rhsPerm.end(), rhsFree.begin(), rhsFree.end());

    MPSGraphTensor* lhs = lhsInput;
    NSArray<NSNumber*>* lhsPermArray = ToNSArray(lhsPerm);
    if (!IsIdentityPermutation(lhsPermArray)) {
        lhs = [ctx.graph transposeTensor:lhsInput permutation:lhsPermArray name:nil];
    }

    MPSGraphTensor* rhs = rhsInput;
    NSArray<NSNumber*>* rhsPermArray = ToNSArray(rhsPerm);
    if (!IsIdentityPermutation(rhsPermArray)) {
        rhs = [ctx.graph transposeTensor:rhsInput permutation:rhsPermArray name:nil];
    }

    std::vector<int64_t> batchSizes = GetDimSizes(lhsShape, lhsBatchDims);
    std::vector<int64_t> lhsFreeSizes = GetDimSizes(lhsShape, lhsFree);
    std::vector<int64_t> rhsFreeSizes = GetDimSizes(rhsShape, rhsFree);
    std::vector<int64_t> contractingSizes = GetDimSizes(lhsShape, lhsContractingDims);

    int64_t m = 1;
    int64_t k = 1;
    int64_t n = 1;
    int64_t batchCheck = 1;
    ProductDimsStatus batchStatus = ProductDimsDetailed(batchSizes, &batchCheck);
    ProductDimsStatus mStatus = ProductDimsDetailed(lhsFreeSizes, &m);
    ProductDimsStatus kStatus = ProductDimsDetailed(contractingSizes, &k);
    ProductDimsStatus nStatus = ProductDimsDetailed(rhsFreeSizes, &n);
    if (batchStatus != ProductDimsStatus::kOk || mStatus != ProductDimsStatus::kOk ||
        kStatus != ProductDimsStatus::kOk || nStatus != ProductDimsStatus::kOk) {
        if (batchStatus == ProductDimsStatus::kDynamic || mStatus == ProductDimsStatus::kDynamic ||
            kStatus == ProductDimsStatus::kDynamic || nStatus == ProductDimsStatus::kDynamic) {
            return ProcessResult::Error("dot_general: dynamic dimensions are not supported (" +
                                        DotGeneralDebugString(dotOp, lhsShape, rhsShape, outShape) + ")");
        }
        if (batchStatus == ProductDimsStatus::kZero || mStatus == ProductDimsStatus::kZero ||
            kStatus == ProductDimsStatus::kZero || nStatus == ProductDimsStatus::kZero) {
            return ProcessResult::Error("dot_general: zero-sized dimensions are not supported yet (" +
                                        DotGeneralDebugString(dotOp, lhsShape, rhsShape, outShape) + ")");
        }

        std::ostringstream oss;
        oss << "dot_general: dimension product overflow (batch_dims=" << JoinInts(lhsBatchDims)
            << ", lhs_free_dims=" << JoinInts(lhsFree) << ", rhs_free_dims=" << JoinInts(rhsFree)
            << ", contracting_dims=" << JoinInts(lhsContractingDims)
            << ", lhs_shape=" << ShapeToString(lhsShape) << ", rhs_shape=" << ShapeToString(rhsShape) << ")";
        return ProcessResult::Error(oss.str());
    }

    // MPSGraph matrixMultiplication only supports rank 2 (M×K * K×N) or rank 3
    // (batch×M×K * batch×K×N).  When dot_general has multiple batch dimensions
    // (e.g. attention einsum "btnh,bsnh->bnts" with batch dims [0,2]) we must
    // flatten them into a single batch dimension before the matmul and reshape
    // the result back afterwards.
    std::vector<int64_t> lhsMatShape;
    std::vector<int64_t> rhsMatShape;
    if (batchSizes.empty()) {
        // No batch dims → rank-2 matmul [M, K] * [K, N]
        lhsMatShape = {m, k};
        rhsMatShape = {k, n};
    } else if (batchSizes.size() == 1) {
        // Single batch dim → rank-3 matmul [B, M, K] * [B, K, N]
        lhsMatShape = {batchSizes[0], m, k};
        rhsMatShape = {batchSizes[0], k, n};
    } else {
        // Multiple batch dims → flatten to single batch [B1*B2*..., M, K]
        lhsMatShape = {batchCheck, m, k};
        rhsMatShape = {batchCheck, k, n};
    }

    NSArray<NSNumber*>* lhsMatShapeArray = ToNSArray(lhsMatShape);
    if (lhsMatShapeArray && lhs.shape && ![lhsMatShapeArray isEqualToArray:lhs.shape]) {
        lhs = [ctx.graph reshapeTensor:lhs withShape:lhsMatShapeArray name:nil];
    }
    NSArray<NSNumber*>* rhsMatShapeArray = ToNSArray(rhsMatShape);
    if (rhsMatShapeArray && rhs.shape && ![rhsMatShapeArray isEqualToArray:rhs.shape]) {
        rhs = [ctx.graph reshapeTensor:rhs withShape:rhsMatShapeArray name:nil];
    }

    MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                              secondaryTensor:rhs
                                                                         name:nil];

    std::vector<int64_t> logicalOutputShape = batchSizes;
    logicalOutputShape.insert(logicalOutputShape.end(), lhsFreeSizes.begin(), lhsFreeSizes.end());
    logicalOutputShape.insert(logicalOutputShape.end(), rhsFreeSizes.begin(), rhsFreeSizes.end());

    NSArray<NSNumber*>* expectedOutputShape = GetOutputShape(ctx.op);
    NSArray<NSNumber*>* finalOutShape =
        expectedOutputShape ? expectedOutputShape : (NSArray<NSNumber*>*)ToNSArray(logicalOutputShape);
    if (finalOutShape && result && ![finalOutShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:finalOutShape name:nil];
    }

    return Result(ctx, result, "dot_general");
}
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

static ProcessResult HandleCompare(HandlerContext& ctx) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(ctx.op);
    if (!compareOp) {
        return ProcessResult::Error("compare: expected CompareOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("compare: missing input tensor");

    auto direction = compareOp.getComparisonDirection();
    using Dir = mlir::stablehlo::ComparisonDirection;

    MPSGraphTensor* result = nil;
    switch (direction) {
    case Dir::LT:
        result = [ctx.graph lessThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    case Dir::LE:
        result = [ctx.graph lessThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    case Dir::GT:
        result = [ctx.graph greaterThanWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    case Dir::GE:
        result = [ctx.graph greaterThanOrEqualToWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    case Dir::EQ:
        result = [ctx.graph equalWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    case Dir::NE:
        result = [ctx.graph notEqualWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
        break;
    default:
        return ProcessResult::Error("compare: unknown compare direction");
    }

    return Result(ctx, result, "compare");
}
REGISTER_MPS_OP("stablehlo.compare", HandleCompare);

static ProcessResult HandleSelect(HandlerContext& ctx) {
    MPSGraphTensor* pred = GetInputTensor(ctx, 0);
    MPSGraphTensor* onTrue = GetInputTensor(ctx, 1);
    MPSGraphTensor* onFalse = GetInputTensor(ctx, 2);
    if (!pred || !onTrue || !onFalse)
        return ProcessResult::Error("select: missing input tensor");

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:pred
                                              truePredicateTensor:onTrue
                                             falsePredicateTensor:onFalse
                                                             name:nil];
    return Result(ctx, result, "select");
}
REGISTER_MPS_OP("stablehlo.select", HandleSelect);

static ProcessResult HandleClamp(HandlerContext& ctx) {
    MPSGraphTensor* minVal = GetInputTensor(ctx, 0);
    MPSGraphTensor* operand = GetInputTensor(ctx, 1);
    MPSGraphTensor* maxVal = GetInputTensor(ctx, 2);
    if (!minVal || !operand || !maxVal)
        return ProcessResult::Error("clamp: missing input tensor");

    MPSGraphTensor* result = [ctx.graph clampWithTensor:operand
                                         minValueTensor:minVal
                                         maxValueTensor:maxVal
                                                   name:nil];
    return Result(ctx, result, "clamp");
}
REGISTER_MPS_OP("stablehlo.clamp", HandleClamp);

static ProcessResult HandleNextAfter(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    MPSGraphTensor* y = GetInputTensor(ctx, 1);
    if (!x || !y)
        return ProcessResult::Error("next_after: missing input tensor");

    MPSDataType dtype = x.dataType;

    NSArray<NSNumber*>* xShape = x.shape;
    bool isScalar = (xShape.count == 0);
    if (isScalar) {
        x = [ctx.graph reshapeTensor:x withShape:@[@1] name:nil];
        y = [ctx.graph reshapeTensor:y withShape:@[@1] name:nil];
    }

    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:dtype];
    MPSGraphTensor* one_int = [ctx.graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg_one_int = [ctx.graph constantWithScalar:-1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_positive_int = [ctx.graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* min_negative_int = [ctx.graph constantWithScalar:0x80000001 dataType:MPSDataTypeInt32];

    MPSGraphTensor* x_as_int = [ctx.graph reinterpretCastTensor:x toType:MPSDataTypeInt32 name:nil];

    MPSGraphTensor* x_eq_y = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:y name:nil];

    MPSGraphTensor* x_is_zero = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:zero name:nil];

    MPSGraphTensor* y_gt_zero = [ctx.graph greaterThanWithPrimaryTensor:y secondaryTensor:zero name:nil];

    MPSGraphTensor* zero_result_int = [ctx.graph selectWithPredicateTensor:y_gt_zero
                                                       truePredicateTensor:min_positive_int
                                                      falsePredicateTensor:min_negative_int
                                                                      name:nil];
    MPSGraphTensor* zero_result = [ctx.graph reinterpretCastTensor:zero_result_int toType:dtype name:nil];

    MPSGraphTensor* y_gt_x = [ctx.graph greaterThanWithPrimaryTensor:y secondaryTensor:x name:nil];

    MPSGraphTensor* x_gt_zero = [ctx.graph greaterThanWithPrimaryTensor:x secondaryTensor:zero name:nil];

    MPSGraphTensor* should_increment = [ctx.graph equalWithPrimaryTensor:x_gt_zero
                                                         secondaryTensor:y_gt_x
                                                                    name:nil];

    MPSGraphTensor* delta = [ctx.graph selectWithPredicateTensor:should_increment
                                             truePredicateTensor:one_int
                                            falsePredicateTensor:neg_one_int
                                                            name:nil];

    MPSGraphTensor* result_int = [ctx.graph additionWithPrimaryTensor:x_as_int
                                                      secondaryTensor:delta
                                                                 name:nil];

    MPSGraphTensor* non_zero_result = [ctx.graph reinterpretCastTensor:result_int toType:dtype name:nil];

    MPSGraphTensor* non_equal_result = [ctx.graph selectWithPredicateTensor:x_is_zero
                                                        truePredicateTensor:zero_result
                                                       falsePredicateTensor:non_zero_result
                                                                       name:nil];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:x_eq_y
                                              truePredicateTensor:y
                                             falsePredicateTensor:non_equal_result
                                                             name:nil];

    if (isScalar) {
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    return Result(ctx, result, "next_after");
}
REGISTER_MPS_OP("chlo.next_after", HandleNextAfter);

} // namespace jax_metallib
