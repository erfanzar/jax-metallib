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

namespace jax_metallib {

static bool isBooleanResult(mlir::Operation* op) {
    if (op->getNumResults() == 0)
        return false;
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return false;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return intType.getWidth() == 1;
    }
    return false;
}

#define REGISTER_LOGICAL_BITWISE_OP(mlir_op_name, logical_method, bitwise_method, reg_suffix)                \
    static ProcessResult Handle##reg_suffix(HandlerContext& ctx) {                                           \
        MPSGraphTensor* lhs = GetInputTensor(ctx, 0);                                                        \
        MPSGraphTensor* rhs = GetInputTensor(ctx, 1);                                                        \
        if (!lhs || !rhs)                                                                                    \
            return ProcessResult::Error(#reg_suffix ": missing input tensor");                               \
        MPSGraphTensor* result = nil;                                                                        \
        if (isBooleanResult(ctx.op)) {                                                                       \
            result = [ctx.graph logical_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];         \
        } else {                                                                                             \
            result = [ctx.graph bitwise_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];         \
        }                                                                                                    \
        return Result(ctx, result, #reg_suffix);                                                             \
    }                                                                                                        \
    REGISTER_MPS_OP(mlir_op_name, Handle##reg_suffix)

REGISTER_LOGICAL_BITWISE_OP("stablehlo.and", logicalAND, bitwiseAND, And);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.or", logicalOR, bitwiseOR, Or);
REGISTER_LOGICAL_BITWISE_OP("stablehlo.xor", logicalXOR, bitwiseXOR, Xor);

static ProcessResult HandleNot(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("not: missing input tensor");

    MPSGraphTensor* result = nil;
    if (isBooleanResult(ctx.op)) {
        MPSGraphTensor* falseTensor = [ctx.graph constantWithScalar:0 dataType:input.dataType];
        result = [ctx.graph equalWithPrimaryTensor:input secondaryTensor:falseTensor name:nil];
    } else {
        result = [ctx.graph bitwiseNOTWithTensor:input name:nil];
    }

    return Result(ctx, result, "not");
}
REGISTER_MPS_OP("stablehlo.not", HandleNot);

static int getBitWidth(mlir::Operation* op) {
    auto resultType = op->getResult(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType)
        return 0;
    auto elemType = tensorType.getElementType();
    if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
        return static_cast<int>(intType.getWidth());
    }
    return 0;
}

static MPSDataType toUnsignedIntegerDataType(MPSDataType dataType) {
    switch (dataType) {
    case MPSDataTypeInt8:
        return MPSDataTypeUInt8;
    case MPSDataTypeInt16:
        return MPSDataTypeUInt16;
    case MPSDataTypeInt32:
        return MPSDataTypeUInt32;
    case MPSDataTypeInt64:
        return MPSDataTypeUInt64;
    case MPSDataTypeUInt8:
    case MPSDataTypeUInt16:
    case MPSDataTypeUInt32:
    case MPSDataTypeUInt64:
        return dataType;
    default:
        return MPSDataTypeInvalid;
    }
}

static MPSGraphTensor* BuildShiftOverflowMask(MPSGraph* g, MPSGraphTensor* shiftAmount, int bitWidth) {
    MPSDataType unsignedShiftType = toUnsignedIntegerDataType(shiftAmount.dataType);
    MPSGraphTensor* shiftForCompare = shiftAmount;
    if (unsignedShiftType != MPSDataTypeInvalid && unsignedShiftType != shiftAmount.dataType) {
        shiftForCompare = [g castTensor:shiftAmount toType:unsignedShiftType name:nil];
    }

    MPSGraphTensor* bitWidthTensor = [g constantWithScalar:bitWidth
                                                     shape:@[@1]
                                                  dataType:shiftForCompare.dataType];
    return [g greaterThanOrEqualToWithPrimaryTensor:shiftForCompare secondaryTensor:bitWidthTensor name:nil];
}

enum class ShiftMode {
    kLeft,
    kRightLogical,
    kRightArithmetic,
};

static MPSGraphTensor* BuildShiftOverflowValue(MPSGraph* g, MPSGraphTensor* input, ShiftMode mode) {
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];
    if (mode != ShiftMode::kRightArithmetic) {
        return zeroTensor;
    }

    MPSGraphTensor* minusOneTensor = [g constantWithScalar:-1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* isNegative = [g lessThanWithPrimaryTensor:input secondaryTensor:zeroTensor name:nil];
    return [g selectWithPredicateTensor:isNegative
                    truePredicateTensor:minusOneTensor
                   falsePredicateTensor:zeroTensor
                                   name:nil];
}

static MPSGraphTensor* HandleShiftOp(HandlerContext& ctx, ShiftMode mode) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(ctx, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(ctx.op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraphTensor* shiftedResult =
        mode == ShiftMode::kLeft
            ? [ctx.graph bitwiseLeftShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil]
            : [ctx.graph bitwiseRightShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil];

    MPSGraphTensor* overflowMask = BuildShiftOverflowMask(ctx.graph, shiftAmount, bitWidth);
    MPSGraphTensor* overflowValue = BuildShiftOverflowValue(ctx.graph, input, mode);

    return [ctx.graph selectWithPredicateTensor:overflowMask
                            truePredicateTensor:overflowValue
                           falsePredicateTensor:shiftedResult
                                           name:nil];
}

static ProcessResult HandleShiftLeft(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kLeft);
    return Result(ctx, result, "shift_left");
}
REGISTER_MPS_OP("stablehlo.shift_left", HandleShiftLeft);

static ProcessResult HandleShiftRightLogical(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightLogical);
    return Result(ctx, result, "shift_right_logical");
}
REGISTER_MPS_OP("stablehlo.shift_right_logical", HandleShiftRightLogical);

static ProcessResult HandleShiftRightArithmetic(HandlerContext& ctx) {
    MPSGraphTensor* result = HandleShiftOp(ctx, ShiftMode::kRightArithmetic);
    return Result(ctx, result, "shift_right_arithmetic");
}
REGISTER_MPS_OP("stablehlo.shift_right_arithmetic", HandleShiftRightArithmetic);

static ProcessResult HandlePopcnt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("popcnt: missing input tensor");

    if (ctx.op->getNumOperands() == 0) {
        return ProcessResult::Error("popcnt: requires one integer operand");
    }
    auto operandType = ctx.op->getOperand(0).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operandType);
    if (!tensorType || !mlir::isa<mlir::IntegerType>(tensorType.getElementType())) {
        return ProcessResult::Error("popcnt: requires an integer operand");
    }

    MPSGraphTensor* count = [ctx.graph bitwisePopulationCountWithTensor:input name:nil];
    if (!count)
        return ProcessResult::Error("popcnt: bitwisePopulationCount returned null");

    MPSDataType outType = GetResultMpsType(ctx.op);
    MPSGraphTensor* result = count;
    if (outType != MPSDataTypeInvalid && count.dataType != outType) {
        result = [ctx.graph castTensor:count toType:outType name:nil];
    }

    return Result(ctx, result, "popcnt");
}
REGISTER_MPS_OP("stablehlo.popcnt", HandlePopcnt);

} // namespace jax_metallib
