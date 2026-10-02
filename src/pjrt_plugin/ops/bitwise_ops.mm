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

static bool IsUnsignedIntDtype(MPSDataType t) {
    return t == MPSDataTypeUInt8 || t == MPSDataTypeUInt16 || t == MPSDataTypeUInt32 ||
           t == MPSDataTypeUInt64;
}

static bool IsSignedIntDtype(MPSDataType t) {
    return t == MPSDataTypeInt8 || t == MPSDataTypeInt16 || t == MPSDataTypeInt32 || t == MPSDataTypeInt64;
}

// ~0 in the given dtype (-1 signed, max unsigned), built without large
// constant conversions.
static MPSGraphTensor* AllOnes(MPSGraph* g, MPSDataType t) {
    return [g bitwiseNOTWithTensor:[g constantWithScalar:0 dataType:t] name:nil];
}

// Bool tensor: the top (sign) bit of `input` is set. Works for either
// signedness, unlike `input < 0`.
static MPSGraphTensor* TopBitSet(MPSGraph* g, MPSGraphTensor* input, int bitWidth) {
    MPSDataType t = input.dataType;
    MPSGraphTensor* topBit = [g bitwiseLeftShiftWithPrimaryTensor:AllOnes(g, t)
                                                  secondaryTensor:[g constantWithScalar:bitWidth - 1
                                                                               dataType:t]
                                                             name:nil];
    MPSGraphTensor* bit = [g bitwiseANDWithPrimaryTensor:input secondaryTensor:topBit name:nil];
    return [g notEqualWithPrimaryTensor:bit secondaryTensor:[g constantWithScalar:0 dataType:t] name:nil];
}

static MPSGraphTensor* BuildShiftOverflowValue(MPSGraph* g, MPSGraphTensor* input, ShiftMode mode,
                                               int bitWidth) {
    MPSGraphTensor* zeroTensor = [g constantWithScalar:0 shape:@[@1] dataType:input.dataType];
    if (mode != ShiftMode::kRightArithmetic) {
        return zeroTensor;
    }

    MPSGraphTensor* minusOneTensor = [g constantWithScalar:-1 shape:@[@1] dataType:input.dataType];
    MPSGraphTensor* isNegative = IsUnsignedIntDtype(input.dataType) ? TopBitSet(g, input, bitWidth)
                                                                    : [g lessThanWithPrimaryTensor:input
                                                                                   secondaryTensor:zeroTensor
                                                                                              name:nil];
    if (IsUnsignedIntDtype(input.dataType)) {
        minusOneTensor = AllOnes(g, input.dataType);
    }
    return [g selectWithPredicateTensor:isNegative
                    truePredicateTensor:minusOneTensor
                   falsePredicateTensor:zeroTensor
                                   name:nil];
}

// x >> k with a constant k lowers to shift(x, broadcast_in_dim(constant k)).
// The overflow guard (k >= bitWidth) and its select over that full-size
// constant made MPSGraph fold an n-element compare and select one APInt at a
// time on the first run: 20 rotate rounds on 4M uint32 spent ~11 s before the
// first result, linear in n (every jax.random call hit this). When k is an
// in-range splat constant and the input already has the output shape, feed k
// as a [1,...,1] constant and skip the guard, which cannot fire.
// JAX_METALLIB_DISABLE_SHIFT_SPLAT_FOLD=1 restores the guarded form.
static bool ShiftSplatFoldEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_SHIFT_SPLAT_FOLD");
        return !(v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
    }();
    return enabled;
}

// Unsigned value of `v` when it is a splat integer constant, possibly behind
// broadcast_in_dim or reshape. Negative signed amounts come out >= bitWidth.
static bool SplatShiftAmount(mlir::Value v, uint64_t* amount) {
    for (int depth = 0; depth < 8 && v; ++depth) {
        mlir::Operation* def = v.getDefiningOp();
        if (!def) {
            return false;
        }
        if (auto c = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
            auto dense = mlir::dyn_cast<mlir::DenseIntElementsAttr>(c.getValue());
            if (!dense || !dense.isSplat()) {
                return false;
            }
            llvm::APInt k = dense.getSplatValue<llvm::APInt>();
            if (k.getActiveBits() > 64) {
                return false;
            }
            *amount = k.getZExtValue();
            return true;
        }
        if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
            v = bc.getOperand();
            continue;
        }
        if (auto rs = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
            v = rs.getOperand();
            continue;
        }
        return false;
    }
    return false;
}

static MPSGraphTensor* HandleShiftOp(HandlerContext& ctx, ShiftMode mode) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* shiftAmount = GetInputTensor(ctx, 1);
    if (!input || !shiftAmount)
        return nullptr;

    int bitWidth = getBitWidth(ctx.op);
    if (bitWidth == 0)
        return nullptr;

    MPSGraph* g = ctx.graph;
    bool inRangeSplat = false;
    uint64_t splatAmount = 0;
    if (ShiftSplatFoldEnabled() && SplatShiftAmount(ctx.op->getOperand(1), &splatAmount) &&
        splatAmount < static_cast<uint64_t>(bitWidth)) {
        NSArray<NSNumber*>* out = GetOutputShape(ctx.op);
        if (out && input.shape && [input.shape isEqualToArray:out]) {
            NSMutableArray<NSNumber*>* ones = [NSMutableArray arrayWithCapacity:out.count];
            for (NSUInteger i = 0; i < out.count; ++i) {
                [ones addObject:@1];
            }
            const double k = static_cast<double>(splatAmount);
            shiftAmount = out.count == 0 ? [g constantWithScalar:k dataType:shiftAmount.dataType]
                                         : [g constantWithScalar:k shape:ones dataType:shiftAmount.dataType];
            inRangeSplat = true;
        }
    }
    MPSGraphTensor* shiftedResult =
        mode == ShiftMode::kLeft
            ? [g bitwiseLeftShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil]
            : [g bitwiseRightShiftWithPrimaryTensor:input secondaryTensor:shiftAmount name:nil];

    // MPSGraph right shift follows the dtype's signedness, not the op: signed
    // ints always sign-extend and unsigned ints always zero-fill. Correct the
    // mismatched combinations with masks (no reinterpret casts, which abort on
    // rank-0 tensors). In-range amounts only; overflow is selected below.
    const MPSDataType dtype = input.dataType;
    if (mode == ShiftMode::kRightLogical && IsSignedIntDtype(dtype)) {
        // keep = ~((INT_MIN >> s) << 1): clears the s sign-extended high bits.
        MPSGraphTensor* one = [g constantWithScalar:1 dataType:dtype];
        MPSGraphTensor* intMin = [g bitwiseLeftShiftWithPrimaryTensor:AllOnes(g, dtype)
                                                      secondaryTensor:[g constantWithScalar:bitWidth - 1
                                                                                   dataType:dtype]
                                                                 name:nil];
        MPSGraphTensor* high = [g bitwiseRightShiftWithPrimaryTensor:intMin
                                                     secondaryTensor:shiftAmount
                                                                name:nil];
        high = [g bitwiseLeftShiftWithPrimaryTensor:high secondaryTensor:one name:nil];
        MPSGraphTensor* keep = [g bitwiseNOTWithTensor:high name:nil];
        shiftedResult = [g bitwiseANDWithPrimaryTensor:shiftedResult secondaryTensor:keep name:nil];
    } else if (mode == ShiftMode::kRightArithmetic && IsUnsignedIntDtype(dtype)) {
        // fill = ~(UMAX >> s): the s high bits, set only when the top bit is.
        MPSGraphTensor* ones = AllOnes(g, dtype);
        MPSGraphTensor* fill = [g bitwiseNOTWithTensor:[g bitwiseRightShiftWithPrimaryTensor:ones
                                                                             secondaryTensor:shiftAmount
                                                                                        name:nil]
                                                  name:nil];
        fill = [g selectWithPredicateTensor:TopBitSet(g, input, bitWidth)
                        truePredicateTensor:fill
                       falsePredicateTensor:[g constantWithScalar:0 dataType:dtype]
                                       name:nil];
        shiftedResult = [g bitwiseORWithPrimaryTensor:shiftedResult secondaryTensor:fill name:nil];
    }

    if (inRangeSplat) {
        return shiftedResult;
    }

    MPSGraphTensor* overflowMask = BuildShiftOverflowMask(ctx.graph, shiftAmount, bitWidth);
    MPSGraphTensor* overflowValue = BuildShiftOverflowValue(ctx.graph, input, mode, bitWidth);

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
