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

#include <cstdlib>
#include <limits>
#include <sstream>

namespace jax_metallib {

// MPSGraph 8-bit integer add/subtract/multiply saturated instead of wrapping
// (e.g. int8 -128 + -128 gave -128, not 0), while 16/32-bit arithmetic and
// int32 -> 8-bit conversion wrapped correctly. Widen 8-bit operands to int32,
// compute there, and convert back to get two's-complement wraparound.
static bool IsEightBitInt(MPSDataType t) {
    return t == MPSDataTypeInt8 || t == MPSDataTypeUInt8;
}

// x + 1.0 lowers to add(x, broadcast_in_dim(constant)). Handing MPSGraph the
// explicitly broadcast operand made it materialise and re-read a full-size
// tensor: a scalar add on 16M f32 took 1.78 ms vs 1.24 ms for a unary op (and
// vs jax-mps). When one operand is a broadcast of a one-element tensor and the
// other already has the output shape, feed the source reshaped to [1,...,1] and
// let MPSGraph broadcast implicitly.
// JAX_METALLIB_DISABLE_SCALAR_OPERAND_BROADCAST=1 restores the explicit form.
static bool ScalarOperandFoldEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_SCALAR_OPERAND_BROADCAST");
        return !(v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
    }();
    return enabled;
}

// Source of operand `index` when it is a broadcast_in_dim of a one-element
// tensor with the same dtype, else nil.
static MPSGraphTensor* ScalarBroadcastSource(HandlerContext& ctx, unsigned index, MPSGraphTensor* t) {
    auto bc =
        mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(ctx.op->getOperand(index).getDefiningOp());
    if (!bc) {
        return nil;
    }
    MPSGraphTensor* src = GetTensor(ctx.values, bc.getOperand());
    if (!src || !src.shape || src.dataType != t.dataType) {
        return nil;
    }
    for (NSNumber* d in src.shape) {
        if ([d longLongValue] != 1) {
            return nil;
        }
    }
    return src;
}

static MPSGraphTensor* BinaryOperand(HandlerContext& ctx, unsigned index) {
    MPSGraphTensor* t = GetInputTensor(ctx, index);
    if (!t || !ScalarOperandFoldEnabled() || ctx.variables || ctx.op->getNumOperands() != 2) {
        return t;
    }
    MPSGraphTensor* src = ScalarBroadcastSource(ctx, index, t);
    MPSGraphTensor* other = GetInputTensor(ctx, 1 - index);
    NSArray<NSNumber*>* out = GetOutputShape(ctx.op);
    // Fold only one side: if both operands are scalar broadcasts, folding both
    // collapses the result to one element (MultivariateNormal grad aborted in
    // a later reshape [1] -> [3,1]).
    if (!src || !other || !out || out.count == 0 || !other.shape || ![other.shape isEqualToArray:out] ||
        ScalarBroadcastSource(ctx, 1 - index, other)) {
        return t;
    }
    NSMutableArray<NSNumber*>* ones = [NSMutableArray arrayWithCapacity:out.count];
    for (NSUInteger i = 0; i < out.count; ++i) {
        [ones addObject:@1];
    }
    return [ctx.graph reshapeTensor:src withShape:ones name:nil];
}

template <typename Op>
static ProcessResult HandleWrappingBinary(HandlerContext& ctx, const char* name, Op op) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error(std::string(name) + ": missing input tensor");
    MPSDataType dtype = lhs.dataType;
    if (IsEightBitInt(dtype) && rhs.dataType == dtype) {
        MPSGraphTensor* l = [ctx.graph castTensor:lhs toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* r = [ctx.graph castTensor:rhs toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* wide = op(l, r);
        return Result(ctx, [ctx.graph castTensor:wide toType:dtype name:nil], name);
    }
    return Result(ctx, op(lhs, rhs), name);
}

// True when `v` is a splat integer constant (possibly behind broadcast_in_dim
// or reshape) equal to the type's signed minimum (only the sign bit set).
static bool IsSignBitSplat(mlir::Value v) {
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
            auto intType = mlir::dyn_cast<mlir::IntegerType>(dense.getElementType());
            if (!intType || intType.getWidth() < 8) {
                return false;
            }
            return dense.getSplatValue<llvm::APInt>().isMinSignedValue();
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

// MPSGraph add of a splat INT32_MIN / INT64_MIN constant returned the other
// operand unchanged wherever the sum overflows (jit `x + int32(-2**31)`), while
// the same value as an argument, a non-splat constant, sub/mul/xor with it,
// and int8/int16 minimums all wrapped correctly. Adding the sign bit only
// flips the top bit, so x + MIN == x ^ MIN exactly in two's complement.
static ProcessResult HandleAdd(HandlerContext& ctx) {
    if (ctx.op->getNumOperands() == 2 &&
        (IsSignBitSplat(ctx.op->getOperand(0)) || IsSignBitSplat(ctx.op->getOperand(1)))) {
        return HandleWrappingBinary(ctx, "add", [&](MPSGraphTensor* a, MPSGraphTensor* b) {
            return [ctx.graph bitwiseXORWithPrimaryTensor:a secondaryTensor:b name:nil];
        });
    }
    return HandleWrappingBinary(ctx, "add", [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [ctx.graph additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    });
}
REGISTER_MPS_OP("stablehlo.add", HandleAdd);

static ProcessResult HandleSubtract(HandlerContext& ctx) {
    return HandleWrappingBinary(ctx, "subtract", [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [ctx.graph subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    });
}
REGISTER_MPS_OP("stablehlo.subtract", HandleSubtract);
// Integer x*x came back saturated and float32-rounded under JIT (int16/int32/
// uint32), consistent with MPSGraph rewriting it into a float square. MPSGraph
// also proves operands equal after its own folding (x*(x+0), duplicate casts),
// so the plugin cannot detect every square. Compute every integer multiply as
// lhs*(rhs+1) - lhs: identical modulo 2**n, and never a literal square.
static ProcessResult HandleMultiply(HandlerContext& ctx) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("multiply: missing input tensor");
    MPSDataType dtype = lhs.dataType;
    const bool isInteger = dtype == MPSDataTypeInt8 || dtype == MPSDataTypeInt16 ||
                           dtype == MPSDataTypeInt32 || dtype == MPSDataTypeInt64 ||
                           dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeUInt16 ||
                           dtype == MPSDataTypeUInt32 || dtype == MPSDataTypeUInt64;
    if (isInteger && rhs.dataType == dtype) {
        // 8-bit multiplies saturate; compute in int32 and wrap on conversion.
        const bool narrow = IsEightBitInt(dtype);
        MPSDataType work = narrow ? MPSDataTypeInt32 : dtype;
        MPSGraphTensor* l = narrow ? [ctx.graph castTensor:lhs toType:work name:nil] : lhs;
        MPSGraphTensor* r = narrow ? [ctx.graph castTensor:rhs toType:work name:nil] : rhs;
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1 dataType:work];
        MPSGraphTensor* next = [ctx.graph additionWithPrimaryTensor:r secondaryTensor:one name:nil];
        MPSGraphTensor* prod = [ctx.graph multiplicationWithPrimaryTensor:l secondaryTensor:next name:nil];
        MPSGraphTensor* out = [ctx.graph subtractionWithPrimaryTensor:prod secondaryTensor:l name:nil];
        if (narrow) {
            out = [ctx.graph castTensor:out toType:dtype name:nil];
        }
        return Result(ctx, out, "multiply");
    }
    MPSGraphTensor* out = [ctx.graph multiplicationWithPrimaryTensor:lhs secondaryTensor:rhs name:nil];
    return Result(ctx, out, "multiply");
}
REGISTER_MPS_OP("stablehlo.multiply", HandleMultiply);
// Integer divide on MPS: 8-bit returned garbage (int8 100/5 -> 0, -1/2 -> -1;
// uint8 254/3 -> 0) while 16/32-bit truncated correctly, and x / 0 returned
// sign-dependent values for every width where XLA defines all ones (-1 signed,
// max unsigned). Divide by a safe divisor (0 -> 1), 8-bit in int32 with a
// wrapping cast back (int8 -128/-1 -> -128), then select all ones where the
// divisor was 0.
static ProcessResult HandleDivide(HandlerContext& ctx) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("divide: missing input tensor");
    MPSGraph* g = ctx.graph;
    const MPSDataType dtype = lhs.dataType;
    const bool isInteger = dtype == MPSDataTypeInt8 || dtype == MPSDataTypeInt16 ||
                           dtype == MPSDataTypeInt32 || dtype == MPSDataTypeInt64 ||
                           dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeUInt16 ||
                           dtype == MPSDataTypeUInt32 || dtype == MPSDataTypeUInt64;
    if (!isInteger || rhs.dataType != dtype) {
        return Result(ctx, [g divisionWithPrimaryTensor:lhs secondaryTensor:rhs name:nil], "divide");
    }
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:dtype];
    MPSGraphTensor* isZero = [g equalWithPrimaryTensor:rhs secondaryTensor:zero name:nil];
    MPSGraphTensor* safe = [g selectWithPredicateTensor:isZero
                                    truePredicateTensor:[g constantWithScalar:1 dataType:dtype]
                                   falsePredicateTensor:rhs
                                                   name:nil];
    MPSGraphTensor* quot = nil;
    if (IsEightBitInt(dtype)) {
        MPSGraphTensor* l = [g castTensor:lhs toType:MPSDataTypeInt32 name:nil];
        MPSGraphTensor* r = [g castTensor:safe toType:MPSDataTypeInt32 name:nil];
        quot = [g castTensor:[g divisionWithPrimaryTensor:l secondaryTensor:r name:nil]
                      toType:dtype
                        name:nil];
    } else {
        quot = [g divisionWithPrimaryTensor:lhs secondaryTensor:safe name:nil];
    }
    MPSGraphTensor* allOnes = [g bitwiseNOTWithTensor:zero name:nil];
    MPSGraphTensor* out = [g selectWithPredicateTensor:isZero
                                   truePredicateTensor:allOnes
                                  falsePredicateTensor:quot
                                                  name:nil];
    return Result(ctx, out, "divide");
}
REGISTER_MPS_OP("stablehlo.divide", HandleDivide);
// StableHLO maximum/minimum propagate NaN. MPSGraph maximum/minimum returned
// the non-NaN operand for float32/bfloat16, so jnp.maximum, relu and clip
// silently hid NaNs. Use the NaN-propagating variants for floating types.
static bool IsRealFloatDtype(MPSDataType t) {
    return t == MPSDataTypeFloat32 || t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
}

static MPSGraphTensor* NaNMax(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    if (IsRealFloatDtype(a.dataType)) {
        return [g maximumWithNaNPropagationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    return [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil];
}

static MPSGraphTensor* NaNMin(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
    if (IsRealFloatDtype(a.dataType)) {
        return [g minimumWithNaNPropagationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    return [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
}

// XLA orders -0 < +0 in max/min: max(-0, +0) == +0 and min(-0, +0) == -0 in
// either operand order. MPSGraph maximum/minimum return the secondary operand
// on a tie, so a +-0 tie came back order-dependent (clip(x, -0., 1.) gave -0
// for x == +0, jnp.maximum(-0., +0.) gave -0). Equal non-zero floats are
// bit-identical, so only zero ties need fixing. Non-zero splat constants
// cannot tie at zero and keep the plain op; zero constants and two tensors get
// a select fix-up (see TieAwareMaxMin).
// JAX_METALLIB_DISABLE_MAXMIN_ZERO_TIE=1 restores the plain lowering.
static bool MaxMinZeroTieEnabled() {
    static const bool enabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_MAXMIN_ZERO_TIE");
        return !(v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0'));
    }();
    return enabled;
}

enum class FloatSplatKind { kNone, kNonZero, kPosZero, kNegZero };

// Kind of `v` when it is a splat float constant (possibly behind
// broadcast_in_dim or reshape), else kNone.
static FloatSplatKind GetFloatSplatKind(mlir::Value v) {
    for (int depth = 0; depth < 8 && v; ++depth) {
        mlir::Operation* def = v.getDefiningOp();
        if (!def) {
            return FloatSplatKind::kNone;
        }
        if (auto c = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(def)) {
            auto dense = mlir::dyn_cast<mlir::DenseFPElementsAttr>(c.getValue());
            if (!dense || !dense.isSplat()) {
                return FloatSplatKind::kNone;
            }
            llvm::APFloat f = dense.getSplatValue<llvm::APFloat>();
            if (!f.isZero()) {
                return FloatSplatKind::kNonZero;
            }
            return f.isNegative() ? FloatSplatKind::kNegZero : FloatSplatKind::kPosZero;
        }
        if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(def)) {
            v = bc.getOperand();
            continue;
        }
        if (auto rs = mlir::dyn_cast<mlir::stablehlo::ReshapeOp>(def)) {
            v = rs.getOperand();
            continue;
        }
        return FloatSplatKind::kNone;
    }
    return FloatSplatKind::kNone;
}

// NaN-propagating max/min with XLA's -0 < +0 ordering. `av`/`bv` are the MLIR
// values behind `a`/`b` (or a null value when unknown).
static MPSGraphTensor* TieAwareMaxMin(MPSGraph* g, MPSGraphTensor* a, mlir::Value av, MPSGraphTensor* b,
                                      mlir::Value bv, bool isMax) {
    auto plain = [&](MPSGraphTensor* x, MPSGraphTensor* y) {
        return isMax ? NaNMax(g, x, y) : NaNMin(g, x, y);
    };
    if (!IsRealFloatDtype(a.dataType) || b.dataType != a.dataType || !MaxMinZeroTieEnabled()) {
        return plain(a, b);
    }
    FloatSplatKind ka = GetFloatSplatKind(av);
    FloatSplatKind kb = GetFloatSplatKind(bv);
    if (ka == FloatSplatKind::kNonZero || kb == FloatSplatKind::kNonZero) {
        return plain(a, b);
    }
    MPSGraphTensor* m = plain(a, b);
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:a.dataType];
    // The zero that should win a tie: +0 for max, -0 for min.
    const FloatSplatKind winner = isMax ? FloatSplatKind::kPosZero : FloatSplatKind::kNegZero;
    if ((ka == FloatSplatKind::kNone) != (kb == FloatSplatKind::kNone)) {
        // One operand is a +-0 splat constant c. MPSGraph canonicalises the
        // operand order of max/min (max(b,a) was CSE'd with max(a,b)), so
        // swapping operands cannot steer the tie. It returned c on a tie, which
        // is right when c is the winning zero (relu, max(x, +0), costs nothing;
        // tests/test_maxmin_signed_zero.py pins both operand orders); when c is
        // the losing zero, take x wherever x == 0.
        const bool constIsB = kb != FloatSplatKind::kNone;
        if ((constIsB ? kb : ka) == winner) {
            return m;
        }
        MPSGraphTensor* x = constIsB ? a : b;
        MPSGraphTensor* xZero = [g equalWithPrimaryTensor:x secondaryTensor:zero name:nil];
        return [g selectWithPredicateTensor:xZero truePredicateTensor:x falsePredicateTensor:m name:nil];
    }
    if (a.dataType == MPSDataTypeFloat32) {
        // f32: with s = a + b (min: s = -((-a) - b)), m == s holds exactly when
        // the other operand is zero or too small to change m, and then s equals
        // m numerically (equal non-zero floats are bit-identical); at a +-0 tie
        // s carries the AND (min: OR) of the signs. NaN fails the compare and
        // keeps m. This kept fused f32 chains at plain speed, where the both-zero
        // select below cost 3.4x on a tanh chain; in f16/bf16 this rule cost up
        // to 5x on affine chains, so half dtypes keep the both-zero select
        // (build/probe_maxmin_heldout_r20.py).
        MPSGraphTensor* s =
            isMax ? [g additionWithPrimaryTensor:a secondaryTensor:b name:nil]
                  : [g negativeWithTensor:[g subtractionWithPrimaryTensor:[g negativeWithTensor:a name:nil]
                                                          secondaryTensor:b
                                                                     name:nil]
                                     name:nil];
        MPSGraphTensor* same = [g equalWithPrimaryTensor:m secondaryTensor:s name:nil];
        return [g selectWithPredicateTensor:same truePredicateTensor:s falsePredicateTensor:m name:nil];
    }
    // Two tensors (or two zero constants): where both are zero, max is a + b
    // (-0 only if both are -0) and min is -((-a) + (-b)) (-0 if either is -0).
    MPSGraphTensor* both =
        [g logicalANDWithPrimaryTensor:[g equalWithPrimaryTensor:a secondaryTensor:zero name:nil]
                       secondaryTensor:[g equalWithPrimaryTensor:b secondaryTensor:zero name:nil]
                                  name:nil];
    MPSGraphTensor* fixed =
        isMax ? [g additionWithPrimaryTensor:a secondaryTensor:b name:nil]
              : [g negativeWithTensor:[g additionWithPrimaryTensor:[g negativeWithTensor:a name:nil]
                                                   secondaryTensor:[g negativeWithTensor:b name:nil]
                                                              name:nil]
                                 name:nil];
    return [g selectWithPredicateTensor:both truePredicateTensor:fixed falsePredicateTensor:m name:nil];
}

static ProcessResult HandleMaximum(HandlerContext& ctx) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("maximum: missing input tensor");
    return Result(ctx,
                  TieAwareMaxMin(ctx.graph, lhs, ctx.op->getOperand(0), rhs, ctx.op->getOperand(1), true),
                  "maximum");
}
REGISTER_MPS_OP("stablehlo.maximum", HandleMaximum);

static ProcessResult HandleMinimum(HandlerContext& ctx) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("minimum: missing input tensor");
    return Result(ctx,
                  TieAwareMaxMin(ctx.graph, lhs, ctx.op->getOperand(0), rhs, ctx.op->getOperand(1), false),
                  "minimum");
}
REGISTER_MPS_OP("stablehlo.minimum", HandleMinimum);
// XLA defines integer x % 0 == x. MPSGraph modulo got that right for a runtime
// zero divisor but returned junk when the zero divisor was a compile-time
// constant ([7,-7,0,5] % 0 -> [1,1,0,1]), which also broke JAX's floor_divide
// correction (a // 0 gave -1 instead of -2). Take the remainder by a safe
// divisor (0 -> 1) and select the dividend where the divisor is 0.
static ProcessResult HandleRemainder(HandlerContext& ctx) {
    MPSGraphTensor* lhs = BinaryOperand(ctx, 0);
    MPSGraphTensor* rhs = BinaryOperand(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("remainder: missing input tensor");
    MPSGraph* g = ctx.graph;
    const MPSDataType dtype = lhs.dataType;
    const bool isInteger = dtype == MPSDataTypeInt8 || dtype == MPSDataTypeInt16 ||
                           dtype == MPSDataTypeInt32 || dtype == MPSDataTypeInt64 ||
                           dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeUInt16 ||
                           dtype == MPSDataTypeUInt32 || dtype == MPSDataTypeUInt64;
    if (!isInteger || rhs.dataType != dtype) {
        return Result(ctx, [g moduloWithPrimaryTensor:lhs secondaryTensor:rhs name:nil], "remainder");
    }
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:dtype];
    MPSGraphTensor* isZero = [g equalWithPrimaryTensor:rhs secondaryTensor:zero name:nil];
    MPSGraphTensor* safe = [g selectWithPredicateTensor:isZero
                                    truePredicateTensor:[g constantWithScalar:1 dataType:dtype]
                                   falsePredicateTensor:rhs
                                                   name:nil];
    MPSGraphTensor* rem = [g moduloWithPrimaryTensor:lhs secondaryTensor:safe name:nil];
    MPSGraphTensor* out = [g selectWithPredicateTensor:isZero
                                   truePredicateTensor:lhs
                                  falsePredicateTensor:rem
                                                  name:nil];
    return Result(ctx, out, "remainder");
}
REGISTER_MPS_OP("stablehlo.remainder", HandleRemainder);
REGISTER_MLIR_BINARY_OP("stablehlo.power", power, power);
REGISTER_MLIR_BINARY_OP("stablehlo.atan2", atan2, atan2);

static bool IsIntegerMatmulDtype(MPSDataType t) {
    return t == MPSDataTypeInt8 || t == MPSDataTypeUInt8 || t == MPSDataTypeInt16 || t == MPSDataTypeUInt16 ||
           t == MPSDataTypeInt32 || t == MPSDataTypeUInt32;
}

// MPSGraph matmul rejects integer operands with a fatal assertion that aborts
// the process. Compute integer matmuls exactly in int32 instead: broadcast
// multiply [.., M, K, 1] x [.., 1, K, N] and sum over K (int32 multiply and
// sum wrap modulo 2**32 like CPU), then convert to the result dtype.
// Accepts rank-1 (vector) or equal-rank 2/3 operands.
static MPSGraphTensor* IntegerMatMul(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b, MPSDataType outType) {
    auto toWork = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
        if (t.dataType == MPSDataTypeInt32)
            return t;
        if (t.dataType == MPSDataTypeUInt32)
            return [g reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
        return [g castTensor:t toType:MPSDataTypeInt32 name:nil];
    };
    a = toWork(a);
    b = toWork(b);
    if (a.shape.count == 1)
        a = [g reshapeTensor:a withShape:@[@1, a.shape[0]] name:nil];
    if (b.shape.count == 1)
        b = [g reshapeTensor:b withShape:@[b.shape[0], @1] name:nil];
    if (a.shape.count != b.shape.count || a.shape.count < 2)
        return nil;
    NSMutableArray<NSNumber*>* aShape = [NSMutableArray arrayWithArray:a.shape];
    [aShape addObject:@1];
    NSMutableArray<NSNumber*>* bShape = [NSMutableArray arrayWithArray:b.shape];
    [bShape insertObject:@1 atIndex:bShape.count - 2];
    a = [g reshapeTensor:a withShape:aShape name:nil];
    b = [g reshapeTensor:b withShape:bShape name:nil];
    MPSGraphTensor* prod = [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    MPSGraphTensor* sum = [g reductionSumWithTensor:prod axis:(NSInteger)prod.shape.count - 2 name:nil];
    NSMutableArray<NSNumber*>* outShape = [NSMutableArray arrayWithArray:prod.shape];
    [outShape removeObjectAtIndex:outShape.count - 2];
    sum = [g reshapeTensor:sum withShape:outShape name:nil];
    if (outType == MPSDataTypeInvalid || outType == MPSDataTypeInt32)
        return sum;
    if (outType == MPSDataTypeUInt32)
        return [g reinterpretCastTensor:sum toType:MPSDataTypeUInt32 name:nil];
    return [g castTensor:sum toType:outType name:nil];
}

static ProcessResult HandleDot(HandlerContext& ctx) {
    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("dot: missing input tensor");
    // StableHLO dot allows a rank-1 operand on either side ([M,K].[K] -> [M],
    // [K].[K,N] -> [N]); MPSGraph matmul does not, and aborted the process with
    // "K dimension is inconsistently defined". Promote the vector to a matrix
    // and reshape the product to the declared result shape below.
    // vector.vector already matches XLA:CPU and is left unchanged.
    if (lhs.shape.count == 2 && rhs.shape.count == 1) {
        rhs = [ctx.graph reshapeTensor:rhs withShape:@[rhs.shape[0], @1] name:nil];
    } else if (lhs.shape.count == 1 && rhs.shape.count == 2) {
        lhs = [ctx.graph reshapeTensor:lhs withShape:@[@1, lhs.shape[0]] name:nil];
    }
    NSArray<NSNumber*>* declaredShape = GetOutputShape(ctx.op);
    if (IsIntegerMatmulDtype(lhs.dataType) && IsIntegerMatmulDtype(rhs.dataType)) {
        MPSGraphTensor* result = IntegerMatMul(ctx.graph, lhs, rhs, GetResultMpsType(ctx.op));
        if (result && declaredShape && ![declaredShape isEqualToArray:result.shape]) {
            result = [ctx.graph reshapeTensor:result withShape:declaredShape name:nil];
        }
        return Result(ctx, result, "dot");
    }
    // float16 matmul overflowed to inf near the float16 max; compute in
    // float32 and cast back (see HandleDotGeneral).
    // bf16 operands with preferred_element_type=f32 are widened too (exact): the
    // bf16 matmul result was stored under the f32 result type and read back as
    // garbage (relative error ~1.0-1.5, no error raised).
    const bool widenF16 = (lhs.dataType == MPSDataTypeFloat16 && rhs.dataType == MPSDataTypeFloat16) ||
                          (lhs.dataType == MPSDataTypeBFloat16 && rhs.dataType == MPSDataTypeBFloat16 &&
                           GetResultMpsType(ctx.op) == MPSDataTypeFloat32);
    if (widenF16) {
        lhs = [ctx.graph castTensor:lhs toType:MPSDataTypeFloat32 name:nil];
        rhs = [ctx.graph castTensor:rhs toType:MPSDataTypeFloat32 name:nil];
    }
    MPSGraphTensor* result = [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                              secondaryTensor:rhs
                                                                         name:nil];
    if (widenF16 && result) {
        MPSDataType outType = GetResultMpsType(ctx.op);
        if (outType == MPSDataTypeInvalid)
            outType = MPSDataTypeFloat16;
        result = [ctx.graph castTensor:result toType:outType name:nil];
    }
    if (result && declaredShape && ![declaredShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:declaredShape name:nil];
    }
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

    // Mixed precision: cast fp32 inputs to fp16 for 2x MMA throughput.
    // Only for large matmuls where the speedup outweighs precision loss and graph overhead.
    // Disabled by JAX_METALLIB_GEMM_TRUE_F32=1.
    bool mixedFp32 = false;
    {
        static const bool useMixed = []() {
            const char* v = std::getenv("JAX_METALLIB_GEMM_TRUE_F32");
            return !(v && v[0] == '1' && v[1] == '\0');
        }();
        if (useMixed && lhsInput.dataType == MPSDataTypeFloat32) {
            // Require both inputs to have >= 500K elements to avoid
            // precision loss and overhead on small/special-shape matmuls.
            // Also require sufficient FLOPs (M*N*K*batch >= 100M) to amortize cast overhead.
            auto totalElems = [](MPSGraphTensor* t) -> uint64_t {
                uint64_t total = 1;
                for (NSNumber* d in t.shape) {
                    uint64_t dim = [d unsignedLongLongValue];
                    if (dim > 0)
                        total *= dim;
                }
                return total;
            };
            if (totalElems(lhsInput) >= 500000 && totalElems(rhsInput) >= 500000) {
                auto dimNumbers = dotOp.getDotDimensionNumbers();
                auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
                auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
                auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
                auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();
                NSArray<NSNumber*>* lhsShape = lhsInput.shape;
                NSArray<NSNumber*>* rhsShape = rhsInput.shape;
                uint64_t m = 1, k = 1, n = 1, batch = 1;
                for (int64_t d : lhsBatchDims) {
                    batch *= [lhsShape[(NSUInteger)d] unsignedLongLongValue];
                }
                for (int64_t d : lhsContractingDims) {
                    k *= [lhsShape[(NSUInteger)d] unsignedLongLongValue];
                }
                for (NSUInteger d = 0; d < lhsShape.count; ++d) {
                    bool used = false;
                    for (int64_t bd : lhsBatchDims) if ((NSUInteger)bd == d) used = true;
                    for (int64_t cd : lhsContractingDims) if ((NSUInteger)cd == d) used = true;
                    if (!used) m *= [lhsShape[d] unsignedLongLongValue];
                }
                for (NSUInteger d = 0; d < rhsShape.count; ++d) {
                    bool used = false;
                    for (int64_t bd : rhsBatchDims) if ((NSUInteger)bd == d) used = true;
                    for (int64_t cd : rhsContractingDims) if ((NSUInteger)cd == d) used = true;
                    if (!used) n *= [rhsShape[d] unsignedLongLongValue];
                }
                uint64_t flops = m * n * k * batch;
                if (flops >= 100000000) {
                    mixedFp32 = true;
                    lhsInput = [ctx.graph castTensor:lhsInput toType:MPSDataTypeFloat16 name:nil];
                    rhsInput = [ctx.graph castTensor:rhsInput toType:MPSDataTypeFloat16 name:nil];
                }
            }
        }
    }
    // MPSGraph float16 matmul returned +/-inf for results between ~60000 and
    // the float16 max (e.g. 60000 * 1.0 with K=1) where CPU is exact. Widen
    // native float16 operands to float32 (exact) and cast the result back to
    // the declared result type. This sits after the fp32->fp16 mixed-precision
    // block, which only triggers on float32 inputs, so it cannot re-narrow.
    // Measured cost: ~+60 us at 256^3 and 1.2-2.7x slower than native f16 at
    // MLP/GPT-2 sizes. A standalone Level0 probe (5x1x7 plus 128x256x768 up to
    // 4096x2048x512 with rows engineered into [60000,65504]) found no spurious
    // inf in native f16 matmul, f32 accumulation (4096 ones sum to 4096), and
    // error within the same bound as the widened path; the overflow appears to
    // have come from Level1/ANE placement. Large dots whose declared result is
    // f16 therefore run natively; small ones and preferred_element_type=f32
    // keep the widening. JAX_METALLIB_F16_NATIVE_DOT=0 widens everything.
    bool nativeF16Dot = false;
    if (!mixedFp32 && lhsInput.dataType == MPSDataTypeFloat16 && rhsInput.dataType == MPSDataTypeFloat16 &&
        GetResultMpsType(ctx.op) == MPSDataTypeFloat16) {
        static const bool enabled = []() {
            const char* v = std::getenv("JAX_METALLIB_F16_NATIVE_DOT");
            return !(v && v[0] == '0' && v[1] == '\0');
        }();
        if (enabled) {
            auto dimNumbers = dotOp.getDotDimensionNumbers();
            auto lhsContracting = dimNumbers.getLhsContractingDimensions();
            NSArray<NSNumber*>* lhsShape = lhsInput.shape;
            NSArray<NSNumber*>* rhsShape = rhsInput.shape;
            uint64_t lhsElems = 1, rhsElems = 1, k = 1;
            for (NSNumber* d in lhsShape) {
                lhsElems *= [d unsignedLongLongValue];
            }
            for (NSNumber* d in rhsShape) {
                rhsElems *= [d unsignedLongLongValue];
            }
            for (int64_t d : lhsContracting) {
                k *= [lhsShape[(NSUInteger)d] unsignedLongLongValue];
            }
            // (lhsElems / k) * rhsElems == batch^2 * M * N * K: an upper bound on
            // the dot's M*N*K*batch work, used only as a size gate.
            uint64_t work = (k > 0) ? (lhsElems / k) * rhsElems : 0;
            nativeF16Dot = lhsShape && rhsShape && work >= (1ULL << 20);
        }
    }
    bool widenedF16 = false;
    if (!mixedFp32 && !nativeF16Dot && lhsInput.dataType == MPSDataTypeFloat16 &&
        rhsInput.dataType == MPSDataTypeFloat16) {
        lhsInput = [ctx.graph castTensor:lhsInput toType:MPSDataTypeFloat32 name:nil];
        rhsInput = [ctx.graph castTensor:rhsInput toType:MPSDataTypeFloat32 name:nil];
        widenedF16 = true;
    }
    // bf16 x bf16 with preferred_element_type=f32: widen (exact) so the matmul
    // accumulates and returns f32 like XLA:CPU. Placed after the fp32->fp16
    // mixed-precision block so it can never be re-narrowed to f16's smaller range.
    if (!mixedFp32 && lhsInput.dataType == MPSDataTypeBFloat16 && rhsInput.dataType == MPSDataTypeBFloat16 &&
        GetResultMpsType(ctx.op) == MPSDataTypeFloat32) {
        lhsInput = [ctx.graph castTensor:lhsInput toType:MPSDataTypeFloat32 name:nil];
        rhsInput = [ctx.graph castTensor:rhsInput toType:MPSDataTypeFloat32 name:nil];
    }
    // Wrap result with fp16→f32 cast when mixed precision is active.
    auto DotResult = [&](MPSGraphTensor* result, const char* name) -> ProcessResult {
        if (mixedFp32 && result)
            result = [ctx.graph castTensor:result toType:MPSDataTypeFloat32 name:nil];
        if (widenedF16 && result) {
            MPSDataType outType = GetResultMpsType(ctx.op);
            if (outType == MPSDataTypeInvalid)
                outType = MPSDataTypeFloat16;
            if (result.dataType != outType)
                result = [ctx.graph castTensor:result toType:outType name:nil];
        }
        // Result() never converts to the declared type; a float result whose
        // dtype differs from it would be reinterpreted, so cast as a backstop.
        if (result) {
            MPSDataType declared = GetResultMpsType(ctx.op);
            auto isFloat = [](MPSDataType t) {
                return t == MPSDataTypeFloat32 || t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
            };
            if (isFloat(declared) && isFloat(result.dataType) && result.dataType != declared)
                result = [ctx.graph castTensor:result toType:declared name:nil];
        }
        return Result(ctx, result, name);
    };
    // Integer dots skip the float matmul fast paths and use the canonical
    // rank-2/3 general path below with IntegerMatMul.
    const bool integerDot =
        IsIntegerMatmulDtype(lhsInput.dataType) && IsIntegerMatmulDtype(rhsInput.dataType);

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
        // XLA: a zero-length contraction is an empty sum (zeros) and an empty output has
        // no elements, so the result is zeros of the declared type in either case.
        MPSDataType declared = GetResultMpsType(ctx.op);
        if (declared == MPSDataTypeInvalid || !outShape) {
            return ProcessResult::Error("dot_general: zero-sized dimensions with unknown result type (" +
                                        DotGeneralDebugString(dotOp, lhsShape, rhsShape, outShape) + ")");
        }
        MPSGraphTensor* zeros = [ctx.graph constantWithScalar:0.0 dataType:declared];
        if (outShape.count > 0) {
            zeros = [ctx.graph broadcastTensor:zeros toShape:outShape name:nil];
        }
        return Result(ctx, zeros, "dot_general_zero_dim");
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

    // A float dot whose contraction has exactly one element (K=1, including the
    // no-contraction outer product) is a plain product. XLA:CPU folds it into a
    // multiply, so 0 * -3 gives -0, while the MPSGraph matmul accumulates from +0
    // and returns +0 (K >= 2 agree: both start from +0). Lower it as a broadcast
    // multiply in f32: the product of two f32 values rounds once, and the product
    // of two halves is exact in f32, so the cast to the declared type rounds once.
    // JAX_METALLIB_DISABLE_DOT_K1_MULTIPLY=1 keeps the matmul.
    {
        static const bool k1Enabled = []() {
            const char* v = std::getenv("JAX_METALLIB_DISABLE_DOT_K1_MULTIPLY");
            return !(v && v[0] == '1' && v[1] == '\0');
        }();
        auto isFloatType = [](MPSDataType t) {
            return t == MPSDataTypeFloat32 || t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
        };
        int64_t contractElems = 1;
        for (int64_t d : lhsContractingDims) {
            contractElems *= [lhsShape[(NSUInteger)d] longLongValue];
        }
        MPSGraphTensor* lhsOrig = GetInputTensor(ctx, 0);
        MPSGraphTensor* rhsOrig = GetInputTensor(ctx, 1);
        MPSDataType declared = GetResultMpsType(ctx.op);
        if (k1Enabled && contractElems == 1 && !integerDot && outShape && isFloatType(declared) &&
            isFloatType(lhsOrig.dataType) && isFloatType(rhsOrig.dataType)) {
            std::vector<NSUInteger> lhsFree;
            std::vector<NSUInteger> rhsFree;
            for (NSUInteger d = 0; d < lhsRank; ++d) {
                if (!lhsUsed[d])
                    lhsFree.push_back(d);
            }
            for (NSUInteger d = 0; d < rhsRank; ++d) {
                if (!rhsUsed[d])
                    rhsFree.push_back(d);
            }
            // Same-type half dots multiply natively: the exact product of two halves
            // fits in f32, so one rounding to the half type is the correctly rounded
            // product either way, and the native multiply skips the f32 intermediate
            // (f16/bf16 [4096,1]@[1,4096] ~1.1 ms widened vs ~0.6 ms matmul).
            // JAX_METALLIB_DOT_K1_WIDEN_HALF=1 keeps the f32 multiply.
            static const bool k1WidenHalf = []() {
                const char* v = std::getenv("JAX_METALLIB_DOT_K1_WIDEN_HALF");
                return v && v[0] == '1' && v[1] == '\0';
            }();
            MPSDataType computeType = MPSDataTypeFloat32;
            if (!k1WidenHalf && lhsOrig.dataType == declared && rhsOrig.dataType == declared &&
                (declared == MPSDataTypeFloat16 || declared == MPSDataTypeBFloat16))
                computeType = declared;
            // Lay each side out as [batch..., lhsFree..., rhsFree...] with size-1
            // placeholders for the other side's free dims; the multiply broadcasts.
            auto arrange = [&](MPSGraphTensor* t, NSArray<NSNumber*>* shape, llvm::ArrayRef<int64_t> batch,
                               const std::vector<NSUInteger>& own, llvm::ArrayRef<int64_t> contracting,
                               bool ownFirst, size_t otherCount) -> MPSGraphTensor* {
                NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
                NSMutableArray<NSNumber*>* target = [NSMutableArray array];
                for (int64_t d : batch) {
                    [perm addObject:@(d)];
                    [target addObject:shape[(NSUInteger)d]];
                }
                if (!ownFirst) {
                    for (size_t i = 0; i < otherCount; ++i)
                        [target addObject:@1];
                }
                for (NSUInteger d : own) {
                    [perm addObject:@(d)];
                    [target addObject:shape[d]];
                }
                if (ownFirst) {
                    for (size_t i = 0; i < otherCount; ++i)
                        [target addObject:@1];
                }
                for (int64_t d : contracting)
                    [perm addObject:@(d)];
                bool identity = true;
                for (NSUInteger i = 0; i < perm.count; ++i) {
                    if ([perm[i] unsignedIntegerValue] != i)
                        identity = false;
                }
                if (!identity)
                    t = [ctx.graph transposeTensor:t permutation:perm name:nil];
                if (target.count == 0)
                    [target addObject:@1];
                if (![t.shape isEqualToArray:target])
                    t = [ctx.graph reshapeTensor:t withShape:target name:nil];
                if (t.dataType != computeType)
                    t = [ctx.graph castTensor:t toType:computeType name:nil];
                return t;
            };
            MPSGraphTensor* lhsK1 =
                arrange(lhsOrig, lhsShape, lhsBatchDims, lhsFree, lhsContractingDims, true, rhsFree.size());
            MPSGraphTensor* rhsK1 =
                arrange(rhsOrig, rhsShape, rhsBatchDims, rhsFree, rhsContractingDims, false, lhsFree.size());
            MPSGraphTensor* product = [ctx.graph multiplicationWithPrimaryTensor:lhsK1
                                                                 secondaryTensor:rhsK1
                                                                            name:nil];
            if (product.dataType != declared)
                product = [ctx.graph castTensor:product toType:declared name:nil];
            if (![product.shape isEqualToArray:outShape])
                product = [ctx.graph reshapeTensor:product withShape:outShape name:nil];
            return Result(ctx, product, "dot_general_k1");
        }
    }

    if (!integerDot && lhsBatchDims.empty() && rhsBatchDims.empty() && lhsContractingDims.size() == 1 &&
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
            return DotResult(result, "dot_general");
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
            return DotResult(result, "dot_general");
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
            return DotResult(result, "dot_general");
        }
    }

    if (!integerDot && lhsBatchDims.size() == 1 && rhsBatchDims.size() == 1 &&
        lhsContractingDims.size() == 1 && rhsContractingDims.size() == 1) {
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
            return DotResult(result, "dot_general");
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
            return DotResult(result3d, "dot_general");
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
            return DotResult(result3d, "dot_general");
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

    MPSGraphTensor* result = integerDot ? IntegerMatMul(ctx.graph, lhs, rhs, GetResultMpsType(ctx.op))
                                        : [ctx.graph matrixMultiplicationWithPrimaryTensor:lhs
                                                                           secondaryTensor:rhs
                                                                                      name:nil];

    std::vector<int64_t> logicalOutputShape = batchSizes;
    logicalOutputShape.insert(logicalOutputShape.end(), lhsFreeSizes.begin(), lhsFreeSizes.end());
    logicalOutputShape.insert(logicalOutputShape.end(), rhsFreeSizes.begin(), rhsFreeSizes.end());

    NSArray<NSNumber*>* expectedOutputShape = GetOutputShape(ctx.op);
    NSArray<NSNumber*>* finalOutShape =
        expectedOutputShape ? expectedOutputShape : (NSArray<NSNumber*>*)ToNSArray(logicalOutputShape);
    // Reshape the matmul result to match the expected output shape.
    // When there are 2+ batch dims, the matmul produced [B1*B2*..., m, n] (flattened)
    // and must be reshaped back to [B1, B2, ..., m, n]. With 0 or 1 batch dims the
    // matmul output may already match, so the isEqualToArray check avoids a no-op reshape.
    if (finalOutShape && result && ![finalOutShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:finalOutShape name:nil];
    }

    return DotResult(result, "dot_general");
}
REGISTER_MPS_OP("stablehlo.dot_general", HandleDotGeneral);

// XLA TOTALORDER compares (jax lt_to/le_to/eq_to, searchsorted, sort
// comparators) order -NaN < -Inf < ... < -0 < +0 < ... < +Inf < +NaN, NaNs by
// payload. Reinterpret the float bits as a signed integer and flip every bit
// but the sign of negative values: the integer order is then the total order.
// Returns nil for types without a total-order key (integers, complex).
static MPSGraphTensor* TotalOrderKey(MPSGraph* g, MPSGraphTensor* x) {
    MPSDataType intType;
    double magnitudeMask;
    switch (x.dataType) {
    case MPSDataTypeFloat32:
        intType = MPSDataTypeInt32;
        magnitudeMask = 2147483647.0;
        break;
    case MPSDataTypeFloat16:
    case MPSDataTypeBFloat16:
        intType = MPSDataTypeInt16;
        magnitudeMask = 32767.0;
        break;
    default:
        return nil;
    }
    MPSGraphTensor* bits = [g reinterpretCastTensor:x toType:intType name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:intType];
    MPSGraphTensor* negative = [g lessThanWithPrimaryTensor:bits secondaryTensor:zero name:nil];
    MPSGraphTensor* mask = [g selectWithPredicateTensor:negative
                                    truePredicateTensor:[g constantWithScalar:magnitudeMask dataType:intType]
                                   falsePredicateTensor:zero
                                                   name:nil];
    return [g bitwiseXORWithPrimaryTensor:bits secondaryTensor:mask name:nil];
}

static ProcessResult HandleCompare(HandlerContext& ctx) {
    auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(ctx.op);
    if (!compareOp) {
        return ProcessResult::Error("compare: expected CompareOp");
    }

    MPSGraphTensor* lhs = GetInputTensor(ctx, 0);
    MPSGraphTensor* rhs = GetInputTensor(ctx, 1);
    if (!lhs || !rhs)
        return ProcessResult::Error("compare: missing input tensor");

    // compare_type was ignored, so TOTALORDER compares used IEEE semantics:
    // lt_to(-0, 0), lt_to(1, nan) and eq_to(nan, nan) were false, and
    // searchsorted placed NaN queries at the end instead of before +NaN.
    // Rank-0 operands are keyed as [1]: MPSGraph's compile-time optimizer
    // asserts (SIGABRT) on reinterpretCastTensor of a scalar (jnp.digitize on
    // a scalar, hence jsl.expm, aborted). The result is reshaped back below.
    bool scalarKeyed = false;
    auto compareType = compareOp.getCompareTypeAttr();
    if (compareType && compareType.getValue() == mlir::stablehlo::ComparisonType::TOTALORDER) {
        MPSGraphTensor* lhsIn = lhs;
        MPSGraphTensor* rhsIn = rhs;
        const bool scalar = (lhs.shape && lhs.shape.count == 0) || (rhs.shape && rhs.shape.count == 0);
        if (scalar) {
            lhsIn = [ctx.graph reshapeTensor:lhs withShape:@[@1] name:nil];
            rhsIn = [ctx.graph reshapeTensor:rhs withShape:@[@1] name:nil];
        }
        MPSGraphTensor* lhsKey = TotalOrderKey(ctx.graph, lhsIn);
        MPSGraphTensor* rhsKey = TotalOrderKey(ctx.graph, rhsIn);
        if (lhsKey && rhsKey) {
            lhs = lhsKey;
            rhs = rhsKey;
            scalarKeyed = scalar;
        }
    }

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

    if (scalarKeyed) {
        NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
        result = [ctx.graph reshapeTensor:result withShape:outShape ? outShape : @[] name:nil];
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

    // bf16 select flushed subnormals and dropped the NaN sign; select the int16
    // view instead (f16 select was already exact). See HalfToBits in registry.h.
    if (onTrue.dataType == MPSDataTypeBFloat16 && onFalse.dataType == MPSDataTypeBFloat16 &&
        HalfBitSignOpsEnabled()) {
        MPSGraphTensor* tBits = HalfToBits(ctx.graph, onTrue);
        MPSGraphTensor* fBits = HalfToBits(ctx.graph, onFalse);
        NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
        if (tBits && fBits && outShape) {
            MPSGraphTensor* p = pred;
            if (p.shape && p.shape.count == 0 && outShape.count > 0) {
                p = [ctx.graph reshapeTensor:p withShape:@[@1] name:nil];
            }
            MPSGraphTensor* bits = [ctx.graph selectWithPredicateTensor:p
                                                    truePredicateTensor:tBits
                                                   falsePredicateTensor:fBits
                                                                   name:nil];
            return Result(ctx, HalfFromBits(ctx.graph, bits, MPSDataTypeBFloat16, outShape), "select");
        }
    }

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

    if (IsRealFloatDtype(operand.dataType)) {
        // clampWithTensor dropped NaN operands; clamp = min(max(x, lo), hi) with NaN propagation.
        MPSGraphTensor* lo = minVal.dataType == operand.dataType
                                 ? minVal
                                 : [ctx.graph castTensor:minVal toType:operand.dataType name:nil];
        MPSGraphTensor* hi = maxVal.dataType == operand.dataType
                                 ? maxVal
                                 : [ctx.graph castTensor:maxVal toType:operand.dataType name:nil];
        // max/min with XLA's -0 < +0 tie order; free when lo/hi are splat constants.
        MPSGraphTensor* lower = TieAwareMaxMin(ctx.graph, operand, ctx.op->getOperand(1), lo,
                                               minVal == lo ? ctx.op->getOperand(0) : mlir::Value(), true);
        return Result(ctx,
                      TieAwareMaxMin(ctx.graph, lower, mlir::Value(), hi,
                                     maxVal == hi ? ctx.op->getOperand(2) : mlir::Value(), false),
                      "clamp");
    }
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
