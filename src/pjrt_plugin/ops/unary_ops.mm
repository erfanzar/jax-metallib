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

REGISTER_MLIR_UNARY_OP("stablehlo.tanh", tanh, tanh);
REGISTER_MLIR_UNARY_OP("stablehlo.exponential", exponent, exp);
REGISTER_MLIR_UNARY_OP("stablehlo.log", logarithm, log);
REGISTER_MLIR_UNARY_OP("stablehlo.negate", negative, negate);

static ProcessResult HandleAbs(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("abs: missing input tensor");
    MPSGraphTensor* result = [ctx.graph absoluteWithTensor:input name:nil];
    if (input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16)
        result = [ctx.graph realPartOfTensor:result name:nil];
    return Result(ctx, result, "abs");
}
REGISTER_MPS_OP("stablehlo.abs", HandleAbs);
REGISTER_MLIR_UNARY_OP("stablehlo.sqrt", squareRoot, sqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.rsqrt", reciprocalSquareRoot, rsqrt);
REGISTER_MLIR_UNARY_OP("stablehlo.erf", erf, erf);
REGISTER_MLIR_UNARY_OP("chlo.erf", erf, chlo_erf);
REGISTER_MLIR_UNARY_OP("stablehlo.floor", floor, floor);

static ProcessResult HandleSign(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("sign: missing input tensor");

    MPSGraphTensor* result = nil;
    if (input.dataType != MPSDataTypeComplexFloat32 && input.dataType != MPSDataTypeComplexFloat16) {
        result = [ctx.graph signWithTensor:input name:nil];
    } else {
        MPSGraphTensor* re = [ctx.graph realPartOfTensor:input name:nil];
        MPSGraphTensor* im = [ctx.graph imaginaryPartOfTensor:input name:nil];
        MPSDataType floatType = re.dataType;

        MPSGraphTensor* magnitude = [ctx.graph realPartOfTensor:[ctx.graph absoluteWithTensor:input name:nil]
                                                           name:nil];

        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:floatType];
        MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:floatType];
        MPSGraphTensor* is_zero = [ctx.graph equalWithPrimaryTensor:magnitude secondaryTensor:zero name:nil];
        MPSGraphTensor* safe_mag = [ctx.graph selectWithPredicateTensor:is_zero
                                                    truePredicateTensor:one
                                                   falsePredicateTensor:magnitude
                                                                   name:nil];

        MPSGraphTensor* norm_re = [ctx.graph divisionWithPrimaryTensor:re secondaryTensor:safe_mag name:nil];
        MPSGraphTensor* norm_im = [ctx.graph divisionWithPrimaryTensor:im secondaryTensor:safe_mag name:nil];
        norm_re = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_re
                                                  name:nil];
        norm_im = [ctx.graph selectWithPredicateTensor:is_zero
                                   truePredicateTensor:zero
                                  falsePredicateTensor:norm_im
                                                  name:nil];

        result = [ctx.graph complexTensorWithRealTensor:norm_re imaginaryTensor:norm_im name:nil];
    }

    return Result(ctx, result, "sign");
}
REGISTER_MPS_OP("stablehlo.sign", HandleSign);
REGISTER_MLIR_UNARY_OP("stablehlo.is_finite", isFinite, is_finite);
REGISTER_MLIR_UNARY_OP("chlo.square", square, chlo_square);
REGISTER_MLIR_UNARY_OP("stablehlo.ceil", ceil, ceil);
REGISTER_MLIR_UNARY_OP("stablehlo.round_nearest_even", rint, round_nearest_even);
REGISTER_MLIR_UNARY_OP("stablehlo.cosine", cos, cosine);
REGISTER_MLIR_UNARY_OP("stablehlo.sine", sin, sine);
REGISTER_MLIR_UNARY_OP("stablehlo.tan", tan, tan);
REGISTER_MLIR_UNARY_OP("chlo.asin", asin, asin);
REGISTER_MLIR_UNARY_OP("chlo.acos", acos, acos);
REGISTER_MLIR_UNARY_OP("chlo.sinh", sinh, sinh);
REGISTER_MLIR_UNARY_OP("chlo.cosh", cosh, cosh);
REGISTER_MLIR_UNARY_OP("chlo.asinh", asinh, asinh);
REGISTER_MLIR_UNARY_OP("chlo.acosh", acosh, acosh);
REGISTER_MLIR_UNARY_OP("chlo.atanh", atanh, atanh);

REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.erf", erf, mhlo_erf);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asin", asin, mhlo_asin);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acos", acos, mhlo_acos);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.sinh", sinh, mhlo_sinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.cosh", cosh, mhlo_cosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.asinh", asinh, mhlo_asinh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.acosh", acosh, mhlo_acosh);
REGISTER_CUSTOM_CALL_UNARY_OP("mhlo.atanh", atanh, mhlo_atanh);

static ProcessResult HandleReal(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("real: missing input tensor");
    MPSGraphTensor* result = [ctx.graph realPartOfTensor:input name:nil];
    return Result(ctx, result, "real");
}
REGISTER_MPS_OP("stablehlo.real", HandleReal);

static ProcessResult HandleImag(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("imag: missing input tensor");
    MPSGraphTensor* result = [ctx.graph imaginaryPartOfTensor:input name:nil];
    return Result(ctx, result, "imag");
}
REGISTER_MPS_OP("stablehlo.imag", HandleImag);

static ProcessResult HandleComplex(HandlerContext& ctx) {
    MPSGraphTensor* real = GetInputTensor(ctx, 0);
    MPSGraphTensor* imag = GetInputTensor(ctx, 1);
    if (!real || !imag)
        return ProcessResult::Error("complex: missing input tensor");
    MPSGraphTensor* result = [ctx.graph complexTensorWithRealTensor:real imaginaryTensor:imag name:nil];
    return Result(ctx, result, "complex");
}
REGISTER_MPS_OP("stablehlo.complex", HandleComplex);

static ProcessResult HandleExpm1(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("exponential_minus_one: missing input tensor");
    MPSGraphTensor* exp_x = [ctx.graph exponentWithTensor:input name:nil];
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* result = [ctx.graph subtractionWithPrimaryTensor:exp_x secondaryTensor:one name:nil];
    return Result(ctx, result, "exponential_minus_one");
}
REGISTER_MPS_OP("stablehlo.exponential_minus_one", HandleExpm1);

static ProcessResult HandleLogPlusOne(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("log_plus_one: missing input tensor");

    const bool isComplex =
        (input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16);
    auto scalarConst = [&](double value) -> MPSGraphTensor* {
        if (isComplex) {
            return [ctx.graph constantWithRealPart:value imaginaryPart:0.0 dataType:input.dataType];
        }
        return [ctx.graph constantWithScalar:value dataType:input.dataType];
    };

    MPSGraphTensor* one = scalarConst(1.0);
    MPSGraphTensor* onePlusX = [ctx.graph additionWithPrimaryTensor:input secondaryTensor:one name:nil];
    MPSGraphTensor* logPath = [ctx.graph logarithmWithTensor:onePlusX name:nil];

    MPSGraphTensor* magnitude = [ctx.graph absoluteWithTensor:input name:nil];
    if (isComplex) {
        magnitude = [ctx.graph realPartOfTensor:magnitude name:nil];
    }

    const double epsScalar = (magnitude.dataType == MPSDataTypeFloat16) ? 1e-2 : 1e-4;
    MPSGraphTensor* eps = [ctx.graph constantWithScalar:epsScalar dataType:magnitude.dataType];
    MPSGraphTensor* isSmall = [ctx.graph lessThanOrEqualToWithPrimaryTensor:magnitude
                                                            secondaryTensor:eps
                                                                       name:nil];

    MPSGraphTensor* x = input;
    MPSGraphTensor* x2 = [ctx.graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];
    MPSGraphTensor* x3 = [ctx.graph multiplicationWithPrimaryTensor:x2 secondaryTensor:x name:nil];
    MPSGraphTensor* x4 = [ctx.graph multiplicationWithPrimaryTensor:x3 secondaryTensor:x name:nil];
    MPSGraphTensor* x5 = [ctx.graph multiplicationWithPrimaryTensor:x4 secondaryTensor:x name:nil];

    MPSGraphTensor* term2 = [ctx.graph multiplicationWithPrimaryTensor:x2
                                                       secondaryTensor:scalarConst(-0.5)
                                                                  name:nil];
    MPSGraphTensor* term3 = [ctx.graph multiplicationWithPrimaryTensor:x3
                                                       secondaryTensor:scalarConst(1.0 / 3.0)
                                                                  name:nil];
    MPSGraphTensor* term4 = [ctx.graph multiplicationWithPrimaryTensor:x4
                                                       secondaryTensor:scalarConst(-0.25)
                                                                  name:nil];
    MPSGraphTensor* term5 = [ctx.graph multiplicationWithPrimaryTensor:x5
                                                       secondaryTensor:scalarConst(0.2)
                                                                  name:nil];

    MPSGraphTensor* series = [ctx.graph additionWithPrimaryTensor:x secondaryTensor:term2 name:nil];
    series = [ctx.graph additionWithPrimaryTensor:series secondaryTensor:term3 name:nil];
    series = [ctx.graph additionWithPrimaryTensor:series secondaryTensor:term4 name:nil];
    series = [ctx.graph additionWithPrimaryTensor:series secondaryTensor:term5 name:nil];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:isSmall
                                              truePredicateTensor:series
                                             falsePredicateTensor:logPath
                                                             name:nil];
    return Result(ctx, result, "log_plus_one");
}
REGISTER_MPS_OP("stablehlo.log_plus_one", HandleLogPlusOne);

static ProcessResult HandleErfInv(HandlerContext& ctx) {
    MPSGraphTensor* x = GetInputTensor(ctx, 0);
    if (!x)
        return ProcessResult::Error("erf_inv: missing input tensor");

    MPSDataType dtype = x.dataType;

    double a = 0.140012;
    double two_over_pi_a = 2.0 / (M_PI * a);

    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:dtype];
    MPSGraphTensor* half = [ctx.graph constantWithScalar:0.5 dataType:dtype];
    MPSGraphTensor* const_a = [ctx.graph constantWithScalar:a dataType:dtype];
    MPSGraphTensor* const_two_pi_a = [ctx.graph constantWithScalar:two_over_pi_a dataType:dtype];

    MPSGraphTensor* x_sq = [ctx.graph multiplicationWithPrimaryTensor:x secondaryTensor:x name:nil];

    MPSGraphTensor* one_minus_x_sq = [ctx.graph subtractionWithPrimaryTensor:one
                                                             secondaryTensor:x_sq
                                                                        name:nil];

    MPSGraphTensor* epsilon = [ctx.graph constantWithScalar:1e-7 dataType:dtype];
    MPSGraphTensor* clamped = [ctx.graph maximumWithPrimaryTensor:one_minus_x_sq
                                                  secondaryTensor:epsilon
                                                             name:nil];

    MPSGraphTensor* log_term = [ctx.graph logarithmWithTensor:clamped name:nil];

    MPSGraphTensor* half_log = [ctx.graph multiplicationWithPrimaryTensor:log_term
                                                          secondaryTensor:half
                                                                     name:nil];
    MPSGraphTensor* t = [ctx.graph additionWithPrimaryTensor:const_two_pi_a
                                             secondaryTensor:half_log
                                                        name:nil];

    MPSGraphTensor* t_sq = [ctx.graph multiplicationWithPrimaryTensor:t secondaryTensor:t name:nil];

    MPSGraphTensor* log_over_a = [ctx.graph divisionWithPrimaryTensor:log_term
                                                      secondaryTensor:const_a
                                                                 name:nil];

    MPSGraphTensor* inner = [ctx.graph subtractionWithPrimaryTensor:t_sq secondaryTensor:log_over_a name:nil];

    MPSGraphTensor* sqrt_inner = [ctx.graph squareRootWithTensor:inner name:nil];

    MPSGraphTensor* diff = [ctx.graph subtractionWithPrimaryTensor:sqrt_inner secondaryTensor:t name:nil];

    MPSGraphTensor* abs_result = [ctx.graph squareRootWithTensor:diff name:nil];

    MPSGraphTensor* sign_x = [ctx.graph signWithTensor:x name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign_x
                                                        secondaryTensor:abs_result
                                                                   name:nil];
    return Result(ctx, result, "erf_inv");
}
REGISTER_MPS_OP("chlo.erf_inv", HandleErfInv);

static ProcessResult HandleCbrt(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("cbrt: missing input tensor");
    MPSGraphTensor* abs_input = [ctx.graph absoluteWithTensor:input name:nil];
    MPSGraphTensor* third = [ctx.graph constantWithScalar:(1.0 / 3.0) dataType:input.dataType];
    MPSGraphTensor* pow_result = [ctx.graph powerWithPrimaryTensor:abs_input secondaryTensor:third name:nil];
    MPSGraphTensor* sign = [ctx.graph signWithTensor:input name:nil];
    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign
                                                        secondaryTensor:pow_result
                                                                   name:nil];
    return Result(ctx, result, "cbrt");
}
REGISTER_MPS_OP("stablehlo.cbrt", HandleCbrt);

REGISTER_MLIR_UNARY_OP("stablehlo.logistic", sigmoid, logistic);

static ProcessResult HandleCountLeadingZeros(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("count_leading_zeros: missing input tensor");

    MPSDataType dtype = input.dataType;
    int bits = 0;
    if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8)
        bits = 8;
    else if (dtype == MPSDataTypeInt16 || dtype == MPSDataTypeUInt16)
        bits = 16;
    else if (dtype == MPSDataTypeInt32 || dtype == MPSDataTypeUInt32)
        bits = 32;
    else
        return ProcessResult::Error("count_leading_zeros: unsupported dtype (only 8/16/32-bit)");

    MPSGraphTensor* x = input;
    if (dtype != MPSDataTypeInt32) {
        if (dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeUInt16 || dtype == MPSDataTypeUInt32) {
            x = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
        } else {

            x = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
            if (bits == 8) {
                MPSGraphTensor* mask = [ctx.graph constantWithScalar:0xFF dataType:MPSDataTypeInt32];
                x = [ctx.graph bitwiseANDWithPrimaryTensor:x secondaryTensor:mask name:nil];
            } else if (bits == 16) {
                MPSGraphTensor* mask = [ctx.graph constantWithScalar:0xFFFF dataType:MPSDataTypeInt32];
                x = [ctx.graph bitwiseANDWithPrimaryTensor:x secondaryTensor:mask name:nil];
            }
        }
    }

    MPSGraphTensor* zero_i32 = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* is_zero = [ctx.graph equalWithPrimaryTensor:x secondaryTensor:zero_i32 name:nil];

    MPSGraphTensor* is_negative = [ctx.graph lessThanWithPrimaryTensor:x secondaryTensor:zero_i32 name:nil];

    MPSGraphTensor* abs_x = [ctx.graph absoluteWithTensor:x name:nil];

    MPSGraphTensor* one_i32 = [ctx.graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* safe_x = [ctx.graph maximumWithPrimaryTensor:abs_x secondaryTensor:one_i32 name:nil];

    MPSGraphTensor* as_float = [ctx.graph castTensor:safe_x toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* log2_val = [ctx.graph logarithmBase2WithTensor:as_float name:nil];
    MPSGraphTensor* msb_pos_f = [ctx.graph floorWithTensor:log2_val name:nil];
    MPSGraphTensor* msb_pos = [ctx.graph castTensor:msb_pos_f toType:MPSDataTypeInt32 name:nil];

    MPSGraphTensor* thirty = [ctx.graph constantWithScalar:30 dataType:MPSDataTypeInt32];
    msb_pos = [ctx.graph minimumWithPrimaryTensor:msb_pos secondaryTensor:thirty name:nil];
    msb_pos = [ctx.graph maximumWithPrimaryTensor:msb_pos secondaryTensor:zero_i32 name:nil];

    MPSGraphTensor* one_shifted = [ctx.graph bitwiseLeftShiftWithPrimaryTensor:one_i32
                                                               secondaryTensor:msb_pos
                                                                          name:nil];
    MPSGraphTensor* too_high = [ctx.graph greaterThanWithPrimaryTensor:one_shifted
                                                       secondaryTensor:abs_x
                                                                  name:nil];
    MPSGraphTensor* correction = [ctx.graph selectWithPredicateTensor:too_high
                                                  truePredicateTensor:one_i32
                                                 falsePredicateTensor:zero_i32
                                                                 name:nil];
    msb_pos = [ctx.graph subtractionWithPrimaryTensor:msb_pos secondaryTensor:correction name:nil];

    MPSGraphTensor* bits_minus_1 = [ctx.graph constantWithScalar:(bits - 1) dataType:MPSDataTypeInt32];
    MPSGraphTensor* clz_val = [ctx.graph subtractionWithPrimaryTensor:bits_minus_1
                                                      secondaryTensor:msb_pos
                                                                 name:nil];

    MPSGraphTensor* total_bits = [ctx.graph constantWithScalar:bits dataType:MPSDataTypeInt32];
    MPSGraphTensor* zero_clz = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:is_zero
                                              truePredicateTensor:total_bits
                                             falsePredicateTensor:clz_val
                                                             name:nil];
    result = [ctx.graph selectWithPredicateTensor:is_negative
                              truePredicateTensor:zero_clz
                             falsePredicateTensor:result
                                             name:nil];

    if (dtype != MPSDataTypeInt32) {
        result = [ctx.graph castTensor:result toType:dtype name:nil];
    }

    return Result(ctx, result, "count_leading_zeros");
}
REGISTER_MPS_OP("stablehlo.count_leading_zeros", HandleCountLeadingZeros);

static ProcessResult HandleConj(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("conj: missing input tensor");
    MPSGraphTensor* re = [ctx.graph realPartOfTensor:input name:nil];
    MPSGraphTensor* im = [ctx.graph imaginaryPartOfTensor:input name:nil];
    MPSGraphTensor* neg_im = [ctx.graph negativeWithTensor:im name:nil];
    MPSGraphTensor* result = [ctx.graph complexTensorWithRealTensor:re imaginaryTensor:neg_im name:nil];
    return Result(ctx, result, "conj");
}
REGISTER_MPS_OP("chlo.conj", HandleConj);

static ProcessResult HandleErfc(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("erfc: missing input tensor");
    MPSGraphTensor* erf_x = [ctx.graph erfWithTensor:input name:nil];
    MPSGraphTensor* one = [ctx.graph constantWithScalar:1.0 dataType:input.dataType];
    MPSGraphTensor* result = [ctx.graph subtractionWithPrimaryTensor:one secondaryTensor:erf_x name:nil];
    return Result(ctx, result, "erfc");
}
REGISTER_MPS_OP("chlo.erfc", HandleErfc);

static ProcessResult HandleIsInf(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("is_inf: missing input tensor");
    MPSGraphTensor* result = [ctx.graph isInfiniteWithTensor:input name:nil];
    return Result(ctx, result, "is_inf");
}
REGISTER_MPS_OP("chlo.is_inf", HandleIsInf);

static ProcessResult HandleIsNegInf(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("is_neg_inf: missing input tensor");
    MPSGraphTensor* is_inf = [ctx.graph isInfiniteWithTensor:input name:nil];
    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:input.dataType];
    MPSGraphTensor* is_neg = [ctx.graph lessThanWithPrimaryTensor:input secondaryTensor:zero name:nil];
    MPSGraphTensor* result = [ctx.graph logicalANDWithPrimaryTensor:is_inf secondaryTensor:is_neg name:nil];
    return Result(ctx, result, "is_neg_inf");
}
REGISTER_MPS_OP("chlo.is_neg_inf", HandleIsNegInf);

static ProcessResult HandleIsPosInf(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("is_pos_inf: missing input tensor");
    MPSGraphTensor* is_inf = [ctx.graph isInfiniteWithTensor:input name:nil];
    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:input.dataType];
    MPSGraphTensor* is_pos = [ctx.graph greaterThanWithPrimaryTensor:input secondaryTensor:zero name:nil];
    MPSGraphTensor* result = [ctx.graph logicalANDWithPrimaryTensor:is_inf secondaryTensor:is_pos name:nil];
    return Result(ctx, result, "is_pos_inf");
}
REGISTER_MPS_OP("chlo.is_pos_inf", HandleIsPosInf);

static ProcessResult HandleDigamma(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("digamma: missing input tensor");

    MPSDataType dtype = input.dataType;

    auto cst = [&](double v) -> MPSGraphTensor* { return [ctx.graph constantWithScalar:v dataType:dtype]; };

    MPSGraphTensor* zero = cst(0.0);
    MPSGraphTensor* is_neg = [ctx.graph lessThanWithPrimaryTensor:input secondaryTensor:zero name:nil];

    MPSGraphTensor* one = cst(1.0);
    MPSGraphTensor* reflected = [ctx.graph subtractionWithPrimaryTensor:one secondaryTensor:input name:nil];
    MPSGraphTensor* x = [ctx.graph selectWithPredicateTensor:is_neg
                                         truePredicateTensor:reflected
                                        falsePredicateTensor:input
                                                        name:nil];

    MPSGraphTensor* result_accum = cst(0.0);
    MPSGraphTensor* six = cst(6.0);
    for (int i = 0; i < 6; i++) {
        MPSGraphTensor* needs_shift = [ctx.graph lessThanWithPrimaryTensor:x secondaryTensor:six name:nil];
        MPSGraphTensor* recip = [ctx.graph reciprocalWithTensor:x name:nil];
        MPSGraphTensor* neg_recip = [ctx.graph negativeWithTensor:recip name:nil];
        result_accum = [ctx.graph selectWithPredicateTensor:needs_shift
                                        truePredicateTensor:[ctx.graph additionWithPrimaryTensor:result_accum
                                                                                 secondaryTensor:neg_recip
                                                                                            name:nil]
                                       falsePredicateTensor:result_accum
                                                       name:nil];
        x = [ctx.graph selectWithPredicateTensor:needs_shift
                             truePredicateTensor:[ctx.graph additionWithPrimaryTensor:x
                                                                      secondaryTensor:one
                                                                                 name:nil]
                            falsePredicateTensor:x
                                            name:nil];
    }

    MPSGraphTensor* log_x = [ctx.graph logarithmWithTensor:x name:nil];
    MPSGraphTensor* inv_x = [ctx.graph reciprocalWithTensor:x name:nil];
    MPSGraphTensor* inv_x2 = [ctx.graph multiplicationWithPrimaryTensor:inv_x secondaryTensor:inv_x name:nil];

    MPSGraphTensor* series = log_x;
    MPSGraphTensor* half_inv_x = [ctx.graph multiplicationWithPrimaryTensor:inv_x
                                                            secondaryTensor:cst(0.5)
                                                                       name:nil];
    series = [ctx.graph subtractionWithPrimaryTensor:series secondaryTensor:half_inv_x name:nil];

    MPSGraphTensor* term = [ctx.graph multiplicationWithPrimaryTensor:inv_x2
                                                      secondaryTensor:cst(1.0 / 12.0)
                                                                 name:nil];
    series = [ctx.graph subtractionWithPrimaryTensor:series secondaryTensor:term name:nil];

    MPSGraphTensor* inv_x4 = [ctx.graph multiplicationWithPrimaryTensor:inv_x2
                                                        secondaryTensor:inv_x2
                                                                   name:nil];
    term = [ctx.graph multiplicationWithPrimaryTensor:inv_x4 secondaryTensor:cst(1.0 / 120.0) name:nil];
    series = [ctx.graph additionWithPrimaryTensor:series secondaryTensor:term name:nil];

    MPSGraphTensor* inv_x6 = [ctx.graph multiplicationWithPrimaryTensor:inv_x4
                                                        secondaryTensor:inv_x2
                                                                   name:nil];
    term = [ctx.graph multiplicationWithPrimaryTensor:inv_x6 secondaryTensor:cst(1.0 / 252.0) name:nil];
    series = [ctx.graph subtractionWithPrimaryTensor:series secondaryTensor:term name:nil];

    MPSGraphTensor* pos_result = [ctx.graph additionWithPrimaryTensor:series
                                                      secondaryTensor:result_accum
                                                                 name:nil];

    MPSGraphTensor* pi = cst(M_PI);
    MPSGraphTensor* pi_x = [ctx.graph multiplicationWithPrimaryTensor:pi secondaryTensor:input name:nil];
    MPSGraphTensor* tan_pi_x = [ctx.graph tanWithTensor:pi_x name:nil];
    MPSGraphTensor* pi_cot = [ctx.graph divisionWithPrimaryTensor:pi secondaryTensor:tan_pi_x name:nil];
    MPSGraphTensor* neg_result = [ctx.graph subtractionWithPrimaryTensor:pos_result
                                                         secondaryTensor:pi_cot
                                                                    name:nil];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:is_neg
                                              truePredicateTensor:neg_result
                                             falsePredicateTensor:pos_result
                                                             name:nil];

    return Result(ctx, result, "digamma");
}
REGISTER_MPS_OP("chlo.digamma", HandleDigamma);

static ProcessResult HandleLgamma(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("lgamma: missing input tensor");

    MPSDataType dtype = input.dataType;

    auto cst = [&](double v) -> MPSGraphTensor* { return [ctx.graph constantWithScalar:v dataType:dtype]; };

    MPSGraphTensor* half = cst(0.5);
    MPSGraphTensor* one = cst(1.0);

    MPSGraphTensor* is_small = [ctx.graph lessThanWithPrimaryTensor:input secondaryTensor:half name:nil];
    MPSGraphTensor* reflected_x = [ctx.graph subtractionWithPrimaryTensor:one secondaryTensor:input name:nil];
    MPSGraphTensor* x = [ctx.graph selectWithPredicateTensor:is_small
                                         truePredicateTensor:reflected_x
                                        falsePredicateTensor:input
                                                        name:nil];

    double lanczos_g = 7.0;
    double lanczos_coeffs[] = {0.99999999999980993,  676.5203681218851,     -1259.1392167224028,
                               771.32342877765313,   -176.61502916214059,   12.507343278686905,
                               -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7};

    MPSGraphTensor* t = [ctx.graph additionWithPrimaryTensor:x secondaryTensor:cst(lanczos_g - 0.5) name:nil];

    MPSGraphTensor* series = cst(lanczos_coeffs[0]);
    for (int i = 1; i < 9; i++) {
        MPSGraphTensor* denom = [ctx.graph additionWithPrimaryTensor:x
                                                     secondaryTensor:cst((double)(i - 1))
                                                                name:nil];
        MPSGraphTensor* term = [ctx.graph divisionWithPrimaryTensor:cst(lanczos_coeffs[i])
                                                    secondaryTensor:denom
                                                               name:nil];
        series = [ctx.graph additionWithPrimaryTensor:series secondaryTensor:term name:nil];
    }

    double log_sqrt_2pi = 0.5 * log(2.0 * M_PI);
    MPSGraphTensor* log_t = [ctx.graph logarithmWithTensor:t name:nil];
    MPSGraphTensor* x_minus_half = [ctx.graph subtractionWithPrimaryTensor:x secondaryTensor:half name:nil];
    MPSGraphTensor* log_series = [ctx.graph logarithmWithTensor:series name:nil];

    MPSGraphTensor* lgamma_val = cst(log_sqrt_2pi);
    MPSGraphTensor* term1 = [ctx.graph multiplicationWithPrimaryTensor:x_minus_half
                                                       secondaryTensor:log_t
                                                                  name:nil];
    lgamma_val = [ctx.graph additionWithPrimaryTensor:lgamma_val secondaryTensor:term1 name:nil];
    lgamma_val = [ctx.graph subtractionWithPrimaryTensor:lgamma_val secondaryTensor:t name:nil];
    lgamma_val = [ctx.graph additionWithPrimaryTensor:lgamma_val secondaryTensor:log_series name:nil];

    MPSGraphTensor* pi = cst(M_PI);
    MPSGraphTensor* log_pi = cst(log(M_PI));
    MPSGraphTensor* pi_x = [ctx.graph multiplicationWithPrimaryTensor:pi secondaryTensor:input name:nil];
    MPSGraphTensor* sin_pi_x = [ctx.graph sinWithTensor:pi_x name:nil];
    MPSGraphTensor* abs_sin = [ctx.graph absoluteWithTensor:sin_pi_x name:nil];

    MPSGraphTensor* eps = cst(1e-7);
    abs_sin = [ctx.graph maximumWithPrimaryTensor:abs_sin secondaryTensor:eps name:nil];
    MPSGraphTensor* log_abs_sin = [ctx.graph logarithmWithTensor:abs_sin name:nil];

    MPSGraphTensor* reflected_result = [ctx.graph subtractionWithPrimaryTensor:log_pi
                                                               secondaryTensor:log_abs_sin
                                                                          name:nil];
    reflected_result = [ctx.graph subtractionWithPrimaryTensor:reflected_result
                                               secondaryTensor:lgamma_val
                                                          name:nil];

    MPSGraphTensor* result = [ctx.graph selectWithPredicateTensor:is_small
                                              truePredicateTensor:reflected_result
                                             falsePredicateTensor:lgamma_val
                                                             name:nil];

    return Result(ctx, result, "lgamma");
}
REGISTER_MPS_OP("chlo.lgamma", HandleLgamma);

static ProcessResult HandleBesselI1e(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("bessel_i1e: missing input tensor");

    MPSDataType dtype = input.dataType;
    auto cst = [&](double v) -> MPSGraphTensor* { return [ctx.graph constantWithScalar:v dataType:dtype]; };

    MPSGraphTensor* abs_x = [ctx.graph absoluteWithTensor:input name:nil];
    MPSGraphTensor* sign_x = [ctx.graph signWithTensor:input name:nil];

    MPSGraphTensor* half_x = [ctx.graph multiplicationWithPrimaryTensor:abs_x
                                                        secondaryTensor:cst(0.5)
                                                                   name:nil];
    MPSGraphTensor* y_small = [ctx.graph subtractionWithPrimaryTensor:half_x
                                                      secondaryTensor:cst(2.0)
                                                                 name:nil];

    double A[] = {2.77791411276104639959E-18,  -2.11142121435816608115E-17, 1.55363195773620046921E-16,
                  -1.10559694773538630805E-15, 7.60068429473891120186E-15,  -5.04218550472791168711E-14,
                  3.22092999180461902994E-13,  -1.98164750424335182583E-12, 1.17117667681494667293E-11,
                  -6.66890520414789543498E-11, 3.62938027230514540899E-10,  -1.88552003064791069494E-9,
                  9.31085715080956094421E-9,   -4.36305746013711072900E-8,  1.93266646512567653440E-7,
                  -8.03631924445498759498E-7,  3.12665082695161709920E-6,   -1.12978710076553605780E-5,
                  3.75070535427644498820E-5,   -1.12440116662891055498E-4,  2.99215419443639548890E-4,
                  -6.87554792585498210570E-4,  1.31199293002609576940E-3,   -1.87579476071609936160E-3,
                  1.46631637804605107430E-3,   1.33565495221819659230E-4,   -5.86470789089560078560E-4,
                  3.44405814992460059170E-4,   -1.42670964488686504710E-4};
    int nA = 29;

    MPSGraphTensor* b0 = cst(0.0);
    MPSGraphTensor* b1 = cst(0.0);
    MPSGraphTensor* two_y = [ctx.graph multiplicationWithPrimaryTensor:y_small
                                                       secondaryTensor:cst(2.0)
                                                                  name:nil];
    for (int i = nA - 1; i >= 0; i--) {
        MPSGraphTensor* b2 = b1;
        b1 = b0;
        MPSGraphTensor* prod = [ctx.graph multiplicationWithPrimaryTensor:two_y secondaryTensor:b1 name:nil];
        b0 = [ctx.graph subtractionWithPrimaryTensor:prod secondaryTensor:b2 name:nil];
        b0 = [ctx.graph additionWithPrimaryTensor:b0 secondaryTensor:cst(A[i]) name:nil];
    }
    MPSGraphTensor* y_b1 = [ctx.graph multiplicationWithPrimaryTensor:y_small secondaryTensor:b1 name:nil];
    MPSGraphTensor* cheb_small = [ctx.graph subtractionWithPrimaryTensor:b0 secondaryTensor:y_b1 name:nil];
    MPSGraphTensor* result_small = [ctx.graph multiplicationWithPrimaryTensor:cheb_small
                                                              secondaryTensor:abs_x
                                                                         name:nil];
    result_small = [ctx.graph multiplicationWithPrimaryTensor:result_small secondaryTensor:cst(0.5) name:nil];

    double B[] = {7.51729631084210481353E-18,  4.41434832307170791151E-18,  -4.65030536848935832153E-17,
                  -3.20952592199342395980E-17, 2.96262899764595013876E-16,  1.82825063549809726860E-16,
                  -1.80657598807301887150E-15, -9.88641919095498401150E-16, 1.06975938468674568490E-14,
                  5.04879099815498641360E-15,  -6.39149612015498801390E-14, -2.45025863512195210280E-14,
                  4.08837690478558987380E-13,  1.11487707916665755640E-13,  -3.04504304168929097590E-12,
                  -5.17853898272741988400E-13, 2.70277090027505620910E-11,  2.51386592793505379390E-12,
                  -3.07267228758395150900E-10, -1.22675164753259048370E-12, 5.27037146400985042200E-9,
                  -2.77626976709066295390E-9,  -2.07790395505109178060E-7,  4.11437181870938413730E-7,
                  2.92488700914420261550E-5,   -1.56025548101744723380E-4,  4.23614013938649619520E-3,
                  5.10478066034814003120E-1};
    int nB = 28;

    MPSGraphTensor* y_large = [ctx.graph divisionWithPrimaryTensor:cst(32.0) secondaryTensor:abs_x name:nil];
    y_large = [ctx.graph subtractionWithPrimaryTensor:y_large secondaryTensor:cst(2.0) name:nil];

    b0 = cst(0.0);
    b1 = cst(0.0);
    two_y = [ctx.graph multiplicationWithPrimaryTensor:y_large secondaryTensor:cst(2.0) name:nil];
    for (int i = nB - 1; i >= 0; i--) {
        MPSGraphTensor* b2 = b1;
        b1 = b0;
        MPSGraphTensor* prod = [ctx.graph multiplicationWithPrimaryTensor:two_y secondaryTensor:b1 name:nil];
        b0 = [ctx.graph subtractionWithPrimaryTensor:prod secondaryTensor:b2 name:nil];
        b0 = [ctx.graph additionWithPrimaryTensor:b0 secondaryTensor:cst(B[i]) name:nil];
    }
    MPSGraphTensor* y_b1_large = [ctx.graph multiplicationWithPrimaryTensor:y_large
                                                            secondaryTensor:b1
                                                                       name:nil];
    MPSGraphTensor* cheb_large = [ctx.graph subtractionWithPrimaryTensor:b0
                                                         secondaryTensor:y_b1_large
                                                                    name:nil];
    MPSGraphTensor* rsqrt_x = [ctx.graph reciprocalSquareRootWithTensor:abs_x name:nil];
    MPSGraphTensor* result_large = [ctx.graph multiplicationWithPrimaryTensor:cheb_large
                                                              secondaryTensor:rsqrt_x
                                                                         name:nil];

    MPSGraphTensor* eight = cst(8.0);
    MPSGraphTensor* is_small = [ctx.graph lessThanOrEqualToWithPrimaryTensor:abs_x
                                                             secondaryTensor:eight
                                                                        name:nil];
    MPSGraphTensor* unsigned_result = [ctx.graph selectWithPredicateTensor:is_small
                                                       truePredicateTensor:result_small
                                                      falsePredicateTensor:result_large
                                                                      name:nil];

    MPSGraphTensor* result = [ctx.graph multiplicationWithPrimaryTensor:sign_x
                                                        secondaryTensor:unsigned_result
                                                                   name:nil];

    return Result(ctx, result, "bessel_i1e");
}
REGISTER_MPS_OP("chlo.bessel_i1e", HandleBesselI1e);

} // namespace jax_metallib
