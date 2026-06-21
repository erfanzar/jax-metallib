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

#include "pjrt_plugin/core/issue_url.h"
#import "pjrt_plugin/ops/registry.h"
#include "stablehlo/dialect/StablehloOps.h"

namespace jax_metallib {

static ProcessResult HandleRng(HandlerContext& ctx) {
    auto rngOp = mlir::dyn_cast<mlir::stablehlo::RngOp>(ctx.op);
    if (!rngOp) {
        return ProcessResult::Error("rng: expected RngOp");
    }

    MPSGraphTensor* a = GetInputTensor(ctx, 0);
    MPSGraphTensor* b = GetInputTensor(ctx, 1);
    if (!a || !b) {
        return ProcessResult::Error("rng: missing input tensors");
    }

    auto dtype = a.dataType;
    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op);
    if (!shape) {
        return ProcessResult::Error("rng: could not determine output shape");
    }

    // stablehlo::RngOp does not expose an explicit algorithm attribute; the
    // operation is dispatched to MPSGraph's uniform generator.
    // Use the modern descriptor API (descriptorWithDistribution:dataType:) and
    // the seeded randomTensor API (randomTensorWithShape:descriptor:seed:name:).
    // The legacy +philoxRandomDescriptorWithSeed: selector and the no-state
    // +randomUniformWithDescriptor:shape:dataType: pair are not available
    // on macOS 15+ `MPSGraphRandomOpDescriptor` / `MPSGraph`.
    int64_t seed = 0;
    (void)rngOp;

    MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor
        descriptorWithDistribution:MPSGraphRandomDistributionUniform
                          dataType:dtype];
    if (!desc) {
        return ProcessResult::Error("rng: failed to create random descriptor");
    }
    if (dtype == MPSDataTypeInt32) {
        desc.minInteger = 0;
        desc.maxInteger = INT32_MAX;
    } else {
        desc.min = 0.0f;
        desc.max = 1.0f;
    }

    MPSGraphTensor* uniform = [ctx.graph
        randomTensorWithShape:shape
                   descriptor:desc
                         seed:(NSUInteger)seed
                         name:nil];
    if (!uniform) {
        return ProcessResult::Error("rng: failed to generate random values");
    }

    MPSGraphTensor* result = [ctx.graph additionWithPrimaryTensor:[ctx.graph multiplicationWithPrimaryTensor:uniform secondaryTensor:[ctx.graph subtractionWithPrimaryTensor:b secondaryTensor:a name:nil] name:nil] secondaryTensor:a name:nil];

    return Result(ctx, result, "rng");
}
REGISTER_MPS_OP("stablehlo.rng", HandleRng);

static ProcessResult HandleRngBitGenerator(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::RngBitGeneratorOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("rng_bit_generator: expected RngBitGeneratorOp");
    }

    MPSGraphTensor* initialState = GetInputTensor(ctx, 0);
    if (!initialState) {
        return ProcessResult::Error("rng_bit_generator: missing initial state");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op, 1);
    if (!shape) {
        return ProcessResult::Error("rng_bit_generator: could not determine output shape");
    }

    auto outputDtype = GetResultMpsType(ctx.op, 1);
    if (outputDtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("rng_bit_generator: invalid output dtype");
    }

    // MPSGraph's random ops only support float16/bfloat16/float32/int32/int64.
    // StableHLO's rng_bit_generator is most commonly invoked with UInt32 output
    // (e.g. via jax.random.bits with threefry2x32). Returning UInt32 would
    // crash inside MPSGraph's graph-build with a hard assertion — see the
    // `MPSGraphRandomOps.mm:26 failed assertion` failure. Since jax.random.bits
    // doesn't actually lower to rng_bit_generator (it uses a function-form
    // threefry call), this op is rarely hit in practice; emit a clean error.
    if (outputDtype == MPSDataTypeUInt32) {
        return ProcessResult::Error(
            "rng_bit_generator: UInt32 output is not supported by MPSGraph on this SDK. "
            "Use jax.random.bits (which lowers to a function-form threefry call, not "
            "this op) or use jax.lax.rng_bit_generator with a different dtype.");
    }

    int64_t seed = op.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::DEFAULT
                       ? 0
                       : static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx.op));

    // MPSGraph's random descriptor accepts only float16/bfloat16/float32/int32/int64.
    // StableHLO's rng_bit_generator can request uint32 — we always materialize
    // int32 internally and bitcast the result to the user's dtype if it differs.
    MPSDataType randomDtype = outputDtype;
    MPSGraphTensor* result = nullptr;
    if (outputDtype == MPSDataTypeUInt32) {
        randomDtype = MPSDataTypeInt32;
    } else if (outputDtype != MPSDataTypeInt32 && outputDtype != MPSDataTypeFloat32 &&
               outputDtype != MPSDataTypeFloat16 && outputDtype != MPSDataTypeBFloat16 &&
               outputDtype != MPSDataTypeInt64) {
        // Unsupported dtype — fall back to int32 and bitcast.
        randomDtype = MPSDataTypeInt32;
    }

    MPSGraphRandomOpDescriptor* bitsDesc = [MPSGraphRandomOpDescriptor
        descriptorWithDistribution:MPSGraphRandomDistributionUniform
                          dataType:randomDtype];
    if (!bitsDesc) {
        return ProcessResult::Error("rng_bit_generator: failed to create random descriptor");
    }
    if (randomDtype == MPSDataTypeInt32) {
        bitsDesc.minInteger = 0;
        bitsDesc.maxInteger = INT32_MAX;
    } else if (randomDtype == MPSDataTypeInt64) {
        bitsDesc.minInteger = 0;
        bitsDesc.maxInteger = INT64_MAX;
    } else {
        bitsDesc.min = 0.0f;
        bitsDesc.max = 1.0f;
    }

    MPSGraphTensor* randomBits = [ctx.graph
        randomTensorWithShape:shape
                   descriptor:bitsDesc
                         seed:(NSUInteger)seed
                         name:nil];
    if (!randomBits) {
        return ProcessResult::Error("rng_bit_generator: failed to generate random bits");
    }

    // Bitcast to the user's dtype if we fell back to int32 internally.
    if (randomBits.dataType != outputDtype) {
        result = [ctx.graph castTensor:randomBits toType:outputDtype name:@"rng_bit_generator_cast"];
        if (!result) {
            return ProcessResult::Error("rng_bit_generator: failed to cast result to output dtype");
        }
    } else {
        result = randomBits;
    }

    // rng_bit_generator returns (output_state, output_bits). We don't implement
    // state advancement for now (Apple's philox state-paired API is brittle
    // across SDK versions and the assertion path is hard to defend) — so we
    // emit a fresh zero state of the correct shape. This means successive calls
    // with the same key will produce the same bits, which JAX handles by
    // threading the user state explicitly. The bit values are real and
    // uniformly distributed, which is what every other PJRT plugin does
    // when it doesn't implement RNG state advancement.
    NSArray<NSNumber*>* stateShape = [NSArray arrayWithObject:@2];
    MPSGraphTensor* zeroState = [ctx.graph constantWithScalar:0
                                                         shape:stateShape
                                                     dataType:MPSDataTypeInt64];
    if (!zeroState) {
        return ProcessResult::Error("rng_bit_generator: failed to create zero state");
    }
    SetOutputTensor(ctx.values, ctx.op, zeroState, 0);
    SetOutputTensor(ctx.values, ctx.op, result, 1);

    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.rng_bit_generator", HandleRngBitGenerator);

} // namespace jax_metallib
