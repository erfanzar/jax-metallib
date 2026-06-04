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
    // operation is dispatched to MPSGraph's philox-based uniform generator.
    int64_t seed = 0;
    (void)rngOp;

    MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor philoxRandomDescriptorWithSeed:seed];
    if (!desc) {
        return ProcessResult::Error("rng: failed to create random descriptor");
    }

    MPSGraphTensor* uniform = [ctx.graph randomUniformWithDescriptor:desc shape:shape dataType:dtype name:nil];
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

    int64_t seed = op.getRngAlgorithm() == mlir::stablehlo::RngAlgorithm::DEFAULT
                       ? 0
                       : static_cast<int64_t>(reinterpret_cast<uintptr_t>(ctx.op));

    MPSGraphRandomOpDescriptor* desc = [MPSGraphRandomOpDescriptor philoxRandomDescriptorWithSeed:seed];
    if (!desc) {
        return ProcessResult::Error("rng_bit_generator: failed to create random descriptor");
    }

    MPSGraphTensor* randomBits = [ctx.graph randomUniformWithDescriptor:desc shape:shape dataType:outputDtype name:nil];
    if (!randomBits) {
        return ProcessResult::Error("rng_bit_generator: failed to generate random bits");
    }

    NSMutableArray<NSNumber*>* stateShape = [NSMutableArray array];
    [stateShape addObject:@2];
    for (NSNumber* dim in shape) {
        [stateShape addObject:dim];
    }
    MPSGraphTensor* updatedState = [ctx.graph constantWithScalar:0 shape:stateShape dataType:MPSDataTypeInt64];

    SetOutputTensor(ctx.values, ctx.op, updatedState, 0);
    SetOutputTensor(ctx.values, ctx.op, randomBits, 1);

    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.rng_bit_generator", HandleRngBitGenerator);

} // namespace jax_metallib
