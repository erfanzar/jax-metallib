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

namespace {

bool IsElementwiseMapDimensions(llvm::ArrayRef<int64_t> dims, int64_t rank) {
    if ((int64_t)dims.size() != rank) {
        return false;
    }
    for (int64_t i = 0; i < rank; ++i) {
        if (dims[(size_t)i] != i) {
            return false;
        }
    }
    return true;
}

} // namespace

static ProcessResult HandleMap(HandlerContext& ctx) {
    auto mapOp = mlir::dyn_cast<mlir::stablehlo::MapOp>(ctx.op);
    if (!mapOp) {
        return ProcessResult::Error("map: expected MapOp");
    }
    if (!ctx.processBlock) {
        return ProcessResult::Error("map: missing block processor in context");
    }
    if (ctx.op->getNumResults() != 1) {
        return ProcessResult::Error("map: only single-result map is supported");
    }
    if (ctx.op->getNumOperands() == 0) {
        return ProcessResult::Error("map: expected at least one input");
    }

    auto firstType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
    if (!firstType) {
        return ProcessResult::Error("map: expected ranked tensor inputs");
    }
    int64_t rank = firstType.getRank();
    if (!IsElementwiseMapDimensions(mapOp.getDimensions(), rank)) {
        return ProcessResult::Error(
            "map: only elementwise map is supported (dimensions must be [0..rank-1])");
    }

    if (mapOp.getComputation().empty()) {
        return ProcessResult::Error("map: computation region is empty");
    }
    mlir::Block& block = mapOp.getComputation().front();
    if (block.getNumArguments() != ctx.op->getNumOperands()) {
        return ProcessResult::Error("map: region argument arity mismatch");
    }

    ValueMap regionValues = ctx.values;
    for (unsigned i = 0; i < ctx.op->getNumOperands(); ++i) {
        MPSGraphTensor* t = GetInputTensor(ctx, i);
        if (!t) {
            return ProcessResult::Error("map: missing input tensor at operand " + std::to_string(i));
        }
        void* key = block.getArgument(i).getAsOpaquePointer();
        regionValues[key] = t;
        if (ctx.variables) {
            auto it = ctx.variables->find(ctx.op->getOperand(i).getAsOpaquePointer());
            if (it != ctx.variables->end()) {
                (*ctx.variables)[key] = it->second;
            }
        }
    }

    HandlerContext regionCtx(ctx.graph, nullptr, regionValues, ctx.module, ctx.depth + 1, ctx.processBlock);
    regionCtx.variables = ctx.variables;
    ProcessResult regionResult = ctx.processBlock(regionCtx, block);
    if (!regionResult.ok()) {
        return regionResult;
    }
    if (regionResult.return_values.size() != 1) {
        return ProcessResult::Error("map: region must return exactly one value");
    }

    MPSGraphTensor* result = GetTensor(regionValues, regionResult.return_values[0]);
    if (!result) {
        return ProcessResult::Error("map: result tensor not found");
    }

    NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
    if (expectedShape && result.shape && ![expectedShape isEqualToArray:result.shape]) {
        result = [ctx.graph reshapeTensor:result withShape:expectedShape name:nil];
    }

    return Result(ctx, result, "map");
}
REGISTER_MPS_OP("stablehlo.map", HandleMap);

} // namespace jax_metallib
