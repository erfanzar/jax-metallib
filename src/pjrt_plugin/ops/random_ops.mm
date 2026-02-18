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

namespace jax_metallib {

static ProcessResult HandleRng(HandlerContext& ctx) {
    auto rngOp = mlir::dyn_cast<mlir::stablehlo::RngOp>(ctx.op);
    if (!rngOp) {
        return ProcessResult::Error("rng: expected RngOp");
    }

    return ProcessResult::Error(
        "rng: stablehlo.rng is intentionally unsupported on MPS (determinism/algorithm semantics "
        "not implemented).\n\n" +
        UnsupportedOpsMessage({"stablehlo.rng"}));
}
REGISTER_MPS_OP("stablehlo.rng", HandleRng);

static ProcessResult HandleRngBitGenerator(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::RngBitGeneratorOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("rng_bit_generator: expected RngBitGeneratorOp");
    }

    return ProcessResult::Error(
        "rng_bit_generator: stablehlo.rng_bit_generator is intentionally unsupported on MPS "
        "(determinism/algorithm semantics not implemented).\n\n" +
        UnsupportedOpsMessage({"stablehlo.rng_bit_generator"}));
}
REGISTER_MPS_OP("stablehlo.rng_bit_generator", HandleRngBitGenerator);

} // namespace jax_metallib
