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

#import "pjrt_plugin/core/issue_url.h"
#import "pjrt_plugin/ops/registry.h"

#include <sstream>

namespace jax_metallib {

namespace {

int64_t GetNumReplicasFromModule(mlir::ModuleOp module) {
    if (!module) {
        return 1;
    }
    if (auto attr = module->getAttrOfType<mlir::IntegerAttr>("mhlo.num_replicas")) {
        return attr.getInt();
    }
    return 1;
}

bool IsSingleReplicaGroups(mlir::Attribute replicaGroupsAttr, std::string& errorOut) {
    // Newer StableHLO exposes getReplicaGroups() as a generic mlir::Attribute;
    // cast to the dense-elements form when present.
    auto replicaGroups = mlir::dyn_cast<mlir::DenseIntElementsAttr>(replicaGroupsAttr);
    if (!replicaGroups) {

        return true;
    }

    auto type = mlir::dyn_cast<mlir::RankedTensorType>(replicaGroups.getType());
    if (!type || type.getRank() != 2) {
        errorOut = "replica_groups must be a rank-2 tensor";
        return false;
    }
    auto shape = type.getShape();
    if (shape.size() != 2) {
        errorOut = "replica_groups must be a rank-2 tensor";
        return false;
    }
    int64_t groupSize = shape[1];
    if (groupSize != 1) {
        errorOut = "replica_groups has group_size=" + std::to_string(groupSize) +
                   " (multi-device collectives are not supported)";
        return false;
    }

    for (llvm::APInt v : replicaGroups.getValues<llvm::APInt>()) {
        if (v.getSExtValue() != 0) {
            errorOut = "replica_groups contains non-zero replica id (multi-device collectives are "
                       "not supported)";
            return false;
        }
    }
    return true;
}

ProcessResult ValidateSingleReplicaCollective(HandlerContext& ctx, mlir::Attribute replicaGroups,
                                              const char* opNameForError) {
    int64_t numReplicas = GetNumReplicasFromModule(ctx.module);
    if (numReplicas != 1) {
        return ProcessResult::Error(std::string(opNameForError) +
                                    ": multi-replica execution is not supported (mhlo.num_replicas=" +
                                    std::to_string(numReplicas) + ")");
    }

    std::string replicaGroupsError;
    if (!IsSingleReplicaGroups(replicaGroups, replicaGroupsError)) {
        return ProcessResult::Error(std::string(opNameForError) + ": " + replicaGroupsError);
    }
    return ProcessResult{};
}

} // namespace

static ProcessResult HandleAllReduce(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::AllReduceOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("all_reduce: expected AllReduceOp");
    }

    ProcessResult valid = ValidateSingleReplicaCollective(ctx, op.getReplicaGroups(), "all_reduce");
    if (!valid.ok()) {
        return valid;
    }

    unsigned numOperands = ctx.op->getNumOperands();
    unsigned numResults = ctx.op->getNumResults();
    if (numOperands != numResults) {
        return ProcessResult::Error(
            "all_reduce: operand/result arity mismatch (operands=" + std::to_string(numOperands) +
            ", results=" + std::to_string(numResults) + ")");
    }

    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* t = GetInputTensor(ctx, i);
        if (!t) {
            return ProcessResult::Error("all_reduce: missing operand tensor " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, t, i);
    }
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.all_reduce", HandleAllReduce);

static ProcessResult HandleAllGather(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::AllGatherOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("all_gather: expected AllGatherOp");
    }

    ProcessResult valid = ValidateSingleReplicaCollective(ctx, op.getReplicaGroups(), "all_gather");
    if (!valid.ok()) {
        return valid;
    }

    unsigned numOperands = ctx.op->getNumOperands();
    unsigned numResults = ctx.op->getNumResults();
    if (numOperands != numResults) {
        return ProcessResult::Error(
            "all_gather: operand/result arity mismatch (operands=" + std::to_string(numOperands) +
            ", results=" + std::to_string(numResults) + ")");
    }

    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* t = GetInputTensor(ctx, i);
        if (!t) {
            return ProcessResult::Error("all_gather: missing operand tensor " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, t, i);
    }
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.all_gather", HandleAllGather);

static ProcessResult HandleReduceScatter(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::ReduceScatterOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("reduce_scatter: expected ReduceScatterOp");
    }

    ProcessResult valid = ValidateSingleReplicaCollective(ctx, op.getReplicaGroups(), "reduce_scatter");
    if (!valid.ok()) {
        return valid;
    }

    unsigned numOperands = ctx.op->getNumOperands();
    unsigned numResults = ctx.op->getNumResults();
    if (numOperands != numResults) {
        return ProcessResult::Error(
            "reduce_scatter: operand/result arity mismatch (operands=" + std::to_string(numOperands) +
            ", results=" + std::to_string(numResults) + ")");
    }

    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* t = GetInputTensor(ctx, i);
        if (!t) {
            return ProcessResult::Error("reduce_scatter: missing operand tensor " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, t, i);
    }
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.reduce_scatter", HandleReduceScatter);

static ProcessResult HandleCollectivePermute(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::CollectivePermuteOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("collective_permute: expected CollectivePermuteOp");
    }

    int64_t numReplicas = GetNumReplicasFromModule(ctx.module);
    if (numReplicas != 1) {
        return ProcessResult::Error("collective_permute: multi-replica execution is not supported "
                                    "(mhlo.num_replicas=" +
                                    std::to_string(numReplicas) + ")");
    }

    auto pairs = op.getSourceTargetPairs();
    for (llvm::APInt v : pairs.getValues<llvm::APInt>()) {
        if (v.getSExtValue() != 0) {
            return ProcessResult::Error("collective_permute: only the identity mapping (0->0) is "
                                        "supported on single-device");
        }
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("collective_permute: missing input tensor");
    }
    SetOutputTensor(ctx.values, ctx.op, input);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.collective_permute", HandleCollectivePermute);

// lax.axis_index under shard_map and pmap lowers to stablehlo.partition_id,
// which had no handler. With a single partition the id is always 0; modules
// declaring more than one partition are rejected rather than silently given 0.
static ProcessResult HandlePartitionId(HandlerContext& ctx) {
    if (ctx.module) {
        if (auto attr = ctx.module->getAttrOfType<mlir::IntegerAttr>("mhlo.num_partitions")) {
            if (attr.getInt() != 1) {
                return ProcessResult::Error("partition_id: multi-partition execution is not supported "
                                            "(mhlo.num_partitions=" +
                                            std::to_string(attr.getInt()) + ")");
            }
        }
    }
    MPSDataType dtype = GetResultMpsType(ctx.op);
    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op);
    if (dtype == MPSDataTypeInvalid || !shape) {
        return ProcessResult::Error("partition_id: could not determine result type");
    }
    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 shape:shape dataType:dtype];
    return Result(ctx, zero, "partition_id");
}
REGISTER_MPS_OP("stablehlo.partition_id", HandlePartitionId);

} // namespace jax_metallib
