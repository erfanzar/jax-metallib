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

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "pjrt_plugin/core/logging.h"
#include "pjrt_plugin/core/type_utils.h"
#include "stablehlo/dialect/StablehloOps.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jax_metallib {

using ValueMap = std::unordered_map<void*, MPSGraphTensor*>;

struct ProcessResult {
    std::string error;
    std::vector<mlir::Value> return_values;

    bool ok() const {
        return error.empty();
    }
    static ProcessResult Error(const std::string& msg) {
        ProcessResult r;
        r.error = msg;
        return r;
    }
};

struct HandlerContext;

using BlockProcessor = ProcessResult (*)(HandlerContext& ctx, mlir::Block& block);

struct HandlerContext {
    MPSGraph* graph;
    mlir::Operation* op;
    ValueMap& values;

    ValueMap* variables = nullptr;

    mlir::ModuleOp module;
    int depth;
    BlockProcessor processBlock;

    HandlerContext(MPSGraph* g, mlir::Operation* o, ValueMap& v)
        : graph(g), op(o), values(v), variables(nullptr), module(nullptr), depth(0), processBlock(nullptr) {}

    HandlerContext(MPSGraph* g, mlir::Operation* o, ValueMap& v, mlir::ModuleOp m, int d, BlockProcessor bp)
        : graph(g), op(o), values(v), variables(nullptr), module(m), depth(d), processBlock(bp) {}
};

using GraphOpHandler = ProcessResult (*)(HandlerContext& ctx);

using NativeOpHandler = id<MTLBuffer> (*)(id<MTLDevice>, id<MTLCommandBuffer>, mlir::Operation*,
                                          const std::vector<id<MTLBuffer>>&);

struct OpHandler {
    enum class Kind { GRAPH, NATIVE } kind;

    GraphOpHandler graph_handler = nullptr;
    NativeOpHandler native_handler = nullptr;

    static OpHandler Graph(GraphOpHandler h) {
        OpHandler handler;
        handler.kind = Kind::GRAPH;
        handler.graph_handler = h;
        return handler;
    }

    static OpHandler Native(NativeOpHandler h) {
        OpHandler handler;
        handler.kind = Kind::NATIVE;
        handler.native_handler = h;
        return handler;
    }

    bool is_native() const {
        return kind == Kind::NATIVE;
    }
    bool is_graph() const {
        return kind == Kind::GRAPH;
    }
};

class OpRegistry {
public:
    static bool Register(const char* name, OpHandler handler) {
        GetMutableHandlers()[name] = handler;
        return true;
    }

    static const OpHandler* Find(const std::string& name) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(name);
        return it != handlers.end() ? &it->second : nullptr;
    }

    static std::string ListRegistered() {
        auto& handlers = GetMutableHandlers();
        std::string result;
        bool first = true;
        for (const auto& pair : handlers) {
            if (!first)
                result += ", ";
            result += pair.first;
            first = false;
        }
        return result;
    }

    static std::unordered_set<std::string> GetRegisteredOps() {
        std::unordered_set<std::string> ops;
        for (const auto& pair : GetMutableHandlers()) {
            ops.insert(pair.first);
        }
        return ops;
    }

private:
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }
};

class CustomCallRegistry {
public:
    static bool Register(const char* target, OpHandler handler) {
        GetMutableHandlers()[target] = handler;
        return true;
    }

    static const OpHandler* Find(const std::string& target) {
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(target);
        return it != handlers.end() ? &it->second : nullptr;
    }

private:
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }
};

inline MPSGraphTensor* GetTensor(ValueMap& values, mlir::Value value) {
    auto it = values.find(value.getAsOpaquePointer());
    return it != values.end() ? it->second : nullptr;
}

inline MPSGraphTensor* GetInputTensor(ValueMap& values, mlir::Operation* op, unsigned index) {
    if (index >= op->getNumOperands()) {
        return nullptr;
    }
    return GetTensor(values, op->getOperand(index));
}

inline void SetOutputTensor(ValueMap& values, mlir::Operation* op, MPSGraphTensor* tensor,
                            unsigned index = 0) {
    if (index < op->getNumResults()) {
        values[op->getResult(index).getAsOpaquePointer()] = tensor;
    }
}

inline MPSDataType GetResultMpsType(mlir::Operation* op, unsigned index = 0) {
    if (index >= op->getNumResults()) {
        return MPSDataTypeInvalid;
    }
    auto resultType = op->getResult(index).getType();
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
        return MlirTypeToMps(tensorType.getElementType());
    }
    return MPSDataTypeInvalid;
}

inline mlir::Type GetElementType(mlir::Type type) {
    if (auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
        return tensorType.getElementType();
    }
    return type;
}

inline NSArray<NSNumber*>* GetOutputShape(mlir::Operation* op, unsigned resultIndex = 0) {
    if (resultIndex >= op->getNumResults()) {
        return nil;
    }
    auto resultType = op->getResult(resultIndex).getType();
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    if (!tensorType) {
        return nil;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (int64_t dim : tensorType.getShape()) {
        [shape addObject:@(dim)];
    }
    return shape;
}

inline MPSGraphTensor* EnsureInt32(MPSGraph* g, MPSGraphTensor* tensor) {
    if (tensor.dataType != MPSDataTypeInt32) {
        return [g castTensor:tensor toType:MPSDataTypeInt32 name:nil];
    }
    return tensor;
}

inline thread_local int g_control_flow_depth = 0;

struct ScopedControlFlowContext {
    ScopedControlFlowContext() {
        ++g_control_flow_depth;
    }
    ~ScopedControlFlowContext() {
        --g_control_flow_depth;
    }
};

inline bool InControlFlowContext() {
    return g_control_flow_depth > 0;
}

inline ProcessResult Result(ValueMap& values, mlir::Operation* op, MPSGraphTensor* result,
                            const char* op_name) {
    if (!result)
        return ProcessResult::Error(std::string(op_name) + ": handler returned null");
    SetOutputTensor(values, op, result);
    return ProcessResult{};
}

inline MPSGraphTensor* GetInputTensor(HandlerContext& ctx, unsigned index) {
    if (!ctx.op) {
        return nil;
    }
    if (ctx.variables) {
        mlir::Value operand = ctx.op->getOperand(index);
        auto it = ctx.variables->find(operand.getAsOpaquePointer());
        if (it != ctx.variables->end()) {
            return [ctx.graph readVariable:it->second name:nil];
        }
    }
    return GetInputTensor(ctx.values, ctx.op, index);
}

inline ProcessResult Result(HandlerContext& ctx, MPSGraphTensor* result, const char* op_name) {
    return Result(ctx.values, ctx.op, result, op_name);
}

#define REGISTER_MPS_OP(mlir_op_name, handler_fn)                                                            \
    static bool _reg_##handler_fn =                                                                          \
        ::jax_metallib::OpRegistry::Register(mlir_op_name, ::jax_metallib::OpHandler::Graph(handler_fn))

#define REGISTER_NATIVE_MPS_OP(mlir_op_name, handler_fn)                                                     \
    static bool _reg_##handler_fn =                                                                          \
        ::jax_metallib::OpRegistry::Register(mlir_op_name, ::jax_metallib::OpHandler::Native(handler_fn))

#define REGISTER_MLIR_BINARY_OP(mlir_op_name, mps_method, reg_suffix)                                        \
    static ::jax_metallib::ProcessResult HandleMlir##reg_suffix(::jax_metallib::HandlerContext& ctx) {       \
        MPSGraphTensor* lhs = GetInputTensor(ctx, 0);                                                        \
        MPSGraphTensor* rhs = GetInputTensor(ctx, 1);                                                        \
        if (!lhs || !rhs)                                                                                    \
            return ::jax_metallib::ProcessResult::Error(#reg_suffix ": missing input tensor");               \
        MPSGraphTensor* out = [ctx.graph mps_method##WithPrimaryTensor:lhs secondaryTensor:rhs name:nil];    \
        return Result(ctx, out, #reg_suffix);                                                                \
    }                                                                                                        \
    REGISTER_MPS_OP(mlir_op_name, HandleMlir##reg_suffix)

#define REGISTER_MLIR_UNARY_OP(mlir_op_name, mps_method, reg_suffix)                                         \
    static ::jax_metallib::ProcessResult HandleMlir##reg_suffix(::jax_metallib::HandlerContext& ctx) {       \
        MPSGraphTensor* input = GetInputTensor(ctx, 0);                                                      \
        if (!input)                                                                                          \
            return ::jax_metallib::ProcessResult::Error(#reg_suffix ": missing input tensor");               \
        MPSGraphTensor* out = [ctx.graph mps_method##WithTensor:input name:nil];                             \
        return Result(ctx, out, #reg_suffix);                                                                \
    }                                                                                                        \
    REGISTER_MPS_OP(mlir_op_name, HandleMlir##reg_suffix)

#define REGISTER_CUSTOM_CALL(target_name, handler_fn, unique_suffix)                                         \
    static bool _cc_reg_##unique_suffix = ::jax_metallib::CustomCallRegistry::Register(                      \
        target_name, ::jax_metallib::OpHandler::Graph(handler_fn))

#define REGISTER_CUSTOM_CALL_UNARY_OP(target_name, mps_method, reg_suffix)                                   \
    static ::jax_metallib::ProcessResult HandleCc##reg_suffix(::jax_metallib::HandlerContext& ctx) {         \
        MPSGraphTensor* input = GetInputTensor(ctx, 0);                                                      \
        if (!input)                                                                                          \
            return ::jax_metallib::ProcessResult::Error(#reg_suffix ": missing input tensor");               \
        MPSGraphTensor* out = [ctx.graph mps_method##WithTensor:input name:nil];                             \
        return Result(ctx, out, #reg_suffix);                                                                \
    }                                                                                                        \
    REGISTER_CUSTOM_CALL(target_name, HandleCc##reg_suffix, reg_suffix)

} // namespace jax_metallib
