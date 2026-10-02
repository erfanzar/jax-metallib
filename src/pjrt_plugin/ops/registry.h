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
#include <cstdlib>
#include <mutex>
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

inline bool InControlFlowContext();

using BlockProcessor = ProcessResult (*)(HandlerContext& ctx, mlir::Block& block);

struct HandlerContext {
    MPSGraph* graph;
    mlir::Operation* op;
    ValueMap& values;

    ValueMap* variables = nullptr;
    bool in_control_flow = false;

    mlir::ModuleOp module;
    int depth;
    BlockProcessor processBlock;

    HandlerContext(MPSGraph* g, mlir::Operation* o, ValueMap& v)
        : graph(g), op(o), values(v), variables(nullptr), in_control_flow(false), module(nullptr), depth(0),
          processBlock(nullptr) {}

    HandlerContext(MPSGraph* g, mlir::Operation* o, ValueMap& v, mlir::ModuleOp m, int d, BlockProcessor bp)
        : graph(g), op(o), values(v), variables(nullptr), in_control_flow(false), module(m), depth(d),
          processBlock(bp) {}

    bool isInControlFlow() const {
        return in_control_flow || InControlFlowContext();
    }
};

using GraphOpHandler = ProcessResult (*)(HandlerContext& ctx);

using NativeOpHandler = id<MTLBuffer> (*)(id<MTLDevice>, id<MTLCommandBuffer>, mlir::Operation*,
                                          const std::vector<id<MTLBuffer>>&);

struct OpHandler {
    enum class Kind { GRAPH, NATIVE } kind;

    GraphOpHandler graph_handler = nullptr;
    NativeOpHandler native_handler = nullptr;
    GraphOpHandler graph_fallback = nullptr;

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

    // Native op that can also be built as MPSGraph ops. The graph form is used
    // only when the op lands inside a graph region (a case/while body), where
    // a native kernel cannot run; top-level occurrences stay native.
    static OpHandler NativeWithGraphFallback(NativeOpHandler h, GraphOpHandler fallback) {
        OpHandler handler = Native(h);
        handler.graph_fallback = fallback;
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
private:
    static std::mutex& GetMutex() {
        static std::mutex mutex;
        return mutex;
    }
    static std::unordered_map<std::string, OpHandler>& GetMutableHandlers() {
        static std::unordered_map<std::string, OpHandler> handlers;
        return handlers;
    }

public:
    static bool Register(const char* name, OpHandler handler) {
        std::lock_guard<std::mutex> lock(GetMutex());
        GetMutableHandlers()[name] = handler;
        return true;
    }

    static const OpHandler* Find(const std::string& name) {
        std::lock_guard<std::mutex> lock(GetMutex());
        auto& handlers = GetMutableHandlers();
        auto it = handlers.find(name);
        return it != handlers.end() ? &it->second : nullptr;
    }

    static std::string ListRegistered() {
        std::lock_guard<std::mutex> lock(GetMutex());
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
        std::lock_guard<std::mutex> lock(GetMutex());
        std::unordered_set<std::string> ops;
        for (const auto& pair : GetMutableHandlers()) {
            ops.insert(pair.first);
        }
        return ops;
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

// MPSGraph evaluates float16/bfloat16 elementwise chains without rounding each
// intermediate to the half type: round(x * 1.5) with bf16 x = 4.34375 gave 7
// (XLA: 6, since the product 6.515625 rounds to 6.5 in bf16), exp(x * 1.5) was
// ~2% off, and f16 (x * 1.5 == 3.00390625) was false for x = 2.001953125.
// A reinterpret round trip through int16 forces the bf16 value to
// materialise (same barrier as HandleConvert), but it also splits MPSGraph's
// elementwise fusion (bf16 tanh/exp chain on 16M elements 1.1 -> 6.4 ms when
// applied to every op). Smooth chains stay within a few bf16 ulp without it,
// so by default only values feeding a discontinuous op (round, floor, ceil,
// sign, compare, rem; traced through data movement) are rounded.
// JAX_METALLIB_BF16_STRICT_ROUNDING=1 rounds every bf16 op result, which is
// what XLA:CPU does for bf16. It is bf16-only because XLA:CPU keeps f16
// arithmetic chains in f32 (f16 x * 1.5 - 3.00390625 is -0.0009765625 there,
// 0 if the product were rounded), so per-op f16 rounding moves away from it.
// JAX_METALLIB_DISABLE_HALF_ROUNDING_BARRIER=1 disables both.
inline bool Bf16BarrierEnvFlag(const char* name) {
    const char* v = std::getenv(name);
    return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
}

inline bool Bf16RoundingBarrierDisabled() {
    static const bool disabled = Bf16BarrierEnvFlag("JAX_METALLIB_DISABLE_HALF_ROUNDING_BARRIER");
    return disabled;
}

inline bool Bf16StrictRounding() {
    static const bool strict = Bf16BarrierEnvFlag("JAX_METALLIB_BF16_STRICT_ROUNDING");
    return strict;
}

inline bool IsDataMovementOp(mlir::Operation* op) {
    namespace shlo = mlir::stablehlo;
    return mlir::isa<shlo::ReshapeOp, shlo::BroadcastInDimOp, shlo::TransposeOp, shlo::SliceOp,
                     shlo::ConstantOp, shlo::ConcatenateOp, shlo::ConvertOp, shlo::ReverseOp, shlo::PadOp,
                     shlo::DynamicSliceOp, shlo::BitcastConvertOp>(op);
}

inline bool IsRoundingSensitiveOp(mlir::Operation* op) {
    namespace shlo = mlir::stablehlo;
    return mlir::isa<shlo::RoundOp, shlo::RoundNearestEvenOp, shlo::FloorOp, shlo::CeilOp, shlo::SignOp,
                     shlo::CompareOp, shlo::RemOp>(op);
}

inline bool FeedsRoundingSensitiveOp(mlir::Operation* op, int depth = 0) {
    if (depth > 4) {
        return false;
    }
    for (mlir::Value r : op->getResults()) {
        for (mlir::Operation* user : r.getUsers()) {
            if (IsRoundingSensitiveOp(user)) {
                return true;
            }
            if (IsDataMovementOp(user) && !mlir::isa<mlir::stablehlo::ConvertOp>(user) &&
                FeedsRoundingSensitiveOp(user, depth + 1)) {
                return true;
            }
        }
    }
    return false;
}

inline MPSGraphTensor* Bf16RoundingBarrier(MPSGraph* g, MPSGraphTensor* t) {
    NSArray<NSNumber*>* shape = t.shape;
    if (shape == nil) {
        return t;
    }
    for (NSNumber* d : shape) {
        if ([d longLongValue] <= 0) {
            return t;
        }
    }
    const bool scalar = shape.count == 0;
    MPSGraphTensor* x = scalar ? [g reshapeTensor:t withShape:@[@1] name:nil] : t;
    MPSGraphTensor* bits = [g reinterpretCastTensor:x toType:MPSDataTypeInt16 name:nil];
    x = [g reinterpretCastTensor:bits toType:t.dataType name:nil];
    return scalar ? [g reshapeTensor:x withShape:@[] name:nil] : x;
}

// MPSGraph negate/abs/select on float16/bfloat16 rewrote every NaN to +qNaN
// (f16 0x7e00, bf16 0x7fc0) and flushed bf16 subnormals to zero (select too).
// XLA:CPU does the f16 sign ops as bit arithmetic (negate 0x7e01 -> 0xfe01,
// abs 0xfe03 -> 0x7e03) and bf16 ops through f32: subnormals and the NaN sign
// survive and every NaN becomes sign|0x7fc0. These helpers work on the int16
// view; HalfToBits returns nil (caller falls back) for empty or dynamic
// shapes. Opt-in (JAX_METALLIB_HALF_BIT_SIGN_OPS=1): the int16 reinterpret
// splits MPSGraph elementwise fusion, so 16M-element chains slowed 3x
// (f16 where(m, -(x*y), abs(x)) + y 1.14 -> 3.52 ms, bf16 1.25 -> 4.22 ms),
// the same trade-off as JAX_METALLIB_BF16_STRICT_ROUNDING.
inline bool HalfBitSignOpsEnabled() {
    static const bool enabled = Bf16BarrierEnvFlag("JAX_METALLIB_HALF_BIT_SIGN_OPS");
    return enabled;
}

inline bool IsF16OrBf16(MPSDataType dt) {
    return dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
}

inline MPSGraphTensor* HalfToBits(MPSGraph* g, MPSGraphTensor* t) {
    NSArray<NSNumber*>* shape = t.shape;
    if (shape == nil) {
        return nil;
    }
    for (NSNumber* d : shape) {
        if ([d longLongValue] <= 0) {
            return nil;
        }
    }
    if (shape.count == 0) {
        t = [g reshapeTensor:t withShape:@[@1] name:nil];
    }
    return [g reinterpretCastTensor:t toType:MPSDataTypeInt16 name:nil];
}

inline MPSGraphTensor* HalfFromBits(MPSGraph* g, MPSGraphTensor* bits, MPSDataType dtype,
                                    NSArray<NSNumber*>* shape) {
    if (dtype == MPSDataTypeBFloat16) {
        MPSGraphTensor* magMask = [g constantWithScalar:32767.0 dataType:MPSDataTypeInt16];
        MPSGraphTensor* infBits = [g constantWithScalar:32640.0 dataType:MPSDataTypeInt16];
        MPSGraphTensor* signMask = [g constantWithScalar:-32768.0 dataType:MPSDataTypeInt16];
        MPSGraphTensor* qnan = [g constantWithScalar:32704.0 dataType:MPSDataTypeInt16];
        MPSGraphTensor* mag = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:magMask name:nil];
        MPSGraphTensor* isNan = [g greaterThanWithPrimaryTensor:mag secondaryTensor:infBits name:nil];
        MPSGraphTensor* sign = [g bitwiseANDWithPrimaryTensor:bits secondaryTensor:signMask name:nil];
        MPSGraphTensor* canon = [g bitwiseORWithPrimaryTensor:sign secondaryTensor:qnan name:nil];
        bits = [g selectWithPredicateTensor:isNan
                        truePredicateTensor:canon
                       falsePredicateTensor:bits
                                       name:nil];
    }
    MPSGraphTensor* out = [g reinterpretCastTensor:bits toType:dtype name:nil];
    if (shape && ![shape isEqualToArray:out.shape]) {
        out = [g reshapeTensor:out withShape:shape name:nil];
    }
    return out;
}

inline ProcessResult Result(ValueMap& values, mlir::Operation* op, MPSGraphTensor* result,
                            const char* op_name) {
    if (!result)
        return ProcessResult::Error(std::string(op_name) + ": handler returned null");
    const bool bf16Result = result.dataType == MPSDataTypeBFloat16;
    const bool halfResult = bf16Result || result.dataType == MPSDataTypeFloat16;
    if (halfResult && op && !IsDataMovementOp(op) && !Bf16RoundingBarrierDisabled() &&
        ((bf16Result && Bf16StrictRounding()) || FeedsRoundingSensitiveOp(op))) {
        result = Bf16RoundingBarrier(result.operation.graph, result);
    }
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

#define REGISTER_NATIVE_MPS_OP_WITH_GRAPH_FALLBACK(mlir_op_name, handler_fn, fallback_fn)                    \
    static bool _reg_##handler_fn = ::jax_metallib::OpRegistry::Register(                                    \
        mlir_op_name, ::jax_metallib::OpHandler::NativeWithGraphFallback(handler_fn, fallback_fn))

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
