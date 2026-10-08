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

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace jax_metallib {

namespace {

bool ReadConstantIntVector(mlir::Value value, std::vector<int64_t>& out) {
    out.clear();
    if (!value) {
        return false;
    }

    auto* defOp = value.getDefiningOp();
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
    if (!constOp) {
        return false;
    }

    auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!denseAttr) {
        return false;
    }

    auto elemType = denseAttr.getElementType();
    auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType);
    if (!intType) {
        return false;
    }

    out.reserve(denseAttr.getNumElements());
    for (llvm::APInt v : denseAttr.getValues<llvm::APInt>()) {

        out.push_back(intType.isUnsigned() ? (int64_t)v.getZExtValue() : v.getSExtValue());
    }
    return true;
}

bool ReadConstantIntScalar(mlir::Value value, int64_t& out) {
    std::vector<int64_t> values;
    if (!ReadConstantIntVector(value, values) || values.size() != 1) {
        return false;
    }
    out = values[0];
    return true;
}

} // namespace

static ProcessResult HandleBroadcast(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("broadcast: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && input.shape && [outputShape isEqualToArray:input.shape]) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    MPSGraphTensor* result = [ctx.graph broadcastTensor:input toShape:outputShape name:nil];
    return Result(ctx, result, "broadcast");
}
REGISTER_MPS_OP("stablehlo.broadcast", HandleBroadcast);

// broadcast_in_dim maps input axis i to output axis dims[i], and dims need not
// be increasing ((1,0) transposes, (2,0) sends [3,4] to [4,_,3]). Transpose the
// input into sorted-dims order first, then reshape to the output rank and
// broadcast. Before this, equal ranks went straight to broadcastTensor and the
// reshape path ignored the order, so permuting broadcasts returned untransposed data.
static MPSGraphTensor* BroadcastInDimTensor(MPSGraph* graph, MPSGraphTensor* input,
                                            llvm::ArrayRef<int64_t> dims, NSArray<NSNumber*>* outputShape) {
    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger inputRank = inputShape.count;
    NSUInteger outputRank = outputShape.count;
    if (dims.empty()) {
        return [graph broadcastTensor:input toShape:outputShape name:nil];
    }
    // order[k] is the input axis that lands on the k-th smallest output axis.
    std::vector<size_t> order(dims.size());
    for (size_t k = 0; k < order.size(); ++k) {
        order[k] = k;
    }
    bool permuted = false;
    if (dims.size() == inputRank) {
        std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) { return dims[a] < dims[b]; });
        for (size_t k = 0; k < order.size(); ++k) {
            permuted = permuted || order[k] != k;
        }
    }
    MPSGraphTensor* src = input;
    if (permuted) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:order.size()];
        for (size_t k : order) {
            [perm addObject:@(k)];
        }
        src = [graph transposeTensor:input permutation:perm name:nil];
    } else if (inputRank == outputRank) {
        return [graph broadcastTensor:input toShape:outputShape name:nil];
    }
    NSMutableArray<NSNumber*>* intermediateShape = [NSMutableArray arrayWithCapacity:outputRank];
    for (NSUInteger i = 0; i < outputRank; i++) {
        [intermediateShape addObject:@1];
    }
    for (size_t k = 0; k < order.size() && order[k] < inputRank; ++k) {
        int64_t outDim = dims[order[k]];
        if (outDim >= 0 && static_cast<NSUInteger>(outDim) < outputRank) {
            intermediateShape[static_cast<NSUInteger>(outDim)] = inputShape[order[k]];
        }
    }
    MPSGraphTensor* reshaped = [graph reshapeTensor:src withShape:intermediateShape name:nil];
    return [graph broadcastTensor:reshaped toShape:outputShape name:nil];
}

static ProcessResult HandleBroadcastInDim(HandlerContext& ctx) {
    auto broadcastOp = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(ctx.op);
    if (!broadcastOp) {
        return ProcessResult::Error("broadcast_in_dim: expected BroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("broadcast_in_dim: input tensor not found");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    MPSGraphTensor* result =
        BroadcastInDimTensor(ctx.graph, input, broadcastOp.getBroadcastDimensions(), outputShape);
    return Result(ctx, result, "broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.broadcast_in_dim", HandleBroadcastInDim);

static ProcessResult HandleDynamicBroadcastInDim(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::DynamicBroadcastInDimOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("dynamic_broadcast_in_dim: expected DynamicBroadcastInDimOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("dynamic_broadcast_in_dim: missing input tensor");
    }

    std::vector<int64_t> outDims;
    if (!ReadConstantIntVector(op.getOutputDimensions(), outDims)) {
        return ProcessResult::Error(
            "dynamic_broadcast_in_dim: dynamic output_dimensions is not supported (must be a "
            "stablehlo.constant)");
    }

    NSMutableArray<NSNumber*>* outputShape =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(outDims.size())];
    for (int64_t d : outDims) {
        if (d < 0) {
            return ProcessResult::Error(
                "dynamic_broadcast_in_dim: dynamic (unknown) output dimension is not supported");
        }
        [outputShape addObject:@(d)];
    }

    MPSGraphTensor* result = BroadcastInDimTensor(ctx.graph, input, op.getBroadcastDimensions(), outputShape);
    return Result(ctx, result, "dynamic_broadcast_in_dim");
}
REGISTER_MPS_OP("stablehlo.dynamic_broadcast_in_dim", HandleDynamicBroadcastInDim);

static ProcessResult HandleReshape(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reshape: missing input tensor");
    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);
    if (outputShape && input.shape && [outputShape isEqualToArray:input.shape]) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    // uint16 reshape under JIT corrupted values near 65535 (int16 and other
    // dtypes were exact); reshape the exact int32 widening and cast back.
    if (input.dataType == MPSDataTypeUInt16) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
        wide = [ctx.graph reshapeTensor:wide withShape:outputShape name:nil];
        return Result(ctx, [ctx.graph castTensor:wide toType:MPSDataTypeUInt16 name:nil], "reshape");
    }
    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "reshape");
}
REGISTER_MPS_OP("stablehlo.reshape", HandleReshape);

static ProcessResult HandleDynamicReshape(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::DynamicReshapeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("dynamic_reshape: expected DynamicReshapeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("dynamic_reshape: missing input tensor");
    }

    std::vector<int64_t> outDims;
    if (!ReadConstantIntVector(op.getOutputShape(), outDims)) {
        return ProcessResult::Error("dynamic_reshape: dynamic output_shape is not supported (must "
                                    "be a stablehlo.constant)");
    }

    NSMutableArray<NSNumber*>* outputShape =
        [NSMutableArray arrayWithCapacity:static_cast<NSUInteger>(outDims.size())];
    for (int64_t d : outDims) {
        if (d < 0) {
            return ProcessResult::Error(
                "dynamic_reshape: dynamic (unknown) output dimension is not supported");
        }
        [outputShape addObject:@(d)];
    }

    MPSGraphTensor* result = [ctx.graph reshapeTensor:input withShape:outputShape name:nil];
    return Result(ctx, result, "dynamic_reshape");
}
REGISTER_MPS_OP("stablehlo.dynamic_reshape", HandleDynamicReshape);

static ProcessResult HandleGetDimensionSize(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::GetDimensionSizeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("get_dimension_size: expected GetDimensionSizeOp");
    }

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(op.getOperand().getType());
    if (!inputType) {
        return ProcessResult::Error("get_dimension_size: expected ranked tensor operand");
    }

    int64_t dim = op.getDimension();
    auto shape = inputType.getShape();
    if (dim < 0 || dim >= (int64_t)shape.size()) {
        return ProcessResult::Error("get_dimension_size: dimension out of bounds");
    }
    if (shape[dim] < 0) {
        return ProcessResult::Error("get_dimension_size: dynamic dimension size is not supported");
    }

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("get_dimension_size: invalid result dtype");
    }

    MPSGraphTensor* result = [ctx.graph constantWithScalar:(double)shape[dim] dataType:dtype];
    return Result(ctx, result, "get_dimension_size");
}
REGISTER_MPS_OP("stablehlo.get_dimension_size", HandleGetDimensionSize);

static ProcessResult HandleSetDimensionSize(HandlerContext& ctx) {
    auto op = mlir::dyn_cast<mlir::stablehlo::SetDimensionSizeOp>(ctx.op);
    if (!op) {
        return ProcessResult::Error("set_dimension_size: expected SetDimensionSizeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input) {
        return ProcessResult::Error("set_dimension_size: missing input tensor");
    }

    int64_t dim = op.getDimension();
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(0).getType());
    if (!inputType) {
        return ProcessResult::Error("set_dimension_size: expected ranked tensor operand");
    }

    auto shape = inputType.getShape();
    if (dim < 0 || dim >= (int64_t)shape.size()) {
        return ProcessResult::Error("set_dimension_size: dimension out of bounds");
    }

    int64_t requested = 0;
    if (shape[dim] >= 0 && ReadConstantIntScalar(op.getSize(), requested) && requested != shape[dim]) {
        return ProcessResult::Error("set_dimension_size: requested size " + std::to_string(requested) +
                                    " does not match static dim size " + std::to_string(shape[dim]));
    }

    SetOutputTensor(ctx.values, ctx.op, input);
    return ProcessResult{};
}
REGISTER_MPS_OP("stablehlo.set_dimension_size", HandleSetDimensionSize);

static ProcessResult HandleTranspose(HandlerContext& ctx) {
    auto transposeOp = mlir::dyn_cast<mlir::stablehlo::TransposeOp>(ctx.op);
    if (!transposeOp) {
        return ProcessResult::Error("transpose: expected TransposeOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("transpose: missing input tensor");

    auto permutation = transposeOp.getPermutation();
    NSMutableArray<NSNumber*>* perm = [NSMutableArray array];
    bool isIdentity = true;
    int64_t i = 0;
    for (int64_t d : permutation) {
        [perm addObject:@(d)];
        if (d != i) {
            isIdentity = false;
        }
        ++i;
    }

    if (isIdentity) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    MPSGraphTensor* result = [ctx.graph transposeTensor:input permutation:perm name:nil];
    return Result(ctx, result, "transpose");
}
REGISTER_MPS_OP("stablehlo.transpose", HandleTranspose);

static bool IsHalfWidthFloat(MPSDataType t) {
    return t == MPSDataTypeFloat16 || t == MPSDataTypeBFloat16;
}

// A narrowing convert (to f16/bf16) of a value computed only from constants is
// folded on the host. MPSGraph folded reinterpretCast(cast(constant)) into a
// reinterpret of the constant's original bytes, so CastFoldBarrier returned the
// f32 bit pattern read as 16-bit values: jnp.clip(f16, -0.5, 0.5) clipped to
// [0, 0] (-0.5f is 0xBF000000) and arange(f32) * c cast to bf16 gave garbage
// (build/probe_const_cast_sources.py: 36/48 wrong). A plain cast is no fix:
// MPSGraph then folds const -> f16 -> f32 and 70000 stays 70000 instead of inf.
// The fold evaluates constant, iota, broadcast_in_dim, reshape, convert and
// float add/subtract/multiply/divide/negate exactly (double arithmetic rounded
// to the element type after every op) and emits a 16-bit constant; anything
// else keeps the barrier path. JAX_METALLIB_DISABLE_CONST_CONVERT_FOLD=1 opts out.
enum class ConstFoldElem { kF64, kF32, kF16, kBF16, kInt };

struct ConstFoldValue {
    std::vector<double> data; // a single element when splat
    bool splat = false;
    std::vector<int64_t> shape;
};

using ConstFoldMemo = std::unordered_map<const void*, std::optional<ConstFoldValue>>;

constexpr int64_t kConstFoldMaxElements = int64_t{1} << 22;

static std::optional<ConstFoldElem> ConstFoldElemKind(mlir::Type t) {
    if (t.isF64())
        return ConstFoldElem::kF64;
    if (t.isF32())
        return ConstFoldElem::kF32;
    if (t.isF16())
        return ConstFoldElem::kF16;
    if (t.isBF16())
        return ConstFoldElem::kBF16;
    if (auto it = mlir::dyn_cast<mlir::IntegerType>(t)) {
        const unsigned w = it.getWidth();
        if (w == 8 || w == 16 || w == 32 || w == 64)
            return ConstFoldElem::kInt;
    }
    return std::nullopt;
}

static bool ConstFoldFitsFloat(double v) {
    return !std::isfinite(v) || std::fabs(v) <= static_cast<double>(std::numeric_limits<float>::max());
}

// Round-to-nearest-even bits of v in f16 or bf16, rounding exactly once.
static uint16_t ConstFoldNarrowBits(double v, ConstFoldElem kind) {
    const bool bf16 = kind == ConstFoldElem::kBF16;
    if (ConstFoldFitsFloat(v)) {
        const float f = static_cast<float>(v);
        if (std::isnan(v) || static_cast<double>(f) == v) {
            if (bf16) {
                uint32_t u = 0;
                std::memcpy(&u, &f, sizeof(u));
                if (std::isnan(f))
                    return static_cast<uint16_t>((u >> 16) | 0x0040u);
                u += 0x7FFFu + ((u >> 16) & 1u);
                return static_cast<uint16_t>(u >> 16);
            }
            const _Float16 h = static_cast<_Float16>(f);
            uint16_t bits = 0;
            std::memcpy(&bits, &h, sizeof(bits));
            return bits;
        }
    }
    llvm::APFloat ap(v);
    bool loses = false;
    ap.convert(bf16 ? llvm::APFloat::BFloat() : llvm::APFloat::IEEEhalf(), llvm::APFloat::rmNearestTiesToEven,
               &loses);
    return static_cast<uint16_t>(ap.bitcastToAPInt().getZExtValue());
}

static double ConstFoldRound(double v, ConstFoldElem kind) {
    switch (kind) {
    case ConstFoldElem::kF64:
    case ConstFoldElem::kInt:
        return v;
    case ConstFoldElem::kF32: {
        if (ConstFoldFitsFloat(v))
            return static_cast<double>(static_cast<float>(v));
        llvm::APFloat ap(v);
        bool loses = false;
        ap.convert(llvm::APFloat::IEEEsingle(), llvm::APFloat::rmNearestTiesToEven, &loses);
        return static_cast<double>(ap.convertToFloat());
    }
    case ConstFoldElem::kF16: {
        const uint16_t bits = ConstFoldNarrowBits(v, kind);
        _Float16 h;
        std::memcpy(&h, &bits, sizeof(h));
        return static_cast<double>(h);
    }
    case ConstFoldElem::kBF16: {
        const uint32_t u = static_cast<uint32_t>(ConstFoldNarrowBits(v, kind)) << 16;
        float f = 0.0f;
        std::memcpy(&f, &u, sizeof(f));
        return static_cast<double>(f);
    }
    }
    return v;
}

static std::optional<ConstFoldValue> ConstFoldEval(mlir::Value v, ConstFoldMemo& memo, int depth);

static std::optional<ConstFoldValue> ConstFoldReadConstant(mlir::stablehlo::ConstantOp op, ConstFoldElem kind,
                                                           mlir::Type elemType, ConstFoldValue r,
                                                           int64_t numel) {
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(op.getValue());
    if (!dense)
        return std::nullopt;
    r.splat = dense.isSplat();
    const int64_t count = r.splat ? 1 : numel;
    if (count > kConstFoldMaxElements)
        return std::nullopt;
    r.data.resize(static_cast<size_t>(count));
    int64_t i = 0;
    if (kind == ConstFoldElem::kInt) {
        const bool isUnsigned = mlir::cast<mlir::IntegerType>(elemType).isUnsigned();
        constexpr int64_t kExact = int64_t{1} << 53;
        for (const llvm::APInt& ap : dense.getValues<llvm::APInt>()) {
            if (i >= count)
                break;
            if (isUnsigned) {
                const uint64_t u = ap.getZExtValue();
                if (u > static_cast<uint64_t>(kExact))
                    return std::nullopt;
                r.data[i++] = static_cast<double>(u);
            } else {
                const int64_t s = ap.getSExtValue();
                if (s > kExact || s < -kExact)
                    return std::nullopt;
                r.data[i++] = static_cast<double>(s);
            }
        }
    } else if (kind == ConstFoldElem::kF32 && !r.splat) {
        llvm::ArrayRef<char> raw = dense.getRawData();
        if (raw.size() != static_cast<size_t>(count) * sizeof(float))
            return std::nullopt;
        for (; i < count; ++i) {
            float f = 0.0f;
            std::memcpy(&f, raw.data() + i * sizeof(float), sizeof(f));
            r.data[i] = static_cast<double>(f);
        }
    } else {
        for (llvm::APFloat ap : dense.getValues<llvm::APFloat>()) {
            if (i >= count)
                break;
            bool loses = false;
            ap.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven, &loses);
            r.data[i++] = ap.convertToDouble();
        }
    }
    if (i != count)
        return std::nullopt;
    return r;
}

static std::optional<ConstFoldValue> ConstFoldEvalUncached(mlir::Value v, ConstFoldMemo& memo, int depth) {
    namespace shlo = mlir::stablehlo;
    mlir::Operation* def = v.getDefiningOp();
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!def || !type || !type.hasStaticShape() || def->getNumResults() != 1)
        return std::nullopt;
    const std::optional<ConstFoldElem> kind = ConstFoldElemKind(type.getElementType());
    if (!kind)
        return std::nullopt;
    ConstFoldValue r;
    r.shape.assign(type.getShape().begin(), type.getShape().end());
    int64_t numel = 1;
    for (int64_t d : r.shape)
        numel *= d;
    if (numel <= 0)
        return std::nullopt;
    auto operand = [&](unsigned i) { return ConstFoldEval(def->getOperand(i), memo, depth + 1); };

    if (auto c = mlir::dyn_cast<shlo::ConstantOp>(def))
        return ConstFoldReadConstant(c, *kind, type.getElementType(), std::move(r), numel);
    if (auto iota = mlir::dyn_cast<shlo::IotaOp>(def)) {
        const uint64_t dim = iota.getIotaDimension();
        if (numel > kConstFoldMaxElements || dim >= r.shape.size())
            return std::nullopt;
        int64_t inner = 1;
        for (size_t d = dim + 1; d < r.shape.size(); ++d)
            inner *= r.shape[d];
        r.data.resize(static_cast<size_t>(numel));
        for (int64_t i = 0; i < numel; ++i)
            r.data[i] = ConstFoldRound(static_cast<double>((i / inner) % r.shape[dim]), *kind);
        return r;
    }
    if (mlir::isa<shlo::ReshapeOp>(def)) {
        auto in = operand(0);
        if (in)
            in->shape = r.shape;
        return in;
    }
    if (auto bcast = mlir::dyn_cast<shlo::BroadcastInDimOp>(def)) {
        auto in = operand(0);
        if (!in)
            return std::nullopt;
        if (in->splat) {
            in->shape = r.shape;
            return in;
        }
        llvm::ArrayRef<int64_t> dims = bcast.getBroadcastDimensions();
        const int64_t outRank = static_cast<int64_t>(r.shape.size());
        if (numel > kConstFoldMaxElements || dims.size() != in->shape.size())
            return std::nullopt;
        for (int64_t d : dims) {
            if (d < 0 || d >= outRank)
                return std::nullopt;
        }
        std::vector<int64_t> inStride(in->shape.size(), 1);
        for (int64_t k = static_cast<int64_t>(in->shape.size()) - 2; k >= 0; --k)
            inStride[k] = inStride[k + 1] * in->shape[k + 1];
        std::vector<int64_t> idx(r.shape.size(), 0);
        r.data.resize(static_cast<size_t>(numel));
        for (int64_t lin = 0; lin < numel; ++lin) {
            int64_t src = 0;
            for (size_t k = 0; k < dims.size(); ++k)
                src += (in->shape[k] == 1 ? 0 : idx[dims[k]]) * inStride[k];
            r.data[lin] = in->data[src];
            for (int64_t d = outRank - 1; d >= 0; --d) {
                if (++idx[d] < r.shape[d])
                    break;
                idx[d] = 0;
            }
        }
        return r;
    }
    if (mlir::isa<shlo::ConvertOp>(def)) {
        if (*kind == ConstFoldElem::kInt)
            return std::nullopt;
        auto in = operand(0);
        if (!in)
            return std::nullopt;
        for (double& x : in->data)
            x = ConstFoldRound(x, *kind);
        in->shape = r.shape;
        return in;
    }
    if (*kind == ConstFoldElem::kInt)
        return std::nullopt;
    if (mlir::isa<shlo::NegOp>(def)) {
        auto in = operand(0);
        if (in) {
            for (double& x : in->data)
                x = -x;
        }
        return in;
    }
    if (mlir::isa<shlo::AddOp, shlo::SubtractOp, shlo::MulOp, shlo::DivOp>(def)) {
        auto a = operand(0);
        auto b = operand(1);
        if (!a || !b)
            return std::nullopt;
        r.splat = a->splat && b->splat;
        const size_t n = r.splat ? 1 : static_cast<size_t>(numel);
        if ((!a->splat && a->data.size() != n) || (!b->splat && b->data.size() != n))
            return std::nullopt;
        r.data.resize(n);
        for (size_t i = 0; i < n; ++i) {
            const double x = a->data[a->splat ? 0 : i];
            const double y = b->data[b->splat ? 0 : i];
            double z = 0.0;
            if (mlir::isa<shlo::AddOp>(def))
                z = x + y;
            else if (mlir::isa<shlo::SubtractOp>(def))
                z = x - y;
            else if (mlir::isa<shlo::MulOp>(def))
                z = x * y;
            else
                z = x / y;
            r.data[i] = ConstFoldRound(z, *kind);
        }
        return r;
    }
    return std::nullopt;
}

static std::optional<ConstFoldValue> ConstFoldEval(mlir::Value v, ConstFoldMemo& memo, int depth) {
    if (depth > 32)
        return std::nullopt;
    const void* key = v.getAsOpaquePointer();
    auto found = memo.find(key);
    if (found != memo.end())
        return found->second;
    std::optional<ConstFoldValue> out = ConstFoldEvalUncached(v, memo, depth);
    memo[key] = out;
    return out;
}

// Returns a 16-bit constant for a foldable narrowing convert, or nil.
static MPSGraphTensor* FoldNarrowingConvert(MPSGraph* g, mlir::Operation* op, MPSDataType dtype) {
    static const bool disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_CONST_CONVERT_FOLD");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    if (disabled || op->getNumResults() != 1 || op->getNumOperands() != 1 ||
        !op->getOperand(0).getDefiningOp())
        return nil;
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!type)
        return nil;
    const std::optional<ConstFoldElem> kind = ConstFoldElemKind(type.getElementType());
    if (!kind || (*kind != ConstFoldElem::kF16 && *kind != ConstFoldElem::kBF16))
        return nil;
    ConstFoldMemo memo;
    std::optional<ConstFoldValue> folded = ConstFoldEval(op->getResult(0), memo, 0);
    if (!folded || folded->data.empty())
        return nil;
    std::vector<uint16_t> bits(folded->data.size());
    for (size_t i = 0; i < bits.size(); ++i)
        bits[i] = ConstFoldNarrowBits(folded->data[i], *kind);
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:folded->shape.size()];
    for (int64_t d : folded->shape)
        [shape addObject:@(d)];
    NSData* data = [NSData dataWithBytes:bits.data() length:bits.size() * sizeof(uint16_t)];
    if (bits.size() == 1) {
        MPSGraphTensor* one = [g constantWithData:data shape:@[@1] dataType:dtype];
        MPSGraphTensor* scalar = [g reshapeTensor:one withShape:@[] name:nil];
        return shape.count == 0 ? scalar : [g broadcastTensor:scalar toShape:shape name:nil];
    }
    return [g constantWithData:data shape:shape dataType:dtype];
}

// Under jit MPSGraph folded narrowing cast chains: x.astype(f16).astype(f32)
// became the identity (70000 stayed 70000 instead of inf, pi kept its f32
// bits), and f16 arithmetic between the casts ran in float32. A reinterpret
// round trip through int16 on the 16-bit side is not folded (verified with a
// standalone MPSGraph program at Level0). Rank-0 is reshaped to [1] because
// reinterpret casts abort on rank 0. JAX_METALLIB_DISABLE_CAST_FOLD_BARRIER=1
// restores the plain cast.
static MPSGraphTensor* CastFoldBarrier(MPSGraph* g, MPSGraphTensor* t) {
    NSArray<NSNumber*>* shape = t.shape;
    if (shape == nil) {
        return t;
    }
    const bool scalar = shape.count == 0;
    MPSGraphTensor* x = scalar ? [g reshapeTensor:t withShape:@[@1] name:nil] : t;
    MPSGraphTensor* bits = [g reinterpretCastTensor:x toType:MPSDataTypeInt16 name:nil];
    x = [g reinterpretCastTensor:bits toType:t.dataType name:nil];
    return scalar ? [g reshapeTensor:x withShape:@[] name:nil] : x;
}

// True unless v depends on a block argument or a nondeterministic/external op,
// i.e. MPSGraph may constant-fold it. Over-approximates on purpose: a walk past
// kLimit values is reported as compile-time, which only costs the exact integer
// narrowing below (correct for any input).
static bool IsCompileTimeValue(mlir::Value root) {
    namespace shlo = mlir::stablehlo;
    constexpr size_t kLimit = 4096;
    std::vector<mlir::Value> stack{root};
    std::unordered_set<const void*> seen;
    while (!stack.empty()) {
        mlir::Value v = stack.back();
        stack.pop_back();
        if (!seen.insert(v.getAsOpaquePointer()).second)
            continue;
        if (seen.size() > kLimit)
            return true;
        mlir::Operation* def = v.getDefiningOp();
        if (!def)
            return false;
        if (mlir::isa<shlo::RngOp, shlo::RngBitGeneratorOp, shlo::CustomCallOp, shlo::InfeedOp, shlo::RecvOp>(
                def))
            return false;
        // A reduce over a statically empty axis is its init values broadcast
        // (HandleSingleResultReduce emits exactly that), so MPSGraph folds it
        // like a constant: f16 prod over [4,0] narrowed 1.0f to [0, 1.875].
        if (auto red = mlir::dyn_cast<shlo::ReduceOp>(def)) {
            auto in = mlir::dyn_cast<mlir::RankedTensorType>(def->getOperand(0).getType());
            bool emptyAxis = false;
            for (int64_t d : red.getDimensions()) {
                if (in && d >= 0 && d < in.getRank() && in.getDimSize(d) == 0)
                    emptyAxis = true;
            }
            if (emptyAxis) {
                const unsigned n = def->getNumOperands();
                for (unsigned i = n / 2; i < n; ++i)
                    stack.push_back(def->getOperand(i));
                continue;
            }
        }
        for (mlir::Value operand : def->getOperands())
            stack.push_back(operand);
    }
    return true;
}

// f32 -> f16/bf16 with round-to-nearest-even done on the f32 bit pattern in
// int32 arithmetic, then reinterpreted. For constant-derived operands MPSGraph
// folded castTensor(const) to a 16-bit tensor that still held the f32 bytes, so
// any later reinterpret (CastFoldBarrier) read them back: every barrier variant
// tried failed identically, while reinterpreting a folded f32 value is exact.
// NaN follows XLA/Eigen: sign | 0x7E00 (f16), sign | 0x7FC0 (bf16).
static MPSGraphTensor* NarrowF32ByBits(MPSGraph* g, MPSGraphTensor* x, MPSDataType dtype) {
    const bool scalar = x.shape.count == 0;
    if (scalar)
        x = [g reshapeTensor:x withShape:@[@1] name:nil];
    const MPSDataType i32 = MPSDataTypeInt32;
    auto k = [&](int64_t v) { return [g constantWithScalar:static_cast<double>(v) dataType:i32]; };
    auto add = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    auto sub = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    auto band = [&](MPSGraphTensor* a, int64_t m) {
        return [g bitwiseANDWithPrimaryTensor:a secondaryTensor:k(m) name:nil];
    };
    auto bor = [&](MPSGraphTensor* a, MPSGraphTensor* b) {
        return [g bitwiseORWithPrimaryTensor:a secondaryTensor:b name:nil];
    };
    auto shr = [&](MPSGraphTensor* a, MPSGraphTensor* s) {
        return [g bitwiseRightShiftWithPrimaryTensor:a secondaryTensor:s name:nil];
    };
    auto shl = [&](MPSGraphTensor* a, MPSGraphTensor* s) {
        return [g bitwiseLeftShiftWithPrimaryTensor:a secondaryTensor:s name:nil];
    };
    auto ge = [&](MPSGraphTensor* a, int64_t v) {
        return [g greaterThanOrEqualToWithPrimaryTensor:a secondaryTensor:k(v) name:nil];
    };
    auto sel = [&](MPSGraphTensor* p, MPSGraphTensor* t, MPSGraphTensor* f) {
        return [g selectWithPredicateTensor:p truePredicateTensor:t falsePredicateTensor:f name:nil];
    };
    MPSGraphTensor* u = [g reinterpretCastTensor:x toType:i32 name:nil];
    MPSGraphTensor* hi = shr(u, k(16));
    MPSGraphTensor* sign = band(hi, 0x8000);
    MPSGraphTensor* a = band(u, 0x7FFFFFFF);
    MPSGraphTensor* isNan = ge(a, 0x7F800001);
    MPSGraphTensor* h = nil;
    if (dtype == MPSDataTypeBFloat16) {
        MPSGraphTensor* rounded = band(shr(add(add(u, k(0x7FFF)), band(hi, 1)), k(16)), 0xFFFF);
        h = sel(isNan, bor(sign, k(0x7FC0)), rounded);
    } else {
        // Normal f16: rebias the exponent (112 << 23) and round the 13 dropped bits.
        MPSGraphTensor* normal =
            shr(sub(add(add(a, k(0xFFF)), band(shr(a, k(13)), 1)), k(0x38000000)), k(13));
        // Subnormal f16: h = RNE(m / 2^s) with m the 24-bit significand, s = 126 - e.
        MPSGraphTensor* m = bor(band(a, 0x7FFFFF), k(0x800000));
        MPSGraphTensor* s = sub(k(126), shr(a, k(23)));
        s = [g minimumWithPrimaryTensor:[g maximumWithPrimaryTensor:s secondaryTensor:k(1) name:nil]
                        secondaryTensor:k(31)
                                   name:nil];
        MPSGraphTensor* halfMinus1 = sub(shl(k(1), sub(s, k(1))), k(1));
        MPSGraphTensor* subnormal = shr(add(add(m, halfMinus1), band(shr(m, s), 1)), s);
        h = sel(ge(a, 0x38800000), normal, subnormal);
        h = sel(ge(a, 0x477FF000), k(0x7C00), h); // >= 65520 rounds to inf
        h = sel(isNan, k(0x7E00), h);
        h = bor(sign, h);
    }
    MPSGraphTensor* bits = [g castTensor:h toType:MPSDataTypeUInt16 name:nil];
    MPSGraphTensor* out = [g reinterpretCastTensor:bits toType:dtype name:nil];
    return scalar ? [g reshapeTensor:out withShape:@[] name:nil] : out;
}

static ProcessResult HandleConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("convert: invalid dtype for convert operation");
    }
    if (input.dataType == dtype) {
        SetOutputTensor(ctx.values, ctx.op, input);
        return ProcessResult{};
    }
    static const bool barrierDisabled = []() {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_CAST_FOLD_BARRIER");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    MPSGraph* g = ctx.graph;
    if (IsHalfWidthFloat(dtype)) {
        if (MPSGraphTensor* folded = FoldNarrowingConvert(g, ctx.op, dtype)) {
            return Result(ctx, folded, "convert");
        }
        if (!barrierDisabled && input.dataType == MPSDataTypeFloat32 && input.shape &&
            IsCompileTimeValue(ctx.op->getOperand(0))) {
            return Result(ctx, NarrowF32ByBits(g, input, dtype), "convert");
        }
    }
    if (!barrierDisabled && IsHalfWidthFloat(input.dataType)) {
        input = CastFoldBarrier(g, input);
    }
    MPSGraphTensor* result = [g castTensor:input toType:dtype name:nil];
    if (!barrierDisabled && IsHalfWidthFloat(dtype)) {
        result = CastFoldBarrier(g, result);
    }
    return Result(ctx, result, "convert");
}
REGISTER_MPS_OP("stablehlo.convert", HandleConvert);

static ProcessResult HandleSlice(HandlerContext& ctx) {
    auto sliceOp = mlir::dyn_cast<mlir::stablehlo::SliceOp>(ctx.op);
    if (!sliceOp) {
        return ProcessResult::Error("slice: expected SliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("slice: missing input tensor");

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    for (int64_t s : sliceOp.getStartIndices()) {
        [starts addObject:@(s)];
    }
    for (int64_t l : sliceOp.getLimitIndices()) {
        [ends addObject:@(l)];
    }
    for (int64_t s : sliceOp.getStrides()) {
        [strides addObject:@(s)];
    }

    // uint16 slice under JIT corrupted values near 65535 (int16 and other
    // dtypes were exact); slice the exact int32 widening and cast back.
    if (input.dataType == MPSDataTypeUInt16) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
        wide = [ctx.graph sliceTensor:wide starts:starts ends:ends strides:strides name:nil];
        return Result(ctx, [ctx.graph castTensor:wide toType:MPSDataTypeUInt16 name:nil], "slice");
    }
    MPSGraphTensor* result = [ctx.graph sliceTensor:input starts:starts ends:ends strides:strides name:nil];
    return Result(ctx, result, "slice");
}
REGISTER_MPS_OP("stablehlo.slice", HandleSlice);

static ProcessResult HandleDynamicSliceImpl(HandlerContext& ctx) {
    auto dynSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicSliceOp>(ctx.op);
    if (!dynSliceOp) {
        return ProcessResult::Error("dynamic_slice: expected DynamicSliceOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("dynamic_slice: missing input tensor");

    auto sliceSizes = dynSliceOp.getSliceSizes();
    NSUInteger rank = sliceSizes.size();
    NSArray<NSNumber*>* inputShape = input.shape;
    if (inputShape.count != rank) {
        return ProcessResult::Error("dynamic_slice: input rank does not match slice_sizes rank");
    }

    NSMutableArray<NSNumber*>* outputShape = [NSMutableArray array];
    for (int64_t s : sliceSizes) {
        [outputShape addObject:@(s)];
    }

    std::vector<int64_t> startConst(rank, 0);
    std::vector<bool> startIsConst(rank, false);
    bool allConst = true;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        int64_t v = 0;
        if (ReadConstantIntScalar(ctx.op->getOperand(dim + 1), v)) {
            startConst[dim] = v;
            startIsConst[dim] = true;
        } else {
            startIsConst[dim] = false;
            allConst = false;
        }
    }
    if (allConst) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            int64_t inputDim = [inputShape[dim] longLongValue];
            int64_t sliceSize = sliceSizes[dim];
            if (inputDim < 0) {
                return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
            }
            if (sliceSize < 0) {
                return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
            }
            int64_t maxStart = inputDim - sliceSize;
            if (maxStart < 0) {
                return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
            }
            int64_t s = std::min<int64_t>(std::max<int64_t>(startConst[dim], 0), maxStart);
            [starts addObject:@(s)];
            [ends addObject:@(s + sliceSize)];
            [strides addObject:@1];
        }
        MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                                 starts:starts
                                                   ends:ends
                                                strides:strides
                                                   name:nil];
        return Result(ctx, result, "dynamic_slice");
    }

    if (rank > 0) {
        if (@available(macOS 15.2, *)) {
            NSMutableArray<MPSGraphTensor*>* startsScalars = [NSMutableArray arrayWithCapacity:rank];
            std::vector<int32_t> sizes32;
            sizes32.reserve(rank);

            for (NSUInteger dim = 0; dim < rank; ++dim) {
                MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
                if (!startIdx) {
                    return ProcessResult::Error("dynamic_slice: missing start index for dimension");
                }

                startIdx = EnsureInt32(ctx.graph, startIdx);
                int64_t inputDim = [inputShape[dim] longLongValue];
                int64_t sliceSize = sliceSizes[dim];
                if (inputDim < 0) {
                    return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
                }
                if (sliceSize < 0) {
                    return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
                }
                int64_t maxStart = inputDim - sliceSize;
                if (maxStart < 0) {
                    return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
                }

                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:startIdx.dataType];
                startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                               secondaryTensor:maxStartTensor
                                                          name:nil];
                [startsScalars addObject:startIdx];

                sizes32.push_back((int32_t)sliceSize);
            }

            MPSGraphTensor* startsTensor = [ctx.graph stackTensors:startsScalars axis:0 name:nil];
            startsTensor = EnsureInt32(ctx.graph, startsTensor);

            NSData* sizeData = [NSData dataWithBytes:sizes32.data()
                                              length:(sizes32.size() * sizeof(int32_t))];
            MPSGraphTensor* sizeTensor = [ctx.graph constantWithData:sizeData
                                                               shape:@[@(rank)]
                                                            dataType:MPSDataTypeInt32];

            MPSGraphTensor* result = [ctx.graph sliceTensor:input
                                                startTensor:startsTensor
                                                 sizeTensor:sizeTensor
                                                squeezeMask:0
                                                       name:nil];
            return Result(ctx, result, "dynamic_slice");
        }
    }

    int64_t dynamicDim = -1;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        if (!startIsConst[dim]) {
            if (dynamicDim != -1) {
                dynamicDim = -2;
                break;
            }
            dynamicDim = (int64_t)dim;
        }
    }
    if (dynamicDim >= 0) {
        bool fullOtherDims = true;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            if ((int64_t)dim == dynamicDim) {
                continue;
            }
            int64_t inputDim = [inputShape[dim] longLongValue];
            int64_t sliceSize = sliceSizes[dim];
            if (inputDim < 0 || sliceSize < 0 || sliceSize != inputDim) {
                fullOtherDims = false;
                break;
            }
        }
        if (fullOtherDims) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dynamicDim + 1));
            if (!startIdx) {
                return ProcessResult::Error("dynamic_slice: missing start index for dimension");
            }

            startIdx = EnsureInt32(ctx.graph, startIdx);
            int64_t inputDim = [inputShape[(NSUInteger)dynamicDim] longLongValue];
            int64_t sliceSize = sliceSizes[(NSUInteger)dynamicDim];
            if (inputDim < 0) {
                return ProcessResult::Error("dynamic_slice: dynamic input shapes are not supported");
            }
            if (sliceSize < 0) {
                return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
            }
            int64_t maxStart = inputDim - sliceSize;
            if (maxStart < 0) {
                return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
            MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                  dataType:startIdx.dataType];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

            MPSGraphTensor* indices = nil;
            if (sliceSize == 1) {
                indices = [ctx.graph broadcastTensor:startIdx toShape:outputShape name:nil];
            } else {
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dynamicDim
                                                              withShape:outputShape
                                                                   name:nil];
                coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];
                indices = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:startIdx name:nil];
            }
            indices = EnsureInt32(ctx.graph, indices);

            MPSGraphTensor* result = [ctx.graph gatherAlongAxis:(NSInteger)dynamicDim
                                              withUpdatesTensor:input
                                                  indicesTensor:indices
                                                           name:nil];
            return Result(ctx, result, "dynamic_slice");
        }
    }

    NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
    for (NSUInteger dim = 0; dim < rank; dim++) {

        MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 1);
        if (!startIdx) {
            return ProcessResult::Error("dynamic_slice: missing start index for dimension");
        }

        startIdx = EnsureInt32(ctx.graph, startIdx);
        int64_t inputDim = [inputShape[dim] longLongValue];
        int64_t sliceSize = sliceSizes[dim];
        if (sliceSize < 0) {
            return ProcessResult::Error("dynamic_slice: negative slice size is not supported");
        }
        int64_t maxStart = inputDim - sliceSize;
        if (maxStart < 0) {
            return ProcessResult::Error("dynamic_slice: slice size exceeds input dimension");
        }
        MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
        MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart dataType:startIdx.dataType];
        startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
        startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

        MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                      withShape:outputShape
                                                           name:nil];
        coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

        MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                              secondaryTensor:startIdx
                                                                         name:nil];

        [indexTensors addObject:adjustedCoords];
    }

    MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];
    indices = EnsureInt32(ctx.graph, indices);

    MPSGraphTensor* result = [ctx.graph gatherNDWithUpdatesTensor:input
                                                    indicesTensor:indices
                                                  batchDimensions:0
                                                             name:nil];
    return Result(ctx, result, "dynamic_slice");
}
// Eager dynamic_slice with runtime start indices on float16/int8/uint8
// operands segfaulted the process (exit 139, alongside ANE compile errors),
// while float32 operands and every JIT form were correct. Widen 8/16-bit
// operands exactly (f16 -> f32, small ints -> int32), slice, and cast back.
static ProcessResult HandleDynamicSlice(HandlerContext& ctx) {
    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    if (!operand || ctx.op->getNumResults() != 1) {
        return HandleDynamicSliceImpl(ctx);
    }
    MPSDataType dtype = operand.dataType;
    MPSDataType wideType = MPSDataTypeInvalid;
    if (dtype == MPSDataTypeFloat16) {
        wideType = MPSDataTypeFloat32;
    } else if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeInt16 ||
               dtype == MPSDataTypeUInt16) {
        wideType = MPSDataTypeInt32;
    }
    if (wideType == MPSDataTypeInvalid) {
        return HandleDynamicSliceImpl(ctx);
    }
    // The segfault was eager-only (a standalone dynamic_slice). Inside a loop
    // body the widening cast covered the whole operand on every iteration:
    // an f16 scan over stacked layer weights read each layer by casting the
    // entire stack (GPT-2 medium f16 9.8 ms in a scan vs 2.3 ms unrolled).
    // JAX_METALLIB_DYNSLICE_WIDEN_IN_LOOP=1 restores widening in loops.
    static const bool widenInLoop = [] {
        const char* v = std::getenv("JAX_METALLIB_DYNSLICE_WIDEN_IN_LOOP");
        return v && v[0] != '\0' && !(v[0] == '0' && v[1] == '\0');
    }();
    if (ctx.isInControlFlow() && !widenInLoop) {
        return HandleDynamicSliceImpl(ctx);
    }
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    if (ctx.variables && ctx.variables->find(operandKey) != ctx.variables->end()) {
        return HandleDynamicSliceImpl(ctx);
    }
    for (unsigned i = 1; i < ctx.op->getNumOperands(); ++i) {
        if (ctx.op->getOperand(i) == ctx.op->getOperand(0)) {
            return HandleDynamicSliceImpl(ctx);
        }
    }

    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;
    ctx.values[operandKey] = [ctx.graph castTensor:operand toType:wideType name:nil];
    ProcessResult r = HandleDynamicSliceImpl(ctx);
    if (original) {
        ctx.values[operandKey] = original;
    } else {
        ctx.values.erase(operandKey);
    }
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* sliced = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!sliced || sliced.dataType != wideType) {
        ctx.values.erase(ctx.op->getResult(0).getAsOpaquePointer());
        return HandleDynamicSliceImpl(ctx);
    }
    return Result(ctx, [ctx.graph castTensor:sliced toType:dtype name:nil], "dynamic_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_slice", HandleDynamicSlice);

static ProcessResult HandleBitcastConvert(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("bitcast_convert: missing input tensor");

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("bitcast_convert: invalid dtype");
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    bool isScalar = (inputShape.count == 0);

    if (isScalar) {

        input = [ctx.graph reshapeTensor:input withShape:@[@1] name:nil];
    }

    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:input toType:dtype name:nil];

    // A scalar source can bitcast to a vector when the destination element is
    // narrower (e.g. f32 -> 2xu16); only restore the scalar shape if requested.
    if (isScalar && GetOutputShape(ctx.op).count == 0) {
        result = [ctx.graph reshapeTensor:result withShape:@[] name:nil];
    }

    // reinterpretCastTensor scales the last dimension when the element width
    // changes (u16[6,4,2] -> u32 gives 6x4x1, u32[6,4] -> u16 gives 6x8), while
    // StableHLO drops or appends that axis (6x4, 6x4x2). Consumers then saw the
    // wrong rank: add/shift/multiply/reverse aborted the process in MPSGraph and
    // a broadcasting add returned wrong values (build/probe_r20h_bitcast_shape.log).
    // The bytes are already in order, so reshape to the declared result shape.
    NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
    bool staticOut = outShape != nil;
    for (NSNumber* d : outShape) {
        staticOut = staticOut && d.longLongValue >= 0;
    }
    if (staticOut && !(result.shape && [result.shape isEqualToArray:outShape])) {
        result = [ctx.graph reshapeTensor:result withShape:outShape name:nil];
    }

    return Result(ctx, result, "bitcast_convert");
}
REGISTER_MPS_OP("stablehlo.bitcast_convert", HandleBitcastConvert);

static ProcessResult HandleConcatenate(HandlerContext& ctx) {
    auto concatOp = mlir::dyn_cast<mlir::stablehlo::ConcatenateOp>(ctx.op);
    if (!concatOp) {
        return ProcessResult::Error("concatenate: expected ConcatenateOp");
    }

    unsigned numOperands = ctx.op->getNumOperands();
    if (numOperands == 0) {
        return ProcessResult::Error("concatenate: no operands");
    }
    if (numOperands == 1) {
        MPSGraphTensor* only = GetInputTensor(ctx, 0);
        if (!only) {
            return ProcessResult::Error("concatenate: missing input tensor at operand 0");
        }
        SetOutputTensor(ctx.values, ctx.op, only);
        return ProcessResult{};
    }

    NSInteger dimension = static_cast<NSInteger>(concatOp.getDimension());

    // Under jit, MPSGraph concatTensors with a zero-size operand in a trailing
    // or middle position shifted the result (rows offset by one element),
    // which broke square jnp.linalg.svd(full_matrices=True). Zero-size operands
    // contribute nothing, so drop them; if all are empty, keep the first (its
    // shape equals the result shape).
    NSMutableArray<MPSGraphTensor*>* input_tensors =
        [NSMutableArray arrayWithCapacity:(NSUInteger)numOperands];
    MPSGraphTensor* firstTensor = nil;
    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* tensor = GetInputTensor(ctx, i);
        if (!tensor) {
            return ProcessResult::Error("concatenate: missing input tensor at operand " + std::to_string(i));
        }
        if (!firstTensor) {
            firstTensor = tensor;
        }
        auto operandType = mlir::dyn_cast<mlir::RankedTensorType>(ctx.op->getOperand(i).getType());
        if (operandType && dimension >= 0 && dimension < (NSInteger)operandType.getRank() &&
            operandType.getDimSize(dimension) == 0) {
            continue;
        }
        [input_tensors addObject:tensor];
    }

    if (input_tensors.count == 0) {
        SetOutputTensor(ctx.values, ctx.op, firstTensor);
        return ProcessResult{};
    }
    if (input_tensors.count == 1) {
        SetOutputTensor(ctx.values, ctx.op, input_tensors[0]);
        return ProcessResult{};
    }

    MPSGraphTensor* result = [ctx.graph concatTensors:input_tensors dimension:dimension name:nil];
    return Result(ctx, result, "concatenate");
}
REGISTER_MPS_OP("stablehlo.concatenate", HandleConcatenate);

static ProcessResult HandleSharding(HandlerContext& ctx) {
    unsigned numOperands = ctx.op->getNumOperands();
    unsigned numResults = ctx.op->getNumResults();

    if (numOperands != numResults) {
        return ProcessResult::Error(
            "sharding: operand/result arity mismatch (operands=" + std::to_string(numOperands) +
            ", results=" + std::to_string(numResults) + ")");
    }

    for (unsigned i = 0; i < numOperands; ++i) {
        MPSGraphTensor* input = GetInputTensor(ctx, i);
        if (!input) {
            return ProcessResult::Error("sharding: missing input tensor at operand " + std::to_string(i));
        }
        SetOutputTensor(ctx.values, ctx.op, input, i);
    }
    return ProcessResult{};
}
REGISTER_CUSTOM_CALL("Sharding", HandleSharding, sharding);

// MPSGraph scatterND (set) is not exact for every dtype: uint32 updates go
// through f32 (1322117303 -> 1322117248, 4294967294 -> 4294967295), bool
// updates are dropped, and complex tensors abort the process
// ("mps_scatter_nd"). Scatter uint32 as int32 bits, bool as uint8 and complex
// as separate real/imaginary parts; every other dtype scatters directly
// (f32/f16/bf16/int32/int8/uint8/uint16 verified exact).
static MPSGraphTensor* ScatterNDSetExact(MPSGraph* g, MPSGraphTensor* data, MPSGraphTensor* updates,
                                         MPSGraphTensor* indices) {
    auto scatter = [&](MPSGraphTensor* d, MPSGraphTensor* u) {
        return [g scatterNDWithDataTensor:d
                            updatesTensor:u
                            indicesTensor:indices
                          batchDimensions:0
                                     mode:MPSGraphScatterModeSet
                                     name:nil];
    };
    const MPSDataType dt = data.dataType;
    if (dt == MPSDataTypeComplexFloat32 || dt == MPSDataTypeComplexFloat16) {
        MPSGraphTensor* re = scatter([g realPartOfTensor:data name:nil], [g realPartOfTensor:updates
                                                                                        name:nil]);
        MPSGraphTensor* im = scatter([g imaginaryPartOfTensor:data name:nil], [g imaginaryPartOfTensor:updates
                                                                                                  name:nil]);
        return [g complexTensorWithRealTensor:re imaginaryTensor:im name:nil];
    }
    // reinterpretCastTensor of a rank-0 tensor aborts in the MPSGraph compiler.
    if (dt == MPSDataTypeUInt32 && data.shape.count > 0 && updates.shape.count > 0) {
        MPSGraphTensor* out = scatter([g reinterpretCastTensor:data toType:MPSDataTypeInt32 name:nil],
                                      [g reinterpretCastTensor:updates toType:MPSDataTypeInt32 name:nil]);
        return [g reinterpretCastTensor:out toType:MPSDataTypeUInt32 name:nil];
    }
    if (dt == MPSDataTypeBool) {
        MPSGraphTensor* out = scatter([g castTensor:data toType:MPSDataTypeUInt8 name:nil],
                                      [g castTensor:updates toType:MPSDataTypeUInt8 name:nil]);
        return [g notEqualWithPrimaryTensor:out
                            secondaryTensor:[g constantWithScalar:0 dataType:MPSDataTypeUInt8]
                                       name:nil];
    }
    return scatter(data, updates);
}

// Pad one axis at a time from reshape/slice/concat with the broadcast pad
// value. Exact for every dtype (the pad value never goes through f32) and needs
// no scatter or slice-update. Requires non-empty input and output.
static MPSGraphTensor* PadByConcat(MPSGraph* g, MPSGraphTensor* input, MPSGraphTensor* paddingValue,
                                   llvm::ArrayRef<int64_t> lowPad, llvm::ArrayRef<int64_t> highPad,
                                   llvm::ArrayRef<int64_t> interiorPad) {
    auto sliceAxis = [&](MPSGraphTensor* t, NSUInteger axis, int64_t start, int64_t end) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:t.shape.count];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:t.shape.count];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:t.shape.count];
        for (NSUInteger d = 0; d < t.shape.count; ++d) {
            [starts addObject:d == axis ? @(start) : @0];
            [ends addObject:d == axis ? @(end) : t.shape[d]];
            [strides addObject:@1];
        }
        return [g sliceTensor:t starts:starts ends:ends strides:strides name:nil];
    };
    auto fill = [&](MPSGraphTensor* t, NSUInteger axis, int64_t count) {
        NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:t.shape];
        shape[axis] = @(count);
        return [g broadcastTensor:paddingValue toShape:shape name:nil];
    };

    MPSGraphTensor* current = input;
    for (NSUInteger axis = 0; axis < input.shape.count; ++axis) {
        int64_t size = [current.shape[axis] longLongValue];
        int64_t interior = interiorPad[axis];
        if (interior > 0 && size > 1) {
            NSMutableArray<NSNumber*>* expanded = [NSMutableArray arrayWithArray:current.shape];
            [expanded insertObject:@1 atIndex:axis + 1];
            MPSGraphTensor* values = [g reshapeTensor:current withShape:expanded name:nil];
            MPSGraphTensor* gaps = fill(values, axis + 1, interior);
            MPSGraphTensor* joined = [g concatTensors:@[values, gaps]
                                            dimension:(NSInteger)(axis + 1)
                                                 name:nil];
            NSMutableArray<NSNumber*>* flat = [NSMutableArray arrayWithArray:current.shape];
            flat[axis] = @(size * (interior + 1));
            current = [g reshapeTensor:joined withShape:flat name:nil];
            size = size + (size - 1) * interior;
            current = sliceAxis(current, axis, 0, size);
        }
        int64_t low = lowPad[axis];
        int64_t high = highPad[axis];
        if (low < 0 || high < 0) {
            current =
                sliceAxis(current, axis, std::max<int64_t>(-low, 0), size - std::max<int64_t>(-high, 0));
        }
        NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:3];
        if (low > 0) {
            [parts addObject:fill(current, axis, low)];
        }
        [parts addObject:current];
        if (high > 0) {
            [parts addObject:fill(current, axis, high)];
        }
        if (parts.count > 1) {
            current = [g concatTensors:parts dimension:(NSInteger)axis name:nil];
        }
    }
    return current;
}

static ProcessResult HandlePad(HandlerContext& ctx) {
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(ctx.op);
    if (!padOp) {
        return ProcessResult::Error("pad: expected PadOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* paddingValue = GetInputTensor(ctx, 1);
    if (!input || !paddingValue)
        return ProcessResult::Error("pad: missing input tensor");

    auto edgePaddingLow = padOp.getEdgePaddingLow();
    auto edgePaddingHigh = padOp.getEdgePaddingHigh();
    auto interiorPadding = padOp.getInteriorPadding();
    if (edgePaddingLow.size() != edgePaddingHigh.size()) {
        return ProcessResult::Error("pad: edge padding rank mismatch");
    }
    if (edgePaddingLow.size() != interiorPadding.size()) {
        return ProcessResult::Error("pad: interior padding rank mismatch");
    }

    bool hasInteriorPadding = false;
    for (int64_t p : interiorPadding) {
        if (p < 0) {
            return ProcessResult::Error("pad: negative interior padding is not supported");
        }
        if (p != 0) {
            hasInteriorPadding = true;
            break;
        }
    }

    NSArray<NSNumber*>* inputShape = input.shape;
    if (inputShape.count != edgePaddingLow.size()) {
        return ProcessResult::Error("pad: input rank does not match padding rank");
    }

    NSArray<NSNumber*>* outputShape = GetOutputShape(ctx.op);

    // An all-zero pad is an identity, regardless of the padding value. Avoid
    // MPSGraph's slice-update of a broadcast tensor, which can discard complex
    // values produced by a preceding FFT in JIT gradients.
    bool noPadding = !hasInteriorPadding;
    for (size_t i = 0; i < edgePaddingLow.size(); ++i) {
        noPadding &= edgePaddingLow[i] == 0 && edgePaddingHigh[i] == 0;
    }
    if (noPadding && [inputShape isEqualToArray:outputShape]) {
        return Result(ctx, input, "pad");
    }

    // MPSGraph scatterND rejects complex tensors (aborting the process), and
    // slice-update into a broadcast complex base can drop the written values.
    // Build complex pads one axis at a time from reshape/slice/concat instead.
    bool complexInput =
        input.dataType == MPSDataTypeComplexFloat32 || input.dataType == MPSDataTypeComplexFloat16;
    bool complexNonEmpty = complexInput && outputShape.count == inputShape.count;
    for (NSNumber* d : inputShape) {
        complexNonEmpty &= [d longLongValue] > 0;
    }
    for (NSNumber* d : outputShape) {
        complexNonEmpty &= [d longLongValue] > 0;
    }
    if (complexNonEmpty) {
        MPSGraphTensor* current =
            PadByConcat(ctx.graph, input, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding);
        if (![current.shape isEqualToArray:outputShape]) {
            return ProcessResult::Error("pad: complex pad produced an unexpected output shape");
        }
        return Result(ctx, current, "pad");
    }

    auto getScalarPaddingValueDouble = [&]() -> std::optional<double> {
        auto* defOp = padOp.getPaddingValue().getDefiningOp();
        auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
        if (!constOp) {
            return std::nullopt;
        }
        auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
        if (!denseAttr || denseAttr.getNumElements() != 1) {
            return std::nullopt;
        }
        auto elemType = denseAttr.getElementType();
        if (elemType.isF16() || elemType.isF32() || elemType.isF64()) {
            llvm::APFloat ap = *denseAttr.getValues<llvm::APFloat>().begin();
            return ap.convertToDouble();
        }
        if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
            llvm::APInt ap = *denseAttr.getValues<llvm::APInt>().begin();

            if (intType.isUnsigned()) {
                return static_cast<double>(ap.getZExtValue());
            }
            return static_cast<double>(ap.getSExtValue());
        }
        return std::nullopt;
    };

    auto canUseNativeEdgePad = [&](bool needsInputSlice, bool hasNonEmptyUpdate) -> bool {
        if (needsInputSlice || !hasNonEmptyUpdate) {
            return false;
        }

        auto padConst = getScalarPaddingValueDouble();
        if (!padConst.has_value()) {
            return false;
        }
        // padTensor takes a double but materialises it through f32: integer pad
        // values beyond 2^24 were rounded or saturated (uint32 0xFFFFFFFD ->
        // 0xFFFFFFFF, int32 2147483645 -> INT32_MAX).
        const double pv = *padConst;
        if (!std::isnan(pv) && static_cast<double>(static_cast<float>(pv)) != pv) {
            return false;
        }
        // It also round-trips the operand through f32: int32 jnp.pad(x, 1,
        // constant_values=-1) returned 1638082997 as 1638082944 for every element
        // above 2^24. Integers wider than 16 bits take PadByConcat (exact).
        switch (input.dataType) {
        case MPSDataTypeInt32:
        case MPSDataTypeUInt32:
        case MPSDataTypeInt64:
        case MPSDataTypeUInt64:
            return false;
        default:
            break;
        }

        for (size_t i = 0; i < edgePaddingLow.size(); ++i) {
            if (edgePaddingLow[i] < 0 || edgePaddingHigh[i] < 0) {
                return false;
            }
        }

        for (NSNumber* d : inputShape) {
            if ([d longLongValue] == 0) {
                return false;
            }
        }
        for (NSNumber* d : outputShape) {
            if ([d longLongValue] == 0) {
                return false;
            }
        }
        return true;
    };

    MPSGraphTensor* padded = nil;
    auto getPadded = [&]() -> MPSGraphTensor* {
        if (!padded) {
            padded = [ctx.graph broadcastTensor:paddingValue toShape:outputShape name:nil];
        }
        return padded;
    };

    if (hasInteriorPadding) {
        NSUInteger rank = inputShape.count;
        NSMutableArray<NSNumber*>* preCropShape = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropStarts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropEnds = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* cropStrides = [NSMutableArray arrayWithCapacity:rank];

        bool hasElements = true;
        bool needsCrop = false;
        for (NSUInteger i = 0; i < rank; ++i) {
            int64_t low = edgePaddingLow[i];
            int64_t high = edgePaddingHigh[i];
            int64_t interior = interiorPadding[i];
            int64_t inputDim = [inputShape[i] longLongValue];
            int64_t outputDim = [outputShape[i] longLongValue];
            if (inputDim < 0 || outputDim < 0) {
                return ProcessResult::Error("pad: negative input/output dimension");
            }

            int64_t baseDim = 0;
            if (inputDim > 0) {
                baseDim = inputDim + (inputDim - 1) * interior;
            }
            int64_t preDim = std::max<int64_t>(low, 0) + baseDim + std::max<int64_t>(high, 0);
            int64_t cropLow = std::max<int64_t>(-low, 0);
            int64_t cropHigh = std::max<int64_t>(-high, 0);
            int64_t expectedOut = preDim - cropLow - cropHigh;
            if (expectedOut != outputDim) {
                return ProcessResult::Error("pad: inconsistent output shape for padding config");
            }

            if (low < 0 || high < 0) {
                needsCrop = true;
            }
            if (inputDim == 0 || outputDim == 0) {
                hasElements = false;
            }
            [preCropShape addObject:@(preDim)];
            [cropStarts addObject:@(cropLow)];
            [cropEnds addObject:@(cropLow + outputDim)];
            [cropStrides addObject:@1];
        }

        // MPSGraph's slice-update can silently discard interior-pad input values.
        // For a float32 row-only dilation, concatenate each row with its padding
        // instead of allocating and scattering a rank-vector index per element.
        if (!ctx.isInControlFlow() && rank == 2 && hasElements && !needsCrop &&
            input.dataType == MPSDataTypeFloat32 && [inputShape[0] longLongValue] > 1 &&
            interiorPadding[0] > 0 && interiorPadding[1] == 0 && edgePaddingLow[0] == 0 &&
            edgePaddingHigh[0] == 0 && edgePaddingLow[1] == 0 && edgePaddingHigh[1] == 0) {
            int64_t rows = [inputShape[0] longLongValue];
            int64_t cols = [inputShape[1] longLongValue];
            int64_t step = interiorPadding[0] + 1;
            MPSGraphTensor* rowValues = [ctx.graph reshapeTensor:input
                                                       withShape:@[@(rows), @1, @(cols)]
                                                            name:nil];
            MPSGraphTensor* gaps = [ctx.graph broadcastTensor:paddingValue
                                                      toShape:@[@(rows), @(interiorPadding[0]), @(cols)]
                                                         name:nil];
            MPSGraphTensor* interleaved = [ctx.graph concatTensors:@[rowValues, gaps] dimension:1 name:nil];
            MPSGraphTensor* flat = [ctx.graph reshapeTensor:interleaved
                                                  withShape:@[@(rows * step), @(cols)]
                                                       name:nil];
            MPSGraphTensor* result = [ctx.graph sliceTensor:flat
                                                     starts:@[@0, @0]
                                                       ends:outputShape
                                                    strides:@[@1, @1]
                                                       name:nil];
            return Result(ctx, result, "pad");
        }

        MPSGraphTensor* prePadded = [ctx.graph broadcastTensor:paddingValue toShape:preCropShape name:nil];

        if (!hasElements) {
            MPSGraphTensor* result = prePadded;
            if (needsCrop) {
                result = [ctx.graph sliceTensor:result
                                         starts:cropStarts
                                           ends:cropEnds
                                        strides:cropStrides
                                           name:nil];
            }
            if (outputShape && result) {
                result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
            }
            return Result(ctx, result, "pad");
        }

        NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                          withShape:inputShape
                                                               name:nil];
            int64_t interior = interiorPadding[dim];
            int64_t step = interior + 1;
            if (step != 1) {
                MPSGraphTensor* stepTensor = [ctx.graph constantWithScalar:step dataType:coords.dataType];
                coords = [ctx.graph multiplicationWithPrimaryTensor:coords
                                                    secondaryTensor:stepTensor
                                                               name:nil];
            }
            int64_t low = edgePaddingLow[dim];
            int64_t shift = std::max<int64_t>(low, 0);
            if (shift != 0) {
                MPSGraphTensor* shiftTensor = [ctx.graph constantWithScalar:shift dataType:coords.dataType];
                coords = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:shiftTensor name:nil];
            }
            [indexTensors addObject:coords];
        }

        MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];
        indices = EnsureInt32(ctx.graph, indices);
        // Plain scatterND dropped bool updates (every element came back as the
        // padding value) and rounded uint32 values through f32.
        MPSGraphTensor* result = ScatterNDSetExact(ctx.graph, prePadded, input, indices);
        if (needsCrop) {
            result = [ctx.graph sliceTensor:result
                                     starts:cropStarts
                                       ends:cropEnds
                                    strides:cropStrides
                                       name:nil];
        }
        if (outputShape && result) {
            result = [ctx.graph reshapeTensor:result withShape:outputShape name:nil];
        }
        return Result(ctx, result, "pad");
    }

    NSMutableArray<NSNumber*>* inputSliceStarts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* inputSliceEnds = [NSMutableArray array];
    NSMutableArray<NSNumber*>* inputSliceStrides = [NSMutableArray array];

    NSMutableArray<NSNumber*>* starts = [NSMutableArray array];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray array];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray array];

    bool needsInputSlice = false;
    bool hasNonEmptyUpdate = true;
    for (NSUInteger i = 0; i < edgePaddingLow.size(); i++) {
        int64_t low = edgePaddingLow[i];
        int64_t high = edgePaddingHigh[i];
        int64_t inputDim = [inputShape[i] longLongValue];
        int64_t cropLow = std::max<int64_t>(-low, 0);
        int64_t cropHigh = std::max<int64_t>(-high, 0);
        int64_t sliceStart = cropLow;
        int64_t sliceEnd = inputDim - cropHigh;
        if (sliceEnd < sliceStart) {
            sliceEnd = sliceStart;
        }
        int64_t slicedDim = sliceEnd - sliceStart;
        if (slicedDim == 0) {
            hasNonEmptyUpdate = false;
        }
        if (sliceStart != 0 || sliceEnd != inputDim) {
            needsInputSlice = true;
        }
        [inputSliceStarts addObject:@(sliceStart)];
        [inputSliceEnds addObject:@(sliceEnd)];
        [inputSliceStrides addObject:@1];

        int64_t outputStart = std::max<int64_t>(low, 0);
        [starts addObject:@(outputStart)];
        [ends addObject:@(outputStart + slicedDim)];
        [strides addObject:@1];
    }

    if (!hasNonEmptyUpdate) {
        return Result(ctx, getPadded(), "pad");
    }

    if (canUseNativeEdgePad(needsInputSlice, hasNonEmptyUpdate)) {
        auto padConst = getScalarPaddingValueDouble();
        NSMutableArray<NSNumber*>* left = [NSMutableArray arrayWithCapacity:inputShape.count];
        NSMutableArray<NSNumber*>* right = [NSMutableArray arrayWithCapacity:inputShape.count];
        for (NSUInteger i = 0; i < edgePaddingLow.size(); ++i) {
            [left addObject:@(edgePaddingLow[i])];
            [right addObject:@(edgePaddingHigh[i])];
        }
        MPSGraphTensor* out = [ctx.graph padTensor:input
                                   withPaddingMode:MPSGraphPaddingModeConstant
                                       leftPadding:left
                                      rightPadding:right
                                     constantValue:*padConst
                                              name:nil];
        return Result(ctx, out, "pad");
    }

    // Traced pad values, and constants f32 cannot represent, pad by
    // slice/concat. The earlier float route ((x - p) padded with 0, then + p)
    // was inexact (9053559 -> 9053558) and gave NaN everywhere for p = +-inf;
    // the slice-update of a broadcast base was the slowest route (f32 4096^2:
    // shift 4.5-4.7 ms, select mask 3.4-3.6 ms, concat 1.8-1.9 ms; native
    // constant pad 1.6 ms).
    MPSGraphTensor* result =
        PadByConcat(ctx.graph, input, paddingValue, edgePaddingLow, edgePaddingHigh, interiorPadding);
    if (![result.shape isEqualToArray:outputShape]) {
        return ProcessResult::Error("pad: edge pad produced an unexpected output shape");
    }
    return Result(ctx, result, "pad");
}
REGISTER_MPS_OP("stablehlo.pad", HandlePad);

// True when v is computed only from stablehlo.constant through pure scalar
// arithmetic/data movement (no block arguments). jnp index normalisation emits
// select(compare(c, 0), c + n, c) for concrete starts; MPSGraph folds such
// starts to constants just like literal ones.
static bool IsConstantDerivedValue(mlir::Value v, int depth = 0) {
    namespace shlo = mlir::stablehlo;
    if (depth > 8) {
        return false;
    }
    mlir::Operation* def = v.getDefiningOp();
    if (!def) {
        return false;
    }
    if (mlir::isa<shlo::ConstantOp>(def)) {
        return true;
    }
    if (!mlir::isa<shlo::CompareOp, shlo::SelectOp, shlo::AddOp, shlo::SubtractOp, shlo::MulOp,
                   shlo::ConvertOp, shlo::ClampOp, shlo::MaxOp, shlo::MinOp, shlo::NegOp, shlo::ReshapeOp,
                   shlo::BroadcastInDimOp, shlo::AndOp, shlo::OrOp, shlo::NotOp>(def)) {
        return false;
    }
    for (mlir::Value operand : def->getOperands()) {
        if (!IsConstantDerivedValue(operand, depth + 1)) {
            return false;
        }
    }
    return true;
}

static ProcessResult HandleDynamicUpdateSlice(HandlerContext& ctx) {
    auto updateSliceOp = mlir::dyn_cast<mlir::stablehlo::DynamicUpdateSliceOp>(ctx.op);
    if (!updateSliceOp) {
        return ProcessResult::Error("dynamic_update_slice: expected DynamicUpdateSliceOp");
    }

    MPSGraphTensor* baseVariable = nil;
    if (ctx.variables) {
        auto it = ctx.variables->find(ctx.op->getOperand(0).getAsOpaquePointer());
        if (it != ctx.variables->end()) {
            baseVariable = it->second;
        }
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* update = GetInputTensor(ctx, 1);
    if (!operand || !update)
        return ProcessResult::Error("dynamic_update_slice: missing input tensor");

    NSArray<NSNumber*>* operandShape = operand.shape;
    NSArray<NSNumber*>* updateShape = update.shape;
    NSUInteger rank = updateShape.count;
    if (operandShape.count != rank) {
        return ProcessResult::Error("dynamic_update_slice: operand rank does not match update rank");
    }

    MPSGraphTensor* outTensor = nil;
    // MPSGraph's slice-update may silently ignore the update when its data
    // tensor is a scalar broadcast (notably in JIT gradients). ScatterND
    // handles that input while preserving the same clamped index semantics.
    // Only route float32 here: MPSGraph's scatterND rejects complex tensors.
    bool scatterBroadcastOperand =
        !ctx.isInControlFlow() && operand.dataType == MPSDataTypeFloat32 &&
        static_cast<bool>(
            mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(ctx.op->getOperand(0).getDefiningOp()));

    std::vector<int64_t> startConst(rank, 0);
    std::vector<bool> startIsConst(rank, false);
    bool allConst = true;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        int64_t v = 0;
        if (ReadConstantIntScalar(ctx.op->getOperand(dim + 2), v)) {
            startConst[dim] = v;
            startIsConst[dim] = true;
        } else {
            startIsConst[dim] = false;
            allConst = false;
        }
    }
    // An update with the operand's shape replaces it entirely (every start
    // clamps to 0). MPSGraph's slice-update dropped that write with traced and
    // constant starts alike: jit dus(x * 2, x * 3, (0, 0)) returned x * 2.
    if ([operandShape isEqualToArray:updateShape]) {
        outTensor = update;
    }
    // Outside control flow, MPSGraph's slice-update with constant starts
    // silently drops the write whenever the operand or update is computed in
    // the same graph: jit dus(a[0], b[0], (5, 9)) and dus(x * 2, y * 3, (5, 9))
    // returned the operand unchanged for every dtype from 8x2 to 4096x1024.
    // The tensor-start variant fed the same constants drops it as well (MPSGraph
    // folds them), so constant starts take the scatterND route (Path D) there.
    // Traced starts and loop bodies (constant or traced) are exact and keep
    // slice-update.
    if (!outTensor && allConst && ctx.isInControlFlow()) {
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            if (operandDim < 0 || updateDim < 0) {
                return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
            }
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            int64_t s = std::min<int64_t>(std::max<int64_t>(startConst[dim], 0), maxStart);
            [starts addObject:@(s)];
            [ends addObject:@(s + updateDim)];
            [strides addObject:@1];
        }
        outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                        updateTensor:update
                                              starts:starts
                                                ends:ends
                                             strides:strides
                                                name:nil];
    }

    // Path A: sliceUpdateDataTensor with tensor starts (macOS 14.4+).
    // Only used outside control flow; inside while_loop bodies, MPSGraph internally
    // lowers this to a scatter op that fails shape verification.
    // Starts folded from constants (literal or via select/compare/add) drop the
    // write through slice-update at top level; they go to Path D instead.
    bool constDerivedStarts = allConst;
    if (!constDerivedStarts) {
        constDerivedStarts = true;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            constDerivedStarts &= IsConstantDerivedValue(ctx.op->getOperand(dim + 2));
        }
    }
    const bool topLevelFoldedStarts = constDerivedStarts && !ctx.isInControlFlow();
    if (!outTensor && !topLevelFoldedStarts && !ctx.isInControlFlow() && !scatterBroadcastOperand) {

        if (rank > 0) {
            if (@available(macOS 14.4, *)) {
                NSMutableArray<MPSGraphTensor*>* startsScalars = [NSMutableArray arrayWithCapacity:rank];
                std::vector<int32_t> sizes32;
                sizes32.reserve(rank);

                for (NSUInteger dim = 0; dim < rank; ++dim) {
                    MPSGraphTensor* startIdx = GetInputTensor(ctx, dim + 2);
                    if (!startIdx) {
                        return ProcessResult::Error("dynamic_update_slice: missing start index");
                    }
                    startIdx = EnsureInt32(ctx.graph, startIdx);

                    int64_t operandDim = [operandShape[dim] longLongValue];
                    int64_t updateDim = [updateShape[dim] longLongValue];
                    if (operandDim < 0 || updateDim < 0) {
                        return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
                    }
                    int64_t maxStart = operandDim - updateDim;
                    if (maxStart < 0) {
                        return ProcessResult::Error(
                            "dynamic_update_slice: update dimension exceeds operand dimension");
                    }

                    MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                    MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                          dataType:startIdx.dataType];
                    startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                    startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                                   secondaryTensor:maxStartTensor
                                                              name:nil];
                    [startsScalars addObject:startIdx];

                    sizes32.push_back((int32_t)updateDim);
                }

                MPSGraphTensor* startsTensor = [ctx.graph stackTensors:startsScalars axis:0 name:nil];
                startsTensor = EnsureInt32(ctx.graph, startsTensor);

                NSData* sizeData = [NSData dataWithBytes:sizes32.data()
                                                  length:(sizes32.size() * sizeof(int32_t))];
                MPSGraphTensor* sizeTensor = [ctx.graph constantWithData:sizeData
                                                                   shape:@[@(rank)]
                                                                dataType:MPSDataTypeInt32];

                MPSGraphTensor* endsTensor = [ctx.graph additionWithPrimaryTensor:startsTensor
                                                                  secondaryTensor:sizeTensor
                                                                             name:nil];
                MPSGraphTensor* stridesTensor = [ctx.graph constantWithScalar:1
                                                                        shape:@[@(rank)]
                                                                     dataType:MPSDataTypeInt32];

                if (@available(macOS 15.0, *)) {

                    outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                                    updateTensor:update
                                                    startsTensor:startsTensor
                                                      endsTensor:endsTensor
                                                   stridesTensor:stridesTensor
                                                            name:nil];
                } else {
                    outTensor = [ctx.graph sliceUpdateDataTensor:operand
                                                    updateTensor:update
                                                    startsTensor:startsTensor
                                                      endsTensor:endsTensor
                                                   stridesTensor:stridesTensor
                                                       startMask:0
                                                         endMask:0
                                                     squeezeMask:0
                                                            name:nil];
                }
            }
        }
    }

    // Path B: scatterAlongAxis. Inside loop bodies it was verified exact (float32
    // and int32 > 2**24, NaN/inf) and ~2-3x faster than the Path C mask for
    // multi-row updates, but slower for single-row updates, so in control flow
    // it is used only for float32/int32 updates spanning more than one row.
    if (!outTensor && !topLevelFoldedStarts) {

        int64_t dynamicDim = -1;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            if (!startIsConst[dim]) {
                if (dynamicDim != -1) {
                    dynamicDim = -2;
                    break;
                }
                dynamicDim = (int64_t)dim;
            }
        }
        if (dynamicDim >= 0) {
            bool fullOtherDims = true;
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                if ((int64_t)dim == dynamicDim) {
                    continue;
                }
                int64_t operandDim = [operandShape[dim] longLongValue];
                int64_t updateDim = [updateShape[dim] longLongValue];
                if (operandDim < 0 || updateDim < 0 || operandDim != updateDim) {
                    fullOtherDims = false;
                    break;
                }
            }
            if (fullOtherDims && ctx.isInControlFlow()) {
                const bool loopDtypeOk =
                    operand.dataType == MPSDataTypeFloat32 || operand.dataType == MPSDataTypeInt32;
                if (!loopDtypeOk || [updateShape[(NSUInteger)dynamicDim] longLongValue] <= 1) {
                    fullOtherDims = false;
                }
            }
            if (fullOtherDims) {
                MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dynamicDim + 2));
                if (!startIdx) {
                    return ProcessResult::Error("dynamic_update_slice: missing start index");
                }
                startIdx = EnsureInt32(ctx.graph, startIdx);

                int64_t operandDim = [operandShape[(NSUInteger)dynamicDim] longLongValue];
                int64_t updateDim = [updateShape[(NSUInteger)dynamicDim] longLongValue];
                if (operandDim < 0 || updateDim < 0) {
                    return ProcessResult::Error("dynamic_update_slice: dynamic shapes are not supported");
                }
                int64_t maxStart = operandDim - updateDim;
                if (maxStart < 0) {
                    return ProcessResult::Error(
                        "dynamic_update_slice: update dimension exceeds operand dimension");
                }
                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
                MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                      dataType:startIdx.dataType];
                startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
                startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx
                                               secondaryTensor:maxStartTensor
                                                          name:nil];

                MPSGraphTensor* indices = nil;
                if (updateDim == 1) {
                    indices = [ctx.graph broadcastTensor:startIdx toShape:updateShape name:nil];
                } else {
                    MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dynamicDim
                                                                  withShape:updateShape
                                                                       name:nil];
                    coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];
                    indices = [ctx.graph additionWithPrimaryTensor:coords secondaryTensor:startIdx name:nil];
                }
                indices = EnsureInt32(ctx.graph, indices);

                outTensor = [ctx.graph scatterAlongAxis:(NSInteger)dynamicDim
                                         withDataTensor:operand
                                          updatesTensor:update
                                          indicesTensor:indices
                                                   mode:MPSGraphScatterModeSet
                                                   name:nil];
            }
        }
    }

    // Path C: Masking-based fallback for control flow contexts.
    // Inside while_loop bodies, MPSGraph's scatter operations have shape verification bugs.
    // We implement dynamic_update_slice using coordinate comparisons and selectWithPredicateTensor
    // which only uses elementwise ops that work reliably inside control flow.
    if (!outTensor && ctx.isInControlFlow() && rank > 0) {
        // Clamp each start index to [0, operandDim - updateDim].
        NSMutableArray<MPSGraphTensor*>* clampedStarts = [NSMutableArray arrayWithCapacity:rank];
        bool dimsValid = true;
        for (NSUInteger dim = 0; dim < rank; ++dim) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, (unsigned)(dim + 2));
            if (!startIdx) {
                return ProcessResult::Error("dynamic_update_slice: missing start index");
            }
            startIdx = EnsureInt32(ctx.graph, startIdx);

            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            if (operandDim < 0 || updateDim < 0) {
                dimsValid = false;
                break;
            }
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
            MPSGraphTensor* maxStartT = [ctx.graph constantWithScalar:maxStart dataType:MPSDataTypeInt32];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartT name:nil];
            [clampedStarts addObject:startIdx];
        }

        if (dimsValid) {
            // Build a boolean mask over operandShape: true where the update region lies.
            // For each dimension, check: start[d] <= coord < start[d] + updateSize[d]
            // The combined mask is the AND of all per-dimension masks.
            MPSGraphTensor* combinedMask = nil;
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                int64_t updateDim = [updateShape[dim] longLongValue];
                MPSGraphTensor* startIdx = clampedStarts[dim];

                // Create coordinate tensor along this axis with operand shape.
                // Shape: broadcast-compatible — size operandDim along dim, 1 elsewhere.
                NSMutableArray<NSNumber*>* coordShape = [NSMutableArray arrayWithCapacity:rank];
                for (NSUInteger d = 0; d < rank; ++d) {
                    [coordShape addObject:(d == dim ? operandShape[d] : @1)];
                }
                MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                              withShape:coordShape
                                                                   name:nil];
                coords = [ctx.graph castTensor:coords toType:MPSDataTypeInt32 name:nil];

                MPSGraphTensor* endIdx =
                    [ctx.graph additionWithPrimaryTensor:startIdx
                                         secondaryTensor:[ctx.graph constantWithScalar:updateDim
                                                                              dataType:MPSDataTypeInt32]
                                                    name:nil];

                // mask_dim = (coords >= start) & (coords < end)
                MPSGraphTensor* geStart = [ctx.graph greaterThanOrEqualToWithPrimaryTensor:coords
                                                                           secondaryTensor:startIdx
                                                                                      name:nil];
                MPSGraphTensor* ltEnd = [ctx.graph lessThanWithPrimaryTensor:coords
                                                             secondaryTensor:endIdx
                                                                        name:nil];
                MPSGraphTensor* dimMask = [ctx.graph logicalANDWithPrimaryTensor:geStart
                                                                 secondaryTensor:ltEnd
                                                                            name:nil];

                if (!combinedMask) {
                    combinedMask = dimMask;
                } else {
                    combinedMask = [ctx.graph logicalANDWithPrimaryTensor:combinedMask
                                                          secondaryTensor:dimMask
                                                                     name:nil];
                }
            }

            // Build paddedUpdate (operand-shaped) using one_hot + matmul per dimension.
            // For each dim where updateDim < operandDim, expand the update using:
            //   indices = arange(updateDim) + start[d]
            //   selector = one_hot(indices, operandDim)  -> [updateDim, operandDim]
            //   selectorT = transpose(selector)          -> [operandDim, updateDim]
            //   Contract update along dim with selectorT to expand it.
            // This only uses one_hot, matmul, reshape - NO scatter/gather.

            MPSGraphTensor* paddedUpdate = update;

            // Expand each partial update axis to the operand extent without a
            // float32 one-hot matmul (which lost int32 precision above 2**24 and
            // turned inf*0 into NaN). Rows outside the update region are
            // discarded by the final select, so any in-range row may fill them.
            // - Real floating types: gather update row clamp(j - start) for each
            //   operand position j (exact, operand-sized intermediate).
            // - Integer/bool/complex: in-loop gatherAlongAxis measurably lost
            //   int32 precision above 2**24, so broadcast against a boolean
            //   one-hot selector and sum over the update axis instead. Each
            //   output receives exactly one value, so the sum is exact; the
            //   intermediate is updateDim x operand-sized.
            // A size-1 update axis needs no expansion: the final select's mask
            // picks exactly one position there and the update broadcasts.
            const MPSDataType updateDtype = paddedUpdate.dataType;
            const bool gatherExact = updateDtype == MPSDataTypeFloat32 || updateDtype == MPSDataTypeFloat16 ||
                                     updateDtype == MPSDataTypeBFloat16;
            const bool boolDtype = updateDtype == MPSDataTypeBool;
            if (boolDtype) {
                paddedUpdate = [ctx.graph castTensor:paddedUpdate toType:MPSDataTypeInt8 name:nil];
            }
            for (NSUInteger dim = 0; dim < rank; ++dim) {
                int64_t opDim = [operandShape[dim] longLongValue];
                int64_t upDim = [updateShape[dim] longLongValue];

                if (upDim == opDim || upDim == 1) {
                    continue; // No expansion needed on this dimension.
                }

                NSMutableArray<NSNumber*>* outShape = [NSMutableArray arrayWithArray:paddedUpdate.shape];
                outShape[dim] = @(opDim);

                if (gatherExact) {
                    MPSGraphTensor* positions = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                                     withShape:outShape
                                                                          name:nil];
                    positions = [ctx.graph castTensor:positions toType:MPSDataTypeInt32 name:nil];
                    MPSGraphTensor* rows = [ctx.graph subtractionWithPrimaryTensor:positions
                                                                   secondaryTensor:clampedStarts[dim]
                                                                              name:nil];
                    MPSGraphTensor* zeroIdx = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
                    MPSGraphTensor* lastIdx = [ctx.graph constantWithScalar:upDim - 1
                                                                   dataType:MPSDataTypeInt32];
                    rows = [ctx.graph maximumWithPrimaryTensor:rows secondaryTensor:zeroIdx name:nil];
                    rows = [ctx.graph minimumWithPrimaryTensor:rows secondaryTensor:lastIdx name:nil];
                    paddedUpdate = [ctx.graph gatherAlongAxis:(NSInteger)dim
                                            withUpdatesTensor:paddedUpdate
                                                indicesTensor:rows
                                                         name:nil];
                    continue;
                }

                // selector[k, j] = (start + k == j) -> [updateDim, operandDim]
                MPSGraphTensor* localCoords = [ctx.graph coordinateAlongAxis:0
                                                                   withShape:@[@(upDim), @1]
                                                                        name:nil];
                localCoords = [ctx.graph castTensor:localCoords toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* globalIndices = [ctx.graph additionWithPrimaryTensor:localCoords
                                                                     secondaryTensor:clampedStarts[dim]
                                                                                name:nil];
                MPSGraphTensor* targets = [ctx.graph coordinateAlongAxis:1
                                                               withShape:@[@1, @(opDim)]
                                                                    name:nil];
                targets = [ctx.graph castTensor:targets toType:MPSDataTypeInt32 name:nil];
                MPSGraphTensor* selector = [ctx.graph equalWithPrimaryTensor:globalIndices
                                                             secondaryTensor:targets
                                                                        name:nil];
                NSUInteger curRank = paddedUpdate.shape.count;
                NSMutableArray<NSNumber*>* selectorShape = [NSMutableArray arrayWithCapacity:curRank + 1];
                for (NSUInteger d = 0; d <= curRank; ++d) {
                    [selectorShape addObject:d == dim ? @(upDim) : (d == dim + 1 ? @(opDim) : @1)];
                }
                selector = [ctx.graph reshapeTensor:selector withShape:selectorShape name:nil];

                NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray arrayWithArray:paddedUpdate.shape];
                [expandedShape insertObject:@1 atIndex:dim + 1];
                MPSGraphTensor* expanded = [ctx.graph reshapeTensor:paddedUpdate
                                                          withShape:expandedShape
                                                               name:nil];
                MPSGraphTensor* zero = [ctx.graph constantWithScalar:0.0 dataType:paddedUpdate.dataType];
                MPSGraphTensor* picked = [ctx.graph selectWithPredicateTensor:selector
                                                          truePredicateTensor:expanded
                                                         falsePredicateTensor:zero
                                                                         name:nil];
                MPSGraphTensor* summed = [ctx.graph reductionSumWithTensor:picked
                                                                      axis:(NSInteger)dim
                                                                      name:nil];
                paddedUpdate = [ctx.graph reshapeTensor:summed withShape:outShape name:nil];
            }
            if (boolDtype) {
                paddedUpdate = [ctx.graph castTensor:paddedUpdate toType:MPSDataTypeBool name:nil];
            }

            // result = where(mask, paddedUpdate, operand)
            outTensor = [ctx.graph selectWithPredicateTensor:combinedMask
                                         truePredicateTensor:paddedUpdate
                                        falsePredicateTensor:operand
                                                        name:nil];
        }
    }

    // Path D: scatterND fallback (outside control flow only)
    if (!outTensor && !ctx.isInControlFlow()) {

        NSMutableArray<MPSGraphTensor*>* startIndices = [NSMutableArray array];
        for (NSUInteger i = 0; i < rank; i++) {
            MPSGraphTensor* startIdx = GetInputTensor(ctx, i + 2);
            if (!startIdx) {
                return ProcessResult::Error("dynamic_update_slice: missing start index");
            }
            [startIndices addObject:startIdx];
        }

        NSMutableArray<MPSGraphTensor*>* indexTensors = [NSMutableArray array];
        for (NSUInteger dim = 0; dim < rank; dim++) {
            MPSGraphTensor* startIdx = EnsureInt32(ctx.graph, startIndices[dim]);

            int64_t operandDim = [operandShape[dim] longLongValue];
            int64_t updateDim = [updateShape[dim] longLongValue];
            int64_t maxStart = operandDim - updateDim;
            if (maxStart < 0) {
                return ProcessResult::Error(
                    "dynamic_update_slice: update dimension exceeds operand dimension");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:startIdx.dataType];
            MPSGraphTensor* maxStartTensor = [ctx.graph constantWithScalar:maxStart
                                                                  dataType:startIdx.dataType];
            startIdx = [ctx.graph maximumWithPrimaryTensor:startIdx secondaryTensor:zero name:nil];
            startIdx = [ctx.graph minimumWithPrimaryTensor:startIdx secondaryTensor:maxStartTensor name:nil];

            MPSGraphTensor* coords = [ctx.graph coordinateAlongAxis:(NSInteger)dim
                                                          withShape:updateShape
                                                               name:nil];
            coords = [ctx.graph castTensor:coords toType:startIdx.dataType name:nil];

            MPSGraphTensor* adjustedCoords = [ctx.graph additionWithPrimaryTensor:coords
                                                                  secondaryTensor:startIdx
                                                                             name:nil];

            [indexTensors addObject:adjustedCoords];
        }

        MPSGraphTensor* indices = [ctx.graph stackTensors:indexTensors axis:(NSInteger)rank name:nil];

        indices = EnsureInt32(ctx.graph, indices);

        outTensor = ScatterNDSetExact(ctx.graph, operand, update, indices);
    }

    if (!outTensor) {
        return ProcessResult::Error("dynamic_update_slice: handler returned null");
    }

    if (baseVariable) {
        MPSGraphOperation* assign = [ctx.graph assignVariable:baseVariable
                                            withValueOfTensor:outTensor
                                                         name:nil];
        if (!assign) {
            return ProcessResult::Error("dynamic_update_slice: assignVariable returned null");
        }
        NSArray<MPSGraphTensor*>* forwarded =
            [ctx.graph controlDependencyWithOperations:@[assign]
                                        dependentBlock:^NSArray<MPSGraphTensor*>*() {
                                          return @[baseVariable];
                                        }
                                                  name:nil];
        if (!forwarded || forwarded.count != 1 || !forwarded[0]) {
            return ProcessResult::Error("dynamic_update_slice: controlDependency returned null");
        }
        MPSGraphTensor* varOut = forwarded[0];

        (*ctx.variables)[ctx.op->getResult(0).getAsOpaquePointer()] = varOut;
        SetOutputTensor(ctx.values, ctx.op, varOut);
        return ProcessResult{};
    }

    return Result(ctx, outTensor, "dynamic_update_slice");
}
REGISTER_MPS_OP("stablehlo.dynamic_update_slice", HandleDynamicUpdateSlice);

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

static bool ContainsDim(llvm::ArrayRef<int64_t> dims, int64_t dim) {
    for (int64_t d : dims) {
        if (d == dim)
            return true;
    }
    return false;
}

static bool IsIdentityPermutation(NSArray<NSNumber*>* permutation) {
    for (NSUInteger i = 0; i < permutation.count; ++i) {
        if ([permutation[i] integerValue] != (NSInteger)i) {
            return false;
        }
    }
    return true;
}

static std::string GatherDebugString(mlir::stablehlo::GatherOp gatherOp, MPSGraphTensor* operand,
                                     MPSGraphTensor* startIndices) {
    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNumbers.getStartIndicesBatchingDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    auto sliceSizes = gatherOp.getSliceSizes();
    std::ostringstream oss;
    oss << "operand_shape=" << ShapeToString(operand.shape)
        << ", indices_shape=" << ShapeToString(startIndices.shape) << ", offset_dims=" << JoinInts(offsetDims)
        << ", collapsed_slice_dims=" << JoinInts(collapsedSliceDims)
        << ", operand_batching_dims=" << JoinInts(operandBatchingDims)
        << ", start_indices_batching_dims=" << JoinInts(startIndicesBatchingDims)
        << ", start_index_map=" << JoinInts(startIndexMap)
        << ", index_vector_dim=" << dimNumbers.getIndexVectorDim()
        << ", slice_sizes=" << JoinInts(sliceSizes);
    return oss.str();
}

// StableHLO gather clamps every start index into [0, operand_dim - slice_size]
// before slicing. JAX relies on that for take(mode="clip") and for the gather
// under mode="fill"; MPSGraph gathers instead returned 0 for any out-of-range or
// negative index (take(x, [7, -2], axis=1, mode="clip") gave zeros). Component k
// of the index vector (along index_vector_dim) addresses operand dim
// start_index_map[k]. Returns the input unchanged when a bound is not static.
static MPSGraphTensor* ClampGatherStartIndices(MPSGraph* g, MPSGraphTensor* indices,
                                               NSArray<NSNumber*>* operandShape,
                                               llvm::ArrayRef<int64_t> startIndexMap,
                                               llvm::ArrayRef<int64_t> sliceSizes, int64_t indexVectorDim) {
    if (!indices || !indices.shape || !operandShape || startIndexMap.empty()) {
        return indices;
    }
    size_t k = startIndexMap.size();
    std::vector<int32_t> upper(k);
    for (size_t i = 0; i < k; ++i) {
        int64_t d = startIndexMap[i];
        if (d < 0 || d >= (int64_t)operandShape.count || d >= (int64_t)sliceSizes.size()) {
            return indices;
        }
        int64_t dim = [operandShape[(NSUInteger)d] longLongValue];
        if (dim < 0 || dim > INT32_MAX) {
            return indices;
        }
        upper[i] = (int32_t)std::max<int64_t>(0, dim - sliceSizes[(size_t)d]);
    }
    MPSDataType dt = indices.dataType;
    MPSGraphTensor* hi = nil;
    if (k == 1) {
        hi = [g constantWithScalar:(double)upper[0] dataType:dt];
    } else {
        NSUInteger rank = indices.shape.count;
        if (indexVectorDim < 0 || indexVectorDim >= (int64_t)rank ||
            [indices.shape[(NSUInteger)indexVectorDim] longLongValue] != (int64_t)k) {
            return indices;
        }
        NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger i = 0; i < rank; ++i) {
            [shape addObject:i == (NSUInteger)indexVectorDim ? @(k) : @1];
        }
        NSData* data = [NSData dataWithBytes:upper.data() length:k * sizeof(int32_t)];
        hi = [g constantWithData:data shape:shape dataType:MPSDataTypeInt32];
        if (dt != MPSDataTypeInt32) {
            hi = [g castTensor:hi toType:dt name:nil];
        }
    }
    MPSGraphTensor* lo = [g constantWithScalar:0 dataType:dt];
    MPSGraphTensor* clamped = [g maximumWithPrimaryTensor:indices secondaryTensor:lo name:nil];
    return [g minimumWithPrimaryTensor:clamped secondaryTensor:hi name:nil];
}

// Moves the index vector of gather/scatter start indices to the last axis
// (appending a unit axis when index_vector_dim == rank) and casts to int32.
static MPSGraphTensor* IndexVectorLast(MPSGraph* g, MPSGraphTensor* indices, int64_t indexVectorDim) {
    const NSUInteger rank = indices.shape.count;
    if (indexVectorDim == (int64_t)rank) {
        NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithArray:indices.shape];
        [shape addObject:@1];
        indices = [g reshapeTensor:indices withShape:shape name:nil];
    } else if (indexVectorDim != (int64_t)rank - 1) {
        NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger d = 0; d < rank; ++d) {
            if ((int64_t)d != indexVectorDim) {
                [perm addObject:@(d)];
            }
        }
        [perm addObject:@(indexVectorDim)];
        indices = [g transposeTensor:indices permutation:perm name:nil];
    }
    return EnsureInt32(g, indices);
}

// Component k of index-vector-last indices [batch..., K] as a rank-`rank`
// tensor with the batch sizes at `batchPositions` and 1 elsewhere, so it
// broadcasts against coordinates of the gather output or scatter updates.
static MPSGraphTensor* IndexComponent(MPSGraph* g, MPSGraphTensor* indices, int64_t k, NSUInteger rank,
                                      const std::vector<int64_t>& batchPositions) {
    const NSUInteger iRank = indices.shape.count;
    NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:iRank];
    NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:iRank];
    NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:iRank];
    for (NSUInteger d = 0; d < iRank; ++d) {
        const bool last = d + 1 == iRank;
        [starts addObject:last ? @(k) : @0];
        [ends addObject:last ? @(k + 1) : indices.shape[d]];
        [strides addObject:@1];
    }
    MPSGraphTensor* comp = [g sliceTensor:indices starts:starts ends:ends strides:strides name:nil];
    NSMutableArray<NSNumber*>* shape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger d = 0; d < rank; ++d) {
        [shape addObject:@1];
    }
    for (size_t b = 0; b < batchPositions.size(); ++b) {
        shape[(NSUInteger)batchPositions[b]] = indices.shape[b];
    }
    return [g reshapeTensor:comp withShape:shape name:nil];
}

// Position of a start-indices dim among the dims other than index_vector_dim.
static int64_t IndexBatchPosition(int64_t indicesDim, int64_t indexVectorDim) {
    return indicesDim < indexVectorDim ? indicesDim : indicesDim - 1;
}

static bool StaticNumel(NSArray<NSNumber*>* shape, int64_t* numel) {
    int64_t n = 1;
    for (NSNumber* d : shape) {
        const int64_t v = [d longLongValue];
        if (v < 0) {
            return false;
        }
        n *= v;
        if (n >= ((int64_t)1 << 31)) {
            return false;
        }
    }
    *numel = n;
    return true;
}

static MPSGraphTensor* AddIndexTerm(MPSGraph* g, MPSGraphTensor* sum, MPSGraphTensor* term) {
    return sum ? [g additionWithPrimaryTensor:sum secondaryTensor:term name:nil] : term;
}

// Explicit-coordinate gather: the StableHLO definition evaluated directly.
// Each output element's operand coordinate is the clamped start index plus
// the batching iota plus the window offset; the per-dim coordinates are
// stacked and gathered with gatherND. Only reached for layouts the specialised paths reject
// (vmapped dynamic_slice: operand_batching_dims with a multi-component index
// vector and windowed slices).
static ProcessResult GatherExplicitCoordinates(HandlerContext& ctx, mlir::stablehlo::GatherOp gatherOp) {
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* indices = GetInputTensor(ctx, 1);
    NSArray<NSNumber*>* outShape = GetOutputShape(ctx.op);
    int64_t operandNumel = 0;
    int64_t outNumel = 0;
    if (!operand || !indices || !outShape || outShape.count == 0 || operand.shape.count == 0 ||
        !StaticNumel(operand.shape, &operandNumel) || !StaticNumel(outShape, &outNumel) ||
        operandNumel == 0 || outNumel == 0) {
        return ProcessResult::Error("explicit gather needs static non-empty shapes");
    }
    auto dn = gatherOp.getDimensionNumbers();
    auto offsetDims = dn.getOffsetDims();
    auto collapsed = dn.getCollapsedSliceDims();
    auto operandBatching = dn.getOperandBatchingDims();
    auto indicesBatching = dn.getStartIndicesBatchingDims();
    auto startIndexMap = dn.getStartIndexMap();
    const int64_t ivd = dn.getIndexVectorDim();
    auto sliceSizes = gatherOp.getSliceSizes();
    NSArray<NSNumber*>* opShape = operand.shape;
    const NSUInteger R = opShape.count;
    const NSUInteger N = outShape.count;

    indices = IndexVectorLast(g, indices, ivd);
    const NSUInteger batchRank = indices.shape.count - 1;
    if (sliceSizes.size() != R || [indices.shape[batchRank] longLongValue] != (int64_t)startIndexMap.size() ||
        operandBatching.size() != indicesBatching.size()) {
        return ProcessResult::Error("explicit gather: inconsistent dimension numbers");
    }
    std::vector<int64_t> outBatchDims;
    for (NSUInteger d = 0; d < N; ++d) {
        if (!ContainsDim(offsetDims, (int64_t)d)) {
            outBatchDims.push_back((int64_t)d);
        }
    }
    std::vector<int64_t> windowDims;
    for (NSUInteger d = 0; d < R; ++d) {
        if (!ContainsDim(collapsed, (int64_t)d) && !ContainsDim(operandBatching, (int64_t)d)) {
            windowDims.push_back((int64_t)d);
        }
    }
    if (outBatchDims.size() != batchRank || windowDims.size() != offsetDims.size()) {
        return ProcessResult::Error("explicit gather: batch/window rank mismatch");
    }
    for (size_t b = 0; b < batchRank; ++b) {
        if ([outShape[(NSUInteger)outBatchDims[b]] longLongValue] != [indices.shape[b] longLongValue]) {
            return ProcessResult::Error("explicit gather: batch size mismatch");
        }
    }
    for (size_t i = 0; i < offsetDims.size(); ++i) {
        if (offsetDims[i] < 0 || offsetDims[i] >= (int64_t)N ||
            [outShape[(NSUInteger)offsetDims[i]] longLongValue] != sliceSizes[windowDims[i]]) {
            return ProcessResult::Error("explicit gather: window size mismatch");
        }
    }

    std::vector<MPSGraphTensor*> coords(R, nil);
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    for (size_t k = 0; k < startIndexMap.size(); ++k) {
        const int64_t d = startIndexMap[k];
        if (d < 0 || d >= (int64_t)R) {
            return ProcessResult::Error("explicit gather: invalid start_index_map");
        }
        const int64_t hi = [opShape[(NSUInteger)d] longLongValue] - sliceSizes[d];
        if (hi < 0) {
            return ProcessResult::Error("explicit gather: slice larger than operand");
        }
        MPSGraphTensor* s = IndexComponent(g, indices, (int64_t)k, N, outBatchDims);
        s = [g maximumWithPrimaryTensor:s secondaryTensor:zero name:nil];
        s = [g minimumWithPrimaryTensor:s
                        secondaryTensor:[g constantWithScalar:(double)hi dataType:MPSDataTypeInt32]
                                   name:nil];
        coords[(size_t)d] = AddIndexTerm(g, coords[(size_t)d], s);
    }
    for (size_t p = 0; p < operandBatching.size(); ++p) {
        const int64_t d = operandBatching[p];
        const int64_t bp = IndexBatchPosition(indicesBatching[p], ivd);
        if (d < 0 || d >= (int64_t)R || bp < 0 || bp >= (int64_t)batchRank) {
            return ProcessResult::Error("explicit gather: invalid batching dims");
        }
        MPSGraphTensor* iota = [g coordinateAlongAxis:(NSInteger)outBatchDims[(size_t)bp]
                                            withShape:outShape
                                                 name:nil];
        coords[(size_t)d] = AddIndexTerm(g, coords[(size_t)d], iota);
    }
    for (size_t i = 0; i < windowDims.size(); ++i) {
        MPSGraphTensor* offset = [g coordinateAlongAxis:(NSInteger)offsetDims[i] withShape:outShape name:nil];
        coords[(size_t)windowDims[i]] = AddIndexTerm(g, coords[(size_t)windowDims[i]], offset);
    }
    // Stack the per-dim coordinates into [out..., R] and gatherND on the
    // unflattened operand. A flat 1-D gatherAlongAxis of linear indices was
    // ~2x slower from 1M operand elements (vmap ds 256x[256,256]->32x32:
    // 1.33 vs 0.65 ms; 64x[256,256]->16: 0.475 vs 0.22 ms) and tied below.
    const bool isBool = operand.dataType == MPSDataTypeBool;
    NSMutableArray<NSNumber*>* coordShape = [outShape mutableCopy];
    [coordShape addObject:@1];
    NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:R];
    for (NSUInteger d = 0; d < R; ++d) {
        MPSGraphTensor* c = coords[(size_t)d] ? coords[(size_t)d] : zero;
        c = [g broadcastTensor:c toShape:outShape name:nil];
        [parts addObject:[g reshapeTensor:c withShape:coordShape name:nil]];
    }
    MPSGraphTensor* nd = R == 1 ? parts[0] : [g concatTensors:parts dimension:(NSInteger)N name:nil];
    MPSGraphTensor* src = isBool ? [g castTensor:operand toType:MPSDataTypeUInt8 name:nil] : operand;
    MPSGraphTensor* out = [g gatherNDWithUpdatesTensor:src indicesTensor:nd batchDimensions:0 name:nil];
    out = [g reshapeTensor:out withShape:outShape name:nil];
    if (isBool) {
        out = [g castTensor:out toType:MPSDataTypeBool name:nil];
    }
    return Result(ctx, out, "gather");
}

// Explicit-coordinate scatter: the StableHLO definition evaluated directly.
// Each update element's result coordinate (start + batching iota + window
// offset) is stacked per dim and scattered with scatterND. A window whose
// start does not fit entirely is sent out of bounds and dropped, like XLA's
// whole-window skip. Only reached for layouts the specialised paths reject
// (vmapped dynamic_update_slice).
static ProcessResult ScatterExplicitCoordinates(HandlerContext& ctx, mlir::stablehlo::ScatterOp scatterOp,
                                                MPSGraphScatterMode mode) {
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* indices = GetInputTensor(ctx, 1);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    int64_t inNumel = 0;
    int64_t updNumel = 0;
    if (!input || !indices || !updates || input.shape.count == 0 || updates.shape.count == 0 ||
        !StaticNumel(input.shape, &inNumel) || !StaticNumel(updates.shape, &updNumel) || inNumel == 0 ||
        updNumel == 0 || updates.dataType != input.dataType) {
        return ProcessResult::Error("explicit scatter needs static non-empty shapes of one dtype");
    }
    auto dn = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dn.getUpdateWindowDims();
    auto insertedWindowDims = dn.getInsertedWindowDims();
    auto toOperandDims = dn.getScatterDimsToOperandDims();
    auto inputBatching = dn.getInputBatchingDims();
    auto indicesBatching = dn.getScatterIndicesBatchingDims();
    const int64_t ivd = dn.getIndexVectorDim();
    NSArray<NSNumber*>* inShape = input.shape;
    NSArray<NSNumber*>* updShape = updates.shape;
    const NSUInteger R = inShape.count;
    const NSUInteger U = updShape.count;

    indices = IndexVectorLast(g, indices, ivd);
    const NSUInteger batchRank = indices.shape.count - 1;
    if ([indices.shape[batchRank] longLongValue] != (int64_t)toOperandDims.size() ||
        inputBatching.size() != indicesBatching.size()) {
        return ProcessResult::Error("explicit scatter: inconsistent dimension numbers");
    }
    std::vector<int64_t> updScatterDims;
    for (NSUInteger d = 0; d < U; ++d) {
        if (!ContainsDim(updateWindowDims, (int64_t)d)) {
            updScatterDims.push_back((int64_t)d);
        }
    }
    std::vector<int64_t> windowOperandDims;
    for (NSUInteger d = 0; d < R; ++d) {
        if (!ContainsDim(insertedWindowDims, (int64_t)d) && !ContainsDim(inputBatching, (int64_t)d)) {
            windowOperandDims.push_back((int64_t)d);
        }
    }
    if (updScatterDims.size() != batchRank || windowOperandDims.size() != updateWindowDims.size()) {
        return ProcessResult::Error("explicit scatter: batch/window rank mismatch");
    }
    for (size_t b = 0; b < batchRank; ++b) {
        if ([updShape[(NSUInteger)updScatterDims[b]] longLongValue] != [indices.shape[b] longLongValue]) {
            return ProcessResult::Error("explicit scatter: batch size mismatch");
        }
    }
    std::vector<int64_t> windowSize(R, 1);
    for (size_t i = 0; i < updateWindowDims.size(); ++i) {
        const int64_t ud = updateWindowDims[i];
        const int64_t wd = windowOperandDims[i];
        if (ud < 0 || ud >= (int64_t)U) {
            return ProcessResult::Error("explicit scatter: invalid update_window_dims");
        }
        windowSize[(size_t)wd] = [updShape[(NSUInteger)ud] longLongValue];
        if (windowSize[(size_t)wd] > [inShape[(NSUInteger)wd] longLongValue]) {
            return ProcessResult::Error("explicit scatter: window larger than input");
        }
    }

    std::vector<MPSGraphTensor*> coords(R, nil);
    MPSGraphTensor* valid = nil;
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    for (size_t k = 0; k < toOperandDims.size(); ++k) {
        const int64_t d = toOperandDims[k];
        if (d < 0 || d >= (int64_t)R) {
            return ProcessResult::Error("explicit scatter: invalid scatter_dims_to_operand_dims");
        }
        const int64_t hi = [inShape[(NSUInteger)d] longLongValue] - windowSize[(size_t)d];
        MPSGraphTensor* s = IndexComponent(g, indices, (int64_t)k, U, updScatterDims);
        MPSGraphTensor* ok =
            [g logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:s
                                                                    secondaryTensor:zero
                                                                               name:nil]
                           secondaryTensor:
                               [g lessThanOrEqualToWithPrimaryTensor:s
                                                     secondaryTensor:[g constantWithScalar:(double)hi
                                                                                  dataType:MPSDataTypeInt32]
                                                                name:nil]
                                      name:nil];
        valid = valid ? [g logicalANDWithPrimaryTensor:valid secondaryTensor:ok name:nil] : ok;
        coords[(size_t)d] = AddIndexTerm(g, coords[(size_t)d], s);
    }
    for (size_t p = 0; p < inputBatching.size(); ++p) {
        const int64_t d = inputBatching[p];
        const int64_t bp = IndexBatchPosition(indicesBatching[p], ivd);
        if (d < 0 || d >= (int64_t)R || bp < 0 || bp >= (int64_t)batchRank) {
            return ProcessResult::Error("explicit scatter: invalid batching dims");
        }
        MPSGraphTensor* iota = [g coordinateAlongAxis:(NSInteger)updScatterDims[(size_t)bp]
                                            withShape:updShape
                                                 name:nil];
        coords[(size_t)d] = AddIndexTerm(g, coords[(size_t)d], iota);
    }
    for (size_t i = 0; i < updateWindowDims.size(); ++i) {
        MPSGraphTensor* offset = [g coordinateAlongAxis:(NSInteger)updateWindowDims[i]
                                              withShape:updShape
                                                   name:nil];
        coords[(size_t)windowOperandDims[i]] = AddIndexTerm(g, coords[(size_t)windowOperandDims[i]], offset);
    }
    // Stack the per-dim coordinates into [upd..., R] and scatterND into the
    // unflattened input. Every element of a window that does not fit gets row
    // coordinate inShape[0]; scatterND drops out-of-bounds elements, so the
    // whole window is skipped without a dummy slot. The earlier flat 1-D
    // scatterAlongAxis with a concatenated dummy slot copied the input twice:
    // vmap dus 256x[256,256]<-32x32 4.31-4.34 ms vs 1.43-1.45 ms here.
    const bool isBool = input.dataType == MPSDataTypeBool;
    NSMutableArray<NSNumber*>* coordShape = [updShape mutableCopy];
    [coordShape addObject:@1];
    NSMutableArray<MPSGraphTensor*>* parts = [NSMutableArray arrayWithCapacity:R];
    for (NSUInteger d = 0; d < R; ++d) {
        MPSGraphTensor* c = coords[(size_t)d] ? coords[(size_t)d] : zero;
        if (valid && d == 0) {
            c = [g selectWithPredicateTensor:valid
                         truePredicateTensor:c
                        falsePredicateTensor:[g constantWithScalar:[inShape[0] doubleValue]
                                                          dataType:MPSDataTypeInt32]
                                        name:nil];
        }
        c = [g broadcastTensor:c toShape:updShape name:nil];
        [parts addObject:[g reshapeTensor:c withShape:coordShape name:nil]];
    }
    MPSGraphTensor* nd = R == 1 ? parts[0] : [g concatTensors:parts dimension:(NSInteger)U name:nil];
    // MPSGraph scatterND drops bool updates; scatter 0/1 bytes instead.
    MPSGraphTensor* data = isBool ? [g castTensor:input toType:MPSDataTypeUInt8 name:nil] : input;
    MPSGraphTensor* upd = isBool ? [g castTensor:updates toType:MPSDataTypeUInt8 name:nil] : updates;
    MPSGraphTensor* out = [g scatterNDWithDataTensor:data
                                       updatesTensor:upd
                                       indicesTensor:nd
                                     batchDimensions:0
                                                mode:mode
                                                name:nil];
    if (isBool) {
        out = [g notEqualWithPrimaryTensor:out
                           secondaryTensor:[g constantWithScalar:0 dataType:MPSDataTypeUInt8]
                                      name:nil];
    }
    return Result(ctx, out, "scatter");
}

static ProcessResult HandleGatherImpl(HandlerContext& ctx) {
    auto gatherOp = mlir::dyn_cast<mlir::stablehlo::GatherOp>(ctx.op);
    if (!gatherOp) {
        return ProcessResult::Error("gather: expected GatherOp");
    }

    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    MPSGraphTensor* startIndices = GetInputTensor(ctx, 1);
    if (!operand || !startIndices)
        return ProcessResult::Error("gather: missing input tensor");

    auto dimNumbers = gatherOp.getDimensionNumbers();
    auto offsetDims = dimNumbers.getOffsetDims();
    auto collapsedSliceDims = dimNumbers.getCollapsedSliceDims();
    auto operandBatchingDims = dimNumbers.getOperandBatchingDims();
    auto startIndicesBatchingDims = dimNumbers.getStartIndicesBatchingDims();
    auto startIndexMap = dimNumbers.getStartIndexMap();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();
    auto sliceSizes = gatherOp.getSliceSizes();
    startIndices = ClampGatherStartIndices(ctx.graph, startIndices, operand.shape, startIndexMap, sliceSizes,
                                           indexVectorDim);

    NSArray<NSNumber*>* indicesShape = startIndices.shape;
    NSArray<NSNumber*>* operandShape = operand.shape;
    NSUInteger indicesRank = indicesShape.count;
    NSUInteger operandRank = operandShape.count;

    bool collapsedGatherDim = startIndexMap.size() == 1 && collapsedSliceDims.size() == 1 &&
                              collapsedSliceDims[0] == startIndexMap[0];
    bool nonCollapsedGatherDim = startIndexMap.size() == 1 && collapsedSliceDims.empty();

    bool embeddingPattern = indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 &&
                            [indicesShape[indicesRank - 1] integerValue] == 1 && startIndexMap.size() == 1 &&
                            (collapsedGatherDim || nonCollapsedGatherDim) && startIndexMap[0] >= 0 &&
                            startIndexMap[0] < (int64_t)sliceSizes.size() &&
                            sliceSizes[startIndexMap[0]] == 1;
    if (embeddingPattern) {
        int64_t gatherAxis = startIndexMap[0];
        if (gatherAxis < 0 || gatherAxis >= (int64_t)operandRank) {
            return ProcessResult::Error("gather: gather axis out of bounds (" +
                                        GatherDebugString(gatherOp, operand, startIndices) + ")");
        }

        NSUInteger batchDimensions = 0;
        bool canonicalizeBatching = false;
        int64_t gatherAxisForMps = gatherAxis;
        std::vector<int64_t> indicesCanonNonVector;
        std::vector<int64_t> indicesCanonNonVectorInverse;

        if (!operandBatchingDims.empty() || !startIndicesBatchingDims.empty()) {
            if (operandBatchingDims.size() != startIndicesBatchingDims.size() ||
                operandBatchingDims.empty()) {
                return ProcessResult::Error("gather: unsupported operand/start_indices batching dims (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }

            size_t B = operandBatchingDims.size();

            auto isPrefixBatchDims = [](llvm::ArrayRef<int64_t> dims) -> bool {
                for (size_t i = 0; i < dims.size(); ++i) {
                    if (dims[i] != (int64_t)i) {
                        return false;
                    }
                }
                return true;
            };

            bool tierA =
                isPrefixBatchDims(operandBatchingDims) && isPrefixBatchDims(startIndicesBatchingDims);

            if (tierA && operandBatchingDims.size() == startIndicesBatchingDims.size()) {
                for (size_t i = 0; i < B; ++i) {
                    if (operandBatchingDims[i] != startIndicesBatchingDims[i]) {
                        tierA = false;
                        break;
                    }
                }
            }

            batchDimensions = (NSUInteger)B;

            if (!tierA) {
                canonicalizeBatching = true;

                NSMutableArray<NSNumber*>* operandPerm = [NSMutableArray arrayWithCapacity:operandRank];
                for (int64_t d : operandBatchingDims) {
                    if (d < 0 || d >= (int64_t)operandRank || d == gatherAxis) {
                        return ProcessResult::Error("gather: invalid operand_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    [operandPerm addObject:@(d)];
                }
                [operandPerm addObject:@(gatherAxis)];
                for (NSUInteger d = 0; d < operandRank; ++d) {
                    if ((int64_t)d == gatherAxis || ContainsDim(operandBatchingDims, (int64_t)d)) {
                        continue;
                    }
                    [operandPerm addObject:@((NSInteger)d)];
                }
                if (operandPerm.count != operandRank) {
                    return ProcessResult::Error("gather: invalid canonical operand permutation (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }

                if (!IsIdentityPermutation(operandPerm)) {
                    operand = [ctx.graph transposeTensor:operand permutation:operandPerm name:nil];
                    operandShape = operand.shape;
                }

                NSMutableArray<NSNumber*>* indicesPerm = [NSMutableArray arrayWithCapacity:indicesRank];
                std::vector<bool> used((size_t)indicesRank, false);
                indicesCanonNonVector.clear();
                indicesCanonNonVector.reserve(indicesRank > 0 ? indicesRank - 1 : 0);

                for (int64_t d : startIndicesBatchingDims) {
                    if (d < 0 || d >= (int64_t)indicesRank - 1) {
                        return ProcessResult::Error("gather: invalid start_indices_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    if (used[(size_t)d]) {
                        return ProcessResult::Error("gather: duplicate start_indices_batching_dims (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    used[(size_t)d] = true;
                    indicesCanonNonVector.push_back(d);
                    [indicesPerm addObject:@(d)];
                }
                for (NSUInteger d = 0; d < indicesRank - 1; ++d) {
                    if (used[(size_t)d]) {
                        continue;
                    }
                    used[(size_t)d] = true;
                    indicesCanonNonVector.push_back((int64_t)d);
                    [indicesPerm addObject:@((NSInteger)d)];
                }
                [indicesPerm addObject:@((NSInteger)indexVectorDim)];
                if (indicesPerm.count != indicesRank) {
                    return ProcessResult::Error("gather: invalid canonical indices permutation (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }

                if (!IsIdentityPermutation(indicesPerm)) {
                    startIndices = [ctx.graph transposeTensor:startIndices permutation:indicesPerm name:nil];
                    indicesShape = startIndices.shape;
                }

                gatherAxisForMps = (int64_t)B;

                size_t nonVectorRank = indicesCanonNonVector.size();
                indicesCanonNonVectorInverse.assign(nonVectorRank, -1);
                for (size_t i = 0; i < nonVectorRank; ++i) {
                    int64_t originalAxis = indicesCanonNonVector[i];
                    if (originalAxis < 0 || (size_t)originalAxis >= nonVectorRank) {
                        return ProcessResult::Error("gather: invalid indices non-vector canonicalization (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    indicesCanonNonVectorInverse[(size_t)originalAxis] = (int64_t)i;
                }
                for (size_t i = 0; i < nonVectorRank; ++i) {
                    if (indicesCanonNonVectorInverse[i] < 0) {
                        return ProcessResult::Error(
                            "gather: invalid indices non-vector inverse permutation (" +
                            GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                }
            }
        }

        if (sliceSizes.size() == operandRank) {
            bool needsSlice = false;
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:operandRank];
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                [starts addObject:@0];
                [strides addObject:@1];
                int64_t operandDim = [operand.shape[dim] longLongValue];
                int64_t end = operandDim;
                if ((int64_t)dim != gatherAxis && !ContainsDim(operandBatchingDims, (int64_t)dim)) {
                    int64_t slice = sliceSizes[dim];
                    if (slice < 0 || operandDim <= 0 || slice > operandDim) {
                        return ProcessResult::Error("gather: invalid slice_sizes for operand dim " +
                                                    std::to_string(dim) + " (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    end = slice;
                    if (slice != operandDim) {
                        needsSlice = true;
                    }
                }
                [ends addObject:@(end)];
            }
            if (needsSlice) {
                operand = [ctx.graph sliceTensor:operand starts:starts ends:ends strides:strides name:nil];
                operandShape = operand.shape;
            }
        }

        MPSGraphTensor* gatherIndices = startIndices;
        if (collapsedGatherDim) {

            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesShape.count - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }
            gatherIndices = [ctx.graph reshapeTensor:startIndices withShape:squeezedShape name:nil];
        }

        gatherIndices = EnsureInt32(ctx.graph, gatherIndices);

        MPSGraphTensor* result = [ctx.graph gatherWithUpdatesTensor:operand
                                                      indicesTensor:gatherIndices
                                                               axis:(NSUInteger)gatherAxisForMps
                                                    batchDimensions:batchDimensions
                                                               name:nil];

        if (canonicalizeBatching && !indicesCanonNonVectorInverse.empty() && result) {
            NSUInteger prefixRank = indicesCanonNonVectorInverse.size();
            NSMutableArray<NSNumber*>* resultPerm = [NSMutableArray arrayWithCapacity:result.shape.count];
            for (NSUInteger i = 0; i < prefixRank; ++i) {
                [resultPerm addObject:@(indicesCanonNonVectorInverse[i])];
            }

            for (NSUInteger i = prefixRank; i < result.shape.count; ++i) {
                [resultPerm addObject:@((NSInteger)i)];
            }
            if (resultPerm.count == result.shape.count && !IsIdentityPermutation(resultPerm)) {
                result = [ctx.graph transposeTensor:result permutation:resultPerm name:nil];
            }
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && result && ![expectedShape isEqualToArray:result.shape]) {
            result = [ctx.graph reshapeTensor:result withShape:expectedShape name:nil];
        }

        return Result(ctx, result, "gather");
    }

    bool pointGatherNdPattern =
        indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 && startIndexMap.size() > 1 &&
        operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
        (NSInteger)startIndexMap.size() == [indicesShape[indicesRank - 1] integerValue] &&
        (collapsedSliceDims.size() == startIndexMap.size() || collapsedSliceDims.empty()) &&
        sliceSizes.size() == operandRank;
    if (pointGatherNdPattern) {
        bool mappingValid = true;
        bool collapsedMappedDims = !collapsedSliceDims.empty();
        for (int64_t mappedDim : startIndexMap) {
            if (mappedDim < 0 || mappedDim >= (int64_t)operandRank ||
                (collapsedMappedDims && !ContainsDim(collapsedSliceDims, mappedDim)) ||
                sliceSizes[mappedDim] != 1) {
                mappingValid = false;
                break;
            }
        }
        if (!mappingValid) {
            pointGatherNdPattern = false;
        }
    }

    if (pointGatherNdPattern) {

        if (sliceSizes.size() == operandRank) {
            bool needsSlice = false;
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:operandRank];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:operandRank];
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                [starts addObject:@0];
                [strides addObject:@1];
                int64_t operandDim = [operand.shape[dim] longLongValue];
                int64_t end = operandDim;
                if (!ContainsDim(startIndexMap, (int64_t)dim)) {
                    int64_t slice = sliceSizes[dim];
                    if (slice < 0 || operandDim <= 0 || slice > operandDim) {
                        return ProcessResult::Error("gather: invalid slice_sizes for operand dim " +
                                                    std::to_string(dim) + " (" +
                                                    GatherDebugString(gatherOp, operand, startIndices) + ")");
                    }
                    end = slice;
                    if (slice != operandDim) {
                        needsSlice = true;
                    }
                }
                [ends addObject:@(end)];
            }
            if (needsSlice) {
                operand = [ctx.graph sliceTensor:operand starts:starts ends:ends strides:strides name:nil];
                operandShape = operand.shape;
            }
        }

        bool collapsedMappedDims = !collapsedSliceDims.empty();
        NSUInteger indexPrefixRank = indicesRank - 1;
        NSUInteger mappedRank = startIndexMap.size();
        NSUInteger remainingRank = operandRank - mappedRank;
        NSUInteger sliceDimsCount = collapsedMappedDims ? remainingRank : operandRank;
        NSUInteger outputRank = indexPrefixRank + sliceDimsCount;

        if (offsetDims.size() != sliceDimsCount) {
            return ProcessResult::Error("gather: unsupported gatherND offset dims (" +
                                        GatherDebugString(gatherOp, operand, startIndices) + ")");
        }

        NSMutableArray<NSNumber*>* operandPerm = [NSMutableArray arrayWithCapacity:operandRank];
        std::vector<bool> isMapped(operandRank, false);
        for (int64_t mappedDim : startIndexMap) {
            if (mappedDim < 0 || mappedDim >= (int64_t)operandRank || isMapped[mappedDim]) {
                return ProcessResult::Error("gather: invalid start_index_map for gatherND (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            isMapped[mappedDim] = true;
            [operandPerm addObject:@(mappedDim)];
        }
        for (NSUInteger dim = 0; dim < operandRank; ++dim) {
            if (!isMapped[dim]) {
                [operandPerm addObject:@((NSInteger)dim)];
            }
        }

        MPSGraphTensor* transposedOperand = operand;
        if (!IsIdentityPermutation(operandPerm)) {
            transposedOperand = [ctx.graph transposeTensor:operand permutation:operandPerm name:nil];
        }

        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, startIndices);
        MPSGraphTensor* gathered = [ctx.graph gatherNDWithUpdatesTensor:transposedOperand
                                                          indicesTensor:int32Indices
                                                        batchDimensions:0
                                                                   name:nil];

        MPSGraphTensor* expanded = gathered;
        if (!collapsedMappedDims) {
            NSMutableArray<NSNumber*>* expandedShape =
                [NSMutableArray arrayWithCapacity:indexPrefixRank + operandRank];
            for (NSUInteger i = 0; i < indexPrefixRank; ++i) {
                [expandedShape addObject:gathered.shape[i]];
            }
            for (NSUInteger dim = 0; dim < operandRank; ++dim) {
                int64_t slice = sliceSizes[dim];
                if (slice < 0) {
                    return ProcessResult::Error("gather: negative slice_sizes not supported (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }
                [expandedShape addObject:@(slice)];
            }
            expanded = [ctx.graph reshapeTensor:gathered withShape:expandedShape name:nil];
        }

        std::vector<int64_t> sliceDimToOutputAxis(sliceDimsCount, -1);
        std::vector<bool> outputAxisUsed(outputRank, false);
        for (NSUInteger s = 0; s < sliceDimsCount; ++s) {
            int64_t axis = offsetDims[s];
            if (axis < 0 || axis >= (int64_t)outputRank || outputAxisUsed[(size_t)axis]) {
                return ProcessResult::Error("gather: invalid offset_dims for gatherND (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            outputAxisUsed[(size_t)axis] = true;
            sliceDimToOutputAxis[s] = axis;
        }

        NSMutableArray<NSNumber*>* outputPermutation = [NSMutableArray arrayWithCapacity:outputRank];
        NSUInteger nextPrefixDim = 0;
        for (NSUInteger outAxis = 0; outAxis < outputRank; ++outAxis) {
            bool isOffsetAxis = false;
            NSUInteger sliceSourceIndex = 0;
            for (; sliceSourceIndex < sliceDimsCount; ++sliceSourceIndex) {
                if (sliceDimToOutputAxis[sliceSourceIndex] == (int64_t)outAxis) {
                    isOffsetAxis = true;
                    break;
                }
            }

            NSInteger sourceAxis = 0;
            if (isOffsetAxis) {
                sourceAxis = (NSInteger)(indexPrefixRank + sliceSourceIndex);
            } else {
                if (nextPrefixDim >= indexPrefixRank) {
                    return ProcessResult::Error("gather: invalid prefix/offset mapping (" +
                                                GatherDebugString(gatherOp, operand, startIndices) + ")");
                }
                sourceAxis = (NSInteger)nextPrefixDim;
                ++nextPrefixDim;
            }
            [outputPermutation addObject:@(sourceAxis)];
        }

        MPSGraphTensor* result = expanded;
        if (outputRank > 1 && !IsIdentityPermutation(outputPermutation)) {
            result = [ctx.graph transposeTensor:expanded permutation:outputPermutation name:nil];
        }
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && result && ![expectedShape isEqualToArray:result.shape]) {
            result = [ctx.graph reshapeTensor:result withShape:expectedShape name:nil];
        }
        return Result(ctx, result, "gather");
    }

    // Dynamic slice gather pattern: dynamic_slice (possibly via vmap) lowered to gather.
    // Handles both identity and non-identity start_index_map, and partial indexing
    // where only some operand dims are mapped.
    //
    // Pattern requirements:
    //   - index_vector_dim == indicesRank - 1 (last dim of indices is the index vector)
    //   - collapsed_slice_dims = [] (no dims collapsed)
    //   - offset_dims = [batchRank, batchRank+1, ..., batchRank+operandRank-1]
    //   - start_index_map entries are unique and valid operand dims
    //   - slice_sizes[mapped_dim] >= 1 (may be > 1, unlike embedding gather)
    //
    // Strategy: expand start indices to cover the full slice by adding range offsets,
    //           build full operandRank-dimensional indices, then use gatherND.
    bool dynamicSliceGatherPattern = indicesRank > 0 && indexVectorDim == (int64_t)indicesRank - 1 &&
                                     operandBatchingDims.empty() && startIndicesBatchingDims.empty() &&
                                     collapsedSliceDims.empty() && startIndexMap.size() > 0 &&
                                     sliceSizes.size() == operandRank;

    if (dynamicSliceGatherPattern) {
        // Validate index vector dim size matches start_index_map size
        // For indicesRank==1: the whole tensor is the index vector (no batch prefix)
        // indexVectorDim is the last axis, and its size should equal startIndexMap.size()
        int64_t indexVectorSize = [indicesShape[indexVectorDim] longLongValue];
        if ((int64_t)startIndexMap.size() != indexVectorSize) {
            dynamicSliceGatherPattern = false;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Validate start_index_map entries are valid and unique
        std::vector<bool> mappedDims(operandRank, false);
        for (int64_t d : startIndexMap) {
            if (d < 0 || d >= (int64_t)operandRank || mappedDims[(size_t)d]) {
                dynamicSliceGatherPattern = false;
                break;
            }
            mappedDims[(size_t)d] = true;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Validate offset_dims has operandRank entries with valid, strictly increasing values.
        // They may be interleaved with batch dims (not necessarily [batchRank, batchRank+1, ...]).
        NSUInteger batchRank = indicesRank - 1;
        NSUInteger outputRankDS = batchRank + operandRank;
        bool validOffsets = offsetDims.size() == operandRank;
        for (size_t i = 0; i < offsetDims.size() && validOffsets; ++i) {
            if (offsetDims[i] < 0 || offsetDims[i] >= (int64_t)outputRankDS) {
                validOffsets = false;
            }
            if (i > 0 && offsetDims[i] <= offsetDims[i - 1]) {
                validOffsets = false;
            }
        }
        if (!validOffsets) {
            dynamicSliceGatherPattern = false;
        }
    }

    if (dynamicSliceGatherPattern) {
        // Build expanded indices for gatherND.
        // For each operand dimension, we create a range covering slice_sizes[dim].
        // For mapped dims, the range is offset by the start index.
        // For unmapped dims, the range starts at 0.
        //
        // Expanded indices shape: [batch..., S0, S1, ..., S_{R-1}, R]
        // where R = operandRank and Si = sliceSizes[i]

        NSUInteger batchRank = indicesRank - 1;
        NSUInteger R = operandRank; // all operand dims in the expanded index

        // Build a mapping: for each operand dim, which index in startIndexMap provides it?
        // mappedIndexPos[dim] = position in startIndexMap, or -1 if unmapped
        std::vector<int64_t> mappedIndexPos(R, -1);
        for (size_t i = 0; i < startIndexMap.size(); ++i) {
            mappedIndexPos[(size_t)startIndexMap[i]] = (int64_t)i;
        }

        // Validate total slice size
        int64_t totalSlice = 1;
        for (NSUInteger d = 0; d < R; ++d) {
            if (sliceSizes[d] <= 0) {
                return ProcessResult::Error("gather: dynamic_slice pattern has non-positive slice_sizes (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
            totalSlice *= sliceSizes[d];
            if (totalSlice > 16 * 1024 * 1024) {
                return ProcessResult::Error("gather: dynamic_slice pattern slice too large (" +
                                            GatherDebugString(gatherOp, operand, startIndices) + ")");
            }
        }

        // expanded shape: [batch..., S0, S1, ..., S_{R-1}, R]
        NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < batchRank; ++i) {
            [expandedShape addObject:indicesShape[i]];
        }
        for (NSUInteger d = 0; d < R; ++d) {
            [expandedShape addObject:@(sliceSizes[d])];
        }
        [expandedShape addObject:@(R)];

        NSUInteger expandedRank = expandedShape.count;

        // Build per-dim index tensors and concatenate along last dim.
        // For each operand dim d:
        //   If mapped (startIndexMap has it): index = start[..., pos] + range(sliceSizes[d])
        //   If unmapped: index = range(sliceSizes[d])  (no start offset)
        //
        // Each produces shape [batch..., S0, ..., S_{R-1}, 1], concat → [..., R]

        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, startIndices);

        NSMutableArray<MPSGraphTensor*>* dimTensors = [NSMutableArray arrayWithCapacity:R];
        for (NSUInteger d = 0; d < R; ++d) {
            int64_t S = sliceSizes[d];

            // Create range [0, 1, ..., S-1] as int32
            int32_t* rangeData = (int32_t*)calloc((size_t)S, sizeof(int32_t));
            for (int32_t s = 0; s < (int32_t)S; ++s) {
                rangeData[s] = s;
            }
            NSData* rangeNSData = [NSData dataWithBytesNoCopy:rangeData
                                                       length:(NSUInteger)S * sizeof(int32_t)
                                                 freeWhenDone:YES];
            MPSGraphTensor* rangeTensor = [ctx.graph constantWithData:rangeNSData
                                                                shape:@[@(S)]
                                                             dataType:MPSDataTypeInt32];

            // Reshape range to [1, ..., 1, S, 1, ..., 1] with S at axis (batchRank + d)
            NSMutableArray<NSNumber*>* rangeShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < batchRank; ++i) {
                [rangeShape addObject:@1];
            }
            for (NSUInteger dd = 0; dd < R; ++dd) {
                [rangeShape addObject:(dd == d) ? @(S) : @1];
            }
            MPSGraphTensor* rangeReshaped = [ctx.graph reshapeTensor:rangeTensor
                                                           withShape:rangeShape
                                                                name:nil];

            // Broadcast to [batch..., S0, S1, ..., S_{R-1}]
            NSMutableArray<NSNumber*>* broadShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < batchRank; ++i) {
                [broadShape addObject:indicesShape[i]];
            }
            for (NSUInteger dd = 0; dd < R; ++dd) {
                [broadShape addObject:@(sliceSizes[dd])];
            }
            MPSGraphTensor* rangeBroadcasted = [ctx.graph broadcastTensor:rangeReshaped
                                                                  toShape:broadShape
                                                                     name:nil];

            MPSGraphTensor* dimIndex = rangeBroadcasted;

            if (mappedIndexPos[d] >= 0) {
                // This dim is mapped: add the start index
                int64_t pos = mappedIndexPos[d];

                // Extract the start value for this dim: startIndices[..., pos]
                // Slice along the index vector dim to get just this component
                NSMutableArray<NSNumber*>* sliceStarts = [NSMutableArray array];
                NSMutableArray<NSNumber*>* sliceEnds = [NSMutableArray array];
                NSMutableArray<NSNumber*>* sliceStrides = [NSMutableArray array];
                for (NSUInteger i = 0; i < indicesRank; ++i) {
                    if (i == (NSUInteger)indexVectorDim) {
                        [sliceStarts addObject:@(pos)];
                        [sliceEnds addObject:@(pos + 1)];
                    } else {
                        [sliceStarts addObject:@0];
                        [sliceEnds addObject:indicesShape[i]];
                    }
                    [sliceStrides addObject:@1];
                }
                MPSGraphTensor* startSlice = [ctx.graph sliceTensor:int32Indices
                                                             starts:sliceStarts
                                                               ends:sliceEnds
                                                            strides:sliceStrides
                                                               name:nil];

                // Reshape to [batch..., 1, 1, ..., 1] for broadcasting
                NSMutableArray<NSNumber*>* startBroadShape = [NSMutableArray array];
                for (NSUInteger i = 0; i < batchRank; ++i) {
                    [startBroadShape addObject:indicesShape[i]];
                }
                for (NSUInteger dd = 0; dd < R; ++dd) {
                    [startBroadShape addObject:@1];
                }
                MPSGraphTensor* startReshaped = [ctx.graph reshapeTensor:startSlice
                                                               withShape:startBroadShape
                                                                    name:nil];

                // Broadcast to [batch..., S0, ..., S_{R-1}]
                MPSGraphTensor* startBroadcasted = [ctx.graph broadcastTensor:startReshaped
                                                                      toShape:broadShape
                                                                         name:nil];

                // Add start + range
                dimIndex = [ctx.graph additionWithPrimaryTensor:startBroadcasted
                                                secondaryTensor:rangeBroadcasted
                                                           name:nil];
            }

            // Expand last dim: [batch..., S0, ..., S_{R-1}, 1]
            NSMutableArray<NSNumber*>* expandedDimShape = [NSMutableArray arrayWithArray:broadShape];
            [expandedDimShape addObject:@1];
            MPSGraphTensor* dimExpanded = [ctx.graph reshapeTensor:dimIndex
                                                         withShape:expandedDimShape
                                                              name:nil];
            [dimTensors addObject:dimExpanded];
        }

        // Concatenate along last dim → [batch..., S0, ..., S_{R-1}, R]
        MPSGraphTensor* expandedIndices = [ctx.graph concatTensors:dimTensors
                                                         dimension:(NSInteger)(expandedRank - 1)
                                                              name:nil];

        // gatherND: operand[expandedIndices]
        // Result shape: [batch..., S0, S1, ..., S_{R-1}]
        MPSGraphTensor* gathered = [ctx.graph gatherNDWithUpdatesTensor:operand
                                                          indicesTensor:expandedIndices
                                                        batchDimensions:0
                                                                   name:nil];

        // If offset_dims are not in the standard [batchRank, batchRank+1, ...] positions,
        // we need to permute the gathered result to place offset dims at the correct output
        // positions. The gathered result has layout [batch0, ..., batch_{B-1}, S0, ..., S_{R-1}].
        // We need to permute so that offset dim i maps to output position offsetDims[i].
        NSUInteger outputRankDS = batchRank + R;
        bool needsPermute = false;
        for (size_t i = 0; i < R; ++i) {
            if (offsetDims[i] != (int64_t)(batchRank + i)) {
                needsPermute = true;
                break;
            }
        }
        if (needsPermute) {
            // Build the output permutation.
            // Gathered layout: positions [0..batchRank-1] are batch dims, [batchRank..batchRank+R-1] are
            // slice dims. We need output[offsetDims[i]] = slice dim i, and the remaining positions filled by
            // batch dims in order.
            std::vector<bool> isOffsetPos(outputRankDS, false);
            for (size_t i = 0; i < R; ++i) {
                isOffsetPos[(size_t)offsetDims[i]] = true;
            }

            // Build inverse: for each output position, find which gathered position maps to it
            NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:outputRankDS];
            for (NSUInteger i = 0; i < outputRankDS; ++i) {
                [perm addObject:@0]; // placeholder
            }

            // Place offset dims: output position offsetDims[i] ← gathered position (batchRank + i)
            for (size_t i = 0; i < R; ++i) {
                perm[(NSUInteger)offsetDims[i]] = @((NSInteger)(batchRank + i));
            }

            // Place batch dims: remaining output positions ← gathered positions 0..batchRank-1
            NSUInteger batchIdx = 0;
            for (NSUInteger outPos = 0; outPos < outputRankDS; ++outPos) {
                if (!isOffsetPos[outPos]) {
                    perm[outPos] = @((NSInteger)batchIdx);
                    ++batchIdx;
                }
            }

            gathered = [ctx.graph transposeTensor:gathered permutation:perm name:nil];
        }

        // Reshape to expected output shape
        NSArray<NSNumber*>* expectedShape = GetOutputShape(ctx.op);
        if (expectedShape && gathered && ![expectedShape isEqualToArray:gathered.shape]) {
            gathered = [ctx.graph reshapeTensor:gathered withShape:expectedShape name:nil];
        }
        return Result(ctx, gathered, "gather");
    }

    ProcessResult explicitGather = GatherExplicitCoordinates(ctx, gatherOp);
    if (explicitGather.ok()) {
        return explicitGather;
    }
    return ProcessResult::Error("gather: unsupported gather pattern (" +
                                GatherDebugString(gatherOp, operand, startIndices) + "; " +
                                explicitGather.error + ")");
}
// MPSGraph gathers 32-bit integers inexactly: int32/uint32 values above 2**24
// came back off by 1-2 (inside and outside loops) while float32 gathers were
// exact, consistent with an internal float32 path. Gather the low and high
// 16-bit halves separately (each exact) and recombine them bitwise.
// mps.gather rejects complex operands with a process-aborting assertion.
// Gather the real and imaginary parts separately (exact) and recombine.
static ProcessResult HandleComplexGather(HandlerContext& ctx, MPSGraphTensor* operand) {
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;
    auto gatherPart = [&](MPSGraphTensor* part, MPSGraphTensor** out) -> ProcessResult {
        ctx.values[operandKey] = part;
        ProcessResult r = HandleGatherImpl(ctx);
        if (r.ok()) {
            *out = GetTensor(ctx.values, ctx.op->getResult(0));
        }
        return r;
    };
    MPSGraphTensor* gRe = nil;
    MPSGraphTensor* gIm = nil;
    ProcessResult r = gatherPart([ctx.graph realPartOfTensor:operand name:nil], &gRe);
    if (r.ok()) {
        r = gatherPart([ctx.graph imaginaryPartOfTensor:operand name:nil], &gIm);
    }
    if (original) {
        ctx.values[operandKey] = original;
    } else {
        ctx.values.erase(operandKey);
    }
    if (!r.ok()) {
        return r;
    }
    if (!gRe || !gIm) {
        return ProcessResult::Error("gather: complex split lowering failed");
    }
    return Result(ctx, [ctx.graph complexTensorWithRealTensor:gRe imaginaryTensor:gIm name:nil], "gather");
}

// MPSGraph gather rewrote 16-bit float NaNs to +qNaN (f16 -NaN 0xfe00 and
// payload NaNs became 0x7e00, bf16 ones 0x7fc0) and flushed bf16 subnormals to
// zero, while XLA:CPU gathers move raw bits (embedding lookups included).
// Gather the int16 view and reinterpret back. Falls back to the plain gather
// if the int16 route fails. JAX_METALLIB_DISABLE_GATHER16_RAW_BITS=1 opts out.
static ProcessResult HandleHalfFloatGather(HandlerContext& ctx, MPSGraphTensor* operand) {
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    void* resultKey = ctx.op->getResult(0).getAsOpaquePointer();
    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;
    const MPSDataType dtype = operand.dataType;
    ctx.values[operandKey] = [ctx.graph reinterpretCastTensor:operand toType:MPSDataTypeInt16 name:nil];
    ProcessResult r = HandleGatherImpl(ctx);
    MPSGraphTensor* raw = r.ok() ? GetTensor(ctx.values, ctx.op->getResult(0)) : nil;
    if (original) {
        ctx.values[operandKey] = original;
    } else {
        ctx.values.erase(operandKey);
    }
    if (!r.ok() || !raw || raw.dataType != MPSDataTypeInt16) {
        ctx.values.erase(resultKey);
        return HandleGatherImpl(ctx);
    }
    return Result(ctx, [ctx.graph reinterpretCastTensor:raw toType:dtype name:nil], "gather");
}

static bool Gather16RawBitsDisabled() {
    static const bool disabled = [] {
        const char* v = std::getenv("JAX_METALLIB_DISABLE_GATHER16_RAW_BITS");
        return v && v[0] == '1';
    }();
    return disabled;
}

// Graph lowering of a gather tagged by FuseTakePatterns (mps_executable.mm):
// operand 1 holds the raw [N] int32 jnp.take indices. Wrap negatives (i + R),
// gather the clamped rows (16-bit tables as raw int16), and select the fill
// bits where the wrapped index is out of range, on integer views so NaN
// payloads survive. Used when the native take step is not taken
// (JAX_METALLIB_TAKE_FUSED_GRAPH=1).
static ProcessResult HandleFusedTake(HandlerContext& ctx) {
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* table = GetInputTensor(ctx, 0);
    MPSGraphTensor* raw = GetInputTensor(ctx, 1);
    auto fillAttr = ctx.op->getAttrOfType<mlir::IntegerAttr>("__take_fill_bits");
    if (!table || !raw || !fillAttr || table.shape.count != 2 || raw.shape.count != 1) {
        return ProcessResult::Error("gather: malformed fused take");
    }
    const MPSDataType dt = table.dataType;
    const bool half = dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
    if (!half && dt != MPSDataTypeFloat32) {
        return ProcessResult::Error("gather: fused take supports f32/f16 tables only");
    }
    const MPSDataType bitsType = half ? MPSDataTypeInt16 : MPSDataTypeInt32;
    const int64_t rows = [table.shape[0] longLongValue];
    raw = EnsureInt32(g, raw);
    MPSGraphTensor* zero = [g constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* nrows = [g constantWithScalar:(double)rows dataType:MPSDataTypeInt32];
    MPSGraphTensor* last = [g constantWithScalar:(double)(rows - 1) dataType:MPSDataTypeInt32];
    MPSGraphTensor* neg = [g lessThanWithPrimaryTensor:raw secondaryTensor:zero name:nil];
    MPSGraphTensor* w = [g selectWithPredicateTensor:neg
                                 truePredicateTensor:[g additionWithPrimaryTensor:raw
                                                                  secondaryTensor:nrows
                                                                             name:nil]
                                falsePredicateTensor:raw
                                                name:nil];
    MPSGraphTensor* valid = [g
        logicalANDWithPrimaryTensor:[g greaterThanOrEqualToWithPrimaryTensor:w secondaryTensor:zero name:nil]
                    secondaryTensor:[g lessThanOrEqualToWithPrimaryTensor:w secondaryTensor:last name:nil]
                               name:nil];
    MPSGraphTensor* clamped = [g maximumWithPrimaryTensor:[g minimumWithPrimaryTensor:w
                                                                      secondaryTensor:last
                                                                                 name:nil]
                                          secondaryTensor:zero
                                                     name:nil];
    MPSGraphTensor* src = half ? [g reinterpretCastTensor:table toType:MPSDataTypeInt16 name:nil] : table;
    MPSGraphTensor* gathered = [g gatherWithUpdatesTensor:src
                                            indicesTensor:clamped
                                                     axis:0
                                          batchDimensions:0
                                                     name:nil];
    MPSGraphTensor* bits =
        half ? gathered : [g reinterpretCastTensor:gathered toType:MPSDataTypeInt32 name:nil];
    const uint64_t fill = static_cast<uint64_t>(fillAttr.getInt());
    const double fillValue = half ? (double)(int16_t)(uint16_t)fill : (double)(int32_t)(uint32_t)fill;
    MPSGraphTensor* fillT = [g constantWithScalar:fillValue dataType:bitsType];
    MPSGraphTensor* mask = [g reshapeTensor:valid withShape:@[raw.shape[0], @1] name:nil];
    MPSGraphTensor* out = [g selectWithPredicateTensor:mask
                                   truePredicateTensor:bits
                                  falsePredicateTensor:fillT
                                                  name:nil];
    return Result(ctx, [g reinterpretCastTensor:out toType:dt name:nil], "gather");
}

static ProcessResult HandleGather(HandlerContext& ctx) {
    if (ctx.op && ctx.op->hasAttr("__take_fused")) {
        return HandleFusedTake(ctx);
    }
    MPSGraphTensor* operand = GetInputTensor(ctx, 0);
    if (operand && (operand.dataType == MPSDataTypeFloat16 || operand.dataType == MPSDataTypeBFloat16) &&
        !Gather16RawBitsDisabled() && ctx.op->getNumOperands() >= 2 && ctx.op->getNumResults() == 1 &&
        ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        !(ctx.variables &&
          ctx.variables->find(ctx.op->getOperand(0).getAsOpaquePointer()) != ctx.variables->end())) {
        return HandleHalfFloatGather(ctx, operand);
    }
    if (operand &&
        (operand.dataType == MPSDataTypeComplexFloat32 || operand.dataType == MPSDataTypeComplexFloat16) &&
        ctx.op->getNumOperands() >= 2 && ctx.op->getNumResults() == 1 &&
        ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        !(ctx.variables &&
          ctx.variables->find(ctx.op->getOperand(0).getAsOpaquePointer()) != ctx.variables->end())) {
        return HandleComplexGather(ctx, operand);
    }
    if (!operand || (operand.dataType != MPSDataTypeInt32 && operand.dataType != MPSDataTypeUInt32) ||
        ctx.op->getNumOperands() < 2 || ctx.op->getNumResults() != 1 ||
        ctx.op->getOperand(0) == ctx.op->getOperand(1)) {
        return HandleGatherImpl(ctx);
    }
    void* operandKey = ctx.op->getOperand(0).getAsOpaquePointer();
    if (ctx.variables && ctx.variables->find(operandKey) != ctx.variables->end()) {
        return HandleGatherImpl(ctx);
    }
    void* resultKey = ctx.op->getResult(0).getAsOpaquePointer();
    auto saved = ctx.values.find(operandKey);
    MPSGraphTensor* original = saved != ctx.values.end() ? saved->second : nil;

    MPSDataType dtype = operand.dataType;
    MPSGraphTensor* mask = [ctx.graph constantWithScalar:0xFFFF dataType:dtype];
    MPSGraphTensor* sixteen = [ctx.graph constantWithScalar:16 dataType:dtype];
    MPSGraphTensor* lo = [ctx.graph bitwiseANDWithPrimaryTensor:operand secondaryTensor:mask name:nil];
    MPSGraphTensor* hi = [ctx.graph bitwiseRightShiftWithPrimaryTensor:operand
                                                       secondaryTensor:sixteen
                                                                  name:nil];
    hi = [ctx.graph bitwiseANDWithPrimaryTensor:hi secondaryTensor:mask name:nil];

    auto restore = [&]() {
        if (original) {
            ctx.values[operandKey] = original;
        } else {
            ctx.values.erase(operandKey);
        }
    };
    auto gatherHalf = [&](MPSGraphTensor* half, MPSGraphTensor** out) -> ProcessResult {
        ctx.values[operandKey] = half;
        ProcessResult r = HandleGatherImpl(ctx);
        if (r.ok()) {
            *out = GetTensor(ctx.values, ctx.op->getResult(0));
        }
        return r;
    };

    MPSGraphTensor* gatheredLo = nil;
    MPSGraphTensor* gatheredHi = nil;
    ProcessResult r = gatherHalf(lo, &gatheredLo);
    if (r.ok()) {
        r = gatherHalf(hi, &gatheredHi);
    }
    restore();
    if (!r.ok()) {
        return r;
    }
    if (!gatheredLo || !gatheredHi || gatheredLo.dataType != dtype || gatheredHi.dataType != dtype) {
        ctx.values.erase(resultKey);
        return HandleGatherImpl(ctx);
    }
    MPSGraphTensor* shifted = [ctx.graph bitwiseLeftShiftWithPrimaryTensor:gatheredHi
                                                           secondaryTensor:sixteen
                                                                      name:nil];
    MPSGraphTensor* combined = [ctx.graph bitwiseORWithPrimaryTensor:shifted
                                                     secondaryTensor:gatheredLo
                                                                name:nil];
    return Result(ctx, combined, "gather");
}
REGISTER_MPS_OP("stablehlo.gather", HandleGather);

static bool IsBoolScatterBody(mlir::Block& block) {
    if (block.getNumArguments() == 0) {
        return false;
    }
    auto t = mlir::dyn_cast<mlir::RankedTensorType>(block.getArgument(0).getType());
    return t && t.getElementType().isInteger(1);
}

// The only update bodies the lowering represents: `return upd` (set) or one
// `op(cur, upd)` with add/mul/max/min (either order), sub/div (cur first), and
// or/and on bool. GetScatterMode takes the first recognised op anywhere in the
// body and defaults to set, so x.at[i].apply(jnp.sin) (body `return sine(cur)`,
// zero updates) silently wrote zeros, and a multi-op combiner ran as its first
// op. apply cannot be lowered as one scatter at all: a duplicated index applies
// f once per occurrence. Anything else is rejected with an error.
static bool IsSupportedScatterBody(mlir::stablehlo::ScatterOp scatterOp, std::string* ops) {
    auto& region = scatterOp.getUpdateComputation();
    if (region.empty()) {
        return false;
    }
    mlir::Block& block = region.front();
    if (block.getNumArguments() != 2 || block.empty()) {
        return false;
    }
    mlir::Value cur = block.getArgument(0);
    mlir::Value upd = block.getArgument(1);
    mlir::Operation* term = &block.back();
    mlir::Operation* only = nullptr;
    size_t count = 0;
    for (mlir::Operation& op : block) {
        if (&op == term) {
            continue;
        }
        *ops += (count++ ? "," : "") + op.getName().getStringRef().str();
        only = &op;
    }
    if (term->getNumOperands() != 1) {
        return false;
    }
    mlir::Value r = term->getOperand(0);
    if (count == 0) {
        return r == upd;
    }
    if (count != 1 || only->getNumResults() != 1 || r != only->getResult(0) || only->getNumOperands() != 2) {
        return false;
    }
    bool fwd = only->getOperand(0) == cur && only->getOperand(1) == upd;
    bool rev = only->getOperand(0) == upd && only->getOperand(1) == cur;
    if (mlir::isa<mlir::stablehlo::AddOp, mlir::stablehlo::MulOp, mlir::stablehlo::MaxOp,
                  mlir::stablehlo::MinOp>(only)) {
        return fwd || rev;
    }
    if (mlir::isa<mlir::stablehlo::OrOp, mlir::stablehlo::AndOp>(only)) {
        return (fwd || rev) && IsBoolScatterBody(block);
    }
    if (mlir::isa<mlir::stablehlo::SubtractOp, mlir::stablehlo::DivOp>(only)) {
        return fwd;
    }
    return false;
}

static MPSGraphScatterMode GetScatterMode(mlir::stablehlo::ScatterOp scatterOp) {
    MPSGraphScatterMode mode = MPSGraphScatterModeSet;
    auto& updateRegion = scatterOp.getUpdateComputation();
    if (!updateRegion.empty()) {
        auto& block = updateRegion.front();
        for (auto& innerOp : block) {
            // bool or/and scatter through the 0/1 int32 widening: max == or, min == and.
            if (mlir::isa<mlir::stablehlo::OrOp>(innerOp) && IsBoolScatterBody(block)) {
                return MPSGraphScatterModeMax;
            }
            if (mlir::isa<mlir::stablehlo::AndOp>(innerOp) && IsBoolScatterBody(block)) {
                return MPSGraphScatterModeMin;
            }
            if (mlir::isa<mlir::stablehlo::AddOp>(innerOp)) {
                return MPSGraphScatterModeAdd;
            } else if (mlir::isa<mlir::stablehlo::SubtractOp>(innerOp)) {
                return MPSGraphScatterModeSub;
            } else if (mlir::isa<mlir::stablehlo::MulOp>(innerOp)) {
                return MPSGraphScatterModeMul;
            } else if (mlir::isa<mlir::stablehlo::DivOp>(innerOp)) {
                return MPSGraphScatterModeDiv;
            } else if (mlir::isa<mlir::stablehlo::MaxOp>(innerOp)) {
                return MPSGraphScatterModeMax;
            } else if (mlir::isa<mlir::stablehlo::MinOp>(innerOp)) {
                return MPSGraphScatterModeMin;
            }
        }
    }
    return mode;
}

static std::string ScatterDebugString(mlir::stablehlo::ScatterOp scatterOp, MPSGraphTensor* input,
                                      MPSGraphTensor* scatterIndices, MPSGraphTensor* updates) {
    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();
    std::ostringstream oss;
    oss << "input_shape=" << ShapeToString(input.shape)
        << ", indices_shape=" << ShapeToString(scatterIndices.shape)
        << ", updates_shape=" << ShapeToString(updates.shape)
        << ", update_window_dims=" << JoinInts(updateWindowDims)
        << ", inserted_window_dims=" << JoinInts(insertedWindowDims)
        << ", scatter_dims_to_operand_dims=" << JoinInts(scatterDimsToOperandDims)
        << ", input_batching_dims=" << JoinInts(inputBatchingDims)
        << ", scatter_indices_batching_dims=" << JoinInts(scatterIndicesBatchingDims)
        << ", index_vector_dim=" << dimNumbers.getIndexVectorDim();
    return oss.str();
}

// Parallel scatter optimization helpers

static bool CanUseParallelScatter(NSArray<NSNumber*>* indices, int64_t indexVectorLen,
                                  NSUInteger updatesRank, bool isAddMode) {
    if (updatesRank != 1) return false;
    if (indexVectorLen != 1) return false;
    if (!isAddMode) return true;  // Set mode is naturally parallel-safe
    return true;
}

static bool IndicesAreUnique(NSArray<NSNumber*>* indices, int64_t indexVectorLen) {
    if (indexVectorLen != 1) return false;
    NSUInteger n = indices.count;
    if (n == 0) return true;
    NSMutableSet<NSNumber*>* seen = [NSMutableSet setWithCapacity:n];
    for (NSUInteger i = 0; i < n; ++i) {
        NSNumber* idx = indices[i];
        if ([seen containsObject:idx]) {
            return false;
        }
        [seen addObject:idx];
    }
    return true;
}

static NSArray<NSNumber*>* ComputeFlattenedIndices(NSArray<NSNumber*>* indices,
                                                   NSArray<NSNumber*>* inputShape,
                                                   int64_t indexVectorLen,
                                                   NSArray<NSNumber*>* updateWindowDims,
                                                   NSArray<NSNumber*>* insertedWindowDims,
                                                   NSArray<NSNumber*>* scatterDimsToOperandDims) {
    if (indexVectorLen != 1) return nil;
    if (updateWindowDims.count != 0) return nil;
    if (insertedWindowDims.count != 1) return nil;
    if (scatterDimsToOperandDims.count != 1) return nil;
    if (insertedWindowDims[0].integerValue != 0) return nil;
    if (scatterDimsToOperandDims[0].integerValue != 0) return nil;

    NSUInteger n = indices.count;
    int64_t totalSize = inputShape[0].longLongValue;

    NSMutableArray<NSNumber*>* flatIndices = [NSMutableArray arrayWithCapacity:n];
    for (NSUInteger i = 0; i < n; ++i) {
        [flatIndices addObject:indices[i]];
    }
    return flatIndices;
}

// Sort indices and updates together by index value for cache-friendly parallel scatter
static std::tuple<MPSGraphTensor*, MPSGraphTensor*>
SortIndicesAndUpdatesForParallelScatter(HandlerContext& ctx,
                                        MPSGraphTensor* indices,
                                        MPSGraphTensor* updates,
                                        MPSGraphTensor* input) {
    (void)input;  // input is only used for graph-side context

    // argSort returns the permutation of indices that would sort them.
    // We then use that permutation to gather the corresponding rows of
    // `updates`, yielding a (sortedIndices, sortedUpdates) pair without
    // needing a fabricated range tensor (the SDK on this platform does not
    // expose a `rangeWithStart:limit:rate:` API).
    MPSGraphTensor* sortedIndices;
    if (indices.dataType == MPSDataTypeInt64) {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, indices);
        sortedIndices = [ctx.graph argSortWithTensor:int32Indices
                                                axis:0
                                          descending:NO
                                                name:nil];
    } else {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, indices);
        sortedIndices = [ctx.graph argSortWithTensor:int32Indices
                                                axis:0
                                          descending:NO
                                                name:nil];
    }

    MPSGraphTensor* sortedUpdates = [ctx.graph gatherAlongAxis:0
                                              withUpdatesTensor:updates
                                                  indicesTensor:sortedIndices
                                                           name:@"scatter_gather_sorted_updates"];

    return {sortedIndices, sortedUpdates};
}

// Parallel scatter using batched atomic updates for additive mode
static MPSGraphTensor* ParallelAddScatter(HandlerContext& ctx,
                                          MPSGraphTensor* input,
                                          MPSGraphTensor* indices,
                                          MPSGraphTensor* updates,
                                          int64_t indexVectorLen,
                                          NSArray<NSNumber*>* inputShape) {
    NSUInteger n = (NSUInteger)[updates.shape[0] unsignedIntegerValue];
    int64_t elemCount = 1;
    for (NSUInteger i = 1; i < inputShape.count; ++i) {
        elemCount *= inputShape[i].longLongValue;
    }

    MPSGraphTensor* result = input;

    if (elemCount > 0 && n > 0) {
        NSUInteger maxBatches = 256;
        NSUInteger batchSize = (n + maxBatches - 1) / maxBatches;
        if (batchSize < 8) batchSize = 8;
        if (batchSize > 256) batchSize = 256;

        NSUInteger numBatches = (n + batchSize - 1) / batchSize;

        NSMutableArray<MPSGraphTensor*>* partialResults = [NSMutableArray arrayWithCapacity:numBatches];

        for (NSUInteger b = 0; b < numBatches; ++b) {
            NSUInteger start = b * batchSize;
            NSUInteger end = start + batchSize;
            if (end > n) end = n;

            NSArray<NSNumber*>* sliceStart = @[@(start)];
            NSArray<NSNumber*>* sliceEnd = @[@(end)];

            MPSGraphTensor* batchIndices = [ctx.graph sliceTensor:indices
                                                           starts:sliceStart
                                                             ends:sliceEnd
                                                          strides:@[@1]
                                                             name:nil];
            MPSGraphTensor* batchUpdates = [ctx.graph sliceTensor:updates
                                                           starts:sliceStart
                                                             ends:sliceEnd
                                                          strides:@[@1]
                                                             name:nil];

            batchIndices = EnsureInt32(ctx.graph, batchIndices);

            MPSGraphTensor* batchResult = [ctx.graph scatterWithDataTensor:input
                                                            updatesTensor:batchUpdates
                                                            indicesTensor:batchIndices
                                                                     axis:0
                                                                     mode:MPSGraphScatterModeAdd
                                                                     name:nil];
            [partialResults addObject:batchResult];
        }

        if (partialResults.count == 1) {
            result = partialResults[0];
        } else {
            result = partialResults[0];
            for (NSUInteger i = 1; i < partialResults.count; ++i) {
                result = [ctx.graph additionWithPrimaryTensor:result
                                             secondaryTensor:partialResults[i]
                                                        name:nil];
            }
        }
    }

    return result;
}

// Write an already-scattered window (the operand's leading corner, starting at
// 0 on every axis) back into the operand. sliceUpdateDataTensor with these
// constant starts drops the write whenever the operand or window is computed
// in the same graph (every dtype), and plain scatterND is not exact for
// uint32/bool/complex, so reassemble axis by axis with slice/concat: at axis a
// the dims before a are already full, and the rest of the operand along a
// (restricted to the window on later dims) is concatenated after it.
static MPSGraphTensor* ReinsertWindowByConcat(MPSGraph* graph, MPSGraphTensor* input,
                                              MPSGraphTensor* window) {
    NSUInteger rank = window.shape.count;
    for (NSUInteger dim = 0; dim < rank; ++dim) {
        if ([window.shape[dim] longLongValue] == 0) {
            return input;
        }
    }
    MPSGraphTensor* current = window;
    for (NSUInteger axis = 0; axis < rank; ++axis) {
        int64_t w = [window.shape[axis] longLongValue];
        int64_t full = [input.shape[axis] longLongValue];
        if (w >= full) {
            continue;
        }
        NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:rank];
        NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:rank];
        for (NSUInteger d = 0; d < rank; ++d) {
            [starts addObject:d == axis ? @(w) : @0];
            [ends addObject:d <= axis ? input.shape[d] : window.shape[d]];
            [strides addObject:@1];
        }
        MPSGraphTensor* rest = [graph sliceTensor:input starts:starts ends:ends strides:strides name:nil];
        current = [graph concatTensors:@[current, rest] dimension:(NSInteger)axis name:nil];
    }
    return current;
}

static ProcessResult HandleScatterImpl(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp) {
        return ProcessResult::Error("scatter: expected ScatterOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* scatterIndices = GetInputTensor(ctx, 1);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    if (!input || !scatterIndices || !updates)
        return ProcessResult::Error("scatter: missing input tensor");

    auto dimNumbers = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dimNumbers.getUpdateWindowDims();
    auto insertedWindowDims = dimNumbers.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dimNumbers.getScatterDimsToOperandDims();
    auto inputBatchingDims = dimNumbers.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dimNumbers.getScatterIndicesBatchingDims();
    int64_t indexVectorDim = dimNumbers.getIndexVectorDim();

    NSArray<NSNumber*>* indicesShape = scatterIndices.shape;
    NSUInteger indicesRank = indicesShape.count;
    MPSGraphScatterMode mode = GetScatterMode(scatterOp);

    // Parallel scatter optimization for independent updates
    // Use sorted scatter + gather approach for additive scatter with unique indices
    bool isParallelOptimizable =
        inputBatchingDims.empty() &&
        scatterIndicesBatchingDims.empty() &&
        updateWindowDims.empty() &&
        insertedWindowDims.size() == 1 &&
        insertedWindowDims[0] == 0 &&
        scatterDimsToOperandDims.size() == 1 &&
        scatterDimsToOperandDims[0] == 0 &&
        indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 &&
        input.shape.count == 2 &&
        updates.shape.count == 1 &&
        scatterOp.getNumOperands() == 3;

    if (isParallelOptimizable) {
        NSUInteger n = (NSUInteger)[scatterIndices.shape[0] unsignedIntegerValue];
        if (n > 16) {
            MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, scatterIndices);

            if (mode == MPSGraphScatterModeSet) {
                // For set mode with unique indices: single parallel scatter
                // Sort indices for cache-friendly writes
                auto [sortedIndices, sortedUpdates] =
                    SortIndicesAndUpdatesForParallelScatter(ctx, int32Indices, updates, input);

                MPSGraphTensor* result = [ctx.graph scatterWithDataTensor:input
                                                            updatesTensor:sortedUpdates
                                                            indicesTensor:sortedIndices
                                                                     axis:0
                                                                     mode:mode
                                                                     name:nil];
                return Result(ctx, result, "scatter");
            }

            if (mode == MPSGraphScatterModeAdd) {
                // For add mode: use batched parallel scatter to avoid read-modify-write
                MPSGraphTensor* result = ParallelAddScatter(ctx, input, int32Indices, updates, 1, input.shape);
                return Result(ctx, result, "scatter");
            }
        }
    }

    // Original specialized paths
    if (mode == MPSGraphScatterModeAdd && input.shape.count == 2 && updates.shape.count == 1 &&
        inputBatchingDims.empty() && scatterIndicesBatchingDims.empty() && updateWindowDims.empty() &&
        insertedWindowDims.size() == 2 && scatterDimsToOperandDims.size() == 2 &&
        insertedWindowDims[0] == 0 && insertedWindowDims[1] == 1 && scatterDimsToOperandDims[0] == 0 &&
        scatterDimsToOperandDims[1] == 1 && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 2) {
        MPSGraphTensor* int32Indices = EnsureInt32(ctx.graph, scatterIndices);
        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:int32Indices
                                                    batchDimensions:0
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    if (mode == MPSGraphScatterModeAdd && input.shape.count == 3 && updates.shape.count == 2 &&
        inputBatchingDims.empty() && scatterIndicesBatchingDims.empty() && updateWindowDims.size() == 1 &&
        updateWindowDims[0] == 0 && insertedWindowDims.size() == 2 && insertedWindowDims[0] == 1 &&
        insertedWindowDims[1] == 2 && scatterDimsToOperandDims.size() == 2 &&
        scatterDimsToOperandDims[0] == 1 && scatterDimsToOperandDims[1] == 2 &&
        indexVectorDim == (int64_t)indicesRank - 1 && [indicesShape[indicesRank - 1] integerValue] == 2) {
        NSMutableArray<NSNumber*>* expandedIndicesShape = [NSMutableArray arrayWithCapacity:indicesRank + 1];
        [expandedIndicesShape addObject:@1];
        for (NSUInteger i = 0; i < indicesRank; ++i) {
            [expandedIndicesShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* expandedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:expandedIndicesShape
                                                              name:nil];

        NSMutableArray<NSNumber*>* broadcastShape = [NSMutableArray arrayWithCapacity:indicesRank + 1];
        [broadcastShape addObject:input.shape[0]];
        for (NSUInteger i = 0; i < indicesRank; ++i) {
            [broadcastShape addObject:indicesShape[i]];
        }
        MPSGraphTensor* batchedIndices = [ctx.graph broadcastTensor:expandedIndices
                                                            toShape:broadcastShape
                                                               name:nil];
        batchedIndices = EnsureInt32(ctx.graph, batchedIndices);

        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:batchedIndices
                                                    batchDimensions:1
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    if (!inputBatchingDims.empty() && !scatterIndicesBatchingDims.empty() &&
        inputBatchingDims.size() == scatterIndicesBatchingDims.size() &&
        scatterDimsToOperandDims.size() == 1 && insertedWindowDims.size() == 1 &&
        indexVectorDim == (int64_t)indicesRank - 1 && [indicesShape[indicesRank - 1] integerValue] == 1) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        if (insertedWindowDims[0] != scatterAxis) {

        } else {

            NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
            for (NSUInteger i = 0; i < indicesRank - 1; i++) {
                [squeezedShape addObject:indicesShape[i]];
            }
            MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                             withShape:squeezedShape
                                                                  name:nil];

            // Several non-batching scatter dims (the gradient of a vmapped row
            // gather with a 2-D index per batch: indices [B,S,T,1], updates
            // [B,S,T,F] for data [B,V,F]) gave scatterAlongAxis updates of
            // higher rank than the data and aborted MPSGraph ("indices rank and
            // result rank must be equal"). When those dims are contiguous and
            // land on the scatter axis, merge them into one: [B,S*T], [B,S*T,F].
            {
                const NSUInteger k = squeezedShape.count;
                const NSUInteger uRank = updates.shape.count;
                std::vector<NSUInteger> nonWindow;
                for (NSUInteger d = 0; d < uRank; ++d) {
                    bool isWindow = false;
                    for (auto w : updateWindowDims) {
                        if (w == static_cast<int64_t>(d)) {
                            isWindow = true;
                        }
                    }
                    if (!isWindow) {
                        nonWindow.push_back(d);
                    }
                }
                std::vector<NSUInteger> scatterJ;
                for (NSUInteger j = 0; j < k; ++j) {
                    bool isBatch = false;
                    for (auto b : scatterIndicesBatchingDims) {
                        if (b == static_cast<int64_t>(j)) {
                            isBatch = true;
                        }
                    }
                    if (!isBatch) {
                        scatterJ.push_back(j);
                    }
                }
                const NSUInteger m = scatterJ.size();
                bool mergeable = nonWindow.size() == k && m > 1 && uRank - (m - 1) == input.shape.count &&
                                 nonWindow[scatterJ[0]] == static_cast<NSUInteger>(scatterAxis);
                int64_t n = 1;
                for (NSUInteger j = 0; mergeable && j < m; ++j) {
                    mergeable = scatterJ[j] == scatterJ[0] + j &&
                                nonWindow[scatterJ[j]] == nonWindow[scatterJ[0]] + j &&
                                [updates.shape[nonWindow[scatterJ[j]]] longLongValue] ==
                                    [squeezedShape[scatterJ[j]] longLongValue];
                    n *= [squeezedShape[scatterJ[j]] longLongValue];
                }
                if (mergeable) {
                    NSMutableArray<NSNumber*>* newIdx = [NSMutableArray arrayWithCapacity:k - m + 1];
                    for (NSUInteger j = 0; j < k; ++j) {
                        if (j == scatterJ[0]) {
                            [newIdx addObject:@(n)];
                        } else if (j < scatterJ[0] || j >= scatterJ[0] + m) {
                            [newIdx addObject:squeezedShape[j]];
                        }
                    }
                    const NSUInteger p = nonWindow[scatterJ[0]];
                    NSMutableArray<NSNumber*>* newUpd = [NSMutableArray arrayWithCapacity:uRank - m + 1];
                    for (NSUInteger d = 0; d < uRank; ++d) {
                        if (d == p) {
                            [newUpd addObject:@(n)];
                        } else if (d < p || d >= p + m) {
                            [newUpd addObject:updates.shape[d]];
                        }
                    }
                    squeezedIndices = [ctx.graph reshapeTensor:squeezedIndices withShape:newIdx name:nil];
                    updates = [ctx.graph reshapeTensor:updates withShape:newUpd name:nil];
                }
            }

            // A vmapped column write (x.at[:, i].set(v) under vmap: input [B,R,C],
            // indices [B,1], updates [B,R], input_batching_dims=[0],
            // inserted_window_dims=[2]) has only batching scatter dims, so the
            // updates lack the inserted scatter axis and scatterAlongAxis aborted
            // the process ("indices rank and result rank must be equal"). When the
            // update dims map in order onto the operand dims without the scatter
            // axis, re-insert that axis as size 1 and place each index at its
            // operand batching dim.
            if (updates.shape.count + 1 == input.shape.count &&
                squeezedIndices.shape.count == scatterIndicesBatchingDims.size()) {
                const NSUInteger inRank = input.shape.count;
                const NSUInteger uRank = updates.shape.count;
                std::vector<int64_t> windowOperandDims;
                for (NSUInteger d = 0; d < inRank; ++d) {
                    if (!ContainsDim(insertedWindowDims, (int64_t)d) &&
                        !ContainsDim(inputBatchingDims, (int64_t)d)) {
                        windowOperandDims.push_back((int64_t)d);
                    }
                }
                bool mappable = updateWindowDims.size() == windowOperandDims.size();
                size_t nextWindow = 0;
                int64_t nextIndexDim = 0;
                NSMutableArray<NSNumber*>* idxShape = [NSMutableArray arrayWithCapacity:inRank];
                for (NSUInteger d = 0; d < inRank; ++d) {
                    [idxShape addObject:@1];
                }
                for (NSUInteger d = 0; mappable && d < uRank; ++d) {
                    const int64_t expected = (int64_t)d < scatterAxis ? (int64_t)d : (int64_t)d + 1;
                    int64_t mapped = -1;
                    if (ContainsDim(updateWindowDims, (int64_t)d)) {
                        if (nextWindow < windowOperandDims.size()) {
                            mapped = windowOperandDims[nextWindow++];
                        }
                    } else {
                        const int64_t j = nextIndexDim++;
                        for (size_t p = 0; p < scatterIndicesBatchingDims.size(); ++p) {
                            if (scatterIndicesBatchingDims[p] == j) {
                                mapped = inputBatchingDims[p];
                            }
                        }
                        if (mapped == expected) {
                            idxShape[(NSUInteger)expected] = squeezedIndices.shape[(NSUInteger)j];
                        }
                    }
                    mappable = mapped == expected && [updates.shape[d] longLongValue] <=
                                                         [input.shape[(NSUInteger)expected] longLongValue];
                }
                if (mappable) {
                    NSMutableArray<NSNumber*>* newUpd = [NSMutableArray arrayWithArray:updates.shape];
                    [newUpd insertObject:@1 atIndex:(NSUInteger)scatterAxis];
                    updates = [ctx.graph reshapeTensor:updates withShape:newUpd name:nil];
                    squeezedIndices = [ctx.graph reshapeTensor:squeezedIndices withShape:idxShape name:nil];
                }
            }

            MPSGraphTensor* scatterInput = input;
            bool needsWindowUpdate = false;
            NSMutableArray<NSNumber*>* windowStarts = [NSMutableArray arrayWithCapacity:input.shape.count];
            NSMutableArray<NSNumber*>* windowEnds = [NSMutableArray arrayWithCapacity:input.shape.count];
            NSMutableArray<NSNumber*>* windowStrides = [NSMutableArray arrayWithCapacity:input.shape.count];
            if (input.shape.count == updates.shape.count) {
                for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                    [windowStarts addObject:@0];
                    [windowStrides addObject:@1];
                    int64_t inDim = [input.shape[dim] longLongValue];
                    int64_t upDim = [updates.shape[dim] longLongValue];
                    int64_t end = inDim;

                    if ((int64_t)dim != scatterAxis) {
                        if (inDim <= 0 || upDim < 0) {
                            return ProcessResult::Error(
                                "scatter: dynamic/unknown shapes not supported for window scatter");
                        }
                        if (upDim > inDim) {
                            return ProcessResult::Error(
                                "scatter: updates window exceeds input dim (" +
                                ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                        }
                        if (upDim < inDim) {
                            end = upDim;
                            needsWindowUpdate = true;
                        }
                    }
                    [windowEnds addObject:@(end)];
                }

                if (needsWindowUpdate) {
                    scatterInput = [ctx.graph sliceTensor:input
                                                   starts:windowStarts
                                                     ends:windowEnds
                                                  strides:windowStrides
                                                     name:nil];
                }
            }

            MPSGraphTensor* indicesForScatter = squeezedIndices;
            NSUInteger idxRank = indicesForScatter.shape.count;
            NSUInteger updRank = updates.shape.count;
            if (idxRank > updRank) {
                return ProcessResult::Error("scatter: indices rank exceeds updates rank (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            if (idxRank < updRank) {
                NSMutableArray<NSNumber*>* expandedShape =
                    [NSMutableArray arrayWithCapacity:idxRank + (updRank - idxRank)];
                for (NSUInteger i = 0; i < idxRank; ++i) {
                    [expandedShape addObject:indicesForScatter.shape[i]];
                }
                for (NSUInteger i = idxRank; i < updRank; ++i) {
                    [expandedShape addObject:@1];
                }
                indicesForScatter = [ctx.graph reshapeTensor:indicesForScatter
                                                   withShape:expandedShape
                                                        name:nil];
            }
            if (updates.shape && indicesForScatter) {
                indicesForScatter = [ctx.graph broadcastTensor:indicesForScatter
                                                       toShape:updates.shape
                                                          name:nil];
            }
            indicesForScatter = EnsureInt32(ctx.graph, indicesForScatter);
            // scatterAlongAxis asserts (SIGABRT) unless data, updates and indices
            // share a rank; lower other layouts with explicit coordinates.
            if (updates.shape.count != scatterInput.shape.count ||
                indicesForScatter.shape.count != scatterInput.shape.count) {
                return ScatterExplicitCoordinates(ctx, scatterOp, mode);
            }

            MPSGraphTensor* resultWindow = [ctx.graph scatterAlongAxis:static_cast<NSInteger>(scatterAxis)
                                                        withDataTensor:scatterInput
                                                         updatesTensor:updates
                                                         indicesTensor:indicesForScatter
                                                                  mode:mode
                                                                  name:nil];

            if (!needsWindowUpdate) {
                return Result(ctx, resultWindow, "scatter");
            }
            return Result(ctx, ReinsertWindowByConcat(ctx.graph, input, resultWindow), "scatter");
        }
    }

    if (inputBatchingDims.empty() && indexVectorDim == (int64_t)indicesRank - 1 &&
        [indicesShape[indicesRank - 1] integerValue] == 1 && scatterDimsToOperandDims.size() == 1 &&
        insertedWindowDims.size() == 1 && insertedWindowDims[0] == scatterDimsToOperandDims[0]) {
        int64_t scatterAxis = scatterDimsToOperandDims[0];

        NSMutableArray<NSNumber*>* squeezedShape = [NSMutableArray array];
        for (NSUInteger i = 0; i < indicesRank - 1; i++) {
            [squeezedShape addObject:indicesShape[i]];
        }

        if (squeezedShape.count == 0)
            [squeezedShape addObject:@1];

        MPSGraphTensor* squeezedIndices = [ctx.graph reshapeTensor:scatterIndices
                                                         withShape:squeezedShape
                                                              name:nil];
        squeezedIndices = EnsureInt32(ctx.graph, squeezedIndices);

        if (updates.shape.count == 0)
            updates = [ctx.graph reshapeTensor:updates withShape:@[@1] name:nil];

        // Several scatter batch dims (the gradient of t[idx] with a 2-D index
        // array: indices [B,T,1], updates [B,T,F]; of t[:, idx]: updates
        // [F,B,T]) reached scatterWithDataTensor with updates rank > data rank
        // and aborted MPSGraph ("updates rank and result rank must be equal").
        // When the scatter dims of the updates are contiguous and start at the
        // scatter axis, merge them into one: indices [N], updates [..., N, ...].
        if (squeezedShape.count > 1) {
            const NSUInteger k = squeezedShape.count;
            const NSUInteger updRank = updates.shape.count;
            std::vector<NSUInteger> scatterPos;
            for (NSUInteger d = 0; d < updRank; ++d) {
                bool isWindow = false;
                for (auto w : updateWindowDims) {
                    if (w == static_cast<int64_t>(d)) {
                        isWindow = true;
                    }
                }
                if (!isWindow) {
                    scatterPos.push_back(d);
                }
            }
            bool mergeable = scatterPos.size() == k && scatterPos[0] == static_cast<NSUInteger>(scatterAxis);
            for (NSUInteger j = 1; mergeable && j < k; ++j) {
                mergeable = scatterPos[j] == scatterPos[0] + j;
            }
            int64_t n = 1;
            for (NSUInteger j = 0; mergeable && j < k; ++j) {
                mergeable = [updates.shape[scatterPos[j]] longLongValue] == [squeezedShape[j] longLongValue];
                n *= [squeezedShape[j] longLongValue];
            }
            if (mergeable) {
                const NSUInteger p = scatterPos[0];
                NSMutableArray<NSNumber*>* flatUpdates = [NSMutableArray arrayWithCapacity:updRank - k + 1];
                for (NSUInteger d = 0; d < p; ++d) {
                    [flatUpdates addObject:updates.shape[d]];
                }
                [flatUpdates addObject:@(n)];
                for (NSUInteger d = p + k; d < updRank; ++d) {
                    [flatUpdates addObject:updates.shape[d]];
                }
                squeezedIndices = [ctx.graph reshapeTensor:squeezedIndices withShape:@[@(n)] name:nil];
                updates = [ctx.graph reshapeTensor:updates withShape:flatUpdates name:nil];
                squeezedShape = [NSMutableArray arrayWithObject:@(n)];
            }
        }

        // Inside control flow, MPSGraph's scatter ops have shape verification bugs.
        // Use one_hot + arithmetic approach instead.
        if (ctx.isInControlFlow() && mode == MPSGraphScatterModeSet) {
            NSUInteger inputRank = input.shape.count;
            int64_t axisDim = [input.shape[(NSUInteger)scatterAxis] longLongValue];
            int64_t numIdx = 1;
            for (NSNumber* d in squeezedShape) {
                numIdx *= [d longLongValue];
            }

            // Sets one slice along scatterAxis: flatIdx is a [1] index, upd has
            // the input's dims except scatterAxis. Elementwise ops only.
            auto setSlice = [&](MPSGraphTensor* base, MPSGraphTensor* flatIdx,
                                MPSGraphTensor* upd) -> MPSGraphTensor* {
                MPSGraphTensor* updates = upd;
                MPSGraphTensor* input = base;
                MPSGraphTensor* oneHot = [ctx.graph oneHotWithIndicesTensor:flatIdx
                                                                      depth:(NSUInteger)axisDim
                                                                       axis:1
                                                                   dataType:MPSDataTypeFloat32
                                                                    onValue:1.0
                                                                   offValue:0.0
                                                                       name:nil];
                // oneHot shape: [1, axisDim]
                // Squeeze to [axisDim]
                oneHot = [ctx.graph reshapeTensor:oneHot withShape:@[@(axisDim)] name:nil];

                // Build a boolean mask over the input shape: true at the scatter position(s).
                // Reshape oneHot to be broadcastable: size axisDim on scatterAxis, 1 elsewhere.
                NSMutableArray<NSNumber*>* maskShape = [NSMutableArray arrayWithCapacity:inputRank];
                for (NSUInteger d = 0; d < inputRank; ++d) {
                    [maskShape addObject:((int64_t)d == scatterAxis ? @(axisDim) : @1)];
                }
                MPSGraphTensor* maskFloat = [ctx.graph reshapeTensor:oneHot withShape:maskShape name:nil];
                // mask is 1.0 at scatter position, 0.0 elsewhere.

                // Expand updates to match input shape.
                // updates has all dims of input EXCEPT the scatterAxis dim is missing
                // (it was "inserted" — i.e., the update covers a single slice along scatterAxis).
                // We need to unsqueeze updates along scatterAxis to make it broadcastable.
                NSMutableArray<NSNumber*>* expandedUpdatesShape =
                    [NSMutableArray arrayWithCapacity:inputRank];
                NSUInteger updateDimIdx = 0;
                for (NSUInteger d = 0; d < inputRank; ++d) {
                    if ((int64_t)d == scatterAxis) {
                        [expandedUpdatesShape addObject:@1];
                    } else {
                        if (updateDimIdx < updates.shape.count) {
                            [expandedUpdatesShape addObject:updates.shape[updateDimIdx]];
                            ++updateDimIdx;
                        } else {
                            [expandedUpdatesShape addObject:@1];
                        }
                    }
                }
                MPSGraphTensor* expandedUpdates = [ctx.graph reshapeTensor:updates
                                                                 withShape:expandedUpdatesShape
                                                                      name:nil];
                // expandedUpdates: [1, 4] for our 10x4 case (1 along scatterAxis, 4 along other dims)

                // result = where(mask, updates, input), in the input dtype. An
                // arithmetic float32 blend (input*(1-mask) + updates*mask) lost
                // int32 precision above 2**24, spread NaN from a non-finite update
                // into every other slice, and let a non-finite target survive.
                MPSGraphTensor* zeroF = [ctx.graph constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
                MPSGraphTensor* maskBool = [ctx.graph notEqualWithPrimaryTensor:maskFloat
                                                                secondaryTensor:zeroF
                                                                           name:nil];
                if (expandedUpdates.dataType != input.dataType) {
                    expandedUpdates = [ctx.graph castTensor:expandedUpdates toType:input.dataType name:nil];
                }
                return [ctx.graph selectWithPredicateTensor:maskBool
                                        truePredicateTensor:expandedUpdates
                                       falsePredicateTensor:input
                                                       name:nil];
            };

            // A single index whose updates keep a size-1 batch dim (full input
            // rank) goes through the per-index path, which drops that dim.
            if (numIdx == 1 && updates.shape.count != inputRank) {
                MPSGraphTensor* flatIdx = [ctx.graph reshapeTensor:squeezedIndices withShape:@[@1] name:nil];
                return Result(ctx, setSlice(input, flatIdx, updates), "scatter");
            }
            // Multi-row window set (e.g. LU row swaps, a.at[idx].set(rows)):
            // the single-index lowering reshaped K updates into one slice and
            // aborted on a broadcast-incompatible select. Apply it per index,
            // in order (later indices win; duplicate order is implementation-
            // defined). Bounded K keeps the graph small; larger K falls through.
            if (indicesRank == 2 && updates.shape.count == inputRank && numIdx <= 64) {
                NSInteger batchPos = -1;
                for (NSUInteger d = 0; d < updates.shape.count && batchPos < 0; ++d) {
                    bool isWindow = false;
                    for (auto w : updateWindowDims) {
                        if ((NSUInteger)w == d) {
                            isWindow = true;
                        }
                    }
                    if (!isWindow) {
                        batchPos = (NSInteger)d;
                    }
                }
                if (batchPos >= 0 && [updates.shape[(NSUInteger)batchPos] longLongValue] == numIdx) {
                    NSMutableArray<NSNumber*>* sliceShape = [NSMutableArray array];
                    for (NSUInteger d = 0; d < updates.shape.count; ++d) {
                        if ((NSInteger)d != batchPos) {
                            [sliceShape addObject:updates.shape[d]];
                        }
                    }
                    MPSGraphTensor* acc = input;
                    for (int64_t k = 0; k < numIdx; ++k) {
                        MPSGraphTensor* idxK = [ctx.graph sliceTensor:squeezedIndices
                                                            dimension:0
                                                                start:(NSInteger)k
                                                               length:1
                                                                 name:nil];
                        MPSGraphTensor* updK = [ctx.graph sliceTensor:updates
                                                            dimension:batchPos
                                                                start:(NSInteger)k
                                                               length:1
                                                                 name:nil];
                        updK = [ctx.graph reshapeTensor:updK withShape:sliceShape name:nil];
                        acc = setSlice(acc, idxK, updK);
                    }
                    return Result(ctx, acc, "scatter");
                }
            }
        }

        // When insertedWindowDims removes the scatter axis from updates, unsqueeze it back.
        // E.g., input=[10,4] scatter_axis=0 updates=[4] → need updates=[1,4]
        if (updates.shape.count + insertedWindowDims.size() == input.shape.count) {
            NSMutableArray<NSNumber*>* expandedShape = [NSMutableArray array];
            NSUInteger updateIdx = 0;
            for (NSUInteger d = 0; d < input.shape.count; ++d) {
                bool isInserted = false;
                for (auto iwd : insertedWindowDims) {
                    if ((NSUInteger)iwd == d) {
                        isInserted = true;
                        break;
                    }
                }
                if (isInserted) {
                    [expandedShape addObject:@1];
                } else {
                    if (updateIdx < updates.shape.count) {
                        [expandedShape addObject:updates.shape[updateIdx++]];
                    } else {
                        [expandedShape addObject:@1];
                    }
                }
            }
            updates = [ctx.graph reshapeTensor:updates withShape:expandedShape name:nil];
        }

        MPSGraphTensor* scatterInput = input;
        bool needsWindowUpdate = false;
        NSMutableArray<NSNumber*>* windowStarts = [NSMutableArray arrayWithCapacity:input.shape.count];
        NSMutableArray<NSNumber*>* windowEnds = [NSMutableArray arrayWithCapacity:input.shape.count];
        NSMutableArray<NSNumber*>* windowStrides = [NSMutableArray arrayWithCapacity:input.shape.count];
        if (input.shape.count == updates.shape.count) {
            for (NSUInteger dim = 0; dim < input.shape.count; ++dim) {
                [windowStarts addObject:@0];
                [windowStrides addObject:@1];
                int64_t inDim = [input.shape[dim] longLongValue];
                int64_t upDim = [updates.shape[dim] longLongValue];
                int64_t end = inDim;
                if ((int64_t)dim != scatterAxis) {
                    if (inDim <= 0 || upDim < 0) {
                        return ProcessResult::Error(
                            "scatter: dynamic/unknown shapes not supported for window scatter");
                    }
                    if (upDim > inDim) {
                        return ProcessResult::Error(
                            "scatter: updates window exceeds input dim (" +
                            ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                    }
                    if (upDim < inDim) {
                        end = upDim;
                        needsWindowUpdate = true;
                    }
                }
                [windowEnds addObject:@(end)];
            }
            if (needsWindowUpdate) {
                scatterInput = [ctx.graph sliceTensor:input
                                               starts:windowStarts
                                                 ends:windowEnds
                                              strides:windowStrides
                                                 name:nil];
            }
        }

        MPSGraphTensor* resultWindow = [ctx.graph scatterWithDataTensor:scatterInput
                                                          updatesTensor:updates
                                                          indicesTensor:squeezedIndices
                                                                   axis:static_cast<NSInteger>(scatterAxis)
                                                                   mode:mode
                                                                   name:nil];

        if (!needsWindowUpdate) {
            return Result(ctx, resultWindow, "scatter");
        }
        return Result(ctx, ReinsertWindowByConcat(ctx.graph, input, resultWindow), "scatter");
    }

    if (inputBatchingDims.empty() && scatterIndicesBatchingDims.empty()) {
        NSUInteger operandRank = input.shape.count;
        NSUInteger updatesRank = updates.shape.count;
        // Scalar update into a rank >= 1 operand (x.at[i, j].set(v)): the index
        // vector addresses one element. Build a mask where every coordinate equals
        // its index component (an out-of-bounds index matches nothing, matching
        // StableHLO's skip semantics), apply the reducer there, and select. Only
        // elementwise ops, so it is also safe inside loop bodies.
        if (operandRank > 0 && updatesRank == 0 && updateWindowDims.empty() &&
            insertedWindowDims.size() == operandRank && scatterDimsToOperandDims.size() == operandRank) {
            int64_t idxCount = 1;
            for (NSNumber* d in scatterIndices.shape) {
                idxCount *= [d longLongValue];
            }
            MPSDataType dt = input.dataType;
            const bool isFloat =
                dt == MPSDataTypeFloat32 || dt == MPSDataTypeFloat16 || dt == MPSDataTypeBFloat16;
            const bool isNarrowInt = dt == MPSDataTypeInt8 || dt == MPSDataTypeUInt8 ||
                                     dt == MPSDataTypeInt16 || dt == MPSDataTypeUInt16;
            const bool isU32 = dt == MPSDataTypeUInt32;
            const bool isInt = isNarrowInt || isU32 || dt == MPSDataTypeInt32;
            if (idxCount == (int64_t)operandRank && (isFloat || isInt) &&
                !(isInt && mode == MPSGraphScatterModeDiv)) {
                MPSGraph* g = ctx.graph;
                MPSGraphTensor* flatIdx = [g reshapeTensor:scatterIndices withShape:@[@(idxCount)] name:nil];
                flatIdx = EnsureInt32(g, flatIdx);
                MPSGraphTensor* mask = nil;
                for (NSUInteger k = 0; k < operandRank; ++k) {
                    NSInteger dim = (NSInteger)scatterDimsToOperandDims[k];
                    MPSGraphTensor* coord = [g coordinateAlongAxis:dim withShape:input.shape name:nil];
                    coord = [g castTensor:coord toType:MPSDataTypeInt32 name:nil];
                    MPSGraphTensor* comp = [g sliceTensor:flatIdx
                                                dimension:0
                                                    start:(NSInteger)k
                                                   length:1
                                                     name:nil];
                    MPSGraphTensor* eq = [g equalWithPrimaryTensor:coord secondaryTensor:comp name:nil];
                    mask = mask ? [g logicalANDWithPrimaryTensor:mask secondaryTensor:eq name:nil] : eq;
                }
                MPSGraphTensor* upd =
                    updates.dataType == dt ? updates : [g castTensor:updates toType:dt name:nil];
                // Lift the rank-0 update to rank >= 1: MPSGraph reinterpret_cast (used
                // for uint32 below) aborts the process on rank-0 tensors.
                {
                    NSMutableArray<NSNumber*>* ones = [NSMutableArray arrayWithCapacity:operandRank];
                    for (NSUInteger d = 0; d < operandRank; ++d) {
                        [ones addObject:@1];
                    }
                    upd = [g reshapeTensor:upd withShape:ones name:nil];
                }
                // Integer arithmetic in int32 (8/16-bit MPSGraph arithmetic saturated;
                // uint32 via bit reinterpretation), converted back with wraparound.
                auto toWork = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
                    if (isNarrowInt)
                        return [g castTensor:t toType:MPSDataTypeInt32 name:nil];
                    if (isU32)
                        return [g reinterpretCastTensor:t toType:MPSDataTypeInt32 name:nil];
                    return t;
                };
                auto fromWork = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
                    if (isNarrowInt)
                        return [g castTensor:t toType:dt name:nil];
                    if (isU32)
                        return [g reinterpretCastTensor:t toType:dt name:nil];
                    return t;
                };
                MPSGraphTensor* combined = nil;
                if (mode == MPSGraphScatterModeSet) {
                    combined = upd;
                } else if (mode == MPSGraphScatterModeMax || mode == MPSGraphScatterModeMin) {
                    const bool isMax = mode == MPSGraphScatterModeMax;
                    if (isFloat) {
                        combined = isMax ? [g maximumWithNaNPropagationWithPrimaryTensor:input
                                                                         secondaryTensor:upd
                                                                                    name:nil]
                                         : [g minimumWithNaNPropagationWithPrimaryTensor:input
                                                                         secondaryTensor:upd
                                                                                    name:nil];
                    } else if (isU32) {
                        // Sign-bit XOR maps uint32 order onto int32 order.
                        MPSGraphTensor* sb = [g constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
                        MPSGraphTensor* a = [g bitwiseXORWithPrimaryTensor:toWork(input)
                                                           secondaryTensor:sb
                                                                      name:nil];
                        MPSGraphTensor* b = [g bitwiseXORWithPrimaryTensor:toWork(upd)
                                                           secondaryTensor:sb
                                                                      name:nil];
                        MPSGraphTensor* r = isMax ? [g maximumWithPrimaryTensor:a secondaryTensor:b name:nil]
                                                  : [g minimumWithPrimaryTensor:a secondaryTensor:b name:nil];
                        combined = fromWork([g bitwiseXORWithPrimaryTensor:r secondaryTensor:sb name:nil]);
                    } else {
                        combined = isMax ? [g maximumWithPrimaryTensor:input secondaryTensor:upd name:nil]
                                         : [g minimumWithPrimaryTensor:input secondaryTensor:upd name:nil];
                    }
                } else {
                    MPSGraphTensor* a = isInt ? toWork(input) : input;
                    MPSGraphTensor* b = isInt ? toWork(upd) : upd;
                    MPSGraphTensor* r = nil;
                    if (mode == MPSGraphScatterModeAdd)
                        r = [g additionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeSub)
                        r = [g subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeMul)
                        r = [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
                    else if (mode == MPSGraphScatterModeDiv)
                        r = [g divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
                    combined = (r && isInt) ? fromWork(r) : r;
                }
                if (combined && mask) {
                    MPSGraphTensor* result = [g selectWithPredicateTensor:mask
                                                      truePredicateTensor:combined
                                                     falsePredicateTensor:input
                                                                     name:nil];
                    return Result(ctx, result, "scatter");
                }
            }
        }
        if (operandRank == 0 || updatesRank == 0) {
            return ProcessResult::Error("scatter: scalar scatter is not supported yet (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }

        MPSGraphTensor* indicesTensor = scatterIndices;
        NSArray<NSNumber*>* indicesTensorShape = indicesTensor.shape;
        NSUInteger indicesTensorRank = indicesTensorShape.count;
        if (indicesTensorRank == 0) {
            return ProcessResult::Error("scatter: invalid scatter_indices rank (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        if (indexVectorDim < 0 || indexVectorDim >= (int64_t)indicesTensorRank) {
            return ProcessResult::Error("scatter: invalid index_vector_dim (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        if (indexVectorDim != (int64_t)indicesTensorRank - 1) {
            NSMutableArray<NSNumber*>* perm = [NSMutableArray arrayWithCapacity:indicesTensorRank];
            for (NSUInteger d = 0; d < indicesTensorRank; ++d) {
                if ((int64_t)d == indexVectorDim) {
                    continue;
                }
                [perm addObject:@((NSInteger)d)];
            }
            [perm addObject:@((NSInteger)indexVectorDim)];
            if (perm.count != indicesTensorRank) {
                return ProcessResult::Error("scatter: invalid scatter_indices permutation (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            indicesTensor = [ctx.graph transposeTensor:indicesTensor permutation:perm name:nil];
            indicesTensorShape = indicesTensor.shape;
            indicesTensorRank = indicesTensorShape.count;
        }
        indicesTensor = EnsureInt32(ctx.graph, indicesTensor);

        std::vector<int64_t> operandDimsWithoutInserted;
        operandDimsWithoutInserted.reserve(operandRank);
        for (NSUInteger d = 0; d < operandRank; ++d) {
            if (ContainsDim(insertedWindowDims, (int64_t)d)) {
                continue;
            }
            operandDimsWithoutInserted.push_back((int64_t)d);
        }
        if (operandDimsWithoutInserted.size() != updateWindowDims.size()) {
            return ProcessResult::Error("scatter: unsupported update_window_dims mapping (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        std::vector<int64_t> operandToUpdateDim(operandRank, -1);
        for (size_t i = 0; i < operandDimsWithoutInserted.size(); ++i) {
            int64_t updateDim = updateWindowDims[i];
            if (updateDim < 0 || updateDim >= (int64_t)updatesRank) {
                return ProcessResult::Error("scatter: invalid update_window_dims (" +
                                            ScatterDebugString(scatterOp, input, scatterIndices, updates) +
                                            ")");
            }
            operandToUpdateDim[(size_t)operandDimsWithoutInserted[i]] = updateDim;
        }

        // scatterND drops out-of-bounds elements one at a time, so a window that
        // only partly fits (mode="drop", start near the edge) was partially
        // written, while XLA skips the whole window. When a scattered dim has a
        // window wider than 1, lower with explicit coordinates, which send every
        // non-fitting window to a dummy slot.
        bool partialWindowPossible = false;
        for (int64_t d : scatterDimsToOperandDims) {
            if (d >= 0 && d < (int64_t)operandRank && operandToUpdateDim[(size_t)d] >= 0 &&
                [updates.shape[(NSUInteger)operandToUpdateDim[(size_t)d]] longLongValue] > 1) {
                partialWindowPossible = true;
            }
        }
        if (partialWindowPossible) {
            ProcessResult explicitWindow = ScatterExplicitCoordinates(ctx, scatterOp, mode);
            if (explicitWindow.ok()) {
                return explicitWindow;
            }
        }

        std::vector<int64_t> updateScatterDims;
        updateScatterDims.reserve(updatesRank);
        for (NSUInteger d = 0; d < updatesRank; ++d) {
            if (!ContainsDim(updateWindowDims, (int64_t)d)) {
                updateScatterDims.push_back((int64_t)d);
            }
        }

        if (indicesTensorRank < 1) {
            return ProcessResult::Error("scatter: invalid scatter_indices rank (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        int64_t indexVectorLen = [indicesTensor.shape[indicesTensorRank - 1] longLongValue];
        if (indexVectorLen <= 0) {
            return ProcessResult::Error("scatter: invalid index vector length (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }
        NSUInteger scatterPrefixRank = indicesTensorRank - 1;
        if (scatterPrefixRank != updateScatterDims.size()) {
            return ProcessResult::Error("scatter: unsupported scatter_indices/update dims (" +
                                        ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
        }

        NSMutableArray<NSNumber*>* reshapedIndicesShape = [NSMutableArray arrayWithCapacity:updatesRank + 1];

        for (NSUInteger d = 0; d < updatesRank; ++d) {
            NSInteger size = 1;
            for (size_t i = 0; i < updateScatterDims.size(); ++i) {
                if (updateScatterDims[i] == (int64_t)d) {
                    size = [indicesTensor.shape[i] integerValue];
                    break;
                }
            }
            [reshapedIndicesShape addObject:@(size)];
        }
        [reshapedIndicesShape addObject:@(indexVectorLen)];

        MPSGraphTensor* expandedIndices = [ctx.graph reshapeTensor:indicesTensor
                                                         withShape:reshapedIndicesShape
                                                              name:nil];
        NSMutableArray<NSNumber*>* broadcastIndicesShape = [NSMutableArray arrayWithCapacity:updatesRank + 1];
        for (NSUInteger d = 0; d < updatesRank; ++d) {
            [broadcastIndicesShape addObject:updates.shape[d]];
        }
        [broadcastIndicesShape addObject:@(indexVectorLen)];
        MPSGraphTensor* broadcastIndices = [ctx.graph broadcastTensor:expandedIndices
                                                              toShape:broadcastIndicesShape
                                                                 name:nil];

        std::vector<MPSGraphTensor*> indexComponents;
        indexComponents.reserve((size_t)indexVectorLen);
        for (int64_t i = 0; i < indexVectorLen; ++i) {
            NSMutableArray<NSNumber*>* starts = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            NSMutableArray<NSNumber*>* ends = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            NSMutableArray<NSNumber*>* strides = [NSMutableArray arrayWithCapacity:updatesRank + 1];
            for (NSUInteger d = 0; d < updatesRank; ++d) {
                [starts addObject:@0];
                [ends addObject:updates.shape[d]];
                [strides addObject:@1];
            }
            [starts addObject:@(i)];
            [ends addObject:@(i + 1)];
            [strides addObject:@1];
            MPSGraphTensor* slice = [ctx.graph sliceTensor:broadcastIndices
                                                    starts:starts
                                                      ends:ends
                                                   strides:strides
                                                      name:nil];

            slice = [ctx.graph reshapeTensor:slice withShape:updates.shape name:nil];
            indexComponents.push_back(slice);
        }

        MPSGraphTensor* zeroScalar = [ctx.graph constantWithScalar:0 dataType:MPSDataTypeInt32];
        MPSGraphTensor* zero = [ctx.graph broadcastTensor:zeroScalar toShape:updates.shape name:nil];

        NSMutableArray<MPSGraphTensor*>* coordTensors = [NSMutableArray arrayWithCapacity:operandRank];
        for (NSUInteger operandDim = 0; operandDim < operandRank; ++operandDim) {
            MPSGraphTensor* coord = zero;
            int64_t updateDim = operandToUpdateDim[operandDim];
            if (updateDim >= 0) {
                coord = [ctx.graph coordinateAlongAxis:(NSInteger)updateDim withShape:updates.shape name:nil];
                coord = EnsureInt32(ctx.graph, coord);
            }

            for (size_t i = 0; i < (size_t)scatterDimsToOperandDims.size(); ++i) {
                if (scatterDimsToOperandDims[i] == (int64_t)operandDim) {
                    if (i >= indexComponents.size()) {
                        return ProcessResult::Error(
                            "scatter: scatter_dims_to_operand_dims out of range (" +
                            ScatterDebugString(scatterOp, input, scatterIndices, updates) + ")");
                    }
                    coord = [ctx.graph additionWithPrimaryTensor:coord
                                                 secondaryTensor:indexComponents[i]
                                                            name:nil];
                    break;
                }
            }
            [coordTensors addObject:coord];
        }

        MPSGraphTensor* ndIndices = [ctx.graph stackTensors:coordTensors
                                                       axis:(NSInteger)updatesRank
                                                       name:nil];
        ndIndices = EnsureInt32(ctx.graph, ndIndices);

        MPSGraphTensor* result = [ctx.graph scatterNDWithDataTensor:input
                                                      updatesTensor:updates
                                                      indicesTensor:ndIndices
                                                    batchDimensions:0
                                                               mode:mode
                                                               name:nil];
        return Result(ctx, result, "scatter");
    }

    ProcessResult explicitScatter = ScatterExplicitCoordinates(ctx, scatterOp, mode);
    if (explicitScatter.ok()) {
        return explicitScatter;
    }
    return ProcessResult::Error("scatter: unsupported scatter pattern (" +
                                ScatterDebugString(scatterOp, input, scatterIndices, updates) + "; " +
                                explicitScatter.error + ")");
}
// MPSGraph scatter moved uint32 values inexactly (off by 1 above 2**24, and
// add saturated to INT32_MAX above 2**31) while int32 scatter was exact. For
// set and add (mod 2**32), scatter the int32 reinterpretation and reinterpret
// the result back; for min/max, additionally flip the sign bit so int32
// ordering matches uint32 ordering.
static ProcessResult HandleScatter(HandlerContext& ctx) {
    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(ctx.op);
    if (!scatterOp || ctx.op->getNumOperands() != 3 || ctx.op->getNumResults() != 1) {
        return HandleScatterImpl(ctx);
    }
    std::string bodyOps;
    if (!IsSupportedScatterBody(scatterOp, &bodyOps)) {
        return ProcessResult::Error("scatter: unsupported update computation (" + bodyOps +
                                    "); only set or a single add/sub/mul/div/max/min of (current, update) "
                                    "is supported (x.at[i].apply(f) is not)");
    }
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* updates = GetInputTensor(ctx, 2);
    MPSGraphScatterMode mode = GetScatterMode(scatterOp);

    // bool scatter returned wrong values (set) or was rejected (scalar
    // updates). Scatter the exact 0/1 int32 widening and map back with != 0,
    // which also yields bool semantics per reducer: add->OR, mul->AND,
    // max->OR, min->AND.
    if (input && updates && input.dataType == MPSDataTypeBool && updates.dataType == MPSDataTypeBool &&
        ctx.op->getOperand(0) != ctx.op->getOperand(2) && ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        ctx.op->getOperand(1) != ctx.op->getOperand(2)) {
        void* inKey = ctx.op->getOperand(0).getAsOpaquePointer();
        void* upKey = ctx.op->getOperand(2).getAsOpaquePointer();
        if (!(ctx.variables && (ctx.variables->find(inKey) != ctx.variables->end() ||
                                ctx.variables->find(upKey) != ctx.variables->end()))) {
            auto savedIn = ctx.values.find(inKey);
            MPSGraphTensor* origIn = savedIn != ctx.values.end() ? savedIn->second : nil;
            auto savedUp = ctx.values.find(upKey);
            MPSGraphTensor* origUp = savedUp != ctx.values.end() ? savedUp->second : nil;
            ctx.values[inKey] = [ctx.graph castTensor:input toType:MPSDataTypeInt32 name:nil];
            ctx.values[upKey] = [ctx.graph castTensor:updates toType:MPSDataTypeInt32 name:nil];
            ProcessResult r = HandleScatterImpl(ctx);
            auto restoreKey = [&](void* key, MPSGraphTensor* orig) {
                if (orig) {
                    ctx.values[key] = orig;
                } else {
                    ctx.values.erase(key);
                }
            };
            restoreKey(inKey, origIn);
            restoreKey(upKey, origUp);
            if (!r.ok()) {
                return r;
            }
            MPSGraphTensor* scattered = GetTensor(ctx.values, ctx.op->getResult(0));
            if (!scattered) {
                return ProcessResult::Error("scatter: bool widening lowering failed");
            }
            MPSGraphTensor* zero = [ctx.graph constantWithScalar:0 dataType:scattered.dataType];
            MPSGraphTensor* result = [ctx.graph notEqualWithPrimaryTensor:scattered
                                                          secondaryTensor:zero
                                                                     name:nil];
            return Result(ctx, result, "scatter");
        }
    }

    // mps.scatter rejects complex operands with a process-aborting assertion.
    // set and add act independently on real and imaginary parts, so scatter
    // each part through the float path and recombine (exact).
    auto isComplex = [](MPSGraphTensor* t) {
        return t && (t.dataType == MPSDataTypeComplexFloat32 || t.dataType == MPSDataTypeComplexFloat16);
    };
    if (isComplex(input) && isComplex(updates) &&
        (mode == MPSGraphScatterModeSet || mode == MPSGraphScatterModeAdd) &&
        ctx.op->getOperand(0) != ctx.op->getOperand(2) && ctx.op->getOperand(0) != ctx.op->getOperand(1) &&
        ctx.op->getOperand(1) != ctx.op->getOperand(2)) {
        void* inKey = ctx.op->getOperand(0).getAsOpaquePointer();
        void* upKey = ctx.op->getOperand(2).getAsOpaquePointer();
        if (!(ctx.variables && (ctx.variables->find(inKey) != ctx.variables->end() ||
                                ctx.variables->find(upKey) != ctx.variables->end()))) {
            auto savedIn = ctx.values.find(inKey);
            MPSGraphTensor* origIn = savedIn != ctx.values.end() ? savedIn->second : nil;
            auto savedUp = ctx.values.find(upKey);
            MPSGraphTensor* origUp = savedUp != ctx.values.end() ? savedUp->second : nil;
            auto scatterPart = [&](bool real, MPSGraphTensor** out) -> ProcessResult {
                ctx.values[inKey] = real ? [ctx.graph realPartOfTensor:input name:nil]
                                         : [ctx.graph imaginaryPartOfTensor:input name:nil];
                ctx.values[upKey] = real ? [ctx.graph realPartOfTensor:updates name:nil]
                                         : [ctx.graph imaginaryPartOfTensor:updates name:nil];
                ProcessResult r = HandleScatterImpl(ctx);
                if (r.ok()) {
                    *out = GetTensor(ctx.values, ctx.op->getResult(0));
                }
                return r;
            };
            MPSGraphTensor* sRe = nil;
            MPSGraphTensor* sIm = nil;
            ProcessResult r = scatterPart(true, &sRe);
            if (r.ok()) {
                r = scatterPart(false, &sIm);
            }
            auto restoreKey = [&](void* key, MPSGraphTensor* orig) {
                if (orig) {
                    ctx.values[key] = orig;
                } else {
                    ctx.values.erase(key);
                }
            };
            restoreKey(inKey, origIn);
            restoreKey(upKey, origUp);
            if (!r.ok()) {
                return r;
            }
            if (!sRe || !sIm) {
                return ProcessResult::Error("scatter: complex split lowering failed");
            }
            return Result(ctx, [ctx.graph complexTensorWithRealTensor:sRe imaginaryTensor:sIm name:nil],
                          "scatter");
        }
    }

    // Unsigned scatters run in int32. MPSGraph scatter saturated unsigned add/mul
    // (1-D x.at[i].add: 5 + 0xFFFFFFFF gave INT32_MAX, 4e9 + 3 stayed 4e9, uint16
    // 5 + 65535 gave 65535) and compared uint32 as signed (max(4e9, 3) gave 3).
    // uint32 is reinterpreted (min/max XOR the sign bit to map uint32 order onto
    // int32 order); uint8/uint16 are widened and masked back. MPSGraph
    // reinterpret_cast aborts on rank-0 tensors, so scalars go through [1].
    const bool orderMode = mode == MPSGraphScatterModeMin || mode == MPSGraphScatterModeMax;
    const MPSDataType udt = input ? input.dataType : MPSDataTypeInvalid;
    const bool isU32 = udt == MPSDataTypeUInt32;
    const bool isNarrowU = udt == MPSDataTypeUInt16 || udt == MPSDataTypeUInt8;
    const bool arithMode = mode == MPSGraphScatterModeAdd || mode == MPSGraphScatterModeMul || orderMode;
    if (!input || !updates || input.shape.count == 0 || !(isU32 || isNarrowU) || updates.dataType != udt ||
        !(arithMode || (isU32 && mode == MPSGraphScatterModeSet && updates.shape.count > 0)) ||
        ctx.op->getOperand(0) == ctx.op->getOperand(2) || ctx.op->getOperand(0) == ctx.op->getOperand(1) ||
        ctx.op->getOperand(1) == ctx.op->getOperand(2)) {
        return HandleScatterImpl(ctx);
    }
    void* inputKey = ctx.op->getOperand(0).getAsOpaquePointer();
    void* updatesKey = ctx.op->getOperand(2).getAsOpaquePointer();
    if (ctx.variables && (ctx.variables->find(inputKey) != ctx.variables->end() ||
                          ctx.variables->find(updatesKey) != ctx.variables->end())) {
        return HandleScatterImpl(ctx);
    }

    auto savedInput = ctx.values.find(inputKey);
    MPSGraphTensor* origInput = savedInput != ctx.values.end() ? savedInput->second : nil;
    auto savedUpdates = ctx.values.find(updatesKey);
    MPSGraphTensor* origUpdates = savedUpdates != ctx.values.end() ? savedUpdates->second : nil;
    auto restore = [&](void* key, MPSGraphTensor* orig) {
        if (orig) {
            ctx.values[key] = orig;
        } else {
            ctx.values.erase(key);
        }
    };

    MPSGraphTensor* signBit = [ctx.graph constantWithScalar:-2147483648.0 dataType:MPSDataTypeInt32];
    auto toInt32 = [&](MPSGraphTensor* t) -> MPSGraphTensor* {
        if (isNarrowU) {
            return [ctx.graph castTensor:t toType:MPSDataTypeInt32 name:nil];
        }
        const bool scalar = t.shape && t.shape.count == 0;
        MPSGraphTensor* v = scalar ? [ctx.graph reshapeTensor:t withShape:@[@1] name:nil] : t;
        MPSGraphTensor* bits = [ctx.graph reinterpretCastTensor:v toType:MPSDataTypeInt32 name:nil];
        if (orderMode) {
            bits = [ctx.graph bitwiseXORWithPrimaryTensor:bits secondaryTensor:signBit name:nil];
        }
        return scalar ? [ctx.graph reshapeTensor:bits withShape:@[] name:nil] : bits;
    };
    ctx.values[inputKey] = toInt32(input);
    ctx.values[updatesKey] = toInt32(updates);
    ProcessResult r = HandleScatterImpl(ctx);
    restore(inputKey, origInput);
    restore(updatesKey, origUpdates);
    if (!r.ok()) {
        return r;
    }
    MPSGraphTensor* scattered = GetTensor(ctx.values, ctx.op->getResult(0));
    if (!scattered || scattered.dataType != MPSDataTypeInt32) {
        ctx.values.erase(ctx.op->getResult(0).getAsOpaquePointer());
        return HandleScatterImpl(ctx);
    }
    if (isNarrowU) {
        const double mask = udt == MPSDataTypeUInt16 ? 65535.0 : 255.0;
        MPSGraphTensor* low = [ctx.graph
            bitwiseANDWithPrimaryTensor:scattered
                        secondaryTensor:[ctx.graph constantWithScalar:mask dataType:MPSDataTypeInt32]
                                   name:nil];
        return Result(ctx, [ctx.graph castTensor:low toType:udt name:nil], "scatter");
    }
    if (orderMode) {
        scattered = [ctx.graph bitwiseXORWithPrimaryTensor:scattered secondaryTensor:signBit name:nil];
    }
    MPSGraphTensor* result = [ctx.graph reinterpretCastTensor:scattered toType:MPSDataTypeUInt32 name:nil];
    return Result(ctx, result, "scatter");
}
REGISTER_MPS_OP("stablehlo.scatter", HandleScatter);

static ProcessResult HandleReverse(HandlerContext& ctx) {
    auto reverseOp = mlir::dyn_cast<mlir::stablehlo::ReverseOp>(ctx.op);
    if (!reverseOp) {
        return ProcessResult::Error("reverse: expected ReverseOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("reverse: missing input tensor");

    auto dimensions = reverseOp.getDimensions();
    NSMutableArray<NSNumber*>* axes = [NSMutableArray array];
    for (int64_t dim : dimensions) {
        [axes addObject:@(dim)];
    }

    // reverseTensor on 8/16-bit tensors segfaulted (1-D float16/int8/uint8,
    // eager), returned wrong values (1-D int16, jit), and float16
    // reverse(sort(x)) under JIT came back unreversed. 32-bit reverses were
    // correct, so widen exactly (f16 -> f32, small ints -> int32), reverse,
    // and cast back.
    MPSDataType dtype = input.dataType;
    // The f16 -> f32 -> f16 cast round trip rewrote every NaN to 0x7e00 (-NaN
    // 0xfe00 and payload NaNs included; XLA:CPU moves raw bits), which broke
    // jnp.sort(descending=True) and jnp.flip on f16. Reverse the int16 view
    // through the exact int32 route instead.
    NSArray<NSNumber*>* inShape = input.shape;
    if (dtype == MPSDataTypeFloat16 && inShape && inShape.count > 0) {
        bool staticNonEmpty = true;
        for (NSNumber* d : inShape) {
            staticNonEmpty = staticNonEmpty && d.longLongValue > 0;
        }
        if (staticNonEmpty) {
            MPSGraphTensor* bits = [ctx.graph reinterpretCastTensor:input toType:MPSDataTypeInt16 name:nil];
            MPSGraphTensor* wide = [ctx.graph castTensor:bits toType:MPSDataTypeInt32 name:nil];
            MPSGraphTensor* reversed = [ctx.graph reverseTensor:wide axes:axes name:nil];
            MPSGraphTensor* narrow = [ctx.graph castTensor:reversed toType:MPSDataTypeInt16 name:nil];
            return Result(ctx, [ctx.graph reinterpretCastTensor:narrow toType:dtype name:nil], "reverse");
        }
    }
    MPSDataType wideType = MPSDataTypeInvalid;
    if (dtype == MPSDataTypeFloat16) {
        wideType = MPSDataTypeFloat32;
    } else if (dtype == MPSDataTypeInt8 || dtype == MPSDataTypeUInt8 || dtype == MPSDataTypeInt16 ||
               dtype == MPSDataTypeUInt16) {
        wideType = MPSDataTypeInt32;
    }
    if (wideType != MPSDataTypeInvalid) {
        MPSGraphTensor* wide = [ctx.graph castTensor:input toType:wideType name:nil];
        MPSGraphTensor* reversed = [ctx.graph reverseTensor:wide axes:axes name:nil];
        return Result(ctx, [ctx.graph castTensor:reversed toType:dtype name:nil], "reverse");
    }

    MPSGraphTensor* result = [ctx.graph reverseTensor:input axes:axes name:nil];
    return Result(ctx, result, "reverse");
}
REGISTER_MPS_OP("stablehlo.reverse", HandleReverse);

} // namespace jax_metallib
