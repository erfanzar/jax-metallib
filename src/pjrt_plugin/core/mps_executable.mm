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

#import "pjrt_plugin/core/mps_executable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#import "pjrt_plugin/core/issue_url.h"
#import "pjrt_plugin/core/mps_buffer.h"
#import "pjrt_plugin/core/mps_client.h"
#import "pjrt_plugin/core/mps_device.h"
#import "pjrt_plugin/core/stablehlo_parser.h"
#import "pjrt_plugin/ops/registry.h"
#import "pjrt_plugin/runtime/metal_kernels.h"
#include "stablehlo/dialect/Serialization.h"
#include "stablehlo/dialect/Version.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace jax_metallib {

using SlotId = int;

struct SlotInfo {
    void* shape;
    MPSDataType dtype;
    size_t byte_size;
};

struct IntermediateCacheSet {
    std::vector<id<MTLBuffer>> bufs;
    std::vector<MPSGraphTensorData*> tensor_datas;
    std::vector<id<MTLBuffer>> tensor_data_bufs;

    explicit IntermediateCacheSet(size_t n) : bufs(n, nil), tensor_datas(n, nil), tensor_data_bufs(n, nil) {}

    ~IntermediateCacheSet() {
        for (MPSGraphTensorData* td : tensor_datas) {
            if (td) {
                [td release];
            }
        }
        for (id<MTLBuffer> buf : tensor_data_bufs) {
            if (buf) {
                CFRelease((__bridge CFTypeRef)buf);
            }
        }
        for (id<MTLBuffer> buf : bufs) {
            if (buf) {
                CFRelease((__bridge CFTypeRef)buf);
            }
        }
    }
};

struct GraphStep {
    void* graph;
    std::vector<std::pair<SlotId, void*>> feeds;
    std::vector<std::pair<SlotId, void*>> targets;

    void* executable = nullptr;
    std::vector<SlotId> exec_feed_slots;
    std::vector<SlotId> exec_target_slots;
};

struct FusedElementwiseInfo {
    std::vector<SlotId> external_input_slots;
    std::vector<SlotId> eliminated_slots;
    uint32_t num_buffer_args = 0;
};

struct NativeStep {
    enum class BakedKind : uint8_t {
        kNone,
        kElementwise,
        kFusedElementwise,
        kReduce,
        kIota,
        kPad2D,
        kScatterAddRows,
        kReshapeAlias,
        kConstant,
    };

    NativeOpHandler handler;
    mlir::Operation* op;
    std::vector<SlotId> input_slots;
    SlotId output_slot;

    BakedKind baked_kind = BakedKind::kNone;
    id<MTLComputePipelineState> baked_pipeline = nil;
    MTLSize baked_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_tg = MTLSizeMake(1, 1, 1);
    bool baked_use_vec4 = false;
    bool baked_reduce_threadgroup = false;

    uint32_t baked_n = 0;
    uint32_t baked_rows = 0;
    uint32_t baked_cols = 0;

    uint32_t baked_in_h = 0;
    uint32_t baked_in_w = 0;
    uint32_t baked_out_w = 0;
    uint32_t baked_low_h = 0;
    uint32_t baked_low_w = 0;
    float baked_pad_value = 0.0f;
    SlotId baked_pad_value_slot = -1;

    bool baked_pad_use_fill_copy = false;
    id<MTLComputePipelineState> baked_fill_pipeline = nil;
    id<MTLComputePipelineState> baked_copy2d_pipeline = nil;
    uint32_t baked_fill_n4 = 0;
    uint32_t baked_src_total = 0;
    uint32_t baked_src_cols = 0;
    uint32_t baked_dst_cols = 0;
    MTLSize baked_fill_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_fill_tg = MTLSizeMake(1, 1, 1);
    MTLSize baked_copy_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_copy_tg = MTLSizeMake(1, 1, 1);

    uint32_t baked_scatter_k = 0;
    uint32_t baked_scatter_cols = 0;
    uint32_t baked_scatter_out_rows = 0;
    uint32_t baked_scatter_total = 0;

    id<MTLBuffer> baked_constant_buffer = nil;

    std::unique_ptr<FusedElementwiseInfo> fused_info;
};

struct Step {
    enum Kind { GRAPH, NATIVE } kind;
    size_t index;
};

struct ExecutionPlan {
    std::vector<Step> steps;
    std::vector<GraphStep> graph_steps;
    std::vector<NativeStep> native_steps;
    std::vector<SlotId> input_slots;
    std::vector<SlotId> output_slots;
    std::vector<mlir::Type> return_types;
    std::vector<SlotInfo> slots;

    std::vector<uint8_t> slot_is_input;
    std::vector<uint8_t> slot_is_output;
    std::vector<uint8_t> slot_is_eliminated;

    struct OutputMeta {
        int pjrt_dtype;
        std::vector<int64_t> dims;
        size_t byte_size;
    };
    std::vector<OutputMeta> output_meta;

    bool needs_retained_cmd_bufs = false;

    struct ExecScratch {
        std::vector<id<MTLBuffer>> slot_bufs;
        std::vector<CompletionEvent> slot_producer_events;
        void reset(size_t n) {
            slot_bufs.assign(n, nil);
            slot_producer_events.assign(n, nullptr);
        }
    };
    std::mutex scratch_pool_mutex;
    std::vector<std::unique_ptr<ExecScratch>> scratch_pool;

    std::mutex intermediate_pool_mutex;
    std::vector<std::unique_ptr<IntermediateCacheSet>> intermediate_pool;
};

static size_t ByteSizeFromType(mlir::Type type);

namespace {

static bool EnvFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }

    return !(v[0] == '0' && v[1] == '\0');
}

static int EnvIntValue(const char* name, int default_value) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return default_value;
    }
    char* end = nullptr;
    long parsed = std::strtol(v, &end, 10);
    if (!end || *end != '\0' || parsed <= 0 || parsed > std::numeric_limits<int>::max()) {
        return default_value;
    }
    return static_cast<int>(parsed);
}

static id<MTLBuffer> AcquireReusableOutputBuffer(id<MTLDevice> device, size_t byte_size);
static id<MTLBuffer> NativeIntrinsicReshapeAliasHandler(id<MTLDevice> device,
                                                        id<MTLCommandBuffer> commandBuffer,
                                                        mlir::Operation* op,
                                                        const std::vector<id<MTLBuffer>>& inputs);
static id<MTLBuffer> NativeIntrinsicConstantHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                    mlir::Operation* op,
                                                    const std::vector<id<MTLBuffer>>& inputs);
static id<MTLBuffer> NativeIntrinsicIotaHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                mlir::Operation* op,
                                                const std::vector<id<MTLBuffer>>& inputs);
static id<MTLBuffer> NativeIntrinsicPadHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                               mlir::Operation* op, const std::vector<id<MTLBuffer>>& inputs);
static id<MTLBuffer> NativeIntrinsicScatterAddRowsHandler(id<MTLDevice> device,
                                                          id<MTLCommandBuffer> commandBuffer,
                                                          mlir::Operation* op,
                                                          const std::vector<id<MTLBuffer>>& inputs);

enum class IntrinsicNativeKind {
    kNone,
    kReshapeAlias,
    kConstantScalar,
    kIota1D,
    kPad2DNoInterior,
    kScatterAddRowsF32,
    kNegate,
    kFloor,
    kRoundNearestEven,
    kRsqrt,
    kSign,
    kSquare,
    kAbs,
    kExp,
    kLog,
    kTanh,
    kCeil,
    kSin,
    kCos,
    kSqrt,
    kLogistic,
    kAdd,
    kSubtract,
    kMultiply,
    kDivide,
    kMaximum,
    kMinimum,
    kSelect,
    kReduceSumLastAxis,
    kReduceMaxLastAxis,
    kReduceMinLastAxis,
    kConvertFp16Fp32,
};

static bool IsFloat16Or32(MPSDataType dtype) {
    return dtype == MPSDataTypeFloat16 || dtype == MPSDataTypeFloat32;
}

static bool IsInt32OrFloat32(MPSDataType dtype) {
    return dtype == MPSDataTypeInt32 || dtype == MPSDataTypeFloat32;
}

static MTLResourceOptions ComputeBufferResourceOptions() {
    static MTLResourceOptions options = [] {
        MTLResourceOptions opt = EnvFlagEnabled("JAX_METALLIB_USE_PRIVATE_COMPUTE_BUFFERS")
                                     ? MTLResourceStorageModePrivate
                                     : MTLResourceStorageModeShared;

        if (!EnvFlagEnabled("JAX_METALLIB_ENABLE_HAZARD_TRACKING")) {
            opt |= MTLResourceHazardTrackingModeUntracked;
        }
        return opt;
    }();
    return options;
}

static bool GetStaticTensorInfo(mlir::Type type, std::vector<int64_t>* shape_out, uint64_t* numel_out,
                                MPSDataType* dtype_out) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType) {
        return false;
    }
    auto shapeRef = tensorType.getShape();
    std::vector<int64_t> shape(shapeRef.begin(), shapeRef.end());
    uint64_t numel = 1;
    for (int64_t d : shape) {
        if (d < 0) {
            return false;
        }
        if (d == 0) {
            numel = 0;
            continue;
        }
        if (numel > std::numeric_limits<uint64_t>::max() / static_cast<uint64_t>(d)) {
            return false;
        }
        numel *= static_cast<uint64_t>(d);
    }
    if (shape_out) {
        *shape_out = std::move(shape);
    }
    if (numel_out) {
        *numel_out = numel;
    }
    if (dtype_out) {
        *dtype_out = MlirTypeToMps(tensorType.getElementType());
    }
    return true;
}

static bool GetScalarFloatConstant(mlir::Value value, double* out) {
    if (!value || !out) {
        return false;
    }
    auto* defOp = value.getDefiningOp();
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(defOp);
    if (!constOp) {
        return false;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!dense || dense.getNumElements() != 1) {
        return false;
    }
    if (!mlir::isa<mlir::FloatType>(dense.getElementType())) {
        return false;
    }
    llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
    *out = ap.convertToDouble();
    return true;
}

static IntrinsicNativeKind ClassifySimpleKind(mlir::Operation* op) {
    if (!op) {
        return IntrinsicNativeKind::kNone;
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "stablehlo.reshape") {
        return IntrinsicNativeKind::kReshapeAlias;
    }
    if (name == "stablehlo.constant") {
        return IntrinsicNativeKind::kConstantScalar;
    }
    if (name == "stablehlo.iota") {
        return IntrinsicNativeKind::kIota1D;
    }
    if (name == "stablehlo.pad") {
        return IntrinsicNativeKind::kPad2DNoInterior;
    }
    if (name == "stablehlo.scatter") {
        return IntrinsicNativeKind::kScatterAddRowsF32;
    }
    return IntrinsicNativeKind::kNone;
}

static bool IsEligibleReshapeAliasOp(mlir::Operation* op) {
    if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return false;
    }
    uint64_t inNumel = 0;
    uint64_t outNumel = 0;
    MPSDataType inDtype = MPSDataTypeInvalid;
    MPSDataType outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &inNumel, &inDtype) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, &outDtype)) {
        return false;
    }
    return inNumel == outNumel && inDtype == outDtype && inNumel > 0;
}

static bool IsEligibleConstantScalarOp(mlir::Operation* op) {
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op);
    if (!constOp || op->getNumOperands() != 0 || op->getNumResults() != 1) {
        return false;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!dense || dense.getNumElements() != 1) {
        return false;
    }
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, nullptr, &dtype)) {
        return false;
    }
    return dtype == MPSDataTypeFloat16 || dtype == MPSDataTypeFloat32 || dtype == MPSDataTypeInt32 ||
           dtype == MPSDataTypeBool;
}

static bool IsEligibleIotaNativeOp(mlir::Operation* op) {
    auto iotaOp = mlir::dyn_cast_or_null<mlir::stablehlo::IotaOp>(op);
    if (!iotaOp || op->getNumOperands() != 0 || op->getNumResults() != 1) {
        return false;
    }
    std::vector<int64_t> shape;
    uint64_t numel = 0;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), &shape, &numel, &dtype)) {
        return false;
    }
    if (shape.size() != 1 || numel == 0 || numel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (iotaOp.getIotaDimension() != 0) {
        return false;
    }
    return IsInt32OrFloat32(dtype);
}

static bool IsEligiblePadNativeOp(mlir::Operation* op) {
    auto padOp = mlir::dyn_cast_or_null<mlir::stablehlo::PadOp>(op);
    if (!padOp || op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return false;
    }
    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    uint64_t inNumel = 0;
    uint64_t outNumel = 0;
    MPSDataType inDtype = MPSDataTypeInvalid;
    MPSDataType outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, &inNumel, &inDtype) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), &outShape, &outNumel, &outDtype)) {
        return false;
    }
    if (!IsFloat16Or32(inDtype) || inDtype != outDtype || inShape.size() != 2 || outShape.size() != 2 ||
        outNumel == 0 || outNumel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (padOp.getEdgePaddingLow().size() != 2 || padOp.getEdgePaddingHigh().size() != 2 ||
        padOp.getInteriorPadding().size() != 2) {
        return false;
    }
    for (int64_t p : padOp.getInteriorPadding()) {
        if (p != 0) {
            return false;
        }
    }
    for (int64_t p : padOp.getEdgePaddingLow()) {
        if (p < 0 || p > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    for (int64_t p : padOp.getEdgePaddingHigh()) {
        if (p < 0 || p > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
    }
    const int64_t low_h = padOp.getEdgePaddingLow()[0];
    const int64_t low_w = padOp.getEdgePaddingLow()[1];
    const int64_t high_h = padOp.getEdgePaddingHigh()[0];
    const int64_t high_w = padOp.getEdgePaddingHigh()[1];
    if (outShape[0] != inShape[0] + low_h + high_h || outShape[1] != inShape[1] + low_w + high_w) {
        return false;
    }
    double padValue = 0.0;
    if (GetScalarFloatConstant(padOp.getPaddingValue(), &padValue)) {
        return true;
    }

    mlir::Value padVal = padOp.getPaddingValue();
    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(padVal)) {
        auto padType = mlir::dyn_cast<mlir::RankedTensorType>(padVal.getType());
        if (padType && padType.getNumElements() == 1 &&
            IsFloat16Or32(MlirTypeToMps(padType.getElementType()))) {
            return true;
        }
    }
    return false;
}

static bool IsScatterAddCombiner(mlir::stablehlo::ScatterOp scatterOp) {
    auto& region = scatterOp.getUpdateComputation();
    if (region.empty()) {
        return false;
    }
    for (auto& inner : region.front()) {
        if (mlir::isa<mlir::stablehlo::AddOp>(inner)) {
            return true;
        }
    }
    return false;
}

static bool IsEligibleScatterAddRowsNativeOp(mlir::Operation* op) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_NATIVE_SCATTER_ROWS")) {
        return false;
    }
    auto scatterOp = mlir::dyn_cast_or_null<mlir::stablehlo::ScatterOp>(op);
    if (!scatterOp || op->getNumOperands() != 3 || op->getNumResults() != 1) {
        return false;
    }
    if (!IsScatterAddCombiner(scatterOp)) {
        return false;
    }

    auto dnums = scatterOp.getScatterDimensionNumbers();
    auto updateWindowDims = dnums.getUpdateWindowDims();
    auto insertedWindowDims = dnums.getInsertedWindowDims();
    auto scatterDimsToOperandDims = dnums.getScatterDimsToOperandDims();
    auto inputBatchingDims = dnums.getInputBatchingDims();
    auto scatterIndicesBatchingDims = dnums.getScatterIndicesBatchingDims();

    if (!inputBatchingDims.empty() || !scatterIndicesBatchingDims.empty()) {
        return false;
    }
    if (dnums.getIndexVectorDim() != 1 || updateWindowDims.size() != 1 || updateWindowDims[0] != 1 ||
        insertedWindowDims.size() != 1 || insertedWindowDims[0] != 0 ||
        scatterDimsToOperandDims.size() != 1 || scatterDimsToOperandDims[0] != 0) {
        return false;
    }

    std::vector<int64_t> inShape, idxShape, updShape, outShape;
    uint64_t inNumel = 0, outNumel = 0, updNumel = 0;
    MPSDataType inType = MPSDataTypeInvalid, idxType = MPSDataTypeInvalid, updType = MPSDataTypeInvalid,
                outType = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, &inNumel, &inType) ||
        !GetStaticTensorInfo(op->getOperand(1).getType(), &idxShape, nullptr, &idxType) ||
        !GetStaticTensorInfo(op->getOperand(2).getType(), &updShape, &updNumel, &updType) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), &outShape, &outNumel, &outType)) {
        return false;
    }
    if (inShape.size() != 2 || outShape.size() != 2 || idxShape.size() != 2 || updShape.size() != 2) {
        return false;
    }
    if (idxType != MPSDataTypeInt32) {
        return false;
    }

    if (!(inType == MPSDataTypeFloat32 && updType == MPSDataTypeFloat32 && outType == MPSDataTypeFloat32)) {
        return false;
    }
    if (inShape != outShape) {
        return false;
    }
    if (idxShape[1] != 1 || idxShape[0] != updShape[0] || updShape[1] != inShape[1]) {
        return false;
    }
    if (inShape[0] <= 0 || inShape[1] <= 0 || idxShape[0] <= 0 || updNumel == 0 ||
        updNumel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return true;
}

static std::string GetReduceBodyOpName(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return "";
    }
    mlir::Block& block = reduceOp.getBody().front();
    for (mlir::Operation& inner : block) {
        std::string opName = inner.getName().getStringRef().str();
        if (opName == "stablehlo.return") {
            continue;
        }
        return opName;
    }
    return "";
}

static bool IsNeutralReduceInit(mlir::Value init, IntrinsicNativeKind kind) {
    double v = 0.0;
    if (!GetScalarFloatConstant(init, &v)) {
        return false;
    }
    if (kind == IntrinsicNativeKind::kReduceSumLastAxis) {
        return v == 0.0;
    }
    if (kind == IntrinsicNativeKind::kReduceMaxLastAxis) {
        return std::isinf(v) && v < 0.0;
    }
    if (kind == IntrinsicNativeKind::kReduceMinLastAxis) {
        return std::isinf(v) && v > 0.0;
    }
    return false;
}

static bool IsWidenedFp16ReducePattern(mlir::Operation* op) {
    if (!op || op->getNumOperands() < 1 || op->getNumResults() != 1) {
        return false;
    }
    auto inConvert = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(op->getOperand(0).getDefiningOp());
    if (!inConvert) {
        return false;
    }
    auto inSrcTy = mlir::dyn_cast<mlir::RankedTensorType>(inConvert.getOperand().getType());
    auto inDstTy = mlir::dyn_cast<mlir::RankedTensorType>(inConvert.getResult().getType());
    if (!inSrcTy || !inDstTy) {
        return false;
    }
    if (MlirTypeToMps(inSrcTy.getElementType()) != MPSDataTypeFloat16 ||
        MlirTypeToMps(inDstTy.getElementType()) != MPSDataTypeFloat32) {
        return false;
    }

    for (mlir::OpOperand& use : op->getResult(0).getUses()) {
        auto outConvert = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(use.getOwner());
        if (!outConvert) {
            continue;
        }
        auto outSrcTy = mlir::dyn_cast<mlir::RankedTensorType>(outConvert.getOperand().getType());
        auto outDstTy = mlir::dyn_cast<mlir::RankedTensorType>(outConvert.getResult().getType());
        if (!outSrcTy || !outDstTy) {
            continue;
        }
        if (MlirTypeToMps(outSrcTy.getElementType()) == MPSDataTypeFloat32 &&
            MlirTypeToMps(outDstTy.getElementType()) == MPSDataTypeFloat16) {
            return true;
        }
    }
    return false;
}

static IntrinsicNativeKind ClassifyElementwiseKind(mlir::Operation* op) {
    if (!op) {
        return IntrinsicNativeKind::kNone;
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name == "stablehlo.negate")
        return IntrinsicNativeKind::kNegate;
    if (name == "stablehlo.floor")
        return IntrinsicNativeKind::kFloor;
    if (name == "stablehlo.round_nearest_even")
        return IntrinsicNativeKind::kRoundNearestEven;
    if (name == "stablehlo.rsqrt")
        return IntrinsicNativeKind::kRsqrt;
    if (name == "stablehlo.sign")
        return IntrinsicNativeKind::kSign;
    if (name == "chlo.square")
        return IntrinsicNativeKind::kSquare;
    if (name == "stablehlo.abs")
        return IntrinsicNativeKind::kAbs;
    if (name == "stablehlo.exponential")
        return IntrinsicNativeKind::kExp;
    if (name == "stablehlo.log")
        return IntrinsicNativeKind::kLog;
    if (name == "stablehlo.tanh")
        return IntrinsicNativeKind::kTanh;
    if (name == "stablehlo.ceil")
        return IntrinsicNativeKind::kCeil;
    if (name == "stablehlo.sine")
        return IntrinsicNativeKind::kSin;
    if (name == "stablehlo.cosine")
        return IntrinsicNativeKind::kCos;
    if (name == "stablehlo.sqrt")
        return IntrinsicNativeKind::kSqrt;
    if (name == "stablehlo.logistic")
        return IntrinsicNativeKind::kLogistic;
    if (name == "stablehlo.add")
        return IntrinsicNativeKind::kAdd;
    if (name == "stablehlo.subtract")
        return IntrinsicNativeKind::kSubtract;
    if (name == "stablehlo.multiply")
        return IntrinsicNativeKind::kMultiply;
    if (name == "stablehlo.divide")
        return IntrinsicNativeKind::kDivide;
    if (name == "stablehlo.maximum")
        return IntrinsicNativeKind::kMaximum;
    if (name == "stablehlo.minimum")
        return IntrinsicNativeKind::kMinimum;
    if (name == "stablehlo.select")
        return IntrinsicNativeKind::kSelect;
    return IntrinsicNativeKind::kNone;
}

static IntrinsicNativeKind ClassifyReduceKind(mlir::Operation* op) {
    auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return IntrinsicNativeKind::kNone;
    }
    std::string bodyOp = GetReduceBodyOpName(reduceOp);
    if (bodyOp == "stablehlo.add")
        return IntrinsicNativeKind::kReduceSumLastAxis;
    if (bodyOp == "stablehlo.maximum")
        return IntrinsicNativeKind::kReduceMaxLastAxis;
    if (bodyOp == "stablehlo.minimum")
        return IntrinsicNativeKind::kReduceMinLastAxis;
    return IntrinsicNativeKind::kNone;
}

static bool IsEligibleElementwiseNativeOp(mlir::Operation* op, IntrinsicNativeKind kind) {
    if (!op || kind == IntrinsicNativeKind::kNone || op->getNumResults() != 1) {
        return false;
    }
    std::vector<int64_t> outShape;
    uint64_t outNumel = 0;
    MPSDataType outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), &outShape, &outNumel, &outDtype)) {
        return false;
    }
    if (!IsFloat16Or32(outDtype) || outNumel == 0 || outNumel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (kind == IntrinsicNativeKind::kSelect) {
        if (op->getNumOperands() != 3) {
            return false;
        }
        auto predType = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
        if (!predType || !predType.getElementType().isInteger(1)) {
            return false;
        }
        uint64_t predNumel = 0;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &predNumel, nullptr)) {
            return false;
        }
        if (predNumel != outNumel) {
            return false;
        }

        uint64_t xNumel = 0, yNumel = 0;
        MPSDataType xDtype = MPSDataTypeInvalid, yDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(1).getType(), nullptr, &xNumel, &xDtype) ||
            !GetStaticTensorInfo(op->getOperand(2).getType(), nullptr, &yNumel, &yDtype)) {
            return false;
        }
        return xNumel == outNumel && yNumel == outNumel && xDtype == outDtype && yDtype == outDtype;
    }

    if (kind == IntrinsicNativeKind::kNegate || kind == IntrinsicNativeKind::kFloor ||
        kind == IntrinsicNativeKind::kRoundNearestEven || kind == IntrinsicNativeKind::kRsqrt ||
        kind == IntrinsicNativeKind::kSign || kind == IntrinsicNativeKind::kSquare ||
        kind == IntrinsicNativeKind::kAbs || kind == IntrinsicNativeKind::kExp ||
        kind == IntrinsicNativeKind::kLog || kind == IntrinsicNativeKind::kTanh ||
        kind == IntrinsicNativeKind::kCeil || kind == IntrinsicNativeKind::kSin ||
        kind == IntrinsicNativeKind::kCos || kind == IntrinsicNativeKind::kSqrt ||
        kind == IntrinsicNativeKind::kLogistic) {
        if (op->getNumOperands() != 1) {
            return false;
        }
        uint64_t inNumel = 0;
        MPSDataType inDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &inNumel, &inDtype)) {
            return false;
        }
        return inNumel == outNumel && inDtype == outDtype;
    }

    if (kind == IntrinsicNativeKind::kAdd || kind == IntrinsicNativeKind::kSubtract ||
        kind == IntrinsicNativeKind::kMultiply || kind == IntrinsicNativeKind::kDivide ||
        kind == IntrinsicNativeKind::kMaximum || kind == IntrinsicNativeKind::kMinimum) {
        if (op->getNumOperands() != 2) {
            return false;
        }
        uint64_t lhsNumel = 0, rhsNumel = 0;
        MPSDataType lhsDtype = MPSDataTypeInvalid, rhsDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &lhsNumel, &lhsDtype) ||
            !GetStaticTensorInfo(op->getOperand(1).getType(), nullptr, &rhsNumel, &rhsDtype)) {
            return false;
        }
        return lhsNumel == outNumel && rhsNumel == outNumel && lhsDtype == outDtype && rhsDtype == outDtype;
    }

    return false;
}

static bool IsEligibleReduceNativeOp(mlir::Operation* op, IntrinsicNativeKind kind) {
    auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp || kind == IntrinsicNativeKind::kNone) {
        return false;
    }
    if (op->getNumResults() != 1 || op->getNumOperands() != 2) {
        return false;
    }
    auto dims = reduceOp.getDimensions();
    if (dims.size() != 1) {
        return false;
    }

    std::vector<int64_t> inShape;
    uint64_t inNumel = 0;
    MPSDataType inDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, &inNumel, &inDtype)) {
        return false;
    }
    uint64_t outNumel = 0;
    MPSDataType outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, &outDtype)) {
        return false;
    }
    if (!IsFloat16Or32(inDtype) || inDtype != outDtype || inShape.empty() || inNumel == 0 ||
        inNumel > std::numeric_limits<uint32_t>::max() || outNumel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }

    if (kind == IntrinsicNativeKind::kReduceSumLastAxis && IsWidenedFp16ReducePattern(op) &&
        !EnvFlagEnabled("JAX_METALLIB_ENABLE_NATIVE_WIDENED_FP16_REDUCE_SUM")) {
        return false;
    }

    int64_t axis = dims[0];
    int64_t rank = static_cast<int64_t>(inShape.size());
    if (axis != rank - 1) {
        return false;
    }
    int64_t cols = inShape.back();
    if (cols <= 0 || cols > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (outNumel * static_cast<uint64_t>(cols) != inNumel) {
        return false;
    }
    return IsNeutralReduceInit(reduceOp.getInitValues()[0], kind);
}

static IntrinsicNativeKind ClassifyConvertKind(mlir::Operation* op) {
    auto convertOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConvertOp>(op);
    if (!convertOp || op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return IntrinsicNativeKind::kNone;
    }
    auto inTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto outTy = mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inTy || !outTy) {
        return IntrinsicNativeKind::kNone;
    }
    MPSDataType inDtype = MlirTypeToMps(inTy.getElementType());
    MPSDataType outDtype = MlirTypeToMps(outTy.getElementType());
    bool isFp16ToFp32 = (inDtype == MPSDataTypeFloat16 && outDtype == MPSDataTypeFloat32);
    bool isFp32ToFp16 = (inDtype == MPSDataTypeFloat32 && outDtype == MPSDataTypeFloat16);
    if (isFp16ToFp32 || isFp32ToFp16) {
        return IntrinsicNativeKind::kConvertFp16Fp32;
    }
    return IntrinsicNativeKind::kNone;
}

static bool IsEligibleConvertNativeOp(mlir::Operation* op) {
    if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return false;
    }
    uint64_t inNumel = 0, outNumel = 0;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &inNumel, nullptr) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, nullptr)) {
        return false;
    }
    if (inNumel == 0 || inNumel != outNumel || inNumel > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    return ClassifyConvertKind(op) != IntrinsicNativeKind::kNone;
}

static std::mutex g_fused_cache_mu;
static std::unordered_map<uintptr_t, std::unordered_map<std::string, id<MTLComputePipelineState>>>
    g_fused_cache;

static id<MTLComputePipelineState> GetFastNativePipeline(id<MTLDevice> device,
                                                         const std::string& functionName) {
    static std::mutex mu;
    static std::unordered_map<uintptr_t, FastNativeKernelCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mu);
    auto& entry = cache[(uintptr_t)device];
    if (!entry.library) {
        NSError* error = nil;
        entry.library = [device newLibraryWithSource:FastNativeKernelsSource() options:nil error:&error];
        if (!entry.library) {
            MPS_LOG_ERROR("native-fast: shader compile failed: %s\n", error.localizedDescription.UTF8String);
            return nil;
        }
    }

    auto it = entry.pipelines.find(functionName);
    if (it != entry.pipelines.end()) {
        return it->second;
    }

    NSString* fnName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> fn = [entry.library newFunctionWithName:fnName];
    if (!fn) {
        NSArray<NSString*>* functionNames = [entry.library functionNames];
        NSString* joined = [functionNames componentsJoinedByString:@","];
        MPS_LOG_ERROR("native-fast: kernel not found: %s\n", functionName.c_str());
        if (joined) {
            MPS_LOG_ERROR("native-fast: available kernels: %s\n", joined.UTF8String);
        }
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    if (!pipeline) {
        MPS_LOG_ERROR("native-fast: pipeline creation failed for %s: %s\n", functionName.c_str(),
                      error.localizedDescription.UTF8String);
        return nil;
    }
    entry.pipelines.emplace(functionName, pipeline);
    return pipeline;
}

static std::string ElementwiseKernelFunctionName(IntrinsicNativeKind kind, MPSDataType dtype) {
    const bool f16 = (dtype == MPSDataTypeFloat16);
    switch (kind) {
    case IntrinsicNativeKind::kNegate:
        return f16 ? "unary_neg_f16" : "unary_neg_f32";
    case IntrinsicNativeKind::kFloor:
        return f16 ? "unary_floor_f16" : "unary_floor_f32";
    case IntrinsicNativeKind::kRoundNearestEven:
        return f16 ? "unary_rint_f16" : "unary_rint_f32";
    case IntrinsicNativeKind::kRsqrt:
        return f16 ? "unary_rsqrt_f16" : "unary_rsqrt_f32";
    case IntrinsicNativeKind::kSign:
        return f16 ? "unary_sign_f16" : "unary_sign_f32";
    case IntrinsicNativeKind::kSquare:
        return f16 ? "unary_square_f16" : "unary_square_f32";
    case IntrinsicNativeKind::kAbs:
        return f16 ? "unary_abs_f16" : "unary_abs_f32";
    case IntrinsicNativeKind::kExp:
        return f16 ? "unary_exp_f16" : "unary_exp_f32";
    case IntrinsicNativeKind::kLog:
        return f16 ? "unary_log_f16" : "unary_log_f32";
    case IntrinsicNativeKind::kTanh:
        return f16 ? "unary_tanh_f16" : "unary_tanh_f32";
    case IntrinsicNativeKind::kCeil:
        return f16 ? "unary_ceil_f16" : "unary_ceil_f32";
    case IntrinsicNativeKind::kSin:
        return f16 ? "unary_sin_f16" : "unary_sin_f32";
    case IntrinsicNativeKind::kCos:
        return f16 ? "unary_cos_f16" : "unary_cos_f32";
    case IntrinsicNativeKind::kSqrt:
        return f16 ? "unary_sqrt_f16" : "unary_sqrt_f32";
    case IntrinsicNativeKind::kLogistic:
        return f16 ? "unary_sigmoid_f16" : "unary_sigmoid_f32";
    case IntrinsicNativeKind::kAdd:
        return f16 ? "binary_add_f16" : "binary_add_f32";
    case IntrinsicNativeKind::kSubtract:
        return f16 ? "binary_sub_f16" : "binary_sub_f32";
    case IntrinsicNativeKind::kMultiply:
        return f16 ? "binary_mul_f16" : "binary_mul_f32";
    case IntrinsicNativeKind::kDivide:
        return f16 ? "binary_div_f16" : "binary_div_f32";
    case IntrinsicNativeKind::kMaximum:
        return f16 ? "binary_max_f16" : "binary_max_f32";
    case IntrinsicNativeKind::kMinimum:
        return f16 ? "binary_min_f16" : "binary_min_f32";
    case IntrinsicNativeKind::kSelect:
        return f16 ? "ternary_select_f16" : "ternary_select_f32";
    default:
        return "";
    }
}

static std::string ElementwiseVec4KernelFunctionName(IntrinsicNativeKind kind, MPSDataType dtype) {
    if (dtype != MPSDataTypeFloat32 && dtype != MPSDataTypeFloat16) {
        return "";
    }
    const bool f16 = (dtype == MPSDataTypeFloat16);
    switch (kind) {
    case IntrinsicNativeKind::kSquare:
        return f16 ? "unary_square_vec4_f16" : "unary_square_vec4_f32";
    case IntrinsicNativeKind::kAdd:
        return f16 ? "binary_add_vec4_f16" : "binary_add_vec4_f32";
    case IntrinsicNativeKind::kSubtract:
        return f16 ? "binary_sub_vec4_f16" : "binary_sub_vec4_f32";
    case IntrinsicNativeKind::kMultiply:
        return f16 ? "binary_mul_vec4_f16" : "binary_mul_vec4_f32";
    case IntrinsicNativeKind::kDivide:
        return f16 ? "binary_div_vec4_f16" : "binary_div_vec4_f32";
    case IntrinsicNativeKind::kMaximum:
        return f16 ? "binary_max_vec4_f16" : "binary_max_vec4_f32";
    case IntrinsicNativeKind::kMinimum:
        return f16 ? "binary_min_vec4_f16" : "binary_min_vec4_f32";
    case IntrinsicNativeKind::kSelect:
        return f16 ? "ternary_select_vec4_f16" : "ternary_select_vec4_f32";
    default:
        return "";
    }
}

static std::string ElementwiseVec8KernelFunctionName(IntrinsicNativeKind kind) {
    switch (kind) {
    case IntrinsicNativeKind::kSquare:
        return "unary_square_vec8_f16";
    case IntrinsicNativeKind::kAdd:
        return "binary_add_vec8_f16";
    case IntrinsicNativeKind::kSubtract:
        return "binary_sub_vec8_f16";
    case IntrinsicNativeKind::kMultiply:
        return "binary_mul_vec8_f16";
    case IntrinsicNativeKind::kDivide:
        return "binary_div_vec8_f16";
    case IntrinsicNativeKind::kMaximum:
        return "binary_max_vec8_f16";
    case IntrinsicNativeKind::kMinimum:
        return "binary_min_vec8_f16";
    case IntrinsicNativeKind::kSelect:
        return "ternary_select_vec8_f16";
    default:
        return "";
    }
}

static const char* FusionSnippet(IntrinsicNativeKind kind) {
    switch (kind) {
    case IntrinsicNativeKind::kNegate:
        return "-({in0})";
    case IntrinsicNativeKind::kFloor:
        return "floor({in0})";
    case IntrinsicNativeKind::kRoundNearestEven:
        return "rint({in0})";
    case IntrinsicNativeKind::kRsqrt:
        return "rsqrt({in0})";
    case IntrinsicNativeKind::kSign:
        return "sign({in0})";
    case IntrinsicNativeKind::kSquare:
        return "({in0}) * ({in0})";
    case IntrinsicNativeKind::kAbs:
        return "abs({in0})";
    case IntrinsicNativeKind::kExp:
        return "exp({in0})";
    case IntrinsicNativeKind::kLog:
        return "log({in0})";
    case IntrinsicNativeKind::kTanh:
        return "tanh({in0})";
    case IntrinsicNativeKind::kCeil:
        return "ceil({in0})";
    case IntrinsicNativeKind::kSin:
        return "sin({in0})";
    case IntrinsicNativeKind::kCos:
        return "cos({in0})";
    case IntrinsicNativeKind::kSqrt:
        return "sqrt({in0})";
    case IntrinsicNativeKind::kLogistic:
        return "(1.0 / (1.0 + exp(-({in0}))))";
    case IntrinsicNativeKind::kAdd:
        return "({in0}) + ({in1})";
    case IntrinsicNativeKind::kSubtract:
        return "({in0}) - ({in1})";
    case IntrinsicNativeKind::kMultiply:
        return "({in0}) * ({in1})";
    case IntrinsicNativeKind::kDivide:
        return "({in0}) / ({in1})";
    case IntrinsicNativeKind::kMaximum:
        return "max({in0}, {in1})";
    case IntrinsicNativeKind::kMinimum:
        return "min({in0}, {in1})";

    default:
        return nullptr;
    }
}

static const char* FusionKindShortName(IntrinsicNativeKind kind) {
    switch (kind) {
    case IntrinsicNativeKind::kNegate:
        return "neg";
    case IntrinsicNativeKind::kFloor:
        return "flr";
    case IntrinsicNativeKind::kRoundNearestEven:
        return "rne";
    case IntrinsicNativeKind::kRsqrt:
        return "rsq";
    case IntrinsicNativeKind::kSign:
        return "sgn";
    case IntrinsicNativeKind::kSquare:
        return "sq";
    case IntrinsicNativeKind::kAbs:
        return "abs";
    case IntrinsicNativeKind::kExp:
        return "exp";
    case IntrinsicNativeKind::kLog:
        return "log";
    case IntrinsicNativeKind::kTanh:
        return "tnh";
    case IntrinsicNativeKind::kCeil:
        return "cel";
    case IntrinsicNativeKind::kSin:
        return "sin";
    case IntrinsicNativeKind::kCos:
        return "cos";
    case IntrinsicNativeKind::kSqrt:
        return "sqt";
    case IntrinsicNativeKind::kLogistic:
        return "sig";
    case IntrinsicNativeKind::kAdd:
        return "add";
    case IntrinsicNativeKind::kSubtract:
        return "sub";
    case IntrinsicNativeKind::kMultiply:
        return "mul";
    case IntrinsicNativeKind::kDivide:
        return "div";
    case IntrinsicNativeKind::kMaximum:
        return "max";
    case IntrinsicNativeKind::kMinimum:
        return "min";
    case IntrinsicNativeKind::kSelect:
        return "sel";
    default:
        return "?";
    }
}

static bool IsFusibleElementwiseKind(IntrinsicNativeKind kind) {
    return FusionSnippet(kind) != nullptr;
}

struct FusionOp {
    IntrinsicNativeKind kind;

    std::vector<int> sources;
};

static std::string GenerateFusedKernelSource(const std::string& fn_name, const std::vector<FusionOp>& ops,
                                             int num_external_inputs, MPSDataType dtype, int vec_mode) {
    const char* scalar_t = (dtype == MPSDataTypeFloat16) ? "half" : "float";
    const char* vec_t;
    const char* n_param;
    if (vec_mode == 8) {
        vec_t = "half4";
        n_param = "n8";
    } else if (vec_mode == 4) {
        vec_t = (dtype == MPSDataTypeFloat16) ? "half4" : "float4";
        n_param = "n4";
    } else {
        vec_t = scalar_t;
        n_param = "n";
    }

    std::ostringstream src;
    src << "#include <metal_stdlib>\nusing namespace metal;\n\n";
    src << "kernel void " << fn_name << "(\n";

    int buf_idx = 0;
    for (int i = 0; i < num_external_inputs; ++i) {
        src << "    device const " << vec_t << "* ext" << i << " [[buffer(" << buf_idx++ << ")]],\n";
    }
    src << "    device " << vec_t << "* out [[buffer(" << buf_idx++ << ")]],\n";
    src << "    constant uint& " << n_param << " [[buffer(" << buf_idx++ << ")]],\n";
    src << "    uint gid [[thread_position_in_grid]]\n";
    src << ") {\n";
    src << "  if (gid < " << n_param << ") {\n";

    auto emit_chain_body = [&](const std::string& idx, const std::string& indent) {
        auto resolve = [&](int ref) -> std::string {
            if (ref < 0) {
                return "ext" + std::to_string(-ref - 1) + "[" + idx + "]";
            }
            return "t" + std::to_string(ref);
        };

        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& fop = ops[i];
            std::string snippet(FusionSnippet(fop.kind));

            if (fop.sources.size() >= 1) {
                std::string val = resolve(fop.sources[0]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in0}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }

            if (fop.sources.size() >= 2) {
                std::string val = resolve(fop.sources[1]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in1}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }

            if (fop.sources.size() >= 3) {
                std::string val = resolve(fop.sources[2]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in2}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }

            bool is_last = (i == ops.size() - 1);
            if (is_last) {
                src << indent << "out[" << idx << "] = " << snippet << ";\n";
            } else {
                src << indent << vec_t << " t" << i << " = " << snippet << ";\n";
            }
        }
    };

    if (vec_mode == 8) {
        src << "    uint i = gid * 2;\n";
        emit_chain_body("i", "    ");

        for (size_t oi = 0; oi < ops.size(); ++oi) {
        }

        src << "    {\n";

        auto resolve2 = [&](int ref) -> std::string {
            if (ref < 0) {
                return "ext" + std::to_string(-ref - 1) + "[i + 1]";
            }
            return "u" + std::to_string(ref);
        };
        for (size_t i = 0; i < ops.size(); ++i) {
            const auto& fop = ops[i];
            std::string snippet(FusionSnippet(fop.kind));
            if (fop.sources.size() >= 1) {
                std::string val = resolve2(fop.sources[0]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in0}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }
            if (fop.sources.size() >= 2) {
                std::string val = resolve2(fop.sources[1]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in1}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }
            if (fop.sources.size() >= 3) {
                std::string val = resolve2(fop.sources[2]);
                std::string::size_type pos;
                while ((pos = snippet.find("{in2}")) != std::string::npos) {
                    snippet.replace(pos, 5, val);
                }
            }
            bool is_last = (i == ops.size() - 1);
            if (is_last) {
                src << "      out[i + 1] = " << snippet << ";\n";
            } else {
                src << "      " << vec_t << " u" << i << " = " << snippet << ";\n";
            }
        }
        src << "    }\n";
    } else {
        emit_chain_body("gid", "    ");
    }

    src << "  }\n";
    src << "}\n";
    return src.str();
}

static id<MTLComputePipelineState> GetOrCompileFusedPipeline(id<MTLDevice> device,
                                                             const std::string& cache_key,
                                                             const std::string& fn_name,
                                                             const std::string& source) {
    std::lock_guard<std::mutex> lock(g_fused_cache_mu);
    auto& device_cache = g_fused_cache[(uintptr_t)device];
    auto it = device_cache.find(cache_key);
    if (it != device_cache.end()) {
        return it->second;
    }

    NSString* src = [[NSString alloc] initWithUTF8String:source.c_str()];
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithSource:src options:nil error:&error];
    [src release];
    if (!library) {
        MPS_LOG_ERROR("fused-kernel: compile failed for %s: %s\n", cache_key.c_str(),
                      error.localizedDescription.UTF8String);
        return nil;
    }

    NSString* fnName = [NSString stringWithUTF8String:fn_name.c_str()];
    id<MTLFunction> fn = [library newFunctionWithName:fnName];
    if (!fn) {
        MPS_LOG_ERROR("fused-kernel: function not found: %s\n", fn_name.c_str());
        [library release];
        return nil;
    }

    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    [library release];
    if (!pipeline) {
        MPS_LOG_ERROR("fused-kernel: pipeline failed for %s: %s\n", fn_name.c_str(),
                      error.localizedDescription.UTF8String);
        return nil;
    }

    device_cache[cache_key] = pipeline;
    MPS_LOG_INFO("fused-kernel: compiled %s (%zu chars)\n", cache_key.c_str(), source.size());
    return pipeline;
}

static std::string ReduceKernelFunctionName(IntrinsicNativeKind kind, MPSDataType dtype,
                                            bool use_threadgroup) {
    const bool f16 = (dtype == MPSDataTypeFloat16);
    switch (kind) {
    case IntrinsicNativeKind::kReduceSumLastAxis:
        if (use_threadgroup) {
            return f16 ? "reduce_sum_last_f16_tg" : "reduce_sum_last_f32_tg";
        }
        return f16 ? "reduce_sum_last_f16_lin" : "reduce_sum_last_f32_lin";
    case IntrinsicNativeKind::kReduceMaxLastAxis:
        if (use_threadgroup) {
            return f16 ? "reduce_max_last_f16_tg" : "reduce_max_last_f32_tg";
        }
        return f16 ? "reduce_max_last_f16_lin" : "reduce_max_last_f32_lin";
    case IntrinsicNativeKind::kReduceMinLastAxis:
        if (use_threadgroup) {
            return f16 ? "reduce_min_last_f16_tg" : "reduce_min_last_f32_tg";
        }
        return f16 ? "reduce_min_last_f16_lin" : "reduce_min_last_f32_lin";
    default:
        return "";
    }
}

static std::string IotaKernelFunctionName(MPSDataType dtype) {
    if (dtype == MPSDataTypeInt32) {
        return "iota_i32";
    }
    if (dtype == MPSDataTypeFloat32) {
        return "iota_f32";
    }
    return "";
}

static std::string PadKernelFunctionName(MPSDataType dtype) {
    if (dtype == MPSDataTypeFloat16) {
        return "pad2d_f16";
    }
    if (dtype == MPSDataTypeFloat32) {
        return "pad2d_f32";
    }
    return "";
}

static int LinearDispatchDefaultThreadgroupCap(MPSDataType) {

    return 512;
}

static NSUInteger ResolveLinearThreadgroupSize(id<MTLComputePipelineState> pipeline, MPSDataType dtype) {
    const int default_tg_cap = LinearDispatchDefaultThreadgroupCap(dtype);
    const int tg_cap = EnvIntValue("JAX_METALLIB_LINEAR_TG_SIZE", default_tg_cap);
    NSUInteger tg = MIN((NSUInteger)tg_cap, pipeline.maxTotalThreadsPerThreadgroup);
    if (tg == 0) {
        tg = 1;
    }
    return tg;
}

static void EncodeLinearDispatch(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                                 uint32_t count, MPSDataType dtype = MPSDataTypeInvalid) {
    NSUInteger tg = ResolveLinearThreadgroupSize(pipeline, dtype);
    [encoder dispatchThreads:MTLSizeMake((NSUInteger)count, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static NSUInteger Pow2Floor(NSUInteger v) {
    if (v >= 256) {
        return 256;
    }
    if (v >= 128) {
        return 128;
    }
    if (v >= 64) {
        return 64;
    }
    if (v >= 32) {
        return 32;
    }
    if (v >= 16) {
        return 16;
    }
    if (v >= 8) {
        return 8;
    }
    if (v >= 4) {
        return 4;
    }
    if (v >= 2) {
        return 2;
    }
    return 1;
}

static void EncodeRowReduceDispatch(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLComputePipelineState> pipeline, uint32_t rows) {
    NSUInteger tg = Pow2Floor(MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup));
    if (tg == 0) {
        tg = 1;
    }
    [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
}

static void ComputeLinearDispatchConfig(id<MTLComputePipelineState> pipeline, uint32_t count,
                                        MTLSize* grid_out, MTLSize* tg_out,
                                        MPSDataType dtype = MPSDataTypeInvalid) {
    NSUInteger tg = ResolveLinearThreadgroupSize(pipeline, dtype);
    if (grid_out) {
        *grid_out = MTLSizeMake((NSUInteger)count, 1, 1);
    }
    if (tg_out) {
        *tg_out = MTLSizeMake(tg, 1, 1);
    }
}

static void ComputeRowReduceDispatchConfig(id<MTLComputePipelineState> pipeline, uint32_t rows,
                                           MTLSize* grid_out, MTLSize* tg_out) {
    NSUInteger tg = Pow2Floor(MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup));
    if (tg == 0) {
        tg = 1;
    }
    if (grid_out) {
        *grid_out = MTLSizeMake((NSUInteger)rows, 1, 1);
    }
    if (tg_out) {
        *tg_out = MTLSizeMake(tg, 1, 1);
    }
}

static id<MTLBuffer> NativeIntrinsicElementwiseHandler(id<MTLDevice> device,
                                                       id<MTLCommandBuffer> commandBuffer,
                                                       mlir::Operation* op,
                                                       const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op) {
        return nil;
    }
    IntrinsicNativeKind kind = ClassifyElementwiseKind(op);
    if (!IsEligibleElementwiseNativeOp(op, kind)) {
        return nil;
    }

    uint64_t numel = 0;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &numel, &dtype) || numel == 0 ||
        numel > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    const bool supports_vec4_kind =
        kind == IntrinsicNativeKind::kSquare || kind == IntrinsicNativeKind::kAdd ||
        kind == IntrinsicNativeKind::kSubtract || kind == IntrinsicNativeKind::kMultiply ||
        kind == IntrinsicNativeKind::kDivide || kind == IntrinsicNativeKind::kMaximum ||
        kind == IntrinsicNativeKind::kMinimum || kind == IntrinsicNativeKind::kSelect;
    const bool vec_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_VEC4_INTRINSICS");
    const bool vec4_dtype_ok = IsFloat16Or32(dtype);

    const bool use_vec8 =
        !vec_disabled && supports_vec4_kind && dtype == MPSDataTypeFloat16 && (numel % 8 == 0);
    const bool use_vec4 =
        !use_vec8 && !vec_disabled && vec4_dtype_ok && supports_vec4_kind && (numel % 4 == 0);

    std::string kernelName;
    if (use_vec8) {
        kernelName = ElementwiseVec8KernelFunctionName(kind);
    } else if (use_vec4) {
        kernelName = ElementwiseVec4KernelFunctionName(kind, dtype);
    } else {
        kernelName = ElementwiseKernelFunctionName(kind, dtype);
    }
    if (kernelName.empty()) {
        return nil;
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    uint32_t n = static_cast<uint32_t>(use_vec8 ? (numel / 8) : (use_vec4 ? (numel / 4) : numel));

    if (kind == IntrinsicNativeKind::kSelect) {
        if (inputs.size() < 3 || !inputs[0] || !inputs[1] || !inputs[2]) {
            CFRelease((__bridge CFTypeRef)out);
            return nil;
        }
        [encoder setBuffer:inputs[0] offset:0 atIndex:0];
        [encoder setBuffer:inputs[1] offset:0 atIndex:1];
        [encoder setBuffer:inputs[2] offset:0 atIndex:2];
        [encoder setBuffer:out offset:0 atIndex:3];
        [encoder setBytes:&n length:sizeof(n) atIndex:4];
    } else if (kind == IntrinsicNativeKind::kNegate || kind == IntrinsicNativeKind::kFloor ||
               kind == IntrinsicNativeKind::kRoundNearestEven || kind == IntrinsicNativeKind::kRsqrt ||
               kind == IntrinsicNativeKind::kSign || kind == IntrinsicNativeKind::kSquare) {
        if (inputs.empty() || !inputs[0]) {
            CFRelease((__bridge CFTypeRef)out);
            return nil;
        }
        [encoder setBuffer:inputs[0] offset:0 atIndex:0];
        [encoder setBuffer:out offset:0 atIndex:1];
        [encoder setBytes:&n length:sizeof(n) atIndex:2];
    } else {
        if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
            CFRelease((__bridge CFTypeRef)out);
            return nil;
        }
        [encoder setBuffer:inputs[0] offset:0 atIndex:0];
        [encoder setBuffer:inputs[1] offset:0 atIndex:1];
        [encoder setBuffer:out offset:0 atIndex:2];
        [encoder setBytes:&n length:sizeof(n) atIndex:3];
    }

    EncodeLinearDispatch(encoder, pipeline, n, dtype);
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicReduceHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                  mlir::Operation* op,
                                                  const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op || inputs.empty() || !inputs[0]) {
        return nil;
    }

    auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return nil;
    }

    IntrinsicNativeKind kind = ClassifyReduceKind(op);
    if (!IsEligibleReduceNativeOp(op, kind)) {
        return nil;
    }

    std::vector<int64_t> inShape;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype) || inShape.empty()) {
        return nil;
    }
    uint64_t outNumel = 0;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, nullptr) || outNumel == 0 ||
        outNumel > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    int64_t cols64 = inShape.back();
    if (cols64 <= 0 || cols64 > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    uint32_t cols = static_cast<uint32_t>(cols64);
    uint32_t rows = static_cast<uint32_t>(outNumel);
    const uint32_t tg_cols_threshold =
        static_cast<uint32_t>(EnvIntValue("JAX_METALLIB_REDUCE_TG_COLS_THRESHOLD", 512));
    const bool use_threadgroup = cols >= tg_cols_threshold;

    std::string kernelName = ReduceKernelFunctionName(kind, dtype, use_threadgroup);
    if (kernelName.empty()) {
        return nil;
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    if (use_threadgroup) {
        EncodeRowReduceDispatch(encoder, pipeline, rows);
    } else {
        EncodeLinearDispatch(encoder, pipeline, rows, dtype);
    }
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicConvertHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                   mlir::Operation* op,
                                                   const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op || inputs.empty() || !inputs[0]) {
        return nil;
    }
    if (!IsEligibleConvertNativeOp(op)) {
        return nil;
    }

    uint64_t numel = 0;
    MPSDataType inDtype = MPSDataTypeInvalid;
    MPSDataType outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &numel, &inDtype) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), nullptr, nullptr, &outDtype) || numel == 0 ||
        numel > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    bool isFp16ToFp32 = (inDtype == MPSDataTypeFloat16 && outDtype == MPSDataTypeFloat32);
    const bool use_vec4 = !EnvFlagEnabled("JAX_METALLIB_DISABLE_VEC4_INTRINSICS") && (numel % 4 == 0);
    std::string kernelName;
    if (isFp16ToFp32) {
        kernelName = use_vec4 ? "convert_f16_to_f32_vec4" : "convert_f16_to_f32";
    } else {
        kernelName = use_vec4 ? "convert_f32_to_f16_vec4" : "convert_f32_to_f16";
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    uint32_t n = static_cast<uint32_t>(use_vec4 ? (numel / 4) : numel);
    [encoder setBytes:&n length:sizeof(n) atIndex:2];

    EncodeLinearDispatch(encoder, pipeline, n, MPSDataTypeFloat32);
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicReshapeAliasHandler(id<MTLDevice> device,
                                                        id<MTLCommandBuffer> commandBuffer,
                                                        mlir::Operation* op,
                                                        const std::vector<id<MTLBuffer>>& inputs) {
    (void)device;
    (void)commandBuffer;
    if (!IsEligibleReshapeAliasOp(op) || inputs.empty() || !inputs[0]) {
        return nil;
    }
    CFRetain((__bridge CFTypeRef)inputs[0]);
    return inputs[0];
}

struct ConstantBufferKey {
    uintptr_t device = 0;
    void* op = nullptr;

    bool operator==(const ConstantBufferKey& other) const {
        return device == other.device && op == other.op;
    }
};

struct ConstantBufferKeyHash {
    size_t operator()(const ConstantBufferKey& key) const {
        size_t h1 = std::hash<uintptr_t>{}(key.device);
        size_t h2 = std::hash<void*>{}(key.op);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
    }
};

static bool WriteScalarConstantData(mlir::stablehlo::ConstantOp constOp, MPSDataType dtype, void* dst) {
    if (!dst) {
        return false;
    }
    auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
    if (!dense || dense.getNumElements() != 1) {
        return false;
    }
    if (dtype == MPSDataTypeFloat32) {
        if (!mlir::isa<mlir::FloatType>(dense.getElementType())) {
            return false;
        }
        float v = (*dense.getValues<llvm::APFloat>().begin()).convertToFloat();
        memcpy(dst, &v, sizeof(v));
        return true;
    }
    if (dtype == MPSDataTypeFloat16) {
        if (!mlir::isa<mlir::FloatType>(dense.getElementType())) {
            return false;
        }
        llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
        uint16_t bits = static_cast<uint16_t>(ap.bitcastToAPInt().getZExtValue());
        memcpy(dst, &bits, sizeof(bits));
        return true;
    }
    if (dtype == MPSDataTypeInt32) {
        auto intType = mlir::dyn_cast<mlir::IntegerType>(dense.getElementType());
        if (!intType) {
            return false;
        }
        llvm::APInt ap = *dense.getValues<llvm::APInt>().begin();
        int32_t v = intType.isUnsigned() ? static_cast<int32_t>(ap.getZExtValue())
                                         : static_cast<int32_t>(ap.getSExtValue());
        memcpy(dst, &v, sizeof(v));
        return true;
    }
    if (dtype == MPSDataTypeBool) {
        auto intType = mlir::dyn_cast<mlir::IntegerType>(dense.getElementType());
        if (!intType || intType.getWidth() != 1) {
            return false;
        }
        llvm::APInt ap = *dense.getValues<llvm::APInt>().begin();
        uint8_t v = ap.getBoolValue() ? 1 : 0;
        memcpy(dst, &v, sizeof(v));
        return true;
    }
    return false;
}

static bool BakeNativeStep(id<MTLDevice> device, NativeStep* ns) {
    if (!device || !ns || !ns->op) {
        return false;
    }

    mlir::Operation* op = ns->op;

    IntrinsicNativeKind simpleKind = ClassifySimpleKind(op);
    if (simpleKind == IntrinsicNativeKind::kReshapeAlias && IsEligibleReshapeAliasOp(op)) {
        ns->baked_kind = NativeStep::BakedKind::kReshapeAlias;
        return true;
    }

    if (simpleKind == IntrinsicNativeKind::kConstantScalar && IsEligibleConstantScalarOp(op)) {
        auto constOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(op);
        if (!constOp) {
            return false;
        }
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, nullptr, &dtype)) {
            return false;
        }
        size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
        if (outBytes == 0) {
            return false;
        }
        id<MTLBuffer> buf = [device newBufferWithLength:outBytes options:ComputeBufferResourceOptions()];
        if (!buf || !WriteScalarConstantData(constOp, dtype, [buf contents])) {
            if (buf) {
                CFRelease((__bridge CFTypeRef)buf);
            }
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kConstant;
        ns->baked_constant_buffer = buf;
        return true;
    }

    if (simpleKind == IntrinsicNativeKind::kIota1D && IsEligibleIotaNativeOp(op)) {
        uint64_t numel = 0;
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &numel, &dtype) || numel == 0 ||
            numel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        std::string kernelName = IotaKernelFunctionName(dtype);
        if (kernelName.empty()) {
            return false;
        }
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kIota;
        ns->baked_pipeline = pipeline;
        ns->baked_n = static_cast<uint32_t>(numel);
        ComputeLinearDispatchConfig(pipeline, ns->baked_n, &ns->baked_grid, &ns->baked_tg, dtype);
        return true;
    }

    if (simpleKind == IntrinsicNativeKind::kPad2DNoInterior && IsEligiblePadNativeOp(op)) {
        auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
        if (!padOp) {
            return false;
        }
        std::vector<int64_t> inShape;
        std::vector<int64_t> outShape;
        uint64_t outNumel = 0;
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype) ||
            !GetStaticTensorInfo(op->getResult(0).getType(), &outShape, &outNumel, nullptr) ||
            inShape.size() != 2 || outShape.size() != 2 || outNumel == 0 ||
            outNumel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        double padValue64 = 0.0;
        bool pad_value_is_const = GetScalarFloatConstant(padOp.getPaddingValue(), &padValue64);
        std::string kernelName = PadKernelFunctionName(dtype);
        if (kernelName.empty()) {
            return false;
        }
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kPad2D;
        ns->baked_pipeline = pipeline;
        ns->baked_n = static_cast<uint32_t>(outNumel);
        ns->baked_in_h = static_cast<uint32_t>(inShape[0]);
        ns->baked_in_w = static_cast<uint32_t>(inShape[1]);
        ns->baked_out_w = static_cast<uint32_t>(outShape[1]);
        ns->baked_low_h = static_cast<uint32_t>(padOp.getEdgePaddingLow()[0]);
        ns->baked_low_w = static_cast<uint32_t>(padOp.getEdgePaddingLow()[1]);
        if (pad_value_is_const) {
            ns->baked_pad_value = static_cast<float>(padValue64);
        } else {

            ns->baked_pad_value_slot = ns->input_slots.size() > 1 ? ns->input_slots[1] : -1;
        }
        ComputeLinearDispatchConfig(pipeline, ns->baked_n, &ns->baked_grid, &ns->baked_tg, dtype);

        return true;
    }

    if (simpleKind == IntrinsicNativeKind::kScatterAddRowsF32 && IsEligibleScatterAddRowsNativeOp(op)) {
        std::vector<int64_t> inShape;
        std::vector<int64_t> updShape;
        MPSDataType scatterDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &scatterDtype) ||
            !GetStaticTensorInfo(op->getOperand(2).getType(), &updShape, nullptr, nullptr) ||
            inShape.size() != 2 || updShape.size() != 2) {
            return false;
        }
        uint64_t total64 = static_cast<uint64_t>(updShape[0]) * static_cast<uint64_t>(inShape[1]);
        if (total64 == 0 || total64 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const char* kernelName =
            (scatterDtype == MPSDataTypeFloat16) ? "scatter_add_rows_f16" : "scatter_add_rows_f32";
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kScatterAddRows;
        ns->baked_pipeline = pipeline;
        ns->baked_scatter_k = static_cast<uint32_t>(updShape[0]);
        ns->baked_scatter_cols = static_cast<uint32_t>(inShape[1]);
        ns->baked_scatter_out_rows = static_cast<uint32_t>(inShape[0]);
        ns->baked_scatter_total = static_cast<uint32_t>(total64);
        ComputeLinearDispatchConfig(pipeline, ns->baked_scatter_total, &ns->baked_grid, &ns->baked_tg,
                                    scatterDtype);
        return true;
    }

    IntrinsicNativeKind elemKind = ClassifyElementwiseKind(op);
    if (elemKind != IntrinsicNativeKind::kNone && IsEligibleElementwiseNativeOp(op, elemKind)) {
        uint64_t numel = 0;
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &numel, &dtype) || numel == 0 ||
            numel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const bool supports_vec4_kind =
            elemKind == IntrinsicNativeKind::kSquare || elemKind == IntrinsicNativeKind::kAdd ||
            elemKind == IntrinsicNativeKind::kSubtract || elemKind == IntrinsicNativeKind::kMultiply ||
            elemKind == IntrinsicNativeKind::kDivide || elemKind == IntrinsicNativeKind::kMaximum ||
            elemKind == IntrinsicNativeKind::kMinimum || elemKind == IntrinsicNativeKind::kSelect;
        const bool vec_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_VEC4_INTRINSICS");
        const bool vec4_dtype_ok = IsFloat16Or32(dtype);
        const bool use_vec8 =
            !vec_disabled && supports_vec4_kind && dtype == MPSDataTypeFloat16 && (numel % 8 == 0);
        const bool use_vec4 =
            !use_vec8 && !vec_disabled && vec4_dtype_ok && supports_vec4_kind && (numel % 4 == 0);
        std::string kernelName;
        if (use_vec8) {
            kernelName = ElementwiseVec8KernelFunctionName(elemKind);
        } else if (use_vec4) {
            kernelName = ElementwiseVec4KernelFunctionName(elemKind, dtype);
        } else {
            kernelName = ElementwiseKernelFunctionName(elemKind, dtype);
        }
        if (kernelName.empty()) {
            return false;
        }
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kElementwise;
        ns->baked_pipeline = pipeline;
        ns->baked_use_vec4 = use_vec4 || use_vec8;
        ns->baked_n = static_cast<uint32_t>(use_vec8 ? (numel / 8) : (use_vec4 ? (numel / 4) : numel));
        ComputeLinearDispatchConfig(pipeline, ns->baked_n, &ns->baked_grid, &ns->baked_tg, dtype);
        return true;
    }

    IntrinsicNativeKind reduceKind = ClassifyReduceKind(op);
    if (reduceKind != IntrinsicNativeKind::kNone && IsEligibleReduceNativeOp(op, reduceKind)) {
        std::vector<int64_t> inShape;
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype) || inShape.empty()) {
            return false;
        }
        uint64_t outNumel = 0;
        if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, nullptr) || outNumel == 0 ||
            outNumel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        uint32_t cols = static_cast<uint32_t>(inShape.back());
        uint32_t rows = static_cast<uint32_t>(outNumel);
        const uint32_t tg_cols_threshold =
            static_cast<uint32_t>(EnvIntValue("JAX_METALLIB_REDUCE_TG_COLS_THRESHOLD", 512));
        bool use_threadgroup = cols >= tg_cols_threshold;
        std::string kernelName = ReduceKernelFunctionName(reduceKind, dtype, use_threadgroup);
        if (kernelName.empty()) {
            return false;
        }
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kReduce;
        ns->baked_pipeline = pipeline;
        ns->baked_rows = rows;
        ns->baked_cols = cols;
        ns->baked_reduce_threadgroup = use_threadgroup;
        if (use_threadgroup) {
            ComputeRowReduceDispatchConfig(pipeline, rows, &ns->baked_grid, &ns->baked_tg);
        } else {
            ComputeLinearDispatchConfig(pipeline, rows, &ns->baked_grid, &ns->baked_tg, dtype);
        }
        return true;
    }

    IntrinsicNativeKind convertKind = ClassifyConvertKind(op);
    if (convertKind != IntrinsicNativeKind::kNone && IsEligibleConvertNativeOp(op)) {
        uint64_t numel = 0;
        MPSDataType inDtype = MPSDataTypeInvalid;
        MPSDataType outDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, &numel, &inDtype) ||
            !GetStaticTensorInfo(op->getResult(0).getType(), nullptr, nullptr, &outDtype) || numel == 0 ||
            numel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        bool isFp16ToFp32 = (inDtype == MPSDataTypeFloat16 && outDtype == MPSDataTypeFloat32);
        const bool use_vec4 = !EnvFlagEnabled("JAX_METALLIB_DISABLE_VEC4_INTRINSICS") && (numel % 4 == 0);
        std::string kernelName;
        if (isFp16ToFp32) {
            kernelName = use_vec4 ? "convert_f16_to_f32_vec4" : "convert_f16_to_f32";
        } else {
            kernelName = use_vec4 ? "convert_f32_to_f16_vec4" : "convert_f32_to_f16";
        }
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kElementwise;
        ns->baked_pipeline = pipeline;
        ns->baked_use_vec4 = use_vec4;
        ns->baked_n = static_cast<uint32_t>(use_vec4 ? (numel / 4) : numel);
        ComputeLinearDispatchConfig(pipeline, ns->baked_n, &ns->baked_grid, &ns->baked_tg,
                                    MPSDataTypeFloat32);
        return true;
    }

    return false;
}

static id<MTLBuffer> NativeIntrinsicConstantHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                    mlir::Operation* op,
                                                    const std::vector<id<MTLBuffer>>& inputs) {
    (void)commandBuffer;
    (void)inputs;
    auto constOp = mlir::dyn_cast_or_null<mlir::stablehlo::ConstantOp>(op);
    if (!device || !constOp || !IsEligibleConstantScalarOp(op)) {
        return nil;
    }

    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, nullptr, &dtype)) {
        return nil;
    }
    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }

    static std::mutex mu;
    static std::unordered_map<ConstantBufferKey, id<MTLBuffer>, ConstantBufferKeyHash> cache;
    ConstantBufferKey key{(uintptr_t)device, op};
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = cache.find(key);
        if (it != cache.end() && it->second) {
            CFRetain((__bridge CFTypeRef)it->second);
            return it->second;
        }
    }

    id<MTLBuffer> buf = [device newBufferWithLength:outBytes options:ComputeBufferResourceOptions()];
    if (!buf || !WriteScalarConstantData(constOp, dtype, [buf contents])) {
        if (buf) {
            CFRelease((__bridge CFTypeRef)buf);
        }
        return nil;
    }

    {
        std::lock_guard<std::mutex> lock(mu);
        cache.emplace(key, buf);
    }
    CFRetain((__bridge CFTypeRef)buf);
    return buf;
}

static id<MTLBuffer> NativeIntrinsicIotaHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                mlir::Operation* op,
                                                const std::vector<id<MTLBuffer>>& inputs) {
    (void)inputs;
    if (!device || !commandBuffer || !IsEligibleIotaNativeOp(op)) {
        return nil;
    }

    uint64_t numel = 0;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &numel, &dtype) || numel == 0 ||
        numel > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    std::string kernelName = IotaKernelFunctionName(dtype);
    if (kernelName.empty()) {
        return nil;
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }
    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    uint32_t n = static_cast<uint32_t>(numel);
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:out offset:0 atIndex:0];
    [encoder setBytes:&n length:sizeof(n) atIndex:1];
    EncodeLinearDispatch(encoder, pipeline, n, dtype);
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicPadHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                               mlir::Operation* op,
                                               const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !IsEligiblePadNativeOp(op) || inputs.empty() || !inputs[0]) {
        return nil;
    }
    auto padOp = mlir::dyn_cast<mlir::stablehlo::PadOp>(op);
    if (!padOp) {
        return nil;
    }

    std::vector<int64_t> inShape;
    std::vector<int64_t> outShape;
    uint64_t outNumel = 0;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), &outShape, &outNumel, nullptr) ||
        inShape.size() != 2 || outShape.size() != 2 || outNumel == 0 ||
        outNumel > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }

    double padValue64 = 0.0;
    if (!GetScalarFloatConstant(padOp.getPaddingValue(), &padValue64)) {

        if (inputs.size() >= 2 && inputs[1]) {
            void* pv_contents = [inputs[1] contents];
            if (pv_contents) {
                MPSDataType pvDtype = MPSDataTypeInvalid;
                GetStaticTensorInfo(op->getOperand(1).getType(), nullptr, nullptr, &pvDtype);
                if (pvDtype == MPSDataTypeFloat16) {
                    __fp16 h;
                    memcpy(&h, pv_contents, sizeof(h));
                    padValue64 = static_cast<double>(h);
                } else {
                    float f;
                    memcpy(&f, pv_contents, sizeof(f));
                    padValue64 = static_cast<double>(f);
                }
            }
        } else {
            return nil;
        }
    }
    float padValue = static_cast<float>(padValue64);

    std::string kernelName = PadKernelFunctionName(dtype);
    if (kernelName.empty()) {
        return nil;
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }
    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    uint32_t in_h = static_cast<uint32_t>(inShape[0]);
    uint32_t in_w = static_cast<uint32_t>(inShape[1]);
    uint32_t out_w = static_cast<uint32_t>(outShape[1]);
    uint32_t low_h = static_cast<uint32_t>(padOp.getEdgePaddingLow()[0]);
    uint32_t low_w = static_cast<uint32_t>(padOp.getEdgePaddingLow()[1]);
    uint32_t n = static_cast<uint32_t>(outNumel);

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&in_h length:sizeof(in_h) atIndex:2];
    [encoder setBytes:&in_w length:sizeof(in_w) atIndex:3];
    [encoder setBytes:&out_w length:sizeof(out_w) atIndex:4];
    [encoder setBytes:&low_h length:sizeof(low_h) atIndex:5];
    [encoder setBytes:&low_w length:sizeof(low_w) atIndex:6];
    [encoder setBytes:&padValue length:sizeof(padValue) atIndex:7];
    [encoder setBytes:&n length:sizeof(n) atIndex:8];
    EncodeLinearDispatch(encoder, pipeline, n, dtype);
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicScatterAddRowsHandler(id<MTLDevice> device,
                                                          id<MTLCommandBuffer> commandBuffer,
                                                          mlir::Operation* op,
                                                          const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !IsEligibleScatterAddRowsNativeOp(op) || inputs.size() < 3 ||
        !inputs[0] || !inputs[1] || !inputs[2]) {
        return nil;
    }

    std::vector<int64_t> inShape;
    std::vector<int64_t> updShape;
    MPSDataType scatterDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &scatterDtype) ||
        !GetStaticTensorInfo(op->getOperand(2).getType(), &updShape, nullptr, nullptr) ||
        inShape.size() != 2 || updShape.size() != 2) {
        return nil;
    }

    const uint32_t out_rows = static_cast<uint32_t>(inShape[0]);
    const uint32_t cols = static_cast<uint32_t>(inShape[1]);
    const uint32_t k = static_cast<uint32_t>(updShape[0]);
    const uint64_t total64 = static_cast<uint64_t>(k) * static_cast<uint64_t>(cols);
    if (total64 == 0 || total64 > std::numeric_limits<uint32_t>::max()) {
        return nil;
    }
    const uint32_t total = static_cast<uint32_t>(total64);

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }

    id<MTLBlitCommandEncoder> blit = [commandBuffer blitCommandEncoder];
    if (!blit) {
        CFRelease((__bridge CFTypeRef)out);
        return nil;
    }
    [blit copyFromBuffer:inputs[0] sourceOffset:0 toBuffer:out destinationOffset:0 size:outBytes];
    [blit endEncoding];

    const char* kernelName =
        (scatterDtype == MPSDataTypeFloat16) ? "scatter_add_rows_f16" : "scatter_add_rows_f32";
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        CFRelease((__bridge CFTypeRef)out);
        return nil;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:out offset:0 atIndex:0];
    [encoder setBuffer:inputs[1] offset:0 atIndex:1];
    [encoder setBuffer:inputs[2] offset:0 atIndex:2];
    [encoder setBytes:&k length:sizeof(k) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBytes:&out_rows length:sizeof(out_rows) atIndex:5];
    [encoder setBytes:&total length:sizeof(total) atIndex:6];
    EncodeLinearDispatch(encoder, pipeline, total, scatterDtype);
    [encoder endEncoding];
    return out;
}

static bool ShouldPreferGraphForIntrinsic(mlir::Operation* op) {
    (void)op;
    return false;
}

static bool ShouldRouteToIntrinsicNative(mlir::Operation* op) {
    if (!op) {
        return false;
    }

    if (EnvFlagEnabled("JAX_METALLIB_FORCE_INTRINSIC_GRAPH")) {
        return false;
    }
    if (!EnvFlagEnabled("JAX_METALLIB_FORCE_INTRINSIC_NATIVE")) {

        if (ShouldPreferGraphForIntrinsic(op)) {
            return false;
        }
    }

    return true;
}

static NativeOpHandler TrySelectIntrinsicNativeHandler(mlir::Operation* op) {
    if (!op || EnvFlagEnabled("JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS")) {
        return nullptr;
    }
    if (!ShouldRouteToIntrinsicNative(op)) {
        return nullptr;
    }
    IntrinsicNativeKind simpleKind = ClassifySimpleKind(op);
    if (simpleKind == IntrinsicNativeKind::kReshapeAlias && IsEligibleReshapeAliasOp(op)) {
        return NativeIntrinsicReshapeAliasHandler;
    }
    if (simpleKind == IntrinsicNativeKind::kConstantScalar && IsEligibleConstantScalarOp(op)) {
        return NativeIntrinsicConstantHandler;
    }
    if (simpleKind == IntrinsicNativeKind::kIota1D && IsEligibleIotaNativeOp(op)) {
        return NativeIntrinsicIotaHandler;
    }
    if (simpleKind == IntrinsicNativeKind::kPad2DNoInterior && IsEligiblePadNativeOp(op)) {
        return NativeIntrinsicPadHandler;
    }
    if (simpleKind == IntrinsicNativeKind::kScatterAddRowsF32 && IsEligibleScatterAddRowsNativeOp(op)) {
        return NativeIntrinsicScatterAddRowsHandler;
    }
    IntrinsicNativeKind elemKind = ClassifyElementwiseKind(op);
    if (elemKind != IntrinsicNativeKind::kNone && IsEligibleElementwiseNativeOp(op, elemKind)) {
        return NativeIntrinsicElementwiseHandler;
    }
    IntrinsicNativeKind reduceKind = ClassifyReduceKind(op);
    if (reduceKind != IntrinsicNativeKind::kNone && IsEligibleReduceNativeOp(op, reduceKind)) {
        return NativeIntrinsicReduceHandler;
    }
    IntrinsicNativeKind convertKind = ClassifyConvertKind(op);
    if (convertKind != IntrinsicNativeKind::kNone && IsEligibleConvertNativeOp(op)) {
        return NativeIntrinsicConvertHandler;
    }
    return nullptr;
}

static bool ShouldInlineForIntrinsicNative(mlir::Operation* op) {
    NativeOpHandler h = TrySelectIntrinsicNativeHandler(op);
    if (!h) {
        return false;
    }

    return h != NativeIntrinsicConstantHandler && h != NativeIntrinsicReshapeAliasHandler;
}

static size_t RoundUpToPow2(size_t v) {
    if (v <= 256)
        return 256;
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

struct ReusableOutputKey {
    void* device = nullptr;
    size_t bucket_size = 0;

    bool operator==(const ReusableOutputKey& other) const {
        return device == other.device && bucket_size == other.bucket_size;
    }
};

struct ReusableOutputKeyHash {
    size_t operator()(const ReusableOutputKey& k) const {
        size_t h1 = std::hash<void*>{}(k.device);
        size_t h2 = std::hash<size_t>{}(k.bucket_size);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
    }
};

struct ReusableOutputPool {
    std::mutex mu;
    std::unordered_map<ReusableOutputKey, std::vector<id<MTLBuffer>>, ReusableOutputKeyHash> bins;
};

static ReusableOutputPool& GetReusableOutputPool() {
    static ReusableOutputPool pool;
    return pool;
}

static id<MTLBuffer> AcquireReusableOutputBuffer(id<MTLDevice> device, size_t byte_size) {
    if (!device || byte_size == 0) {
        return nil;
    }

    const size_t bucket = RoundUpToPow2(byte_size);
    ReusableOutputPool& pool = GetReusableOutputPool();
    ReusableOutputKey key{(__bridge void*)device, bucket};
    {
        std::scoped_lock lock(pool.mu);
        auto it = pool.bins.find(key);
        if (it != pool.bins.end() && !it->second.empty()) {
            id<MTLBuffer> buf = it->second.back();
            it->second.pop_back();
            return buf;
        }
    }

    return [device newBufferWithLength:bucket options:ComputeBufferResourceOptions()];
}

static void RecycleReusableOutputBuffer(void* device_key, size_t byte_size, void* metal_buffer) {
    if (!device_key || byte_size == 0 || !metal_buffer) {
        if (metal_buffer) {
            CFRelease((__bridge CFTypeRef)metal_buffer);
        }
        return;
    }

    const size_t bucket = RoundUpToPow2(byte_size);
    ReusableOutputPool& pool = GetReusableOutputPool();
    ReusableOutputKey key{device_key, bucket};
    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)metal_buffer;

    std::scoped_lock lock(pool.mu);
    auto& bin = pool.bins[key];

    if (bin.size() >= 16) {
        CFRelease((__bridge CFTypeRef)metal_buffer);
        return;
    }
    bin.push_back(buf);
}

static id<MTLBuffer> BlitCopyToNewBuffer(id<MTLDevice> device, id<MTLCommandQueue> commandQueue,
                                         id<MTLBuffer> src, size_t byte_size, MTLResourceOptions dst_options,
                                         std::string* error) {
    if (!device || !commandQueue || !src) {
        if (error) {
            *error = "BlitCopyToNewBuffer: missing device/queue/src";
        }
        return nil;
    }
    if (byte_size == 0) {
        if (error) {
            *error = "BlitCopyToNewBuffer: zero byte_size";
        }
        return nil;
    }

    id<MTLBuffer> dst = [device newBufferWithLength:byte_size options:dst_options];
    if (!dst) {
        if (error) {
            *error = "BlitCopyToNewBuffer: failed to allocate destination buffer";
        }
        return nil;
    }

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    if (!cmd) {
        if (error) {
            *error = "BlitCopyToNewBuffer: failed to create command buffer";
        }
        CFRelease((__bridge CFTypeRef)dst);
        return nil;
    }

    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!blit) {
        if (error) {
            *error = "BlitCopyToNewBuffer: failed to create blit encoder";
        }
        CFRelease((__bridge CFTypeRef)dst);
        return nil;
    }

    [blit copyFromBuffer:src sourceOffset:0 toBuffer:dst destinationOffset:0 size:byte_size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
        NSString* desc = cmd.error ? cmd.error.localizedDescription : @"unknown error";
        if (error) {
            *error = "BlitCopyToNewBuffer: command buffer error: " + std::string([desc UTF8String]);
        }
        CFRelease((__bridge CFTypeRef)dst);
        return nil;
    }

    return dst;
}

static size_t ByteSizeFromTensorShapeAndType(NSArray<NSNumber*>* shape, MPSDataType dtype) {
    if (!shape) {
        return 0;
    }
    size_t elem = 0;
    switch (dtype) {
    case MPSDataTypeBool:
    case MPSDataTypeInt8:
    case MPSDataTypeUInt8:
        elem = 1;
        break;
    case MPSDataTypeFloat16:
    case MPSDataTypeBFloat16:
    case MPSDataTypeInt16:
    case MPSDataTypeUInt16:
        elem = 2;
        break;
    case MPSDataTypeFloat32:
    case MPSDataTypeInt32:
    case MPSDataTypeUInt32:
        elem = 4;
        break;
    case MPSDataTypeInt64:
    case MPSDataTypeUInt64:
        elem = 8;
        break;
    default:
        return 0;
    }
    size_t total = elem;
    for (NSNumber* dim_num in shape) {
        long long dim = dim_num.longLongValue;
        if (dim < 0) {
            return 0;
        }
        size_t dim_sz = static_cast<size_t>(dim);
        if (dim_sz != 0 && total > (std::numeric_limits<size_t>::max() / dim_sz)) {
            return 0;
        }
        total *= dim_sz;
    }
    return total;
}

static void MarkCompletionFromCommandBuffer(const CompletionEvent& completion,
                                            id<MTLCommandBuffer> completed) {
    if (!completion) {
        return;
    }
    if (!completed || completed.status == MTLCommandBufferStatusCompleted) {
        MarkCompletionEventReady(completion);
        return;
    }
    NSString* desc = completed.error ? completed.error.localizedDescription : @"unknown error";
    MarkCompletionEventError(completion, "Metal command buffer error: " + std::string([desc UTF8String]));
}

static CompletionEvent CompletionEventForCommandBuffer(id<MTLCommandBuffer> cmd_buf, bool async_completion) {
    if (!cmd_buf) {
        return MakeReadyCompletionEvent();
    }
    CompletionEvent completion = MakePendingCompletionEvent();
    if (!async_completion) {
        return completion;
    }
    CompletionEvent captured = completion;
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      MarkCompletionFromCommandBuffer(captured, completed);
    }];
    return completion;
}

static MPSGraphTensorData* GetTensorDataForSlot(IntermediateCacheSet* cache, SlotId slot,
                                                id<MTLBuffer> buffer, NSArray<NSNumber*>* shape,
                                                MPSDataType dtype, bool* borrowed_from_cache) {
    if (borrowed_from_cache) {
        *borrowed_from_cache = false;
    }
    if (!buffer || !shape || dtype == MPSDataTypeInvalid) {
        return nil;
    }

    size_t required_bytes = ByteSizeFromTensorShapeAndType(shape, dtype);
    if (required_bytes > 0 && [buffer length] < required_bytes) {
        MPS_LOG_ERROR("Slot %d buffer too small: have=%zu need=%zu dtype=%d shape=%s\n", slot,
                      (size_t)[buffer length], required_bytes, (int)dtype, [[shape description] UTF8String]);
        return nil;
    }

    if (cache && slot >= 0 && (size_t)slot < cache->tensor_datas.size() &&
        (size_t)slot < cache->tensor_data_bufs.size()) {
        size_t idx = (size_t)slot;
        if (cache->tensor_datas[idx] && cache->tensor_data_bufs[idx] == buffer) {
            if (borrowed_from_cache) {
                *borrowed_from_cache = true;
            }
            return cache->tensor_datas[idx];
        }

        if (cache->tensor_datas[idx]) {
            [cache->tensor_datas[idx] release];
            cache->tensor_datas[idx] = nil;
        }
        if (cache->tensor_data_bufs[idx]) {
            CFRelease((__bridge CFTypeRef)cache->tensor_data_bufs[idx]);
            cache->tensor_data_bufs[idx] = nil;
        }

        MPSGraphTensorData* td = [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer
                                                                         shape:shape
                                                                      dataType:dtype];
        if (!td) {
            return nil;
        }
        cache->tensor_datas[idx] = td;
        cache->tensor_data_bufs[idx] = buffer;
        CFRetain((__bridge CFTypeRef)buffer);
        if (borrowed_from_cache) {
            *borrowed_from_cache = true;
        }
        return td;
    }

    return [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:shape dataType:dtype];
}

static std::string JsonEscape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (unsigned char c : s) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '"':
            out += "\\\"";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 0x20) {
                char buf[7];
                std::snprintf(buf, sizeof(buf), "\\u%04x", (unsigned)c);
                out += buf;
            } else {
                out += (char)c;
            }
        }
    }
    return out;
}

static void EmitPlanJsonLine(const std::string& json) {
    std::fprintf(stderr, "JAX_METALLIB_PLAN_JSON:%s\n", json.c_str());
}

} // namespace

MpsExecutable::MpsExecutable(MpsClient* client, mps::ParsedModule module) : client_(client), name_("main") {
    if (!module.ok()) {
        error_ = "Invalid ParsedModule";
        valid_ = false;
        return;
    }

    context_ = std::move(module.context);
    module_ = std::move(module.module);
    entry_func_ = module.entry_func;

    name_ = entry_func_.getName().str();

    num_outputs_ = static_cast<int>(entry_func_.getNumResults());

    valid_ = true;
}

MpsExecutable::~MpsExecutable() {

    if (plan_) {

        for (auto& gs : plan_->graph_steps) {
            if (gs.graph) {
                CFRelease(gs.graph);
            }
            if (gs.executable) {
                CFRelease(gs.executable);
            }
        }

        for (auto& slot : plan_->slots) {
            if (slot.shape) {
                CFRelease(slot.shape);
            }
        }
        for (auto& ns : plan_->native_steps) {
            if (ns.baked_constant_buffer) {
                CFRelease((__bridge CFTypeRef)ns.baked_constant_buffer);
                ns.baked_constant_buffer = nil;
            }
        }
    }
}

std::string MpsExecutable::Serialize() const {
    if (!module_ || !context_) {
        return "";
    }

    std::string result;
    llvm::raw_string_ostream os(result);

    std::string version = mlir::vhlo::Version::getCurrentVersion().toString();
    auto status = mlir::stablehlo::serializePortableArtifact(*module_, version, os);
    if (mlir::failed(status)) {
        return "";
    }

    os.flush();
    return result;
}

std::string MpsExecutable::SerializeToText() const {
    if (!module_ || !context_) {
        return "";
    }

    std::string result;
    llvm::raw_string_ostream os(result);

    mlir::ModuleOp mod = const_cast<mlir::OwningOpRef<mlir::ModuleOp>&>(module_).get();
    mod.print(os);
    os.flush();
    return result;
}

static NSArray<NSNumber*>* GetShapeFromType(mlir::Type type) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType) {
        return nil;
    }

    NSMutableArray<NSNumber*>* shape = [NSMutableArray array];
    for (int64_t dim : tensorType.getShape()) {
        [shape addObject:@(dim)];
    }
    return shape;
}

static size_t ByteSizeFromType(mlir::Type type) {
    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(type);
    if (!tensorType)
        return 0;
    int pjrt_dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    size_t total = elem_size;
    for (int64_t dim : tensorType.getShape()) {
        total *= dim;
    }
    return total;
}

static const OpHandler* FindHandler(mlir::Operation* op) {
    std::string op_name = op->getName().getStringRef().str();
    return OpRegistry::Find(op_name);
}

static ProcessResult processOperations(HandlerContext& ctx, mlir::Block& block);

static ProcessResult processOperationsAdapter(HandlerContext& ctx, mlir::Block& block) {
    return processOperations(ctx, block);
}

static ProcessResult processCallOp(HandlerContext& ctx, mlir::func::CallOp callOp) {
    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum call depth exceeded - possible recursive function");
    }

    auto callee = ctx.module.lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
    if (!callee) {
        return ProcessResult::Error("Could not find callee function: " + callOp.getCallee().str());
    }

    if (callee.empty()) {
        return ProcessResult::Error("Callee function has no body: " + callOp.getCallee().str());
    }

    auto& calleeBlock = callee.front();
    for (size_t i = 0; i < callOp.getNumOperands() && i < calleeBlock.getNumArguments(); i++) {
        mlir::Value callArg = callOp.getOperand(i);
        mlir::BlockArgument funcArg = calleeBlock.getArgument(i);

        MPSGraphTensor* argTensor = GetTensor(ctx.values, callArg);
        if (!argTensor) {
            return ProcessResult::Error("Call argument " + std::to_string(i) + " not found");
        }

        ctx.values[funcArg.getAsOpaquePointer()] = argTensor;

        if (ctx.variables) {
            auto it = ctx.variables->find(callArg.getAsOpaquePointer());
            if (it != ctx.variables->end()) {
                (*ctx.variables)[funcArg.getAsOpaquePointer()] = it->second;
            }
        }
    }

    HandlerContext calleeCtx(ctx.graph, nullptr, ctx.values, ctx.module, ctx.depth + 1,
                             processOperationsAdapter);
    calleeCtx.variables = ctx.variables;
    ProcessResult calleeResult = processOperations(calleeCtx, calleeBlock);
    if (!calleeResult.ok()) {
        return calleeResult;
    }

    for (size_t i = 0; i < callOp.getNumResults() && i < calleeResult.return_values.size(); i++) {
        mlir::Value returnValue = calleeResult.return_values[i];
        MPSGraphTensor* returnTensor = GetTensor(ctx.values, returnValue);
        if (!returnTensor) {
            return ProcessResult::Error("Return value " + std::to_string(i) + " not found in callee");
        }
        ctx.values[callOp.getResult(i).getAsOpaquePointer()] = returnTensor;

        if (ctx.variables) {
            auto it = ctx.variables->find(returnValue.getAsOpaquePointer());
            if (it != ctx.variables->end()) {
                (*ctx.variables)[callOp.getResult(i).getAsOpaquePointer()] = it->second;
            }
        }
    }

    return ProcessResult{};
}

static ProcessResult processOperations(HandlerContext& ctx, mlir::Block& block) {
    ProcessResult result;

    for (mlir::Operation& operation : block) {
        mlir::Operation* op = &operation;
        std::string op_name = op->getName().getStringRef().str();
        MPS_LOG_DEBUG("Processing op: %s (results=%u)\n", op_name.c_str(), op->getNumResults());

        if (mlir::isa<mlir::func::ReturnOp>(op) || mlir::isa<mlir::stablehlo::ReturnOp>(op)) {
            for (mlir::Value operand : op->getOperands()) {
                result.return_values.push_back(operand);
            }
            continue;
        }

        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
            ProcessResult callResult = processCallOp(ctx, callOp);
            if (!callResult.ok()) {
                return callResult;
            }
            continue;
        }

        const OpHandler* handler = FindHandler(op);
        if (!handler) {
            return ProcessResult::Error(UnsupportedOpsMessage({op_name}) +
                                        "\n\n"
                                        "Supported operations: " +
                                        OpRegistry::ListRegistered());
        }

        if (handler->is_native()) {
            return ProcessResult::Error("Native op '" + op_name +
                                        "' encountered during graph building - "
                                        "this should have been handled by segmented execution");
        }

        HandlerContext opCtx(ctx.graph, op, ctx.values, ctx.module, ctx.depth, processOperationsAdapter);
        opCtx.variables = ctx.variables;
        ProcessResult opResult = handler->graph_handler(opCtx);
        if (!opResult.ok())
            return opResult;
    }

    return result;
}

bool MpsExecutable::BuildExecutionPlan() {
    MPS_LOG_DEBUG("BuildExecutionPlan: start, plan_built_=%d\n", plan_built_);

    std::scoped_lock lock(plan_mutex_);
    if (plan_built_)
        return true;

    @autoreleasepool {
        MPS_LOG_DEBUG("BuildExecutionPlan: in autoreleasepool\n");
        if (!client_) {
            error_ = "No MPS client available";
            return false;
        }

        const bool trace_plan = EnvFlagEnabled("JAX_METALLIB_TRACE_PLAN");
        const bool disable_executable = EnvFlagEnabled("JAX_METALLIB_DISABLE_MPSGRAPH_EXECUTABLE");

        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)client_->metal_device();
        if (!mtl_device) {
            error_ = "No Metal GPU device available";
            return false;
        }
        MPSGraphDevice* graphDevice = [MPSGraphDevice deviceWithMTLDevice:mtl_device];

        MPS_LOG_DEBUG("BuildExecutionPlan: starting inline pass\n");

        {
            bool changed = true;
            while (changed) {
                changed = false;
                mlir::Block& block = entry_func_.front();

                for (auto it = block.begin(), end = block.end(); it != end; ++it) {
                    auto callOp = mlir::dyn_cast<mlir::func::CallOp>(&*it);
                    if (!callOp)
                        continue;
                    MPS_LOG_DEBUG("BuildExecutionPlan: found call to %s\n", callOp.getCallee().str().c_str());

                    auto callee = module_->lookupSymbol<mlir::func::FuncOp>(callOp.getCallee());
                    if (!callee || callee.empty())
                        continue;

                    bool has_native = false;
                    callee.walk([&](mlir::Operation* inner) {
                        if (has_native)
                            return;
                        std::string inner_name = inner->getName().getStringRef().str();
                        const OpHandler* h = OpRegistry::Find(inner_name);
                        if ((h && h->is_native()) || ShouldInlineForIntrinsicNative(inner))
                            has_native = true;
                    });
                    if (!has_native)
                        continue;

                    mlir::Block& calleeBlock = callee.front();
                    mlir::IRMapping mapping;
                    for (unsigned i = 0; i < calleeBlock.getNumArguments(); i++) {
                        mapping.map(calleeBlock.getArgument(i), callOp.getOperand(i));
                    }

                    mlir::OpBuilder builder(callOp);
                    for (auto& op : calleeBlock) {
                        if (auto retOp = mlir::dyn_cast<mlir::func::ReturnOp>(&op)) {

                            for (unsigned i = 0; i < callOp.getNumResults(); i++) {
                                callOp.getResult(i).replaceAllUsesWith(mapping.lookup(retOp.getOperand(i)));
                            }
                            continue;
                        }
                        builder.clone(op, mapping);
                    }
                    callOp.erase();
                    changed = true;
                    break;
                }
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: inline pass complete\n");

        auto plan = std::make_unique<ExecutionPlan>();

        mlir::Block& block = entry_func_.front();
        MPS_LOG_DEBUG("BuildExecutionPlan: got entry block with %zu ops\n",
                      std::distance(block.begin(), block.end()));

        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 1 (segmentation)\n");
        struct SegmentInfo {
            bool is_native;
            std::vector<mlir::Operation*> ops;
            NativeOpHandler native_handler = nullptr;
        };

        std::vector<SegmentInfo> segments;
        mlir::func::ReturnOp returnOp = nullptr;

        auto build_segments = [&](bool allow_intrinsic_native) -> bool {
            segments.clear();
            SegmentInfo current_segment{false, {}, nullptr};
            returnOp = nullptr;

            for (mlir::Operation& operation : block) {
                mlir::Operation* op = &operation;

                if (auto ret = mlir::dyn_cast<mlir::func::ReturnOp>(op)) {
                    returnOp = ret;
                    continue;
                }

                std::string op_name = op->getName().getStringRef().str();
                const OpHandler* handler = OpRegistry::Find(op_name);
                NativeOpHandler native_handler = nullptr;
                if (handler && handler->is_native()) {
                    native_handler = handler->native_handler;
                } else if (allow_intrinsic_native) {
                    native_handler = TrySelectIntrinsicNativeHandler(op);
                }
                bool is_native = (native_handler != nullptr);

                if (is_native) {

                    if (op->getNumResults() != 1) {
                        error_ = "Native op '" + op_name + "' must have exactly one result";
                        return false;
                    }

                    if (!current_segment.ops.empty()) {
                        segments.push_back(std::move(current_segment));
                        current_segment = {false, {}, nullptr};
                    }

                    segments.push_back({true, {op}, native_handler});
                } else {
                    current_segment.ops.push_back(op);
                }
            }
            if (!current_segment.ops.empty()) {
                segments.push_back(std::move(current_segment));
            }
            return true;
        };

        if (!build_segments(true)) {
            return false;
        }

        const int mixed_intrinsic_segment_threshold =
            EnvIntValue("JAX_METALLIB_MIXED_INTRINSIC_SEGMENT_THRESHOLD", 1);
        const bool keep_mixed_intrinsics = EnvFlagEnabled("JAX_METALLIB_ENABLE_MIXED_INTRINSIC_NATIVE");
        bool has_graph_segment = false;
        bool has_native_segment = false;
        bool has_explicit_native_segment = false;
        for (const auto& seg : segments) {
            if (!seg.is_native) {
                has_graph_segment = true;
                continue;
            }
            has_native_segment = true;
            if (!seg.ops.empty()) {
                const std::string native_op_name = seg.ops[0]->getName().getStringRef().str();
                const OpHandler* native_handler = OpRegistry::Find(native_op_name);
                if (native_handler && native_handler->is_native()) {
                    has_explicit_native_segment = true;
                }
            }
        }
        if (!keep_mixed_intrinsics && !has_explicit_native_segment && has_graph_segment &&
            has_native_segment && (int)segments.size() > mixed_intrinsic_segment_threshold) {
            MPS_LOG_DEBUG("BuildExecutionPlan: fallback to graph-only intrinsics (%zu mixed segments)\n",
                          segments.size());
            if (!build_segments(false)) {
                return false;
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: Pass 1 complete, %zu segments\n", segments.size());

        if (trace_plan) {
            std::ostringstream oss;
            oss << "{\"event\":\"segments\",\"num_segments\":" << segments.size() << ",\"segments\":[";
            for (size_t i = 0; i < segments.size(); ++i) {
                if (i > 0) {
                    oss << ",";
                }
                const auto& seg = segments[i];
                if (seg.is_native) {
                    std::string op_name =
                        seg.ops.empty() ? "<missing>" : seg.ops[0]->getName().getStringRef().str();
                    oss << "{\"kind\":\"native\",\"op\":\"" << JsonEscape(op_name) << "\"}";
                    continue;
                }

                size_t op_count = 0;
                std::map<std::string, size_t> hist;
                for (auto* op : seg.ops) {
                    op->walk([&](mlir::Operation* nested) {
                        ++op_count;
                        hist[nested->getName().getStringRef().str()]++;
                    });
                }
                oss << "{\"kind\":\"graph\",\"ops\":" << op_count << ",\"op_hist\":{";
                bool first = true;
                for (const auto& kv : hist) {
                    if (!first) {
                        oss << ",";
                    }
                    first = false;
                    oss << "\"" << JsonEscape(kv.first) << "\":" << kv.second;
                }
                oss << "}}";
            }
            oss << "]}";
            EmitPlanJsonLine(oss.str());
        }

        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 1b (slot allocation)\n");

        std::unordered_map<mlir::Operation*, int> op_to_seg;
        for (int i = 0; i < (int)segments.size(); i++) {
            for (auto* op : segments[i].ops) {
                op_to_seg[op] = i;
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: op_to_seg map built\n");

        std::unordered_map<void*, SlotId> value_to_slot;
        auto getOrCreateSlot = [&](mlir::Value value) -> SlotId {
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot called\n");
            void* key = value.getAsOpaquePointer();
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot key=%p\n", key);
            auto it = value_to_slot.find(key);
            if (it != value_to_slot.end()) {
                MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot found existing slot %d\n", it->second);
                return it->second;
            }

            SlotId slot = (SlotId)plan->slots.size();
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot creating new slot %d\n", slot);
            value_to_slot[key] = slot;

            SlotInfo info;
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot getting shape\n");
            NSArray<NSNumber*>* shape = GetShapeFromType(value.getType());
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot shape=%p\n", (void*)shape);
            info.shape = (void*)CFBridgingRetain(shape);
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot getting dtype\n");
            auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
            info.dtype = tensorType ? MlirTypeToMps(tensorType.getElementType()) : MPSDataTypeFloat32;
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot dtype=%d\n", (int)info.dtype);
            info.byte_size = ByteSizeFromType(value.getType());
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot byte_size=%zu\n", info.byte_size);
            plan->slots.push_back(info);
            MPS_LOG_DEBUG("BuildExecutionPlan: getOrCreateSlot done, slot=%d\n", slot);
            return slot;
        };

        MPS_LOG_DEBUG("BuildExecutionPlan: allocating slots for %u function arguments\n",
                      entry_func_.getNumArguments());

        for (unsigned i = 0; i < entry_func_.getNumArguments(); i++) {
            MPS_LOG_DEBUG("BuildExecutionPlan: arg %u\n", i);
            auto arg = entry_func_.getArgument(i);
            SlotId slot = getOrCreateSlot(arg);
            plan->input_slots.push_back(slot);
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: %zu input slots\n", plan->input_slots.size());

        MPS_LOG_DEBUG("BuildExecutionPlan: allocating return value slots\n");
        if (returnOp) {
            MPS_LOG_DEBUG("BuildExecutionPlan: returnOp has %u operands\n", returnOp->getNumOperands());
            for (mlir::Value operand : returnOp->getOperands()) {
                SlotId slot = getOrCreateSlot(operand);
                plan->output_slots.push_back(slot);
                plan->return_types.push_back(operand.getType());
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: %zu output slots\n", plan->output_slots.size());

        MPS_LOG_DEBUG("BuildExecutionPlan: checking segment boundary crossings\n");
        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            MPS_LOG_DEBUG("BuildExecutionPlan: segment %d, %zu ops\n", seg_idx, segments[seg_idx].ops.size());
            for (auto* op : segments[seg_idx].ops) {

                for (mlir::Value operand : op->getOperands()) {
                    auto* defOp = operand.getDefiningOp();
                    if (!defOp)
                        continue;
                    auto it = op_to_seg.find(defOp);
                    if (it == op_to_seg.end() || it->second != seg_idx) {
                        getOrCreateSlot(operand);
                    }
                }
                MPS_LOG_DEBUG("BuildExecutionPlan:   operands done, checking results\n");

                for (mlir::Value result : op->getResults()) {
                    for (auto* user : result.getUsers()) {
                        if (mlir::isa<mlir::func::ReturnOp>(user)) {
                            getOrCreateSlot(result);
                            break;
                        }
                        auto it = op_to_seg.find(user);
                        if (it == op_to_seg.end() || it->second != seg_idx) {
                            getOrCreateSlot(result);
                            break;
                        }
                    }
                }
                MPS_LOG_DEBUG("BuildExecutionPlan:   results done\n");
            }
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: segment boundary crossings done, %zu slots\n", plan->slots.size());

        if (trace_plan) {
            size_t total = 0;
            size_t max_slot = 0;
            for (const auto& slot : plan->slots) {
                total += slot.byte_size;
                if (slot.byte_size > max_slot) {
                    max_slot = slot.byte_size;
                }
            }
            std::ostringstream oss;
            oss << "{\"event\":\"slots\",\"num_slots\":" << plan->slots.size()
                << ",\"total_slot_bytes\":" << total << ",\"max_slot_bytes\":" << max_slot << "}";
            EmitPlanJsonLine(oss.str());
        }

        MPS_LOG_DEBUG("BuildExecutionPlan: starting Pass 2 (step building)\n");

        for (int seg_idx = 0; seg_idx < (int)segments.size(); seg_idx++) {
            auto& seg = segments[seg_idx];
            MPS_LOG_DEBUG("BuildExecutionPlan: Pass 2 segment %d, is_native=%d, %zu ops\n", seg_idx,
                          seg.is_native, seg.ops.size());

            if (seg.is_native) {

                mlir::Operation* op = seg.ops[0];
                NativeOpHandler native_handler = seg.native_handler;
                if (!native_handler) {
                    std::string op_name = op->getName().getStringRef().str();
                    const OpHandler* handler = OpRegistry::Find(op_name);
                    if (!handler || !handler->is_native() || !handler->native_handler) {
                        error_ = "Native segment missing handler for op '" + op_name + "'";
                        return false;
                    }
                    native_handler = handler->native_handler;
                }

                NativeStep ns;
                ns.handler = native_handler;
                ns.op = op;
                for (mlir::Value operand : op->getOperands()) {
                    ns.input_slots.push_back(value_to_slot[operand.getAsOpaquePointer()]);
                }

                ns.output_slot = value_to_slot[op->getResult(0).getAsOpaquePointer()];

                (void)BakeNativeStep(mtl_device, &ns);

                if (ns.baked_kind == NativeStep::BakedKind::kScatterAddRows) {
                    plan->needs_retained_cmd_bufs = true;
                }

                plan->native_steps.push_back(std::move(ns));
                plan->steps.push_back({Step::NATIVE, plan->native_steps.size() - 1});
            } else {

                MPS_LOG_DEBUG("BuildExecutionPlan: creating MPSGraph\n");
                MPSGraph* graph = [[MPSGraph alloc] init];
                if (!graph) {
                    error_ = "Failed to create MPSGraph";
                    return false;
                }

                graph.options = MPSGraphOptionsNone;
                MPS_LOG_DEBUG("BuildExecutionPlan: MPSGraph created at %p\n", (void*)graph);
                ValueMap values;
                GraphStep gs;
                gs.graph = (void*)CFBridgingRetain(graph);
                MPS_LOG_DEBUG("BuildExecutionPlan: gs.graph set to %p\n", gs.graph);

                MPS_LOG_DEBUG("BuildExecutionPlan: collecting all ops in segment\n");
                std::unordered_set<mlir::Operation*> all_seg_ops;
                for (auto* op : seg.ops) {
                    all_seg_ops.insert(op);
                    op->walk([&](mlir::Operation* nestedOp) { all_seg_ops.insert(nestedOp); });
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: collected %zu ops in segment\n", all_seg_ops.size());

                std::unordered_set<void*> created_ph;

                auto maybeCreatePlaceholder = [&](mlir::Value operand) {
                    if (auto blockArg = mlir::dyn_cast<mlir::BlockArgument>(operand)) {
                        if (blockArg.getOwner() != &entry_func_.front()) {
                            return;
                        }
                    }

                    mlir::Operation* defOp = operand.getDefiningOp();

                    if (defOp && all_seg_ops.count(defOp)) {
                        return;
                    }

                    void* key = operand.getAsOpaquePointer();
                    if (created_ph.count(key)) {
                        return;
                    }

                    NSArray<NSNumber*>* shape = GetShapeFromType(operand.getType());
                    auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(operand.getType());
                    MPSDataType mps_dtype =
                        tensorType ? MlirTypeToMps(tensorType.getElementType()) : MPSDataTypeFloat32;

                    MPSGraphTensor* placeholder = [graph placeholderWithShape:shape
                                                                     dataType:mps_dtype
                                                                         name:nil];
                    values[key] = placeholder;
                    created_ph.insert(key);

                    auto slot_it = value_to_slot.find(key);
                    if (slot_it != value_to_slot.end()) {
                        gs.feeds.emplace_back(slot_it->second, (__bridge void*)placeholder);
                    }
                };

                MPS_LOG_DEBUG("BuildExecutionPlan: creating placeholders\n");
                for (auto* op : seg.ops) {
                    op->walk([&](mlir::Operation* nestedOp) {
                        for (mlir::Value operand : nestedOp->getOperands()) {
                            maybeCreatePlaceholder(operand);
                        }
                    });
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: created %zu placeholders\n", created_ph.size());

                MPS_LOG_DEBUG("BuildExecutionPlan: processing ops in segment\n");
                @try {
                    for (auto* op : seg.ops) {
                        std::string op_name = op->getName().getStringRef().str();

                        if (auto callOp = mlir::dyn_cast<mlir::func::CallOp>(op)) {
                            HandlerContext callCtx(graph, op, values, *module_, 0, processOperationsAdapter);
                            ProcessResult callResult = processCallOp(callCtx, callOp);
                            if (!callResult.ok()) {
                                error_ = callResult.error;
                                return false;
                            }
                            continue;
                        }

                        const OpHandler* handler = FindHandler(op);
                        if (!handler) {
                            error_ = UnsupportedOpsMessage({op_name}) +
                                     "\n\nSupported operations: " + OpRegistry::ListRegistered();
                            return false;
                        }

                        HandlerContext opCtx(graph, op, values, *module_, 0, processOperationsAdapter);
                        ProcessResult opResult = handler->graph_handler(opCtx);
                        if (!opResult.ok()) {
                            error_ = opResult.error;
                            return false;
                        }
                    }
                } @catch (NSException* exception) {
                    error_ = "MPS graph build failed: " + std::string([[exception reason] UTF8String]);
                    return false;
                }

                MPS_LOG_DEBUG("BuildExecutionPlan: identifying target values\n");
                for (auto* op : seg.ops) {
                    for (mlir::Value result : op->getResults()) {
                        void* key = result.getAsOpaquePointer();
                        if (value_to_slot.count(key)) {
                            SlotId slot = value_to_slot[key];
                            MPSGraphTensor* tensor = values[key];
                            MPS_LOG_DEBUG("BuildExecutionPlan: target slot=%d, key=%p, tensor=%p\n", slot,
                                          key, (void*)tensor);
                            if (tensor) {
                                MPS_LOG_DEBUG("BuildExecutionPlan: tensor shape=%s, dtype=%d\n",
                                              [[tensor.shape description] UTF8String], (int)tensor.dataType);
                                gs.targets.emplace_back(slot, (__bridge void*)tensor);
                            } else {
                                MPS_LOG_DEBUG("BuildExecutionPlan: WARNING tensor is nil!\n");
                            }
                        }
                    }
                }
                MPS_LOG_DEBUG("BuildExecutionPlan: identified %zu targets\n", gs.targets.size());

                if (!disable_executable && graphDevice && !gs.targets.empty()) {
                    @try {
                        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feedTypes =
                            [NSMutableDictionary dictionaryWithCapacity:gs.feeds.size()];
                        for (const auto& [slot, tensor_ptr] : gs.feeds) {
                            (void)slot;
                            MPSGraphTensor* t = (__bridge MPSGraphTensor*)tensor_ptr;
                            if (!t) {
                                continue;
                            }
                            MPSGraphShapedType* st = [[MPSGraphShapedType alloc] initWithShape:t.shape
                                                                                      dataType:t.dataType];
                            if (st) {
                                feedTypes[t] = st;
                                [st release];
                            }
                        }

                        NSMutableArray<MPSGraphTensor*>* targetTensors =
                            [NSMutableArray arrayWithCapacity:gs.targets.size()];
                        for (const auto& [slot, tensor_ptr] : gs.targets) {
                            (void)slot;
                            MPSGraphTensor* t = (__bridge MPSGraphTensor*)tensor_ptr;
                            if (t) {
                                [targetTensors addObject:t];
                            }
                        }

                        MPSGraphExecutable* exec = [graph compileWithDevice:graphDevice
                                                                      feeds:feedTypes
                                                              targetTensors:targetTensors
                                                           targetOperations:nil
                                                      compilationDescriptor:nil];
                        if (exec) {
                            exec.options = MPSGraphOptionsNone;
                            gs.executable = (void*)CFBridgingRetain(exec);

                            std::unordered_map<void*, SlotId> feedTensorToSlot;
                            feedTensorToSlot.reserve(gs.feeds.size());
                            for (const auto& [slot, tensor_ptr] : gs.feeds) {
                                feedTensorToSlot[tensor_ptr] = slot;
                            }

                            NSArray<MPSGraphTensor*>* feedOrder = exec.feedTensors;
                            if (feedOrder) {
                                gs.exec_feed_slots.clear();
                                gs.exec_feed_slots.reserve(feedOrder.count);
                                bool ok = true;
                                for (MPSGraphTensor* ft in feedOrder) {
                                    auto it = feedTensorToSlot.find((__bridge void*)ft);
                                    if (it == feedTensorToSlot.end()) {
                                        ok = false;
                                        break;
                                    }
                                    gs.exec_feed_slots.push_back(it->second);
                                }
                                if (!ok) {
                                    gs.exec_feed_slots.clear();
                                }
                            }

                            std::unordered_map<void*, SlotId> targetTensorToSlot;
                            targetTensorToSlot.reserve(gs.targets.size());
                            for (const auto& [slot, tensor_ptr] : gs.targets) {
                                targetTensorToSlot[tensor_ptr] = slot;
                            }

                            NSArray<MPSGraphTensor*>* targetOrder = exec.targetTensors;
                            if (targetOrder) {
                                gs.exec_target_slots.clear();
                                gs.exec_target_slots.reserve(targetOrder.count);
                                bool ok = true;
                                for (MPSGraphTensor* tt in targetOrder) {
                                    auto it = targetTensorToSlot.find((__bridge void*)tt);
                                    if (it == targetTensorToSlot.end()) {
                                        ok = false;
                                        break;
                                    }
                                    gs.exec_target_slots.push_back(it->second);
                                }
                                if (!ok) {
                                    gs.exec_target_slots.clear();
                                }
                            }
                        }
                    } @catch (NSException*) {
                    }
                }

                plan->graph_steps.push_back(gs);
                plan->steps.push_back({Step::GRAPH, plan->graph_steps.size() - 1});
            }
        }

        MPS_LOG_DEBUG("BuildExecutionPlan: setting plan_ and plan_built_=true\n");
        MPS_LOG_DEBUG("BuildExecutionPlan: %zu steps, %zu graph_steps, %zu native_steps, %zu slots\n",
                      plan->steps.size(), plan->graph_steps.size(), plan->native_steps.size(),
                      plan->slots.size());

        if (trace_plan) {
            std::ostringstream oss;
            oss << "{\"event\":\"steps\",\"num_steps\":" << plan->steps.size()
                << ",\"graph_steps\":" << plan->graph_steps.size()
                << ",\"native_steps\":" << plan->native_steps.size() << ",\"graph_step_io\":[";
            for (size_t i = 0; i < plan->graph_steps.size(); ++i) {
                if (i > 0) {
                    oss << ",";
                }
                const auto& gs = plan->graph_steps[i];
                oss << "{\"feeds\":" << gs.feeds.size() << ",\"targets\":" << gs.targets.size() << "}";
            }
            oss << "]}";
            EmitPlanJsonLine(oss.str());
        }

        static const bool enable_fusion = !EnvFlagEnabled("JAX_METALLIB_DISABLE_ELEMENTWISE_FUSION");
        MPS_LOG_DEBUG("Fusion pass: enable=%d, native_steps=%zu, steps=%zu, graph_steps=%zu\n",
                      (int)enable_fusion, plan->native_steps.size(), plan->steps.size(),
                      plan->graph_steps.size());
        if (enable_fusion && !plan->native_steps.empty()) {
            const size_t num_slots = plan->slots.size();

            std::vector<int> slot_producer_step(num_slots, -1);

            std::vector<int> slot_consumer_count(num_slots, 0);

            for (size_t si = 0; si < plan->steps.size(); ++si) {
                const auto& step = plan->steps[si];
                if (step.kind != Step::NATIVE)
                    continue;
                const auto& ns = plan->native_steps[step.index];
                if (ns.baked_kind == NativeStep::BakedKind::kElementwise) {
                    if (ns.output_slot >= 0 && (size_t)ns.output_slot < num_slots) {
                        slot_producer_step[(size_t)ns.output_slot] = (int)step.index;
                    }
                }
                for (auto slot : ns.input_slots) {
                    if (slot >= 0 && (size_t)slot < num_slots) {
                        slot_consumer_count[(size_t)slot]++;
                    }
                }
            }

            for (const auto& gs : plan->graph_steps) {
                for (const auto& [slot, _] : gs.feeds) {
                    if (slot >= 0 && (size_t)slot < num_slots) {
                        slot_consumer_count[(size_t)slot]++;
                    }
                }
            }

            std::vector<uint8_t> is_func_io(num_slots, 0);
            for (auto s : plan->input_slots) {
                if (s >= 0 && (size_t)s < num_slots)
                    is_func_io[(size_t)s] = 1;
            }
            for (auto s : plan->output_slots) {
                if (s >= 0 && (size_t)s < num_slots)
                    is_func_io[(size_t)s] = 1;
            }

            auto dtype_byte_size = [](MPSDataType d) -> size_t { return (d == MPSDataTypeFloat16) ? 2 : 4; };

            struct ChainRange {
                size_t steps_begin;
                size_t steps_end;
            };
            std::vector<ChainRange> chains;

            size_t chain_start = 0;
            bool in_chain = false;
            MPSDataType chain_dtype = MPSDataTypeInvalid;
            uint64_t chain_numel = 0;

            std::unordered_set<int> chain_ns_indices;

            auto flush_chain = [&](size_t chain_end) {
                if (in_chain && chain_end - chain_start >= 2) {
                    chains.push_back({chain_start, chain_end});
                }
                in_chain = false;
                chain_ns_indices.clear();
            };

            for (size_t si = 0; si < plan->steps.size(); ++si) {
                const auto& step = plan->steps[si];
                if (step.kind != Step::NATIVE) {
                    flush_chain(si);
                    continue;
                }
                const auto& ns = plan->native_steps[step.index];
                if (ns.baked_kind != NativeStep::BakedKind::kElementwise) {
                    flush_chain(si);
                    continue;
                }

                IntrinsicNativeKind kind = ClassifyElementwiseKind(ns.op);
                if (!IsFusibleElementwiseKind(kind)) {
                    flush_chain(si);
                    continue;
                }

                MPSDataType ns_dtype = plan->slots[(size_t)ns.output_slot].dtype;
                size_t dsize = dtype_byte_size(ns_dtype);
                uint64_t ns_numel = (dsize > 0) ? plan->slots[(size_t)ns.output_slot].byte_size / dsize : 0;

                if (!in_chain) {

                    chain_start = si;
                    in_chain = true;
                    chain_dtype = ns_dtype;
                    chain_numel = ns_numel;
                    chain_ns_indices.clear();
                    chain_ns_indices.insert((int)step.index);
                    continue;
                }

                if (ns_dtype != chain_dtype || ns_numel != chain_numel) {
                    flush_chain(si);

                    chain_start = si;
                    in_chain = true;
                    chain_dtype = ns_dtype;
                    chain_numel = ns_numel;
                    chain_ns_indices.clear();
                    chain_ns_indices.insert((int)step.index);
                    continue;
                }

                bool intermediates_ok = true;
                for (auto slot : ns.input_slots) {
                    if (slot >= 0 && (size_t)slot < num_slots) {
                        int prod = slot_producer_step[(size_t)slot];
                        if (prod >= 0 && chain_ns_indices.count(prod)) {

                            if (slot_consumer_count[(size_t)slot] != 1 || is_func_io[(size_t)slot]) {
                                intermediates_ok = false;
                                break;
                            }
                        }
                    }
                }
                if (!intermediates_ok) {
                    flush_chain(si);
                    chain_start = si;
                    in_chain = true;
                    chain_dtype = ns_dtype;
                    chain_numel = ns_numel;
                    chain_ns_indices.clear();
                    chain_ns_indices.insert((int)step.index);
                    continue;
                }

                chain_ns_indices.insert((int)step.index);
            }
            flush_chain(plan->steps.size());

            if (!chains.empty()) {

                std::vector<int> step_chain_id(plan->steps.size(), -1);
                for (size_t ci = 0; ci < chains.size(); ++ci) {
                    for (size_t si = chains[ci].steps_begin; si < chains[ci].steps_end; ++si) {
                        step_chain_id[si] = (int)ci;
                    }
                }

                std::vector<NativeStep> fused_steps;
                fused_steps.reserve(chains.size());

                for (size_t ci = 0; ci < chains.size(); ++ci) {
                    const auto& cr = chains[ci];

                    std::vector<size_t> chain_ns_idxs;
                    for (size_t si = cr.steps_begin; si < cr.steps_end; ++si) {
                        chain_ns_idxs.push_back(plan->steps[si].index);
                    }

                    std::unordered_map<SlotId, int> slot_to_chain_op;
                    for (size_t i = 0; i < chain_ns_idxs.size(); ++i) {
                        const auto& ns = plan->native_steps[chain_ns_idxs[i]];
                        slot_to_chain_op[ns.output_slot] = (int)i;
                    }

                    std::vector<FusionOp> fusion_ops;
                    std::vector<SlotId> external_input_slots;
                    std::unordered_map<SlotId, int> ext_slot_to_idx;
                    std::vector<SlotId> eliminated_slots;

                    for (size_t i = 0; i < chain_ns_idxs.size(); ++i) {
                        const auto& ns = plan->native_steps[chain_ns_idxs[i]];
                        IntrinsicNativeKind kind = ClassifyElementwiseKind(ns.op);
                        FusionOp fop;
                        fop.kind = kind;
                        for (auto slot : ns.input_slots) {
                            auto chain_it = slot_to_chain_op.find(slot);
                            if (chain_it != slot_to_chain_op.end()) {

                                fop.sources.push_back(chain_it->second);
                            } else {

                                auto ext_it = ext_slot_to_idx.find(slot);
                                if (ext_it == ext_slot_to_idx.end()) {
                                    int idx = (int)external_input_slots.size();
                                    ext_slot_to_idx[slot] = idx;
                                    external_input_slots.push_back(slot);
                                    fop.sources.push_back(-(idx + 1));
                                } else {
                                    fop.sources.push_back(-(ext_it->second + 1));
                                }
                            }
                        }
                        fusion_ops.push_back(std::move(fop));

                        if (i < chain_ns_idxs.size() - 1) {
                            eliminated_slots.push_back(ns.output_slot);
                        }
                    }

                    bool vec_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_VEC4_INTRINSICS");
                    int vec_mode = 0;
                    if (!vec_disabled) {
                        if (chain_dtype == MPSDataTypeFloat16 && chain_numel % 8 == 0) {
                            vec_mode = 8;
                        } else if (IsFloat16Or32(chain_dtype) && chain_numel % 4 == 0) {
                            vec_mode = 4;
                        }
                    }

                    std::string cache_key = "fused_";
                    cache_key += (chain_dtype == MPSDataTypeFloat16) ? "f16" : "f32";
                    if (vec_mode == 8)
                        cache_key += "_v8";
                    else if (vec_mode == 4)
                        cache_key += "_v4";
                    for (size_t i = 0; i < fusion_ops.size(); ++i) {
                        cache_key += "_";
                        cache_key += FusionKindShortName(fusion_ops[i].kind);
                        cache_key += "(";
                        for (size_t j = 0; j < fusion_ops[i].sources.size(); ++j) {
                            if (j > 0)
                                cache_key += ",";
                            cache_key += std::to_string(fusion_ops[i].sources[j]);
                        }
                        cache_key += ")";
                    }

                    std::string fn_name = cache_key;

                    for (char& c : fn_name) {
                        if (c == '(' || c == ')' || c == ',' || c == '-')
                            c = '_';
                    }

                    std::string source = GenerateFusedKernelSource(
                        fn_name, fusion_ops, (int)external_input_slots.size(), chain_dtype, vec_mode);

                    MPS_LOG_DEBUG("Fusion: chain %zu has %zu ops, %zu external inputs, "
                                  "%zu eliminated slots, vec_mode=%d, cache_key=%s\n",
                                  ci, fusion_ops.size(), external_input_slots.size(), eliminated_slots.size(),
                                  vec_mode, cache_key.c_str());

                    id<MTLComputePipelineState> pipeline =
                        GetOrCompileFusedPipeline(mtl_device, cache_key, fn_name, source);
                    if (!pipeline) {
                        MPS_LOG_WARN("Fusion: JIT compile failed for chain %zu (%s), skipping\n", ci,
                                     cache_key.c_str());

                        for (size_t si = cr.steps_begin; si < cr.steps_end; ++si) {
                            step_chain_id[si] = -1;
                        }
                        NativeStep dummy;
                        dummy.baked_kind = NativeStep::BakedKind::kNone;
                        fused_steps.push_back(std::move(dummy));
                        continue;
                    }

                    NativeStep fns;
                    fns.handler = nullptr;
                    fns.op = plan->native_steps[chain_ns_idxs[0]].op;
                    fns.input_slots = external_input_slots;
                    fns.output_slot = plan->native_steps[chain_ns_idxs.back()].output_slot;
                    fns.baked_kind = NativeStep::BakedKind::kFusedElementwise;
                    fns.baked_pipeline = pipeline;
                    uint32_t n;
                    if (vec_mode == 8)
                        n = static_cast<uint32_t>(chain_numel / 8);
                    else if (vec_mode == 4)
                        n = static_cast<uint32_t>(chain_numel / 4);
                    else
                        n = static_cast<uint32_t>(chain_numel);
                    fns.baked_n = n;
                    fns.baked_use_vec4 = (vec_mode >= 4);
                    ComputeLinearDispatchConfig(pipeline, n, &fns.baked_grid, &fns.baked_tg, chain_dtype);
                    fns.fused_info = std::make_unique<FusedElementwiseInfo>();
                    fns.fused_info->external_input_slots = external_input_slots;
                    fns.fused_info->eliminated_slots = eliminated_slots;
                    fns.fused_info->num_buffer_args = (uint32_t)external_input_slots.size() + 2;

                    fused_steps.push_back(std::move(fns));
                }

                std::vector<Step> new_steps;
                std::vector<NativeStep> new_native_steps;
                new_steps.reserve(plan->steps.size());
                new_native_steps.reserve(plan->native_steps.size());

                std::vector<size_t> old_to_new_ns(plan->native_steps.size(), SIZE_MAX);
                for (size_t si = 0; si < plan->steps.size(); ++si) {
                    const auto& step = plan->steps[si];
                    if (step_chain_id[si] >= 0) {

                        int ci = step_chain_id[si];

                        if (si == chains[(size_t)ci].steps_begin &&
                            fused_steps[(size_t)ci].baked_kind == NativeStep::BakedKind::kFusedElementwise) {
                            size_t new_idx = new_native_steps.size();
                            new_native_steps.push_back(std::move(fused_steps[(size_t)ci]));
                            new_steps.push_back({Step::NATIVE, new_idx});
                        }

                    } else if (step.kind == Step::NATIVE) {
                        size_t new_idx = new_native_steps.size();
                        old_to_new_ns[step.index] = new_idx;
                        new_native_steps.push_back(std::move(plan->native_steps[step.index]));
                        new_steps.push_back({Step::NATIVE, new_idx});
                    } else {

                        new_steps.push_back(step);
                    }
                }

                plan->steps = std::move(new_steps);
                plan->native_steps = std::move(new_native_steps);

                std::vector<uint8_t> elim(num_slots, 0);
                for (const auto& ns : plan->native_steps) {
                    if (ns.fused_info) {
                        for (auto s : ns.fused_info->eliminated_slots) {
                            if (s >= 0 && (size_t)s < num_slots) {
                                elim[(size_t)s] = 1;
                            }
                        }
                    }
                }
                plan->slot_is_eliminated = std::move(elim);

                MPS_LOG_DEBUG("BuildExecutionPlan: fused %zu chains, %zu steps -> %zu steps, "
                              "%zu native_steps -> %zu native_steps\n",
                              chains.size(), plan->steps.size() + chains.size(), plan->steps.size(),
                              plan->native_steps.size() + chains.size(), plan->native_steps.size());
            } else {
                plan->slot_is_eliminated.assign(num_slots, 0);
            }
        } else {
            plan->slot_is_eliminated.assign(plan->slots.size(), 0);
        }

        plan->slot_is_input.assign(plan->slots.size(), 0);
        for (auto s : plan->input_slots) {
            if (s >= 0 && (size_t)s < plan->slot_is_input.size()) {
                plan->slot_is_input[(size_t)s] = 1;
            }
        }
        plan->slot_is_output.assign(plan->slots.size(), 0);
        for (auto s : plan->output_slots) {
            if (s >= 0 && (size_t)s < plan->slot_is_output.size()) {
                plan->slot_is_output[(size_t)s] = 1;
            }
        }

        plan->output_meta.reserve(plan->output_slots.size());
        for (size_t i = 0; i < plan->output_slots.size(); i++) {
            ExecutionPlan::OutputMeta meta;
            auto tensorType = mlir::cast<mlir::RankedTensorType>(plan->return_types[i]);
            meta.pjrt_dtype = MlirTypeToPjrtDtype(tensorType.getElementType());
            auto shape = tensorType.getShape();
            meta.dims.assign(shape.begin(), shape.end());
            SlotId slot = plan->output_slots[i];
            meta.byte_size =
                (slot >= 0 && (size_t)slot < plan->slots.size()) ? plan->slots[(size_t)slot].byte_size : 0;
            plan->output_meta.push_back(std::move(meta));
        }

        plan_ = std::move(plan);
        plan_built_ = true;
    }
    MPS_LOG_DEBUG("BuildExecutionPlan: returning true\n");
    return true;
}

ExecutionResult MpsExecutable::Execute(const std::vector<MpsBuffer*>& inputs, MpsDevice* device) {

    if (!valid_) {
        return ExecutionResult::Error("Cannot execute: compilation failed - " + error_);
    }

    MPS_LOG_DEBUG("Execute: plan_built_=%d\n", plan_built_);
    if (!plan_built_) {
        MPS_LOG_DEBUG("Execute: building execution plan\n");
        if (!BuildExecutionPlan()) {
            return ExecutionResult::Error("Execution plan build failed: " + error_);
        }
        MPS_LOG_DEBUG("Execute: plan built successfully\n");
    }

    @autoreleasepool {
        MPS_LOG_DEBUG("Execute: entering autoreleasepool\n");

        if (!client_) {
            return ExecutionResult::Error("No MPS client available");
        }
        id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)client_->metal_device();
        if (!mtl_device) {
            return ExecutionResult::Error(
                "No Metal GPU device available. MPS backend requires Apple Silicon or AMD GPU.");
        }

        size_t num_args = plan_->input_slots.size();
        if (inputs.size() < num_args) {
            return ExecutionResult::Error("Input count mismatch: expected " + std::to_string(num_args) +
                                          " inputs, got " + std::to_string(inputs.size()));
        }

        if (plan_->steps.empty() && !plan_->output_slots.empty() && !inputs.empty()) {
            ExecutionResult result;
            for (size_t i = 0; i < plan_->output_slots.size(); i++) {
                SlotId slot = plan_->output_slots[i];

                int input_idx = -1;
                for (size_t j = 0; j < plan_->input_slots.size(); j++) {
                    if (plan_->input_slots[j] == slot) {
                        input_idx = (int)j;
                        break;
                    }
                }
                if (input_idx < 0 || !inputs[input_idx]) {
                    return ExecutionResult::Error("Identity function with unmapped output slot");
                }
                MpsBuffer* input = inputs[input_idx];
                std::string input_error;
                if (!input->WaitForProducer(&input_error)) {
                    return ExecutionResult::Error("Identity input not ready: " + input_error);
                }
                size_t byte_size = input->byte_size();
                id<MTLBuffer> src = (__bridge id<MTLBuffer>)input->metal_buffer();
                if (!src) {
                    return ExecutionResult::Error("Identity function input has no Metal buffer");
                }
                void* contents = [src contents];
                id<MTLBuffer> dst = nil;
                if (contents) {
                    dst = [mtl_device newBufferWithBytes:contents
                                                  length:byte_size
                                                 options:ComputeBufferResourceOptions()];
                } else {

                    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)client_->command_queue();
                    if (!commandQueue) {
                        return ExecutionResult::Error("No Metal command queue available");
                    }
                    std::string copy_err;
                    dst = BlitCopyToNewBuffer(mtl_device, commandQueue, src, byte_size,
                                              MTLResourceStorageModeShared, &copy_err);
                    if (!dst && !copy_err.empty()) {
                        return ExecutionResult::Error(copy_err);
                    }
                }
                if (!dst) {
                    return ExecutionResult::Error("Failed to allocate buffer for identity output");
                }

                const auto& meta = plan_->output_meta[i];
                result.buffers.push_back(
                    std::make_unique<MpsBuffer>(device, (__bridge void*)dst, meta.pjrt_dtype, meta.dims));

                CFRelease((__bridge CFTypeRef)dst);
            }
            result.completion = MakeReadyCompletionEvent();
            return result;
        }

        if (plan_->output_slots.empty()) {
            ExecutionResult result;
            result.completion = MakeReadyCompletionEvent();
            return result;
        }

        id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)client_->command_queue();
        if (!commandQueue) {
            return ExecutionResult::Error("No Metal command queue available");
        }

        std::unique_ptr<ExecutionPlan::ExecScratch> scratch;
        {
            std::scoped_lock lock(plan_->scratch_pool_mutex);
            if (!plan_->scratch_pool.empty()) {
                scratch = std::move(plan_->scratch_pool.back());
                plan_->scratch_pool.pop_back();
            }
        }
        if (!scratch) {
            scratch = std::make_unique<ExecutionPlan::ExecScratch>();
        }
        scratch->reset(plan_->slots.size());
        auto& slot_bufs = scratch->slot_bufs;
        auto& slot_producer_events = scratch->slot_producer_events;

        struct ScratchGuard {
            std::unique_ptr<ExecutionPlan::ExecScratch>& scratch_;
            ExecutionPlan* plan_;
            ~ScratchGuard() {
                if (scratch_) {
                    std::scoped_lock lock(plan_->scratch_pool_mutex);
                    plan_->scratch_pool.push_back(std::move(scratch_));
                }
            }
        } scratch_guard{scratch, plan_.get()};

        CompletionEvent execution_completion = MakeReadyCompletionEvent();

        for (size_t i = 0; i < num_args; i++) {
            MpsBuffer* input = inputs[i];
            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }
            id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)input->metal_buffer();
            if (!mtl_buf) {
                return ExecutionResult::Error("Input buffer at index " + std::to_string(i) +
                                              " has no Metal buffer");
            }
            SlotId slot = plan_->input_slots[i];
            slot_bufs[slot] = mtl_buf;
            slot_producer_events[slot] = input->producer_completion();
        }

        static const bool cache_intermediates = !EnvFlagEnabled("JAX_METALLIB_DISABLE_INTERMEDIATE_CACHE");
        static const bool enable_output_pool = !EnvFlagEnabled("JAX_METALLIB_DISABLE_OUTPUT_BUFFER_POOL");
        std::unique_ptr<IntermediateCacheSet> cache_set;
        IntermediateCacheSet* cache = nullptr;
        if (cache_intermediates) {
            std::scoped_lock lock(plan_->intermediate_pool_mutex);
            if (!plan_->intermediate_pool.empty()) {
                cache_set = std::move(plan_->intermediate_pool.back());
                plan_->intermediate_pool.pop_back();
            }
        }
        if (cache_intermediates && !cache_set) {
            cache_set = std::make_unique<IntermediateCacheSet>(plan_->slots.size());
        }
        cache = cache_set.get();

        for (const auto& gs : plan_->graph_steps) {
            for (const auto& [slot, tensor] : gs.targets) {
                if (slot_bufs[slot])
                    continue;
                size_t byte_size = plan_->slots[slot].byte_size;
                MPSGraphTensor* target_tensor = (__bridge MPSGraphTensor*)tensor;
                if (target_tensor) {
                    size_t tensor_bytes =
                        ByteSizeFromTensorShapeAndType(target_tensor.shape, target_tensor.dataType);
                    if (tensor_bytes > byte_size) {
                        byte_size = tensor_bytes;
                    }
                }
                if (byte_size == 0) {
                    return ExecutionResult::Error("Invalid (zero) slot buffer size for slot " +
                                                  std::to_string(slot));
                }

                const bool is_input = slot >= 0 && (size_t)slot < plan_->slot_is_input.size() &&
                                      plan_->slot_is_input[(size_t)slot];
                const bool is_output = slot >= 0 && (size_t)slot < plan_->slot_is_output.size() &&
                                       plan_->slot_is_output[(size_t)slot];

                id<MTLBuffer> buf = nil;
                if (cache && !is_input && !is_output && slot >= 0 && (size_t)slot < cache->bufs.size()) {
                    if (cache->bufs[(size_t)slot]) {
                        buf = cache->bufs[(size_t)slot];
                        if ([buf length] < byte_size) {
                            CFRelease((__bridge CFTypeRef)cache->bufs[(size_t)slot]);
                            cache->bufs[(size_t)slot] = nil;
                            buf = nil;
                        }
                    } else {
                        buf = [mtl_device newBufferWithLength:byte_size
                                                      options:ComputeBufferResourceOptions()];
                        if (buf) {

                            cache->bufs[(size_t)slot] = buf;
                        }
                    }
                    if (!buf) {
                        buf = [mtl_device newBufferWithLength:byte_size
                                                      options:ComputeBufferResourceOptions()];
                        if (buf) {
                            cache->bufs[(size_t)slot] = buf;
                        }
                    }
                } else {
                    if (is_output && enable_output_pool) {
                        buf = AcquireReusableOutputBuffer(mtl_device, byte_size);
                    } else {
                        buf = [mtl_device newBufferWithLength:byte_size
                                                      options:ComputeBufferResourceOptions()];
                    }
                }
                if (!buf) {
                    return ExecutionResult::Error("Failed to pre-allocate slot buffer of " +
                                                  std::to_string(byte_size) + " bytes");
                }
                slot_bufs[slot] = buf;
            }
        }

        static const bool disable_executable = EnvFlagEnabled("JAX_METALLIB_DISABLE_MPSGRAPH_EXECUTABLE");

        static const bool async_completion = !EnvFlagEnabled("JAX_METALLIB_DISABLE_ASYNC_COMPLETION");
        static const bool enable_direct_run = !EnvFlagEnabled("JAX_METALLIB_DISABLE_MPSGRAPH_DIRECT_RUN");
        bool executed_direct_run = false;

        if (enable_direct_run && plan_->steps.size() == 1 && plan_->steps[0].kind == Step::GRAPH) {
            auto& gs = plan_->graph_steps[plan_->steps[0].index];
            MPSGraphExecutable* exec = (__bridge MPSGraphExecutable*)gs.executable;
            if (exec && !disable_executable && !gs.exec_target_slots.empty()) {
                NSArray<MPSGraphTensor*>* feedOrder = exec.feedTensors;
                NSArray<MPSGraphTensor*>* targetOrder = exec.targetTensors;
                if (feedOrder && feedOrder.count == gs.exec_feed_slots.size() && targetOrder &&
                    targetOrder.count == gs.exec_target_slots.size()) {
                    NSMutableArray<MPSGraphTensorData*>* inputsArray =
                        [NSMutableArray arrayWithCapacity:gs.exec_feed_slots.size()];
                    for (NSUInteger i = 0; i < feedOrder.count; ++i) {
                        SlotId slot = gs.exec_feed_slots[i];
                        if (slot < 0 || (size_t)slot >= slot_bufs.size() || !slot_bufs[slot]) {
                            return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                          " is nil during direct-run inputs");
                        }
                        MPSGraphTensor* t = feedOrder[i];
                        bool borrowed = false;
                        MPSGraphTensorData* data = GetTensorDataForSlot(cache, slot, slot_bufs[slot], t.shape,
                                                                        t.dataType, &borrowed);
                        if (!data) {
                            return ExecutionResult::Error("Failed to create tensor data for slot " +
                                                          std::to_string(slot));
                        }
                        [inputsArray addObject:data];
                        if (!borrowed) {
                            [data release];
                        }
                    }

                    NSMutableArray<MPSGraphTensorData*>* resultsArray =
                        [NSMutableArray arrayWithCapacity:gs.exec_target_slots.size()];
                    for (NSUInteger i = 0; i < targetOrder.count; ++i) {
                        SlotId slot = gs.exec_target_slots[i];
                        if (slot < 0 || (size_t)slot >= slot_bufs.size() || !slot_bufs[slot]) {
                            return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                          " is nil during direct-run results");
                        }
                        MPSGraphTensor* t = targetOrder[i];
                        bool borrowed = false;
                        MPSGraphTensorData* out = GetTensorDataForSlot(cache, slot, slot_bufs[slot], t.shape,
                                                                       t.dataType, &borrowed);
                        if (!out) {
                            return ExecutionResult::Error("Failed to create tensor data for slot " +
                                                          std::to_string(slot));
                        }
                        [resultsArray addObject:out];
                        if (!borrowed) {
                            [out release];
                        }
                    }

                    @try {
                        [exec runWithMTLCommandQueue:commandQueue
                                         inputsArray:inputsArray
                                        resultsArray:resultsArray
                                 executionDescriptor:nil];

                        CompletionEvent ready = MakeReadyCompletionEvent();

                        executed_direct_run = true;
                        for (SlotId slot : gs.exec_target_slots) {
                            if (slot >= 0 && (size_t)slot < slot_producer_events.size()) {
                                slot_producer_events[(size_t)slot] = ready;
                            }
                        }
                        execution_completion = ready;
                    } @catch (NSException* exception) {
                        return ExecutionResult::Error("MPS executable direct run failed: " +
                                                      std::string([[exception reason] UTF8String]));
                    }
                }
            }
        }

        if (!executed_direct_run && plan_->graph_steps.empty()) {

            static const bool use_unretained_default =
                !EnvFlagEnabled("JAX_METALLIB_DISABLE_NATIVE_UNRETAINED_CMDBUF");
            const bool use_unretained = use_unretained_default && !plan_->needs_retained_cmd_bufs;
            static const int max_ops_per_cmd = EnvIntValue("JAX_METALLIB_MAX_OPS_PER_CMDBUF", 48);
            static const int max_mb_per_cmd = EnvIntValue("JAX_METALLIB_MAX_MB_PER_CMDBUF", 48);

            const bool queue_supports_unretained = [&] {
                static int cached = -1;
                if (cached < 0) {
                    cached = commandQueue && [commandQueue respondsToSelector:@selector
                                                           (commandBufferWithUnretainedReferences)]
                                 ? 1
                                 : 0;
                }
                return cached == 1;
            }();

            auto make_native_cmd_buffer = [&]() -> id<MTLCommandBuffer> {
                id<MTLCommandBuffer> cb = nil;
                if (use_unretained && queue_supports_unretained) {
                    cb = [commandQueue commandBufferWithUnretainedReferences];
                }
                if (!cb) {
                    cb = [commandQueue commandBuffer];
                }
                return cb;
            };

            id<MTLCommandBuffer> cmdBuf = nil;
            id<MTLComputeCommandEncoder> nativeComputeEncoder = nil;
            CompletionEvent current_cmd_completion;
            std::vector<CompletionEvent> submitted_events;
            submitted_events.reserve(4);
            int ops_in_cmd = 0;
            size_t bytes_in_cmd = 0;

            auto ensure_cmd_buf = [&]() -> bool {
                if (cmdBuf) {
                    return true;
                }
                cmdBuf = make_native_cmd_buffer();
                if (!cmdBuf) {
                    return false;
                }
                current_cmd_completion = CompletionEventForCommandBuffer(cmdBuf, async_completion);
                return true;
            };

            auto ensure_output_buffer = [&](SlotId slot) -> bool {
                if (slot < 0 || (size_t)slot >= slot_bufs.size()) {
                    return false;
                }
                if (slot_bufs[slot]) {
                    return true;
                }

                if ((size_t)slot < plan_->slot_is_eliminated.size() &&
                    plan_->slot_is_eliminated[(size_t)slot]) {
                    return true;
                }
                size_t out_bytes = plan_->slots[slot].byte_size;
                if (out_bytes == 0) {
                    return false;
                }
                slot_bufs[slot] = AcquireReusableOutputBuffer(mtl_device, out_bytes);
                return slot_bufs[slot] != nil;
            };

            auto flush_native_cmd = [&]() -> bool {
                if (nativeComputeEncoder) {
                    [nativeComputeEncoder endEncoding];
                    nativeComputeEncoder = nil;
                }
                if (!cmdBuf) {
                    return true;
                }
                if (ops_in_cmd == 0) {
                    cmdBuf = nil;
                    current_cmd_completion.reset();
                    return true;
                }
                [cmdBuf commit];
                if (!async_completion) {
                    [cmdBuf waitUntilCompleted];
                    MarkCompletionFromCommandBuffer(current_cmd_completion, cmdBuf);
                }
                submitted_events.push_back(current_cmd_completion);
                cmdBuf = nil;
                current_cmd_completion.reset();
                ops_in_cmd = 0;
                bytes_in_cmd = 0;
                return true;
            };

            std::vector<id<MTLBuffer>> input_bufs;
            input_bufs.reserve(4);

            for (size_t step_idx = 0; step_idx < plan_->steps.size(); ++step_idx) {
                const auto& step = plan_->steps[step_idx];
                if (step.kind != Step::NATIVE) {
                    return ExecutionResult::Error("Native-only execution path encountered non-native step");
                }

                auto& ns = plan_->native_steps[step.index];
                input_bufs.clear();
                for (auto slot : ns.input_slots) {
                    input_bufs.push_back(slot_bufs[slot]);
                }

                if (ns.baked_kind == NativeStep::BakedKind::kReshapeAlias) {
                    if (ns.input_slots.empty()) {
                        return ExecutionResult::Error("Baked reshape missing input slot");
                    }
                    SlotId in_slot = ns.input_slots[0];
                    if (in_slot < 0 || (size_t)in_slot >= slot_bufs.size() || !slot_bufs[in_slot]) {
                        return ExecutionResult::Error("Baked reshape input slot is nil");
                    }
                    id<MTLBuffer> out = slot_bufs[in_slot];
                    CFRetain((__bridge CFTypeRef)out);
                    slot_bufs[ns.output_slot] = out;
                    slot_producer_events[(size_t)ns.output_slot] = slot_producer_events[(size_t)in_slot];
                    continue;
                }

                if (ns.baked_kind == NativeStep::BakedKind::kConstant) {
                    if (!ns.baked_constant_buffer) {
                        return ExecutionResult::Error("Baked constant missing buffer");
                    }
                    CFRetain((__bridge CFTypeRef)ns.baked_constant_buffer);
                    slot_bufs[ns.output_slot] = ns.baked_constant_buffer;
                    slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                    continue;
                }

                bool encoded = false;
                if (ns.baked_kind != NativeStep::BakedKind::kNone && ns.baked_pipeline) {
                    if (!ensure_output_buffer(ns.output_slot)) {
                        return ExecutionResult::Error("Failed to allocate baked output buffer");
                    }
                    if (!ensure_cmd_buf()) {
                        return ExecutionResult::Error("Failed to create native command buffer");
                    }

                    if (ns.baked_kind == NativeStep::BakedKind::kScatterAddRows) {
                        if (nativeComputeEncoder) {
                            [nativeComputeEncoder endEncoding];
                            nativeComputeEncoder = nil;
                        }
                        if (ns.input_slots.size() < 3) {
                            return ExecutionResult::Error("Baked scatter missing input slots");
                        }
                        id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
                        if (!blit) {
                            return ExecutionResult::Error("Failed to create blit encoder for baked scatter");
                        }
                        [blit copyFromBuffer:slot_bufs[ns.input_slots[0]]
                                 sourceOffset:0
                                     toBuffer:slot_bufs[ns.output_slot]
                            destinationOffset:0
                                         size:plan_->slots[ns.output_slot].byte_size];
                        [blit endEncoding];
                    }

                    if (!nativeComputeEncoder) {
                        nativeComputeEncoder = [cmdBuf computeCommandEncoder];
                        if (!nativeComputeEncoder) {
                            return ExecutionResult::Error("Failed to create native compute encoder");
                        }
                    }

                    [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                    switch (ns.baked_kind) {
                    case NativeStep::BakedKind::kElementwise: {
                        if (ns.input_slots.size() == 1) {
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:2];
                        } else if (ns.input_slots.size() == 2) {
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:3];
                        } else if (ns.input_slots.size() == 3) {
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[2]] offset:0 atIndex:2];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:3];
                            [nativeComputeEncoder setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:4];
                        } else {
                            return ExecutionResult::Error("Unsupported baked elementwise arity");
                        }
                        [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                        threadsPerThreadgroup:ns.baked_tg];
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kFusedElementwise: {
                        if (!ns.fused_info) {
                            return ExecutionResult::Error("Fused elementwise missing info");
                        }
                        auto& fi = *ns.fused_info;
                        uint32_t buf_idx = 0;
                        for (SlotId s : fi.external_input_slots) {
                            [nativeComputeEncoder setBuffer:slot_bufs[s] offset:0 atIndex:buf_idx++];
                        }
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:buf_idx++];
                        [nativeComputeEncoder setBytes:&ns.baked_n
                                                length:sizeof(ns.baked_n)
                                               atIndex:buf_idx++];
                        [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                        threadsPerThreadgroup:ns.baked_tg];
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kReduce: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked reduce missing input slot");
                        }
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                        [nativeComputeEncoder setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                        if (ns.baked_reduce_threadgroup) {
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                        } else {
                            [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                            threadsPerThreadgroup:ns.baked_tg];
                        }
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kIota: {
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:0];
                        [nativeComputeEncoder setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:1];
                        [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                        threadsPerThreadgroup:ns.baked_tg];
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kPad2D: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked pad missing input slot");
                        }
                        if (ns.baked_pad_use_fill_copy) {

                            float pv = ns.baked_pad_value;
                            if (ns.baked_pad_value_slot >= 0 &&
                                (size_t)ns.baked_pad_value_slot < slot_bufs.size() &&
                                slot_bufs[ns.baked_pad_value_slot]) {
                                id<MTLBuffer> pv_buf = slot_bufs[ns.baked_pad_value_slot];
                                void* pv_contents = [pv_buf contents];
                                if (pv_contents) {
                                    MPSDataType pvDtype = plan_->slots[(size_t)ns.baked_pad_value_slot].dtype;
                                    if (pvDtype == MPSDataTypeFloat16) {
                                        __fp16 h;
                                        memcpy(&h, pv_contents, sizeof(h));
                                        pv = static_cast<float>(h);
                                    } else {
                                        memcpy(&pv, pv_contents, sizeof(pv));
                                    }
                                }
                            }

                            [nativeComputeEncoder setComputePipelineState:ns.baked_fill_pipeline];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:0];
                            [nativeComputeEncoder setBytes:&pv length:sizeof(pv) atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_fill_n4
                                                    length:sizeof(ns.baked_fill_n4)
                                                   atIndex:2];
                            [nativeComputeEncoder dispatchThreads:ns.baked_fill_grid
                                            threadsPerThreadgroup:ns.baked_fill_tg];

                            [nativeComputeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];

                            [nativeComputeEncoder setComputePipelineState:ns.baked_copy2d_pipeline];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_src_cols
                                                    length:sizeof(ns.baked_src_cols)
                                                   atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_dst_cols
                                                    length:sizeof(ns.baked_dst_cols)
                                                   atIndex:3];
                            [nativeComputeEncoder setBytes:&ns.baked_low_h
                                                    length:sizeof(ns.baked_low_h)
                                                   atIndex:4];
                            [nativeComputeEncoder setBytes:&ns.baked_low_w
                                                    length:sizeof(ns.baked_low_w)
                                                   atIndex:5];
                            [nativeComputeEncoder setBytes:&ns.baked_src_total
                                                    length:sizeof(ns.baked_src_total)
                                                   atIndex:6];
                            [nativeComputeEncoder dispatchThreads:ns.baked_copy_grid
                                            threadsPerThreadgroup:ns.baked_copy_tg];
                        } else {

                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_in_h
                                                    length:sizeof(ns.baked_in_h)
                                                   atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_in_w
                                                    length:sizeof(ns.baked_in_w)
                                                   atIndex:3];
                            [nativeComputeEncoder setBytes:&ns.baked_out_w
                                                    length:sizeof(ns.baked_out_w)
                                                   atIndex:4];
                            [nativeComputeEncoder setBytes:&ns.baked_low_h
                                                    length:sizeof(ns.baked_low_h)
                                                   atIndex:5];
                            [nativeComputeEncoder setBytes:&ns.baked_low_w
                                                    length:sizeof(ns.baked_low_w)
                                                   atIndex:6];
                            if (ns.baked_pad_value_slot >= 0 &&
                                (size_t)ns.baked_pad_value_slot < slot_bufs.size() &&
                                slot_bufs[ns.baked_pad_value_slot]) {
                                id<MTLBuffer> pv_buf = slot_bufs[ns.baked_pad_value_slot];
                                void* pv_contents = [pv_buf contents];
                                float pv = 0.0f;
                                if (pv_contents) {
                                    MPSDataType pvDtype = plan_->slots[(size_t)ns.baked_pad_value_slot].dtype;
                                    if (pvDtype == MPSDataTypeFloat16) {
                                        __fp16 h;
                                        memcpy(&h, pv_contents, sizeof(h));
                                        pv = static_cast<float>(h);
                                    } else {
                                        memcpy(&pv, pv_contents, sizeof(pv));
                                    }
                                }
                                [nativeComputeEncoder setBytes:&pv length:sizeof(pv) atIndex:7];
                            } else {
                                [nativeComputeEncoder setBytes:&ns.baked_pad_value
                                                        length:sizeof(ns.baked_pad_value)
                                                       atIndex:7];
                            }
                            [nativeComputeEncoder setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:8];
                            [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                            threadsPerThreadgroup:ns.baked_tg];
                        }
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kScatterAddRows: {
                        if (ns.input_slots.size() < 3) {
                            return ExecutionResult::Error("Baked scatter missing input slots");
                        }
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[2]] offset:0 atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_scatter_k
                                                length:sizeof(ns.baked_scatter_k)
                                               atIndex:3];
                        [nativeComputeEncoder setBytes:&ns.baked_scatter_cols
                                                length:sizeof(ns.baked_scatter_cols)
                                               atIndex:4];
                        [nativeComputeEncoder setBytes:&ns.baked_scatter_out_rows
                                                length:sizeof(ns.baked_scatter_out_rows)
                                               atIndex:5];
                        [nativeComputeEncoder setBytes:&ns.baked_scatter_total
                                                length:sizeof(ns.baked_scatter_total)
                                               atIndex:6];
                        [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                        threadsPerThreadgroup:ns.baked_tg];
                        encoded = true;
                        break;
                    }
                    default:
                        break;
                    }
                }

                if (!encoded) {
                    if (nativeComputeEncoder) {
                        [nativeComputeEncoder endEncoding];
                        nativeComputeEncoder = nil;
                    }
                    const bool needs_cmd = ns.handler != NativeIntrinsicReshapeAliasHandler &&
                                           ns.handler != NativeIntrinsicConstantHandler;
                    if (needs_cmd && !ensure_cmd_buf()) {
                        return ExecutionResult::Error("Failed to create native command buffer");
                    }
                    id<MTLBuffer> output =
                        ns.handler(mtl_device, needs_cmd ? cmdBuf : nil, ns.op, input_bufs);
                    if (!output) {
                        return ExecutionResult::Error("Native op handler returned nil");
                    }
                    slot_bufs[ns.output_slot] = output;
                    if (needs_cmd) {
                        slot_producer_events[(size_t)ns.output_slot] = current_cmd_completion;
                    } else if (ns.handler == NativeIntrinsicReshapeAliasHandler && !ns.input_slots.empty()) {
                        SlotId in_slot = ns.input_slots[0];
                        if (in_slot >= 0 && (size_t)in_slot < slot_producer_events.size()) {
                            slot_producer_events[(size_t)ns.output_slot] =
                                slot_producer_events[(size_t)in_slot];
                        }
                    } else {
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                    }
                    if (!needs_cmd) {
                        continue;
                    }
                }

                if (encoded) {
                    slot_producer_events[(size_t)ns.output_slot] = current_cmd_completion;
                }

                ops_in_cmd += 1;
                bytes_in_cmd += plan_->slots[ns.output_slot].byte_size;
                const bool has_more = (step_idx + 1) < plan_->steps.size();
                const bool over_ops = max_ops_per_cmd > 0 && ops_in_cmd >= max_ops_per_cmd;
                const bool over_mb = max_mb_per_cmd > 0 && ((bytes_in_cmd >> 20) >= (size_t)max_mb_per_cmd);
                if (has_more && (over_ops || over_mb)) {
                    if (!flush_native_cmd()) {
                        return ExecutionResult::Error("Failed to rotate native command buffer");
                    }
                }
            }

            if (!flush_native_cmd()) {
                return ExecutionResult::Error("Failed to finalize native command buffer");
            }
            if (!submitted_events.empty()) {
                execution_completion = submitted_events.back();
            }
        } else if (!executed_direct_run) {

            MPSCommandBuffer* cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
            if (!cmdBuf) {
                return ExecutionResult::Error("Failed to create command buffer");
            }
            CompletionEvent cmd_completion = CompletionEventForCommandBuffer(cmdBuf, async_completion);

            MPS_LOG_DEBUG("Executing %zu steps\n", plan_->steps.size());
            std::vector<id<MTLBuffer>> mixed_input_bufs;
            mixed_input_bufs.reserve(4);
            for (size_t step_idx = 0; step_idx < plan_->steps.size(); step_idx++) {
                const auto& step = plan_->steps[step_idx];
                MPS_LOG_DEBUG("Step %zu: kind=%s\n", step_idx, step.kind == Step::GRAPH ? "GRAPH" : "NATIVE");
                if (step.kind == Step::GRAPH) {
                    auto& gs = plan_->graph_steps[step.index];
                    MPS_LOG_DEBUG("  graph_step index=%zu, feeds=%zu, targets=%zu\n", step.index,
                                  gs.feeds.size(), gs.targets.size());
                    MPSGraph* graph = (__bridge MPSGraph*)gs.graph;
                    if (!graph) {
                        return ExecutionResult::Error("Graph is nil at step " + std::to_string(step_idx));
                    }

                    MPSGraphExecutable* exec = (__bridge MPSGraphExecutable*)gs.executable;
                    const bool use_exec = exec && !disable_executable && !gs.exec_target_slots.empty();

                    if (use_exec) {
                        NSArray<MPSGraphTensor*>* feedOrder = exec.feedTensors;
                        NSArray<MPSGraphTensor*>* targetOrder = exec.targetTensors;
                        if (!feedOrder || feedOrder.count != gs.exec_feed_slots.size() || !targetOrder ||
                            targetOrder.count != gs.exec_target_slots.size()) {

                        } else {

                            NSMutableArray<MPSGraphTensorData*>* inputsArray =
                                [NSMutableArray arrayWithCapacity:gs.exec_feed_slots.size()];
                            for (NSUInteger i = 0; i < feedOrder.count; ++i) {
                                SlotId slot = gs.exec_feed_slots[i];
                                if (slot < 0 || (size_t)slot >= slot_bufs.size() || !slot_bufs[slot]) {
                                    return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                                  " is nil during exec inputs");
                                }
                                MPSGraphTensor* t = feedOrder[i];
                                bool borrowed = false;
                                MPSGraphTensorData* data = GetTensorDataForSlot(
                                    cache, slot, slot_bufs[slot], t.shape, t.dataType, &borrowed);
                                if (!data) {
                                    return ExecutionResult::Error("Failed to create tensor data for slot " +
                                                                  std::to_string(slot));
                                }
                                [inputsArray addObject:data];
                                if (!borrowed) {
                                    [data release];
                                }
                            }

                            NSMutableArray<MPSGraphTensorData*>* resultsArray =
                                [NSMutableArray arrayWithCapacity:gs.exec_target_slots.size()];
                            for (NSUInteger i = 0; i < targetOrder.count; ++i) {
                                SlotId slot = gs.exec_target_slots[i];
                                if (slot < 0 || (size_t)slot >= slot_bufs.size() || !slot_bufs[slot]) {
                                    return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                                  " is nil during exec results");
                                }
                                MPSGraphTensor* t = targetOrder[i];
                                bool borrowed = false;
                                MPSGraphTensorData* out = GetTensorDataForSlot(
                                    cache, slot, slot_bufs[slot], t.shape, t.dataType, &borrowed);
                                if (!out) {
                                    return ExecutionResult::Error("Failed to create tensor data for slot " +
                                                                  std::to_string(slot));
                                }
                                [resultsArray addObject:out];
                                if (!borrowed) {
                                    [out release];
                                }
                            }

                            @try {
                                [exec encodeToCommandBuffer:cmdBuf
                                                inputsArray:inputsArray
                                               resultsArray:resultsArray
                                        executionDescriptor:nil];
                            } @catch (NSException* exception) {
                                return ExecutionResult::Error("MPS executable execution failed: " +
                                                              std::string([[exception reason] UTF8String]));
                            }

                            for (SlotId target_slot : gs.exec_target_slots) {
                                if (target_slot >= 0 && (size_t)target_slot < slot_producer_events.size()) {
                                    slot_producer_events[(size_t)target_slot] = cmd_completion;
                                }
                            }
                            MPS_LOG_DEBUG("  Graph executable encoded\n");
                            continue;
                        }
                    }

                    MPS_LOG_DEBUG("  Building feeds dictionary\n");
                    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds =
                        [NSMutableDictionary dictionaryWithCapacity:gs.feeds.size()];
                    for (auto& [slot, tensor_ptr] : gs.feeds) {
                        MPS_LOG_DEBUG("    Feed slot=%d, tensor_ptr=%p\n", slot, tensor_ptr);
                        if (!slot_bufs[slot]) {
                            return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                          " is nil during feed construction");
                        }
                        MPSGraphTensor* tensor = (__bridge MPSGraphTensor*)tensor_ptr;
                        if (!tensor) {
                            return ExecutionResult::Error("Feed tensor is nil for slot " +
                                                          std::to_string(slot));
                        }
                        MPS_LOG_DEBUG("    Tensor shape: %s, dtype: %d\n",
                                      [[tensor.shape description] UTF8String], (int)tensor.dataType);
                        bool borrowed = false;
                        MPSGraphTensorData* data = GetTensorDataForSlot(
                            cache, slot, slot_bufs[slot], tensor.shape, tensor.dataType, &borrowed);
                        if (!data) {
                            return ExecutionResult::Error("Failed to create feed tensor data for slot " +
                                                          std::to_string(slot));
                        }
                        feeds[tensor] = data;
                        if (!borrowed) {
                            [data release];
                        }
                    }
                    MPS_LOG_DEBUG("  Feeds built: %lu entries\n", (unsigned long)[feeds count]);

                    MPS_LOG_DEBUG("  Building results dictionary\n");
                    NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results =
                        [NSMutableDictionary dictionaryWithCapacity:gs.targets.size()];
                    for (auto& [slot, tensor_ptr] : gs.targets) {
                        MPS_LOG_DEBUG("    Target slot=%d, tensor_ptr=%p\n", slot, tensor_ptr);
                        if (!slot_bufs[slot]) {
                            return ExecutionResult::Error("Slot buffer " + std::to_string(slot) +
                                                          " is nil during results construction");
                        }
                        MPSGraphTensor* tensor = (__bridge MPSGraphTensor*)tensor_ptr;
                        if (!tensor) {
                            return ExecutionResult::Error("Target tensor is nil for slot " +
                                                          std::to_string(slot));
                        }
                        if (plan_->slots[slot].dtype != tensor.dataType) {
                            return ExecutionResult::Error(
                                "Slot dtype mismatch for target slot " + std::to_string(slot) +
                                ": slot dtype=" + std::to_string((int)plan_->slots[slot].dtype) +
                                ", tensor dtype=" + std::to_string((int)tensor.dataType));
                        }
                        bool borrowed = false;
                        MPSGraphTensorData* out = GetTensorDataForSlot(
                            cache, slot, slot_bufs[slot], tensor.shape, tensor.dataType, &borrowed);
                        if (!out) {
                            return ExecutionResult::Error("Failed to create result tensor data for slot " +
                                                          std::to_string(slot));
                        }
                        results[tensor] = out;
                        if (!borrowed) {
                            [out release];
                        }
                    }
                    MPS_LOG_DEBUG("  Results built: %lu entries\n", (unsigned long)[results count]);

                    MPS_LOG_DEBUG("  Encoding graph to command buffer\n");
                    MPS_LOG_DEBUG("  graph=%p, cmdBuf=%p, feeds count=%lu, results count=%lu\n", (void*)graph,
                                  (void*)cmdBuf, (unsigned long)[feeds count],
                                  (unsigned long)[results count]);
                    @try {
                        MPS_LOG_DEBUG("  calling encodeToCommandBuffer (resultsDictionary)...\n");
                        [graph encodeToCommandBuffer:cmdBuf
                                               feeds:feeds
                                    targetOperations:nil
                                   resultsDictionary:results
                                 executionDescriptor:nil];
                        MPS_LOG_DEBUG("  encodeToCommandBuffer returned\n");
                    } @catch (NSException* exception) {
                        return ExecutionResult::Error("MPS graph execution failed: " +
                                                      std::string([[exception reason] UTF8String]));
                    }
                    MPS_LOG_DEBUG("  Graph encoded\n");
                    for (const auto& target : gs.targets) {
                        SlotId target_slot = target.first;
                        if (target_slot >= 0 && (size_t)target_slot < slot_producer_events.size()) {
                            slot_producer_events[(size_t)target_slot] = cmd_completion;
                        }
                    }

                    MPS_LOG_DEBUG("  Graph step %zu complete\n", step_idx);
                } else {

                    auto& ns = plan_->native_steps[step.index];

                    mixed_input_bufs.clear();
                    for (auto slot : ns.input_slots) {
                        mixed_input_bufs.push_back(slot_bufs[slot]);
                    }

                    id<MTLBuffer> output = ns.handler(mtl_device, cmdBuf, ns.op, mixed_input_bufs);
                    if (!output) {
                        return ExecutionResult::Error("Native op handler returned nil");
                    }

                    slot_bufs[ns.output_slot] = output;
                    if (ns.handler == NativeIntrinsicReshapeAliasHandler && !ns.input_slots.empty()) {
                        SlotId in_slot = ns.input_slots[0];
                        if (in_slot >= 0 && (size_t)in_slot < slot_producer_events.size()) {
                            slot_producer_events[(size_t)ns.output_slot] =
                                slot_producer_events[(size_t)in_slot];
                        }
                    } else if (ns.handler == NativeIntrinsicConstantHandler) {
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                    } else {
                        slot_producer_events[(size_t)ns.output_slot] = cmd_completion;
                    }
                }
            }

            [cmdBuf commit];
            if (!async_completion) {
                [cmdBuf waitUntilCompleted];
                MarkCompletionFromCommandBuffer(cmd_completion, cmdBuf);
            }
            execution_completion = cmd_completion;
        }

        const std::vector<uint8_t>& slot_is_input = plan_->slot_is_input;
        const std::vector<uint8_t>& slot_is_output = plan_->slot_is_output;
        void* output_device_key = (__bridge void*)mtl_device;

        ExecutionResult result;
        for (size_t i = 0; i < plan_->output_slots.size(); i++) {
            SlotId slot = plan_->output_slots[i];
            id<MTLBuffer> buf = slot_bufs[slot];
            CompletionEvent slot_completion = (slot >= 0 && (size_t)slot < slot_producer_events.size())
                                                  ? slot_producer_events[(size_t)slot]
                                                  : nullptr;
            if (!slot_completion) {
                slot_completion = execution_completion;
            }

            const auto& meta = plan_->output_meta[i];
            int dtype = meta.pjrt_dtype;
            if (dtype < 0) {
                return ExecutionResult::Error("Unsupported output type for result " + std::to_string(i));
            }

            const auto& dims = meta.dims;
            size_t out_byte_size = meta.byte_size;
            if (buf) {
                size_t buf_len = (size_t)[buf length];
                if (buf_len > out_byte_size) {
                    out_byte_size = buf_len;
                }
            }

            if (slot_is_input[slot]) {
                AwaitCompletionEvent(slot_completion);
                const std::string slot_error = CompletionEventError(slot_completion);
                if (!slot_error.empty()) {
                    return ExecutionResult::Error("Output slot completion failed: " + slot_error);
                }
                size_t byte_size = out_byte_size;
                void* contents = [buf contents];
                id<MTLBuffer> copy = nil;
                if (contents) {
                    if (enable_output_pool &&
                        ComputeBufferResourceOptions() == MTLResourceStorageModeShared) {
                        copy = AcquireReusableOutputBuffer(mtl_device, byte_size);
                        if (copy && [copy contents]) {
                            memcpy([copy contents], contents, byte_size);
                        }
                    } else {
                        copy = [mtl_device newBufferWithBytes:contents
                                                       length:byte_size
                                                      options:ComputeBufferResourceOptions()];
                    }
                } else {
                    std::string copy_err;
                    copy = BlitCopyToNewBuffer(mtl_device, commandQueue, buf, byte_size,
                                               MTLResourceStorageModeShared, &copy_err);
                    if (!copy && !copy_err.empty()) {
                        return ExecutionResult::Error(copy_err);
                    }
                }
                if (!copy) {
                    return ExecutionResult::Error("Failed to allocate output buffer copy");
                }
                if (enable_output_pool) {
                    result.buffers.push_back(std::make_unique<MpsBuffer>(
                        device, (__bridge void*)copy, dtype, dims,
                        [output_device_key, out_byte_size](void* metal_buffer) {
                            RecycleReusableOutputBuffer(output_device_key, out_byte_size, metal_buffer);
                        },
                        MakeReadyCompletionEvent()));
                } else {
                    result.buffers.push_back(std::make_unique<MpsBuffer>(device, (__bridge void*)copy, dtype,
                                                                         dims, MpsBuffer::BufferReleaser{},
                                                                         MakeReadyCompletionEvent()));
                }

                CFRelease((__bridge CFTypeRef)copy);
            } else {
                if (enable_output_pool) {
                    result.buffers.push_back(std::make_unique<MpsBuffer>(
                        device, (__bridge void*)buf, dtype, dims,
                        [output_device_key, out_byte_size](void* metal_buffer) {
                            RecycleReusableOutputBuffer(output_device_key, out_byte_size, metal_buffer);
                        },
                        slot_completion));
                } else {
                    result.buffers.push_back(std::make_unique<MpsBuffer>(device, (__bridge void*)buf, dtype,
                                                                         dims, MpsBuffer::BufferReleaser{},
                                                                         slot_completion));
                }

                CFRelease((__bridge CFTypeRef)buf);
                slot_bufs[slot] = nil;
            }
        }

        std::vector<std::pair<void*, size_t>> deferred_recycle;
        std::vector<void*> deferred_release;
        deferred_recycle.reserve(slot_bufs.size());
        deferred_release.reserve(slot_bufs.size());

        for (size_t slot = 0; slot < slot_bufs.size(); ++slot) {
            if (!slot_bufs[slot]) {
                continue;
            }
            if (slot_is_input[slot] || slot_is_output[slot]) {
                continue;
            }

            if (cache && slot < cache->bufs.size() && slot_bufs[slot] == cache->bufs[slot]) {
                slot_bufs[slot] = nil;
                continue;
            }
            if (enable_output_pool && plan_->slots[slot].byte_size > 0) {
                deferred_recycle.emplace_back((__bridge void*)slot_bufs[slot], plan_->slots[slot].byte_size);
            } else {
                deferred_release.push_back((__bridge void*)slot_bufs[slot]);
            }
            slot_bufs[slot] = nil;
        }

        if (!deferred_recycle.empty() || !deferred_release.empty()) {
            auto recycle_items =
                std::make_shared<std::vector<std::pair<void*, size_t>>>(std::move(deferred_recycle));
            auto release_items = std::make_shared<std::vector<void*>>(std::move(deferred_release));
            CompletionEvent cleanup_completion =
                execution_completion ? execution_completion : MakeReadyCompletionEvent();
            cleanup_completion->AddReadyCallback([output_device_key, recycle_items, release_items] {
                for (const auto& item : *recycle_items) {
                    RecycleReusableOutputBuffer(output_device_key, item.second, item.first);
                }
                for (void* metal_buffer : *release_items) {
                    CFRelease((__bridge CFTypeRef)metal_buffer);
                }
            });
        }

        if (cache_intermediates && cache_set) {
            if (execution_completion && !execution_completion->IsReady()) {
                auto delayed_cache = std::shared_ptr<IntermediateCacheSet>(cache_set.release());
                execution_completion->AddReadyCallback([delayed_cache] {});
            } else {
                std::scoped_lock lock(plan_->intermediate_pool_mutex);
                plan_->intermediate_pool.push_back(std::move(cache_set));
            }
        }

        result.completion = execution_completion ? execution_completion : MakeReadyCompletionEvent();
        return result;
    }
}

} // namespace jax_metallib
