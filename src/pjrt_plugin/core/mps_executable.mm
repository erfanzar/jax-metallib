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
#include "stablehlo/dialect/StablehloOps.h"
#include "stablehlo/dialect/Version.h"

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dispatch/dispatch.h>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <sys/stat.h>
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

    // Synchronously release all owned resources. Caller is responsible for
    // ensuring the GPU is no longer using any of the MTLBuffers referenced
    // here (typically by holding a CompletionEvent fence).
    void ReleaseAll() {
        for (MPSGraphTensorData* td : tensor_datas) {
            if (td) {
                [td release];
            }
        }
        tensor_datas.clear();
        for (id<MTLBuffer> buf : tensor_data_bufs) {
            if (buf) {
                CFRelease((__bridge CFTypeRef)buf);
            }
        }
        tensor_data_bufs.clear();
        for (id<MTLBuffer> buf : bufs) {
            if (buf) {
                CFRelease((__bridge CFTypeRef)buf);
            }
        }
        bufs.clear();
    }

    ~IntermediateCacheSet() {
        ReleaseAll();
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
        kSoftmax,
        kLayerNorm,
        kGatherRows,
        kIota,
        kPad2D,
        kScatterAddRows,
        kReshapeAlias,
        kTransposeAlias,
        kConstant,
        kGEMM,
        kCpuBLAS,
        kSort,
        kTranspose2D,
        kCpuSort,
        kCpuCholesky,
        kReduceSplit,
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
    bool baked_use_stride = false;
    bool baked_reduce_threadgroup = false;

    // Split reduce (two-stage single-row huge reduce): stage 1 reduces G
    // column groups into G partials, stage 2 combines them into the scalar.
    id<MTLComputePipelineState> baked_reduce_split_combine_pipeline = nil;
    id<MTLBuffer> baked_reduce_split_partials = nil;
    uint32_t baked_reduce_split_groups = 0;
    MTLSize baked_reduce_split_combine_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_reduce_split_combine_tg = MTLSizeMake(1, 1, 1);
    bool baked_reduce_split_combine_tg_dispatch = false;

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

    id<MTLComputePipelineState> baked_softmax_pass2_pipeline = nil;
    id<MTLBuffer> baked_softmax_maxsum = nil;
    MTLSize baked_softmax_pass2_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_softmax_pass2_tg = MTLSizeMake(1, 1, 1);

    // LayerNorm fields (2-pass, reuses baked_rows/baked_cols from reduce)
    id<MTLComputePipelineState> baked_layernorm_pass2_pipeline = nil;
    id<MTLBuffer> baked_layernorm_stats = nil; // [rows × 2] floats: [mean, inv_std]
    MTLSize baked_layernorm_pass2_grid = MTLSizeMake(1, 1, 1);
    MTLSize baked_layernorm_pass2_tg = MTLSizeMake(1, 1, 1);
    float baked_layernorm_eps = 1e-5f;

    // Gather fields
    uint32_t baked_gather_num_indices = 0;
    uint32_t baked_gather_cols = 0;
    uint32_t baked_gather_table_rows = 0;

    uint32_t baked_gemm_m = 0;
    uint32_t baked_gemm_n = 0;
    uint32_t baked_gemm_k = 0;
    uint32_t baked_gemm_batch = 0;
    int baked_gemm_variant = 0; // kMppGemmVariant (-1) for MPP path

    // Transpose fields (last-2-dims swap)
    uint32_t baked_transpose_rows = 0;
    uint32_t baked_transpose_cols = 0;
    uint32_t baked_transpose_batch = 1;

    // Sort fields
    uint32_t baked_sort_n = 0; // elements per row (or total for 1D)
    uint32_t baked_sort_padded_n = 0;
    uint32_t baked_sort_num_rows = 1; // 1 for 1D sort, >1 for 2D row-wise sort

    // CPU Cholesky fields
    uint32_t baked_cholesky_n = 0;
    uint32_t baked_cholesky_batch = 1;
    bool baked_cholesky_lower = true;

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
        // For transpose-alias outputs: the permutation and the input shape so
        // we can compute the correct byte_strides at output buffer creation.
        std::vector<int64_t> transpose_perm; // empty → contiguous output
        std::vector<int64_t> transpose_input_dims;
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

// A nil compilation descriptor behaves like MPSGraphOptimizationLevel1, whose
// placement pass can run float16 subgraphs on the Neural Engine. There f16
// `nan != nan` was false, log(0) gave -45440, sqrt(-x) gave 0, exp(nan) gave inf
// and f32->f16 casts turned NaN into inf and -0 into +0 (reproduced with a
// standalone MPSGraph program). Level0 keeps core optimizations on the GPU and
// matches XLA. JAX_METALLIB_MPSGRAPH_OPT_LEVEL1=1 restores the old default.
static MPSGraphCompilationDescriptor* GraphCompilationDescriptor() {
    static MPSGraphCompilationDescriptor* desc = []() -> MPSGraphCompilationDescriptor* {
        if (EnvFlagEnabled("JAX_METALLIB_MPSGRAPH_OPT_LEVEL1")) {
            return nil;
        }
        MPSGraphCompilationDescriptor* d = [MPSGraphCompilationDescriptor new];
        d.optimizationLevel = MPSGraphOptimizationLevel0;
        return d;
    }();
    return desc;
}

static MPSGraphExecutionDescriptor* GraphExecutionDescriptor() {
    static MPSGraphExecutionDescriptor* desc = []() -> MPSGraphExecutionDescriptor* {
        MPSGraphCompilationDescriptor* c = GraphCompilationDescriptor();
        if (!c) {
            return nil;
        }
        MPSGraphExecutionDescriptor* d = [MPSGraphExecutionDescriptor new];
        d.compilationDescriptor = c;
        return d;
    }();
    return desc;
}

// Intrinsic-native kernels are DISABLED by default. Under JAX 0.11's eager
// dispatch, the fused-chain path still has a slot-aliasing defect (a graph-step
// target sharing a slot with a fused-chain output) that yields wrong results
// for some MultivariateNormal log-prob gradients when enabled. The CFRelease(nil)
// abort and the broadcast-ordering wrong-results are fixed (0.11.1.0). Opt in
// with JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS=1 for experiments;
// JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS=1 always wins.
static bool IntrinsicNativeKernelsEnabled() {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_INTRINSIC_NATIVE_KERNELS")) {
        return false;
    }
    return EnvFlagEnabled("JAX_METALLIB_ENABLE_INTRINSIC_NATIVE_KERNELS");
}

// Full last-axis softmax (reduce_max -> sub -> exp -> reduce_sum -> div) is
// fused and routed to the standalone softmax kernel even when the intrinsic
// family is off: it does not use the fused elementwise-chain machinery where
// the known defects live, and MPSGraph softmax was ~3.7x slower at
// [4096,4096]. Fusion and routing must stay enabled together: a fused
// reduce_max reaching MPSGraph has the wrong shape.
// JAX_METALLIB_DISABLE_STANDALONE_SOFTMAX=1 restores the graph path.
static bool StandaloneSoftmaxEnabled() {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_SOFTMAX_FUSION")) {
        return false;
    }
    return IntrinsicNativeKernelsEnabled() || !EnvFlagEnabled("JAX_METALLIB_DISABLE_STANDALONE_SOFTMAX");
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
static void RecycleReusableOutputBuffer(void* device_key, size_t byte_size, void* metal_buffer);
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
static id<MTLBuffer> NativeIntrinsicGatherRowsHandler(id<MTLDevice> device,
                                                      id<MTLCommandBuffer> commandBuffer, mlir::Operation* op,
                                                      const std::vector<id<MTLBuffer>>& inputs);

enum class IntrinsicNativeKind {
    kNone,
    kReshapeAlias,
    kTransposeAlias,
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
    kReduceMeanLastAxis,
    kReduceSumAxis0,
    kReduceMaxAxis0,
    kReduceMinAxis0,
    kReduceMeanAxis0,
    kSortAsc,
    kSoftmaxLastAxis,
    kSoftmaxNumeratorLastAxis,
    kLayerNormLastAxis,
    kGatherRows,
    kConvertFp16Fp32,
    kDotGeneralNN,
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

        // Untracked buffers let a native kernel read a slot while an MPSGraph
        // step on the same command buffer was still writing it: a >=4-op graph
        // chain feeding the native last-axis reduce returned zero rows at
        // ~16M elements (f16 softmax [4096,4096] gave NaN). Tracking measured
        // no cost (9 workloads within noise), so it is the default now;
        // JAX_METALLIB_DISABLE_HAZARD_TRACKING=1 restores untracked buffers.
        if (EnvFlagEnabled("JAX_METALLIB_DISABLE_HAZARD_TRACKING") &&
            !EnvFlagEnabled("JAX_METALLIB_ENABLE_HAZARD_TRACKING")) {
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
    if (name == "stablehlo.transpose") {
        return IntrinsicNativeKind::kTransposeAlias;
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
    if (name == "stablehlo.gather") {
        return IntrinsicNativeKind::kGatherRows;
    }
    return IntrinsicNativeKind::kNone;
}

static bool IsEligibleTransposeAliasOp(mlir::Operation* op) {
    if (!op || op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return false;
    }
    // Only alias transposes whose result goes directly to the function return
    // (i.e., no downstream ops consume it within this computation).  Internal
    // transposes must remain as graph ops because native steps expect contiguous
    // data in slot buffers.
    mlir::Value result = op->getResult(0);
    for (auto& use : result.getUses()) {
        mlir::Operation* user = use.getOwner();
        if (!mlir::isa<mlir::func::ReturnOp>(user)) {
            return false; // consumed by a non-return op → can't alias
        }
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

// Native transpose: handles transposes that swap only the last 2 dims on f32 tensors.
// These are NOT eligible for alias (result consumed internally), but can use a fast Metal kernel.
static bool IsEligibleTransposeNativeOp(mlir::Operation* op, uint32_t* out_rows = nullptr,
                                        uint32_t* out_cols = nullptr, uint32_t* out_batch = nullptr) {
    auto transposeOp = mlir::dyn_cast_or_null<mlir::stablehlo::TransposeOp>(op);
    if (!transposeOp) {
        return false;
    }
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return false;
    }
    std::vector<int64_t> inShape;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype)) {
        return false;
    }
    if (dtype != MPSDataTypeFloat32 || inShape.size() < 2) {
        return false;
    }
    // Check permutation: only last 2 dims swapped, all others identity
    auto perm = transposeOp.getPermutation();
    int64_t rank = static_cast<int64_t>(inShape.size());
    for (int64_t i = 0; i < rank - 2; ++i) {
        if (perm[i] != i)
            return false;
    }
    if (perm[rank - 2] != rank - 1 || perm[rank - 1] != rank - 2) {
        return false;
    }
    uint32_t rows = static_cast<uint32_t>(inShape[rank - 2]);
    uint32_t cols = static_cast<uint32_t>(inShape[rank - 1]);
    if (rows == 0 || cols == 0)
        return false;
    uint32_t batch = 1;
    for (int64_t i = 0; i < rank - 2; ++i) {
        batch *= static_cast<uint32_t>(inShape[i]);
    }
    if (out_rows)
        *out_rows = rows;
    if (out_cols)
        *out_cols = cols;
    if (out_batch)
        *out_batch = batch;
    return true;
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

// Returns: "add", "mul", "max", "min", or "" for unsupported combiners.
static std::string ClassifyScatterCombiner(mlir::stablehlo::ScatterOp scatterOp) {
    auto& region = scatterOp.getUpdateComputation();
    if (region.empty())
        return "";
    for (auto& inner : region.front()) {
        if (mlir::isa<mlir::stablehlo::AddOp>(inner))
            return "add";
        if (mlir::isa<mlir::stablehlo::MulOp>(inner))
            return "mul";
        if (mlir::isa<mlir::stablehlo::MaxOp>(inner))
            return "max";
        if (mlir::isa<mlir::stablehlo::MinOp>(inner))
            return "min";
    }
    return "";
}

static bool IsScatterAddCombiner(mlir::stablehlo::ScatterOp scatterOp) {
    return ClassifyScatterCombiner(scatterOp) == "add";
}

static bool IsEligibleScatterAddRowsNativeOp(mlir::Operation* op) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_NATIVE_SCATTER_ROWS")) {
        return false;
    }
    auto scatterOp = mlir::dyn_cast_or_null<mlir::stablehlo::ScatterOp>(op);
    if (!scatterOp || op->getNumOperands() != 3 || op->getNumResults() != 1) {
        return false;
    }
    std::string combiner = ClassifyScatterCombiner(scatterOp);
    if (combiner.empty()) {
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

    if (!IsFloat16Or32(inType) || inType != updType || inType != outType) {
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

// Eligibility check for native gather (embedding lookup): table[indices, :]
static bool IsEligibleGatherRowsNativeOp(mlir::Operation* op) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_NATIVE_GATHER_ROWS")) {
        return false;
    }
    auto gatherOp = mlir::dyn_cast_or_null<mlir::stablehlo::GatherOp>(op);
    if (!gatherOp || op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return false;
    }
    auto dnums = gatherOp.getDimensionNumbers();
    auto offsetDims = dnums.getOffsetDims();
    auto collapsedDims = dnums.getCollapsedSliceDims();
    auto startIndexMap = dnums.getStartIndexMap();
    int64_t indexVectorDim = dnums.getIndexVectorDim();

    // Must be: offset_dims=[1], collapsed_slice_dims=[0], start_index_map=[0], index_vector_dim=1
    if (offsetDims.size() != 1 || offsetDims[0] != 1 || collapsedDims.size() != 1 || collapsedDims[0] != 0 ||
        startIndexMap.size() != 1 || startIndexMap[0] != 0 || indexVectorDim != 1) {
        return false;
    }

    std::vector<int64_t> operandShape, indexShape, outShape;
    MPSDataType operandDtype = MPSDataTypeInvalid, indexDtype = MPSDataTypeInvalid,
                outDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &operandShape, nullptr, &operandDtype) ||
        !GetStaticTensorInfo(op->getOperand(1).getType(), &indexShape, nullptr, &indexDtype) ||
        !GetStaticTensorInfo(op->getResult(0).getType(), &outShape, nullptr, &outDtype)) {
        return false;
    }
    if (operandShape.size() != 2 || indexShape.size() != 2 || outShape.size() != 2) {
        return false;
    }
    if (indexDtype != MPSDataTypeInt32 || !IsFloat16Or32(operandDtype) || operandDtype != outDtype) {
        return false;
    }
    // indexShape should be [num_indices, 1]
    if (indexShape[1] != 1)
        return false;
    // slice_sizes should be [1, cols]
    auto sliceSizes = gatherOp.getSliceSizes();
    if (sliceSizes.size() != 2 || sliceSizes[0] != 1 || sliceSizes[1] != operandShape[1]) {
        return false;
    }
    // outShape should be [num_indices, cols]
    if (outShape[0] != indexShape[0] || outShape[1] != operandShape[1])
        return false;
    if (operandShape[0] <= 0 || operandShape[1] <= 0 || indexShape[0] <= 0)
        return false;
    uint64_t total = static_cast<uint64_t>(indexShape[0]) * static_cast<uint64_t>(operandShape[1]);
    if (total > std::numeric_limits<uint32_t>::max())
        return false;
    return true;
}

static std::string GetReduceBodyOpName(mlir::stablehlo::ReduceOp reduceOp) {
    if (reduceOp.getBody().empty()) {
        return "";
    }
    mlir::Block& block = reduceOp.getBody().front();
    if (block.getNumArguments() != 2) {
        return "";
    }
    // Strict validation: the reducer must be exactly one binary compute op
    // over the two block arguments whose result is returned directly. The
    // previous "first non-return op name" heuristic could misroute a
    // multi-op reducer (e.g. add-then-convert) as a plain sum/max/min.
    mlir::Operation* compute = nullptr;
    mlir::Operation* ret = nullptr;
    for (mlir::Operation& inner : block) {
        std::string opName = inner.getName().getStringRef().str();
        if (opName == "stablehlo.return") {
            if (ret) {
                return "";
            }
            ret = &inner;
            continue;
        }
        if (compute) {
            return "";
        }
        compute = &inner;
    }
    if (!compute || !ret || compute->getNumOperands() != 2 || compute->getNumResults() != 1 ||
        ret->getNumOperands() != 1 || ret->getOperand(0) != compute->getResult(0)) {
        return "";
    }
    auto arg0 = mlir::dyn_cast<mlir::BlockArgument>(compute->getOperand(0));
    auto arg1 = mlir::dyn_cast<mlir::BlockArgument>(compute->getOperand(1));
    if (!arg0 || !arg1 || arg0.getOwner() != &block || arg1.getOwner() != &block ||
        arg0.getArgNumber() == arg1.getArgNumber()) {
        return "";
    }
    return compute->getName().getStringRef().str();
}

static bool IsNeutralReduceInit(mlir::Value init, IntrinsicNativeKind kind) {
    double v = 0.0;
    if (!GetScalarFloatConstant(init, &v)) {
        return false;
    }
    if (kind == IntrinsicNativeKind::kReduceSumLastAxis || kind == IntrinsicNativeKind::kReduceMeanLastAxis ||
        kind == IntrinsicNativeKind::kReduceSumAxis0 || kind == IntrinsicNativeKind::kReduceMeanAxis0 ||
        kind == IntrinsicNativeKind::kLayerNormLastAxis) {
        return v == 0.0 && !std::signbit(v);
    }
    if (kind == IntrinsicNativeKind::kReduceMaxLastAxis || kind == IntrinsicNativeKind::kReduceMaxAxis0 ||
        kind == IntrinsicNativeKind::kSoftmaxLastAxis ||
        kind == IntrinsicNativeKind::kSoftmaxNumeratorLastAxis) {
        return std::isinf(v) && v < 0.0;
    }
    if (kind == IntrinsicNativeKind::kReduceMinLastAxis || kind == IntrinsicNativeKind::kReduceMinAxis0) {
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

static bool IsEligibleReduceNativeOp(mlir::Operation* op, IntrinsicNativeKind kind);
static bool IsEligibleSortNativeOp(mlir::Operation* op);

static std::optional<double> TraceScalarConstant(mlir::Value value, int depth = 0) {
    if (!value) {
        return std::nullopt;
    }
    double v = 0.0;
    if (GetScalarFloatConstant(value, &v)) {
        return v;
    }
    auto* defOp = value.getDefiningOp();
    if (!defOp) {
        return std::nullopt;
    }
    if (mlir::isa<mlir::stablehlo::BroadcastInDimOp>(defOp) || mlir::isa<mlir::stablehlo::ReshapeOp>(defOp) ||
        mlir::isa<mlir::stablehlo::BroadcastOp>(defOp) || mlir::isa<mlir::stablehlo::ConvertOp>(defOp)) {
        if (defOp->getNumOperands() >= 1) {
            return TraceScalarConstant(defOp->getOperand(0), depth + 1);
        }
    }
    auto constOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(defOp);
    if (constOp) {
        auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(constOp.getValue());
        if (dense && dense.isSplat() && mlir::isa<mlir::FloatType>(dense.getElementType())) {
            llvm::APFloat ap = *dense.getValues<llvm::APFloat>().begin();
            return ap.convertToDouble();
        }
    }
    return std::nullopt;
}

static bool IsReduceMeanPattern(mlir::Operation* reduceOp,
                                llvm::SmallVectorImpl<mlir::Operation*>& absorbed_ops,
                                mlir::Value& final_result) {
    auto reduce = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(reduceOp);
    if (!reduce) {
        return false;
    }
    std::string bodyOp = GetReduceBodyOpName(reduce);
    if (bodyOp != "stablehlo.add") {
        return false;
    }
    if (reduce->getNumOperands() < 1 || reduce->getNumResults() != 1) {
        return false;
    }
    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(reduce->getOperand(0).getType());
    if (!inType || !inType.hasStaticShape() || inType.getRank() == 0) {
        return false;
    }
    auto dims = reduce.getDimensions();
    if (dims.size() != 1) {
        return false;
    }
    int64_t axis = dims[0];
    int64_t rank = inType.getRank();
    // Accept last-axis or axis-0 on rank-2 tensors
    if (axis != rank - 1 && !(axis == 0 && rank == 2)) {
        return false;
    }
    int64_t cols = inType.getShape()[axis]; // size of the reduction dimension
    if (cols <= 0) {
        return false;
    }

    mlir::Value reduceResult = reduce->getResult(0);
    if (!reduceResult.hasOneUse()) {
        return false;
    }

    mlir::Operation* userOp = *reduceResult.getUsers().begin();
    if (!userOp) {
        return false;
    }

    if (auto divOp = mlir::dyn_cast<mlir::stablehlo::DivOp>(userOp)) {
        if (divOp.getLhs() != reduceResult) {
            return false;
        }
        std::optional<double> divisor = TraceScalarConstant(divOp.getRhs());
        if (!divisor.has_value()) {
            return false;
        }
        double expected = static_cast<double>(cols);
        if (std::abs(*divisor - expected) > 1e-6 * std::abs(expected)) {
            return false;
        }
        if (reduceResult.getType() != divOp.getResult().getType()) {
            return false;
        }
        final_result = divOp.getResult();
        absorbed_ops.push_back(userOp);
        if (auto* rhsDef = divOp.getRhs().getDefiningOp()) {
            absorbed_ops.push_back(rhsDef);
            if (rhsDef->getNumOperands() >= 1) {
                if (auto* constDef = rhsDef->getOperand(0).getDefiningOp()) {
                    absorbed_ops.push_back(constDef);
                }
            }
        }
        return true;
    }
    if (auto mulOp = mlir::dyn_cast<mlir::stablehlo::MulOp>(userOp)) {
        mlir::Value otherOperand;
        if (mulOp.getLhs() == reduceResult) {
            otherOperand = mulOp.getRhs();
        } else if (mulOp.getRhs() == reduceResult) {
            otherOperand = mulOp.getLhs();
        } else {
            return false;
        }
        std::optional<double> factor = TraceScalarConstant(otherOperand);
        if (!factor.has_value()) {
            return false;
        }
        double expected = 1.0 / static_cast<double>(cols);
        if (std::abs(*factor - expected) > 1e-6 * std::abs(expected)) {
            return false;
        }
        if (reduceResult.getType() != mulOp.getResult().getType()) {
            return false;
        }
        final_result = mulOp.getResult();
        absorbed_ops.push_back(userOp);
        if (auto* otherDef = otherOperand.getDefiningOp()) {
            absorbed_ops.push_back(otherDef);
            if (otherDef->getNumOperands() >= 1) {
                if (auto* constDef = otherDef->getOperand(0).getDefiningOp()) {
                    absorbed_ops.push_back(constDef);
                }
            }
        }
        return true;
    }
    return false;
}

static void FuseReduceMeanPatterns(mlir::Block& block) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_REDUCE_MEAN_FUSION") ||
        !IntrinsicNativeKernelsEnabled()) {
        return;
    }

    struct Candidate {
        mlir::stablehlo::ReduceOp reduceOp;
        llvm::SmallVector<mlir::Operation*, 4> absorbed_ops;
        mlir::Value final_result;
    };
    llvm::SmallVector<Candidate, 4> candidates;

    for (mlir::Operation& operation : block) {
        auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(&operation);
        if (!reduceOp) {
            continue;
        }
        Candidate c;
        c.reduceOp = reduceOp;
        if (!IsReduceMeanPattern(&operation, c.absorbed_ops, c.final_result)) {
            continue;
        }
        // Determine correct kind based on reduce axis
        auto dims = reduceOp.getDimensions();
        auto inType = mlir::dyn_cast<mlir::RankedTensorType>(operation.getOperand(0).getType());
        IntrinsicNativeKind meanKind = IntrinsicNativeKind::kReduceMeanLastAxis;
        if (inType && dims.size() == 1 && dims[0] == 0 && inType.getRank() == 2) {
            meanKind = IntrinsicNativeKind::kReduceMeanAxis0;
        }
        if (!IsEligibleReduceNativeOp(&operation, meanKind)) {
            continue;
        }
        candidates.push_back(std::move(c));
    }

    for (auto& c : candidates) {
        mlir::Value reduceResult = c.reduceOp->getResult(0);

        MPS_LOG_INFO("reduce_mean fusion: fusing reduce_sum + divide\n");

        c.final_result.replaceAllUsesWith(reduceResult);

        mlir::OpBuilder builder(c.reduceOp);
        c.reduceOp->setAttr("__mean_fused", builder.getUnitAttr());

        for (auto* absOp : c.absorbed_ops) {
            if (absOp->use_empty()) {
                absOp->erase();
            }
        }
    }
}

// JAX's f16 softmax widens exp to f32 for the row sum and narrows the sum
// back to f16 before the divide. It is fused into softmax_full_f16_xla_tg,
// which reproduces XLA:CPU's rounding points bit for bit (MPSGraph's fused
// sub->exp skipped the f16 rounding of x - max: up to 9 ulp off).
// JAX_METALLIB_DISABLE_F16_WIDENED_SOFTMAX_FUSION=1 keeps the graph path.
static bool F16WidenedSoftmaxFusionEnabled() {
    static const bool enabled = !EnvFlagEnabled("JAX_METALLIB_DISABLE_F16_WIDENED_SOFTMAX_FUSION");
    return enabled;
}

static bool IsZeroFloatSplat(mlir::Value v) {
    auto c = v.getDefiningOp<mlir::stablehlo::ConstantOp>();
    if (!c) {
        return false;
    }
    auto attr = mlir::dyn_cast<mlir::DenseFPElementsAttr>(c.getValue());
    return attr && attr.isSplat() && attr.getSplatValue<llvm::APFloat>().isZero();
}

static mlir::Value TraceBroadcastChain(mlir::Value start, llvm::SmallVectorImpl<mlir::Operation*>& absorbed) {
    mlir::Value cur = start;
    for (int step = 0; step < 8; ++step) {
        if (!cur.hasOneUse()) {
            return cur;
        }
        mlir::Operation* user = *cur.getUsers().begin();
        if (!user)
            return cur;
        if (mlir::isa<mlir::stablehlo::BroadcastInDimOp>(user) ||
            mlir::isa<mlir::stablehlo::ReshapeOp>(user) || mlir::isa<mlir::stablehlo::BroadcastOp>(user)) {
            absorbed.push_back(user);
            cur = user->getResult(0);
            continue;
        }
        if (auto maxOp = mlir::dyn_cast<mlir::stablehlo::MaxOp>(user)) {
            if (maxOp.getLhs() != cur && maxOp.getRhs() != cur) {
                return cur;
            }
            mlir::Value other = (maxOp.getLhs() == cur) ? maxOp.getRhs() : maxOp.getLhs();
            absorbed.push_back(user);
            {
                mlir::Value chain = other;
                for (int ci = 0; ci < 8 && chain; ++ci) {
                    auto* chainDef = chain.getDefiningOp();
                    if (!chainDef)
                        break;
                    if (!chainDef->getResult(0).hasOneUse())
                        break;
                    absorbed.push_back(chainDef);
                    if (mlir::isa<mlir::stablehlo::ConstantOp>(chainDef))
                        break;
                    if (chainDef->getNumOperands() >= 1) {
                        chain = chainDef->getOperand(0);
                    } else {
                        break;
                    }
                }
            }
            cur = user->getResult(0);
            continue;
        }
        return cur;
    }
    return cur;
}

// Detect LayerNorm pattern:
//   mean = reduce_sum(x, axis=-1) / N
//   var  = reduce_sum((x - mean)^2, axis=-1) / N
//   y    = (x - mean) / sqrt(var + eps)
// The HLO IR is:
//   reduce_sum(x) → broadcast → divide_by_N  (mean)
//   broadcast(mean) → subtract(x, mean) → multiply(diff, diff) →
//   reduce_sum(sq) → broadcast → divide_by_N  (var)
//   add(var, eps) → sqrt → broadcast → divide(diff, sqrt_var_eps)
static bool IsLayerNormPattern(mlir::Operation* reduceSumOp,
                               llvm::SmallVectorImpl<mlir::Operation*>& absorbed_ops, mlir::Value& ln_input,
                               mlir::Value& final_result, float& eps_value) {
    auto reduce = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(reduceSumOp);
    if (!reduce)
        return false;

    std::string bodyOp = GetReduceBodyOpName(reduce);
    if (bodyOp != "stablehlo.add")
        return false;
    if (reduce->getNumOperands() < 1 || reduce->getNumResults() != 1)
        return false;

    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(reduce->getOperand(0).getType());
    if (!inType || !inType.hasStaticShape() || inType.getRank() == 0)
        return false;
    auto dims = reduce.getDimensions();
    if (dims.size() != 1 || dims[0] != inType.getRank() - 1)
        return false;
    if (!IsFloat16Or32(MlirTypeToMps(inType.getElementType())))
        return false;

    ln_input = reduce->getOperand(0);
    mlir::Value reduceResult = reduce->getResult(0);

    // Follow reduce_sum → broadcast_in_dim → divide (mean = sum / N)
    llvm::SmallVector<mlir::Operation*, 8> meanBroadcastOps;
    mlir::Value broadcastedSum = TraceBroadcastChain(reduceResult, meanBroadcastOps);

    // broadcastedSum should be used in a divide to compute mean
    mlir::stablehlo::DivOp meanDivOp = nullptr;
    for (auto* user : broadcastedSum.getUsers()) {
        if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(user)) {
            if (div.getLhs() == broadcastedSum) {
                meanDivOp = div;
                break;
            }
        }
    }
    if (!meanDivOp)
        return false;

    mlir::Value meanValue = meanDivOp.getResult();

    // Absorb the broadcast chain + constant divisor + div op
    for (auto* op : meanBroadcastOps)
        absorbed_ops.push_back(op);
    if (auto* constDef = meanDivOp.getRhs().getDefiningOp())
        absorbed_ops.push_back(constDef);
    absorbed_ops.push_back(meanDivOp.getOperation());

    // mean may be broadcast before subtract
    llvm::SmallVector<mlir::Operation*, 8> meanBcast2;
    mlir::Value broadcastedMean = TraceBroadcastChain(meanValue, meanBcast2);
    for (auto* op : meanBcast2)
        absorbed_ops.push_back(op);

    // Find subtract: x - mean (may appear multiple times)
    mlir::stablehlo::SubtractOp subOp1 = nullptr;
    mlir::stablehlo::SubtractOp subOp2 = nullptr;
    for (auto* user : broadcastedMean.getUsers()) {
        if (auto sub = mlir::dyn_cast<mlir::stablehlo::SubtractOp>(user)) {
            if (sub.getLhs() == ln_input && sub.getRhs() == broadcastedMean) {
                if (!subOp1)
                    subOp1 = sub;
                else if (!subOp2)
                    subOp2 = sub;
            }
        }
    }
    if (!subOp1)
        return false;

    // One of the subtract results goes to multiply(diff, diff) for variance
    // The other goes to the final divide for normalization
    mlir::stablehlo::SubtractOp varSubOp = nullptr;
    mlir::stablehlo::SubtractOp normSubOp = nullptr;

    auto checkVariancePath = [&](mlir::stablehlo::SubtractOp sub) -> bool {
        mlir::Value subResult = sub.getResult();
        for (auto* user : subResult.getUsers()) {
            if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user)) {
                if (mul.getLhs() == subResult && mul.getRhs() == subResult) {
                    return true;
                }
            }
        }
        return false;
    };

    if (checkVariancePath(subOp1)) {
        varSubOp = subOp1;
        normSubOp = subOp2 ? subOp2 : subOp1; // If only one subtract, it serves both
    } else if (subOp2 && checkVariancePath(subOp2)) {
        varSubOp = subOp2;
        normSubOp = subOp1;
    } else {
        return false;
    }

    absorbed_ops.push_back(varSubOp.getOperation());
    if (normSubOp != varSubOp) {
        absorbed_ops.push_back(normSubOp.getOperation());
    }

    // Follow variance path: subtract → multiply(diff, diff) → reduce_sum → broadcast → divide
    mlir::Value varSubResult = varSubOp.getResult();
    mlir::stablehlo::MulOp squareOp = nullptr;
    for (auto* user : varSubResult.getUsers()) {
        if (auto mul = mlir::dyn_cast<mlir::stablehlo::MulOp>(user)) {
            if (mul.getLhs() == varSubResult && mul.getRhs() == varSubResult) {
                squareOp = mul;
                break;
            }
        }
    }
    if (!squareOp)
        return false;
    absorbed_ops.push_back(squareOp.getOperation());

    mlir::Value squareResult = squareOp.getResult();
    // Should feed into reduce_sum
    mlir::stablehlo::ReduceOp varReduceOp = nullptr;
    for (auto* user : squareResult.getUsers()) {
        if (auto red = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
            std::string body = GetReduceBodyOpName(red);
            if (body == "stablehlo.add") {
                varReduceOp = red;
                break;
            }
        }
    }
    if (!varReduceOp || varReduceOp->getNumResults() != 1)
        return false;
    absorbed_ops.push_back(varReduceOp.getOperation());

    // variance reduce → broadcast → divide (var = sum_sq / N)
    mlir::Value varReduceResult = varReduceOp->getResult(0);
    llvm::SmallVector<mlir::Operation*, 8> varBroadcastOps;
    mlir::Value broadcastedVarSum = TraceBroadcastChain(varReduceResult, varBroadcastOps);
    for (auto* op : varBroadcastOps)
        absorbed_ops.push_back(op);

    mlir::stablehlo::DivOp varDivOp = nullptr;
    for (auto* user : broadcastedVarSum.getUsers()) {
        if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(user)) {
            if (div.getLhs() == broadcastedVarSum) {
                varDivOp = div;
                break;
            }
        }
    }
    if (!varDivOp)
        return false;
    if (auto* constDef = varDivOp.getRhs().getDefiningOp())
        absorbed_ops.push_back(constDef);
    absorbed_ops.push_back(varDivOp.getOperation());

    mlir::Value variance = varDivOp.getResult();

    // var + eps → sqrt → broadcast → divide(diff, sqrt)
    mlir::stablehlo::AddOp epsAddOp = nullptr;
    for (auto* user : variance.getUsers()) {
        if (auto add = mlir::dyn_cast<mlir::stablehlo::AddOp>(user)) {
            epsAddOp = add;
            break;
        }
    }
    if (!epsAddOp)
        return false;

    // Extract epsilon value
    mlir::Value epsOperand = (epsAddOp.getLhs() == variance) ? epsAddOp.getRhs() : epsAddOp.getLhs();
    double epsDouble = 0.0;
    if (!GetScalarFloatConstant(epsOperand, &epsDouble)) {
        // eps might be broadcast from a constant
        if (auto* def = epsOperand.getDefiningOp()) {
            if (mlir::isa<mlir::stablehlo::BroadcastInDimOp>(def) && def->getNumOperands() >= 1) {
                if (!GetScalarFloatConstant(def->getOperand(0), &epsDouble))
                    return false;
                absorbed_ops.push_back(def);
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    eps_value = static_cast<float>(epsDouble);
    if (auto* epsDef = epsOperand.getDefiningOp())
        absorbed_ops.push_back(epsDef);
    absorbed_ops.push_back(epsAddOp.getOperation());

    mlir::Value varPlusEps = epsAddOp.getResult();

    // sqrt(var + eps)
    mlir::stablehlo::SqrtOp sqrtOp = nullptr;
    for (auto* user : varPlusEps.getUsers()) {
        if (auto sq = mlir::dyn_cast<mlir::stablehlo::SqrtOp>(user)) {
            sqrtOp = sq;
            break;
        }
    }
    if (!sqrtOp)
        return false;
    absorbed_ops.push_back(sqrtOp.getOperation());

    mlir::Value sqrtResult = sqrtOp.getResult();

    // sqrt result broadcast
    llvm::SmallVector<mlir::Operation*, 8> sqrtBroadcastOps;
    mlir::Value broadcastedSqrt = TraceBroadcastChain(sqrtResult, sqrtBroadcastOps);
    for (auto* op : sqrtBroadcastOps)
        absorbed_ops.push_back(op);

    // final divide: (x - mean) / sqrt(var + eps)
    mlir::stablehlo::DivOp finalDivOp = nullptr;
    for (auto* user : broadcastedSqrt.getUsers()) {
        if (auto div = mlir::dyn_cast<mlir::stablehlo::DivOp>(user)) {
            if (div.getRhs() == broadcastedSqrt) {
                finalDivOp = div;
                break;
            }
        }
    }
    if (!finalDivOp)
        return false;

    // LHS of final divide should be (x - mean)
    mlir::Value normSubResult = normSubOp ? normSubOp.getResult() : varSubOp.getResult();
    if (finalDivOp.getLhs() != normSubResult) {
        // Could be that the diff was broadcast
        llvm::SmallVector<mlir::Operation*, 4> diffBcast;
        mlir::Value bcastDiff = TraceBroadcastChain(normSubResult, diffBcast);
        if (finalDivOp.getLhs() != bcastDiff)
            return false;
        for (auto* op : diffBcast)
            absorbed_ops.push_back(op);
    }

    absorbed_ops.push_back(finalDivOp.getOperation());

    // Absorb reduce init values
    if (reduce->getNumOperands() >= 2) {
        if (auto* initDef = reduce->getOperand(1).getDefiningOp())
            absorbed_ops.push_back(initDef);
    }
    if (varReduceOp->getNumOperands() >= 2) {
        if (auto* initDef = varReduceOp->getOperand(1).getDefiningOp())
            absorbed_ops.push_back(initDef);
    }

    final_result = finalDivOp.getResult();
    return true;
}

static void FuseLayerNormPatterns(mlir::Block& block) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_LAYERNORM_FUSION") ||
        !IntrinsicNativeKernelsEnabled()) {
        return;
    }

    struct Candidate {
        mlir::stablehlo::ReduceOp reduceOp;
        llvm::SmallVector<mlir::Operation*, 32> absorbed_ops;
        mlir::Value ln_input;
        mlir::Value final_result;
        float eps;
    };
    llvm::SmallVector<Candidate, 4> candidates;

    for (mlir::Operation& operation : block) {
        auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(&operation);
        if (!reduceOp)
            continue;
        std::string bodyOp = GetReduceBodyOpName(reduceOp);
        if (bodyOp != "stablehlo.add")
            continue;

        Candidate c;
        c.reduceOp = reduceOp;
        if (!IsLayerNormPattern(&operation, c.absorbed_ops, c.ln_input, c.final_result, c.eps)) {
            continue;
        }
        // Ensure the input is eligible for native reduce
        if (!IsEligibleReduceNativeOp(&operation, IntrinsicNativeKind::kReduceSumLastAxis)) {
            continue;
        }
        uint64_t inputNumel = 0;
        if (!GetStaticTensorInfo(c.ln_input.getType(), nullptr, &inputNumel, nullptr) ||
            inputNumel < (uint64_t)EnvIntValue("JAX_METALLIB_LAYERNORM_FUSION_MIN_ELEMENTS", 131072)) {
            continue;
        }
        candidates.push_back(std::move(c));
    }

    for (auto& c : candidates) {
        llvm::SmallPtrSet<mlir::Operation*, 32> absorbedSet;
        for (auto* op : c.absorbed_ops)
            absorbedSet.insert(op);
        mlir::Operation* finalOp = c.final_result.getDefiningOp();

        // Check that absorbed ops don't have external users
        bool canFuse = true;
        for (auto* op : c.absorbed_ops) {
            if (op == finalOp)
                continue;
            for (mlir::Value res : op->getResults()) {
                for (mlir::Operation* user : res.getUsers()) {
                    if (user == finalOp)
                        continue;
                    if (absorbedSet.count(user))
                        continue;
                    if (user == c.reduceOp.getOperation())
                        continue;
                    canFuse = false;
                    break;
                }
                if (!canFuse)
                    break;
            }
            if (!canFuse)
                break;
        }
        if (!canFuse)
            continue;

        MPS_LOG_INFO("layernorm fusion: fusing reduce_sum + var + sqrt + div (eps=%g)\n", c.eps);

        // Rewrite: the first reduce_sum becomes the fused layernorm op
        mlir::Value reduceResult = c.reduceOp->getResult(0);
        reduceResult.setType(c.final_result.getType());
        c.final_result.replaceAllUsesWith(reduceResult);

        mlir::OpBuilder builder(c.reduceOp);
        c.reduceOp->setAttr("__layernorm_fused", builder.getUnitAttr());
        // Store eps as a float attribute
        c.reduceOp->setAttr("__layernorm_eps", builder.getF32FloatAttr(c.eps));

        // Erase absorbed ops
        llvm::SmallPtrSet<mlir::Operation*, 32> eraseSet;
        for (auto* op : c.absorbed_ops)
            eraseSet.insert(op);
        bool progress = true;
        while (progress) {
            progress = false;
            llvm::SmallVector<mlir::Operation*, 32> remaining;
            for (auto* op : eraseSet) {
                if (op->use_empty()) {
                    op->erase();
                    progress = true;
                } else {
                    remaining.push_back(op);
                }
            }
            eraseSet.clear();
            for (auto* op : remaining)
                eraseSet.insert(op);
        }
    }
}

// The softmax matchers follow a last-axis reduce_max / reduce_sum result through
// broadcast_in_dim / reshape / broadcast (and elementwise max / convert) back to
// the input shape. The native kernel reduces and rebroadcasts along the last
// axis, so the chain must put every leading input axis back where it came from.
// Without this check x - max(x, axis=1)[None, :] on a square input was fused as
// a row softmax (relative error ~6e2 vs CPU at n=1024). Axes of extent 1 are
// not tracked: where they land cannot change any value.
static bool BroadcastChainRestoresLeadingAxes(mlir::Value reduced, mlir::Value end) {
    auto redType = mlir::dyn_cast<mlir::RankedTensorType>(reduced.getType());
    auto endType = mlir::dyn_cast<mlir::RankedTensorType>(end.getType());
    if (!redType || !endType || !redType.hasStaticShape() || !endType.hasStaticShape() ||
        endType.getRank() != redType.getRank() + 1) {
        return false;
    }
    // pos[k]: current axis of reduced axis k, or -1 when it has extent 1.
    llvm::SmallVector<int64_t, 8> pos;
    for (int64_t k = 0; k < redType.getRank(); ++k) {
        pos.push_back(redType.getDimSize(k) == 1 ? -1 : k);
    }
    mlir::Value cur = reduced;
    for (int step = 0; step < 32 && cur != end; ++step) {
        if (!cur.hasOneUse()) {
            return false;
        }
        mlir::Operation* user = *cur.getUsers().begin();
        auto inT = mlir::dyn_cast<mlir::RankedTensorType>(cur.getType());
        auto outT = mlir::dyn_cast<mlir::RankedTensorType>(user->getResult(0).getType());
        if (!inT || !outT || !inT.hasStaticShape() || !outT.hasStaticShape()) {
            return false;
        }
        if (auto bc = mlir::dyn_cast<mlir::stablehlo::BroadcastInDimOp>(user)) {
            auto dims = bc.getBroadcastDimensions();
            for (int64_t& p : pos) {
                if (p < 0) {
                    continue;
                }
                if (p >= static_cast<int64_t>(dims.size())) {
                    return false;
                }
                p = dims[p];
            }
        } else if (mlir::isa<mlir::stablehlo::BroadcastOp>(user)) {
            const int64_t added = outT.getRank() - inT.getRank();
            if (added < 0) {
                return false;
            }
            for (int64_t& p : pos) {
                if (p >= 0) {
                    p += added;
                }
            }
        } else if (mlir::isa<mlir::stablehlo::ReshapeOp>(user)) {
            // Only reshapes that add or drop size-1 axes (non-1 extents keep their order).
            llvm::SmallVector<int64_t, 8> inAxes;
            llvm::SmallVector<int64_t, 8> outAxes;
            for (int64_t a = 0; a < inT.getRank(); ++a) {
                if (inT.getDimSize(a) != 1) {
                    inAxes.push_back(a);
                }
            }
            for (int64_t a = 0; a < outT.getRank(); ++a) {
                if (outT.getDimSize(a) != 1) {
                    outAxes.push_back(a);
                }
            }
            if (inAxes.size() != outAxes.size()) {
                return false;
            }
            for (size_t i = 0; i < inAxes.size(); ++i) {
                if (inT.getDimSize(inAxes[i]) != outT.getDimSize(outAxes[i])) {
                    return false;
                }
            }
            for (int64_t& p : pos) {
                if (p < 0) {
                    continue;
                }
                auto it = llvm::find(inAxes, p);
                if (it == inAxes.end()) {
                    return false;
                }
                p = outAxes[static_cast<size_t>(it - inAxes.begin())];
            }
        } else if (mlir::isa<mlir::stablehlo::MaxOp, mlir::stablehlo::ConvertOp>(user)) {
            if (inT.getShape() != outT.getShape()) {
                return false;
            }
        } else {
            return false;
        }
        cur = user->getResult(0);
    }
    if (cur != end) {
        return false;
    }
    for (int64_t k = 0; k < static_cast<int64_t>(pos.size()); ++k) {
        if (pos[k] >= 0 && pos[k] != k) {
            return false;
        }
    }
    return true;
}

static bool IsSoftmaxPattern(mlir::Operation* reduceMaxOp,
                             llvm::SmallVectorImpl<mlir::Operation*>& absorbed_ops,
                             mlir::Value& softmax_input, mlir::Value& final_result,
                             bool* widenedF16Out = nullptr) {
    if (widenedF16Out) {
        *widenedF16Out = false;
    }
    auto reduce = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(reduceMaxOp);
    if (!reduce)
        return false;

    std::string bodyOp = GetReduceBodyOpName(reduce);
    if (bodyOp != "stablehlo.maximum")
        return false;
    if (reduce->getNumOperands() < 1 || reduce->getNumResults() != 1)
        return false;

    auto inType = mlir::dyn_cast<mlir::RankedTensorType>(reduce->getOperand(0).getType());
    if (!inType || !inType.hasStaticShape() || inType.getRank() == 0)
        return false;
    auto dims = reduce.getDimensions();
    if (dims.size() != 1 || dims[0] != inType.getRank() - 1)
        return false;
    if (!IsFloat16Or32(MlirTypeToMps(inType.getElementType())))
        return false;

    softmax_input = reduce->getOperand(0);
    mlir::Value reduceResult = reduce->getResult(0);

    llvm::SmallVector<mlir::Operation*, 8> broadcastOps;
    mlir::Value broadcastedMax = TraceBroadcastChain(reduceResult, broadcastOps);
    if (!broadcastedMax.hasOneUse()) {
        return false;
    }
    if (!BroadcastChainRestoresLeadingAxes(reduceResult, broadcastedMax)) {
        return false;
    }

    mlir::Operation* subUser = *broadcastedMax.getUsers().begin();
    auto subOp = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(subUser);
    if (!subOp) {
        return false;
    }
    if (subOp.getLhs() != softmax_input || subOp.getRhs() != broadcastedMax) {
        return false;
    }
    mlir::Value subResult = subOp.getResult();
    if (!subResult.hasOneUse()) {
        return false;
    }

    mlir::Operation* expUser = *subResult.getUsers().begin();
    auto expOp = mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(expUser);
    if (!expOp) {
        return false;
    }
    mlir::Value expResult = expOp.getResult();

    int expUseCount = 0;
    for ([[maybe_unused]] auto& u : expResult.getUses())
        expUseCount++;
    if (expUseCount != 2) {
        return false;
    }

    mlir::stablehlo::ReduceOp reduceSumOp = nullptr;
    mlir::stablehlo::DivOp divOp = nullptr;
    mlir::stablehlo::ConvertOp widenOp = nullptr;
    for (auto* user : expResult.getUsers()) {
        if (auto rs = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(user)) {
            reduceSumOp = rs;
        } else if (auto dv = mlir::dyn_cast<mlir::stablehlo::DivOp>(user)) {
            divOp = dv;
        } else if (auto cv = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(user)) {
            widenOp = cv;
        }
    }
    // f16: exp -> convert f32 (single use) -> reduce_sum(add, zero init).
    // Rows over 8192 would need the two-pass kernel, which cannot round the
    // sum like XLA, so they stay on the graph path.
    bool widened = false;
    if (!reduceSumOp && widenOp && divOp && F16WidenedSoftmaxFusionEnabled() &&
        inType.getElementType().isF16() && inType.getShape().back() <= 8192) {
        auto wOut = mlir::dyn_cast<mlir::RankedTensorType>(widenOp->getResult(0).getType());
        if (wOut && wOut.getElementType().isF32() && widenOp->getResult(0).hasOneUse()) {
            reduceSumOp =
                mlir::dyn_cast<mlir::stablehlo::ReduceOp>(*widenOp->getResult(0).getUsers().begin());
            widened = static_cast<bool>(reduceSumOp);
        }
        if (widened && (reduceSumOp->getNumOperands() < 2 || !IsZeroFloatSplat(reduceSumOp->getOperand(1)))) {
            return false;
        }
    }
    if (!reduceSumOp || !divOp)
        return false;

    std::string sumBody = GetReduceBodyOpName(reduceSumOp);
    if (sumBody != "stablehlo.add")
        return false;
    auto sumDims = reduceSumOp.getDimensions();
    if (sumDims.size() != 1 || sumDims[0] != dims[0])
        return false;
    if (reduceSumOp->getNumResults() != 1)
        return false;
    mlir::Value sumResult = reduceSumOp->getResult(0);

    llvm::SmallVector<mlir::Operation*, 8> sumBroadcastOps;
    mlir::Value broadcastedSum = TraceBroadcastChain(sumResult, sumBroadcastOps);
    if (!broadcastedSum.hasOneUse())
        return false;
    if (widened) {
        auto narrow = mlir::dyn_cast<mlir::stablehlo::ConvertOp>(*broadcastedSum.getUsers().begin());
        if (!narrow) {
            return false;
        }
        auto nOut = mlir::dyn_cast<mlir::RankedTensorType>(narrow->getResult(0).getType());
        if (!nOut || !nOut.getElementType().isF16()) {
            return false;
        }
        sumBroadcastOps.push_back(narrow.getOperation());
        broadcastedSum = TraceBroadcastChain(narrow->getResult(0), sumBroadcastOps);
        if (!broadcastedSum.hasOneUse()) {
            return false;
        }
    }

    if (!BroadcastChainRestoresLeadingAxes(sumResult, broadcastedSum)) {
        return false;
    }

    mlir::Operation* divUser = *broadcastedSum.getUsers().begin();
    if (divUser != divOp.getOperation()) {
        divOp = mlir::dyn_cast_or_null<mlir::stablehlo::DivOp>(divUser);
        if (!divOp)
            return false;
    }
    if (divOp.getLhs() != expResult || divOp.getRhs() != broadcastedSum)
        return false;

    for (auto* op : broadcastOps)
        absorbed_ops.push_back(op);
    absorbed_ops.push_back(subOp);
    absorbed_ops.push_back(expOp);
    if (widened) {
        absorbed_ops.push_back(widenOp.getOperation());
    }
    absorbed_ops.push_back(reduceSumOp.getOperation());
    for (auto* op : sumBroadcastOps)
        absorbed_ops.push_back(op);
    absorbed_ops.push_back(divOp.getOperation());

    if (reduce->getNumOperands() >= 2) {
        if (auto* initDef = reduce->getOperand(1).getDefiningOp()) {
            absorbed_ops.push_back(initDef);
        }
    }
    if (reduceSumOp->getNumOperands() >= 2) {
        if (auto* initDef = reduceSumOp->getOperand(1).getDefiningOp()) {
            absorbed_ops.push_back(initDef);
        }
    }

    final_result = divOp.getResult();
    if (widenedF16Out) {
        *widenedF16Out = widened;
    }
    return true;
}

static void FuseSoftmaxPatterns(mlir::Block& block) {
    // NOTE: the fused softmax lowering produces a shape-invalid MPSGraph for
    // 2-D inputs (e.g. softmax(4096x64) aborts in mps.reshape verification),
    // so this pass must follow the intrinsic-kernel gate. Do not revert to
    // checking the raw DISABLE flag.
    if (!StandaloneSoftmaxEnabled()) {
        return;
    }

    struct Candidate {
        mlir::stablehlo::ReduceOp reduceMaxOp;
        llvm::SmallVector<mlir::Operation*, 16> absorbed_ops;
        mlir::Value softmax_input;
        mlir::Value final_result;
        bool widened_f16 = false;
    };
    llvm::SmallVector<Candidate, 4> candidates;

    for (mlir::Operation& operation : block) {
        auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(&operation);
        if (!reduceOp)
            continue;
        std::string bodyOp = GetReduceBodyOpName(reduceOp);
        if (bodyOp != "stablehlo.maximum")
            continue;

        Candidate c;
        c.reduceMaxOp = reduceOp;
        if (!IsSoftmaxPattern(&operation, c.absorbed_ops, c.softmax_input, c.final_result, &c.widened_f16)) {
            continue;
        }
        if (!IsEligibleReduceNativeOp(&operation, IntrinsicNativeKind::kReduceMaxLastAxis)) {
            continue;
        }
        uint64_t inputNumel = 0;
        if (!GetStaticTensorInfo(c.softmax_input.getType(), nullptr, &inputNumel, nullptr) ||
            inputNumel < (uint64_t)EnvIntValue("JAX_METALLIB_SOFTMAX_FUSION_MIN_ELEMENTS", 1024)) {
            continue;
        }
        // Multi-row f32 softmax with short rows stays in MPSGraph: the native
        // kernel split the plan into graph/native/graph segments, and between
        // matmuls (attention) the switch cost more than the kernel saved
        // ([2,8,128,32] attention 0.27 ms fused vs 0.22 ms graph; standalone
        // rows < 1024 are within ~0.02 ms either way, rows >= 1024 are 2x+
        // faster native). f16 keeps the native XLA-rounding kernel.
        // JAX_METALLIB_SOFTMAX_NATIVE_MIN_COLS=1 restores the old routing.
        if (!IntrinsicNativeKernelsEnabled() && !c.widened_f16) {
            auto inT = mlir::dyn_cast<mlir::RankedTensorType>(c.softmax_input.getType());
            if (inT && inT.hasStaticShape() && inT.getRank() >= 1 && inT.getElementType().isF32()) {
                const int64_t cols = inT.getDimSize(inT.getRank() - 1);
                const int64_t minCols = EnvIntValue("JAX_METALLIB_SOFTMAX_NATIVE_MIN_COLS", 1024);
                if (cols > 0 && static_cast<int64_t>(inputNumel) / cols > 1 && cols < minCols) {
                    continue;
                }
            }
        }
        candidates.push_back(std::move(c));
    }

    // --- Softmax numerator pattern: reduce_max → broadcast → sub → exp (final output) ---
    // This matches exp(x - max(x, axis=-1, keepdims=True)) where exp is the return value.
    // The numerator kernel has no standalone route, so it stays with the family.
    if (candidates.empty() && IntrinsicNativeKernelsEnabled()) {
        struct NumCandidate {
            mlir::stablehlo::ReduceOp reduceMaxOp;
            llvm::SmallVector<mlir::Operation*, 8> absorbed_ops;
            mlir::Value softmax_input;
            mlir::Value exp_result;
        };
        llvm::SmallVector<NumCandidate, 4> num_candidates;

        for (mlir::Operation& operation : block) {
            auto reduceOp = mlir::dyn_cast<mlir::stablehlo::ReduceOp>(&operation);
            if (!reduceOp)
                continue;
            std::string bodyOp = GetReduceBodyOpName(reduceOp);
            if (bodyOp != "stablehlo.maximum")
                continue;
            if (!IsEligibleReduceNativeOp(&operation, IntrinsicNativeKind::kReduceMaxLastAxis))
                continue;

            // Check: reduce_max result has one use (broadcast_in_dim)
            mlir::Value reduceResult = operation.getResult(0);
            if (!reduceResult.hasOneUse())
                continue;
            mlir::Operation* broadcastUser = *reduceResult.getUsers().begin();
            auto broadcastOp = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(broadcastUser);
            if (!broadcastOp)
                continue;

            mlir::Value broadcastedMax = broadcastOp.getResult();
            // Follow chain of broadcast_in_dim ops (keepdims may produce multiple)
            llvm::SmallVector<mlir::Operation*, 4> extraBroadcasts;
            while (broadcastedMax.hasOneUse()) {
                mlir::Operation* user = *broadcastedMax.getUsers().begin();
                auto nextBroadcast = mlir::dyn_cast_or_null<mlir::stablehlo::BroadcastInDimOp>(user);
                if (!nextBroadcast)
                    break;
                extraBroadcasts.push_back(nextBroadcast);
                broadcastedMax = nextBroadcast.getResult();
            }
            if (!broadcastedMax.hasOneUse())
                continue;

            // Check: subtract x - broadcast(max)
            mlir::Operation* subUser = *broadcastedMax.getUsers().begin();
            auto subOp = mlir::dyn_cast_or_null<mlir::stablehlo::SubtractOp>(subUser);
            if (!subOp)
                continue;

            // Identify the softmax input: lhs of subtract
            mlir::Value softmax_input = subOp.getLhs();
            if (subOp.getRhs() != broadcastedMax)
                continue;
            if (!BroadcastChainRestoresLeadingAxes(reduceResult, broadcastedMax))
                continue;

            mlir::Value subResult = subOp.getResult();
            if (!subResult.hasOneUse())
                continue;

            // Check: exp(sub_result)
            mlir::Operation* expUser = *subResult.getUsers().begin();
            auto expOp = mlir::dyn_cast_or_null<mlir::stablehlo::ExpOp>(expUser);
            if (!expOp)
                continue;

            mlir::Value expResult = expOp.getResult();
            // exp result must have EXACTLY 1 use, and that use must be a
            // return/func_return op. This prevents matching partial softmax
            // patterns where exp feeds into reduce_sum.
            int expUseCount = 0;
            for ([[maybe_unused]] auto& u : expResult.getUses())
                expUseCount++;
            if (expUseCount != 1)
                continue;
            // Verify the single use is a return-like op
            mlir::Operation* expSingleUser = *expResult.getUsers().begin();
            if (!mlir::isa<mlir::func::ReturnOp>(expSingleUser) &&
                !mlir::isa<mlir::stablehlo::ReturnOp>(expSingleUser)) {
                // Not a direct return — could be part of a larger pattern
                continue;
            }

            // Also check: the reduce input (operand 0 of reduceOp) comes from softmax_input
            if (operation.getOperand(0) != softmax_input)
                continue;

            uint64_t inputNumel = 0;
            if (!GetStaticTensorInfo(softmax_input.getType(), nullptr, &inputNumel, nullptr) ||
                inputNumel < (uint64_t)EnvIntValue("JAX_METALLIB_SOFTMAX_FUSION_MIN_ELEMENTS", 1024))
                continue;

            NumCandidate nc;
            nc.reduceMaxOp = reduceOp;
            nc.softmax_input = softmax_input;
            nc.exp_result = expResult;
            // Absorb: broadcasts, sub, exp, and their init value ops
            nc.absorbed_ops.push_back(broadcastOp);
            for (auto* eb : extraBroadcasts)
                nc.absorbed_ops.push_back(eb);
            nc.absorbed_ops.push_back(subOp);
            nc.absorbed_ops.push_back(expOp);
            if (auto* initDef = reduceOp->getOperand(1).getDefiningOp()) {
                nc.absorbed_ops.push_back(initDef);
            }
            num_candidates.push_back(std::move(nc));
        }

        for (auto& nc : num_candidates) {
            MPS_LOG_INFO("softmax_numerator fusion: fusing reduce_max + sub + exp\n");

            mlir::Value reduceResult = nc.reduceMaxOp->getResult(0);
            reduceResult.setType(nc.exp_result.getType());
            nc.exp_result.replaceAllUsesWith(reduceResult);

            mlir::OpBuilder builder(nc.reduceMaxOp);
            nc.reduceMaxOp->setAttr("__softmax_numerator_fused", builder.getUnitAttr());

            llvm::SmallPtrSet<mlir::Operation*, 8> eraseSet;
            for (auto* op : nc.absorbed_ops)
                eraseSet.insert(op);
            bool progress = true;
            while (progress) {
                progress = false;
                llvm::SmallVector<mlir::Operation*, 8> remaining;
                for (auto* op : eraseSet) {
                    if (op->use_empty()) {
                        op->erase();
                        progress = true;
                    } else {
                        remaining.push_back(op);
                    }
                }
                eraseSet.clear();
                for (auto* op : remaining)
                    eraseSet.insert(op);
            }
        }
    }

    for (auto& c : candidates) {
        llvm::SmallPtrSet<mlir::Operation*, 16> absorbedSet;
        for (auto* op : c.absorbed_ops)
            absorbedSet.insert(op);
        mlir::Operation* divOp = c.final_result.getDefiningOp();

        bool canFuse = true;
        for (auto* op : c.absorbed_ops) {
            if (op == divOp)
                continue;
            for (mlir::Value res : op->getResults()) {
                for (mlir::Operation* user : res.getUsers()) {
                    if (user == divOp)
                        continue;
                    if (absorbedSet.count(user))
                        continue;
                    if (user == c.reduceMaxOp.getOperation())
                        continue;
                    MPS_LOG_INFO("softmax fusion: skipping — absorbed op has external user\n");
                    canFuse = false;
                    break;
                }
                if (!canFuse)
                    break;
            }
            if (!canFuse)
                break;
        }
        if (canFuse) {
            for (auto* op : c.absorbed_ops) {
                for (mlir::Value operand : op->getOperands()) {
                    if (auto* def = operand.getDefiningOp()) {
                        if (absorbedSet.count(def)) {
                            for (mlir::Operation* user : def->getResult(0).getUsers()) {
                                if (absorbedSet.count(user))
                                    continue;
                                if (user == c.reduceMaxOp.getOperation())
                                    continue;
                            }
                        }
                    }
                }
            }
        }
        if (!canFuse) {
            continue;
        }

        MPS_LOG_INFO("softmax fusion: fusing reduce_max + sub + exp + reduce_sum + div\n");

        mlir::Value reduceResult = c.reduceMaxOp->getResult(0);
        reduceResult.setType(c.final_result.getType());
        c.final_result.replaceAllUsesWith(reduceResult);

        mlir::OpBuilder builder(c.reduceMaxOp);
        c.reduceMaxOp->setAttr("__softmax_fused", builder.getUnitAttr());
        if (c.widened_f16) {
            c.reduceMaxOp->setAttr("__softmax_f16_xla", builder.getUnitAttr());
        }

        llvm::SmallPtrSet<mlir::Operation*, 16> eraseSet;
        for (auto* op : c.absorbed_ops)
            eraseSet.insert(op);
        bool progress = true;
        while (progress) {
            progress = false;
            llvm::SmallVector<mlir::Operation*, 16> remaining;
            for (auto* op : eraseSet) {
                if (op->use_empty()) {
                    op->erase();
                    progress = true;
                } else {
                    remaining.push_back(op);
                }
            }
            eraseSet.clear();
            for (auto* op : remaining)
                eraseSet.insert(op);
        }
    }
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

// True when the reduce dimensions are exactly the trailing axes {rank-k .. rank-1}.
// A suffix reduce over contiguous row-major data is a per-row reduction with
// cols = product(suffix dims), so the last-axis native kernels handle it
// (a full reduce to a scalar is the special case rows == 1).
static bool IsSuffixReduceDimensions(llvm::ArrayRef<int64_t> dims, int64_t rank) {
    if (dims.empty() || rank <= 0 || static_cast<int64_t>(dims.size()) > rank) {
        return false;
    }
    int64_t first = rank - static_cast<int64_t>(dims.size());
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] != first + static_cast<int64_t>(i)) {
            return false;
        }
    }
    return true;
}

static IntrinsicNativeKind ClassifyReduceKind(mlir::Operation* op) {
    auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
    if (!reduceOp) {
        return IntrinsicNativeKind::kNone;
    }
    if (op->hasAttr("__softmax_fused")) {
        return IntrinsicNativeKind::kSoftmaxLastAxis;
    }
    if (op->hasAttr("__softmax_numerator_fused")) {
        return IntrinsicNativeKind::kSoftmaxNumeratorLastAxis;
    }
    if (op->hasAttr("__layernorm_fused")) {
        return IntrinsicNativeKind::kLayerNormLastAxis;
    }
    if (op->hasAttr("__mean_fused")) {
        auto dims = reduceOp.getDimensions();
        auto inType = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
        if (inType && dims.size() == 1 && dims[0] == 0 && inType.getRank() == 2) {
            return IntrinsicNativeKind::kReduceMeanAxis0;
        }
        return IntrinsicNativeKind::kReduceMeanLastAxis;
    }
    // Plain last-axis sum: route to native per-row simd kernel (same family as
    // the mean path, which wins 1.05-1.09x). IsEligibleReduceNativeOp re-validates
    // axis/dtype/100K-element threshold. Max/min stay on MPSGraph (no headroom).
    if (auto inType2 = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType())) {
        auto dims2 = reduceOp.getDimensions();
        int64_t rank2 = static_cast<int64_t>(inType2.getRank());
        if (IsSuffixReduceDimensions(dims2, rank2)) {
            const std::string body2 = GetReduceBodyOpName(reduceOp);
            if (body2 == "stablehlo.add") {
                return IntrinsicNativeKind::kReduceSumLastAxis;
            }
            // Full reduce to a scalar: max/min also win big natively (the
            // MPSGraph full-reduce path is effectively serial, ~45ns/elem).
            if (static_cast<int64_t>(dims2.size()) == rank2) {
                if (body2 == "stablehlo.maximum") {
                    return IntrinsicNativeKind::kReduceMaxLastAxis;
                }
                if (body2 == "stablehlo.minimum") {
                    return IntrinsicNativeKind::kReduceMinLastAxis;
                }
            }
        }
    }
    // Plain reduce ops route to MPSGraph: the graph path fuses the entire
    // reduce into a single MPSGraphExecutable dispatch, avoiding the separate
    // command-buffer overhead of the native-only path.  Fused patterns
    // (softmax, layernorm, mean-fused) stay native because their multi-pass
    // kernels outperform the graph path.
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
    if (dims.empty()) {
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

    int64_t rank = static_cast<int64_t>(inShape.size());
    bool is_axis0_kind =
        (kind == IntrinsicNativeKind::kReduceSumAxis0 || kind == IntrinsicNativeKind::kReduceMaxAxis0 ||
         kind == IntrinsicNativeKind::kReduceMinAxis0 || kind == IntrinsicNativeKind::kReduceMeanAxis0);
    int64_t reduce_dim_size = 0;
    if (is_axis0_kind) {
        if (dims.size() != 1 || dims[0] != 0 || rank != 2) {
            return false;
        }
        reduce_dim_size = inShape[0];
    } else {
        // Trailing-suffix reduces (including full reduces to a scalar) are
        // per-row reductions over contiguous memory with cols = product(suffix).
        if (!IsSuffixReduceDimensions(dims, rank)) {
            return false;
        }
        if (dims.size() != 1 && kind != IntrinsicNativeKind::kReduceSumLastAxis &&
            kind != IntrinsicNativeKind::kReduceMaxLastAxis &&
            kind != IntrinsicNativeKind::kReduceMinLastAxis) {
            // Fused softmax/layernorm/mean kernels are single-axis only.
            return false;
        }
        reduce_dim_size = 1;
        for (auto d : dims) {
            reduce_dim_size *= inShape[d];
        }
    }
    if (reduce_dim_size <= 0 || reduce_dim_size > std::numeric_limits<uint32_t>::max()) {
        return false;
    }
    if (kind == IntrinsicNativeKind::kSoftmaxLastAxis || kind == IntrinsicNativeKind::kLayerNormLastAxis ||
        kind == IntrinsicNativeKind::kSoftmaxNumeratorLastAxis) {
        return outNumel == inNumel && IsNeutralReduceInit(reduceOp.getInitValues()[0], kind);
    }
    if (outNumel * static_cast<uint64_t>(reduce_dim_size) != inNumel) {
        return false;
    }
    if (!IsNeutralReduceInit(reduceOp.getInitValues()[0], kind)) {
        return false;
    }
    // Size threshold for sum/max/min last-axis reductions: small reductions have
    // high dispatch overhead relative to compute, so stay on the graph path.
    if (kind == IntrinsicNativeKind::kReduceSumLastAxis || kind == IntrinsicNativeKind::kReduceMaxLastAxis ||
        kind == IntrinsicNativeKind::kReduceMinLastAxis) {
        // Single-row (full) reduces are effectively serial on the MPSGraph path
        // (~45ns/elem), so the native TG kernel wins at much smaller sizes than
        // the multi-row dispatch-overhead-bound case.
        const bool single_row = (outNumel == 1);
        uint64_t minElems = static_cast<uint64_t>(
            EnvIntValue(single_row ? "JAX_METALLIB_REDUCE_NATIVE_MIN_ELEMENTS_SINGLE_ROW"
                                   : "JAX_METALLIB_REDUCE_NATIVE_MIN_ELEMENTS",
                        single_row ? 4096 : 32768));
        if (inNumel < minElems) {
            return false;
        }
    }
    return true;
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

// Forward declarations: AutotuneGemm and kMpsGraphGemmVariant are defined later but
// needed by ClassifyDotGeneralNN to check whether MPSGraph would win.
static const int kMpsGraphGemmVariant = -2;
static int AutotuneGemm(id<MTLDevice> device, int dtype, uint32_t M, uint32_t N, uint32_t K);

static bool ClassifyDotGeneralNN(mlir::Operation* op, uint32_t* out_m, uint32_t* out_n, uint32_t* out_k,
                                 uint32_t* out_batch) {
    if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return false;
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name != "stablehlo.dot_general") {
        return false;
    }
    // Batched dot_general bypass: check batch dims early to skip env flag gate
    // for batched cases where native GEMM with grid.z dispatch is much faster
    // than MPSGraph (which dispatches batch-many separate graph evaluations).
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        return false;
    }

    std::vector<int64_t> lhsShape, rhsShape;
    MPSDataType lhsDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &lhsShape, nullptr, &lhsDtype) ||
        !GetStaticTensorInfo(op->getOperand(1).getType(), &rhsShape, nullptr, nullptr)) {
        return false;
    }
    if (lhsDtype != MPSDataTypeFloat32 && lhsDtype != MPSDataTypeFloat16) {
        return false;
    }

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
        return false;
    }

    uint32_t M = 0, N = 0, K = 0, batch = 1;

    const int64_t lhs_rank = static_cast<int64_t>(lhsShape.size());
    const int64_t rhs_rank = static_cast<int64_t>(rhsShape.size());
    if (lhs_rank != rhs_rank || lhs_rank < 2) {
        return false;
    }
    if (lhsBatchDims.size() != rhsBatchDims.size()) {
        return false;
    }

    if (lhs_rank == 2) {
        if (!lhsBatchDims.empty()) {
            return false;
        }
        if (lhsContractingDims[0] != 1 || rhsContractingDims[0] != 0) {
            return false;
        }
        if (lhsShape[1] != rhsShape[0]) {
            return false;
        }
        M = static_cast<uint32_t>(lhsShape[0]);
        K = static_cast<uint32_t>(lhsShape[1]);
        N = static_cast<uint32_t>(rhsShape[1]);
    } else {
        // Canonical batched matmul: [..., M, K] x [..., K, N] -> [..., M, N]
        const int64_t expected_batch_dims = lhs_rank - 2;
        if (static_cast<int64_t>(lhsBatchDims.size()) != expected_batch_dims) {
            return false;
        }
        if (lhsContractingDims[0] != lhs_rank - 1 || rhsContractingDims[0] != rhs_rank - 2) {
            return false;
        }

        uint64_t batch64 = 1;
        for (int64_t i = 0; i < expected_batch_dims; ++i) {
            if (lhsBatchDims[i] != i || rhsBatchDims[i] != i) {
                return false;
            }
            if (lhsShape[i] != rhsShape[i] || lhsShape[i] <= 0) {
                return false;
            }
            batch64 *= static_cast<uint64_t>(lhsShape[i]);
            if (batch64 > std::numeric_limits<uint32_t>::max()) {
                return false;
            }
        }

        const int64_t m_dim = lhs_rank - 2;
        const int64_t k_lhs_dim = lhs_rank - 1;
        const int64_t k_rhs_dim = rhs_rank - 2;
        const int64_t n_dim = rhs_rank - 1;
        if (lhsShape[k_lhs_dim] != rhsShape[k_rhs_dim]) {
            return false;
        }
        if (lhsShape[m_dim] <= 0 || lhsShape[k_lhs_dim] <= 0 || rhsShape[n_dim] <= 0) {
            return false;
        }

        batch = static_cast<uint32_t>(batch64);
        M = static_cast<uint32_t>(lhsShape[m_dim]);
        K = static_cast<uint32_t>(lhsShape[k_lhs_dim]);
        N = static_cast<uint32_t>(rhsShape[n_dim]);
    }

    if (M == 0 || N == 0 || K == 0) {
        return false;
    }
    // For non-batched GEMM, require the env flag gate.
    // For batched GEMM (batch > 1): only force native when each individual matmul
    // is large enough (M*N*K >= 1M FLOPs). Small batched matmuls are often faster
    // on MPSGraph due to better batch scheduling and lower dispatch overhead.
    if (batch <= 1 && !EnvFlagEnabled("JAX_METALLIB_ENABLE_NATIVE_GEMM")) {
        return false;
    }
    if (batch > 1) {
        uint64_t per_matmul_flops = static_cast<uint64_t>(M) * N * K;
        if (per_matmul_flops < 1000000 && !EnvFlagEnabled("JAX_METALLIB_ENABLE_NATIVE_GEMM")) {
            return false;
        }
    }
    // Minimum alignment across all GEMM variants: M%32, N%32, K%16.
    // (Variants: 32x32x32, 64x64x32, 64x64x16, 64x32x32)
    // Exact variant fitting is checked at autotune time.
    if ((M % 32 != 0) || (N % 32 != 0) || (K % 16 != 0)) {
        return false;
    }
    uint64_t mnk = static_cast<uint64_t>(M) * N * K;
    if (mnk < 262144) {
        return false;
    }

    // Smart routing for f16:
    // - With MPP available: intercept ALL f16 (MPP uses AMX hardware, matches/beats MPSGraph)
    // - Without MPP: MPSGraph achieves ~15 TFLOPS for f16 up to ~5120x5120 but collapses
    //   to ~2 TFLOPS beyond that. Only intercept when M or N exceeds the threshold.
    if (lhsDtype == MPSDataTypeFloat16) {
        // Check if MPP (and hence AMX) is available — use a lazy static check.
        // Can't call IsMppGemmAvailable here (no device), so check env + macOS version.
        static const bool mpp_likely_available = []() {
            if (@available(macOS 14.0, *)) {
                // M3+ with macOS 14+ likely has tensor support
                return true;
            }
            return false;
        }();
        static const bool mpp_disabled = []() {
            const char* v = std::getenv("JAX_METALLIB_GEMM_NO_MPP");
            return v && v[0] == '1' && v[1] == '\0';
        }();
        if (!mpp_likely_available || mpp_disabled) {
            static const int f16_threshold = EnvIntValue("JAX_METALLIB_GEMM_F16_THRESHOLD", 5120);
            if (M <= (uint32_t)f16_threshold && N <= (uint32_t)f16_threshold) {
                return false; // Let MPSGraph handle small/medium f16 (faster hardware path)
            }
        }
        // With MPP available: intercept all f16 sizes
    }

    // Pre-check autotune: if MPSGraph would win, don't intercept this op.
    // The autotuner cache makes this O(1) after the first call per shape.
    {
        static const bool autotune_disabled = []() {
            const char* v = std::getenv("JAX_METALLIB_GEMM_AUTOTUNE");
            return v && v[0] == '0' && v[1] == '\0';
        }();
        if (!autotune_disabled) {
            id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
            if (dev) {
                int dtypeIdx = (lhsDtype == MPSDataTypeFloat16) ? 1 : 0;
                int vi = AutotuneGemm(dev, dtypeIdx, M, N, K);
                [dev release];
                if (vi == kMpsGraphGemmVariant) {
                    return false; // MPSGraph is faster — don't intercept
                }
            }
        }
    }

    if (out_m)
        *out_m = M;
    if (out_n)
        *out_n = N;
    if (out_k)
        *out_k = K;
    if (out_batch)
        *out_batch = batch;
    return true;
}

static const int kNaiveGemmVariant = -3; // Special index: use naive per-element GEMM

// Classify small/unaligned dot_general that fails ClassifyDotGeneralNN's alignment checks.
// Returns true for M*N <= 8192 (small enough that one-thread-per-element is efficient).
static bool ClassifyDotGeneralSmall(mlir::Operation* op, uint32_t* out_m, uint32_t* out_n, uint32_t* out_k,
                                    uint32_t* out_batch) {
    if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1) {
        return false;
    }
    llvm::StringRef name = op->getName().getStringRef();
    if (name != "stablehlo.dot_general") {
        return false;
    }
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_NAIVE_GEMM")) {
        return false;
    }
    auto dotOp = mlir::dyn_cast<mlir::stablehlo::DotGeneralOp>(op);
    if (!dotOp) {
        return false;
    }

    std::vector<int64_t> lhsShape, rhsShape;
    MPSDataType lhsDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &lhsShape, nullptr, &lhsDtype) ||
        !GetStaticTensorInfo(op->getOperand(1).getType(), &rhsShape, nullptr, nullptr)) {
        return false;
    }
    if (lhsDtype != MPSDataTypeFloat32 && lhsDtype != MPSDataTypeFloat16) {
        return false;
    }

    auto dimNumbers = dotOp.getDotDimensionNumbers();
    auto lhsContractingDims = dimNumbers.getLhsContractingDimensions();
    auto rhsContractingDims = dimNumbers.getRhsContractingDimensions();
    auto lhsBatchDims = dimNumbers.getLhsBatchingDimensions();
    auto rhsBatchDims = dimNumbers.getRhsBatchingDimensions();

    if (lhsContractingDims.size() != 1 || rhsContractingDims.size() != 1) {
        return false;
    }

    uint32_t M = 0, N = 0, K = 0, batch = 1;
    const int64_t lhs_rank = static_cast<int64_t>(lhsShape.size());
    const int64_t rhs_rank = static_cast<int64_t>(rhsShape.size());
    if (lhs_rank != rhs_rank || lhs_rank < 2) {
        return false;
    }
    if (lhsBatchDims.size() != rhsBatchDims.size()) {
        return false;
    }

    if (lhs_rank == 2) {
        if (!lhsBatchDims.empty())
            return false;
        if (lhsContractingDims[0] != 1 || rhsContractingDims[0] != 0)
            return false;
        if (lhsShape[1] != rhsShape[0])
            return false;
        M = static_cast<uint32_t>(lhsShape[0]);
        K = static_cast<uint32_t>(lhsShape[1]);
        N = static_cast<uint32_t>(rhsShape[1]);
    } else {
        const int64_t expected_batch_dims = lhs_rank - 2;
        if (static_cast<int64_t>(lhsBatchDims.size()) != expected_batch_dims)
            return false;
        if (lhsContractingDims[0] != lhs_rank - 1 || rhsContractingDims[0] != rhs_rank - 2)
            return false;
        uint64_t batch64 = 1;
        for (int64_t i = 0; i < expected_batch_dims; ++i) {
            if (lhsBatchDims[i] != i || rhsBatchDims[i] != i)
                return false;
            if (lhsShape[i] != rhsShape[i] || lhsShape[i] <= 0)
                return false;
            batch64 *= static_cast<uint64_t>(lhsShape[i]);
            if (batch64 > std::numeric_limits<uint32_t>::max())
                return false;
        }
        const int64_t m_dim = lhs_rank - 2;
        const int64_t k_lhs_dim = lhs_rank - 1;
        const int64_t k_rhs_dim = rhs_rank - 2;
        const int64_t n_dim = rhs_rank - 1;
        if (lhsShape[k_lhs_dim] != rhsShape[k_rhs_dim])
            return false;
        batch = static_cast<uint32_t>(batch64);
        M = static_cast<uint32_t>(lhsShape[m_dim]);
        K = static_cast<uint32_t>(lhsShape[k_lhs_dim]);
        N = static_cast<uint32_t>(rhsShape[n_dim]);
    }

    if (M == 0 || N == 0 || K == 0)
        return false;
    // Only handle small matrices where naive one-thread-per-element is efficient.
    // Keep threshold low to avoid intercepting medium matrices (e.g. 64x64).
    if (static_cast<uint64_t>(M) * N > 2048)
        return false;

    if (out_m)
        *out_m = M;
    if (out_n)
        *out_n = N;
    if (out_k)
        *out_k = K;
    if (out_batch)
        *out_batch = batch;
    return true;
}

static std::mutex g_fused_cache_mu;
static std::unordered_map<uintptr_t, std::unordered_map<std::string, id<MTLComputePipelineState>>>
    g_fused_cache;

// --- MPP (MetalPerformancePrimitives) GEMM Pipeline ---
// Compiled with Metal 3.1+ language version for tensor type support.
// Returns nil if tensor types are not supported (pre-M3 hardware or older macOS).
static id<MTLComputePipelineState> GetMppGemmPipeline(id<MTLDevice> device, const std::string& functionName,
                                                      uint32_t num_simdgroups) {
    static std::mutex mu;
    static std::unordered_map<uintptr_t, FastNativeKernelCacheEntry> cache;
    static bool mpp_available_checked = false;
    static bool mpp_available = false;

    std::lock_guard<std::mutex> lock(mu);

    // Check availability once
    if (!mpp_available_checked) {
        mpp_available_checked = true;
        // Try to compile a minimal tensor shader to check if __HAVE_TENSOR__ is available
        @autoreleasepool {
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            if (@available(macOS 26.0, *)) {
                opts.languageVersion = MTLLanguageVersion4_0;
            } else if (@available(macOS 15.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_2;
            } else if (@available(macOS 14.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_1;
            } else {
                [opts release];
                MPS_LOG_DEBUG("mpp-gemm: macOS too old for tensor support\n");
                return nil;
            }
            NSError* error = nil;
            NSString* testSrc = @"#include <metal_stdlib>\nusing namespace metal;\n"
                                 "#if defined(__HAVE_TENSOR__)\n"
                                 "kernel void _mpp_test() {}\n"
                                 "#else\n"
                                 "#error no tensor support\n"
                                 "#endif\n";
            id<MTLLibrary> testLib = [device newLibraryWithSource:testSrc options:opts error:&error];
            [opts release];
            if (testLib) {
                mpp_available = true;
                [testLib release];
                MPS_LOG_DEBUG("mpp-gemm: tensor support available\n");
            } else {
                MPS_LOG_DEBUG("mpp-gemm: tensor support NOT available: %s\n",
                              error.localizedDescription.UTF8String);
            }
        }
    }

    if (!mpp_available) {
        return nil;
    }

    auto& entry = cache[(uintptr_t)device];
    if (!entry.library) {
        @autoreleasepool {
            MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
            if (@available(macOS 26.0, *)) {
                opts.languageVersion = MTLLanguageVersion4_0;
            } else if (@available(macOS 15.0, *)) {
                opts.languageVersion = MTLLanguageVersion3_2;
            } else {
                opts.languageVersion = MTLLanguageVersion3_1;
            }
            if (@available(macOS 15.0, *)) {
                opts.mathMode = MTLMathModeRelaxed;
            } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                opts.fastMathEnabled = YES;
#pragma clang diagnostic pop
            }
            NSError* error = nil;
            entry.library = [device newLibraryWithSource:MppGemmKernelSource() options:opts error:&error];
            [opts release];
            if (!entry.library) {
                MPS_LOG_ERROR("mpp-gemm: shader compile failed: %s\n", error.localizedDescription.UTF8String);
                return nil;
            }
        }
    }

    auto it = entry.pipelines.find(functionName);
    if (it != entry.pipelines.end()) {
        return it->second;
    }

    NSString* fnName = [NSString stringWithUTF8String:functionName.c_str()];
    id<MTLFunction> fn = [entry.library newFunctionWithName:fnName];
    if (!fn) {
        MPS_LOG_ERROR("mpp-gemm: kernel not found: %s\n", functionName.c_str());
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    if (!pipeline) {
        MPS_LOG_ERROR("mpp-gemm: pipeline creation failed for %s: %s\n", functionName.c_str(),
                      error.localizedDescription.UTF8String);
        return nil;
    }
    entry.pipelines.emplace(functionName, pipeline);
    return pipeline;
}

// Returns true if MPP GEMM is available on this device.
static bool IsMppGemmAvailable(id<MTLDevice> device) {
    // Try to get the f32 pipeline — this will trigger availability check
    return GetMppGemmPipeline(device, "gemm_mpp_f32", 4) != nil;
}

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

// --- MX Quantized Kernel Pipeline Cache ---
static id<MTLComputePipelineState> GetMxQuantizedPipeline(id<MTLDevice> device,
                                                          const std::string& functionName) {
    static std::mutex mu;
    static std::unordered_map<uintptr_t, FastNativeKernelCacheEntry> cache;

    std::lock_guard<std::mutex> lock(mu);
    auto& entry = cache[(uintptr_t)device];
    if (!entry.library) {
        NSError* error = nil;
        entry.library = [device newLibraryWithSource:MxQuantizedKernelSource() options:nil error:&error];
        if (!entry.library) {
            MPS_LOG_ERROR("mx-quantized: shader compile failed: %s\n", error.localizedDescription.UTF8String);
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
        MPS_LOG_ERROR("mx-quantized: kernel not found: %s\n", functionName.c_str());
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:fn error:&error];
    [fn release];
    if (!pipeline) {
        MPS_LOG_ERROR("mx-quantized: pipeline creation failed for %s: %s\n", functionName.c_str(),
                      error.localizedDescription.UTF8String);
        return nil;
    }
    entry.pipelines.emplace(functionName, pipeline);
    return pipeline;
}

// --- GEMM Autotuning ---

struct GemmVariant {
    uint32_t bm, bn, bk;
    uint32_t num_threads;
    const char* f32_name;
    const char* f16_name;
};

static const GemmVariant kGemmVariants[] = {
    {32, 32, 16, 128, "gemm_nn_f32_bm32_bn32_bk16", "gemm_nn_f16_bm32_bn32_bk16"},
    {64, 64, 16, 128, "gemm_nn_f32_bm64_bn64_bk16", "gemm_nn_f16_bm64_bn64_bk16"},
    {64, 64, 32, 128, "gemm_nn_f32_bm64_bn64_bk32", "gemm_nn_f16_bm64_bn64_bk32"},
    {64, 32, 32, 128, "gemm_nn_f32_bm64_bn32_bk32", "gemm_nn_f16_bm64_bn32_bk32"},
};
static const int kNumGemmVariants = sizeof(kGemmVariants) / sizeof(kGemmVariants[0]);
static const int kDefaultGemmVariant = 1; // BM=64, BN=64, BK=16 (MLX default)
static const int kMppGemmVariant = -1;    // Special index: use MPP tensor ops path
// kMpsGraphGemmVariant = -2 is forward-declared above ClassifyDotGeneralNN

struct AutotuneKey {
    uintptr_t device_id;
    int dtype; // 0=f32, 1=f16
    uint32_t M, N, K;
    bool operator==(const AutotuneKey& o) const {
        return device_id == o.device_id && dtype == o.dtype && M == o.M && N == o.N && K == o.K;
    }
};

struct AutotuneKeyHash {
    size_t operator()(const AutotuneKey& k) const {
        size_t h = std::hash<uintptr_t>()(k.device_id);
        h ^= std::hash<int>()(k.dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(k.M) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(k.N) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint32_t>()(k.K) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

static std::mutex g_autotune_mutex;

// ── Elementwise autotuning: native kernel vs MPSGraph ──
// Keyed by (device, is_binary, dtype, numel) — all bandwidth-bound ops
// of the same arity share a cache entry since they have identical perf profiles.
struct ElemAutotuneKey {
    uintptr_t device_id;
    int is_binary; // 0=unary, 1=binary
    int dtype;     // 0=f32, 1=f16
    uint64_t numel;
    bool operator==(const ElemAutotuneKey& o) const {
        return device_id == o.device_id && is_binary == o.is_binary && dtype == o.dtype && numel == o.numel;
    }
};

struct ElemAutotuneKeyHash {
    size_t operator()(const ElemAutotuneKey& k) const {
        size_t h = std::hash<uintptr_t>()(k.device_id);
        h ^= std::hash<int>()(k.is_binary) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<int>()(k.dtype) + 0x9e3779b9 + (h << 6) + (h >> 2);
        h ^= std::hash<uint64_t>()(k.numel) + 0x9e3779b9 + (h << 6) + (h >> 2);
        return h;
    }
};

// true = native kernel wins, false = MPSGraph wins (demote to graph)
static std::mutex g_elem_autotune_mutex;
static std::unordered_map<ElemAutotuneKey, bool, ElemAutotuneKeyHash> g_elem_autotune_cache;
static std::unordered_map<AutotuneKey, int, AutotuneKeyHash> g_autotune_cache;

// ── Autotune cache persistence ──
static std::string GetAutotuneCachePath() {
    const char* env = std::getenv("JAX_METALLIB_AUTOTUNE_CACHE_DIR");
    std::string dir;
    if (env && env[0]) {
        dir = env;
    } else {
        const char* home = std::getenv("HOME");
        if (!home)
            return "";
        dir = std::string(home) + "/.cache/jax-silicon";
    }
    // Ensure directory exists
    mkdir(dir.c_str(), 0755);
    return dir + "/autotune_cache.bin";
}

// Format version of autotune_cache.bin. Bump when AutotuneKey/ElemAutotuneKey
// layouts, variant numbering, or field types change.
static constexpr uint32_t kAutotuneCacheVersion = 2;

static void SaveAutotuneCacheToDisk() {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_AUTOTUNE_CACHE_PERSIST"))
        return;
    std::string path = GetAutotuneCachePath();
    if (path.empty())
        return;

    FILE* f = fopen(path.c_str(), "wb");
    if (!f)
        return;

    // Header: magic + format version. Bump kAutotuneCacheVersion whenever the
    // key/value layouts or variant numbering change; loaders discard
    // mismatched files instead of silently misreading stale entries.
    const char magic[4] = {'J', 'X', 'M', 'T'};
    fwrite(magic, 1, sizeof(magic), f);
    fwrite(&kAutotuneCacheVersion, sizeof(kAutotuneCacheVersion), 1, f);

    // Write GEMM cache
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        uint32_t count = static_cast<uint32_t>(g_autotune_cache.size());
        fwrite(&count, sizeof(count), 1, f);
        for (auto& [k, v] : g_autotune_cache) {
            fwrite(&k.dtype, sizeof(k.dtype), 1, f);
            fwrite(&k.M, sizeof(k.M), 1, f);
            fwrite(&k.N, sizeof(k.N), 1, f);
            fwrite(&k.K, sizeof(k.K), 1, f);
            fwrite(&v, sizeof(v), 1, f);
        }
    }
    // Write elementwise cache
    {
        std::lock_guard<std::mutex> lock(g_elem_autotune_mutex);
        uint32_t count = static_cast<uint32_t>(g_elem_autotune_cache.size());
        fwrite(&count, sizeof(count), 1, f);
        for (auto& [k, v] : g_elem_autotune_cache) {
            fwrite(&k.is_binary, sizeof(k.is_binary), 1, f);
            fwrite(&k.dtype, sizeof(k.dtype), 1, f);
            fwrite(&k.numel, sizeof(k.numel), 1, f);
            uint8_t bv = v ? 1 : 0;
            fwrite(&bv, sizeof(bv), 1, f);
        }
    }
    fclose(f);
    MPS_LOG_DEBUG("autotune cache saved to %s\n", path.c_str());
}

static void LoadAutotuneCacheFromDisk(uintptr_t device_id) {
    if (EnvFlagEnabled("JAX_METALLIB_DISABLE_AUTOTUNE_CACHE_PERSIST"))
        return;
    std::string path = GetAutotuneCachePath();
    if (path.empty())
        return;

    FILE* f = fopen(path.c_str(), "rb");
    if (!f)
        return;

    // Header check: discard unversioned or mismatched files (e.g. caches
    // written by older plugin builds — they must not steer kernel selection).
    const char kMagic[4] = {'J', 'X', 'M', 'T'};
    char magic[4] = {0, 0, 0, 0};
    uint32_t version = 0;
    if (fread(magic, 1, sizeof(magic), f) != sizeof(magic) ||
        std::memcmp(magic, kMagic, sizeof(magic)) != 0) {
        fclose(f);
        MPS_LOG_DEBUG("autotune cache %s: missing magic, ignoring\n", path.c_str());
        return;
    }
    if (fread(&version, sizeof(version), 1, f) != 1 || version != kAutotuneCacheVersion) {
        fclose(f);
        MPS_LOG_DEBUG("autotune cache %s: version mismatch (%u), ignoring\n", path.c_str(), version);
        return;
    }

    // Read GEMM cache
    uint32_t gemm_count = 0;
    if (fread(&gemm_count, sizeof(gemm_count), 1, f) != 1) {
        fclose(f);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        for (uint32_t i = 0; i < gemm_count && i < 10000; ++i) {
            AutotuneKey k;
            k.device_id = device_id;
            int v;
            if (fread(&k.dtype, sizeof(k.dtype), 1, f) != 1)
                break;
            if (fread(&k.M, sizeof(k.M), 1, f) != 1)
                break;
            if (fread(&k.N, sizeof(k.N), 1, f) != 1)
                break;
            if (fread(&k.K, sizeof(k.K), 1, f) != 1)
                break;
            if (fread(&v, sizeof(v), 1, f) != 1)
                break;
            g_autotune_cache[k] = v;
        }
    }
    // Read elementwise cache
    uint32_t elem_count = 0;
    if (fread(&elem_count, sizeof(elem_count), 1, f) != 1) {
        fclose(f);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(g_elem_autotune_mutex);
        for (uint32_t i = 0; i < elem_count && i < 10000; ++i) {
            ElemAutotuneKey k;
            k.device_id = device_id;
            uint8_t bv;
            if (fread(&k.is_binary, sizeof(k.is_binary), 1, f) != 1)
                break;
            if (fread(&k.dtype, sizeof(k.dtype), 1, f) != 1)
                break;
            if (fread(&k.numel, sizeof(k.numel), 1, f) != 1)
                break;
            if (fread(&bv, sizeof(bv), 1, f) != 1)
                break;
            g_elem_autotune_cache[k] = (bv != 0);
        }
    }
    fclose(f);
    MPS_LOG_DEBUG("autotune cache loaded from %s (gemm=%u, elem=%u)\n", path.c_str(), gemm_count, elem_count);
}

static bool g_autotune_cache_loaded = false;

// Returns true if the variant's tile sizes evenly divide M, N, K
static bool GemmVariantFits(const GemmVariant& v, uint32_t M, uint32_t N, uint32_t K) {
    return (M % v.bm == 0) && (N % v.bn == 0) && (K % v.bk == 0);
}

// Forward declaration for use in AutotuneGemm benchmark loop.
static void DispatchMppGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> bufA, size_t offA, id<MTLBuffer> bufB, size_t offB,
                            id<MTLBuffer> bufC, size_t offC, uint32_t M, uint32_t N, uint32_t K,
                            bool isFloat16);

// Find the best GEMM variant for a given shape via GPU benchmarking.
// Returns variant index: kMppGemmVariant (-1) for MPP path, or 0..kNumGemmVariants-1 for simdgroup path.
static int AutotuneGemm(id<MTLDevice> device, int dtype, uint32_t M, uint32_t N, uint32_t K) {
    // Load persistent cache on first call
    if (!g_autotune_cache_loaded) {
        g_autotune_cache_loaded = true;
        LoadAutotuneCacheFromDisk((uintptr_t)device);
    }

    AutotuneKey key{(uintptr_t)device, dtype, M, N, K};

    // Check cache
    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        auto it = g_autotune_cache.find(key);
        if (it != g_autotune_cache.end()) {
            return it->second;
        }
    }

    // Check for forced variant (supports 0..N-1, -1 for MPP, -2 for MPSGraph)
    {
        const char* fv = std::getenv("JAX_METALLIB_GEMM_VARIANT");
        if (fv && fv[0] != '\0') {
            int forced = std::atoi(fv);
            if (forced == kMpsGraphGemmVariant) {
                // Force MPSGraph path
                std::lock_guard<std::mutex> lock(g_autotune_mutex);
                g_autotune_cache[key] = kMpsGraphGemmVariant;
                return kMpsGraphGemmVariant;
            } else if (forced == kMppGemmVariant) {
                // Force MPP path
                if (IsMppGemmAvailable(device)) {
                    std::lock_guard<std::mutex> lock(g_autotune_mutex);
                    g_autotune_cache[key] = kMppGemmVariant;
                    return kMppGemmVariant;
                }
            } else if (forced >= 0 && forced < kNumGemmVariants &&
                       GemmVariantFits(kGemmVariants[forced], M, N, K)) {
                std::lock_guard<std::mutex> lock(g_autotune_mutex);
                g_autotune_cache[key] = forced;
                return forced;
            }
        }
    }

    // Check MPP availability — will be benchmarked alongside simdgroup variants.
    static const bool mpp_disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_GEMM_NO_MPP");
        return v && v[0] == '1' && v[1] == '\0';
    }();
    bool mpp_avail = !mpp_disabled && IsMppGemmAvailable(device);

    // Check if autotuning is disabled
    static const bool autotune_disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_GEMM_AUTOTUNE");
        return v && v[0] == '0' && v[1] == '\0';
    }();
    if (autotune_disabled) {
        // When autotuning is disabled, prefer MPP if available (no benchmark overhead).
        if (mpp_avail) {
            std::lock_guard<std::mutex> lock(g_autotune_mutex);
            g_autotune_cache[key] = kMppGemmVariant;
            return kMppGemmVariant;
        }
        // Use default simdgroup variant if it fits, else first fitting
        if (GemmVariantFits(kGemmVariants[kDefaultGemmVariant], M, N, K)) {
            std::lock_guard<std::mutex> lock(g_autotune_mutex);
            g_autotune_cache[key] = kDefaultGemmVariant;
            return kDefaultGemmVariant;
        }
        for (int i = 0; i < kNumGemmVariants; ++i) {
            if (GemmVariantFits(kGemmVariants[i], M, N, K)) {
                std::lock_guard<std::mutex> lock(g_autotune_mutex);
                g_autotune_cache[key] = i;
                return i;
            }
        }
        return kDefaultGemmVariant;
    }

    // Autotuning: benchmark all fitting variants.
    // Use enough iterations for stable GPU measurements (power state, thermal).
    static const int warmup_iters = EnvIntValue("JAX_METALLIB_GEMM_AUTOTUNE_WARMUP", 5);
    static const int bench_iters = EnvIntValue("JAX_METALLIB_GEMM_AUTOTUNE_ITERS", 10);

    size_t elem_size = (dtype == 1) ? 2 : 4; // f16 : f32
    size_t a_bytes = (size_t)M * K * elem_size;
    size_t b_bytes = (size_t)K * N * elem_size;
    size_t c_bytes = (size_t)M * N * elem_size;

    id<MTLBuffer> tmpA = [device newBufferWithLength:a_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> tmpB = [device newBufferWithLength:b_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> tmpC = [device newBufferWithLength:c_bytes options:MTLResourceStorageModePrivate];

    int best_variant = kDefaultGemmVariant;
    double best_time = 1e30;

    id<MTLCommandQueue> queue = [device newCommandQueue];

    for (int vi = 0; vi < kNumGemmVariants; ++vi) {
        const auto& v = kGemmVariants[vi];
        if (!GemmVariantFits(v, M, N, K))
            continue;

        const char* kname = (dtype == 1) ? v.f16_name : v.f32_name;
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kname);
        if (!pipeline)
            continue;

        uint32_t grid_x = M / v.bm;
        uint32_t grid_y = N / v.bn;
        MTLSize grid = MTLSizeMake(grid_x, grid_y, 1);
        MTLSize tg = MTLSizeMake(v.num_threads, 1, 1);

        // Warmup
        for (int w = 0; w < warmup_iters; ++w) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:tmpA offset:0 atIndex:0];
            [enc setBuffer:tmpB offset:0 atIndex:1];
            [enc setBuffer:tmpC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&K length:sizeof(K) atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }

        // Benchmark — wall-clock (includes dispatch overhead, same as MPSGraph)
        double total_time = 0.0;
        for (int it = 0; it < bench_iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:tmpA offset:0 atIndex:0];
            [enc setBuffer:tmpB offset:0 atIndex:1];
            [enc setBuffer:tmpC offset:0 atIndex:2];
            [enc setBytes:&M length:sizeof(M) atIndex:3];
            [enc setBytes:&N length:sizeof(N) atIndex:4];
            [enc setBytes:&K length:sizeof(K) atIndex:5];
            [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            auto t1 = std::chrono::high_resolution_clock::now();
            total_time += std::chrono::duration<double>(t1 - t0).count();
        }
        double avg_time = total_time / bench_iters;

        double tflops = (2.0 * M * N * K) / (avg_time * 1e12);
        MPS_LOG_WARN("autotune-gemm: simdgroup v%d (%s) M=%u N=%u K=%u time=%.3fms (%.2f TFLOPS)\n", vi,
                     kname, M, N, K, avg_time * 1000.0, tflops);

        if (avg_time < best_time) {
            best_time = avg_time;
            best_variant = vi;
        }
    }

    // Benchmark MPP variant alongside simdgroup variants
    if (mpp_avail) {
        const char* mpp_kname = (dtype == 1) ? "gemm_mpp_f16" : "gemm_mpp_f32";
        id<MTLComputePipelineState> mppPipeline = GetMppGemmPipeline(device, mpp_kname, 4);
        if (mppPipeline) {
            // Warmup
            for (int w = 0; w < warmup_iters; ++w) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                DispatchMppGemm(enc, mppPipeline, tmpA, 0, tmpB, 0, tmpC, 0, M, N, K, dtype == 1);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
            }

            // Benchmark — wall-clock (includes dispatch overhead, same as MPSGraph)
            double mpp_total_time = 0.0;
            for (int it = 0; it < bench_iters; ++it) {
                auto t0 = std::chrono::high_resolution_clock::now();
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
                DispatchMppGemm(enc, mppPipeline, tmpA, 0, tmpB, 0, tmpC, 0, M, N, K, dtype == 1);
                [enc endEncoding];
                [cb commit];
                [cb waitUntilCompleted];
                auto t1 = std::chrono::high_resolution_clock::now();
                mpp_total_time += std::chrono::duration<double>(t1 - t0).count();
            }
            double mpp_avg = mpp_total_time / bench_iters;

            double mpp_tflops = (2.0 * M * N * K) / (mpp_avg * 1e12);
            MPS_LOG_WARN("autotune-gemm: MPP (%s) M=%u N=%u K=%u time=%.3fms (%.2f TFLOPS)\n", mpp_kname, M,
                         N, K, mpp_avg * 1000.0, mpp_tflops);

            if (mpp_avg < best_time) {
                best_time = mpp_avg;
                best_variant = kMppGemmVariant;
            }
        }
    }

    // Benchmark MPSGraph matmul as a third candidate
    {
        @autoreleasepool {
            MPSGraph* graph = [[MPSGraph alloc] init];
            MPSDataType mpsdt = (dtype == 1) ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            NSArray<NSNumber*>* shapeA = @[@(M), @(K)];
            NSArray<NSNumber*>* shapeB = @[@(K), @(N)];

            MPSGraphTensor* tA = [graph placeholderWithShape:shapeA dataType:mpsdt name:@"A"];
            MPSGraphTensor* tB = [graph placeholderWithShape:shapeB dataType:mpsdt name:@"B"];
            MPSGraphTensor* tC = [graph matrixMultiplicationWithPrimaryTensor:tA
                                                              secondaryTensor:tB
                                                                         name:@"C"];

            MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];
            MPSGraphShapedType* stA = [[MPSGraphShapedType alloc] initWithShape:shapeA dataType:mpsdt];
            MPSGraphShapedType* stB = [[MPSGraphShapedType alloc] initWithShape:shapeB dataType:mpsdt];

            MPSGraphExecutable* exec = [graph compileWithDevice:gdev
                                                          feeds:@{tA: stA, tB: stB}
                                                  targetTensors:@[tC]
                                               targetOperations:nil
                                          compilationDescriptor:nil];

            if (exec) {
                MPSGraphTensorData* dataA = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpA
                                                                                    shape:shapeA
                                                                                 dataType:mpsdt];
                MPSGraphTensorData* dataB = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpB
                                                                                    shape:shapeB
                                                                                 dataType:mpsdt];
                MPSGraphTensorData* dataC = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpC
                                                                                    shape:@[@(M), @(N)]
                                                                                 dataType:mpsdt];

                // Warmup
                for (int w = 0; w < warmup_iters; ++w) {
                    [exec runWithMTLCommandQueue:queue
                                     inputsArray:@[dataA, dataB]
                                    resultsArray:@[dataC]
                             executionDescriptor:nil];
                }

                // Benchmark — wall-clock (MPSGraph manages its own command buffers internally)
                double graph_total_time = 0.0;
                for (int it = 0; it < bench_iters; ++it) {
                    auto t0 = std::chrono::high_resolution_clock::now();
                    [exec runWithMTLCommandQueue:queue
                                     inputsArray:@[dataA, dataB]
                                    resultsArray:@[dataC]
                             executionDescriptor:nil];
                    auto t1 = std::chrono::high_resolution_clock::now();
                    graph_total_time += std::chrono::duration<double>(t1 - t0).count();
                }
                double graph_avg = graph_total_time / bench_iters;
                double graph_tflops = (2.0 * M * N * K) / (graph_avg * 1e12);
                MPS_LOG_WARN("autotune-gemm: MPSGraph M=%u N=%u K=%u time=%.3fms (%.2f TFLOPS)\n", M, N, K,
                             graph_avg * 1000.0, graph_tflops);

                if (graph_avg < best_time) {
                    best_time = graph_avg;
                    best_variant = kMpsGraphGemmVariant;
                }
            }
        }
    }

    [tmpA release];
    [tmpB release];
    [tmpC release];
    [queue release];

    {
        double sel_tflops = (2.0 * M * N * K) / (best_time * 1e12);
        const char* variant_name = (best_variant == kMppGemmVariant)        ? " (MPP)"
                                   : (best_variant == kMpsGraphGemmVariant) ? " (MPSGraph)"
                                                                            : " (simdgroup)";
        MPS_LOG_WARN("autotune-gemm: >>> selected variant %d%s for M=%u N=%u K=%u (%.3fms, %.2f TFLOPS)\n",
                     best_variant, variant_name, M, N, K, best_time * 1000.0, sel_tflops);
    }

    {
        std::lock_guard<std::mutex> lock(g_autotune_mutex);
        g_autotune_cache[key] = best_variant;
    }
    SaveAutotuneCacheToDisk();
    return best_variant;
}

// Forward declarations for classifiers and helpers used by AutotuneElementwise
static bool IsComputeBoundElementwise(IntrinsicNativeKind kind);
static bool IsElementwiseBinaryOp(IntrinsicNativeKind kind);
static uint32_t ComputeStrideGridSize(uint32_t n4);

// ── Elementwise autotuning: benchmark native kernel vs MPSGraph ──
// Returns true if native kernel wins, false if MPSGraph is faster.
// Compute-bound ops (exp, tanh, sin, etc.) always return true (native wins).
// Bandwidth-bound ops are benchmarked with representative kernels.
static bool AutotuneElementwise(id<MTLDevice> device, IntrinsicNativeKind kind, MPSDataType dtype,
                                uint64_t numel) {
    // Load persistent cache on first call
    if (!g_autotune_cache_loaded) {
        g_autotune_cache_loaded = true;
        LoadAutotuneCacheFromDisk((uintptr_t)device);
    }

    // Compute-bound ops always win with native kernels — skip benchmarking.
    if (IsComputeBoundElementwise(kind)) {
        return true;
    }

    int is_binary = IsElementwiseBinaryOp(kind) ? 1 : 0;
    int dtype_idx = (dtype == MPSDataTypeFloat16) ? 1 : 0;
    ElemAutotuneKey key{(uintptr_t)device, is_binary, dtype_idx, numel};

    // Check cache
    {
        std::lock_guard<std::mutex> lock(g_elem_autotune_mutex);
        auto it = g_elem_autotune_cache.find(key);
        if (it != g_elem_autotune_cache.end()) {
            return it->second;
        }
    }

    // Check if autotuning is disabled
    static const bool autotune_disabled = []() {
        const char* v = std::getenv("JAX_METALLIB_ELEMWISE_AUTOTUNE");
        return v && v[0] == '0' && v[1] == '\0';
    }();
    if (autotune_disabled) {
        std::lock_guard<std::mutex> lock(g_elem_autotune_mutex);
        g_elem_autotune_cache[key] = true; // default: keep native
        return true;
    }

    static const int warmup_iters = EnvIntValue("JAX_METALLIB_ELEMWISE_AUTOTUNE_WARMUP", 5);
    static const int bench_iters = EnvIntValue("JAX_METALLIB_ELEMWISE_AUTOTUNE_ITERS", 10);

    size_t elem_size = (dtype == MPSDataTypeFloat16) ? 2 : 4;
    size_t buf_bytes = numel * elem_size;

    id<MTLBuffer> tmpA = [device newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> tmpB =
        is_binary ? [device newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate] : nil;
    id<MTLBuffer> tmpC = [device newBufferWithLength:buf_bytes options:MTLResourceStorageModePrivate];
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // --- Benchmark native kernel ---
    // Use representative: binary_add or unary_abs (both bandwidth-bound).
    const char* native_kname = nullptr;
    // Select the best vec mode: stride > vec8 > vec4 > scalar
    static const bool stride_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_STRIDE_KERNELS");
    static const uint64_t stride_min =
        static_cast<uint64_t>(EnvIntValue("JAX_METALLIB_STRIDE_MIN_NUMEL", 1048576));
    bool use_stride = !stride_disabled && (numel % 4 == 0) && numel >= stride_min;
    bool use_vec8 = !use_stride && (numel % 8 == 0) && (dtype == MPSDataTypeFloat16 || numel >= 262144);
    bool use_vec4 = !use_stride && !use_vec8 && (numel % 4 == 0);
    uint32_t dispatch_n = use_stride ? (uint32_t)(numel / 4)
                          : use_vec8 ? (uint32_t)(numel / 8)
                          : use_vec4 ? (uint32_t)(numel / 4)
                                     : (uint32_t)numel;

    if (is_binary) {
        if (use_stride)
            native_kname = (dtype_idx == 1) ? "binary_add_stride_f16" : "binary_add_stride_f32";
        else if (use_vec8)
            native_kname = (dtype_idx == 1) ? "binary_add_vec8_f16" : "binary_add_vec8_f32";
        else if (use_vec4)
            native_kname = (dtype_idx == 1) ? "binary_add_vec4_f16" : "binary_add_vec4_f32";
        else
            native_kname = (dtype_idx == 1) ? "binary_add_f16" : "binary_add_f32";
    } else {
        if (use_stride)
            native_kname = (dtype_idx == 1) ? "unary_abs_stride_f16" : "unary_abs_stride_f32";
        else if (use_vec8)
            native_kname = (dtype_idx == 1) ? "unary_abs_vec8_f16" : "unary_abs_vec8_f32";
        else
            native_kname = (dtype_idx == 1) ? "unary_abs_f16" : "unary_abs_f32";
    }

    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, native_kname);
    double native_avg = 1e30;
    if (pipeline) {
        NSUInteger tg_size = [pipeline maxTotalThreadsPerThreadgroup];
        if (tg_size > 256)
            tg_size = 256;
        uint32_t grid_threads = use_stride ? ComputeStrideGridSize(dispatch_n) : dispatch_n;
        MTLSize tg = MTLSizeMake(tg_size, 1, 1);
        // For stride: use dispatchThreadgroups (lower overhead); for others: dispatchThreads
        NSUInteger tg_count = ((NSUInteger)grid_threads + tg_size - 1) / tg_size;
        MTLSize grid = use_stride ? MTLSizeMake(tg_count, 1, 1) : MTLSizeMake(grid_threads, 1, 1);

        // Warmup
        for (int w = 0; w < warmup_iters; ++w) {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:tmpA offset:0 atIndex:0];
            if (is_binary) {
                [enc setBuffer:tmpB offset:0 atIndex:1];
                [enc setBuffer:tmpC offset:0 atIndex:2];
                [enc setBytes:&dispatch_n length:sizeof(dispatch_n) atIndex:3];
            } else {
                [enc setBuffer:tmpC offset:0 atIndex:1];
                [enc setBytes:&dispatch_n length:sizeof(dispatch_n) atIndex:2];
            }
            if (use_stride) {
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            } else {
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
        }
        // Benchmark — wall-clock (includes dispatch overhead, same as MPSGraph)
        double native_total = 0.0;
        for (int it = 0; it < bench_iters; ++it) {
            auto t0 = std::chrono::high_resolution_clock::now();
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:tmpA offset:0 atIndex:0];
            if (is_binary) {
                [enc setBuffer:tmpB offset:0 atIndex:1];
                [enc setBuffer:tmpC offset:0 atIndex:2];
                [enc setBytes:&dispatch_n length:sizeof(dispatch_n) atIndex:3];
            } else {
                [enc setBuffer:tmpC offset:0 atIndex:1];
                [enc setBytes:&dispatch_n length:sizeof(dispatch_n) atIndex:2];
            }
            if (use_stride) {
                [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
            } else {
                [enc dispatchThreads:grid threadsPerThreadgroup:tg];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            auto t1 = std::chrono::high_resolution_clock::now();
            native_total += std::chrono::duration<double>(t1 - t0).count();
        }
        native_avg = native_total / bench_iters;
    }

    // --- Benchmark MPSGraph ---
    double graph_avg = 1e30;
    {
        @autoreleasepool {
            MPSGraph* graph = [[MPSGraph alloc] init];
            MPSDataType mpsdt = (dtype == MPSDataTypeFloat16) ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
            NSArray<NSNumber*>* shape = @[@((NSUInteger)numel)];

            MPSGraphTensor* tA = [graph placeholderWithShape:shape dataType:mpsdt name:@"A"];
            MPSGraphTensor* tC = nil;

            if (is_binary) {
                MPSGraphTensor* tB = [graph placeholderWithShape:shape dataType:mpsdt name:@"B"];
                tC = [graph additionWithPrimaryTensor:tA secondaryTensor:tB name:@"C"];

                MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];
                MPSGraphShapedType* stA = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsdt];
                MPSGraphShapedType* stB = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsdt];

                MPSGraphExecutable* exec = [graph compileWithDevice:gdev
                                                              feeds:@{tA: stA, tB: stB}
                                                      targetTensors:@[tC]
                                                   targetOperations:nil
                                              compilationDescriptor:nil];
                if (exec) {
                    MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpA
                                                                                     shape:shape
                                                                                  dataType:mpsdt];
                    MPSGraphTensorData* dB = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpB
                                                                                     shape:shape
                                                                                  dataType:mpsdt];
                    MPSGraphTensorData* dC = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpC
                                                                                     shape:shape
                                                                                  dataType:mpsdt];
                    // Warmup
                    for (int w = 0; w < warmup_iters; ++w) {
                        [exec runWithMTLCommandQueue:queue
                                         inputsArray:@[dA, dB]
                                        resultsArray:@[dC]
                                 executionDescriptor:nil];
                    }
                    // Benchmark
                    double graph_total = 0.0;
                    for (int it = 0; it < bench_iters; ++it) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        [exec runWithMTLCommandQueue:queue
                                         inputsArray:@[dA, dB]
                                        resultsArray:@[dC]
                                 executionDescriptor:nil];
                        auto t1 = std::chrono::high_resolution_clock::now();
                        graph_total += std::chrono::duration<double>(t1 - t0).count();
                    }
                    graph_avg = graph_total / bench_iters;
                }
            } else {
                tC = [graph absoluteWithTensor:tA name:@"C"];

                MPSGraphDevice* gdev = [MPSGraphDevice deviceWithMTLDevice:device];
                MPSGraphShapedType* stA = [[MPSGraphShapedType alloc] initWithShape:shape dataType:mpsdt];

                MPSGraphExecutable* exec = [graph compileWithDevice:gdev
                                                              feeds:@{tA: stA}
                                                      targetTensors:@[tC]
                                                   targetOperations:nil
                                              compilationDescriptor:nil];
                if (exec) {
                    MPSGraphTensorData* dA = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpA
                                                                                     shape:shape
                                                                                  dataType:mpsdt];
                    MPSGraphTensorData* dC = [[MPSGraphTensorData alloc] initWithMTLBuffer:tmpC
                                                                                     shape:shape
                                                                                  dataType:mpsdt];
                    // Warmup
                    for (int w = 0; w < warmup_iters; ++w) {
                        [exec runWithMTLCommandQueue:queue
                                         inputsArray:@[dA]
                                        resultsArray:@[dC]
                                 executionDescriptor:nil];
                    }
                    // Benchmark
                    double graph_total = 0.0;
                    for (int it = 0; it < bench_iters; ++it) {
                        auto t0 = std::chrono::high_resolution_clock::now();
                        [exec runWithMTLCommandQueue:queue
                                         inputsArray:@[dA]
                                        resultsArray:@[dC]
                                 executionDescriptor:nil];
                        auto t1 = std::chrono::high_resolution_clock::now();
                        graph_total += std::chrono::duration<double>(t1 - t0).count();
                    }
                    graph_avg = graph_total / bench_iters;
                }
            }
        }
    }

    [tmpA release];
    if (tmpB)
        [tmpB release];
    [tmpC release];
    [queue release];

    // Hysteresis: keep native unless MPSGraph is >margin% faster.
    // Default 10% margin avoids noise-driven demotions that break fusion opportunities.
    static const double margin = 1.0 + EnvIntValue("JAX_METALLIB_ELEMWISE_AUTOTUNE_MARGIN_PCT", 10) / 100.0;
    bool native_wins = (native_avg <= graph_avg * margin);
    const char* dtype_str = (dtype_idx == 1) ? "f16" : "f32";
    MPS_LOG_WARN("autotune-elemwise: native %s (%s) numel=%llu time=%.3fms | MPSGraph time=%.3fms "
                 "(margin=%.0f%%) → %s\n",
                 native_kname, dtype_str, (unsigned long long)numel, native_avg * 1000.0, graph_avg * 1000.0,
                 (margin - 1.0) * 100.0, native_wins ? "native" : "MPSGraph");

    {
        std::lock_guard<std::mutex> lock(g_elem_autotune_mutex);
        g_elem_autotune_cache[key] = native_wins;
    }
    SaveAutotuneCacheToDisk();
    return native_wins;
}

// Helper: dispatch MPP GEMM for one batch slice.
// The kernel takes raw device pointers + M,N,K dimensions and constructs
// tensor views internally via make_tensor().
static void DispatchMppGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> bufA, size_t offA, id<MTLBuffer> bufB, size_t offB,
                            id<MTLBuffer> bufC, size_t offC, uint32_t M, uint32_t N, uint32_t K,
                            bool isFloat16) {
    // MPP kernel takes raw device pointers + M,N,K dimensions.
    // The kernel uses make_tensor() internally to construct tensor views.
    // Grid: ceil(N/64) x ceil(M/64) — tgid.x maps to N cols, tgid.y maps to M rows.
    uint32_t grid_m = (M + 63) / 64;
    uint32_t grid_n = (N + 63) / 64;
    uint32_t num_threads = [pipeline threadExecutionWidth] * 4;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:bufA offset:offA atIndex:0];
    [enc setBuffer:bufB offset:offB atIndex:1];
    [enc setBuffer:bufC offset:offC atIndex:2];
    [enc setBytes:&M length:sizeof(M) atIndex:3];
    [enc setBytes:&N length:sizeof(N) atIndex:4];
    [enc setBytes:&K length:sizeof(K) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(grid_n, grid_m, 1)
        threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
}

static id<MTLBuffer> NativeIntrinsicGEMMHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                mlir::Operation* op,
                                                const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op || inputs.size() < 2 || !inputs[0] || !inputs[1]) {
        return nil;
    }
    uint32_t M = 0, N = 0, K = 0, batch = 1;
    if (!ClassifyDotGeneralNN(op, &M, &N, &K, &batch)) {
        return nil;
    }
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, nullptr, &dtype)) {
        return nil;
    }
    int dtypeIdx = (dtype == MPSDataTypeFloat16) ? 1 : 0;
    int vi = AutotuneGemm(device, dtypeIdx, M, N, K);
    if (vi == kMpsGraphGemmVariant) {
        return nil; // Safety fallback: MPSGraph won, shouldn't reach here
    }

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0) {
        return nil;
    }
    id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:ComputeBufferResourceOptions()];
    if (!outBuf) {
        return nil;
    }

    id<MTLComputeCommandEncoder> enc = [commandBuffer computeCommandEncoder];
    if (!enc) {
        CFRelease((__bridge CFTypeRef)outBuf);
        return nil;
    }

    size_t elem_size = (dtype == MPSDataTypeFloat16) ? 2 : 4;

    if (vi == kMppGemmVariant) {
        // MPP tensor ops path — single dispatch with grid.z = batch
        const char* kernelName = (dtypeIdx == 1) ? "gemm_mpp_f16" : "gemm_mpp_f32";
        id<MTLComputePipelineState> pipeline = GetMppGemmPipeline(device, kernelName, 4);
        if (!pipeline) {
            [enc endEncoding];
            CFRelease((__bridge CFTypeRef)outBuf);
            return nil;
        }
        uint32_t grid_m = (M + 63) / 64;
        uint32_t grid_n = (N + 63) / 64;
        uint32_t num_threads = [pipeline threadExecutionWidth] * 4;
        [enc setComputePipelineState:pipeline];
        [enc setBuffer:inputs[0] offset:0 atIndex:0];
        [enc setBuffer:inputs[1] offset:0 atIndex:1];
        [enc setBuffer:outBuf offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&K length:sizeof(K) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(grid_n, grid_m, batch)
            threadsPerThreadgroup:MTLSizeMake(num_threads, 1, 1)];
    } else {
        // Simdgroup MMA path — single dispatch with grid.z = batch
        const auto& variant = kGemmVariants[vi];
        const char* kernelName = (dtypeIdx == 1) ? variant.f16_name : variant.f32_name;
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            [enc endEncoding];
            CFRelease((__bridge CFTypeRef)outBuf);
            return nil;
        }

        uint32_t grid_x = M / variant.bm;
        uint32_t grid_y = N / variant.bn;

        [enc setComputePipelineState:pipeline];
        [enc setBuffer:inputs[0] offset:0 atIndex:0];
        [enc setBuffer:inputs[1] offset:0 atIndex:1];
        [enc setBuffer:outBuf offset:0 atIndex:2];
        [enc setBytes:&M length:sizeof(M) atIndex:3];
        [enc setBytes:&N length:sizeof(N) atIndex:4];
        [enc setBytes:&K length:sizeof(K) atIndex:5];
        [enc dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
            threadsPerThreadgroup:MTLSizeMake(variant.num_threads, 1, 1)];
    }

    [enc endEncoding];
    return outBuf;
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

static std::string ElementwiseVec8KernelFunctionName(IntrinsicNativeKind kind, MPSDataType dtype) {
    const bool f16 = (dtype == MPSDataTypeFloat16);
    switch (kind) {
    case IntrinsicNativeKind::kNegate:
        return f16 ? "unary_neg_vec8_f16" : "unary_neg_vec8_f32";
    case IntrinsicNativeKind::kFloor:
        return f16 ? "unary_floor_vec8_f16" : "unary_floor_vec8_f32";
    case IntrinsicNativeKind::kRoundNearestEven:
        return f16 ? "unary_rint_vec8_f16" : "unary_rint_vec8_f32";
    case IntrinsicNativeKind::kRsqrt:
        return f16 ? "unary_rsqrt_vec8_f16" : "unary_rsqrt_vec8_f32";
    case IntrinsicNativeKind::kSign:
        return f16 ? "unary_sign_vec8_f16" : "unary_sign_vec8_f32";
    case IntrinsicNativeKind::kSquare:
        return f16 ? "unary_square_vec8_f16" : "unary_square_vec8_f32";
    case IntrinsicNativeKind::kAbs:
        return f16 ? "unary_abs_vec8_f16" : "unary_abs_vec8_f32";
    case IntrinsicNativeKind::kExp:
        return f16 ? "unary_exp_vec8_f16" : "unary_exp_vec8_f32";
    case IntrinsicNativeKind::kLog:
        return f16 ? "unary_log_vec8_f16" : "unary_log_vec8_f32";
    case IntrinsicNativeKind::kTanh:
        return f16 ? "unary_tanh_vec8_f16" : "unary_tanh_vec8_f32";
    case IntrinsicNativeKind::kCeil:
        return f16 ? "unary_ceil_vec8_f16" : "unary_ceil_vec8_f32";
    case IntrinsicNativeKind::kSin:
        return f16 ? "unary_sin_vec8_f16" : "unary_sin_vec8_f32";
    case IntrinsicNativeKind::kCos:
        return f16 ? "unary_cos_vec8_f16" : "unary_cos_vec8_f32";
    case IntrinsicNativeKind::kSqrt:
        return f16 ? "unary_sqrt_vec8_f16" : "unary_sqrt_vec8_f32";
    case IntrinsicNativeKind::kLogistic:
        return f16 ? "unary_sigmoid_vec8_f16" : "unary_sigmoid_vec8_f32";
    case IntrinsicNativeKind::kAdd:
        return f16 ? "binary_add_vec8_f16" : "binary_add_vec8_f32";
    case IntrinsicNativeKind::kSubtract:
        return f16 ? "binary_sub_vec8_f16" : "binary_sub_vec8_f32";
    case IntrinsicNativeKind::kMultiply:
        return f16 ? "binary_mul_vec8_f16" : "binary_mul_vec8_f32";
    case IntrinsicNativeKind::kDivide:
        return f16 ? "binary_div_vec8_f16" : "binary_div_vec8_f32";
    case IntrinsicNativeKind::kMaximum:
        return f16 ? "binary_max_vec8_f16" : "binary_max_vec8_f32";
    case IntrinsicNativeKind::kMinimum:
        return f16 ? "binary_min_vec8_f16" : "binary_min_vec8_f32";
    case IntrinsicNativeKind::kSelect:
        return f16 ? "ternary_select_vec8_f16" : "";
    default:
        return "";
    }
}

// Grid-stride kernel name lookup — covers all elementwise ops
static std::string ElementwiseStrideKernelFunctionName(IntrinsicNativeKind kind, MPSDataType dtype) {
    const bool f16 = (dtype == MPSDataTypeFloat16);
    switch (kind) {
    case IntrinsicNativeKind::kNegate:
        return f16 ? "unary_neg_stride_f16" : "unary_neg_stride_f32";
    case IntrinsicNativeKind::kFloor:
        return f16 ? "unary_floor_stride_f16" : "unary_floor_stride_f32";
    case IntrinsicNativeKind::kRoundNearestEven:
        return f16 ? "unary_rint_stride_f16" : "unary_rint_stride_f32";
    case IntrinsicNativeKind::kRsqrt:
        return f16 ? "unary_rsqrt_stride_f16" : "unary_rsqrt_stride_f32";
    case IntrinsicNativeKind::kSign:
        return f16 ? "unary_sign_stride_f16" : "unary_sign_stride_f32";
    case IntrinsicNativeKind::kSquare:
        return f16 ? "unary_square_stride_f16" : "unary_square_stride_f32";
    case IntrinsicNativeKind::kAbs:
        return f16 ? "unary_abs_stride_f16" : "unary_abs_stride_f32";
    case IntrinsicNativeKind::kExp:
        return f16 ? "unary_exp_stride_f16" : "unary_exp_stride_f32";
    case IntrinsicNativeKind::kLog:
        return f16 ? "unary_log_stride_f16" : "unary_log_stride_f32";
    case IntrinsicNativeKind::kTanh:
        return f16 ? "unary_tanh_stride_f16" : "unary_tanh_stride_f32";
    case IntrinsicNativeKind::kCeil:
        return f16 ? "unary_ceil_stride_f16" : "unary_ceil_stride_f32";
    case IntrinsicNativeKind::kSin:
        return f16 ? "unary_sin_stride_f16" : "unary_sin_stride_f32";
    case IntrinsicNativeKind::kCos:
        return f16 ? "unary_cos_stride_f16" : "unary_cos_stride_f32";
    case IntrinsicNativeKind::kSqrt:
        return f16 ? "unary_sqrt_stride_f16" : "unary_sqrt_stride_f32";
    case IntrinsicNativeKind::kLogistic:
        return f16 ? "unary_sigmoid_stride_f16" : "unary_sigmoid_stride_f32";
    case IntrinsicNativeKind::kAdd:
        return f16 ? "binary_add_stride_f16" : "binary_add_stride_f32";
    case IntrinsicNativeKind::kSubtract:
        return f16 ? "binary_sub_stride_f16" : "binary_sub_stride_f32";
    case IntrinsicNativeKind::kMultiply:
        return f16 ? "binary_mul_stride_f16" : "binary_mul_stride_f32";
    case IntrinsicNativeKind::kDivide:
        return f16 ? "binary_div_stride_f16" : "binary_div_stride_f32";
    case IntrinsicNativeKind::kMaximum:
        return f16 ? "binary_max_stride_f16" : "binary_max_stride_f32";
    case IntrinsicNativeKind::kMinimum:
        return f16 ? "binary_min_stride_f16" : "binary_min_stride_f32";
    case IntrinsicNativeKind::kSelect:
        return f16 ? "ternary_select_stride_f16" : "ternary_select_stride_f32";
    default:
        return "";
    }
}

// Compute capped grid size for stride kernels
static uint32_t ComputeStrideGridSize(uint32_t n4) {
    static const uint32_t cap = static_cast<uint32_t>(EnvIntValue("JAX_METALLIB_STRIDE_GRID_CAP", 65536));
    return (n4 < cap) ? n4 : cap;
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

// Compute-bound ops (exp, tanh, sin, etc.) are always faster as native kernels.
// Bandwidth-bound ops (add, mul, abs, etc.) may lose to MPSGraph due to dispatch overhead.
static bool IsComputeBoundElementwise(IntrinsicNativeKind kind) {
    switch (kind) {
    case IntrinsicNativeKind::kExp:
    case IntrinsicNativeKind::kLog:
    case IntrinsicNativeKind::kTanh:
    case IntrinsicNativeKind::kSin:
    case IntrinsicNativeKind::kCos:
    case IntrinsicNativeKind::kLogistic:
        return true;
    default:
        return false;
    }
}

// Returns true if the elementwise op is a binary op (2 inputs), false if unary (1 input).
static bool IsElementwiseBinaryOp(IntrinsicNativeKind kind) {
    switch (kind) {
    case IntrinsicNativeKind::kAdd:
    case IntrinsicNativeKind::kSubtract:
    case IntrinsicNativeKind::kMultiply:
    case IntrinsicNativeKind::kDivide:
    case IntrinsicNativeKind::kMaximum:
    case IntrinsicNativeKind::kMinimum:
        return true;
    default:
        return false;
    }
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
    if (vec_mode == 16 || vec_mode == 8) {
        vec_t = (dtype == MPSDataTypeFloat16) ? "half4" : "float4";
        n_param = (vec_mode == 16) ? "n4" : "n8";
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
    if (vec_mode == 16) {
        src << "    uint gid [[thread_position_in_grid]],\n";
        src << "    uint grid_size [[threads_per_grid]]\n";
    } else {
        src << "    uint gid [[thread_position_in_grid]]\n";
    }
    src << ") {\n";
    if (vec_mode == 16) {
        src << "  for (uint i = gid; i < " << n_param << "; i += grid_size) {\n";
    } else {
        src << "  if (gid < " << n_param << ") {\n";
    }

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

    if (vec_mode == 16) {
        // Stride mode: loop variable is 'i', single chain body
        emit_chain_body("i", "    ");
    } else if (vec_mode == 8) {
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
    case IntrinsicNativeKind::kReduceMeanLastAxis:
        if (use_threadgroup) {
            return f16 ? "reduce_mean_last_f16_tg" : "reduce_mean_last_f32_tg";
        }
        return f16 ? "reduce_mean_last_f16_lin" : "reduce_mean_last_f32_lin";
    case IntrinsicNativeKind::kReduceSumAxis0:
        return f16 ? "reduce_sum_axis0_f16" : "reduce_sum_axis0_f32";
    case IntrinsicNativeKind::kReduceMaxAxis0:
        return f16 ? "reduce_max_axis0_f16" : "reduce_max_axis0_f32";
    case IntrinsicNativeKind::kReduceMinAxis0:
        return f16 ? "reduce_min_axis0_f16" : "reduce_min_axis0_f32";
    case IntrinsicNativeKind::kReduceMeanAxis0:
        return f16 ? "reduce_mean_axis0_f16" : "reduce_mean_axis0_f32";
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
    if (v >= 1024) {
        return 1024;
    }
    if (v >= 512) {
        return 512;
    }
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

static NSUInteger OptimalReduceThreadgroupSize(id<MTLComputePipelineState> pipeline, uint32_t cols) {
    static const bool disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_ADAPTIVE_REDUCE_TG");
    NSUInteger max_tg = Pow2Floor(MIN((NSUInteger)1024, pipeline.maxTotalThreadsPerThreadgroup));
    if (max_tg == 0)
        max_tg = 1;
    if (disabled)
        return max_tg;
    const int min_work = EnvIntValue("JAX_METALLIB_REDUCE_MIN_WORK_PER_THREAD", 8);
    if (min_work <= 0)
        return max_tg;
    NSUInteger desired_tg = Pow2Floor((NSUInteger)(cols / (uint32_t)min_work));
    if (desired_tg < 32)
        desired_tg = 32;
    return MIN(desired_tg, max_tg);
}

static void EncodeRowReduceDispatch(id<MTLComputeCommandEncoder> encoder,
                                    id<MTLComputePipelineState> pipeline, uint32_t rows, uint32_t cols) {
    NSUInteger tg = OptimalReduceThreadgroupSize(pipeline, cols);
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

static void ComputeRowReduceDispatchConfig(id<MTLComputePipelineState> pipeline, uint32_t rows, uint32_t cols,
                                           MTLSize* grid_out, MTLSize* tg_out) {
    NSUInteger tg = OptimalReduceThreadgroupSize(pipeline, cols);
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

    // Stride kernels: top tier for large tensors — grid-stride loop for high bandwidth utilization
    const bool stride_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_STRIDE_KERNELS");
    static const uint64_t stride_min_numel =
        static_cast<uint64_t>(EnvIntValue("JAX_METALLIB_STRIDE_MIN_NUMEL",
                                          1048576)); // 1M elements default (~4MB f32)
    const bool stride_candidate = !stride_disabled && !vec_disabled && IsFloat16Or32(dtype) &&
                                  (numel % 4 == 0) && numel >= stride_min_numel;
    std::string strideName = stride_candidate ? ElementwiseStrideKernelFunctionName(kind, dtype) : "";
    const bool use_stride = !strideName.empty();

    const uint64_t vec8_min_numel = (dtype == MPSDataTypeFloat32) ? 262144u : 8u;
    const bool vec8_candidate =
        !use_stride && !vec_disabled && IsFloat16Or32(dtype) && (numel % 8 == 0) && numel >= vec8_min_numel;
    std::string vec8Name = vec8_candidate ? ElementwiseVec8KernelFunctionName(kind, dtype) : "";
    const bool use_vec8 = !vec8Name.empty();
    const bool use_vec4 =
        !use_vec8 && !use_stride && !vec_disabled && vec4_dtype_ok && supports_vec4_kind && (numel % 4 == 0);

    std::string kernelName;
    if (use_stride) {
        kernelName = std::move(strideName);
    } else if (use_vec8) {
        kernelName = std::move(vec8Name);
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
    uint32_t n = static_cast<uint32_t>(
        use_stride ? (numel / 4) : (use_vec8 ? (numel / 8) : (use_vec4 ? (numel / 4) : numel)));

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

    if (use_stride) {
        // Stride kernels: dispatch capped grid as threadgroups (avoids non-uniform dispatch overhead)
        uint32_t grid_threads = ComputeStrideGridSize(n);
        NSUInteger tg = ResolveLinearThreadgroupSize(pipeline, dtype);
        NSUInteger tg_count = ((NSUInteger)grid_threads + tg - 1) / tg;
        [encoder dispatchThreadgroups:MTLSizeMake(tg_count, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    } else {
        EncodeLinearDispatch(encoder, pipeline, n, dtype);
    }
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
    bool is_axis0 =
        (kind == IntrinsicNativeKind::kReduceSumAxis0 || kind == IntrinsicNativeKind::kReduceMaxAxis0 ||
         kind == IntrinsicNativeKind::kReduceMinAxis0 || kind == IntrinsicNativeKind::kReduceMeanAxis0);

    uint32_t rows, cols;
    bool use_threadgroup;
    if (is_axis0) {
        // Axis-0 reduce on [R, C]: rows = reduce dim, cols = output dim
        rows = static_cast<uint32_t>(inShape[0]);
        cols = static_cast<uint32_t>(inShape[1]);
        use_threadgroup = false;
    } else {
        // cols = product of the reduced (trailing-suffix) dims; for a single
        // last-axis reduce this is simply inShape.back().
        uint64_t cols64 = 1;
        for (auto d : reduceOp.getDimensions()) {
            cols64 *= static_cast<uint64_t>(inShape[d]);
        }
        if (cols64 == 0 || cols64 > std::numeric_limits<uint32_t>::max()) {
            return nil;
        }
        cols = static_cast<uint32_t>(cols64);
        rows = static_cast<uint32_t>(outNumel);
        const uint32_t tg_cols_threshold =
            static_cast<uint32_t>(EnvIntValue("JAX_METALLIB_REDUCE_TG_COLS_THRESHOLD", 512));
        use_threadgroup = cols >= tg_cols_threshold;
    }

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
        // A single threadgroup serializes on one SM; for huge single-row
        // reductions, split the columns across G threadgroups into partials,
        // then combine with a second dispatch of the same kernel (~10-100x).
        NSUInteger tg_split = MIN((NSUInteger)1024, (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup);
        if (tg_split == 0) {
            tg_split = 1;
        }
        uint32_t groups = 0;
        if (rows == 1 && dtype == MPSDataTypeFloat32 && cols >= 4096) {
            const uint32_t candidates[] = {1024u, 512u, 256u, 128u, 64u, 32u, 16u};
            for (uint32_t g : candidates) {
                if (cols % g == 0 && cols / g >= 512) {
                    groups = g;
                    break;
                }
            }
        }
        if (groups > 0) {
            const size_t partials_bytes = (size_t)groups * sizeof(float);
            id<MTLBuffer> partials = AcquireReusableOutputBuffer(device, partials_bytes);
            if (!partials) {
                CFRelease((__bridge CFTypeRef)out);
                return nil;
            }
            // Recycle the scratch partials only after the GPU work that
            // produces/consumes them has completed (they are written by the
            // first dispatch and read by the combine dispatch on this command
            // buffer, which may use unretained references).
            void* partials_dev_key = (__bridge void*)device;
            id<MTLBuffer> captured_partials = partials;
            [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
              RecycleReusableOutputBuffer(partials_dev_key, partials_bytes,
                                          (__bridge void*)captured_partials);
            }];
            uint32_t rows_g = groups;
            uint32_t cols_g = cols / groups;
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:inputs[0] offset:0 atIndex:0];
            [encoder setBuffer:partials offset:0 atIndex:1];
            [encoder setBytes:&rows_g length:sizeof(rows_g) atIndex:2];
            [encoder setBytes:&cols_g length:sizeof(cols_g) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(tg_split, 1, 1)];
            [encoder endEncoding];
            // Combine: reduce the G partials into the scalar output.
            id<MTLComputeCommandEncoder> enc2 = [commandBuffer computeCommandEncoder];
            const bool combine_tg = groups >= 512;
            id<MTLComputePipelineState> combine_pipeline =
                GetFastNativePipeline(device, ReduceKernelFunctionName(kind, dtype, combine_tg));
            if (!combine_pipeline) {
                [enc2 endEncoding];
                CFRelease((__bridge CFTypeRef)out);
                return nil;
            }
            // The kernels read rows/cols from buffer indices 2/3; they must be
            // set on the combine encoder too (this was previously missing).
            uint32_t rows_c = 1;
            uint32_t cols_c = groups;
            [enc2 setComputePipelineState:combine_pipeline];
            [enc2 setBuffer:partials offset:0 atIndex:0];
            [enc2 setBuffer:out offset:0 atIndex:1];
            [enc2 setBytes:&rows_c length:sizeof(rows_c) atIndex:2];
            [enc2 setBytes:&cols_c length:sizeof(cols_c) atIndex:3];
            if (combine_tg) {
                EncodeRowReduceDispatch(enc2, combine_pipeline, rows_c, cols_c);
            } else {
                EncodeLinearDispatch(enc2, combine_pipeline, rows_c, dtype);
            }
            [enc2 endEncoding];
            return out;
        }
        EncodeRowReduceDispatch(encoder, pipeline, rows, cols);
    } else {
        uint32_t dispatch_count = is_axis0 ? cols : rows;
        EncodeLinearDispatch(encoder, pipeline, dispatch_count, dtype);
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

// Forward declarations for sort constants (used in BakeNativeStep, defined with IsEligibleSortNativeOp)
// The stable keyed bitonic kernels keep a uint key and a ushort index per
// padded element (6 bytes), so rows are capped at 4096 (24 KiB); longer
// rows use the MPSGraph argsort + gather path.
static constexpr uint32_t kMaxNativeSortElements = 4096;
static NSUInteger SortThreadgroupBytes(uint32_t padded_n) {
    NSUInteger bytes = (NSUInteger)padded_n * (sizeof(uint32_t) + sizeof(uint16_t));
    return (bytes + 15) & ~(NSUInteger)15;
}
// Threads per row for the bitonic row sort. A fixed 32 (one simdgroup) left
// the GPU idle when there were few rows: 8x4096 took 0.53 ms at 32 threads vs
// 0.24 ms at 256. The best size measured gives each thread ~8 compare-swaps
// per stage, i.e. padded_n / 16, clamped to [32, 256] (4096x256 stays at 32,
// 4096x1024 -> 64, 8x4096 -> 256). JAX_METALLIB_SORT_TG_CAP=N forces a fixed
// cap of N (32 restores the old behaviour); 0 (default) is automatic.
static NSUInteger SortThreadgroupSize(uint32_t padded_n, id<MTLComputePipelineState> pipeline) {
    static const int64_t fixed = EnvIntValue("JAX_METALLIB_SORT_TG_CAP", 0);
    NSUInteger tg = fixed > 0 ? (NSUInteger)std::min<int64_t>(fixed, 1024)
                              : std::max<NSUInteger>(32, std::min<NSUInteger>(256, padded_n / 16));
    tg = std::min(tg, (NSUInteger)(padded_n / 2));
    if (pipeline) {
        tg = std::min(tg, (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup);
    }
    return std::max<NSUInteger>(tg, 1);
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
    if (simpleKind == IntrinsicNativeKind::kTransposeAlias && IsEligibleTransposeAliasOp(op)) {
        ns->baked_kind = NativeStep::BakedKind::kTransposeAlias;
        return true;
    }
    // Native transpose: swap last 2 dims using a fast Metal kernel
    if (simpleKind == IntrinsicNativeKind::kTransposeAlias) {
        uint32_t rows = 0, cols = 0, batch = 1;
        if (IsEligibleTransposeNativeOp(op, &rows, &cols, &batch)) {
            id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, "transpose2d_f32");
            if (pipeline) {
                ns->baked_kind = NativeStep::BakedKind::kTranspose2D;
                ns->baked_pipeline = pipeline;
                ns->baked_transpose_rows = rows;
                ns->baked_transpose_cols = cols;
                ns->baked_transpose_batch = batch;
                uint32_t grid_x = (cols + 31) / 32;
                uint32_t grid_y = (rows + 31) / 32;
                ns->baked_grid = MTLSizeMake(grid_x, grid_y, batch);
                ns->baked_tg = MTLSizeMake(32, 8, 1);
                return true;
            }
        }
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
        // Constants need CPU write → use Shared storage.
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
        // Generalized scatter: detect combiner type (add/mul/max/min)
        auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
        std::string combiner = scatterOp ? ClassifyScatterCombiner(scatterOp) : "add";
        const bool f16 = (scatterDtype == MPSDataTypeFloat16);
        std::string kernelName = "scatter_" + combiner + "_rows_" + (f16 ? "f16" : "f32");
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

    // Native gather (embedding lookup)
    if (simpleKind == IntrinsicNativeKind::kGatherRows && IsEligibleGatherRowsNativeOp(op)) {
        std::vector<int64_t> operandShape, indexShape;
        MPSDataType gatherDtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &operandShape, nullptr, &gatherDtype) ||
            !GetStaticTensorInfo(op->getOperand(1).getType(), &indexShape, nullptr, nullptr) ||
            operandShape.size() != 2 || indexShape.size() != 2) {
            return false;
        }
        uint32_t num_indices = static_cast<uint32_t>(indexShape[0]);
        uint32_t cols = static_cast<uint32_t>(operandShape[1]);
        uint32_t table_rows = static_cast<uint32_t>(operandShape[0]);
        uint64_t total64 = static_cast<uint64_t>(num_indices) * static_cast<uint64_t>(cols);
        if (total64 == 0 || total64 > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        const bool f16 = (gatherDtype == MPSDataTypeFloat16);
        std::string kernelName = f16 ? "gather_rows_f16" : "gather_rows_f32";
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        ns->baked_kind = NativeStep::BakedKind::kGatherRows;
        ns->baked_pipeline = pipeline;
        ns->baked_gather_num_indices = num_indices;
        ns->baked_gather_cols = cols;
        ns->baked_gather_table_rows = table_rows;
        ComputeLinearDispatchConfig(pipeline, static_cast<uint32_t>(total64), &ns->baked_grid, &ns->baked_tg,
                                    gatherDtype);
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

        // Stride kernels: top tier for large tensors
        const bool stride_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_STRIDE_KERNELS");
        static const uint64_t stride_min_numel =
            static_cast<uint64_t>(EnvIntValue("JAX_METALLIB_STRIDE_MIN_NUMEL", 1048576));
        const bool stride_candidate = !stride_disabled && !vec_disabled && IsFloat16Or32(dtype) &&
                                      (numel % 4 == 0) && numel >= stride_min_numel;
        std::string strideName = stride_candidate ? ElementwiseStrideKernelFunctionName(elemKind, dtype) : "";
        const bool use_stride = !strideName.empty();

        const uint64_t vec8_min_numel = (dtype == MPSDataTypeFloat32) ? 262144u : 8u;
        const bool vec8_candidate = !use_stride && !vec_disabled && IsFloat16Or32(dtype) &&
                                    (numel % 8 == 0) && numel >= vec8_min_numel;
        std::string vec8Name = vec8_candidate ? ElementwiseVec8KernelFunctionName(elemKind, dtype) : "";
        const bool use_vec8 = !vec8Name.empty();
        const bool use_vec4 = !use_vec8 && !use_stride && !vec_disabled && vec4_dtype_ok &&
                              supports_vec4_kind && (numel % 4 == 0);
        std::string kernelName;
        if (use_stride) {
            kernelName = std::move(strideName);
        } else if (use_vec8) {
            kernelName = std::move(vec8Name);
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
        ns->baked_use_vec4 = use_vec4 || use_vec8 || use_stride;
        ns->baked_use_stride = use_stride;
        ns->baked_n = static_cast<uint32_t>(
            use_stride ? (numel / 4) : (use_vec8 ? (numel / 8) : (use_vec4 ? (numel / 4) : numel)));
        if (use_stride) {
            // Stride: store threadgroup count (not thread count) for dispatchThreadgroups
            uint32_t grid_threads = ComputeStrideGridSize(ns->baked_n);
            NSUInteger tg = ResolveLinearThreadgroupSize(pipeline, dtype);
            NSUInteger tg_count = ((NSUInteger)grid_threads + tg - 1) / tg;
            ns->baked_grid = MTLSizeMake(tg_count, 1, 1);
            ns->baked_tg = MTLSizeMake(tg, 1, 1);
        } else {
            ComputeLinearDispatchConfig(pipeline, ns->baked_n, &ns->baked_grid, &ns->baked_tg, dtype);
        }
        return true;
    }

    IntrinsicNativeKind reduceKind = ClassifyReduceKind(op);
    if (reduceKind != IntrinsicNativeKind::kNone && IsEligibleReduceNativeOp(op, reduceKind)) {
        std::vector<int64_t> inShape;
        MPSDataType dtype = MPSDataTypeInvalid;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype) || inShape.empty()) {
            return false;
        }

        uint64_t inNumel = 1;
        for (auto d : inShape)
            inNumel *= static_cast<uint64_t>(d);
        // cols = product of the reduced dims (trailing suffix); equals
        // inShape.back() for the single last-axis case.
        uint64_t colsProd = 1;
        if (auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op)) {
            for (auto d : reduceOp.getDimensions()) {
                colsProd *= static_cast<uint64_t>(inShape[d]);
            }
        } else {
            colsProd = static_cast<uint64_t>(inShape.back());
        }
        if (colsProd == 0 || colsProd > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        uint32_t cols = static_cast<uint32_t>(colsProd);
        uint32_t rows = static_cast<uint32_t>(inNumel / cols);

        if (reduceKind == IntrinsicNativeKind::kSoftmaxLastAxis) {
            const bool f16 = (dtype == MPSDataTypeFloat16);
            // Use single-pass full softmax when cols fit in threadgroup memory
            if (cols <= 8192) {
                std::string fullKernelName = op->hasAttr("__softmax_f16_xla") ? "softmax_full_f16_xla_tg"
                                             : f16                            ? "softmax_full_f16_tg"
                                                                              : "softmax_full_f32_tg";
                id<MTLComputePipelineState> pFull = GetFastNativePipeline(device, fullKernelName);
                if (pFull) {
                    ns->baked_kind = NativeStep::BakedKind::kSoftmax;
                    ns->baked_pipeline = pFull;
                    ns->baked_softmax_pass2_pipeline = nil; // single-pass
                    ns->baked_softmax_maxsum = nil;
                    ns->baked_rows = rows;
                    ns->baked_cols = cols;
                    ComputeRowReduceDispatchConfig(pFull, rows, cols, &ns->baked_grid, &ns->baked_tg);
                    return true;
                }
            }
            // The XLA-rounding f16 form is only fused for cols <= 8192.
            if (op->hasAttr("__softmax_f16_xla")) {
                return false;
            }
            // Fallback: 2-pass softmax for large cols (>8K)
            std::string pass1Name = f16 ? "softmax_pass1_f16_tg" : "softmax_pass1_f32_tg";
            std::string pass2Name = f16 ? "softmax_pass2_f16" : "softmax_pass2_f32";
            id<MTLComputePipelineState> p1 = GetFastNativePipeline(device, pass1Name);
            id<MTLComputePipelineState> p2 = GetFastNativePipeline(device, pass2Name);
            if (!p1 || !p2) {
                return false;
            }

            size_t maxsumBytes = (size_t)rows * 2 * sizeof(float);
            id<MTLBuffer> maxsumBuf = [device newBufferWithLength:maxsumBytes
                                                          options:ComputeBufferResourceOptions()];
            if (!maxsumBuf)
                return false;

            ns->baked_kind = NativeStep::BakedKind::kSoftmax;
            ns->baked_pipeline = p1;
            ns->baked_softmax_pass2_pipeline = p2;
            ns->baked_softmax_maxsum = maxsumBuf;
            ns->baked_rows = rows;
            ns->baked_cols = cols;
            ComputeRowReduceDispatchConfig(p1, rows, cols, &ns->baked_grid, &ns->baked_tg);

            uint32_t totalElements = rows * cols;
            ComputeLinearDispatchConfig(p2, totalElements, &ns->baked_softmax_pass2_grid,
                                        &ns->baked_softmax_pass2_tg, dtype);
            return true;
        }

        if (reduceKind == IntrinsicNativeKind::kSoftmaxNumeratorLastAxis) {
            // Fused softmax numerator: single kernel exp(x - max(x, axis=-1))
            // Uses threadgroup memory to cache row data (1 global read + 1 global write).
            const bool f16 = (dtype == MPSDataTypeFloat16);
            // Only use TG kernel if cols fit in threadgroup memory (32KB max → 8K floats)
            if (cols <= 8192) {
                std::string kname = f16 ? "softmax_numerator_f16_tg" : "softmax_numerator_f32_tg";
                id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kname);
                if (pipeline) {
                    ns->baked_kind = NativeStep::BakedKind::kSoftmax;
                    ns->baked_pipeline = pipeline;
                    ns->baked_softmax_pass2_pipeline = nil; // single-pass, no pass2
                    ns->baked_softmax_maxsum = nil;
                    ns->baked_rows = rows;
                    ns->baked_cols = cols;
                    ComputeRowReduceDispatchConfig(pipeline, rows, cols, &ns->baked_grid, &ns->baked_tg);
                    return true;
                }
            }
            // Fallback: let it go to graph
            return false;
        }

        if (reduceKind == IntrinsicNativeKind::kLayerNormLastAxis) {
            const bool f16 = (dtype == MPSDataTypeFloat16);
            std::string pass1Name = f16 ? "layernorm_pass1_f16_tg" : "layernorm_pass1_f32_tg";
            std::string pass2Name = f16 ? "layernorm_pass2_f16" : "layernorm_pass2_f32";
            id<MTLComputePipelineState> p1 = GetFastNativePipeline(device, pass1Name);
            id<MTLComputePipelineState> p2 = GetFastNativePipeline(device, pass2Name);
            if (!p1 || !p2) {
                return false;
            }

            size_t statsBytes = (size_t)rows * 2 * sizeof(float);
            id<MTLBuffer> statsBuf = [device newBufferWithLength:statsBytes
                                                         options:ComputeBufferResourceOptions()];
            if (!statsBuf)
                return false;

            // Extract eps from the fused attribute
            float eps = 1e-5f;
            if (auto epsAttr = op->getAttrOfType<mlir::FloatAttr>("__layernorm_eps")) {
                eps = epsAttr.getValueAsDouble();
            }

            ns->baked_kind = NativeStep::BakedKind::kLayerNorm;
            ns->baked_pipeline = p1;
            ns->baked_layernorm_pass2_pipeline = p2;
            ns->baked_layernorm_stats = statsBuf;
            ns->baked_rows = rows;
            ns->baked_cols = cols;
            ns->baked_layernorm_eps = eps;
            ComputeRowReduceDispatchConfig(p1, rows, cols, &ns->baked_grid, &ns->baked_tg);

            uint32_t totalElements = rows * cols;
            ComputeLinearDispatchConfig(p2, totalElements, &ns->baked_layernorm_pass2_grid,
                                        &ns->baked_layernorm_pass2_tg, dtype);
            return true;
        }

        bool is_axis0 = (reduceKind == IntrinsicNativeKind::kReduceSumAxis0 ||
                         reduceKind == IntrinsicNativeKind::kReduceMaxAxis0 ||
                         reduceKind == IntrinsicNativeKind::kReduceMinAxis0 ||
                         reduceKind == IntrinsicNativeKind::kReduceMeanAxis0);
        if (is_axis0) {
            // Axis-0 reduce on [R, C]: rows = reduce dim (R), cols = output dim (C)
            // Kernel: each thread handles one column, loops over rows.
            // use_threadgroup is false (no TG variant needed for axis-0).
            std::string kernelName = ReduceKernelFunctionName(reduceKind, dtype, false);
            if (kernelName.empty()) {
                return false;
            }
            id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
            if (!pipeline) {
                return false;
            }
            ns->baked_kind = NativeStep::BakedKind::kReduce;
            ns->baked_pipeline = pipeline;
            ns->baked_rows = static_cast<uint32_t>(inShape[0]); // reduce dimension
            ns->baked_cols = static_cast<uint32_t>(inShape[1]); // output dimension
            ns->baked_reduce_threadgroup = false;
            ComputeLinearDispatchConfig(pipeline, ns->baked_cols, &ns->baked_grid, &ns->baked_tg, dtype);
            return true;
        }

        uint64_t outNumel = 0;
        if (!GetStaticTensorInfo(op->getResult(0).getType(), nullptr, &outNumel, nullptr) || outNumel == 0 ||
            outNumel > std::numeric_limits<uint32_t>::max()) {
            return false;
        }
        rows = static_cast<uint32_t>(outNumel);
        const uint32_t tg_cols_threshold =
            static_cast<uint32_t>(EnvIntValue("JAX_METALLIB_REDUCE_TG_COLS_THRESHOLD", 512));
        bool use_threadgroup = cols >= tg_cols_threshold;
        std::string kernelName = ReduceKernelFunctionName(reduceKind, dtype, use_threadgroup);
        if (kernelName.empty()) {
            return false;
        }
        // Single-row huge reduces use a two-stage group split (stage 1: G
        // threadgroups reduce column groups into partials; stage 2: combine).
        // Baked directly with a retained partials scratch buffer so hot loops
        // avoid the per-execution handler path entirely.
        if (rows == 1 && dtype == MPSDataTypeFloat32 && cols >= 4096 &&
            (reduceKind == IntrinsicNativeKind::kReduceSumLastAxis ||
             reduceKind == IntrinsicNativeKind::kReduceMaxLastAxis ||
             reduceKind == IntrinsicNativeKind::kReduceMinLastAxis)) {
            uint32_t groups = 0;
            const uint32_t candidates[] = {1024u, 512u, 256u, 128u, 64u, 32u, 16u};
            for (uint32_t g : candidates) {
                if (cols % g == 0 && cols / g >= 512) {
                    groups = g;
                    break;
                }
            }
            if (groups > 0) {
                id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
                if (!pipeline) {
                    return false;
                }
                const bool combine_tg = groups >= 512;
                std::string combineName = ReduceKernelFunctionName(reduceKind, dtype, combine_tg);
                if (combineName.empty()) {
                    return false;
                }
                id<MTLComputePipelineState> combine = GetFastNativePipeline(device, combineName);
                if (!combine) {
                    return false;
                }
                id<MTLBuffer> partials =
                    [device newBufferWithLength:(size_t)groups * sizeof(float)
                                        options:ComputeBufferResourceOptions()];
                if (!partials) {
                    return false;
                }
                NSUInteger tg_split =
                    MIN((NSUInteger)1024, (NSUInteger)pipeline.maxTotalThreadsPerThreadgroup);
                if (tg_split == 0) {
                    tg_split = 1;
                }
                ns->baked_kind = NativeStep::BakedKind::kReduceSplit;
                ns->baked_pipeline = pipeline;
                ns->baked_reduce_split_combine_pipeline = combine;
                ns->baked_reduce_split_partials = partials;
                ns->baked_reduce_split_groups = groups;
                ns->baked_rows = groups;        // stage-1 rows
                ns->baked_cols = cols / groups; // stage-1 cols
                ns->baked_grid = MTLSizeMake((NSUInteger)groups, 1, 1);
                ns->baked_tg = MTLSizeMake(tg_split, 1, 1);
                ns->baked_reduce_split_combine_tg_dispatch = combine_tg;
                if (combine_tg) {
                    ComputeRowReduceDispatchConfig(combine, 1, groups,
                                                   &ns->baked_reduce_split_combine_grid,
                                                   &ns->baked_reduce_split_combine_tg);
                } else {
                    ComputeLinearDispatchConfig(combine, 1, &ns->baked_reduce_split_combine_grid,
                                                &ns->baked_reduce_split_combine_tg, dtype);
                }
                return true;
            }
            // groups == 0: fall through to the regular single-dispatch bake.
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
            ComputeRowReduceDispatchConfig(pipeline, rows, cols, &ns->baked_grid, &ns->baked_tg);
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

    {
        uint32_t gM = 0, gN = 0, gK = 0, gBatch = 0;
        if (ClassifyDotGeneralNN(op, &gM, &gN, &gK, &gBatch)) {
            MPSDataType dtype = MPSDataTypeInvalid;
            if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, nullptr, &dtype)) {
                return false;
            }
            int dtypeIdx = (dtype == MPSDataTypeFloat16) ? 1 : 0;
            int vi = AutotuneGemm(device, dtypeIdx, gM, gN, gK);
            if (vi == kMpsGraphGemmVariant) {
                return false; // Safety fallback: MPSGraph won, shouldn't reach baking
            }

            id<MTLComputePipelineState> pipeline = nil;
            uint32_t grid_x = 0, grid_y = 0;
            uint32_t num_threads = 0;

            if (vi == kMppGemmVariant) {
                const char* kernelName = (dtypeIdx == 1) ? "gemm_mpp_f16" : "gemm_mpp_f32";
                pipeline = GetMppGemmPipeline(device, kernelName, 4);
                if (!pipeline)
                    return false;
                // MPP: ceil division for edge-check tiles
                grid_x = (gN + 63) / 64; // N tiles (tgid.x)
                grid_y = (gM + 63) / 64; // M tiles (tgid.y)
                num_threads = [pipeline threadExecutionWidth] * 4;
            } else {
                const auto& variant = kGemmVariants[vi];
                const char* kernelName = (dtypeIdx == 1) ? variant.f16_name : variant.f32_name;
                pipeline = GetFastNativePipeline(device, kernelName);
                if (!pipeline)
                    return false;
                grid_x = gM / variant.bm;
                grid_y = gN / variant.bn;
                num_threads = variant.num_threads;
            }

            ns->baked_kind = NativeStep::BakedKind::kGEMM;
            ns->baked_pipeline = pipeline;
            ns->baked_gemm_m = gM;
            ns->baked_gemm_n = gN;
            ns->baked_gemm_k = gK;
            ns->baked_gemm_batch = gBatch;
            ns->baked_gemm_variant = vi;
            ns->baked_grid = MTLSizeMake(grid_x, grid_y, gBatch);
            ns->baked_tg = MTLSizeMake(num_threads, 1, 1);
            return true;
        }
    }

    // --- Small/unaligned GEMM baking (fallback for matrices that fail ClassifyDotGeneralNN) ---
    {
        uint32_t gM = 0, gN = 0, gK = 0, gBatch = 0;
        if (ClassifyDotGeneralSmall(op, &gM, &gN, &gK, &gBatch)) {
            MPSDataType dtype = MPSDataTypeInvalid;
            if (!GetStaticTensorInfo(op->getOperand(0).getType(), nullptr, nullptr, &dtype)) {
                return false;
            }

            // CPU BLAS fallback for small f32 matmuls — avoids GPU dispatch overhead.
            // 128x128x128 (2.1M) measures ~7.8ms on the GPU naive path vs ~0.1ms on
            // the CPU; the crossover vs the mixed-fp16 GPU path sits near ~4.5M.
            static const uint64_t cpu_blas_max_mnk =
                static_cast<uint64_t>(EnvIntValue("JAX_METALLIB_CPU_BLAS_MAX_MNK", 4500000));
            uint64_t flops = (uint64_t)gM * gN * gK;
            if (dtype == MPSDataTypeFloat32 && gBatch == 1 && flops < cpu_blas_max_mnk) {
                ns->baked_kind = NativeStep::BakedKind::kCpuBLAS;
                ns->baked_gemm_m = gM;
                ns->baked_gemm_n = gN;
                ns->baked_gemm_k = gK;
                ns->baked_gemm_batch = 1;
                return true;
            }

            const char* kernelName = (dtype == MPSDataTypeFloat16) ? "gemm_naive_f16" : "gemm_naive_f32";
            id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
            if (!pipeline) {
                return false;
            }
            uint32_t tg_x = std::min(gN, (uint32_t)16);
            uint32_t tg_y = std::min(gM, (uint32_t)16);

            ns->baked_kind = NativeStep::BakedKind::kGEMM;
            ns->baked_pipeline = pipeline;
            ns->baked_gemm_m = gM;
            ns->baked_gemm_n = gN;
            ns->baked_gemm_k = gK;
            ns->baked_gemm_batch = gBatch;
            ns->baked_gemm_variant = kNaiveGemmVariant;
            // Grid: ceil(N/tg_x) x ceil(M/tg_y) x batch threadgroups
            ns->baked_grid = MTLSizeMake((gN + tg_x - 1) / tg_x, (gM + tg_y - 1) / tg_y, gBatch);
            ns->baked_tg = MTLSizeMake(tg_x, tg_y, 1);
            return true;
        }
    }

    // --- Sort baking ---
    if (IsEligibleSortNativeOp(op)) {
        std::vector<int64_t> inShape;
        if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, nullptr) ||
            inShape.empty()) {
            return false;
        }
        uint32_t num_rows = 1;
        uint32_t n;
        if (inShape.size() == 1) {
            n = static_cast<uint32_t>(inShape[0]);
        } else {
            // 2D: sort each row independently
            num_rows = static_cast<uint32_t>(inShape[0]);
            n = static_cast<uint32_t>(inShape[1]);
        }

        uint32_t padded_n = 1;
        while (padded_n < n)
            padded_n <<= 1;

        const char* kernelName = (num_rows > 1) ? "bitonic_sort_rows_asc_f32" : "bitonic_sort_asc_f32";
        id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
        if (!pipeline) {
            return false;
        }
        // Threads per row: see SortThreadgroupSize.
        NSUInteger tg_size = SortThreadgroupSize(padded_n, pipeline);
        if (tg_size == 0)
            tg_size = 1;

        ns->baked_kind = NativeStep::BakedKind::kSort;
        ns->baked_pipeline = pipeline;
        ns->baked_sort_n = n;
        ns->baked_sort_padded_n = padded_n;
        ns->baked_sort_num_rows = num_rows;
        ns->baked_grid = MTLSizeMake(num_rows, 1, 1);
        ns->baked_tg = MTLSizeMake(tg_size, 1, 1);
        return true;
    }

    // --- CPU Cholesky baking (LAPACK spotrf_ for small matrices) ---
    {
        auto choleskyOp = mlir::dyn_cast_or_null<mlir::stablehlo::CholeskyOp>(op);
        if (choleskyOp) {
            auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
            auto shape = resultType.getShape();
            if (shape.size() >= 2 && resultType.getElementType().isF32()) {
                int64_t n = shape.back();
                int64_t batch = 1;
                for (int64_t d = 0; d + 2 < static_cast<int64_t>(shape.size()); ++d) {
                    batch *= shape[d];
                }
                if (shape[shape.size() - 2] == n && n > 0 && n <= 256 && batch > 0 &&
                    batch <= 1024) {
                    bool lower = true;
                    if (choleskyOp.getLowerAttr()) {
                        lower = choleskyOp.getLower();
                    }
                    ns->baked_kind = NativeStep::BakedKind::kCpuCholesky;
                    ns->baked_cholesky_n = static_cast<uint32_t>(n);
                    ns->baked_cholesky_batch = static_cast<uint32_t>(batch);
                    ns->baked_cholesky_lower = lower;
                    return true;
                }
            }
        }
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

    // Constants need CPU write → use Shared storage.
    id<MTLBuffer> buf = [device newBufferWithLength:outBytes options:ComputeBufferResourceOptions()];
    if (!buf || !WriteScalarConstantData(constOp, dtype, [buf contents])) {
        if (buf) {
            CFRelease((__bridge CFTypeRef)buf);
        }
        return nil;
    }
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

    auto scatterOp = mlir::dyn_cast<mlir::stablehlo::ScatterOp>(op);
    std::string combiner = scatterOp ? ClassifyScatterCombiner(scatterOp) : "add";
    const bool f16 = (scatterDtype == MPSDataTypeFloat16);
    std::string kernelNameStr = "scatter_" + combiner + "_rows_" + (f16 ? "f16" : "f32");
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelNameStr);
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

// Non-baked handler for native gather (embedding lookup)
static id<MTLBuffer> NativeIntrinsicGatherRowsHandler(id<MTLDevice> device,
                                                      id<MTLCommandBuffer> commandBuffer, mlir::Operation* op,
                                                      const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !IsEligibleGatherRowsNativeOp(op) || inputs.size() < 2 || !inputs[0] ||
        !inputs[1]) {
        return nil;
    }
    std::vector<int64_t> operandShape, indexShape;
    MPSDataType gatherDtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &operandShape, nullptr, &gatherDtype) ||
        !GetStaticTensorInfo(op->getOperand(1).getType(), &indexShape, nullptr, nullptr) ||
        operandShape.size() != 2 || indexShape.size() != 2) {
        return nil;
    }
    const uint32_t num_indices = static_cast<uint32_t>(indexShape[0]);
    const uint32_t cols = static_cast<uint32_t>(operandShape[1]);
    const uint32_t table_rows = static_cast<uint32_t>(operandShape[0]);
    const uint64_t total64 = static_cast<uint64_t>(num_indices) * static_cast<uint64_t>(cols);
    if (total64 == 0 || total64 > std::numeric_limits<uint32_t>::max())
        return nil;

    size_t outBytes = ByteSizeFromType(op->getResult(0).getType());
    if (outBytes == 0)
        return nil;
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out)
        return nil;

    const bool f16 = (gatherDtype == MPSDataTypeFloat16);
    std::string kernelName = f16 ? "gather_rows_f16" : "gather_rows_f32";
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        CFRelease((__bridge CFTypeRef)out);
        return nil;
    }

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0]; // operand table
    [encoder setBuffer:inputs[1] offset:0 atIndex:1]; // indices
    [encoder setBuffer:out offset:0 atIndex:2];       // output
    [encoder setBytes:&num_indices length:sizeof(num_indices) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    [encoder setBytes:&table_rows length:sizeof(table_rows) atIndex:5];
    EncodeLinearDispatch(encoder, pipeline, static_cast<uint32_t>(total64), gatherDtype);
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

// --- Native sort classification ---

static bool IsEligibleSortNativeOp(mlir::Operation* op) {
    auto sortOp = mlir::dyn_cast_or_null<mlir::stablehlo::SortOp>(op);
    if (!sortOp) {
        return false;
    }
    // Only single-operand, single-result sort
    if (op->getNumOperands() != 1 || op->getNumResults() != 1) {
        return false;
    }
    std::vector<int64_t> inShape;
    MPSDataType dtype = MPSDataTypeInvalid;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, &dtype)) {
        return false;
    }
    // Only f32
    if (dtype != MPSDataTypeFloat32) {
        return false;
    }
    // 1D or 2D with sort along last axis
    if (inShape.size() == 1) {
        uint64_t n = static_cast<uint64_t>(inShape[0]);
        if (n == 0 || n > kMaxNativeSortElements) {
            return false;
        }
    } else if (inShape.size() == 2) {
        // 2D: sort must be along last axis (dimension == rank-1 == 1)
        int64_t sortDim = sortOp.getDimension();
        if (sortDim != 1) {
            return false;
        }
        uint64_t cols = static_cast<uint64_t>(inShape[1]);
        if (cols == 0 || cols > kMaxNativeSortElements || inShape[0] <= 0) {
            return false;
        }
    } else {
        return false;
    }
    // Only ascending sort
    bool descending = false;
    if (!sortOp.getComparator().empty()) {
        for (mlir::Operation& nestedOp : sortOp.getComparator().front()) {
            auto compareOp = mlir::dyn_cast<mlir::stablehlo::CompareOp>(&nestedOp);
            if (!compareOp) {
                continue;
            }
            auto direction = compareOp.getComparisonDirection();
            if (direction == mlir::stablehlo::ComparisonDirection::GT ||
                direction == mlir::stablehlo::ComparisonDirection::GE) {
                descending = true;
            }
            break;
        }
    }
    if (descending) {
        return false;
    }
    // Small sorts used to stay on MPSGraph, but the exact graph path is argsort
    // + gather: [4096,16] 4.95 ms vs 0.22 ms native, [1024,64] 1.64 vs 0.35,
    // 1-D 4096 0.29 vs 0.24. Every eligible sort now runs natively;
    // JAX_METALLIB_SORT_NATIVE_MIN_ELEMS=100000 restores the old size gate
    // (EnvIntValue ignores values <= 0).
    uint64_t totalElems = 1;
    for (int64_t d : inShape) {
        totalElems *= static_cast<uint64_t>(d);
    }
    uint64_t cols =
        (inShape.size() == 2) ? static_cast<uint64_t>(inShape[1]) : static_cast<uint64_t>(inShape[0]);
    uint64_t rows = (inShape.size() == 2) ? static_cast<uint64_t>(inShape[0]) : 1;
    static const int64_t minElems = EnvIntValue("JAX_METALLIB_SORT_NATIVE_MIN_ELEMS", 1);
    if (static_cast<int64_t>(totalElems) < minElems && !(rows > 1 && cols > 256)) {
        return false;
    }
    return true;
}

static id<MTLBuffer> NativeIntrinsicTransposeHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                     mlir::Operation* op,
                                                     const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op || inputs.empty() || !inputs[0]) {
        return nil;
    }
    uint32_t rows = 0, cols = 0, batch = 1;
    if (!IsEligibleTransposeNativeOp(op, &rows, &cols, &batch)) {
        return nil;
    }
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, "transpose2d_f32");
    if (!pipeline) {
        return nil;
    }
    size_t outBytes = (size_t)batch * rows * cols * sizeof(float);
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }
    NSUInteger tg_mem = 32 * 33 * sizeof(float); // 32×33 tile (padded)
    uint32_t grid_x = (cols + 31) / 32;
    uint32_t grid_y = (rows + 31) / 32;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&rows length:sizeof(rows) atIndex:2];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:3];
    [encoder setThreadgroupMemoryLength:tg_mem atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, batch)
            threadsPerThreadgroup:MTLSizeMake(32, 8, 1)];
    [encoder endEncoding];
    return out;
}

static id<MTLBuffer> NativeIntrinsicSortHandler(id<MTLDevice> device, id<MTLCommandBuffer> commandBuffer,
                                                mlir::Operation* op,
                                                const std::vector<id<MTLBuffer>>& inputs) {
    if (!device || !commandBuffer || !op || inputs.empty() || !inputs[0]) {
        return nil;
    }
    if (!IsEligibleSortNativeOp(op)) {
        return nil;
    }
    std::vector<int64_t> inShape;
    if (!GetStaticTensorInfo(op->getOperand(0).getType(), &inShape, nullptr, nullptr) || inShape.empty()) {
        return nil;
    }
    uint32_t num_rows = 1;
    uint32_t n;
    if (inShape.size() == 1) {
        n = static_cast<uint32_t>(inShape[0]);
    } else {
        num_rows = static_cast<uint32_t>(inShape[0]);
        n = static_cast<uint32_t>(inShape[1]);
    }
    uint32_t padded_n = 1;
    while (padded_n < n)
        padded_n <<= 1;

    const char* kernelName = (num_rows > 1) ? "bitonic_sort_rows_asc_f32" : "bitonic_sort_asc_f32";
    id<MTLComputePipelineState> pipeline = GetFastNativePipeline(device, kernelName);
    if (!pipeline) {
        return nil;
    }
    size_t totalElements = (size_t)num_rows * (size_t)n;
    size_t outBytes = totalElements * sizeof(float);
    id<MTLBuffer> out = AcquireReusableOutputBuffer(device, outBytes);
    if (!out) {
        return nil;
    }
    NSUInteger tg_mem = SortThreadgroupBytes(padded_n);
    NSUInteger tg_size = SortThreadgroupSize(padded_n, pipeline);
    if (tg_size == 0)
        tg_size = 1;

    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inputs[0] offset:0 atIndex:0];
    [encoder setBuffer:out offset:0 atIndex:1];
    [encoder setBytes:&n length:sizeof(n) atIndex:2];
    [encoder setBytes:&padded_n length:sizeof(padded_n) atIndex:3];
    if (num_rows > 1) {
        [encoder setBytes:&n length:sizeof(n) atIndex:4]; // row_stride = n for contiguous 2D
    }
    [encoder setThreadgroupMemoryLength:tg_mem atIndex:0];
    [encoder dispatchThreadgroups:MTLSizeMake(num_rows, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(tg_size, 1, 1)];
    [encoder endEncoding];
    return out;
}

static NativeOpHandler TrySelectIntrinsicNativeHandler(mlir::Operation* op) {
    if (!op) {
        return nullptr;
    }
    // Large last-axis sum/mean reductions route natively EVEN when the
    // intrinsic-kernel family is gated off: the TG reduce kernel is 13-100x
    // faster than the MPSGraph reduce path at these sizes (2822us -> 372us at
    // 65536; 84ms -> ~300us at 1M), and this route does not use the fused-chain
    // elimination machinery where the intrinsic-family defects live.
    if (!IntrinsicNativeKernelsEnabled()) {
        auto reduceOp = mlir::dyn_cast_or_null<mlir::stablehlo::ReduceOp>(op);
        if (reduceOp && reduceOp->getNumOperands() == 2 && reduceOp->getNumResults() == 1) {
            const std::string bodyName = GetReduceBodyOpName(reduceOp);
            auto inType = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
            auto dims = reduceOp.getDimensions();
            if (inType && inType.hasStaticShape() &&
                IsSuffixReduceDimensions(dims, static_cast<int64_t>(inType.getRank()))) {
                IntrinsicNativeKind bypassKind = IntrinsicNativeKind::kNone;
                if (bodyName == "stablehlo.add") {
                    bypassKind = IntrinsicNativeKind::kReduceSumLastAxis;
                } else if (static_cast<int64_t>(dims.size()) == static_cast<int64_t>(inType.getRank()) &&
                           bodyName == "stablehlo.maximum") {
                    bypassKind = IntrinsicNativeKind::kReduceMaxLastAxis;
                } else if (static_cast<int64_t>(dims.size()) == static_cast<int64_t>(inType.getRank()) &&
                           bodyName == "stablehlo.minimum") {
                    bypassKind = IntrinsicNativeKind::kReduceMinLastAxis;
                }
                // Multi-row reduces with short rows (LayerNorm, in-layer softmax
                // sums) stay in MPSGraph: routing them natively split the plan into
                // graph/native/graph segments, and that switch cost dominated
                // (LayerNorm [16,256,512] 1.19 ms native vs 0.35 ms in graph).
                // Rows >= 1024 measured equal or faster natively; single-row
                // (full) reduces are unaffected.
                // JAX_METALLIB_REDUCE_BYPASS_MIN_ROW=0 restores the old routing.
                if (bypassKind != IntrinsicNativeKind::kNone) {
                    int64_t rowLen = 1;
                    for (auto d : dims) {
                        rowLen *= inType.getDimSize(d);
                    }
                    const int64_t numel = inType.getNumElements();
                    const bool multiRow = rowLen > 0 && numel / rowLen > 1;
                    const int64_t minRow = EnvIntValue("JAX_METALLIB_REDUCE_BYPASS_MIN_ROW", 1024);
                    if (multiRow && rowLen < minRow) {
                        bypassKind = IntrinsicNativeKind::kNone;
                    }
                }
                if (bypassKind != IntrinsicNativeKind::kNone && IsEligibleReduceNativeOp(op, bypassKind)) {
                    return NativeIntrinsicReduceHandler;
                }
            }
        }
        // The f32 bitonic row sort is likewise a standalone kernel outside the
        // fused-chain machinery; MPSGraph sort was ~17x slower at [4096,256]
        // (7.7 ms vs 0.46 ms). The kernel uses XLA's NaN-last total order.
        if (IsEligibleSortNativeOp(op)) {
            return NativeIntrinsicSortHandler;
        }
        if (op->hasAttr("__softmax_fused") &&
            IsEligibleReduceNativeOp(op, IntrinsicNativeKind::kSoftmaxLastAxis)) {
            return NativeIntrinsicReduceHandler;
        }
        return nullptr;
    }
    if (!ShouldRouteToIntrinsicNative(op)) {
        return nullptr;
    }
    IntrinsicNativeKind simpleKind = ClassifySimpleKind(op);
    if (simpleKind == IntrinsicNativeKind::kReshapeAlias && IsEligibleReshapeAliasOp(op)) {
        return NativeIntrinsicReshapeAliasHandler;
    }
    if (simpleKind == IntrinsicNativeKind::kTransposeAlias && IsEligibleTransposeAliasOp(op)) {
        return NativeIntrinsicReshapeAliasHandler; // Same handler — alias the input buffer.
    }
    if (simpleKind == IntrinsicNativeKind::kTransposeAlias && IsEligibleTransposeNativeOp(op)) {
        return NativeIntrinsicTransposeHandler; // Fast native transpose for internal transposes.
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
    if (simpleKind == IntrinsicNativeKind::kGatherRows && IsEligibleGatherRowsNativeOp(op)) {
        return NativeIntrinsicGatherRowsHandler;
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
    if (ClassifyDotGeneralNN(op, nullptr, nullptr, nullptr, nullptr)) {
        return NativeIntrinsicGEMMHandler;
    }
    if (ClassifyDotGeneralSmall(op, nullptr, nullptr, nullptr, nullptr)) {
        return NativeIntrinsicGEMMHandler; // Same handler, variant selection differs at bake time
    }
    if (IsEligibleSortNativeOp(op)) {
        return NativeIntrinsicSortHandler;
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

static bool OutputBufferPoolEnabled() {
    // Default-on for throughput/latency: repeated executions of large kernels (benchmarks,
    // training loops) avoid costly MTLBuffer allocations. Users can disable for debugging.
    static const bool enabled = !EnvFlagEnabled("JAX_METALLIB_DISABLE_OUTPUT_BUFFER_POOL");
    return enabled;
}

// Thread-local single-element cache for output buffer reuse.
// Avoids the global pool mutex on the hot path for repeated dispatches
// of the same shape (the common case in JAX benchmarks and training loops).
struct TLOutputBufSlot {
    static constexpr int kRingSize = 4;
    struct Entry {
        void* device_key = nullptr;
        size_t bucket = 0;
        void* metal_buffer = nullptr; // CFRetained when non-null
    };
    Entry entries[kRingSize];
    int next = 0;
    ~TLOutputBufSlot() {
        for (auto& e : entries) {
            if (e.metal_buffer) {
                CFRelease(e.metal_buffer);
            }
        }
    }
};
static TLOutputBufSlot& GetTLOutputSlot() {
    static thread_local TLOutputBufSlot slot;
    return slot;
}

static id<MTLBuffer> AcquireReusableOutputBuffer(id<MTLDevice> device, size_t byte_size) {
    if (!device || byte_size == 0) {
        return nil;
    }
    if (!OutputBufferPoolEnabled()) {
        return [device newBufferWithLength:RoundUpToPow2(byte_size) options:ComputeBufferResourceOptions()];
    }

    const size_t bucket = RoundUpToPow2(byte_size);
    void* dev_key = (__bridge void*)device;

    // Thread-local fast path: scan the ring first (no mutex).
    auto& tls = GetTLOutputSlot();
    for (auto& e : tls.entries) {
        if (e.metal_buffer && e.device_key == dev_key && e.bucket == bucket) {
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)e.metal_buffer;
            e.metal_buffer = nullptr;
            if ([buf length] >= byte_size) {
                return buf;
            }
            CFRelease((__bridge CFTypeRef)buf);
            break;
        }
    }

    ReusableOutputPool& pool = GetReusableOutputPool();
    ReusableOutputKey key{dev_key, bucket};
    {
        std::scoped_lock lock(pool.mu);
        auto it = pool.bins.find(key);
        if (it != pool.bins.end() && !it->second.empty()) {
            id<MTLBuffer> buf = it->second.back();
            it->second.pop_back();
            if ([buf length] >= byte_size) {
                return buf;
            }
            CFRelease((__bridge CFTypeRef)buf);
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
    if (!OutputBufferPoolEnabled()) {
        CFRelease((__bridge CFTypeRef)metal_buffer);
        return;
    }

    id<MTLBuffer> buf = (__bridge id<MTLBuffer>)metal_buffer;
    const size_t actual_len = (size_t)[buf length];
    const size_t bucket = RoundUpToPow2(actual_len > 0 ? actual_len : byte_size);

    // Thread-local fast path: store in the ring (evict oldest to the bin).
    auto& tls = GetTLOutputSlot();
    auto& slot = tls.entries[tls.next];
    tls.next = (tls.next + 1) % TLOutputBufSlot::kRingSize;
    void* evicted = slot.metal_buffer;
    void* evicted_dk = slot.device_key;
    size_t evicted_bk = slot.bucket;
    slot.device_key = device_key;
    slot.bucket = bucket;
    slot.metal_buffer = metal_buffer;
    if (!evicted) {
        return;
    }

    // Recycle evicted entry to global pool.
    ReusableOutputPool& pool = GetReusableOutputPool();
    ReusableOutputKey key{evicted_dk, evicted_bk};
    std::scoped_lock lock(pool.mu);
    auto& bin = pool.bins[key];
    if (bin.size() >= 16) {
        CFRelease(evicted);
        return;
    }
    bin.push_back((__bridge id<MTLBuffer>)evicted);
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

// Mixed graph+native execution encodes MPSGraph work into an MPSCommandBuffer,
// which MPSGraph may commit-and-continue for large graphs. A completed handler
// added when the buffer was created then belongs to the first underlying Metal
// command buffer and fired before later native steps ran (a 4096x4095 graph
// chain feeding the native last-axis reduce returned zero rows). The handler
// is attached right before the final commit instead.
// JAX_METALLIB_DISABLE_DEFERRED_COMPLETION=1 restores the old behaviour.
static bool DeferMixedCompletion() {
    static const bool defer = !EnvFlagEnabled("JAX_METALLIB_DISABLE_DEFERRED_COMPLETION");
    return defer;
}

static CompletionEvent MixedCompletionEvent(id<MTLCommandBuffer> cmd_buf, bool async_completion) {
    return DeferMixedCompletion() ? MakePendingCompletionEvent()
                                  : CompletionEventForCommandBuffer(cmd_buf, async_completion);
}

static void AttachDeferredCompletion(id<MTLCommandBuffer> cmd_buf, const CompletionEvent& completion,
                                     bool async_completion) {
    if (!DeferMixedCompletion() || !async_completion || !cmd_buf || !completion) {
        return;
    }
    CompletionEvent captured = completion;
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> completed) {
      MarkCompletionFromCommandBuffer(captured, completed);
    }];
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
    calleeCtx.in_control_flow = ctx.in_control_flow;
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

// stablehlo.composite references an out-of-line decomposition function by name
// (FlatSymbolRefAttr), structurally identical to func::CallOp. jax 0.10 lowers
// many higher-level operations (jnp.arcsin, jnp.argsort, jnp.sum, ...) as
// composites instead of as direct stablehlo primitives. We just look up the
// referenced function and walk its body — the same flow processCallOp uses.
static ProcessResult processCompositeOp(HandlerContext& ctx, mlir::stablehlo::CompositeOp compositeOp) {
    if (ctx.depth > 100) {
        return ProcessResult::Error("Maximum composite depth exceeded - possible recursive composite");
    }

    auto callee = ctx.module.lookupSymbol<mlir::func::FuncOp>(compositeOp.getDecomposition());
    if (!callee) {
        return ProcessResult::Error("Could not find composite decomposition function: " +
                                    compositeOp.getDecomposition().str());
    }

    if (callee.empty()) {
        return ProcessResult::Error("Composite decomposition function has no body: " +
                                    compositeOp.getName().str());
    }

    auto& calleeBlock = callee.front();
    for (size_t i = 0; i < compositeOp.getNumOperands() && i < calleeBlock.getNumArguments(); i++) {
        mlir::Value callArg = compositeOp.getOperand(i);
        mlir::BlockArgument funcArg = calleeBlock.getArgument(i);

        MPSGraphTensor* argTensor = GetTensor(ctx.values, callArg);
        if (!argTensor) {
            return ProcessResult::Error("Composite '" + compositeOp.getName().str() + "' argument " +
                                        std::to_string(i) + " not found");
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
    calleeCtx.in_control_flow = ctx.in_control_flow;
    ProcessResult calleeResult = processOperations(calleeCtx, calleeBlock);
    if (!calleeResult.ok()) {
        return calleeResult;
    }

    for (size_t i = 0; i < compositeOp.getNumResults() && i < calleeResult.return_values.size(); i++) {
        mlir::Value returnValue = calleeResult.return_values[i];
        MPSGraphTensor* returnTensor = GetTensor(ctx.values, returnValue);
        if (!returnTensor) {
            return ProcessResult::Error("Composite '" + compositeOp.getName().str() + "' return value " +
                                        std::to_string(i) + " not found");
        }
        ctx.values[compositeOp.getResult(i).getAsOpaquePointer()] = returnTensor;

        if (ctx.variables) {
            auto it = ctx.variables->find(returnValue.getAsOpaquePointer());
            if (it != ctx.variables->end()) {
                (*ctx.variables)[compositeOp.getResult(i).getAsOpaquePointer()] = it->second;
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

        if (auto compositeOp = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
            ProcessResult compositeResult = processCompositeOp(ctx, compositeOp);
            if (!compositeResult.ok()) {
                return compositeResult;
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

        auto graphFn = handler->graph_handler;
        if (handler->is_native()) {
            // A native-only op inside a graph region (a case/while body) cannot
            // run its native kernel; use the op's graph fallback when it has one.
            if (!handler->graph_fallback) {
                return ProcessResult::Error("Native op '" + op_name +
                                            "' encountered during graph building - "
                                            "this should have been handled by segmented execution");
            }
            graphFn = handler->graph_fallback;
        }

        HandlerContext opCtx(ctx.graph, op, ctx.values, ctx.module, ctx.depth, processOperationsAdapter);
        opCtx.variables = ctx.variables;
        opCtx.in_control_flow = ctx.in_control_flow;
        ProcessResult opResult = graphFn(opCtx);
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

        {
            mlir::Block& preBlock = entry_func_.front();
            FuseLayerNormPatterns(preBlock);
            FuseSoftmaxPatterns(preBlock);
            FuseReduceMeanPatterns(preBlock);
        }
        MPS_LOG_DEBUG("BuildExecutionPlan: layernorm/softmax/reduce_mean fusion pass complete\n");

        std::unique_ptr<ExecutionPlan> plan;

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

        auto build_segments = [&](bool allow_intrinsic_native,
                                  const std::unordered_set<mlir::Operation*>* force_graph_ops =
                                      nullptr) -> bool {
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
                    // Demote: if this op is in the force_graph set, don't use native
                    if (native_handler && force_graph_ops && force_graph_ops->count(op)) {
                        native_handler = nullptr;
                    }
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

        // Elementwise autotuning: after fusion, standalone bandwidth-bound elementwise ops
        // are benchmarked against MPSGraph. If MPSGraph wins, the op is demoted to graph and
        // the plan is rebuilt. Runs at most twice: once optimistically, and if any demotions
        // are needed, once more with the demotion set.
        static const bool enable_elem_autotune = !EnvFlagEnabled("JAX_METALLIB_DISABLE_ELEMWISE_AUTOTUNE");
        std::unordered_set<mlir::Operation*> demoted_elem_ops;

        for (int build_pass = 0; build_pass < 2; ++build_pass) {
            plan = std::make_unique<ExecutionPlan>();

            if (!build_segments(true, demoted_elem_ops.empty() ? nullptr : &demoted_elem_ops)) {
                return false;
            }

            const int mixed_intrinsic_segment_threshold =
                EnvIntValue("JAX_METALLIB_MIXED_INTRINSIC_SEGMENT_THRESHOLD", 1);
            // Mixed graph+native plans are the default: large last-axis
            // reductions run natively (8-190x faster than the MPSGraph reduce
            // path). Opt out with JAX_METALLIB_DISABLE_MIXED_INTRINSIC_NATIVE=1
            // if a workload exposes a fused-chain defect.
            const bool keep_mixed_intrinsics =
                !EnvFlagEnabled("JAX_METALLIB_DISABLE_MIXED_INTRINSIC_NATIVE");
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
                    if (seg.ops[0]->hasAttr("__softmax_fused") || seg.ops[0]->hasAttr("__mean_fused") ||
                        seg.ops[0]->hasAttr("__softmax_numerator_fused")) {
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
                MPS_LOG_DEBUG("BuildExecutionPlan: segment %d, %zu ops\n", seg_idx,
                              segments[seg_idx].ops.size());
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
            MPS_LOG_DEBUG("BuildExecutionPlan: segment boundary crossings done, %zu slots\n",
                          plan->slots.size());

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
                                HandlerContext callCtx(graph, op, values, *module_, 0,
                                                       processOperationsAdapter);
                                ProcessResult callResult = processCallOp(callCtx, callOp);
                                if (!callResult.ok()) {
                                    error_ = callResult.error;
                                    return false;
                                }
                                continue;
                            }

                            if (auto compositeOp = mlir::dyn_cast<mlir::stablehlo::CompositeOp>(op)) {
                                HandlerContext compositeCtx(graph, op, values, *module_, 0,
                                                            processOperationsAdapter);
                                ProcessResult compositeResult = processCompositeOp(compositeCtx, compositeOp);
                                if (!compositeResult.ok()) {
                                    error_ = compositeResult.error;
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
                                                  [[tensor.shape description] UTF8String],
                                                  (int)tensor.dataType);
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
                                MPSGraphShapedType* st =
                                    [[MPSGraphShapedType alloc] initWithShape:t.shape dataType:t.dataType];
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
                                                          compilationDescriptor:GraphCompilationDescriptor()];
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

                auto dtype_byte_size = [](MPSDataType d) -> size_t {
                    return (d == MPSDataTypeFloat16) ? 2 : 4;
                };

                struct ChainRange {
                    size_t steps_begin;
                    size_t steps_end;
                };
                std::vector<ChainRange> chains;

                size_t chain_start = 0;
                // Slot -> positions of steps that read it (native inputs and
                // graph feeds). Used to keep fused chains from eliminating an
                // intermediate that a later step still consumes.
                std::unordered_map<SlotId, std::vector<size_t>> slot_consumers;
                for (size_t si = 0; si < plan->steps.size(); ++si) {
                    const auto& step = plan->steps[si];
                    if (step.kind == Step::NATIVE) {
                        for (SlotId s : plan->native_steps[step.index].input_slots) {
                            slot_consumers[s].push_back(si);
                        }
                    } else if (step.kind == Step::GRAPH) {
                        for (auto& feed : plan->graph_steps[step.index].feeds) {
                            slot_consumers[feed.first].push_back(si);
                        }
                        for (SlotId s : plan->graph_steps[step.index].exec_feed_slots) {
                            slot_consumers[s].push_back(si);
                        }
                    }
                }

                bool in_chain = false;
                MPSDataType chain_dtype = MPSDataTypeInvalid;
                uint64_t chain_numel = 0;

                std::unordered_set<int> chain_ns_indices;

                auto flush_chain = [&](size_t chain_end) {
                    if (in_chain && chain_end - chain_start >= 2) {
                        // Split so that no interior output is consumed outside
                        // its chain: such a slot would be marked eliminated and
                        // never written, while the external reader (e.g. a second
                        // fused chain after an intervening graph step) would read
                        // zeros.
                        size_t run_start = chain_start;
                        for (size_t p = run_start; p < chain_end; ++p) {
                            const auto& st = plan->steps[p];
                            SlotId out_slot = (st.kind == Step::NATIVE)
                                                  ? plan->native_steps[st.index].output_slot
                                                  : -1;
                            bool externally_consumed = false;
                            if (out_slot >= 0) {
                                auto it = slot_consumers.find(out_slot);
                                if (it != slot_consumers.end()) {
                                    for (size_t cpos : it->second) {
                                        if (cpos < run_start || cpos >= chain_end) {
                                            externally_consumed = true;
                                            break;
                                        }
                                    }
                                }
                            }
                            if (externally_consumed) {
                                // The op at `p` has a reader outside this run, so
                                // it can never be an interior member. Emit the
                                // prefix if it is a real chain, and always skip
                                // past `p`: absorbing it (e.g. as the first member)
                                // would leave its output eliminated while the
                                // external reader still needs the buffer.
                                if (p + 1 - run_start >= 2) {
                                    chains.push_back({run_start, p + 1});
                                }
                                run_start = p + 1;
                            }
                        }
                        if (chain_end - run_start >= 2) {
                            chains.push_back({run_start, chain_end});
                        }
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
                    uint64_t ns_numel =
                        (dsize > 0) ? plan->slots[(size_t)ns.output_slot].byte_size / dsize : 0;

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
                        bool stride_disabled = EnvFlagEnabled("JAX_METALLIB_DISABLE_STRIDE_KERNELS");
                        static const uint64_t stride_min_numel_fuse =
                            static_cast<uint64_t>(EnvIntValue("JAX_METALLIB_STRIDE_MIN_NUMEL", 1048576));
                        const uint64_t fusion_vec8_min = (chain_dtype == MPSDataTypeFloat32) ? 262144u : 8u;
                        int vec_mode = 0;
                        if (!vec_disabled) {
                            if (!stride_disabled && IsFloat16Or32(chain_dtype) && chain_numel % 4 == 0 &&
                                chain_numel >= stride_min_numel_fuse) {
                                vec_mode = 16; // stride
                            } else if (IsFloat16Or32(chain_dtype) && chain_numel % 8 == 0 &&
                                       chain_numel >= fusion_vec8_min) {
                                vec_mode = 8;
                            } else if (IsFloat16Or32(chain_dtype) && chain_numel % 4 == 0) {
                                vec_mode = 4;
                            }
                        }

                        std::string cache_key = "fused_";
                        cache_key += (chain_dtype == MPSDataTypeFloat16) ? "f16" : "f32";
                        if (vec_mode == 16)
                            cache_key += "_stride";
                        else if (vec_mode == 8)
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
                                      ci, fusion_ops.size(), external_input_slots.size(),
                                      eliminated_slots.size(), vec_mode, cache_key.c_str());

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
                        if (vec_mode == 16)
                            n = static_cast<uint32_t>(chain_numel / 4);
                        else if (vec_mode == 8)
                            n = static_cast<uint32_t>(chain_numel / 8);
                        else if (vec_mode == 4)
                            n = static_cast<uint32_t>(chain_numel / 4);
                        else
                            n = static_cast<uint32_t>(chain_numel);
                        fns.baked_n = n;
                        fns.baked_use_vec4 = (vec_mode >= 4);
                        fns.baked_use_stride = (vec_mode == 16);
                        if (vec_mode == 16) {
                            uint32_t grid_threads = ComputeStrideGridSize(n);
                            NSUInteger tg = ResolveLinearThreadgroupSize(pipeline, chain_dtype);
                            NSUInteger tg_count = ((NSUInteger)grid_threads + tg - 1) / tg;
                            fns.baked_grid = MTLSizeMake(tg_count, 1, 1);
                            fns.baked_tg = MTLSizeMake(tg, 1, 1);
                        } else {
                            ComputeLinearDispatchConfig(pipeline, n, &fns.baked_grid, &fns.baked_tg,
                                                        chain_dtype);
                        }
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
                                fused_steps[(size_t)ci].baked_kind ==
                                    NativeStep::BakedKind::kFusedElementwise) {
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

            // ── Pass 3b: elementwise autotune demotion ──
            // Scan for standalone (non-fused) bandwidth-bound elementwise native steps.
            // Autotune each against MPSGraph; if MPSGraph is faster, demote to graph and rebuild.
            if (enable_elem_autotune && demoted_elem_ops.empty()) {
                std::unordered_set<mlir::Operation*> new_demotions;
                for (size_t si = 0; si < plan->steps.size(); ++si) {
                    const auto& step = plan->steps[si];
                    if (step.kind != Step::NATIVE)
                        continue;
                    const auto& ns = plan->native_steps[step.index];
                    // Only standalone kElementwise (not fused, not other kinds)
                    if (ns.baked_kind != NativeStep::BakedKind::kElementwise)
                        continue;

                    IntrinsicNativeKind kind = ClassifyElementwiseKind(ns.op);
                    if (kind == IntrinsicNativeKind::kNone)
                        continue;

                    // Compute-bound ops always win native — skip autotuning
                    if (IsComputeBoundElementwise(kind))
                        continue;

                    // Get dtype and numel from the output slot
                    MPSDataType dtype = plan->slots[(size_t)ns.output_slot].dtype;
                    size_t dsize = (dtype == MPSDataTypeFloat16) ? 2 : 4;
                    uint64_t numel = (dsize > 0) ? plan->slots[(size_t)ns.output_slot].byte_size / dsize : 0;

                    if (numel == 0)
                        continue;

                    // Autotune: returns true if native wins, false if MPSGraph wins
                    bool native_wins = AutotuneElementwise(mtl_device, kind, dtype, numel);
                    if (!native_wins) {
                        new_demotions.insert(ns.op);
                        MPS_LOG_WARN("elemwise-autotune: demoting op to MPSGraph (numel=%llu dtype=%s)\n",
                                     (unsigned long long)numel,
                                     (dtype == MPSDataTypeFloat16) ? "f16" : "f32");
                    }
                }
                if (!new_demotions.empty()) {
                    MPS_LOG_WARN(
                        "elemwise-autotune: demoting %zu standalone ops to MPSGraph, rebuilding plan\n",
                        new_demotions.size());
                    demoted_elem_ops = std::move(new_demotions);
                    continue; // rebuild plan with demotions
                }
            }

            break; // no demotions needed, exit build loop
        } // end for (build_pass)

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

        // Build a map from output slot → native step index for transpose alias detection.
        std::unordered_map<SlotId, size_t> slot_to_native_step;
        for (size_t si = 0; si < plan->native_steps.size(); ++si) {
            const auto& ns = plan->native_steps[si];
            if (ns.baked_kind == NativeStep::BakedKind::kTransposeAlias) {
                slot_to_native_step[ns.output_slot] = si;
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

            // If this output was produced by a transpose alias, record the
            // permutation and input shape so we can build strided output buffers.
            auto tn_it = slot_to_native_step.find(slot);
            if (tn_it != slot_to_native_step.end()) {
                const auto& ns = plan->native_steps[tn_it->second];
                auto transposeOp = mlir::dyn_cast_or_null<mlir::stablehlo::TransposeOp>(ns.op);
                if (transposeOp) {
                    auto perm = transposeOp.getPermutation();
                    meta.transpose_perm.assign(perm.begin(), perm.end());
                    auto inType = mlir::cast<mlir::RankedTensorType>(ns.op->getOperand(0).getType());
                    auto inShape = inType.getShape();
                    meta.transpose_input_dims.assign(inShape.begin(), inShape.end());
                }
            }

            plan->output_meta.push_back(std::move(meta));
        }

        plan_ = std::move(plan);
        plan_built_ = true;
    }
    MPS_LOG_DEBUG("BuildExecutionPlan: returning true\n");
    return true;
}

// Global queue for delayed IntermediateCacheSet releases. A background
// thread waits for each cache's completion event to fire (via Await()) and
// then destroys the shared_ptr, which calls IntermediateCacheSet::ReleaseAll
// to release the owned MTLBuffers.
//
// This pattern guarantees the GPU is idle when the buffers are released, so
// the next Execute() can safely allocate new buffers from a fresh cache set
// without colliding with in-flight GPU work.
struct PendingCacheRelease {
    std::shared_ptr<IntermediateCacheSet> cache;
    CompletionEvent fence;
};

namespace {
std::mutex& GetCacheReleaseQueueMutex() {
    static std::mutex m;
    return m;
}
std::vector<PendingCacheRelease>& GetCacheReleaseQueue() {
    static std::vector<PendingCacheRelease> q;
    return q;
}
std::condition_variable& GetCacheReleaseCV() {
    static std::condition_variable cv;
    return cv;
}
std::atomic<bool>& GetCacheReleaseShutdown() {
    static std::atomic<bool> s{false};
    return s;
}

void CacheReleaseWorker() {
    std::unique_lock<std::mutex> lock(GetCacheReleaseQueueMutex());
    while (!GetCacheReleaseShutdown().load()) {
        GetCacheReleaseCV().wait(
            lock, [] { return GetCacheReleaseShutdown().load() || !GetCacheReleaseQueue().empty(); });
        if (GetCacheReleaseShutdown().load() && GetCacheReleaseQueue().empty()) {
            return;
        }
        // Drain the entire queue. For each entry, we wait for its completion
        // event WITHOUT holding the queue mutex so other threads can enqueue
        // while we wait. CompletionEvent::Await() is the safe way to wait for
        // GPU work to finish.
        std::vector<PendingCacheRelease> batch;
        batch.swap(GetCacheReleaseQueue());
        lock.unlock();
        for (auto& item : batch) {
            if (item.fence) {
                item.fence->Await();
            }
            // shared_ptr destructor in `item.cache` runs here, releasing the
            // MTLBuffers via IntermediateCacheSet::ReleaseAll.
        }
        lock.lock();
    }
}

// Start the worker thread lazily on first enqueue. Idempotent.
void EnsureCacheReleaseWorker() {
    static std::once_flag once;
    std::call_once(once, [] {
        std::thread t(CacheReleaseWorker);
        t.detach();
    });
}
} // namespace

void EnqueueCacheForDelayedRelease(std::shared_ptr<IntermediateCacheSet> cache, CompletionEvent fence) {
    if (!cache) {
        return;
    }
    EnsureCacheReleaseWorker();
    {
        std::scoped_lock lock(GetCacheReleaseQueueMutex());
        GetCacheReleaseQueue().push_back({std::move(cache), std::move(fence)});
    }
    GetCacheReleaseCV().notify_one();
}

// Submit a no-op command buffer to the given Metal command queue and
// return a CompletionEvent that fires when it completes. Because the
// Metal command queue is serial, this serves as a synchronization
// barrier that fires only after all prior work submitted to the queue
// has finished. The queue parameter is void* to avoid pulling the
// Objective-C Metal headers into a C++ header.
static CompletionEvent SubmitSyncBarrier(void* mtlCommandQueue) {
    // Create an empty command buffer and chain a completion handler on it.
    // Because Metal serializes command buffer execution on a single queue,
    // this handler fires only after every prior command buffer on the queue
    // (including the one created by [MPSGraphExecutable runWithMTLCommandQueue:])
    // has finished executing. This gives us a real GPU-completion fence without
    // needing to expose the underlying MPSCommandBuffer.
    id<MTLCommandQueue> queue = (__bridge id<MTLCommandQueue>)mtlCommandQueue;
    if (!queue) {
        return MakeReadyCompletionEvent();
    }
    id<MTLCommandBuffer> barrier = [queue commandBuffer];
    if (!barrier) {
        return MakeReadyCompletionEvent();
    }
    // Use an empty blit encoder pass to ensure the command buffer actually has
    // work to do (otherwise Metal may skip submitting it).
    id<MTLBlitCommandEncoder> blit = [barrier blitCommandEncoder];
    if (blit) {
        [blit endEncoding];
    }
    CompletionEvent completion = MakePendingCompletionEvent();
    CompletionEvent captured = completion;
    [barrier addCompletedHandler:^(id<MTLCommandBuffer> /*completed*/) {
      MarkCompletionFromCommandBuffer(captured, barrier);
    }];
    [barrier commit];
    return completion;
}

// Batched small-matrix Cholesky via LAPACK spotrf_ on the host. `lower`
// selects the kept triangle; the non-factored triangle is zeroed and any
// non-PD batch is filled with NaN (matching the GPU kernel's behavior).
static void RunCpuCholeskyBatch(float* dst, const float* src, uint32_t n, uint32_t batch,
                                bool lower) {
    const size_t mat_floats = static_cast<size_t>(n) * n;
    memcpy(dst, src, mat_floats * batch * sizeof(float));
    const char uplo = lower ? 'U' : 'L'; // LAPACK is column-major, JAX is row-major
    int n_int = static_cast<int>(n);
    for (uint32_t b = 0; b < batch; ++b) {
        float* m = dst + static_cast<size_t>(b) * mat_floats;
        int info = 0;
        spotrf_(&uplo, &n_int, m, &n_int, &info);
        if (info != 0) {
            // Non-posdef or error: fill with NaN to match GPU behavior
            for (uint32_t i = 0; i < mat_floats; ++i) {
                m[i] = NAN;
            }
            continue;
        }
        if (lower) {
            for (uint32_t row = 0; row < n; ++row) {
                for (uint32_t col = row + 1; col < n; ++col) {
                    m[row * n + col] = 0.0f;
                }
            }
        } else {
            for (uint32_t row = 1; row < n; ++row) {
                for (uint32_t col = 0; col < row; ++col) {
                    m[row * n + col] = 0.0f;
                }
            }
        }
    }
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

    // Validation outside autoreleasepool to reduce overhead on fast path.
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

    @autoreleasepool {
        MPS_LOG_DEBUG("Execute: entering autoreleasepool\n");

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

        // ── Single-step native baked fast path ──────────────────────────
        // For single baked native step plans, skip scratch pool, generic step
        // loop, and deferred cleanup. Supports: elementwise, fused elementwise,
        // reduce, softmax, gather, GEMM, and sort.
        if (plan_->steps.size() == 1 && plan_->graph_steps.empty() && plan_->output_slots.size() == 1 &&
            plan_->steps[0].kind == Step::NATIVE && plan_->slots.size() <= 16) {
            auto& ns_fp = plan_->native_steps[plan_->steps[0].index];
            const auto fp_kind = ns_fp.baked_kind;
            const bool fp_has_pipeline = (ns_fp.baked_pipeline != nil);
            // Eligible baked kinds for the fast path
            const bool fp_eligible =
                fp_has_pipeline &&
                (fp_kind == NativeStep::BakedKind::kElementwise ||
                 fp_kind == NativeStep::BakedKind::kFusedElementwise ||
                 fp_kind == NativeStep::BakedKind::kReduce ||
                 fp_kind == NativeStep::BakedKind::kReduceSplit ||
                 fp_kind == NativeStep::BakedKind::kSoftmax ||
                 fp_kind == NativeStep::BakedKind::kGatherRows || fp_kind == NativeStep::BakedKind::kGEMM ||
                 fp_kind == NativeStep::BakedKind::kSort);
            if (fp_eligible) {
                bool fp_ok = true;
                for (size_t i = 0; i < num_args && fp_ok; ++i) {
                    fp_ok = inputs[i] && inputs[i]->metal_buffer() && inputs[i]->is_contiguous();
                }
                if (fp_ok) {
                    size_t out_bytes = plan_->slots[ns_fp.output_slot].byte_size;
                    id<MTLBuffer> out_buf =
                        (out_bytes > 0) ? AcquireReusableOutputBuffer(mtl_device, out_bytes) : nil;
                    if (out_buf) {
                        // Create command buffer. We use the default
                        // commandBufferWithDescriptor-less path with
                        // retainedReferences=YES (the Metal default) so that
                        // MTLBuffers bound to the command encoder stay alive
                        // until the GPU finishes executing the command buffer.
                        //
                        // The previous "fp_unretained" optimization (setting
                        // retainedReferences=NO) saved a few hundred bytes of
                        // memory per command buffer but was the root cause of
                        // the "Metal command buffer error: Invalid Resource"
                        // failures in the test suite: the next eager Execute()
                        // would free/reuse an MTLBuffer that was still bound to
                        // the previous command buffer's compute encoder.
                        static const bool fp_unretained = false;
                        id<MTLCommandBuffer> fp_cmd =
                            fp_unretained ? [commandQueue commandBuffer] // unused, kept for the type
                                          : [commandQueue commandBuffer];
                        if (fp_cmd) {
                            static const bool fp_async =
                                !EnvFlagEnabled("JAX_METALLIB_DISABLE_ASYNC_COMPLETION");
                            // Reuse CompletionEvent from thread-local pool when possible.
                            // TryReset only checks readiness; it does NOT prove
                            // there are no remaining observers. Only reuse when
                            // the TLS slot is the sole owner — otherwise prior
                            // output buffers / PJRT events would flip back to
                            // pending or observe a later execution's error.
                            thread_local CompletionEvent tl_fp_completion;
                            CompletionEvent fp_completion;
                            if (tl_fp_completion && tl_fp_completion.use_count() == 1 &&
                                tl_fp_completion->TryReset()) {
                                fp_completion = tl_fp_completion;
                            } else {
                                fp_completion = MakePendingCompletionEvent();
                                tl_fp_completion = fp_completion;
                            }
                            if (fp_async) {
                                // Capture raw pointer to avoid shared_ptr copy (2 atomics) in block.
                                // Safe: MpsBuffer holds a shared_ptr keeping the event alive.
                                CompletionEventState* fp_raw = fp_completion.get();
                                [fp_cmd addCompletedHandler:^(id<MTLCommandBuffer> completed) {
                                  if (!completed || completed.status == MTLCommandBufferStatusCompleted) {
                                      fp_raw->MarkReady();
                                  } else {
                                      NSString* desc = completed.error ? completed.error.localizedDescription
                                                                       : @"unknown error";
                                      fp_raw->MarkError("Metal command buffer error: " +
                                                        std::string([desc UTF8String]));
                                  }
                                }];
                            }
                            id<MTLComputeCommandEncoder> fp_enc = [fp_cmd computeCommandEncoder];
                            if (fp_enc) {
                                // Build small stack mapping: slot ID → MTLBuffer.
                                id<MTLBuffer> fp_slot[16] = {};
                                for (size_t i = 0; i < num_args; ++i) {
                                    SlotId s = plan_->input_slots[i];
                                    if (s >= 0 && s < 16)
                                        fp_slot[s] = (__bridge id<MTLBuffer>)inputs[i]->metal_buffer();
                                }
                                bool fp_encoded = false;
                                [fp_enc setComputePipelineState:ns_fp.baked_pipeline];
                                switch (fp_kind) {
                                case NativeStep::BakedKind::kElementwise: {
                                    if (ns_fp.input_slots.size() == 1) {
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:1];
                                        [fp_enc setBytes:&ns_fp.baked_n
                                                  length:sizeof(ns_fp.baked_n)
                                                 atIndex:2];
                                    } else if (ns_fp.input_slots.size() == 2) {
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[1]] offset:0 atIndex:1];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_n
                                                  length:sizeof(ns_fp.baked_n)
                                                 atIndex:3];
                                    } else if (ns_fp.input_slots.size() == 3) {
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[1]] offset:0 atIndex:1];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[2]] offset:0 atIndex:2];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:3];
                                        [fp_enc setBytes:&ns_fp.baked_n
                                                  length:sizeof(ns_fp.baked_n)
                                                 atIndex:4];
                                    }
                                    if (ns_fp.baked_use_stride) {
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                               threadsPerThreadgroup:ns_fp.baked_tg];
                                    } else {
                                        [fp_enc dispatchThreads:ns_fp.baked_grid
                                            threadsPerThreadgroup:ns_fp.baked_tg];
                                    }
                                    fp_encoded = true;
                                    break;
                                }
                                case NativeStep::BakedKind::kFusedElementwise: {
                                    if (ns_fp.fused_info) {
                                        auto& fi = *ns_fp.fused_info;
                                        uint32_t buf_idx = 0;
                                        for (SlotId s : fi.external_input_slots) {
                                            [fp_enc setBuffer:fp_slot[s] offset:0 atIndex:buf_idx++];
                                        }
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:buf_idx++];
                                        [fp_enc setBytes:&ns_fp.baked_n
                                                  length:sizeof(ns_fp.baked_n)
                                                 atIndex:buf_idx++];
                                        if (ns_fp.baked_use_stride) {
                                            [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                                   threadsPerThreadgroup:ns_fp.baked_tg];
                                        } else {
                                            [fp_enc dispatchThreads:ns_fp.baked_grid
                                                threadsPerThreadgroup:ns_fp.baked_tg];
                                        }
                                        fp_encoded = true;
                                    }
                                    break;
                                }
                                case NativeStep::BakedKind::kReduce: {
                                    [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                    [fp_enc setBuffer:out_buf offset:0 atIndex:1];
                                    [fp_enc setBytes:&ns_fp.baked_rows
                                              length:sizeof(ns_fp.baked_rows)
                                             atIndex:2];
                                    [fp_enc setBytes:&ns_fp.baked_cols
                                              length:sizeof(ns_fp.baked_cols)
                                             atIndex:3];
                                    if (ns_fp.baked_reduce_threadgroup) {
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                               threadsPerThreadgroup:ns_fp.baked_tg];
                                    } else {
                                        [fp_enc dispatchThreads:ns_fp.baked_grid
                                            threadsPerThreadgroup:ns_fp.baked_tg];
                                    }
                                    fp_encoded = true;
                                    break;
                                }
                                case NativeStep::BakedKind::kReduceSplit: {
                                    // Stage 1: G threadgroups reduce column groups into partials.
                                    [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                    [fp_enc setBuffer:ns_fp.baked_reduce_split_partials
                                               offset:0
                                              atIndex:1];
                                    [fp_enc setBytes:&ns_fp.baked_rows
                                              length:sizeof(ns_fp.baked_rows)
                                             atIndex:2];
                                    [fp_enc setBytes:&ns_fp.baked_cols
                                              length:sizeof(ns_fp.baked_cols)
                                             atIndex:3];
                                    [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                           threadsPerThreadgroup:ns_fp.baked_tg];
                                    [fp_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                                    // Stage 2: combine the G partials into the scalar output.
                                    [fp_enc setComputePipelineState:ns_fp.baked_reduce_split_combine_pipeline];
                                    [fp_enc setBuffer:ns_fp.baked_reduce_split_partials
                                               offset:0
                                              atIndex:0];
                                    [fp_enc setBuffer:out_buf offset:0 atIndex:1];
                                    uint32_t rows_c = 1;
                                    uint32_t cols_c = ns_fp.baked_reduce_split_groups;
                                    [fp_enc setBytes:&rows_c length:sizeof(rows_c) atIndex:2];
                                    [fp_enc setBytes:&cols_c length:sizeof(cols_c) atIndex:3];
                                    if (ns_fp.baked_reduce_split_combine_tg_dispatch) {
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_reduce_split_combine_grid
                                               threadsPerThreadgroup:ns_fp.baked_reduce_split_combine_tg];
                                    } else {
                                        [fp_enc dispatchThreads:ns_fp.baked_reduce_split_combine_grid
                                            threadsPerThreadgroup:ns_fp.baked_reduce_split_combine_tg];
                                    }
                                    fp_encoded = true;
                                    break;
                                }
                                case NativeStep::BakedKind::kSoftmax: {
                                    if (!ns_fp.baked_softmax_pass2_pipeline) {
                                        // Single-pass softmax_numerator: exp(x - max(x))
                                        // Uses threadgroup memory to cache row data.
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:1];
                                        [fp_enc setBytes:&ns_fp.baked_rows
                                                  length:sizeof(ns_fp.baked_rows)
                                                 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_cols
                                                  length:sizeof(ns_fp.baked_cols)
                                                 atIndex:3];
                                        NSUInteger tg_mem = (NSUInteger)ns_fp.baked_cols * sizeof(float);
                                        [fp_enc setThreadgroupMemoryLength:tg_mem atIndex:0];
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                               threadsPerThreadgroup:ns_fp.baked_tg];
                                        fp_encoded = true;
                                    } else if (ns_fp.baked_softmax_maxsum) {
                                        // 2-pass full softmax
                                        // Pass 1: compute max+sum per row
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:ns_fp.baked_softmax_maxsum offset:0 atIndex:1];
                                        [fp_enc setBytes:&ns_fp.baked_rows
                                                  length:sizeof(ns_fp.baked_rows)
                                                 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_cols
                                                  length:sizeof(ns_fp.baked_cols)
                                                 atIndex:3];
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                               threadsPerThreadgroup:ns_fp.baked_tg];
                                        [fp_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                                        // Pass 2: normalize
                                        [fp_enc setComputePipelineState:ns_fp.baked_softmax_pass2_pipeline];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:ns_fp.baked_softmax_maxsum offset:0 atIndex:1];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_cols
                                                  length:sizeof(ns_fp.baked_cols)
                                                 atIndex:3];
                                        [fp_enc dispatchThreads:ns_fp.baked_softmax_pass2_grid
                                            threadsPerThreadgroup:ns_fp.baked_softmax_pass2_tg];
                                        fp_encoded = true;
                                    }
                                    break;
                                }
                                case NativeStep::BakedKind::kGatherRows: {
                                    if (ns_fp.input_slots.size() >= 2) {
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[1]] offset:0 atIndex:1];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_gather_num_indices
                                                  length:sizeof(ns_fp.baked_gather_num_indices)
                                                 atIndex:3];
                                        [fp_enc setBytes:&ns_fp.baked_gather_cols
                                                  length:sizeof(ns_fp.baked_gather_cols)
                                                 atIndex:4];
                                        [fp_enc setBytes:&ns_fp.baked_gather_table_rows
                                                  length:sizeof(ns_fp.baked_gather_table_rows)
                                                 atIndex:5];
                                        if (ns_fp.baked_use_stride) {
                                            [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                                   threadsPerThreadgroup:ns_fp.baked_tg];
                                        } else {
                                            [fp_enc dispatchThreads:ns_fp.baked_grid
                                                threadsPerThreadgroup:ns_fp.baked_tg];
                                        }
                                        fp_encoded = true;
                                    }
                                    break;
                                }
                                case NativeStep::BakedKind::kGEMM: {
                                    if (ns_fp.input_slots.size() >= 2) {
                                        bool isNaive = (ns_fp.baked_gemm_variant == kNaiveGemmVariant);
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[1]] offset:0 atIndex:1];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_gemm_m
                                                  length:sizeof(ns_fp.baked_gemm_m)
                                                 atIndex:3];
                                        [fp_enc setBytes:&ns_fp.baked_gemm_n
                                                  length:sizeof(ns_fp.baked_gemm_n)
                                                 atIndex:4];
                                        [fp_enc setBytes:&ns_fp.baked_gemm_k
                                                  length:sizeof(ns_fp.baked_gemm_k)
                                                 atIndex:5];
                                        if (isNaive) {
                                            MTLSize naive_grid =
                                                MTLSizeMake(ns_fp.baked_gemm_n, ns_fp.baked_gemm_m,
                                                            ns_fp.baked_gemm_batch);
                                            [fp_enc dispatchThreads:naive_grid
                                                threadsPerThreadgroup:ns_fp.baked_tg];
                                        } else {
                                            [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                                   threadsPerThreadgroup:ns_fp.baked_tg];
                                        }
                                        fp_encoded = true;
                                    }
                                    break;
                                }
                                case NativeStep::BakedKind::kSort: {
                                    if (!ns_fp.input_slots.empty()) {
                                        [fp_enc setBuffer:fp_slot[ns_fp.input_slots[0]] offset:0 atIndex:0];
                                        [fp_enc setBuffer:out_buf offset:0 atIndex:1];
                                        [fp_enc setBytes:&ns_fp.baked_sort_n
                                                  length:sizeof(ns_fp.baked_sort_n)
                                                 atIndex:2];
                                        [fp_enc setBytes:&ns_fp.baked_sort_padded_n
                                                  length:sizeof(ns_fp.baked_sort_padded_n)
                                                 atIndex:3];
                                        if (ns_fp.baked_sort_num_rows > 1) {
                                            [fp_enc setBytes:&ns_fp.baked_sort_n
                                                      length:sizeof(ns_fp.baked_sort_n)
                                                     atIndex:4];
                                        }
                                        NSUInteger tg_mem = SortThreadgroupBytes(ns_fp.baked_sort_padded_n);
                                        [fp_enc setThreadgroupMemoryLength:tg_mem atIndex:0];
                                        [fp_enc dispatchThreadgroups:ns_fp.baked_grid
                                               threadsPerThreadgroup:ns_fp.baked_tg];
                                        fp_encoded = true;
                                    }
                                    break;
                                }
                                default:
                                    break;
                                }
                                if (fp_encoded) {
                                    [fp_enc endEncoding];
                                    [fp_cmd commit];
                                    if (!fp_async) {
                                        [fp_cmd waitUntilCompleted];
                                        MarkCompletionFromCommandBuffer(fp_completion, fp_cmd);
                                    }
                                    // Wrap output.
                                    const auto& fp_meta = plan_->output_meta[0];
                                    static const bool fp_pool = OutputBufferPoolEnabled();
                                    void* fp_dk = (__bridge void*)mtl_device;
                                    ExecutionResult fp_result;
                                    if (fp_pool) {
                                        fp_result.buffers.push_back(std::make_unique<MpsBuffer>(
                                            device, (__bridge void*)out_buf, fp_meta.pjrt_dtype, fp_meta.dims,
                                            [fp_dk, out_bytes](void* mb) {
                                                RecycleReusableOutputBuffer(fp_dk, out_bytes, mb);
                                            },
                                            fp_completion));
                                    } else {
                                        fp_result.buffers.push_back(std::make_unique<MpsBuffer>(
                                            device, (__bridge void*)out_buf, fp_meta.pjrt_dtype, fp_meta.dims,
                                            MpsBuffer::BufferReleaser{}, fp_completion));
                                    }
                                    CFRelease((__bridge CFTypeRef)out_buf);
                                    fp_result.completion = fp_completion;
                                    return fp_result;
                                }
                                // Encoding failed for this kind — close encoder and fall through.
                                [fp_enc endEncoding];
                            }
                        }
                        // Encoder/cmd creation failed — recycle buffer and fall through.
                        RecycleReusableOutputBuffer((__bridge void*)mtl_device, out_bytes,
                                                    (__bridge void*)out_buf);
                    }
                }
            }
        }

        // Thread-local scratch arrays — avoids mutex-protected pool acquire/return.
        thread_local std::vector<id<MTLBuffer>> tl_slot_bufs;
        thread_local std::vector<CompletionEvent> tl_slot_events;
        const size_t num_slots = plan_->slots.size();
        tl_slot_bufs.assign(num_slots, nil);
        tl_slot_events.assign(num_slots, nullptr);
        auto& slot_bufs = tl_slot_bufs;
        auto& slot_producer_events = tl_slot_events;

        CompletionEvent execution_completion = MakeReadyCompletionEvent();

        // Materialized copies of non-contiguous inputs (lazy — only allocated if needed).
        std::vector<std::unique_ptr<MpsBuffer>> materialized_inputs;

        for (size_t i = 0; i < num_args; i++) {
            MpsBuffer* input = inputs[i];
            if (!input) {
                return ExecutionResult::Error("Null input buffer at index " + std::to_string(i));
            }
            if (__builtin_expect(!input->is_contiguous(), 0)) {
                std::string mat_err;
                auto contig = input->MaterializeContiguous(&mat_err);
                if (!contig) {
                    return ExecutionResult::Error("Failed to materialize contiguous input " +
                                                  std::to_string(i) + ": " + mat_err);
                }
                materialized_inputs.push_back(std::move(contig));
                input = materialized_inputs.back().get();
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
        static const bool enable_output_pool = OutputBufferPoolEnabled();
        std::unique_ptr<IntermediateCacheSet> cache_set;
        IntermediateCacheSet* cache = nullptr;
        // Only acquire intermediate cache for plans that have graph steps (avoids
        // mutex lock on the hot path for native-only programs).
        if (cache_intermediates && !plan_->graph_steps.empty()) {
            std::scoped_lock lock(plan_->intermediate_pool_mutex);
            if (!plan_->intermediate_pool.empty()) {
                cache_set = std::move(plan_->intermediate_pool.back());
                plan_->intermediate_pool.pop_back();
            }
            if (!cache_set) {
                cache_set = std::make_unique<IntermediateCacheSet>(plan_->slots.size());
            }
            cache = cache_set.get();
        }

        for (const auto& gs : plan_->graph_steps) {
            for (const auto& [slot, tensor] : gs.targets) {
                if (slot_bufs[slot]) {
                    continue;
                }
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
                    // Zero-element tensors (e.g. shape [0] or [5,0,3]) are valid in JAX/XLA.
                    // Metal doesn't allow zero-length buffers, so allocate a minimum 4-byte
                    // placeholder.  The tensor carries no data so the buffer is never read.
                    byte_size = 4;
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
                        // Use executionDescriptor.completionHandler to get async completion
                        // notification directly from MPSGraphExecutable, eliminating the
                        // separate no-op barrier command buffer (~150-300us savings).
                        MPSGraphExecutableExecutionDescriptor* execDesc =
                            [[MPSGraphExecutableExecutionDescriptor alloc] init];
                        CompletionEvent ready = MakePendingCompletionEvent();
                        CompletionEvent captured = ready;
                        execDesc.completionHandler =
                            ^(NSArray<MPSGraphTensorData*>* results, NSError* error) {
                              if (error) {
                                  captured->MarkError([[error localizedDescription] UTF8String]);
                              } else {
                                  captured->MarkReady();
                              }
                            };
                        [exec runWithMTLCommandQueue:commandQueue
                                         inputsArray:inputsArray
                                        resultsArray:resultsArray
                                 executionDescriptor:execDesc];

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

            // Retained command buffers by default. The plan manages its own slot
            // buffers, but not caller-owned inputs: with unretained references an
            // input released before the GPU finished gave intermittent
            // "Invalid Resource" (native row sort under the intrinsic family:
            // 11/480 failed cases unretained vs 0/480 retained), and retaining
            // measured no cost (sort [4096,256] ~0.49 ms, row sum [4096,4096]
            // ~0.75 ms, 64x64 sum ~0.16 ms either way).
            // JAX_METALLIB_ENABLE_NATIVE_UNRETAINED_CMDBUF=1 restores unretained
            // buffers; JAX_METALLIB_DISABLE_NATIVE_UNRETAINED_CMDBUF=1 still wins.
            static const bool use_unretained_default =
                EnvFlagEnabled("JAX_METALLIB_ENABLE_NATIVE_UNRETAINED_CMDBUF") &&
                !EnvFlagEnabled("JAX_METALLIB_DISABLE_NATIVE_UNRETAINED_CMDBUF");
            const bool use_unretained = use_unretained_default && !plan_->needs_retained_cmd_bufs;
            static const int max_ops_per_cmd = EnvIntValue("JAX_METALLIB_MAX_OPS_PER_CMDBUF", 256);
            static const int max_mb_per_cmd = EnvIntValue("JAX_METALLIB_MAX_MB_PER_CMDBUF", 256);

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

                if (ns.baked_kind == NativeStep::BakedKind::kReshapeAlias ||
                    ns.baked_kind == NativeStep::BakedKind::kTransposeAlias) {
                    if (ns.input_slots.empty()) {
                        return ExecutionResult::Error("Baked reshape/transpose missing input slot");
                    }
                    SlotId in_slot = ns.input_slots[0];
                    if (in_slot < 0 || (size_t)in_slot >= slot_bufs.size() || !slot_bufs[in_slot]) {
                        return ExecutionResult::Error("Baked reshape/transpose input slot is nil");
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

                // CPU sort for medium arrays — faster than GPU bitonic for 8K-64K range
                if (ns.baked_kind == NativeStep::BakedKind::kCpuSort) {
                    if (ns.input_slots.empty()) {
                        return ExecutionResult::Error("CPU sort missing input slot");
                    }
                    // Flush any pending GPU work so input buffers are ready on CPU
                    if (ops_in_cmd > 0) {
                        if (!flush_native_cmd()) {
                            return ExecutionResult::Error("Failed to flush before CPU sort");
                        }
                    }
                    // The flush above only commits local work; in async mode the
                    // GPU may still be running, and inputs produced by earlier
                    // executions are not covered by it at all. Wait for the
                    // producers of the actual input buffers before CPU reads.
                    for (auto slot : ns.input_slots) {
                        if (slot >= 0 && (size_t)slot < slot_producer_events.size()) {
                            const CompletionEvent& prod = slot_producer_events[(size_t)slot];
                            try {
                                AwaitCompletionEvent(prod);
                            } catch (const std::exception& e) {
                                return ExecutionResult::Error(std::string("CPU sort: input wait failed: ") +
                                                              e.what());
                            }
                            std::string prod_err = CompletionEventError(prod);
                            if (!prod_err.empty()) {
                                return ExecutionResult::Error("CPU sort: input producer failed: " + prod_err);
                            }
                        }
                    }
                    if (!ensure_output_buffer(ns.output_slot)) {
                        return ExecutionResult::Error("Failed to allocate CPU sort output");
                    }
                    const float* src = (const float*)[slot_bufs[ns.input_slots[0]] contents];
                    float* dst = (float*)[slot_bufs[ns.output_slot] contents];
                    vDSP_Length sort_n = (vDSP_Length)ns.baked_sort_n;
                    uint32_t num_rows = ns.baked_sort_num_rows;
                    if (num_rows <= 1) {
                        memcpy(dst, src, sort_n * sizeof(float));
                        vDSP_vsort(dst, sort_n, 1); // 1 = ascending
                    } else {
                        // Parallel 2D row-wise sort using GCD dispatch_apply
                        size_t row_bytes = sort_n * sizeof(float);
                        memcpy(dst, src, (size_t)num_rows * row_bytes);
                        dispatch_apply((size_t)num_rows, DISPATCH_APPLY_AUTO, ^(size_t row) {
                          vDSP_vsort(dst + row * sort_n, sort_n, 1);
                        });
                    }
                    slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                    continue;
                }

                // CPU Cholesky for small matrices — LAPACK spotrf_ on CPU
                if (ns.baked_kind == NativeStep::BakedKind::kCpuCholesky) {
                    if (ns.input_slots.empty()) {
                        return ExecutionResult::Error("CPU Cholesky missing input slot");
                    }
                    // Flush any pending GPU work so input buffers are ready on CPU
                    if (ops_in_cmd > 0) {
                        if (!flush_native_cmd()) {
                            return ExecutionResult::Error("Failed to flush before CPU Cholesky");
                        }
                    }
                    // See CPU sort above: flush does not guarantee input readiness.
                    for (auto slot : ns.input_slots) {
                        if (slot >= 0 && (size_t)slot < slot_producer_events.size()) {
                            const CompletionEvent& prod = slot_producer_events[(size_t)slot];
                            try {
                                AwaitCompletionEvent(prod);
                            } catch (const std::exception& e) {
                                return ExecutionResult::Error(std::string("CPU Cholesky: input wait failed: ") +
                                                              e.what());
                            }
                            std::string prod_err = CompletionEventError(prod);
                            if (!prod_err.empty()) {
                                return ExecutionResult::Error("CPU Cholesky: input producer failed: " + prod_err);
                            }
                        }
                    }
                    if (!ensure_output_buffer(ns.output_slot)) {
                        return ExecutionResult::Error("Failed to allocate CPU Cholesky output");
                    }
                    RunCpuCholeskyBatch((float*)[slot_bufs[ns.output_slot] contents],
                                        (const float*)[slot_bufs[ns.input_slots[0]] contents],
                                        ns.baked_cholesky_n, ns.baked_cholesky_batch,
                                        ns.baked_cholesky_lower);
                    slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                    continue;
                }

                // CPU BLAS for tiny matmuls — bypasses GPU dispatch entirely
                if (ns.baked_kind == NativeStep::BakedKind::kCpuBLAS) {
                    if (ns.input_slots.size() < 2) {
                        return ExecutionResult::Error("CPU BLAS missing input slots");
                    }
                    // Flush any pending GPU work so input buffers are ready on CPU
                    if (ops_in_cmd > 0) {
                        if (!flush_native_cmd()) {
                            return ExecutionResult::Error("Failed to flush before CPU BLAS");
                        }
                    }
                    // See CPU sort above: flush does not guarantee input readiness.
                    for (auto slot : ns.input_slots) {
                        if (slot >= 0 && (size_t)slot < slot_producer_events.size()) {
                            const CompletionEvent& prod = slot_producer_events[(size_t)slot];
                            try {
                                AwaitCompletionEvent(prod);
                            } catch (const std::exception& e) {
                                return ExecutionResult::Error(std::string("CPU BLAS: input wait failed: ") +
                                                              e.what());
                            }
                            std::string prod_err = CompletionEventError(prod);
                            if (!prod_err.empty()) {
                                return ExecutionResult::Error("CPU BLAS: input producer failed: " + prod_err);
                            }
                        }
                    }
                    if (!ensure_output_buffer(ns.output_slot)) {
                        return ExecutionResult::Error("Failed to allocate CPU BLAS output");
                    }
                    float* A = (float*)[slot_bufs[ns.input_slots[0]] contents];
                    float* B = (float*)[slot_bufs[ns.input_slots[1]] contents];
                    float* C = (float*)[slot_bufs[ns.output_slot] contents];
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)ns.baked_gemm_m,
                                (int)ns.baked_gemm_n, (int)ns.baked_gemm_k, 1.0f, A, (int)ns.baked_gemm_k, B,
                                (int)ns.baked_gemm_n, 0.0f, C, (int)ns.baked_gemm_n);
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
                        if (ns.baked_use_stride) {
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                        } else {
                            [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                            threadsPerThreadgroup:ns.baked_tg];
                        }
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
                        if (ns.baked_use_stride) {
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                        } else {
                            [nativeComputeEncoder dispatchThreads:ns.baked_grid
                                            threadsPerThreadgroup:ns.baked_tg];
                        }
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
                    case NativeStep::BakedKind::kReduceSplit: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked split reduce missing input slot");
                        }
                        // Stage 1: G threadgroups reduce column groups into partials.
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:ns.baked_reduce_split_partials
                                                 offset:0
                                                atIndex:1];
                        [nativeComputeEncoder setBytes:&ns.baked_rows
                                                length:sizeof(ns.baked_rows)
                                               atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_cols
                                                length:sizeof(ns.baked_cols)
                                               atIndex:3];
                        [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                             threadsPerThreadgroup:ns.baked_tg];
                        [nativeComputeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        // Stage 2: combine the G partials into the scalar output.
                        [nativeComputeEncoder
                            setComputePipelineState:ns.baked_reduce_split_combine_pipeline];
                        [nativeComputeEncoder setBuffer:ns.baked_reduce_split_partials
                                                 offset:0
                                                atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                        uint32_t rows_c = 1;
                        uint32_t cols_c = ns.baked_reduce_split_groups;
                        [nativeComputeEncoder setBytes:&rows_c length:sizeof(rows_c) atIndex:2];
                        [nativeComputeEncoder setBytes:&cols_c length:sizeof(cols_c) atIndex:3];
                        if (ns.baked_reduce_split_combine_tg_dispatch) {
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_reduce_split_combine_grid
                                                 threadsPerThreadgroup:ns.baked_reduce_split_combine_tg];
                        } else {
                            [nativeComputeEncoder dispatchThreads:ns.baked_reduce_split_combine_grid
                                            threadsPerThreadgroup:ns.baked_reduce_split_combine_tg];
                        }
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kSoftmax: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked softmax missing input slot");
                        }
                        [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                        if (!ns.baked_softmax_pass2_pipeline) {
                            // Single-pass softmax_numerator
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_rows
                                                    length:sizeof(ns.baked_rows)
                                                   atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_cols
                                                    length:sizeof(ns.baked_cols)
                                                   atIndex:3];
                            NSUInteger tg_mem = (NSUInteger)ns.baked_cols * sizeof(float);
                            [nativeComputeEncoder setThreadgroupMemoryLength:tg_mem atIndex:0];
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                        } else if (ns.baked_softmax_maxsum) {
                            // 2-pass full softmax
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:ns.baked_softmax_maxsum offset:0 atIndex:1];
                            [nativeComputeEncoder setBytes:&ns.baked_rows
                                                    length:sizeof(ns.baked_rows)
                                                   atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_cols
                                                    length:sizeof(ns.baked_cols)
                                                   atIndex:3];
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                            [nativeComputeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            [nativeComputeEncoder setComputePipelineState:ns.baked_softmax_pass2_pipeline];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [nativeComputeEncoder setBuffer:ns.baked_softmax_maxsum offset:0 atIndex:1];
                            [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [nativeComputeEncoder setBytes:&ns.baked_cols
                                                    length:sizeof(ns.baked_cols)
                                                   atIndex:3];
                            [nativeComputeEncoder dispatchThreads:ns.baked_softmax_pass2_grid
                                            threadsPerThreadgroup:ns.baked_softmax_pass2_tg];
                        } else {
                            return ExecutionResult::Error("Baked softmax missing state");
                        }
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kLayerNorm: {
                        if (ns.input_slots.empty() || !ns.baked_layernorm_stats ||
                            !ns.baked_layernorm_pass2_pipeline) {
                            return ExecutionResult::Error("Baked layernorm missing state");
                        }
                        // Pass 1: compute mean + inv_std per row
                        [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:ns.baked_layernorm_stats offset:0 atIndex:1];
                        [nativeComputeEncoder setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                        [nativeComputeEncoder setBytes:&ns.baked_layernorm_eps
                                                length:sizeof(ns.baked_layernorm_eps)
                                               atIndex:4];
                        [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                             threadsPerThreadgroup:ns.baked_tg];
                        [nativeComputeEncoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        // Pass 2: normalize y = (x - mean) * inv_std
                        [nativeComputeEncoder setComputePipelineState:ns.baked_layernorm_pass2_pipeline];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:ns.baked_layernorm_stats offset:0 atIndex:1];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                        [nativeComputeEncoder dispatchThreads:ns.baked_layernorm_pass2_grid
                                        threadsPerThreadgroup:ns.baked_layernorm_pass2_tg];
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kGatherRows: {
                        if (ns.input_slots.size() < 2) {
                            return ExecutionResult::Error("Baked gather missing input slots");
                        }
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_gather_num_indices
                                                length:sizeof(ns.baked_gather_num_indices)
                                               atIndex:3];
                        [nativeComputeEncoder setBytes:&ns.baked_gather_cols
                                                length:sizeof(ns.baked_gather_cols)
                                               atIndex:4];
                        [nativeComputeEncoder setBytes:&ns.baked_gather_table_rows
                                                length:sizeof(ns.baked_gather_table_rows)
                                               atIndex:5];
                        if (ns.baked_use_stride) {
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
                    case NativeStep::BakedKind::kGEMM: {
                        if (ns.input_slots.size() < 2) {
                            return ExecutionResult::Error("Baked GEMM missing input slots");
                        }
                        bool isMpp = (ns.baked_gemm_variant == kMppGemmVariant);
                        bool isNaive = (ns.baked_gemm_variant == kNaiveGemmVariant);
                        // Single dispatch with grid.z = batch (kernels use tgid.z/gid.z for batch offset)
                        [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_gemm_m
                                                length:sizeof(ns.baked_gemm_m)
                                               atIndex:3];
                        [nativeComputeEncoder setBytes:&ns.baked_gemm_n
                                                length:sizeof(ns.baked_gemm_n)
                                               atIndex:4];
                        [nativeComputeEncoder setBytes:&ns.baked_gemm_k
                                                length:sizeof(ns.baked_gemm_k)
                                               atIndex:5];
                        if (isNaive) {
                            MTLSize naive_grid =
                                MTLSizeMake(ns.baked_gemm_n, ns.baked_gemm_m, ns.baked_gemm_batch);
                            [nativeComputeEncoder dispatchThreads:naive_grid
                                            threadsPerThreadgroup:ns.baked_tg];
                        } else {
                            [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                                 threadsPerThreadgroup:ns.baked_tg];
                        }
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kTranspose2D: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked transpose missing input slot");
                        }
                        // Reuse existing encoder if available, else create one
                        if (!nativeComputeEncoder) {
                            if (!ensure_cmd_buf()) {
                                return ExecutionResult::Error("Failed to create cmd buf for transpose");
                            }
                            nativeComputeEncoder = [cmdBuf computeCommandEncoder];
                            if (!nativeComputeEncoder) {
                                return ExecutionResult::Error("Failed to create transpose compute encoder");
                            }
                        }
                        [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                        [nativeComputeEncoder setBytes:&ns.baked_transpose_rows
                                                length:sizeof(ns.baked_transpose_rows)
                                               atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_transpose_cols
                                                length:sizeof(ns.baked_transpose_cols)
                                               atIndex:3];
                        NSUInteger tg_mem = 32 * 33 * sizeof(float);
                        [nativeComputeEncoder setThreadgroupMemoryLength:tg_mem atIndex:0];
                        [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
                                             threadsPerThreadgroup:ns.baked_tg];
                        encoded = true;
                        break;
                    }
                    case NativeStep::BakedKind::kSort: {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("Baked sort missing input slot");
                        }
                        // Sort needs threadgroup memory — use a fresh encoder
                        if (nativeComputeEncoder) {
                            [nativeComputeEncoder endEncoding];
                            nativeComputeEncoder = nil;
                        }
                        nativeComputeEncoder = [cmdBuf computeCommandEncoder];
                        if (!nativeComputeEncoder) {
                            return ExecutionResult::Error("Failed to create sort compute encoder");
                        }
                        [nativeComputeEncoder setComputePipelineState:ns.baked_pipeline];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                        [nativeComputeEncoder setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                        [nativeComputeEncoder setBytes:&ns.baked_sort_n
                                                length:sizeof(ns.baked_sort_n)
                                               atIndex:2];
                        [nativeComputeEncoder setBytes:&ns.baked_sort_padded_n
                                                length:sizeof(ns.baked_sort_padded_n)
                                               atIndex:3];
                        if (ns.baked_sort_num_rows > 1) {
                            [nativeComputeEncoder setBytes:&ns.baked_sort_n
                                                    length:sizeof(ns.baked_sort_n)
                                                   atIndex:4];
                        }
                        NSUInteger tg_mem = SortThreadgroupBytes(ns.baked_sort_padded_n);
                        [nativeComputeEncoder setThreadgroupMemoryLength:tg_mem atIndex:0];
                        [nativeComputeEncoder dispatchThreadgroups:ns.baked_grid
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
                    // Handlers may read input buffers on the CPU (e.g. the
                    // Cholesky LAPACK fallback). Commit and wait for any
                    // pending GPU work in this execution first so host reads
                    // observe finished writes.
                    if (needs_cmd && cmdBuf && ops_in_cmd > 0) {
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        MarkCompletionFromCommandBuffer(current_cmd_completion, cmdBuf);
                        submitted_events.push_back(current_cmd_completion);
                        cmdBuf = nil;
                        current_cmd_completion.reset();
                        ops_in_cmd = 0;
                        bytes_in_cmd = 0;
                    }
                    if (needs_cmd && !cmdBuf) {
                        // The dependency wait above nils cmdBuf; handlers that
                        // encode GPU work (e.g. NativeIntrinsicReduceHandler)
                        // require a valid command buffer, so recreate it.
                        if (!ensure_cmd_buf()) {
                            return ExecutionResult::Error("Failed to recreate native command buffer");
                        }
                    }
                    if (needs_cmd) {
                        // Handlers may read input buffers on the CPU; the local
                        // commit+wait above does not cover inputs produced by
                        // earlier asynchronous work.
                        for (auto slot : ns.input_slots) {
                            if (slot >= 0 && (size_t)slot < slot_producer_events.size()) {
                                const CompletionEvent& prod = slot_producer_events[(size_t)slot];
                                try {
                                    AwaitCompletionEvent(prod);
                                } catch (const std::exception& e) {
                                    return ExecutionResult::Error(
                                        std::string("Native handler: input wait failed: ") + e.what());
                                }
                                std::string prod_err = CompletionEventError(prod);
                                if (!prod_err.empty()) {
                                    return ExecutionResult::Error("Native handler: input producer failed: " +
                                                                  prod_err);
                                }
                            }
                        }
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
            CompletionEvent cmd_completion = MixedCompletionEvent(cmdBuf, async_completion);

            MPS_LOG_DEBUG("Executing %zu steps\n", plan_->steps.size());
            std::vector<id<MTLBuffer>> mixed_input_bufs;
            mixed_input_bufs.reserve(4);
            // Persistent compute encoder reused across consecutive native steps to
            // avoid the ~15-20µs overhead of creating a new encoder per dispatch.
            id<MTLComputeCommandEncoder> mixedNativeEncoder = nil;
            bool mixedNativeEncoderHasWork = false;
            auto endMixedNativeEncoder = [&]() {
                if (mixedNativeEncoder) {
                    [mixedNativeEncoder endEncoding];
                    mixedNativeEncoder = nil;
                    mixedNativeEncoderHasWork = false;
                }
            };
            auto ensureMixedNativeEncoder = [&]() -> id<MTLComputeCommandEncoder> {
                if (!mixedNativeEncoder) {
                    mixedNativeEncoder = [cmdBuf computeCommandEncoder];
                    mixedNativeEncoderHasWork = false;
                }
                return mixedNativeEncoder;
            };
            for (size_t step_idx = 0; step_idx < plan_->steps.size(); step_idx++) {
                const auto& step = plan_->steps[step_idx];
                MPS_LOG_DEBUG("Step %zu: kind=%s\n", step_idx, step.kind == Step::GRAPH ? "GRAPH" : "NATIVE");
                if (step.kind == Step::GRAPH) {
                    endMixedNativeEncoder();
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
                                 executionDescriptor:GraphExecutionDescriptor()];
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

                    if (ns.baked_kind == NativeStep::BakedKind::kReshapeAlias ||
                        ns.baked_kind == NativeStep::BakedKind::kTransposeAlias) {
                        if (!ns.input_slots.empty()) {
                            SlotId in_slot = ns.input_slots[0];
                            if (in_slot >= 0 && (size_t)in_slot < slot_bufs.size() && slot_bufs[in_slot]) {
                                id<MTLBuffer> out = slot_bufs[in_slot];
                                CFRetain((__bridge CFTypeRef)out);
                                slot_bufs[ns.output_slot] = out;
                                slot_producer_events[(size_t)ns.output_slot] =
                                    slot_producer_events[(size_t)in_slot];
                                continue;
                            }
                        }
                    }
                    if (ns.baked_kind == NativeStep::BakedKind::kConstant && ns.baked_constant_buffer) {
                        CFRetain((__bridge CFTypeRef)ns.baked_constant_buffer);
                        slot_bufs[ns.output_slot] = ns.baked_constant_buffer;
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                        continue;
                    }

                    // CPU sort for medium arrays in mixed path
                    if (ns.baked_kind == NativeStep::BakedKind::kCpuSort) {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("CPU sort missing input slot (mixed)");
                        }
                        // Flush all pending GPU work so input buffers are ready
                        endMixedNativeEncoder();
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        if (DeferMixedCompletion()) {
                            MarkCompletionFromCommandBuffer(cmd_completion, cmdBuf);
                        }
                        // Create a new command buffer for subsequent steps
                        cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
                        if (!cmdBuf) {
                            return ExecutionResult::Error("Failed to recreate cmdBuf after CPU sort");
                        }
                        cmd_completion = MixedCompletionEvent(cmdBuf, async_completion);
                        if (ns.output_slot >= 0 && (size_t)ns.output_slot < slot_bufs.size() &&
                            !slot_bufs[ns.output_slot]) {
                            size_t out_bytes = plan_->slots[ns.output_slot].byte_size;
                            if (out_bytes > 0) {
                                slot_bufs[ns.output_slot] =
                                    [mtl_device newBufferWithLength:out_bytes
                                                            options:ComputeBufferResourceOptions()];
                            }
                        }
                        const float* src = (const float*)[slot_bufs[ns.input_slots[0]] contents];
                        float* dst = (float*)[slot_bufs[ns.output_slot] contents];
                        vDSP_Length sort_n = (vDSP_Length)ns.baked_sort_n;
                        uint32_t num_rows_mixed = ns.baked_sort_num_rows;
                        if (num_rows_mixed <= 1) {
                            memcpy(dst, src, sort_n * sizeof(float));
                            vDSP_vsort(dst, sort_n, 1);
                        } else {
                            size_t row_bytes = sort_n * sizeof(float);
                            memcpy(dst, src, (size_t)num_rows_mixed * row_bytes);
                            dispatch_apply((size_t)num_rows_mixed, DISPATCH_APPLY_AUTO, ^(size_t row) {
                              vDSP_vsort(dst + row * sort_n, sort_n, 1);
                            });
                        }
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                        continue;
                    }

                    // CPU Cholesky for small matrices in mixed path
                    if (ns.baked_kind == NativeStep::BakedKind::kCpuCholesky) {
                        if (ns.input_slots.empty()) {
                            return ExecutionResult::Error("CPU Cholesky missing input slot (mixed)");
                        }
                        // Flush all pending GPU work so input buffers are ready
                        endMixedNativeEncoder();
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        if (DeferMixedCompletion()) {
                            MarkCompletionFromCommandBuffer(cmd_completion, cmdBuf);
                        }
                        // Create a new command buffer for subsequent steps
                        cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
                        if (!cmdBuf) {
                            return ExecutionResult::Error("Failed to recreate cmdBuf after CPU Cholesky");
                        }
                        cmd_completion = MixedCompletionEvent(cmdBuf, async_completion);
                        if (ns.output_slot >= 0 && (size_t)ns.output_slot < slot_bufs.size() &&
                            !slot_bufs[ns.output_slot]) {
                            size_t out_bytes = plan_->slots[ns.output_slot].byte_size;
                            if (out_bytes > 0) {
                                slot_bufs[ns.output_slot] =
                                    [mtl_device newBufferWithLength:out_bytes
                                                            options:ComputeBufferResourceOptions()];
                            }
                        }
                        RunCpuCholeskyBatch((float*)[slot_bufs[ns.output_slot] contents],
                                            (const float*)[slot_bufs[ns.input_slots[0]] contents],
                                            ns.baked_cholesky_n, ns.baked_cholesky_batch,
                                            ns.baked_cholesky_lower);
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                        continue;
                    }

                    // CPU BLAS for tiny matmuls in mixed path
                    if (ns.baked_kind == NativeStep::BakedKind::kCpuBLAS) {
                        if (ns.input_slots.size() < 2) {
                            return ExecutionResult::Error("CPU BLAS missing input slots (mixed)");
                        }
                        // Flush all pending GPU work so input buffers are ready
                        endMixedNativeEncoder();
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        if (DeferMixedCompletion()) {
                            MarkCompletionFromCommandBuffer(cmd_completion, cmdBuf);
                        }
                        // Create a new command buffer for subsequent steps
                        cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
                        if (!cmdBuf) {
                            return ExecutionResult::Error("Failed to recreate cmdBuf after CPU BLAS");
                        }
                        cmd_completion = MixedCompletionEvent(cmdBuf, async_completion);
                        if (ns.output_slot >= 0 && (size_t)ns.output_slot < slot_bufs.size() &&
                            !slot_bufs[ns.output_slot]) {
                            size_t out_bytes = plan_->slots[ns.output_slot].byte_size;
                            if (out_bytes > 0) {
                                slot_bufs[ns.output_slot] =
                                    [mtl_device newBufferWithLength:out_bytes
                                                            options:ComputeBufferResourceOptions()];
                            }
                        }
                        float* A = (float*)[slot_bufs[ns.input_slots[0]] contents];
                        float* B = (float*)[slot_bufs[ns.input_slots[1]] contents];
                        float* C = (float*)[slot_bufs[ns.output_slot] contents];
                        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, (int)ns.baked_gemm_m,
                                    (int)ns.baked_gemm_n, (int)ns.baked_gemm_k, 1.0f, A, (int)ns.baked_gemm_k,
                                    B, (int)ns.baked_gemm_n, 0.0f, C, (int)ns.baked_gemm_n);
                        slot_producer_events[(size_t)ns.output_slot] = MakeReadyCompletionEvent();
                        continue;
                    }

                    auto supports_baked_mixed = [](NativeStep::BakedKind kind) {
                        switch (kind) {
                        case NativeStep::BakedKind::kSoftmax:
                        case NativeStep::BakedKind::kLayerNorm:
                        case NativeStep::BakedKind::kGatherRows:
                        case NativeStep::BakedKind::kElementwise:
                        case NativeStep::BakedKind::kReduce:
                        case NativeStep::BakedKind::kFusedElementwise:
                        case NativeStep::BakedKind::kGEMM:
                        case NativeStep::BakedKind::kSort:
                        case NativeStep::BakedKind::kTranspose2D:
                            return true;
                        default:
                            return false;
                        }
                    };

                    if (ns.baked_kind != NativeStep::BakedKind::kNone && ns.baked_pipeline &&
                        supports_baked_mixed(ns.baked_kind)) {
                        if (ns.output_slot >= 0 && (size_t)ns.output_slot < slot_bufs.size() &&
                            !slot_bufs[ns.output_slot]) {
                            size_t out_bytes = plan_->slots[ns.output_slot].byte_size;
                            if (out_bytes > 0) {
                                slot_bufs[ns.output_slot] =
                                    AcquireReusableOutputBuffer(mtl_device, out_bytes);
                            }
                        }

                        // Reuse persistent encoder; add memory barrier if we already
                        // dispatched something (ensures prior writes are visible).
                        id<MTLComputeCommandEncoder> enc = ensureMixedNativeEncoder();
                        if (!enc) {
                            return ExecutionResult::Error("Failed to create compute encoder in mixed path");
                        }
                        // Only insert a memory barrier if we already dispatched work
                        // into this encoder (prior writes must be visible to this read).
                        if (mixedNativeEncoderHasWork) {
                            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                        }

                        bool baked_ok = true;
                        switch (ns.baked_kind) {
                        case NativeStep::BakedKind::kSoftmax: {
                            if (ns.input_slots.empty()) {
                                endMixedNativeEncoder();
                                return ExecutionResult::Error(
                                    "Baked softmax missing input slot (mixed path)");
                            }
                            if (!ns.baked_softmax_pass2_pipeline) {
                                // Single-pass full softmax or softmax numerator
                                [enc setComputePipelineState:ns.baked_pipeline];
                                [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                                [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                                [enc setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                                [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                                NSUInteger tg_mem = (NSUInteger)ns.baked_cols * sizeof(float);
                                [enc setThreadgroupMemoryLength:tg_mem atIndex:0];
                                [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                                break;
                            }
                            if (!ns.baked_softmax_maxsum) {
                                endMixedNativeEncoder();
                                return ExecutionResult::Error("Baked softmax missing state (mixed path)");
                            }
                            [enc setComputePipelineState:ns.baked_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:ns.baked_softmax_maxsum offset:0 atIndex:1];
                            [enc setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                            [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                            [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            [enc setComputePipelineState:ns.baked_softmax_pass2_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:ns.baked_softmax_maxsum offset:0 atIndex:1];
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                            [enc dispatchThreads:ns.baked_softmax_pass2_grid
                                threadsPerThreadgroup:ns.baked_softmax_pass2_tg];
                            break;
                        }
                        case NativeStep::BakedKind::kLayerNorm: {
                            if (ns.input_slots.empty() || !ns.baked_layernorm_stats ||
                                !ns.baked_layernorm_pass2_pipeline) {
                                endMixedNativeEncoder();
                                return ExecutionResult::Error("Baked layernorm missing state (mixed path)");
                            }
                            [enc setComputePipelineState:ns.baked_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:ns.baked_layernorm_stats offset:0 atIndex:1];
                            [enc setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                            [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                            [enc setBytes:&ns.baked_layernorm_eps
                                   length:sizeof(ns.baked_layernorm_eps)
                                  atIndex:4];
                            [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                            [enc setComputePipelineState:ns.baked_layernorm_pass2_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:ns.baked_layernorm_stats offset:0 atIndex:1];
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                            [enc dispatchThreads:ns.baked_layernorm_pass2_grid
                                threadsPerThreadgroup:ns.baked_layernorm_pass2_tg];
                            break;
                        }
                        case NativeStep::BakedKind::kGatherRows: {
                            if (ns.input_slots.size() < 2) {
                                endMixedNativeEncoder();
                                return ExecutionResult::Error(
                                    "Baked gather missing input slots (mixed path)");
                            }
                            [enc setComputePipelineState:ns.baked_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [enc setBytes:&ns.baked_gather_num_indices
                                   length:sizeof(ns.baked_gather_num_indices)
                                  atIndex:3];
                            [enc setBytes:&ns.baked_gather_cols
                                   length:sizeof(ns.baked_gather_cols)
                                  atIndex:4];
                            [enc setBytes:&ns.baked_gather_table_rows
                                   length:sizeof(ns.baked_gather_table_rows)
                                  atIndex:5];
                            if (ns.baked_use_stride) {
                                [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            } else {
                                [enc dispatchThreads:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            }
                            break;
                        }
                        case NativeStep::BakedKind::kElementwise: {
                            [enc setComputePipelineState:ns.baked_pipeline];
                            if (ns.input_slots.size() == 1) {
                                [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                                [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                                [enc setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:2];
                            } else if (ns.input_slots.size() == 2) {
                                [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                                [enc setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                                [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                                [enc setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:3];
                            } else if (ns.input_slots.size() == 3) {
                                [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                                [enc setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                                [enc setBuffer:slot_bufs[ns.input_slots[2]] offset:0 atIndex:2];
                                [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:3];
                                [enc setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:4];
                            } else {
                                baked_ok = false;
                            }
                            if (baked_ok) {
                                if (ns.baked_use_stride) {
                                    [enc dispatchThreadgroups:ns.baked_grid
                                        threadsPerThreadgroup:ns.baked_tg];
                                } else {
                                    [enc dispatchThreads:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                                }
                            }
                            break;
                        }
                        case NativeStep::BakedKind::kReduce: {
                            [enc setComputePipelineState:ns.baked_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [enc setBytes:&ns.baked_rows length:sizeof(ns.baked_rows) atIndex:2];
                            [enc setBytes:&ns.baked_cols length:sizeof(ns.baked_cols) atIndex:3];
                            if (ns.baked_reduce_threadgroup) {
                                [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            } else {
                                [enc dispatchThreads:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            }
                            break;
                        }
                        case NativeStep::BakedKind::kFusedElementwise: {
                            if (!ns.fused_info) {
                                baked_ok = false;
                                break;
                            }
                            [enc setComputePipelineState:ns.baked_pipeline];
                            uint32_t buf_idx = 0;
                            for (SlotId s : ns.fused_info->external_input_slots) {
                                [enc setBuffer:slot_bufs[s] offset:0 atIndex:buf_idx++];
                            }
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:buf_idx++];
                            [enc setBytes:&ns.baked_n length:sizeof(ns.baked_n) atIndex:buf_idx++];
                            if (ns.baked_use_stride) {
                                [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            } else {
                                [enc dispatchThreads:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            }
                            break;
                        }
                        case NativeStep::BakedKind::kGEMM: {
                            if (ns.input_slots.size() < 2) {
                                baked_ok = false;
                                break;
                            }
                            bool isNaive = (ns.baked_gemm_variant == kNaiveGemmVariant);
                            // Single dispatch with grid.z = batch
                            [enc setComputePipelineState:ns.baked_pipeline];
                            [enc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [enc setBuffer:slot_bufs[ns.input_slots[1]] offset:0 atIndex:1];
                            [enc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:2];
                            [enc setBytes:&ns.baked_gemm_m length:sizeof(ns.baked_gemm_m) atIndex:3];
                            [enc setBytes:&ns.baked_gemm_n length:sizeof(ns.baked_gemm_n) atIndex:4];
                            [enc setBytes:&ns.baked_gemm_k length:sizeof(ns.baked_gemm_k) atIndex:5];
                            if (isNaive) {
                                MTLSize naive_grid =
                                    MTLSizeMake(ns.baked_gemm_n, ns.baked_gemm_m, ns.baked_gemm_batch);
                                [enc dispatchThreads:naive_grid threadsPerThreadgroup:ns.baked_tg];
                            } else {
                                [enc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            }
                            break;
                        }
                        case NativeStep::BakedKind::kTranspose2D: {
                            if (ns.input_slots.empty()) {
                                baked_ok = false;
                                break;
                            }
                            endMixedNativeEncoder();
                            id<MTLComputeCommandEncoder> trEnc = ensureMixedNativeEncoder();
                            if (!trEnc) {
                                baked_ok = false;
                                break;
                            }
                            [trEnc setComputePipelineState:ns.baked_pipeline];
                            [trEnc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [trEnc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [trEnc setBytes:&ns.baked_transpose_rows
                                     length:sizeof(ns.baked_transpose_rows)
                                    atIndex:2];
                            [trEnc setBytes:&ns.baked_transpose_cols
                                     length:sizeof(ns.baked_transpose_cols)
                                    atIndex:3];
                            NSUInteger tg_mem = 32 * 33 * sizeof(float);
                            [trEnc setThreadgroupMemoryLength:tg_mem atIndex:0];
                            [trEnc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            break;
                        }
                        case NativeStep::BakedKind::kSort: {
                            if (ns.input_slots.empty()) {
                                baked_ok = false;
                                break;
                            }
                            // Sort needs threadgroup memory — end current encoder, use fresh one
                            endMixedNativeEncoder();
                            id<MTLComputeCommandEncoder> sortEnc = ensureMixedNativeEncoder();
                            if (!sortEnc) {
                                baked_ok = false;
                                break;
                            }
                            [sortEnc setComputePipelineState:ns.baked_pipeline];
                            [sortEnc setBuffer:slot_bufs[ns.input_slots[0]] offset:0 atIndex:0];
                            [sortEnc setBuffer:slot_bufs[ns.output_slot] offset:0 atIndex:1];
                            [sortEnc setBytes:&ns.baked_sort_n length:sizeof(ns.baked_sort_n) atIndex:2];
                            [sortEnc setBytes:&ns.baked_sort_padded_n
                                       length:sizeof(ns.baked_sort_padded_n)
                                      atIndex:3];
                            if (ns.baked_sort_num_rows > 1) {
                                [sortEnc setBytes:&ns.baked_sort_n length:sizeof(ns.baked_sort_n) atIndex:4];
                            }
                            NSUInteger tg_mem = SortThreadgroupBytes(ns.baked_sort_padded_n);
                            [sortEnc setThreadgroupMemoryLength:tg_mem atIndex:0];
                            [sortEnc dispatchThreadgroups:ns.baked_grid threadsPerThreadgroup:ns.baked_tg];
                            break;
                        }
                        default:
                            baked_ok = false;
                            break;
                        }

                        // Mark that this encoder has dispatched work (for barrier logic).
                        mixedNativeEncoderHasWork = true;

                        // Don't end encoder here — it persists for the next native step.
                        // Softmax/LayerNorm already inserted internal barriers.
                        // ScatterAddRows uses blit and needs its own encoder cycle;
                        // flush if we just dispatched one.
                        if (ns.baked_kind == NativeStep::BakedKind::kScatterAddRows) {
                            endMixedNativeEncoder();
                        }

                        if (!baked_ok) {
                            endMixedNativeEncoder();
                            return ExecutionResult::Error("Unsupported baked kind in mixed execution path");
                        }

                        slot_bufs[ns.output_slot] = slot_bufs[ns.output_slot];
                        slot_producer_events[(size_t)ns.output_slot] = cmd_completion;
                        continue;
                    }

                    // Non-baked handler creates its own encoder, so flush ours first.
                    endMixedNativeEncoder();

                    // Handlers may read input buffers on the CPU (e.g. the
                    // Cholesky LAPACK fallback). Commit and block until all
                    // pending GPU work completes so those host reads observe
                    // finished writes (same discipline as the baked CPU paths).
                    if (cmdBuf) {
                        [cmdBuf commit];
                        [cmdBuf waitUntilCompleted];
                        MarkCompletionFromCommandBuffer(cmd_completion, cmdBuf);
                        cmdBuf = [MPSCommandBuffer commandBufferFromCommandQueue:commandQueue];
                        if (!cmdBuf) {
                            return ExecutionResult::Error("Failed to recreate cmdBuf after CPU handler sync");
                        }
                        cmd_completion = MixedCompletionEvent(cmdBuf, async_completion);
                    }

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

            endMixedNativeEncoder();
            AttachDeferredCompletion(cmdBuf, cmd_completion, async_completion);
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

            if (!buf) {
                // Eliminated (constant-folded away) output slot: no buffer was
                // produced. Hand back a zeroed placeholder instead of wrapping
                // or releasing nil — CFRelease(nil) aborts the process.
                id<MTLBuffer> placeholder = [mtl_device newBufferWithLength:out_byte_size > 0
                                                                                ? out_byte_size
                                                                                : 4
                                                                    options:MTLResourceStorageModeShared];
                if (!placeholder) {
                    return ExecutionResult::Error(
                        "Failed to allocate placeholder buffer for eliminated output slot " +
                        std::to_string(slot));
                }
                result.buffers.push_back(std::make_unique<MpsBuffer>(device, (__bridge void*)placeholder,
                                                                     dtype, dims, MpsBuffer::BufferReleaser{},
                                                                     MakeReadyCompletionEvent()));
                CFRelease((__bridge CFTypeRef)placeholder);
                continue;
            }

            // Transpose aliases: create strided view — no copy needed.
            if (!meta.transpose_perm.empty() && buf) {
                size_t elem_size = DtypeByteSize(dtype);
                auto input_strides = ComputeRowMajorByteStrides(meta.transpose_input_dims, elem_size);
                std::vector<int64_t> transposed_strides(meta.transpose_perm.size());
                for (size_t d = 0; d < meta.transpose_perm.size(); ++d) {
                    transposed_strides[d] = input_strides[meta.transpose_perm[d]];
                }
                size_t storage_size = buf ? (size_t)[buf length] : out_byte_size;
                MpsBuffer::BufferReleaser rel;
                if (enable_output_pool) {
                    rel = [output_device_key, storage_size](void* metal_buffer) {
                        RecycleReusableOutputBuffer(output_device_key, storage_size, metal_buffer);
                    };
                }
                result.buffers.push_back(std::make_unique<MpsBuffer>(device, (__bridge void*)buf, dtype, dims,
                                                                     transposed_strides, storage_size,
                                                                     std::move(rel), slot_completion));
                CFRelease((__bridge CFTypeRef)buf);
                slot_bufs[slot] = nil;
                continue;
            }

            bool aliases_input = false;
            if (!slot_is_input[slot] && buf) {
                for (auto in_slot : plan_->input_slots) {
                    if (in_slot >= 0 && (size_t)in_slot < slot_bufs.size() && slot_bufs[in_slot] == buf) {
                        aliases_input = true;
                        break;
                    }
                }
            }
            if (slot_is_input[slot] || aliases_input) {
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

        // The intermediate buffer cache owns MTLBuffers that may still be in
        // flight on the GPU (in flight either because the direct-run command
        // buffer hasn't been observed to complete, or because the MPSGraph
        // framework holds internal references to the MPSGraphTensorData
        // wrappers). Releasing them at scope exit can corrupt an in-flight
        // command buffer and crash the process with "Invalid Resource".
        //
        // We always enqueue the cache onto a global release queue, which is
        // drained by a background thread. The background thread waits for the
        // completion event to fire (via Await()) before destroying the cache,
        // so the GPU is guaranteed to be idle when ReleaseAll() runs.
        //
        // Returning the cache to the pool is never safe because the next
        // Execute() would grab it and reuse buffers before the previous
        // command buffer's GPU work finishes.
        if (cache_set) {
            CompletionEvent fence = execution_completion ? execution_completion : MakeReadyCompletionEvent();
            EnqueueCacheForDelayedRelease(std::shared_ptr<IntermediateCacheSet>(cache_set.release()), fence);
        }

        result.completion = execution_completion ? execution_completion : MakeReadyCompletionEvent();
        return result;
    }
}

} // namespace jax_metallib
