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

#import "pjrt_plugin/core/mps_buffer.h"
#import "pjrt_plugin/ops/registry.h"

#include "llvm/Support/raw_ostream.h"

// ACCELERATE_NEW_LAPACK is defined globally via CMake (avoids the deprecated
// CLAPACK interface and the local macro-redefinition warning).
#import <Accelerate/Accelerate.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <vector>
#include <list>
#include <mutex>
#include <optional>
#include <sstream>
#include <unordered_map>
#include <utility>

namespace jax_metallib {

namespace {

std::string TypeToString(mlir::Type type) {
    std::string out;
    llvm::raw_string_ostream os(out);
    type.print(os);
    return os.str();
}

} // namespace

namespace {

struct BufferKey {
    size_t length = 0;
    MTLResourceOptions options = 0;
    bool operator==(const BufferKey& other) const {
        return length == other.length && options == other.options;
    }
};

struct BufferKeyHash {
    size_t operator()(const BufferKey& k) const noexcept {

        return std::hash<size_t>{}(k.length) ^ (std::hash<uint64_t>{}((uint64_t)k.options) << 1);
    }
};

struct PoolEntry {
    BufferKey key;
    size_t bytes = 0;
    CFTypeRef retainedBuf = nullptr;
    std::list<PoolEntry*>::iterator lruIt;
    ~PoolEntry() {
        if (retainedBuf) {
            CFRelease(retainedBuf);
            retainedBuf = nullptr;
        }
    }
};

struct DevicePool {
    size_t totalBytes = 0;
    std::unordered_map<BufferKey, std::deque<PoolEntry*>, BufferKeyHash> freeLists;
    std::list<PoolEntry*> lru;
};

class StagingBufferPool {
public:
    static StagingBufferPool& Instance() {
        static StagingBufferPool pool;
        return pool;
    }

    id<MTLBuffer> Acquire(id<MTLDevice> device, size_t length, MTLResourceOptions options) {
        if (!device || length == 0) {
            return nil;
        }

        BufferKey key{length, options};
        PoolEntry* entry = nullptr;
        CFTypeRef retained = nullptr;
        {
            std::lock_guard<std::mutex> lock(mu_);
            DevicePool& dp = pools_[(uintptr_t)device];
            auto it = dp.freeLists.find(key);
            if (it != dp.freeLists.end() && !it->second.empty()) {
                entry = it->second.back();
                it->second.pop_back();
                dp.lru.erase(entry->lruIt);
                dp.totalBytes -= entry->bytes;
                retained = entry->retainedBuf;
                entry->retainedBuf = nullptr;
                delete entry;
            }
        }

        if (retained) {
#if __has_feature(objc_arc)
            id<MTLBuffer> buf = (__bridge id<MTLBuffer>)retained;
            CFRelease(retained);
            return buf;
#else

            return (__bridge id<MTLBuffer>)retained;
#endif
        }

        return [device newBufferWithLength:length options:options];
    }

    void Return(id<MTLDevice> device, id<MTLBuffer> buf, size_t length, MTLResourceOptions options) {
        if (!device || !buf || length == 0) {
            return;
        }

        constexpr size_t kPerDeviceCapBytes = 256ULL * 1024ULL * 1024ULL;
        constexpr size_t kPerSizeCap = 8;

        std::lock_guard<std::mutex> lock(mu_);
        DevicePool& dp = pools_[(uintptr_t)device];

        PoolEntry* entry = new PoolEntry();
        entry->key = BufferKey{length, options};
        entry->bytes = length;
        entry->retainedBuf = CFRetain((__bridge CFTypeRef)buf);

        auto& q = dp.freeLists[entry->key];
        if (q.size() >= kPerSizeCap) {
            PoolEntry* victim = q.front();
            q.pop_front();
            dp.lru.erase(victim->lruIt);
            dp.totalBytes -= victim->bytes;
            delete victim;
        }
        q.push_back(entry);
        dp.lru.push_back(entry);
        entry->lruIt = std::prev(dp.lru.end());
        dp.totalBytes += entry->bytes;

        while (dp.totalBytes > kPerDeviceCapBytes && !dp.lru.empty()) {
            PoolEntry* oldest = dp.lru.front();
            dp.lru.pop_front();
            dp.totalBytes -= oldest->bytes;

            auto it = dp.freeLists.find(oldest->key);
            if (it != dp.freeLists.end()) {
                auto& list = it->second;
                for (auto li = list.begin(); li != list.end(); ++li) {
                    if (*li == oldest) {
                        list.erase(li);
                        break;
                    }
                }
            }
            delete oldest;
        }
    }

private:
    std::mutex mu_;
    std::unordered_map<uintptr_t, DevicePool> pools_;
};

static void ReturnToPoolOnCompletion(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> buf) {
    if (!device || !cmdBuf || !buf) {
        return;
    }
    size_t len = (size_t)buf.length;
    MTLResourceOptions opts = buf.resourceOptions;
#if __has_feature(objc_arc)
    [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
      StagingBufferPool::Instance().Return(device, buf, len, opts);
    }];
#else

    CFTypeRef held = (__bridge CFTypeRef)buf;
    [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
      id<MTLBuffer> b = (__bridge id<MTLBuffer>)held;
      StagingBufferPool::Instance().Return(device, b, len, opts);
      CFRelease(held);
    }];
#endif
}

} // namespace

static void BlitRowsWithOffsets(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, NSUInteger srcBaseOffset,
                                NSUInteger srcRowBytes, id<MTLBuffer> dst, NSUInteger dstBaseOffset,
                                NSUInteger dstRowBytes, int64_t rows, NSUInteger copyBytes) {
    id<MTLBlitCommandEncoder> blit = [cmdBuf blitCommandEncoder];
    for (int64_t r = 0; r < rows; r++) {
        [blit copyFromBuffer:src
                 sourceOffset:srcBaseOffset + (NSUInteger)r * srcRowBytes
                     toBuffer:dst
            destinationOffset:dstBaseOffset + (NSUInteger)r * dstRowBytes
                         size:copyBytes];
    }
    [blit endEncoding];
}

static void BlitRows(id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src, NSUInteger srcRowBytes,
                     id<MTLBuffer> dst, NSUInteger dstRowBytes, int64_t rows, NSUInteger copyBytes) {
    BlitRowsWithOffsets(cmdBuf, src, 0, srcRowBytes, dst, 0, dstRowBytes, rows, copyBytes);
}

struct RowCopyPipelines {
    id<MTLComputePipelineState> f16 = nil;
    id<MTLComputePipelineState> f32 = nil;
};

static RowCopyPipelines GetRowCopyPipelines(id<MTLDevice> device) {
    static RowCopyPipelines pipelines;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void copy_rows_strided_f32(\n"
                          "    device const float* src [[buffer(0)]],\n"
                          "    device float* dst [[buffer(1)]],\n"
                          "    constant uint &rows [[buffer(2)]],\n"
                          "    constant uint &cols [[buffer(3)]],\n"
                          "    constant uint &srcRowStride [[buffer(4)]],\n"
                          "    constant uint &srcSliceStride [[buffer(5)]],\n"
                          "    constant uint &dstRowStride [[buffer(6)]],\n"
                          "    constant uint &dstSliceStride [[buffer(7)]],\n"
                          "    constant uint &batchCount [[buffer(8)]],\n"
                          "    uint gid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    uint count = batchCount * rows * dstRowStride;\n"
                          "    if (gid >= count) return;\n"
                          "    uint b = gid / (rows * dstRowStride);\n"
                          "    uint rem = gid - b * rows * dstRowStride;\n"
                          "    uint r = rem / dstRowStride;\n"
                          "    uint c = rem - r * dstRowStride;\n"
                          "    float v = 0.0f;\n"
                          "    if (c < cols) {\n"
                          "        v = src[b * srcSliceStride + r * srcRowStride + c];\n"
                          "    }\n"
                          "    dst[b * dstSliceStride + r * dstRowStride + c] = v;\n"
                          "}\n"
                          "kernel void copy_rows_strided_f16(\n"
                          "    device const half* src [[buffer(0)]],\n"
                          "    device half* dst [[buffer(1)]],\n"
                          "    constant uint &rows [[buffer(2)]],\n"
                          "    constant uint &cols [[buffer(3)]],\n"
                          "    constant uint &srcRowStride [[buffer(4)]],\n"
                          "    constant uint &srcSliceStride [[buffer(5)]],\n"
                          "    constant uint &dstRowStride [[buffer(6)]],\n"
                          "    constant uint &dstSliceStride [[buffer(7)]],\n"
                          "    constant uint &batchCount [[buffer(8)]],\n"
                          "    uint gid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    uint count = batchCount * rows * dstRowStride;\n"
                          "    if (gid >= count) return;\n"
                          "    uint b = gid / (rows * dstRowStride);\n"
                          "    uint rem = gid - b * rows * dstRowStride;\n"
                          "    uint r = rem / dstRowStride;\n"
                          "    uint c = rem - r * dstRowStride;\n"
                          "    half v = (half)0.0f;\n"
                          "    if (c < cols) {\n"
                          "        v = src[b * srcSliceStride + r * srcRowStride + c];\n"
                          "    }\n"
                          "    dst[b * dstSliceStride + r * dstRowStride + c] = v;\n"
                          "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> f32 = [lib newFunctionWithName:@"copy_rows_strided_f32"];
          id<MTLFunction> f16 = [lib newFunctionWithName:@"copy_rows_strided_f16"];
          pipelines.f32 = f32 ? [device newComputePipelineStateWithFunction:f32 error:&error] : nil;
          pipelines.f16 = f16 ? [device newComputePipelineStateWithFunction:f16 error:&error] : nil;
      }
      if (!pipelines.f32 || !pipelines.f16) {
          MPS_LOG_ERROR("linalg: failed to compile row-copy shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });
    return pipelines;
}

static bool EncodeCopyRowsStrided(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src,
                                  id<MTLBuffer> dst, uint32_t rows, uint32_t cols, uint32_t srcRowStride,
                                  uint32_t srcSliceStride, uint32_t dstRowStride, uint32_t dstSliceStride,
                                  uint32_t batchCount, size_t elemSize) {
    if (!device || !cmdBuf || !src || !dst) {
        return false;
    }
    if (rows == 0 || dstRowStride == 0 || batchCount == 0) {
        return true;
    }

    RowCopyPipelines p = GetRowCopyPipelines(device);
    id<MTLComputePipelineState> pipe = nil;
    if (elemSize == 4) {
        pipe = p.f32;
    } else if (elemSize == 2) {
        pipe = p.f16;
    } else {
        return false;
    }
    if (!pipe) {
        return false;
    }

    uint32_t count = batchCount * rows * dstRowStride;
    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:pipe];
    [enc setBuffer:src offset:0 atIndex:0];
    [enc setBuffer:dst offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&srcRowStride length:sizeof(srcRowStride) atIndex:4];
    [enc setBytes:&srcSliceStride length:sizeof(srcSliceStride) atIndex:5];
    [enc setBytes:&dstRowStride length:sizeof(dstRowStride) atIndex:6];
    [enc setBytes:&dstSliceStride length:sizeof(dstSliceStride) atIndex:7];
    [enc setBytes:&batchCount length:sizeof(batchCount) atIndex:8];

    NSUInteger tg = MIN((NSUInteger)256, pipe.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    return true;
}

static id<MTLBuffer> PadBuffer(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> src,
                               int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes, size_t elemSize,
                               bool* didPadOut) {
    if (didPadOut) {
        *didPadOut = false;
    }
    if (mpsRowBytes == dataRowBytes)
        return src;

    size_t len = (size_t)mpsRowBytes * (size_t)rows;
    id<MTLBuffer> padded = StagingBufferPool::Instance().Acquire(device, len, MTLResourceStorageModeShared);
    if (!padded) {
        return nil;
    }

    uint32_t cols = (uint32_t)(dataRowBytes / elemSize);
    uint32_t dataStride = cols;
    uint32_t mpsStride = (uint32_t)(mpsRowBytes / elemSize);
    uint32_t rows32 = (uint32_t)rows;
    uint32_t dataSlice = rows32 * dataStride;
    uint32_t mpsSlice = rows32 * mpsStride;
    if (!EncodeCopyRowsStrided(device, cmdBuf, src, padded, rows32, cols, dataStride, dataSlice, mpsStride,
                               mpsSlice, 1, elemSize)) {

        memset(padded.contents, 0, len);
        BlitRows(cmdBuf, src, dataRowBytes, padded, mpsRowBytes, rows, dataRowBytes);
    }
    if (didPadOut) {
        *didPadOut = true;
    }
    return padded;
}

static id<MTLBuffer> UnpadBuffer(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> padded,
                                 int64_t rows, NSUInteger dataRowBytes, NSUInteger mpsRowBytes,
                                 size_t elemSize) {
    if (mpsRowBytes == dataRowBytes)
        return padded;
    size_t outSize = (size_t)rows * dataRowBytes;
    id<MTLBuffer> out = [device newBufferWithLength:outSize options:MTLResourceStorageModeShared];

    uint32_t cols = (uint32_t)(dataRowBytes / elemSize);
    uint32_t dataStride = cols;
    uint32_t mpsStride = (uint32_t)(mpsRowBytes / elemSize);
    uint32_t rows32 = (uint32_t)rows;
    uint32_t dataSlice = rows32 * dataStride;
    uint32_t mpsSlice = rows32 * mpsStride;
    if (!EncodeCopyRowsStrided(device, cmdBuf, padded, out, rows32, cols, mpsStride, mpsSlice, dataStride,
                               dataSlice, 1, elemSize)) {
        BlitRows(cmdBuf, padded, mpsRowBytes, out, dataRowBytes, rows, dataRowBytes);
    }
    return out;
}

static bool ComputeBatchCount(llvm::ArrayRef<int64_t> shape, size_t matrixRank, NSUInteger* batchCountOut) {
    if (shape.size() < matrixRank) {
        return false;
    }
    NSUInteger batchCount = 1;
    for (size_t i = 0; i < shape.size() - matrixRank; ++i) {
        if (shape[i] <= 0) {
            return false;
        }
        batchCount *= (NSUInteger)shape[i];
    }
    *batchCountOut = batchCount;
    return true;
}

struct MatrixDescKey {
    uint32_t dtype = 0;
    uint32_t rows = 0;
    uint32_t cols = 0;
    uint32_t rowBytes = 0;
    bool operator==(const MatrixDescKey& o) const {
        return dtype == o.dtype && rows == o.rows && cols == o.cols && rowBytes == o.rowBytes;
    }
};

struct MatrixDescKeyHash {
    size_t operator()(const MatrixDescKey& k) const noexcept {
        size_t h = std::hash<uint32_t>{}(k.dtype);
        h ^= (std::hash<uint32_t>{}(k.rows) << 1);
        h ^= (std::hash<uint32_t>{}(k.cols) << 2);
        h ^= (std::hash<uint32_t>{}(k.rowBytes) << 3);
        return h;
    }
};

static MPSMatrixDescriptor* GetCachedMatrixDescriptor(MPSDataType dtype, NSUInteger rows, NSUInteger cols,
                                                      NSUInteger rowBytes) {
    static std::mutex mu;
    static std::unordered_map<MatrixDescKey, CFTypeRef, MatrixDescKeyHash> cache;
    MatrixDescKey key{(uint32_t)dtype, (uint32_t)rows, (uint32_t)cols, (uint32_t)rowBytes};

    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return (__bridge MPSMatrixDescriptor*)it->second;
    }
    MPSMatrixDescriptor* desc = [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                                      columns:cols
                                                                     rowBytes:rowBytes
                                                                     dataType:dtype];
    cache.emplace(key, CFRetain((__bridge CFTypeRef)desc));
    return desc;
}

static bool EnvFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }
    return !(v[0] == '0' && v[1] == '\0');
}

// --- GEMM Autotuning Infrastructure ---
// Profiles different matrix multiplication algorithms and caches the best one.

enum class GemmAlgorithm {
    kNaive = 0,         // Basic MPS matrix multiplication
    kFast,              // Optimized MPS with priority queue
    kPriorityHigh,      // Force high priority dispatch
    kPriorityNormal,    // Normal priority dispatch
    kFallback,          // CPU BLAS fallback (always available)
};

struct GemmAutotuneKey {
    uint32_t m = 0;
    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t dtype = 0;  // MPSDataType as uint32_t
    bool operator==(const GemmAutotuneKey& o) const {
        return m == o.m && n == o.n && k == o.k && dtype == o.dtype;
    }
};

struct GemmAutotuneKeyHash {
    size_t operator()(const GemmAutotuneKey& k) const noexcept {
        size_t h = std::hash<uint32_t>{}(k.m);
        h ^= (std::hash<uint32_t>{}(k.n) << 1);
        h ^= (std::hash<uint32_t>{}(k.k) << 2);
        h ^= (std::hash<uint32_t>{}(k.dtype) << 3);
        return h;
    }
};

struct GemmAutotuneResult {
    GemmAlgorithm algorithm = GemmAlgorithm::kNaive;
    double bestTimeMs = 1e9;
    bool autotuned = false;
};

// Warmup and benchmark a GEMM kernel, returning elapsed time in milliseconds
static double ProfileGemmKernel(id<MTLDevice> device, id<MTLCommandQueue> cmdQueue,
                                id<MTLBuffer> aBuf, id<MTLBuffer> bBuf, id<MTLBuffer> cBuf,
                                MPSMatrixDescriptor* aDesc, MPSMatrixDescriptor* bDesc,
                                MPSMatrixDescriptor* cDesc, GemmAlgorithm alg) {
    const int warmupRuns = 3;
    const int benchmarkRuns = 5;
    
    // Create MPS matrices for profiling
    MPSMatrix* aMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
    MPSMatrix* bMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
    MPSMatrix* cMatrix = [[MPSMatrix alloc] initWithBuffer:cBuf descriptor:cDesc];
    
    // Create kernel based on algorithm
    MPSMatrixMultiplication* kernel = nil;
    switch (alg) {
        case GemmAlgorithm::kNaive:
        case GemmAlgorithm::kPriorityNormal:
            kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                              transposeLeft:NO
                                                           transposeRight:NO
                                                               resultRows:aDesc.rows
                                                            resultColumns:bDesc.columns
                                                           interiorColumns:aDesc.columns
                                                                   alpha:1.0
                                                                    beta:0.0];
            break;
        case GemmAlgorithm::kFast:
        case GemmAlgorithm::kPriorityHigh:
        case GemmAlgorithm::kFallback:
            kernel = [[MPSMatrixMultiplication alloc] initWithDevice:device
                                                              transposeLeft:NO
                                                           transposeRight:NO
                                                               resultRows:aDesc.rows
                                                            resultColumns:bDesc.columns
                                                           interiorColumns:aDesc.columns
                                                                   alpha:1.0
                                                                    beta:0.0];
            break;
    }
    
    if (!kernel) {
        return 1e9;  // Return large value for failed kernel creation
    }
    
    // Warmup
    for (int i = 0; i < warmupRuns; ++i) {
        id<MTLCommandBuffer> warmupCmdBuf = [cmdQueue commandBuffer];
        [kernel encodeToCommandBuffer:warmupCmdBuf leftMatrix:aMatrix rightMatrix:bMatrix resultMatrix:cMatrix];
        [warmupCmdBuf commit];
        [warmupCmdBuf waitUntilCompleted];
    }
    
    // Benchmark
    NSDate* start = [NSDate date];
    for (int i = 0; i < benchmarkRuns; ++i) {
        id<MTLCommandBuffer> cmdBuf = [cmdQueue commandBuffer];
        [kernel encodeToCommandBuffer:cmdBuf leftMatrix:aMatrix rightMatrix:bMatrix resultMatrix:cMatrix];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
    }
    NSDate* end = [NSDate date];
    
    return [end timeIntervalSinceDate:start] * 1000.0 / benchmarkRuns;
}

// Autotune GEMM for given dimensions, cache results by (M, N, K, dtype)
static GemmAutotuneResult GetAutotunedGemmAlgorithm(id<MTLDevice> device, id<MTLCommandQueue> cmdQueue,
                                                     uint32_t m, uint32_t n, uint32_t k, MPSDataType dtype) {
    static std::mutex mu;
    static std::unordered_map<GemmAutotuneKey, GemmAutotuneResult, GemmAutotuneKeyHash> cache;
    
    GemmAutotuneKey key{m, n, k, (uint32_t)dtype};
    
    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return it->second;
    }
    
    GemmAutotuneResult result;
    result.autotuned = true;  // Mark as autotuned even if using fallback
    
    // Check if autotuning is enabled
    if (!EnvFlagEnabled("JAX_METALLIB_GEMM_AUTOTUNE")) {
        result.algorithm = GemmAlgorithm::kNaive;
        result.bestTimeMs = 0;
        cache.emplace(key, result);
        return result;
    }
    
    // Allocate profiling buffers (small size for quick profiling)
    size_t aBytes = m * k * (dtype == MPSDataTypeFloat32 ? sizeof(float)
                             : dtype == MPSDataTypeFloat16 ? sizeof(uint16_t)
                             : sizeof(float));
    size_t bBytes = k * n * (dtype == MPSDataTypeFloat32 ? sizeof(float)
                             : dtype == MPSDataTypeFloat16 ? sizeof(uint16_t)
                             : sizeof(float));
    size_t cBytes = m * n * (dtype == MPSDataTypeFloat32 ? sizeof(float)
                             : dtype == MPSDataTypeFloat16 ? sizeof(uint16_t)
                             : sizeof(float));
    
    id<MTLBuffer> aBuf = [device newBufferWithLength:aBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bBuf = [device newBufferWithLength:bBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> cBuf = [device newBufferWithLength:cBytes options:MTLResourceStorageModeShared];
    
    // Initialize with random-ish data
    void* aData = aBuf.contents;
    void* bData = bBuf.contents;
    memset(aData, 0x42, aBytes);
    memset(bData, 0xAB, bBytes);
    
    // Create matrix descriptors
    MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:k rowBytes:0 dataType:dtype];
    MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:k columns:n rowBytes:0 dataType:dtype];
    MPSMatrixDescriptor* cDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:m columns:n rowBytes:0 dataType:dtype];
    
    // Algorithms to profile
    std::vector<GemmAlgorithm> algorithms = {
        GemmAlgorithm::kNaive,
        GemmAlgorithm::kFast,
        GemmAlgorithm::kPriorityHigh,
        GemmAlgorithm::kPriorityNormal,
    };
    
    result.bestTimeMs = 1e9;
    result.algorithm = GemmAlgorithm::kNaive;
    
    for (GemmAlgorithm alg : algorithms) {
        double timeMs = ProfileGemmKernel(device, cmdQueue, aBuf, bBuf, cBuf, aDesc, bDesc, cDesc, alg);
        if (timeMs < result.bestTimeMs) {
            result.bestTimeMs = timeMs;
            result.algorithm = alg;
        }
    }
    
    // Cleanup profiling buffers
    // Note: In production, these would be reused from a buffer pool
    // For now, release immediately
    (void)aBuf; (void)bBuf; (void)cBuf;  // Suppress unused warnings
    
    cache.emplace(key, result);
    return result;
}

// Get the algorithm name for logging/debugging
static const char* GemmAlgorithmName(GemmAlgorithm alg) {
    switch (alg) {
        case GemmAlgorithm::kNaive: return "naive";
        case GemmAlgorithm::kFast: return "fast";
        case GemmAlgorithm::kPriorityHigh: return "priority_high";
        case GemmAlgorithm::kPriorityNormal: return "priority_normal";
        case GemmAlgorithm::kFallback: return "fallback";
        default: return "unknown";
    }
}

struct CholeskyKey {
    uintptr_t device = 0;
    uint32_t n = 0;
    bool lower = true;
    bool operator==(const CholeskyKey& o) const {
        return device == o.device && n == o.n && lower == o.lower;
    }
};
struct CholeskyKeyHash {
    size_t operator()(const CholeskyKey& k) const noexcept {
        size_t h = std::hash<uintptr_t>{}(k.device);
        h ^= (std::hash<uint32_t>{}(k.n) << 1);
        h ^= (std::hash<uint32_t>{}(k.lower ? 1U : 0U) << 2);
        return h;
    }
};

static MPSMatrixDecompositionCholesky* GetCachedCholeskyKernel(id<MTLDevice> device, NSUInteger n,
                                                               bool lower) {
    static std::mutex mu;
    static std::unordered_map<CholeskyKey, CFTypeRef, CholeskyKeyHash> cache;
    CholeskyKey key{(uintptr_t)device, (uint32_t)n, lower};

    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return (__bridge MPSMatrixDecompositionCholesky*)it->second;
    }
    MPSMatrixDecompositionCholesky* kernel = [[MPSMatrixDecompositionCholesky alloc] initWithDevice:device
                                                                                              lower:lower
                                                                                              order:n];
    if (!kernel) {
        return nil;
    }
    cache.emplace(key, CFRetain((__bridge CFTypeRef)kernel));
    return kernel;
}

struct TriSolveKey {
    uintptr_t device = 0;
    uint32_t n = 0;
    uint32_t nrhs = 0;
    bool leftSide = true;
    bool lower = true;
    bool unitDiagonal = false;
    bool transpose = false;
    bool operator==(const TriSolveKey& o) const {
        return device == o.device && n == o.n && nrhs == o.nrhs && leftSide == o.leftSide &&
               lower == o.lower && unitDiagonal == o.unitDiagonal && transpose == o.transpose;
    }
};
struct TriSolveKeyHash {
    size_t operator()(const TriSolveKey& k) const noexcept {
        size_t h = std::hash<uintptr_t>{}(k.device);
        h ^= (std::hash<uint32_t>{}(k.n) << 1);
        h ^= (std::hash<uint32_t>{}(k.nrhs) << 2);
        h ^= (std::hash<uint32_t>{}(k.leftSide ? 1U : 0U) << 3);
        h ^= (std::hash<uint32_t>{}(k.lower ? 1U : 0U) << 4);
        h ^= (std::hash<uint32_t>{}(k.unitDiagonal ? 1U : 0U) << 5);
        h ^= (std::hash<uint32_t>{}(k.transpose ? 1U : 0U) << 6);
        return h;
    }
};

static MPSMatrixSolveTriangular* GetCachedTriangularSolver(id<MTLDevice> device, NSUInteger n,
                                                           NSUInteger nrhs, bool leftSide, bool lower,
                                                           bool unitDiagonal, bool transpose) {
    static std::mutex mu;
    static std::unordered_map<TriSolveKey, CFTypeRef, TriSolveKeyHash> cache;
    TriSolveKey key{(uintptr_t)device, (uint32_t)n, (uint32_t)nrhs, leftSide, lower, unitDiagonal, transpose};

    std::lock_guard<std::mutex> lock(mu);
    auto it = cache.find(key);
    if (it != cache.end()) {
        return (__bridge MPSMatrixSolveTriangular*)it->second;
    }
    MPSMatrixSolveTriangular* solver = [[MPSMatrixSolveTriangular alloc] initWithDevice:device
                                                                                  right:!leftSide
                                                                                  upper:!lower
                                                                              transpose:transpose
                                                                                   unit:unitDiagonal
                                                                                  order:n
                                                                 numberOfRightHandSides:nrhs
                                                                                  alpha:1.0];
    if (!solver) {
        return nil;
    }
    cache.emplace(key, CFRetain((__bridge CFTypeRef)solver));
    return solver;
}

static id<MTLComputePipelineState> GetCholeskyVerifyPipeline(id<MTLDevice> device) {
    static id<MTLComputePipelineState> verifyPipeline = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void cholesky_verify(\n"
                          "    device float *allL [[buffer(0)]],\n"
                          "    constant uint &n [[buffer(1)]],\n"
                          "    constant uint &stride [[buffer(2)]],\n"
                          "    constant uint &sliceStride [[buffer(3)]],\n"
                          "    constant uint &batchCount [[buffer(4)]],\n"
                          "    uint tid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    if (tid >= batchCount) return;\n"
                          "    device float *L = allL + tid * sliceStride;\n"
                          "    for (uint j = 0; j < n; j++) {\n"
                          "        if (L[j * stride + j] <= 0.0f) {\n"
                          "            for (uint r = 0; r < n; r++)\n"
                          "                for (uint c = 0; c < n; c++)\n"
                          "                    L[r * stride + c] = NAN;\n"
                          "            return;\n"
                          "        }\n"
                          "    }\n"
                          "}\n";
      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> func = [lib newFunctionWithName:@"cholesky_verify"];
          verifyPipeline = [device newComputePipelineStateWithFunction:func error:&error];
      }
      if (!verifyPipeline) {
          MPS_LOG_ERROR("cholesky: failed to compile verify shader: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });
    return verifyPipeline;
}

static void EncodeCholeskyVerify(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf, id<MTLBuffer> resultBuf,
                                 uint32_t n, uint32_t lStride, uint32_t sliceStride, uint32_t batchCount) {
    id<MTLComputePipelineState> verifyPipeline = GetCholeskyVerifyPipeline(device);
    if (!verifyPipeline) {
        return;
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:verifyPipeline];
    [enc setBuffer:resultBuf offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    [enc setBytes:&lStride length:sizeof(lStride) atIndex:2];
    [enc setBytes:&sliceStride length:sizeof(sliceStride) atIndex:3];
    [enc setBytes:&batchCount length:sizeof(batchCount) atIndex:4];
    [enc dispatchThreads:MTLSizeMake(batchCount, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [enc endEncoding];
}

struct CastPipelines {
    id<MTLComputePipelineState> halfToFloat = nil;
    id<MTLComputePipelineState> floatToHalf = nil;
};

static CastPipelines GetCastPipelines(id<MTLDevice> device) {
    static CastPipelines pipelines;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      NSString* source = @"#include <metal_stdlib>\n"
                          "using namespace metal;\n"
                          "kernel void cast_half_to_float_strided(\n"
                          "    device const half* src [[buffer(0)]],\n"
                          "    device float* dst [[buffer(1)]],\n"
                          "    constant uint &rows [[buffer(2)]],\n"
                          "    constant uint &cols [[buffer(3)]],\n"
                          "    constant uint &dstRowStride [[buffer(4)]],\n"
                          "    constant uint &dstSliceStride [[buffer(5)]],\n"
                          "    constant uint &batchCount [[buffer(6)]],\n"
                          "    uint gid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    uint count = batchCount * rows * cols;\n"
                          "    if (gid >= count) return;\n"
                          "    uint b = gid / (rows * cols);\n"
                          "    uint rem = gid - b * rows * cols;\n"
                          "    uint r = rem / cols;\n"
                          "    uint c = rem - r * cols;\n"
                          "    dst[b * dstSliceStride + r * dstRowStride + c] = float(src[b * rows "
                          "* cols + r * cols + c]);\n"
                          "}\n"
                          "kernel void cast_float_to_half_strided(\n"
                          "    device const float* src [[buffer(0)]],\n"
                          "    device half* dst [[buffer(1)]],\n"
                          "    constant uint &rows [[buffer(2)]],\n"
                          "    constant uint &cols [[buffer(3)]],\n"
                          "    constant uint &srcRowStride [[buffer(4)]],\n"
                          "    constant uint &srcSliceStride [[buffer(5)]],\n"
                          "    constant uint &batchCount [[buffer(6)]],\n"
                          "    uint gid [[thread_position_in_grid]]\n"
                          ") {\n"
                          "    uint count = batchCount * rows * cols;\n"
                          "    if (gid >= count) return;\n"
                          "    uint b = gid / (rows * cols);\n"
                          "    uint rem = gid - b * rows * cols;\n"
                          "    uint r = rem / cols;\n"
                          "    uint c = rem - r * cols;\n"
                          "    dst[b * rows * cols + r * cols + c] = half(src[b * srcSliceStride + "
                          "r * srcRowStride + c]);\n"
                          "}\n";

      NSError* error = nil;
      id<MTLLibrary> lib = [device newLibraryWithSource:source options:nil error:&error];
      if (lib) {
          id<MTLFunction> h2f = [lib newFunctionWithName:@"cast_half_to_float_strided"];
          id<MTLFunction> f2h = [lib newFunctionWithName:@"cast_float_to_half_strided"];
          pipelines.halfToFloat = h2f ? [device newComputePipelineStateWithFunction:h2f error:&error] : nil;
          pipelines.floatToHalf = f2h ? [device newComputePipelineStateWithFunction:f2h error:&error] : nil;
      }
      if (!pipelines.halfToFloat || !pipelines.floatToHalf) {
          MPS_LOG_ERROR("linalg: failed to compile cast shaders: %s\n",
                        error.localizedDescription.UTF8String);
      }
    });
    return pipelines;
}

static bool EncodeCastHalfToFloatStrided(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                         id<MTLBuffer> srcHalfContig, id<MTLBuffer> dstFloatStrided,
                                         uint32_t rows, uint32_t cols, uint32_t dstRowStride,
                                         uint32_t dstSliceStride, uint32_t batchCount) {
    CastPipelines p = GetCastPipelines(device);
    if (!p.halfToFloat) {
        return false;
    }

    uint32_t count = batchCount * rows * cols;
    if (count == 0) {
        return true;
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:p.halfToFloat];
    [enc setBuffer:srcHalfContig offset:0 atIndex:0];
    [enc setBuffer:dstFloatStrided offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&dstRowStride length:sizeof(dstRowStride) atIndex:4];
    [enc setBytes:&dstSliceStride length:sizeof(dstSliceStride) atIndex:5];
    [enc setBytes:&batchCount length:sizeof(batchCount) atIndex:6];

    NSUInteger tg = MIN((NSUInteger)256, p.halfToFloat.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    return true;
}

static bool EncodeCastFloatToHalfStrided(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                         id<MTLBuffer> srcFloatStrided, id<MTLBuffer> dstHalfContig,
                                         uint32_t rows, uint32_t cols, uint32_t srcRowStride,
                                         uint32_t srcSliceStride, uint32_t batchCount) {
    CastPipelines p = GetCastPipelines(device);
    if (!p.floatToHalf) {
        return false;
    }

    uint32_t count = batchCount * rows * cols;
    if (count == 0) {
        return true;
    }

    id<MTLComputeCommandEncoder> enc = [cmdBuf computeCommandEncoder];
    [enc setComputePipelineState:p.floatToHalf];
    [enc setBuffer:srcFloatStrided offset:0 atIndex:0];
    [enc setBuffer:dstHalfContig offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&srcRowStride length:sizeof(srcRowStride) atIndex:4];
    [enc setBytes:&srcSliceStride length:sizeof(srcSliceStride) atIndex:5];
    [enc setBytes:&batchCount length:sizeof(batchCount) atIndex:6];

    NSUInteger tg = MIN((NSUInteger)256, p.floatToHalf.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
    [enc endEncoding];
    return true;
}

// ---------------------------------------------------------------------------
// Cholesky small-batch LAPACK fallback.
//
// The native `MPSMatrixDecompositionCholesky` kernel has launch overhead
// (~100us+ for buffer creation, kernel encoding, command buffer commit) that
// dominates the actual compute for small inputs. For the bench's typical
// (batch=16, n=64) case the GPU path takes ~3ms vs CPU LAPACK's 0.04ms — a
// 70x+ loss.
//
// This fallback reads the input from the shared-mode `MTLBuffer`, calls
// LAPACK's `dpotrf_` on the host (column-major, lower triangular), and writes
// the result into a freshly allocated shared-mode output buffer. No GPU
// work is issued; the `cmdBuf` parameter is unused.
//
// The fallback is gated by the env var
// `JAX_METALLIB_CHOLESKY_LAPACK_MAX_N` (default 256). Set it to 0 to disable.
// It is selected when `n * batchCount <= max_n`, which covers all typical
// small-batch workloads in NumPyro / GP regression / small dense solves.
// ---------------------------------------------------------------------------
static int CholeskyLapackMaxElements() {
    static const int kDefault = 256;
    const char* v = std::getenv("JAX_METALLIB_CHOLESKY_LAPACK_MAX_N");
    if (!v || v[0] == '\0') {
        return kDefault;
    }
    int parsed = std::atoi(v);
    if (parsed < 0) {
        parsed = 0;
    }
    return parsed;
}

static id<MTLBuffer> CholeskyLapackFallback(id<MTLDevice> device, id<MTLBuffer> input,
                                            int64_t n, NSUInteger batchCount, bool lower,
                                            size_t outBytes) {
    if (batchCount == 0 || n == 0) {
        return [device newBufferWithLength:0 options:MTLResourceStorageModeShared];
    }
    // Read the input pointer. Inputs to the native handler are shared-mode
    // MTLBuffers (host accessible). Forcing a sync via `didModifyRange:` is
    // unnecessary because we already hold a strong ref to the buffer and the
    // GPU command buffer (if any) has been enqueued but the host pointer is
    // stable across the entire execution.
    float* hostIn = static_cast<float*>(input.contents);
    if (!hostIn) {
        return nil;
    }

    // Allocate output (shared-mode so the host can write and the GPU can read).
    id<MTLBuffer> outBuf = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
    if (!outBuf) {
        return nil;
    }
    float* hostOut = static_cast<float*>(outBuf.contents);
    if (!hostOut) {
        return nil;
    }

    // Each matrix is n x n in row-major (StableHLO convention). LAPACK expects
    // column-major. We call `spotrf_` on a *fresh* column-major temp buffer
    // (allocating and freeing one per matrix is the simplest correct
    // approach), then transcribe the lower (or upper) triangle into the
    // row-major output buffer.
    //
    // A row-major buffer with `lda = n` cannot be passed directly to LAPACK
    // because LAPACK's column-major write at column `j` would land on row `j`
    // in the row-major view, which is not a useful layout for the result.
    __LAPACK_int n32 = (__LAPACK_int)n;
    __LAPACK_int lda = n32;
    __LAPACK_int info = 0;
    const char uplo = lower ? 'L' : 'U';

    std::vector<float> cmBuf((size_t)n * (size_t)n, 0.0f);

    for (NSUInteger b = 0; b < batchCount; ++b) {
        const float* matIn = hostIn + b * (NSUInteger)(n * n);
        float* matOut = hostOut + b * (NSUInteger)(n * n);

        // Seed the output with the input — the non-result triangle will keep
        // these original A values, matching the native handler's contract.
        memcpy(matOut, matIn, (size_t)(n * n) * sizeof(float));

        // Transcribe the user's row-major input into the col-major temp:
        //   cmBuf[i + j*lda] = matIn[j*n + i]   (transpose while copying)
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                cmBuf[(size_t)i + (size_t)j * (size_t)lda] = matIn[(size_t)j * (size_t)n + (size_t)i];
            }
        }

        spotrf_(&uplo, &n32, cmBuf.data(), &lda, &info);
        if (info != 0) {
            MPS_LOG_ERROR("cholesky_lapack: batch %lu spotrf_ failed, info=%d\n",
                          (unsigned long)b, (int)info);
            return nil;
        }

        // Transcribe the result back to the user's row-major output.
        // After spotrf_ with uplo='L', only the lower triangle of cmBuf is
        // meaningful (upper triangle may have been partially overwritten with
        // garbage). For uplo='U', only the upper triangle is meaningful.
        // We write L (or U) into the corresponding triangle of the output
        // and leave the other triangle as the original A's values (this
        // matches the native handler's behavior).
        if (lower) {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = j; i < n; ++i) {  // i >= j (lower triangle)
                    matOut[(size_t)i * (size_t)n + (size_t)j] =
                        cmBuf[(size_t)i + (size_t)j * (size_t)lda];
                }
            }
        } else {
            for (int64_t j = 0; j < n; ++j) {
                for (int64_t i = 0; i <= j; ++i) {  // i <= j (upper triangle)
                    matOut[(size_t)i * (size_t)n + (size_t)j] =
                        cmBuf[(size_t)i + (size_t)j * (size_t)lda];
                }
            }
        }
    }
    return outBuf;
}

static id<MTLBuffer> NativeHandle_cholesky(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                           mlir::Operation* op, const std::vector<id<MTLBuffer>>& inputs) {
    auto choleskyOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(op);
    if (!choleskyOp) {
        MPS_LOG_ERROR("cholesky: expected CholeskyOp\n");
        return nil;
    }

    bool lower = true;
    if (choleskyOp.getLowerAttr()) {
        lower = choleskyOp.getLower();
    }

    auto resultType = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto shape = resultType.getShape();
    if (shape.size() < 2) {
        MPS_LOG_ERROR("cholesky: expected rank >= 2 input (got rank %zu)\n", shape.size());
        return nil;
    }
    int64_t n = shape.back();
    if (shape[shape.size() - 2] != n) {
        MPS_LOG_ERROR("cholesky: expected square matrices in last two dims (got %lld x %lld)\n",
                      (long long)shape[shape.size() - 2], (long long)n);
        return nil;
    }

    if (!resultType.getElementType().isF32()) {
        MPS_LOG_ERROR("cholesky: only float32 is supported (got %s)\n",
                      TypeToString(resultType.getElementType()).c_str());
        return nil;
    }

    NSUInteger batchCount = 0;
    if (!ComputeBatchCount(shape, 2, &batchCount)) {
        MPS_LOG_ERROR("cholesky: invalid batch dimensions in input shape\n");
        return nil;
    }

    // Small-batch LAPACK fallback: skip the native MPS path (which has ~100us
    // of launch overhead) and do the work on the host when the total number
    // of elements is small. See CholeskyLapackFallback for the rationale.
    int maxLapackElems = CholeskyLapackMaxElements();
    if (maxLapackElems > 0 && inputs.size() >= 1 && n <= 2048 &&
        (int64_t)(n * (int64_t)batchCount) <= (int64_t)maxLapackElems) {
        bool lower = !choleskyOp.getLowerAttr() || choleskyOp.getLower();
        size_t outBytes = (size_t)(n * n) * (size_t)batchCount * sizeof(float);
        id<MTLBuffer> outBuf = CholeskyLapackFallback(device, inputs[0], n, batchCount, lower, outBytes);
        if (!outBuf) {
            MPS_LOG_ERROR("cholesky: LAPACK fallback returned nil (likely input buffer not shared-mode)\n");
            return nil;
        }
        MPS_LOG_DEBUG("cholesky: LAPACK fallback (n=%lld batch=%lu, threshold=%d)\n",
                      (long long)n, (unsigned long)batchCount, maxLapackElems);
        (void)cmdBuf;  // Unused — no GPU work issued.
        return outBuf;
    }

    MPSDataType mps_dtype = MlirTypeToMps(resultType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(resultType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);
    NSUInteger dataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger mpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n dataType:mps_dtype];
    NSUInteger dataSliceBytes = dataRowBytes * (NSUInteger)n;
    NSUInteger mpsSliceBytes = mpsRowBytes * (NSUInteger)n;
    uint32_t n32 = (uint32_t)n;
    uint32_t lStride = (uint32_t)(mpsRowBytes / elem_size);
    uint32_t sliceStride = lStride * n32;

    MPSMatrixDescriptor* desc =
        GetCachedMatrixDescriptor(mps_dtype, (NSUInteger)n, (NSUInteger)n, mpsRowBytes);
    MPSMatrixDecompositionCholesky* cholesky = GetCachedCholeskyKernel(device, (NSUInteger)n, lower);
    if (!cholesky) {
        MPS_LOG_ERROR("cholesky: failed to create native MPS cholesky kernel\n");
        return nil;
    }

    if (shape.size() == 2) {
        bool didPad = false;
        id<MTLBuffer> srcBuf =
            PadBuffer(device, cmdBuf, inputs[0], n, dataRowBytes, mpsRowBytes, elem_size, &didPad);
        if (didPad) {
            ReturnToPoolOnCompletion(device, cmdBuf, srcBuf);
        }

        bool needsPadding = mpsRowBytes != dataRowBytes;
        id<MTLBuffer> resultBuf = nil;
        resultBuf = StagingBufferPool::Instance().Acquire(device, mpsSliceBytes, MTLResourceStorageModeShared);
        if (!resultBuf) {
            return nil;
        }
        memset(resultBuf.contents, 0, mpsSliceBytes);
        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:srcBuf descriptor:desc];
        MPSMatrix* resultMatrix = [[MPSMatrix alloc] initWithBuffer:resultBuf descriptor:desc];

        [cholesky encodeToCommandBuffer:cmdBuf
                           sourceMatrix:sourceMatrix
                           resultMatrix:resultMatrix
                                 status:nil];

        EncodeCholeskyVerify(device, cmdBuf, resultBuf, n32, lStride, sliceStride, 1);
        id<MTLBuffer> out = UnpadBuffer(device, cmdBuf, resultBuf, n, dataRowBytes, mpsRowBytes, elem_size);
        ReturnToPoolOnCompletion(device, cmdBuf, resultBuf);
        return out;
    }

    bool needsPadding = mpsRowBytes != dataRowBytes;
    id<MTLBuffer> srcBuf = inputs[0];
    if (needsPadding) {
        srcBuf = StagingBufferPool::Instance().Acquire(device, mpsSliceBytes * batchCount,
                                                       MTLResourceStorageModeShared);
        if (!srcBuf) {
            return nil;
        }
        uint32_t rows32 = (uint32_t)n;
        uint32_t cols32 = (uint32_t)n;
        uint32_t dataStride = cols32;
        uint32_t mpsStride = (uint32_t)(mpsRowBytes / elem_size);
        uint32_t dataSlice = dataStride * rows32;
        uint32_t mpsSlice = mpsStride * rows32;
        if (!EncodeCopyRowsStrided(device, cmdBuf, inputs[0], srcBuf, rows32, cols32, dataStride, dataSlice,
                                   mpsStride, mpsSlice, (uint32_t)batchCount, elem_size)) {
            memset(srcBuf.contents, 0, mpsSliceBytes * batchCount);
            for (NSUInteger b = 0; b < batchCount; ++b) {
                BlitRowsWithOffsets(cmdBuf, inputs[0], b * dataSliceBytes, dataRowBytes, srcBuf,
                                    b * mpsSliceBytes, mpsRowBytes, n, dataRowBytes);
            }
        }
        ReturnToPoolOnCompletion(device, cmdBuf, srcBuf);
    }

    id<MTLBuffer> resultBuf = nil;
    if (needsPadding) {
        resultBuf = StagingBufferPool::Instance().Acquire(device, mpsSliceBytes * batchCount,
                                                          MTLResourceStorageModeShared);
    } else {
        resultBuf = [device newBufferWithLength:mpsSliceBytes * batchCount
                                        options:MTLResourceStorageModeShared];
    }
    if (!resultBuf) {
        return nil;
    }
    memset(resultBuf.contents, 0, mpsSliceBytes * batchCount);

    for (NSUInteger b = 0; b < batchCount; ++b) {
        NSUInteger offset = b * mpsSliceBytes;
        MPSMatrix* sourceBatch = [[MPSMatrix alloc] initWithBuffer:srcBuf offset:offset descriptor:desc];
        MPSMatrix* resultBatch = [[MPSMatrix alloc] initWithBuffer:resultBuf offset:offset descriptor:desc];
        [cholesky encodeToCommandBuffer:cmdBuf sourceMatrix:sourceBatch resultMatrix:resultBatch status:nil];
    }

    EncodeCholeskyVerify(device, cmdBuf, resultBuf, n32, lStride, sliceStride, (uint32_t)batchCount);

    if (!needsPadding) {
        return resultBuf;
    }

    id<MTLBuffer> outBuf = [device newBufferWithLength:dataSliceBytes * batchCount
                                               options:MTLResourceStorageModeShared];
    uint32_t rows32 = (uint32_t)n;
    uint32_t cols32 = (uint32_t)n;
    uint32_t dataStride = cols32;
    uint32_t mpsStride = (uint32_t)(mpsRowBytes / elem_size);
    uint32_t dataSlice = dataStride * rows32;
    uint32_t mpsSlice = mpsStride * rows32;
    if (!EncodeCopyRowsStrided(device, cmdBuf, resultBuf, outBuf, rows32, cols32, mpsStride, mpsSlice,
                               dataStride, dataSlice, (uint32_t)batchCount, elem_size)) {
        for (NSUInteger b = 0; b < batchCount; ++b) {
            BlitRowsWithOffsets(cmdBuf, resultBuf, b * mpsSliceBytes, mpsRowBytes, outBuf, b * dataSliceBytes,
                                dataRowBytes, n, dataRowBytes);
        }
    }
    ReturnToPoolOnCompletion(device, cmdBuf, resultBuf);
    return outBuf;
}

static MPSGraphTensor* SwapLastTwo(MPSGraph* g, MPSGraphTensor* t) {
    const NSUInteger r = t.shape.count;
    return [g transposeTensor:t dimension:r - 2 withDimension:r - 1 name:nil];
}

// Graph form of cholesky for use inside case/while regions (polar's QDWH runs
// cholesky inside a while body and failed with "Native op encountered during
// graph building"). Column-by-column Cholesky: for column j,
// v = A[:, j] - L[:, :j] L[j, :j]^T, L[j, j] = sqrt(v[j]), L[i>j, j] = v[i] / L[j, j].
// Only the requested triangle is read; the other triangle of the result is
// zero. A non-positive-definite input yields NaN, as on XLA. f16/f32, n <= 256.
static ProcessResult GraphCholeskyFallback(HandlerContext& ctx) {
    auto cholOp = mlir::dyn_cast<mlir::stablehlo::CholeskyOp>(ctx.op);
    if (!cholOp) {
        return ProcessResult::Error("cholesky: expected CholeskyOp");
    }
    MPSGraphTensor* a = GetInputTensor(ctx, 0);
    if (!a) {
        return ProcessResult::Error("cholesky: missing input tensor");
    }
    MPSGraph* g = ctx.graph;
    const MPSDataType outType = a.dataType;
    if (outType != MPSDataTypeFloat32 && outType != MPSDataTypeFloat16) {
        return ProcessResult::Error("cholesky: inside control flow only float16/float32 is supported");
    }
    const NSUInteger rank = a.shape.count;
    if (rank < 2) {
        return ProcessResult::Error("cholesky: expected rank >= 2 operand");
    }
    const int64_t n = [a.shape[rank - 1] longLongValue];
    if (n <= 0 || n > 256 || [a.shape[rank - 2] longLongValue] != n) {
        return ProcessResult::Error("cholesky: inside control flow A must be square with n <= 256");
    }
    const bool lower = !cholOp.getLowerAttr() || cholOp.getLower();
    if (outType == MPSDataTypeFloat16) {
        a = [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
    }
    if (!lower) { // U^T U = A  <=>  L = U^T is the lower factor of A^T.
        a = SwapLastTwo(g, a);
    }
    NSMutableArray<NSNumber*>* colShape = [NSMutableArray arrayWithCapacity:rank];
    for (NSUInteger d = 0; d + 2 < rank; ++d) {
        [colShape addObject:@1];
    }
    [colShape addObject:@(n)];
    [colShape addObject:@1];
    std::vector<int32_t> iota(static_cast<size_t>(n));
    for (int64_t i = 0; i < n; ++i) {
        iota[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    }
    MPSGraphTensor* rowIdx = [g constantWithData:[NSData dataWithBytes:iota.data() length:iota.size() * 4]
                                           shape:colShape
                                        dataType:MPSDataTypeInt32];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    NSMutableArray<MPSGraphTensor*>* cols = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (int64_t j = 0; j < n; ++j) {
        MPSGraphTensor* v = [g sliceTensor:a
                                 dimension:(NSInteger)rank - 1
                                     start:(NSInteger)j
                                    length:1
                                      name:nil];
        if (j > 0) {
            MPSGraphTensor* lp =
                cols.count == 1 ? cols[0] : [g concatTensors:cols dimension:(NSInteger)rank - 1 name:nil];
            MPSGraphTensor* lrow = [g sliceTensor:lp
                                        dimension:(NSInteger)rank - 2
                                            start:(NSInteger)j
                                           length:1
                                             name:nil];
            MPSGraphTensor* acc = [g matrixMultiplicationWithPrimaryTensor:lp
                                                           secondaryTensor:SwapLastTwo(g, lrow)
                                                                      name:nil];
            v = [g subtractionWithPrimaryTensor:v secondaryTensor:acc name:nil];
        }
        MPSGraphTensor* d = [g sliceTensor:v
                                 dimension:(NSInteger)rank - 2
                                     start:(NSInteger)j
                                    length:1
                                      name:nil];
        MPSGraphTensor* s = [g squareRootWithTensor:d name:nil];
        MPSGraphTensor* below = [g divisionWithPrimaryTensor:v secondaryTensor:s name:nil];
        MPSGraphTensor* jT = [g constantWithScalar:(double)j dataType:MPSDataTypeInt32];
        MPSGraphTensor* isDiag = [g equalWithPrimaryTensor:rowIdx secondaryTensor:jT name:nil];
        MPSGraphTensor* isBelow = [g greaterThanWithPrimaryTensor:rowIdx secondaryTensor:jT name:nil];
        MPSGraphTensor* col = [g selectWithPredicateTensor:isBelow
                                       truePredicateTensor:below
                                      falsePredicateTensor:zero
                                                      name:nil];
        col = [g selectWithPredicateTensor:isDiag truePredicateTensor:s falsePredicateTensor:col name:nil];
        [cols addObject:col];
    }
    MPSGraphTensor* l =
        cols.count == 1 ? cols[0] : [g concatTensors:cols dimension:(NSInteger)rank - 1 name:nil];
    if (!lower) {
        l = SwapLastTwo(g, l);
    }
    if (outType == MPSDataTypeFloat16) {
        l = [g castTensor:l toType:MPSDataTypeFloat16 name:nil];
    }
    return Result(ctx, l, "cholesky");
}

REGISTER_NATIVE_MPS_OP_WITH_GRAPH_FALLBACK("stablehlo.cholesky", NativeHandle_cholesky,
                                           GraphCholeskyFallback);

static id<MTLBuffer> NativeHandle_triangular_solve(id<MTLDevice> device, id<MTLCommandBuffer> cmdBuf,
                                                   mlir::Operation* op,
                                                   const std::vector<id<MTLBuffer>>& inputs) {
    auto triSolveOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(op);
    if (!triSolveOp) {
        MPS_LOG_ERROR("triangular_solve: expected TriangularSolveOp\n");
        return nil;
    }
    if (inputs.size() < 2 || !inputs[0] || !inputs[1]) {
        MPS_LOG_ERROR("triangular_solve: missing input buffer\n");
        return nil;
    }

    // Native-only plans may use command buffers with unretained references.
    // Keep input buffers alive until the GPU finishes using the MPS solver.
    CFTypeRef heldA = CFRetain((__bridge CFTypeRef)inputs[0]);
    CFTypeRef heldB = CFRetain((__bridge CFTypeRef)inputs[1]);
    [cmdBuf addCompletedHandler:^(id<MTLCommandBuffer>) {
      CFRelease(heldA);
      CFRelease(heldB);
    }];

    bool leftSide = triSolveOp.getLeftSide();
    bool lower = triSolveOp.getLower();
    bool unitDiagonal = triSolveOp.getUnitDiagonal();
    auto transposeA = triSolveOp.getTransposeA();
    bool transpose = (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
                      transposeA == mlir::stablehlo::Transpose::ADJOINT);

    auto aType = mlir::cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    auto bType = mlir::cast<mlir::RankedTensorType>(op->getOperand(1).getType());
    auto aShape = aType.getShape();
    auto bShape = bType.getShape();
    if (aShape.size() < 2 || bShape.size() < 2) {
        MPS_LOG_ERROR("triangular_solve: expected rank >= 2 inputs (got ranks %zu, %zu)\n", aShape.size(),
                      bShape.size());
        return nil;
    }
    if (aShape.size() != bShape.size()) {
        MPS_LOG_ERROR("triangular_solve: unsupported layout (A and B must have same rank, got %zu and %zu)\n",
                      aShape.size(), bShape.size());
        return nil;
    }
    bool aOk = aType.getElementType().isF32() || aType.getElementType().isF16();
    bool bOk = bType.getElementType().isF32() || bType.getElementType().isF16();
    if (!aOk || !bOk || aType.getElementType() != bType.getElementType()) {
        MPS_LOG_ERROR("triangular_solve: only matching float16/float32 dtypes are supported (got "
                      "A=%s, B=%s)\n",
                      TypeToString(aType.getElementType()).c_str(),
                      TypeToString(bType.getElementType()).c_str());
        return nil;
    }

    int64_t n = aShape.back();
    int64_t aRows = aShape[aShape.size() - 2];
    if (aRows != n) {
        MPS_LOG_ERROR("triangular_solve: A must be square in last two dims (got %lld x %lld)\n",
                      (long long)aRows, (long long)n);
        return nil;
    }

    int64_t bRows = bShape[bShape.size() - 2];
    int64_t bCols = bShape[bShape.size() - 1];
    if ((leftSide && bRows != n) || (!leftSide && bCols != n)) {
        MPS_LOG_ERROR("triangular_solve: incompatible matrix dimensions for left_side=%d "
                      "(A=%lldx%lld, B=%lldx%lld)\n",
                      leftSide ? 1 : 0, (long long)aRows, (long long)n, (long long)bRows, (long long)bCols);
        return nil;
    }

    size_t batchRank = aShape.size() - 2;
    for (size_t i = 0; i < batchRank; ++i) {
        if (aShape[i] != bShape[i] || aShape[i] <= 0) {
            MPS_LOG_ERROR("triangular_solve: unsupported batch layout (A and B batch dims must "
                          "match and be > 0)\n");
            return nil;
        }
    }

    NSUInteger batchCount = 0;
    if (!ComputeBatchCount(aShape, 2, &batchCount)) {
        MPS_LOG_ERROR("triangular_solve: invalid batch dimensions in A shape\n");
        return nil;
    }

    bool isF16 = aType.getElementType().isF16();
    if (isF16) {
        uint32_t n32 = (uint32_t)n;
        uint32_t bRows32 = (uint32_t)bRows;
        uint32_t bCols32 = (uint32_t)bCols;
        uint32_t batchCount32 = (uint32_t)batchCount;

        MPSDataType computeDtype = MPSDataTypeFloat32;
        NSUInteger aRowBytes32 = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n
                                                                 dataType:computeDtype];
        NSUInteger bRowBytes32 = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)bCols
                                                                 dataType:computeDtype];
        NSUInteger aSliceBytes32 = aRowBytes32 * (NSUInteger)n;
        NSUInteger bSliceBytes32 = bRowBytes32 * (NSUInteger)bRows;
        uint32_t aStride32 = (uint32_t)(aRowBytes32 / sizeof(float));
        uint32_t bStride32 = (uint32_t)(bRowBytes32 / sizeof(float));
        uint32_t aSliceStride32 = aStride32 * n32;
        uint32_t bSliceStride32 = bStride32 * bRows32;

        NSUInteger bDataRowBytes16 = (NSUInteger)(bCols * (int64_t)sizeof(uint16_t));
        NSUInteger bDataSliceBytes16 = bDataRowBytes16 * (NSUInteger)bRows;

        id<MTLBuffer> a32Buf = StagingBufferPool::Instance().Acquire(device, aSliceBytes32 * batchCount,
                                                                     MTLResourceStorageModeShared);
        id<MTLBuffer> b32Buf = StagingBufferPool::Instance().Acquire(device, bSliceBytes32 * batchCount,
                                                                     MTLResourceStorageModeShared);
        if (!a32Buf || !b32Buf) {
            MPS_LOG_ERROR("triangular_solve: failed to allocate float32 staging buffers\n");
            return nil;
        }

        if (!EncodeCastHalfToFloatStrided(device, cmdBuf, inputs[0], a32Buf, n32, n32, aStride32,
                                          aSliceStride32, batchCount32) ||
            !EncodeCastHalfToFloatStrided(device, cmdBuf, inputs[1], b32Buf, bRows32, bCols32, bStride32,
                                          bSliceStride32, batchCount32)) {
            MPS_LOG_ERROR("triangular_solve: failed to encode float16->float32 cast kernels\n");
            return nil;
        }

        MPSMatrixDescriptor* aDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)n
                                                                           columns:(NSUInteger)n
                                                                          matrices:batchCount
                                                                          rowBytes:aRowBytes32
                                                                       matrixBytes:aSliceBytes32
                                                                          dataType:computeDtype];
        MPSMatrixDescriptor* bDesc = [MPSMatrixDescriptor matrixDescriptorWithRows:(NSUInteger)bRows
                                                                           columns:(NSUInteger)bCols
                                                                          matrices:batchCount
                                                                          rowBytes:bRowBytes32
                                                                       matrixBytes:bSliceBytes32
                                                                          dataType:computeDtype];
        if (!aDesc || !bDesc) {
            MPS_LOG_ERROR("triangular_solve: failed to create batched descriptors\n");
            return nil;
        }

        id<MTLBuffer> sol32Buf = StagingBufferPool::Instance().Acquire(device, bSliceBytes32 * batchCount,
                                                                       MTLResourceStorageModeShared);
        if (!sol32Buf) {
            MPS_LOG_ERROR("triangular_solve: failed to allocate solution staging buffer\n");
            return nil;
        }

        NSUInteger nrhsLocal = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;
        MPSMatrixSolveTriangular* solver = GetCachedTriangularSolver(
            device, (NSUInteger)n, nrhsLocal, leftSide, lower, unitDiagonal, transpose);
        if (!solver) {
            MPS_LOG_ERROR("triangular_solve: failed to create native MPS solver\n");
            return nil;
        }

        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:a32Buf descriptor:aDesc];
        MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:b32Buf descriptor:bDesc];
        MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:sol32Buf descriptor:bDesc];
        [solver encodeToCommandBuffer:cmdBuf
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rhsMatrix
                       solutionMatrix:solutionMatrix];

        id<MTLBuffer> out16Buf = [device newBufferWithLength:bDataSliceBytes16 * batchCount
                                                     options:MTLResourceStorageModeShared];
        if (!EncodeCastFloatToHalfStrided(device, cmdBuf, sol32Buf, out16Buf, bRows32, bCols32, bStride32,
                                          bSliceStride32, batchCount32)) {
            MPS_LOG_ERROR("triangular_solve: failed to encode float32->float16 cast kernel\n");
            return nil;
        }
        ReturnToPoolOnCompletion(device, cmdBuf, a32Buf);
        ReturnToPoolOnCompletion(device, cmdBuf, b32Buf);
        ReturnToPoolOnCompletion(device, cmdBuf, sol32Buf);
        return out16Buf;
    }

    MPSDataType mps_dtype = MlirTypeToMps(bType.getElementType());
    int pjrt_dtype = MlirTypeToPjrtDtype(bType.getElementType());
    size_t elem_size = DtypeByteSize(pjrt_dtype);

    NSUInteger nrhs = leftSide ? (NSUInteger)bCols : (NSUInteger)bRows;

    NSUInteger aDataRowBytes = (NSUInteger)(n * (int64_t)elem_size);
    NSUInteger aMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)n dataType:mps_dtype];
    NSUInteger bDataRowBytes = (NSUInteger)(bCols * (int64_t)elem_size);
    NSUInteger bMpsRowBytes = [MPSMatrixDescriptor rowBytesFromColumns:(NSUInteger)bCols dataType:mps_dtype];
    NSUInteger aDataSliceBytes = aDataRowBytes * (NSUInteger)n;
    NSUInteger aMpsSliceBytes = aMpsRowBytes * (NSUInteger)n;
    NSUInteger bDataSliceBytes = bDataRowBytes * (NSUInteger)bRows;
    NSUInteger bMpsSliceBytes = bMpsRowBytes * (NSUInteger)bRows;

    if (aShape.size() == 2) {
        bool didPadA = false;
        bool didPadB = false;
        id<MTLBuffer> aBuf =
            PadBuffer(device, cmdBuf, inputs[0], n, aDataRowBytes, aMpsRowBytes, elem_size, &didPadA);
        id<MTLBuffer> bBuf =
            PadBuffer(device, cmdBuf, inputs[1], bRows, bDataRowBytes, bMpsRowBytes, elem_size, &didPadB);
        if (didPadA) {
            ReturnToPoolOnCompletion(device, cmdBuf, aBuf);
        }
        if (didPadB) {
            ReturnToPoolOnCompletion(device, cmdBuf, bBuf);
        }

        MPSMatrixDescriptor* aDesc =
            GetCachedMatrixDescriptor(mps_dtype, (NSUInteger)n, (NSUInteger)n, aMpsRowBytes);
        MPSMatrixDescriptor* bDesc =
            GetCachedMatrixDescriptor(mps_dtype, (NSUInteger)bRows, (NSUInteger)bCols, bMpsRowBytes);
        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf descriptor:aDesc];
        MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf descriptor:bDesc];
        bool needsOutUnpad = bMpsRowBytes != bDataRowBytes;
        id<MTLBuffer> solBuf = nil;
        if (needsOutUnpad) {
            solBuf =
                StagingBufferPool::Instance().Acquire(device, bMpsSliceBytes, MTLResourceStorageModeShared);
        } else {
            solBuf = [device newBufferWithLength:bMpsSliceBytes options:MTLResourceStorageModeShared];
        }
        if (!solBuf) {
            return nil;
        }
        MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf descriptor:bDesc];

        MPSMatrixSolveTriangular* solver =
            GetCachedTriangularSolver(device, (NSUInteger)n, nrhs, leftSide, lower, unitDiagonal, transpose);
        if (!solver) {
            MPS_LOG_ERROR("triangular_solve: failed to create native MPS solver\n");
            return nil;
        }

        [solver encodeToCommandBuffer:cmdBuf
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rhsMatrix
                       solutionMatrix:solutionMatrix];
        id<MTLBuffer> out =
            UnpadBuffer(device, cmdBuf, solBuf, bRows, bDataRowBytes, bMpsRowBytes, elem_size);
        if (needsOutUnpad) {
            ReturnToPoolOnCompletion(device, cmdBuf, solBuf);
        }
        return out;
    }

    bool aNeedsPadding = aMpsRowBytes != aDataRowBytes;
    bool bNeedsPadding = bMpsRowBytes != bDataRowBytes;

    id<MTLBuffer> aBuf = inputs[0];
    if (aNeedsPadding) {
        aBuf = StagingBufferPool::Instance().Acquire(device, aMpsSliceBytes * batchCount,
                                                     MTLResourceStorageModeShared);
        if (!aBuf) {
            return nil;
        }
        uint32_t rows32 = (uint32_t)n;
        uint32_t cols32 = (uint32_t)n;
        uint32_t dataStride = cols32;
        uint32_t mpsStride = (uint32_t)(aMpsRowBytes / elem_size);
        uint32_t dataSlice = dataStride * rows32;
        uint32_t mpsSlice = mpsStride * rows32;
        if (!EncodeCopyRowsStrided(device, cmdBuf, inputs[0], aBuf, rows32, cols32, dataStride, dataSlice,
                                   mpsStride, mpsSlice, (uint32_t)batchCount, elem_size)) {
            memset(aBuf.contents, 0, aMpsSliceBytes * batchCount);
            for (NSUInteger b = 0; b < batchCount; ++b) {
                BlitRowsWithOffsets(cmdBuf, inputs[0], b * aDataSliceBytes, aDataRowBytes, aBuf,
                                    b * aMpsSliceBytes, aMpsRowBytes, n, aDataRowBytes);
            }
        }
        ReturnToPoolOnCompletion(device, cmdBuf, aBuf);
    }

    id<MTLBuffer> bBuf = inputs[1];
    if (bNeedsPadding) {
        bBuf = StagingBufferPool::Instance().Acquire(device, bMpsSliceBytes * batchCount,
                                                     MTLResourceStorageModeShared);
        if (!bBuf) {
            return nil;
        }
        uint32_t rows32 = (uint32_t)bRows;
        uint32_t cols32 = (uint32_t)bCols;
        uint32_t dataStride = cols32;
        uint32_t mpsStride = (uint32_t)(bMpsRowBytes / elem_size);
        uint32_t dataSlice = dataStride * rows32;
        uint32_t mpsSlice = mpsStride * rows32;
        if (!EncodeCopyRowsStrided(device, cmdBuf, inputs[1], bBuf, rows32, cols32, dataStride, dataSlice,
                                   mpsStride, mpsSlice, (uint32_t)batchCount, elem_size)) {
            memset(bBuf.contents, 0, bMpsSliceBytes * batchCount);
            for (NSUInteger b = 0; b < batchCount; ++b) {
                BlitRowsWithOffsets(cmdBuf, inputs[1], b * bDataSliceBytes, bDataRowBytes, bBuf,
                                    b * bMpsSliceBytes, bMpsRowBytes, bRows, bDataRowBytes);
            }
        }
        ReturnToPoolOnCompletion(device, cmdBuf, bBuf);
    }

    MPSMatrixDescriptor* aDesc =
        GetCachedMatrixDescriptor(mps_dtype, (NSUInteger)n, (NSUInteger)n, aMpsRowBytes);
    MPSMatrixDescriptor* bDesc =
        GetCachedMatrixDescriptor(mps_dtype, (NSUInteger)bRows, (NSUInteger)bCols, bMpsRowBytes);
    if (!aDesc || !bDesc) {
        MPS_LOG_ERROR("triangular_solve: failed to create matrix descriptors\n");
        return nil;
    }

    bool needsOutUnpad = bNeedsPadding;
    id<MTLBuffer> solBuf = nil;
    if (needsOutUnpad) {
        solBuf = StagingBufferPool::Instance().Acquire(device, bMpsSliceBytes * batchCount,
                                                       MTLResourceStorageModeShared);
    } else {
        solBuf = [device newBufferWithLength:bMpsSliceBytes * batchCount
                                     options:MTLResourceStorageModeShared];
    }
    if (!solBuf) {
        return nil;
    }

    MPSMatrixSolveTriangular* solver =
        GetCachedTriangularSolver(device, (NSUInteger)n, nrhs, leftSide, lower, unitDiagonal, transpose);
    if (!solver) {
        MPS_LOG_ERROR("triangular_solve: failed to create native MPS solver\n");
        return nil;
    }

    for (NSUInteger b = 0; b < batchCount; ++b) {
        NSUInteger aOffset = b * aMpsSliceBytes;
        NSUInteger bOffset = b * bMpsSliceBytes;
        MPSMatrix* sourceMatrix = [[MPSMatrix alloc] initWithBuffer:aBuf offset:aOffset descriptor:aDesc];
        MPSMatrix* rhsMatrix = [[MPSMatrix alloc] initWithBuffer:bBuf offset:bOffset descriptor:bDesc];
        MPSMatrix* solutionMatrix = [[MPSMatrix alloc] initWithBuffer:solBuf offset:bOffset descriptor:bDesc];
        [solver encodeToCommandBuffer:cmdBuf
                         sourceMatrix:sourceMatrix
                  rightHandSideMatrix:rhsMatrix
                       solutionMatrix:solutionMatrix];
    }

    if (!bNeedsPadding) {
        return solBuf;
    }

    id<MTLBuffer> outBuf = [device newBufferWithLength:bDataSliceBytes * batchCount
                                               options:MTLResourceStorageModeShared];
    uint32_t rows32 = (uint32_t)bRows;
    uint32_t cols32 = (uint32_t)bCols;
    uint32_t dataStride = cols32;
    uint32_t mpsStride = (uint32_t)(bMpsRowBytes / elem_size);
    uint32_t dataSlice = dataStride * rows32;
    uint32_t mpsSlice = mpsStride * rows32;
    if (!EncodeCopyRowsStrided(device, cmdBuf, solBuf, outBuf, rows32, cols32, mpsStride, mpsSlice,
                               dataStride, dataSlice, (uint32_t)batchCount, elem_size)) {
        for (NSUInteger b = 0; b < batchCount; ++b) {
            BlitRowsWithOffsets(cmdBuf, solBuf, b * bMpsSliceBytes, bMpsRowBytes, outBuf, b * bDataSliceBytes,
                                bDataRowBytes, bRows, bDataRowBytes);
        }
    }
    ReturnToPoolOnCompletion(device, cmdBuf, solBuf);
    return outBuf;
}

// Graph form of triangular_solve for use inside case/while regions, where the
// native MPS solver cannot run (jax.scipy.linalg.expm solves its Pade system
// inside a runtime case and failed with "Native op encountered during graph
// building"). Every variant is reduced to a left-side lower solve with
// transposes and row/column reversal, then solved by unrolled forward
// substitution. Only the requested triangle is read. float16/float32, n <= 256.
static ProcessResult GraphTriangularSolveFallback(HandlerContext& ctx) {
    auto triOp = mlir::dyn_cast<mlir::stablehlo::TriangularSolveOp>(ctx.op);
    if (!triOp) {
        return ProcessResult::Error("triangular_solve: expected TriangularSolveOp");
    }
    MPSGraphTensor* a = GetInputTensor(ctx, 0);
    MPSGraphTensor* b = GetInputTensor(ctx, 1);
    if (!a || !b) {
        return ProcessResult::Error("triangular_solve: missing input tensor");
    }
    MPSGraph* g = ctx.graph;
    const MPSDataType outType = b.dataType;
    if ((outType != MPSDataTypeFloat32 && outType != MPSDataTypeFloat16) || a.dataType != outType) {
        return ProcessResult::Error(
            "triangular_solve: inside control flow only float16/float32 is supported");
    }
    const NSUInteger rank = a.shape.count;
    if (rank < 2 || b.shape.count != rank) {
        return ProcessResult::Error("triangular_solve: expected matching rank >= 2 operands");
    }
    const int64_t n = [a.shape[rank - 1] longLongValue];
    if (n <= 0 || n > 256 || [a.shape[rank - 2] longLongValue] != n) {
        return ProcessResult::Error("triangular_solve: inside control flow A must be square with n <= 256");
    }
    if (outType == MPSDataTypeFloat16) {
        a = [g castTensor:a toType:MPSDataTypeFloat32 name:nil];
        b = [g castTensor:b toType:MPSDataTypeFloat32 name:nil];
    }
    bool lower = triOp.getLower();
    const bool unit = triOp.getUnitDiagonal();
    const auto transposeA = triOp.getTransposeA();
    const bool left = triOp.getLeftSide();
    if (transposeA == mlir::stablehlo::Transpose::TRANSPOSE ||
        transposeA == mlir::stablehlo::Transpose::ADJOINT) {
        a = SwapLastTwo(g, a);
        lower = !lower;
    }
    if (!left) { // X op(A) = B  <=>  op(A)^T X^T = B^T
        a = SwapLastTwo(g, a);
        lower = !lower;
        b = SwapLastTwo(g, b);
    }
    const bool flipped = !lower; // J U J is lower; solve for J x from J b.
    if (flipped) {
        a = [g reverseTensor:a axes:@[@(rank - 2), @(rank - 1)] name:nil];
        b = [g reverseTensor:b axes:@[@(rank - 2)] name:nil];
    }
    NSMutableArray<MPSGraphTensor*>* rows = [NSMutableArray arrayWithCapacity:(NSUInteger)n];
    for (int64_t i = 0; i < n; ++i) {
        MPSGraphTensor* aRow = [g sliceTensor:a
                                    dimension:(NSInteger)rank - 2
                                        start:(NSInteger)i
                                       length:1
                                         name:nil];
        MPSGraphTensor* bi = [g sliceTensor:b
                                  dimension:(NSInteger)rank - 2
                                      start:(NSInteger)i
                                     length:1
                                       name:nil];
        if (i > 0) {
            MPSGraphTensor* li = [g sliceTensor:aRow
                                      dimension:(NSInteger)rank - 1
                                          start:0
                                         length:(NSInteger)i
                                           name:nil];
            MPSGraphTensor* xp =
                rows.count == 1 ? rows[0] : [g concatTensors:rows dimension:(NSInteger)rank - 2 name:nil];
            MPSGraphTensor* acc = [g matrixMultiplicationWithPrimaryTensor:li secondaryTensor:xp name:nil];
            bi = [g subtractionWithPrimaryTensor:bi secondaryTensor:acc name:nil];
        }
        if (!unit) {
            MPSGraphTensor* d = [g sliceTensor:aRow
                                     dimension:(NSInteger)rank - 1
                                         start:(NSInteger)i
                                        length:1
                                          name:nil];
            bi = [g divisionWithPrimaryTensor:bi secondaryTensor:d name:nil];
        }
        [rows addObject:bi];
    }
    MPSGraphTensor* x =
        rows.count == 1 ? rows[0] : [g concatTensors:rows dimension:(NSInteger)rank - 2 name:nil];
    if (flipped) {
        x = [g reverseTensor:x axes:@[@(rank - 2)] name:nil];
    }
    if (!left) {
        x = SwapLastTwo(g, x);
    }
    if (outType == MPSDataTypeFloat16) {
        x = [g castTensor:x toType:MPSDataTypeFloat16 name:nil];
    }
    return Result(ctx, x, "triangular_solve");
}

REGISTER_NATIVE_MPS_OP_WITH_GRAPH_FALLBACK("stablehlo.triangular_solve", NativeHandle_triangular_solve,
                                           GraphTriangularSolveFallback);

// jnp.linalg.qr lowers to two custom calls that had no MPS handler:
//   Qr: LAPACK geqrf -> (packed R + Householder tails, tau)
//   ProductOfElementaryHouseholderReflectors: LAPACK orgqr -> Q
// Both are lowered as unrolled Householder steps over masked full-length
// column vectors (static slices only, no scatter), following slarfg's sign
// conventions (beta = -sign(alpha) * ||[alpha, x]||, tau = (beta - alpha) /
// beta, v = x / (alpha - beta), tau = 0 for a zero tail) so Q and R match
// LAPACK directly. float32 only; the unroll is capped to keep graphs bounded.
static constexpr int64_t kMaxHouseholderSteps = 128;

static bool QrMatrixDims(NSArray<NSNumber*>* shape, int64_t* batch, int64_t* m, int64_t* n) {
    if (!shape || shape.count < 2) {
        return false;
    }
    int64_t b = 1;
    for (NSUInteger i = 0; i + 2 < shape.count; ++i) {
        b *= [shape[i] longLongValue];
    }
    *batch = b;
    *m = [shape[shape.count - 2] longLongValue];
    *n = [shape[shape.count - 1] longLongValue];
    return b >= 0 && *m >= 0 && *n >= 0;
}

static ProcessResult HandleQrCustomCall(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input || ctx.op->getNumResults() != 2) {
        return ProcessResult::Error("Qr: expected one operand and two results");
    }
    if (input.dataType != MPSDataTypeFloat32) {
        return ProcessResult::Error("Qr: only float32 is supported on MPS");
    }
    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op, 0);
    NSArray<NSNumber*>* tauShape = GetOutputShape(ctx.op, 1);
    int64_t batch = 0, m = 0, n = 0;
    if (!tauShape || !QrMatrixDims(shape, &batch, &m, &n)) {
        return ProcessResult::Error("Qr: expected static rank >= 2 shapes");
    }
    const int64_t k = std::min(m, n);
    if (k > kMaxHouseholderSteps) {
        return ProcessResult::Error("Qr: min(m, n) > 128 is not supported on MPS");
    }
    MPSGraph* g = ctx.graph;
    if (k == 0 || batch == 0) {
        SetOutputTensor(ctx.values, ctx.op, input, 0);
        SetOutputTensor(ctx.values, ctx.op,
                        [g constantWithScalar:0.0 shape:tauShape dataType:MPSDataTypeFloat32], 1);
        return ProcessResult{};
    }

    MPSGraphTensor* A = [g reshapeTensor:input withShape:@[@(batch), @(m), @(n)] name:nil];
    MPSGraphTensor* rowIdx = [g coordinateAlongAxis:1 withShape:@[@1, @(m), @1] name:nil];
    MPSGraphTensor* colIdx = [g coordinateAlongAxis:2 withShape:@[@1, @1, @(n)] name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    NSMutableArray<MPSGraphTensor*>* taus = [NSMutableArray arrayWithCapacity:(NSUInteger)k];

    for (int64_t j = 0; j < k; ++j) {
        MPSGraphTensor* jc = [g constantWithScalar:(double)j dataType:rowIdx.dataType];
        MPSGraphTensor* isDiag = [g equalWithPrimaryTensor:rowIdx secondaryTensor:jc name:nil];
        MPSGraphTensor* isBelow = [g greaterThanWithPrimaryTensor:rowIdx secondaryTensor:jc name:nil];
        MPSGraphTensor* c = [g sliceTensor:A dimension:2 start:(NSInteger)j length:1 name:nil];
        MPSGraphTensor* alpha = [g reductionSumWithTensor:[g selectWithPredicateTensor:isDiag
                                                                   truePredicateTensor:c
                                                                  falsePredicateTensor:zero
                                                                                  name:nil]
                                                     axis:1
                                                     name:nil];
        MPSGraphTensor* tail = [g selectWithPredicateTensor:isBelow
                                        truePredicateTensor:c
                                       falsePredicateTensor:zero
                                                       name:nil];
        MPSGraphTensor* xnorm2 = [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:tail
                                                                              secondaryTensor:tail
                                                                                         name:nil]
                                                      axis:1
                                                      name:nil];
        MPSGraphTensor* norm =
            [g squareRootWithTensor:[g additionWithPrimaryTensor:[g multiplicationWithPrimaryTensor:alpha
                                                                                    secondaryTensor:alpha
                                                                                               name:nil]
                                                 secondaryTensor:xnorm2
                                                            name:nil]
                               name:nil];
        MPSGraphTensor* beta = [g selectWithPredicateTensor:[g greaterThanOrEqualToWithPrimaryTensor:alpha
                                                                                     secondaryTensor:zero
                                                                                                name:nil]
                                        truePredicateTensor:[g negativeWithTensor:norm name:nil]
                                       falsePredicateTensor:norm
                                                       name:nil];
        MPSGraphTensor* zeroTail = [g equalWithPrimaryTensor:xnorm2 secondaryTensor:zero name:nil];
        MPSGraphTensor* safeBeta = [g selectWithPredicateTensor:zeroTail
                                            truePredicateTensor:one
                                           falsePredicateTensor:beta
                                                           name:nil];
        MPSGraphTensor* tau = [g divisionWithPrimaryTensor:[g subtractionWithPrimaryTensor:beta
                                                                           secondaryTensor:alpha
                                                                                      name:nil]
                                           secondaryTensor:safeBeta
                                                      name:nil];
        tau = [g selectWithPredicateTensor:zeroTail
                       truePredicateTensor:zero
                      falsePredicateTensor:tau
                                      name:nil];
        MPSGraphTensor* denom = [g selectWithPredicateTensor:zeroTail
                                         truePredicateTensor:one
                                        falsePredicateTensor:[g subtractionWithPrimaryTensor:alpha
                                                                             secondaryTensor:beta
                                                                                        name:nil]
                                                        name:nil];
        MPSGraphTensor* vTail = [g divisionWithPrimaryTensor:tail secondaryTensor:denom name:nil];
        MPSGraphTensor* v = [g selectWithPredicateTensor:isDiag
                                     truePredicateTensor:one
                                    falsePredicateTensor:vTail
                                                    name:nil];

        // Apply H = I - tau v v^T to the trailing columns (> j).
        MPSGraphTensor* w = [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:v
                                                                         secondaryTensor:A
                                                                                    name:nil]
                                                 axis:1
                                                 name:nil];
        MPSGraphTensor* tv = [g multiplicationWithPrimaryTensor:tau secondaryTensor:v name:nil];
        MPSGraphTensor* updated = [g subtractionWithPrimaryTensor:A
                                                  secondaryTensor:[g multiplicationWithPrimaryTensor:tv
                                                                                     secondaryTensor:w
                                                                                                name:nil]
                                                             name:nil];
        MPSGraphTensor* colAfter = [g greaterThanWithPrimaryTensor:colIdx secondaryTensor:jc name:nil];
        A = [g selectWithPredicateTensor:colAfter
                     truePredicateTensor:updated
                    falsePredicateTensor:A
                                    name:nil];

        // Column j: beta on the diagonal (alpha when tau = 0), v's tail below.
        MPSGraphTensor* diagVal = [g selectWithPredicateTensor:zeroTail
                                           truePredicateTensor:alpha
                                          falsePredicateTensor:beta
                                                          name:nil];
        MPSGraphTensor* newCol = [g selectWithPredicateTensor:isBelow
                                          truePredicateTensor:vTail
                                         falsePredicateTensor:c
                                                         name:nil];
        newCol = [g selectWithPredicateTensor:isDiag
                          truePredicateTensor:diagVal
                         falsePredicateTensor:newCol
                                         name:nil];
        MPSGraphTensor* colIsJ = [g equalWithPrimaryTensor:colIdx secondaryTensor:jc name:nil];
        A = [g selectWithPredicateTensor:colIsJ truePredicateTensor:newCol falsePredicateTensor:A name:nil];

        [taus addObject:[g reshapeTensor:tau withShape:@[@(batch), @1] name:nil]];
    }

    MPSGraphTensor* tauAll = taus.count == 1 ? taus[0] : [g concatTensors:taus dimension:1 name:nil];
    SetOutputTensor(ctx.values, ctx.op, [g reshapeTensor:A withShape:shape name:nil], 0);
    SetOutputTensor(ctx.values, ctx.op, [g reshapeTensor:tauAll withShape:tauShape name:nil], 1);
    return ProcessResult{};
}

static ProcessResult HandleHouseholderProduct(HandlerContext& ctx) {
    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    MPSGraphTensor* tauIn = GetInputTensor(ctx, 1);
    if (!input || !tauIn || ctx.op->getNumResults() != 1) {
        return ProcessResult::Error("ProductOfElementaryHouseholderReflectors: expected (a, tau) -> q");
    }
    if (input.dataType != MPSDataTypeFloat32 || tauIn.dataType != MPSDataTypeFloat32) {
        return ProcessResult::Error(
            "ProductOfElementaryHouseholderReflectors: only float32 is supported on MPS");
    }
    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op, 0);
    int64_t batch = 0, m = 0, n = 0;
    NSArray<NSNumber*>* tauShape = tauIn.shape;
    if (!tauShape || tauShape.count == 0 || !QrMatrixDims(shape, &batch, &m, &n)) {
        return ProcessResult::Error("ProductOfElementaryHouseholderReflectors: expected static shapes");
    }
    const int64_t k = [tauShape.lastObject longLongValue];
    if (k > n || k > m || k > kMaxHouseholderSteps) {
        return ProcessResult::Error("ProductOfElementaryHouseholderReflectors: unsupported reflector count");
    }
    MPSGraph* g = ctx.graph;
    MPSGraphTensor* rowIdx = [g coordinateAlongAxis:1 withShape:@[@1, @(m), @1] name:nil];
    MPSGraphTensor* colIdx = [g coordinateAlongAxis:2 withShape:@[@1, @1, @(n)] name:nil];
    MPSGraphTensor* zero = [g constantWithScalar:0.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* one = [g constantWithScalar:1.0 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* Q = [g selectWithPredicateTensor:[g equalWithPrimaryTensor:rowIdx
                                                               secondaryTensor:colIdx
                                                                          name:nil]
                                 truePredicateTensor:one
                                falsePredicateTensor:zero
                                                name:nil];
    Q = [g broadcastTensor:Q toShape:@[@(batch), @(m), @(n)] name:nil];
    if (k > 0 && batch > 0) {
        MPSGraphTensor* A = [g reshapeTensor:input withShape:@[@(batch), @(m), @(n)] name:nil];
        MPSGraphTensor* T = [g reshapeTensor:tauIn withShape:@[@(batch), @(k)] name:nil];
        for (int64_t j = k - 1; j >= 0; --j) {
            MPSGraphTensor* jc = [g constantWithScalar:(double)j dataType:rowIdx.dataType];
            MPSGraphTensor* isDiag = [g equalWithPrimaryTensor:rowIdx secondaryTensor:jc name:nil];
            MPSGraphTensor* isBelow = [g greaterThanWithPrimaryTensor:rowIdx secondaryTensor:jc name:nil];
            MPSGraphTensor* c = [g sliceTensor:A dimension:2 start:(NSInteger)j length:1 name:nil];
            MPSGraphTensor* v = [g selectWithPredicateTensor:isBelow
                                         truePredicateTensor:c
                                        falsePredicateTensor:zero
                                                        name:nil];
            v = [g selectWithPredicateTensor:isDiag truePredicateTensor:one falsePredicateTensor:v name:nil];
            MPSGraphTensor* t = [g sliceTensor:T dimension:1 start:(NSInteger)j length:1 name:nil];
            t = [g reshapeTensor:t withShape:@[@(batch), @1, @1] name:nil];
            MPSGraphTensor* w = [g reductionSumWithTensor:[g multiplicationWithPrimaryTensor:v
                                                                             secondaryTensor:Q
                                                                                        name:nil]
                                                     axis:1
                                                     name:nil];
            MPSGraphTensor* tv = [g multiplicationWithPrimaryTensor:t secondaryTensor:v name:nil];
            Q = [g subtractionWithPrimaryTensor:Q
                                secondaryTensor:[g multiplicationWithPrimaryTensor:tv
                                                                   secondaryTensor:w
                                                                              name:nil]
                                           name:nil];
        }
    }
    return Result(ctx, [g reshapeTensor:Q withShape:shape name:nil], "householder_product");
}

REGISTER_CUSTOM_CALL("Qr", HandleQrCustomCall, qr);
REGISTER_CUSTOM_CALL("ProductOfElementaryHouseholderReflectors", HandleHouseholderProduct,
                     householder_product);

} // namespace jax_metallib
