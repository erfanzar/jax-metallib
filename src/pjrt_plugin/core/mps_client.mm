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

#import "pjrt_plugin/core/mps_client.h"

#include "pjrt_plugin/core/logging.h"
#import "pjrt_plugin/core/mps_buffer.h"
#import "pjrt_plugin/core/mps_device.h"
#import "pjrt_plugin/core/mps_executable.h"
#import "pjrt_plugin/core/stablehlo_parser.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cstdlib>

namespace jax_metallib {

static bool EnvFlagEnabled(const char* name) {
    const char* v = std::getenv(name);
    if (!v || v[0] == '\0') {
        return false;
    }
    return !(v[0] == '0' && v[1] == '\0');
}

MpsClient::MpsClient()
    : platform_name_("mps"), platform_version_("0.1.0"), metal_device_(nullptr), command_queue_(nullptr) {}

MpsClient::~MpsClient() {
    devices_.clear();
    if (command_queue_) {
        CFRelease((__bridge CFTypeRef)command_queue_);
        command_queue_ = nullptr;
    }
    if (metal_device_) {
        CFRelease((__bridge CFTypeRef)metal_device_);
        metal_device_ = nullptr;
    }
}

std::unique_ptr<MpsClient> MpsClient::Create() {
    auto client = std::unique_ptr<MpsClient>(new MpsClient());
    if (!client->Initialize()) {
        return nullptr;
    }
    return client;
}

bool MpsClient::Initialize() {

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) {
        NSLog(@"Failed to create Metal device");
        return false;
    }

    metal_device_ = (void*)CFBridgingRetain(device);

    id<MTLCommandQueue> queue = [device newCommandQueue];
    if (!queue) {
        NSLog(@"Failed to create Metal command queue");
        return false;
    }
    command_queue_ = (void*)CFBridgingRetain(queue);

    NSString* name = [device name];
    auto mps_device = std::make_unique<MpsDevice>(this, 0, [name UTF8String]);
    devices_.push_back(std::move(mps_device));

    MPS_LOG_INFO("Initialized MPS client with device: %s\n", [name UTF8String]);
    return true;
}

int MpsClient::device_count() const {
    return static_cast<int>(devices_.size());
}

int MpsClient::addressable_device_count() const {
    return device_count();
}

MpsDevice* MpsClient::device(int index) {
    if (index < 0 || (size_t)index >= devices_.size()) {
        return nullptr;
    }
    return devices_[index].get();
}

MpsDevice* MpsClient::addressable_device(int index) {
    return device(index);
}

MpsDevice* MpsClient::LookupDevice(int device_id) {
    for (auto& dev : devices_) {
        if (dev->id() == device_id) {
            return dev.get();
        }
    }
    return nullptr;
}

static bool IsContiguous(const std::vector<int64_t>& dims, const std::vector<int64_t>& byte_strides,
                         size_t element_size) {
    if (byte_strides.empty()) {
        return true;
    }
    if (byte_strides.size() != dims.size()) {
        return false;
    }

    int64_t expected_stride = static_cast<int64_t>(element_size);
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        if (byte_strides[i] != expected_stride) {
            return false;
        }
        expected_stride *= dims[i];
    }
    return true;
}

static void CopyStridedToContiguous(const void* src, void* dst, const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& byte_strides, size_t element_size) {
    size_t ndim = dims.size();
    if (ndim == 0) {
        memcpy(dst, src, element_size);
        return;
    }

    std::vector<int64_t> contiguous_strides(ndim);
    int64_t stride = static_cast<int64_t>(element_size);
    for (int i = static_cast<int>(ndim) - 1; i >= 0; --i) {
        contiguous_strides[i] = stride;
        stride *= dims[i];
    }

    std::vector<int64_t> indices(ndim, 0);
    int64_t total_elements = 1;
    for (auto d : dims) {
        total_elements *= d;
    }

    const char* src_base = static_cast<const char*>(src);
    char* dst_base = static_cast<char*>(dst);

    for (int64_t elem = 0; elem < total_elements; ++elem) {

        int64_t src_offset = 0;
        for (size_t d = 0; d < ndim; ++d) {
            src_offset += indices[d] * byte_strides[d];
        }

        int64_t dst_offset = elem * static_cast<int64_t>(element_size);

        memcpy(dst_base + dst_offset, src_base + src_offset, element_size);

        for (int d = static_cast<int>(ndim) - 1; d >= 0; --d) {
            indices[d]++;
            if (indices[d] < dims[d]) {
                break;
            }
            indices[d] = 0;
        }
    }
}

std::unique_ptr<MpsBuffer> MpsClient::BufferFromHostBuffer(const void* data, int dtype,
                                                           const std::vector<int64_t>& dims,
                                                           const std::vector<int64_t>& byte_strides,
                                                           MpsDevice* device) {
    if (!metal_device_) {
        NSLog(@"ERROR: BufferFromHostBuffer called with no Metal device");
        return nullptr;
    }
    if (!data) {
        NSLog(@"ERROR: BufferFromHostBuffer called with null data pointer");
        return nullptr;
    }

    int64_t element_count = 1;
    for (int64_t dim : dims) {
        element_count *= dim;
    }
    size_t element_size = DtypeByteSize(dtype);
    size_t byte_size = element_count * element_size;

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)metal_device_;
    id<MTLBuffer> buffer = nil;

    // Zero-element tensors (e.g. shape [0] or [5,0,3]) are valid in JAX/XLA. Metal rejects
    // zero-length buffers and MPSNDArray binds them as if each zero dim were 1, so allocate a
    // zeroed placeholder of that size. The host array is empty: copy nothing from `data`.
    if (byte_size == 0) {
        buffer = [mtl_device newBufferWithLength:ZeroElementPlaceholderBytes(dtype, dims)
                                         options:MTLResourceStorageModeShared];
        if (!buffer) {
            NSLog(@"Failed to create zero-element Metal placeholder buffer");
            return nullptr;
        }
        auto empty = std::make_unique<MpsBuffer>(device ? device : devices_[0].get(), (__bridge void*)buffer,
                                                 dtype, dims);
        CFRelease((__bridge CFTypeRef)buffer);
        return empty;
    }

    const bool contiguous = IsContiguous(dims, byte_strides, element_size);
    const bool prefer_private = EnvFlagEnabled("JAX_METALLIB_USE_PRIVATE_INPUT_BUFFERS");
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)command_queue_;

    if (prefer_private && commandQueue) {
        id<MTLBuffer> staging = [mtl_device newBufferWithLength:byte_size
                                                        options:MTLResourceStorageModeShared];
        if (staging) {
            if (contiguous) {
                memcpy([staging contents], data, byte_size);
            } else {
                MPS_LOG_DEBUG("BufferFromHostBuffer: copying non-contiguous array to private GPU buffer\n");
                CopyStridedToContiguous(data, [staging contents], dims, byte_strides, element_size);
            }

            id<MTLBuffer> privateBuf = [mtl_device newBufferWithLength:byte_size
                                                               options:MTLResourceStorageModePrivate];
            if (privateBuf) {
                id<MTLCommandBuffer> cmd = nil;
                if ([commandQueue respondsToSelector:@selector(commandBufferWithUnretainedReferences)]) {
                    cmd = [commandQueue commandBufferWithUnretainedReferences];
                }
                if (!cmd) {
                    cmd = [commandQueue commandBuffer];
                }
                if (cmd) {
                    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                    if (blit) {
                        [blit copyFromBuffer:staging
                                 sourceOffset:0
                                     toBuffer:privateBuf
                            destinationOffset:0
                                         size:byte_size];
                        [blit endEncoding];
                        [cmd commit];
                        [cmd waitUntilCompleted];
                        if (cmd.status != MTLCommandBufferStatusError) {
                            buffer = privateBuf;
                        }
                    }
                }
                if (!buffer) {
                    CFRelease((__bridge CFTypeRef)privateBuf);
                }
            }
            CFRelease((__bridge CFTypeRef)staging);
        }
    }

    if (!buffer) {
        if (contiguous) {

            buffer = [mtl_device newBufferWithBytes:data
                                             length:byte_size
                                            options:MTLResourceStorageModeShared];
        } else {

            MPS_LOG_DEBUG("BufferFromHostBuffer: copying non-contiguous array to contiguous buffer\n");
            buffer = [mtl_device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
            if (buffer) {
                CopyStridedToContiguous(data, [buffer contents], dims, byte_strides, element_size);
            }
        }
    }

    if (!buffer) {
        NSLog(@"Failed to create Metal buffer of size %zu", byte_size);
        return nullptr;
    }

    auto result =
        std::make_unique<MpsBuffer>(device ? device : devices_[0].get(), (__bridge void*)buffer, dtype, dims);
    CFRelease((__bridge CFTypeRef)buffer);
    return result;
}

std::unique_ptr<MpsBuffer> MpsClient::CreateUninitializedBuffer(int dtype, const std::vector<int64_t>& dims,
                                                                MpsDevice* device) {
    if (!metal_device_) {
        NSLog(@"ERROR: CreateUninitializedBuffer called with no Metal device");
        return nullptr;
    }

    int64_t element_count = 1;
    for (int64_t dim : dims) {
        element_count *= dim;
    }
    size_t element_size = DtypeByteSize(dtype);
    size_t byte_size = element_count * element_size;

    if (byte_size == 0) {
        byte_size = ZeroElementPlaceholderBytes(dtype, dims);
    }

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)metal_device_;
    id<MTLBuffer> buffer = [mtl_device newBufferWithLength:byte_size options:MTLResourceStorageModeShared];
    if (!buffer) {
        NSLog(@"Failed to create uninitialized Metal buffer of size %zu", byte_size);
        return nullptr;
    }

    auto result =
        std::make_unique<MpsBuffer>(device ? device : devices_[0].get(), (__bridge void*)buffer, dtype, dims);
    CFRelease((__bridge CFTypeRef)buffer);
    return result;
}

MpsClient::MemoryStats MpsClient::GetMemoryStats() const {
    MemoryStats stats;
    if (!metal_device_) {
        return stats;
    }
    id<MTLDevice> device = (__bridge id<MTLDevice>)metal_device_;

    stats.bytes_in_use = static_cast<int64_t>(device.currentAllocatedSize);

    stats.bytes_limit = static_cast<int64_t>(device.recommendedMaxWorkingSetSize);
    stats.bytes_limit_is_set = true;

    return stats;
}

std::unique_ptr<MpsExecutable> MpsClient::CompileStableHLO(mps::ParsedModule module, MpsDevice* device) {

    auto exec = std::make_unique<MpsExecutable>(this, std::move(module));
    if (!exec->IsValid()) {
        NSLog(@"Failed to compile StableHLO module: %s", exec->error().c_str());
        return nullptr;
    }

    return exec;
}

} // namespace jax_metallib
