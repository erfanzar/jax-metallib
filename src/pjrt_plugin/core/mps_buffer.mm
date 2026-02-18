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

#import "pjrt_plugin/core/mps_client.h"
#import "pjrt_plugin/core/mps_device.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <utility>
#include <xla/pjrt/c/pjrt_c_api.h>

namespace jax_metallib {

size_t DtypeByteSize(int dtype) {
    switch (dtype) {
    case PJRT_Buffer_Type_PRED:
    case PJRT_Buffer_Type_S8:
    case PJRT_Buffer_Type_U8:
        return 1;
    case PJRT_Buffer_Type_S16:
    case PJRT_Buffer_Type_U16:
    case PJRT_Buffer_Type_F16:
    case PJRT_Buffer_Type_BF16:
        return 2;
    case PJRT_Buffer_Type_S32:
    case PJRT_Buffer_Type_U32:
    case PJRT_Buffer_Type_F32:
        return 4;
    case PJRT_Buffer_Type_S64:
    case PJRT_Buffer_Type_U64:
    case PJRT_Buffer_Type_F64:
    case PJRT_Buffer_Type_C64:
        return 8;
    case PJRT_Buffer_Type_C128:
        return 16;
    default:

        NSLog(@"ERROR: Unknown dtype %d in DtypeByteSize", dtype);
        return 0;
    }
}

MpsBuffer::MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype, const std::vector<int64_t>& dims,
                     BufferReleaser releaser, CompletionEvent producer_completion)
    : device_(device), metal_buffer_(metal_buffer), dtype_(dtype), dims_(dims),
      releaser_(std::move(releaser)),
      producer_completion_(producer_completion ? std::move(producer_completion)
                                               : MakeReadyCompletionEvent()) {

    if (metal_buffer_) {
        CFRetain((__bridge CFTypeRef)metal_buffer_);
    }
}

MpsBuffer::~MpsBuffer() {
    Delete();
}

int64_t MpsBuffer::element_count() const {
    int64_t count = 1;
    for (int64_t dim : dims_) {
        count *= dim;
    }
    return count;
}

size_t MpsBuffer::byte_size() const {
    return element_count() * DtypeByteSize(dtype_);
}

bool MpsBuffer::WaitForProducer(std::string* error) const {
    AwaitCompletionEvent(producer_completion_);
    const std::string producer_error = CompletionEventError(producer_completion_);
    if (error) {
        *error = producer_error;
    }
    return producer_error.empty();
}

bool MpsBuffer::ToHostBuffer(void* dst, std::function<void()> on_done, std::string* error) {
    std::string producer_error;
    if (!WaitForProducer(&producer_error)) {
        if (error) {
            *error = producer_error;
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    if (is_deleted_) {
        NSLog(@"ERROR: ToHostBuffer called on deleted buffer");
        if (error) {
            *error = "ToHostBuffer called on deleted buffer";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }
    if (!metal_buffer_) {
        NSLog(@"ERROR: ToHostBuffer called with null Metal buffer");
        if (error) {
            *error = "ToHostBuffer called with null Metal buffer";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }
    if (!dst) {
        NSLog(@"ERROR: ToHostBuffer called with null destination pointer");
        if (error) {
            *error = "ToHostBuffer called with null destination pointer";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer_;

    void* contents = [buffer contents];
    if (contents) {
        memcpy(dst, contents, byte_size());
        if (error) {
            error->clear();
        }
        if (on_done) {
            on_done();
        }
        return true;
    }

    if (!device_ || !device_->client()) {
        NSLog(@"ERROR: ToHostBuffer called with no device/client for private buffer readback");
        if (error) {
            *error = "ToHostBuffer called with no device/client for private buffer readback";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_->client()->metal_device();
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)device_->client()->command_queue();
    if (!mtl_device || !commandQueue) {
        NSLog(@"ERROR: ToHostBuffer missing Metal device/command queue");
        if (error) {
            *error = "ToHostBuffer missing Metal device/command queue";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    size_t nbytes = byte_size();
    id<MTLBuffer> staging = [mtl_device newBufferWithLength:nbytes options:MTLResourceStorageModeShared];
    if (!staging) {
        NSLog(@"ERROR: ToHostBuffer failed to allocate staging buffer (%zu bytes)", nbytes);
        if (error) {
            *error = "ToHostBuffer failed to allocate staging buffer";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    if (!cmd) {
        NSLog(@"ERROR: ToHostBuffer failed to create command buffer for blit");
        CFRelease((__bridge CFTypeRef)staging);
        if (error) {
            *error = "ToHostBuffer failed to create command buffer for blit";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!blit) {
        NSLog(@"ERROR: ToHostBuffer failed to create blit encoder");
        CFRelease((__bridge CFTypeRef)staging);
        if (error) {
            *error = "ToHostBuffer failed to create blit encoder";
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    [blit copyFromBuffer:buffer sourceOffset:0 toBuffer:staging destinationOffset:0 size:nbytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status == MTLCommandBufferStatusError) {
        NSString* desc = cmd.error ? cmd.error.localizedDescription : @"unknown error";
        NSLog(@"ERROR: ToHostBuffer blit command buffer error: %@", desc);
        CFRelease((__bridge CFTypeRef)staging);
        if (error) {
            *error = std::string("ToHostBuffer blit command buffer error: ") + [desc UTF8String];
        }
        if (on_done) {
            on_done();
        }
        return false;
    }

    memcpy(dst, [staging contents], nbytes);
    CFRelease((__bridge CFTypeRef)staging);

    if (error) {
        error->clear();
    }
    if (on_done) {
        on_done();
    }
    return true;
}

bool MpsBuffer::CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size, std::string* error) {
    std::string producer_error;
    if (!WaitForProducer(&producer_error)) {
        if (error)
            *error = producer_error;
        return false;
    }
    if (is_deleted_ || !metal_buffer_) {
        if (error)
            *error = "CopyRawToHost: buffer is deleted or null";
        return false;
    }
    if (!dst) {
        if (error)
            *error = "CopyRawToHost: null destination pointer";
        return false;
    }

    size_t buf_size = byte_size();
    if (offset < 0 || transfer_size < 0 || (size_t)(offset + transfer_size) > buf_size) {
        if (error)
            *error = "CopyRawToHost: offset/size out of range";
        return false;
    }

    id<MTLBuffer> buffer = (__bridge id<MTLBuffer>)metal_buffer_;
    void* contents = [buffer contents];
    if (contents) {

        memcpy(dst, static_cast<const char*>(contents) + offset, transfer_size);
        if (error)
            error->clear();
        return true;
    }

    if (!device_ || !device_->client()) {
        if (error)
            *error = "CopyRawToHost: no device/client for private buffer readback";
        return false;
    }
    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_->client()->metal_device();
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)device_->client()->command_queue();
    if (!mtl_device || !commandQueue) {
        if (error)
            *error = "CopyRawToHost: missing Metal device/command queue";
        return false;
    }

    id<MTLBuffer> staging = [mtl_device newBufferWithLength:transfer_size
                                                    options:MTLResourceStorageModeShared];
    if (!staging) {
        if (error)
            *error = "CopyRawToHost: failed to allocate staging buffer";
        return false;
    }

    id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
    if (!cmd) {
        CFRelease((__bridge CFTypeRef)staging);
        if (error)
            *error = "CopyRawToHost: failed to create command buffer";
        return false;
    }

    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    if (!blit) {
        CFRelease((__bridge CFTypeRef)staging);
        if (error)
            *error = "CopyRawToHost: failed to create blit encoder";
        return false;
    }

    [blit copyFromBuffer:buffer sourceOffset:offset toBuffer:staging destinationOffset:0 size:transfer_size];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];

    if (cmd.status == MTLCommandBufferStatusError) {
        CFRelease((__bridge CFTypeRef)staging);
        if (error)
            *error = "CopyRawToHost: blit command buffer error";
        return false;
    }

    memcpy(dst, [staging contents], transfer_size);
    CFRelease((__bridge CFTypeRef)staging);
    if (error)
        error->clear();
    return true;
}

std::unique_ptr<MpsBuffer> MpsBuffer::Clone(std::string* error) const {
    std::string producer_error;
    if (!WaitForProducer(&producer_error)) {
        if (error)
            *error = producer_error;
        return nullptr;
    }
    if (is_deleted_ || !metal_buffer_) {
        if (error)
            *error = "Clone: buffer is deleted or null";
        return nullptr;
    }
    if (!device_ || !device_->client()) {
        if (error)
            *error = "Clone: no device/client available";
        return nullptr;
    }

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_->client()->metal_device();
    id<MTLCommandQueue> commandQueue = (__bridge id<MTLCommandQueue>)device_->client()->command_queue();
    if (!mtl_device || !commandQueue) {
        if (error)
            *error = "Clone: missing Metal device/command queue";
        return nullptr;
    }

    size_t nbytes = byte_size();
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)metal_buffer_;

    void* src_contents = [src contents];
    id<MTLBuffer> dst_buffer = nil;

    if (src_contents) {

        dst_buffer = [mtl_device newBufferWithBytes:src_contents
                                             length:nbytes
                                            options:MTLResourceStorageModeShared];
    } else {

        dst_buffer = [mtl_device newBufferWithLength:nbytes options:MTLResourceStorageModePrivate];
        if (dst_buffer) {
            id<MTLCommandBuffer> cmd = [commandQueue commandBuffer];
            if (cmd) {
                id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
                if (blit) {
                    [blit copyFromBuffer:src
                             sourceOffset:0
                                 toBuffer:dst_buffer
                        destinationOffset:0
                                     size:nbytes];
                    [blit endEncoding];
                    [cmd commit];
                    [cmd waitUntilCompleted];
                    if (cmd.status == MTLCommandBufferStatusError) {
                        CFRelease((__bridge CFTypeRef)dst_buffer);
                        if (error)
                            *error = "Clone: blit command buffer error";
                        return nullptr;
                    }
                } else {
                    CFRelease((__bridge CFTypeRef)dst_buffer);
                    if (error)
                        *error = "Clone: failed to create blit encoder";
                    return nullptr;
                }
            } else {
                CFRelease((__bridge CFTypeRef)dst_buffer);
                if (error)
                    *error = "Clone: failed to create command buffer";
                return nullptr;
            }
        }
    }

    if (!dst_buffer) {
        if (error)
            *error = "Clone: failed to allocate Metal buffer";
        return nullptr;
    }

    auto result = std::make_unique<MpsBuffer>(device_, (__bridge void*)dst_buffer, dtype_, dims_);
    CFRelease((__bridge CFTypeRef)dst_buffer);
    if (error)
        error->clear();
    return result;
}

void MpsBuffer::Delete() {
    if (!is_deleted_ && metal_buffer_) {
        (void)WaitForProducer(nullptr);
        if (releaser_) {
            releaser_(metal_buffer_);
        } else {
            CFRelease((__bridge CFTypeRef)metal_buffer_);
        }
        metal_buffer_ = nullptr;
    }
    is_deleted_ = true;
}

} // namespace jax_metallib
