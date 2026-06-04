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

std::vector<int64_t> ComputeRowMajorByteStrides(const std::vector<int64_t>& dims, size_t elem_size) {
    std::vector<int64_t> strides(dims.size());
    if (dims.empty()) {
        return strides;
    }
    int64_t stride = static_cast<int64_t>(elem_size);
    for (int i = static_cast<int>(dims.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= dims[i];
    }
    return strides;
}

// N-dimensional strided → contiguous copy.  Walks the logical shape in
// row-major order and gathers bytes from the source at strided offsets.
static void CopyStridedToContiguous(const char* src, char* dst, const std::vector<int64_t>& dims,
                                    const std::vector<int64_t>& byte_strides, size_t elem_size) {
    int64_t ndim = static_cast<int64_t>(dims.size());
    if (ndim == 0) {
        memcpy(dst, src, elem_size);
        return;
    }

    // Fast path for 2-D (the common transpose case).
    if (ndim == 2) {
        int64_t rows = dims[0];
        int64_t cols = dims[1];
        int64_t s0 = byte_strides[0];
        int64_t s1 = byte_strides[1];
        for (int64_t r = 0; r < rows; ++r) {
            const char* row_src = src + r * s0;
            for (int64_t c = 0; c < cols; ++c) {
                memcpy(dst, row_src + c * s1, elem_size);
                dst += elem_size;
            }
        }
        return;
    }

    // General N-D case: iterate with index vector.
    std::vector<int64_t> idx(ndim, 0);
    int64_t total = 1;
    for (int64_t d : dims) {
        total *= d;
    }
    for (int64_t i = 0; i < total; ++i) {
        int64_t offset = 0;
        for (int64_t d = 0; d < ndim; ++d) {
            offset += idx[d] * byte_strides[d];
        }
        memcpy(dst, src + offset, elem_size);
        dst += elem_size;
        // Increment index in row-major order.
        for (int64_t d = ndim - 1; d >= 0; --d) {
            if (++idx[d] < dims[d]) {
                break;
            }
            idx[d] = 0;
        }
    }
}

// Contiguous constructor (existing).
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

// Strided (non-contiguous) constructor.
MpsBuffer::MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype, const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& byte_strides, size_t storage_byte_size,
                     BufferReleaser releaser, CompletionEvent producer_completion)
    : device_(device), metal_buffer_(metal_buffer), dtype_(dtype), dims_(dims), byte_strides_(byte_strides),
      storage_byte_size_(storage_byte_size), releaser_(std::move(releaser)),
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

size_t MpsBuffer::storage_byte_size() const {
    if (storage_byte_size_ > 0) {
        return storage_byte_size_;
    }
    return byte_size();
}

const std::vector<int64_t>& MpsBuffer::byte_strides() const {
    if (!byte_strides_.empty()) {
        return byte_strides_;
    }
    // Lazily compute row-major strides.
    if (cached_strides_.empty() && !dims_.empty()) {
        cached_strides_ = ComputeRowMajorByteStrides(dims_, DtypeByteSize(dtype_));
    }
    return cached_strides_;
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

    // ---------- Non-contiguous (strided) path ----------
    if (!is_contiguous() && contents) {
        CopyStridedToContiguous(static_cast<const char*>(contents), static_cast<char*>(dst), dims_,
                                byte_strides_, DtypeByteSize(dtype_));
        if (error) {
            error->clear();
        }
        if (on_done) {
            on_done();
        }
        return true;
    }
    if (!is_contiguous() && !contents) {
        // Private-storage non-contiguous: materialize contiguous first, then copy.
        std::string mat_err;
        auto contig = MaterializeContiguous(&mat_err);
        if (!contig) {
            if (error) {
                *error = "ToHostBuffer: failed to materialize contiguous copy: " + mat_err;
            }
            if (on_done) {
                on_done();
            }
            return false;
        }
        return contig->ToHostBuffer(dst, std::move(on_done), error);
    }

    // ---------- Contiguous path (existing) ----------
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

    // Zero-element tensors have byte_size 0 — nothing to copy.
    if (nbytes == 0) {
        if (on_done) {
            on_done();
        }
        return true;
    }

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
    // CopyRawToHost operates on raw byte offsets — not meaningful for strided buffers.
    if (!is_contiguous()) {
        if (error)
            *error = "CopyRawToHost: not supported for non-contiguous buffers";
        return false;
    }

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
    // Non-contiguous clone materializes a contiguous copy.
    if (!is_contiguous()) {
        return MaterializeContiguous(error);
    }

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
    // Zero-element tensors — allocate a minimum 4-byte placeholder for Metal.
    size_t alloc_nbytes = (nbytes == 0) ? 4 : nbytes;
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)metal_buffer_;

    void* src_contents = [src contents];
    id<MTLBuffer> dst_buffer = nil;

    if (src_contents) {

        dst_buffer = [mtl_device newBufferWithBytes:src_contents
                                             length:alloc_nbytes
                                            options:MTLResourceStorageModeShared];
    } else {

        dst_buffer = [mtl_device newBufferWithLength:alloc_nbytes options:MTLResourceStorageModePrivate];
        if (dst_buffer && nbytes > 0) {
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

std::unique_ptr<MpsBuffer> MpsBuffer::MaterializeContiguous(std::string* error) const {
    if (is_contiguous()) {
        return Clone(error);
    }

    std::string producer_error;
    if (!WaitForProducer(&producer_error)) {
        if (error)
            *error = producer_error;
        return nullptr;
    }
    if (is_deleted_ || !metal_buffer_) {
        if (error)
            *error = "MaterializeContiguous: buffer is deleted or null";
        return nullptr;
    }
    if (!device_ || !device_->client()) {
        if (error)
            *error = "MaterializeContiguous: no device/client available";
        return nullptr;
    }

    id<MTLDevice> mtl_device = (__bridge id<MTLDevice>)device_->client()->metal_device();
    if (!mtl_device) {
        if (error)
            *error = "MaterializeContiguous: missing Metal device";
        return nullptr;
    }

    size_t nbytes = byte_size();
    size_t alloc_nbytes = (nbytes == 0) ? 4 : nbytes;

    // Allocate contiguous output buffer.
    id<MTLBuffer> dst_buffer = [mtl_device newBufferWithLength:alloc_nbytes
                                                       options:MTLResourceStorageModeShared];
    if (!dst_buffer) {
        if (error)
            *error = "MaterializeContiguous: failed to allocate buffer";
        return nullptr;
    }

    // The source must be readable — shared storage.
    id<MTLBuffer> src = (__bridge id<MTLBuffer>)metal_buffer_;
    void* src_contents = [src contents];
    if (!src_contents) {
        CFRelease((__bridge CFTypeRef)dst_buffer);
        if (error)
            *error = "MaterializeContiguous: source buffer is not shared-storage";
        return nullptr;
    }

    void* dst_contents = [dst_buffer contents];
    if (!dst_contents) {
        CFRelease((__bridge CFTypeRef)dst_buffer);
        if (error)
            *error = "MaterializeContiguous: dst buffer has no contents";
        return nullptr;
    }

    CopyStridedToContiguous(static_cast<const char*>(src_contents), static_cast<char*>(dst_contents), dims_,
                            byte_strides_, DtypeByteSize(dtype_));

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
