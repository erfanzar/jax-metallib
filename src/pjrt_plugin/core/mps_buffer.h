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

#include "pjrt_plugin/core/completion_event.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace jax_metallib {

class MpsDevice;

class MpsBuffer {
public:
    using BufferReleaser = std::function<void(void* metal_buffer)>;

    MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype, const std::vector<int64_t>& dims,
              BufferReleaser releaser = {}, CompletionEvent producer_completion = nullptr);

    // Strided (non-contiguous) buffer constructor.  byte_strides has one entry
    // per dimension; storage_byte_size is the actual MTLBuffer length.
    MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype, const std::vector<int64_t>& dims,
              const std::vector<int64_t>& byte_strides, size_t storage_byte_size,
              BufferReleaser releaser = {}, CompletionEvent producer_completion = nullptr);

    ~MpsBuffer();

    MpsDevice* device() const {
        return device_;
    }
    int dtype() const {
        return dtype_;
    }
    const std::vector<int64_t>& dimensions() const {
        return dims_;
    }
    int64_t element_count() const;
    size_t byte_size() const;

    void* metal_buffer() const {
        return metal_buffer_;
    }

    // Stride accessors.  byte_strides() always returns a valid vector; when the
    // buffer is contiguous it computes row-major strides on the fly.
    const std::vector<int64_t>& byte_strides() const;
    bool is_contiguous() const {
        return byte_strides_.empty();
    }
    // Actual backing MTLBuffer size (may differ from byte_size() for strided buffers).
    size_t storage_byte_size() const;

    bool ToHostBuffer(void* dst, std::function<void()> on_done, std::string* error = nullptr);

    bool CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size, std::string* error = nullptr);

    std::unique_ptr<MpsBuffer> Clone(std::string* error = nullptr) const;

    // Materialize a contiguous copy of a non-contiguous buffer.
    std::unique_ptr<MpsBuffer> MaterializeContiguous(std::string* error = nullptr) const;

    CompletionEvent producer_completion() const {
        return producer_completion_;
    }
    void set_producer_completion(CompletionEvent completion) {
        producer_completion_ = completion ? std::move(completion) : MakeReadyCompletionEvent();
    }
    bool WaitForProducer(std::string* error = nullptr) const;

    bool IsDeleted() const {
        return is_deleted_;
    }
    void Delete();

private:
    MpsDevice* device_;
    void* metal_buffer_;
    int dtype_;
    std::vector<int64_t> dims_;
    std::vector<int64_t> byte_strides_;           // empty → contiguous row-major
    mutable std::vector<int64_t> cached_strides_; // lazily computed for byte_strides()
    size_t storage_byte_size_ = 0;                // 0 → use byte_size()
    BufferReleaser releaser_;
    CompletionEvent producer_completion_;
    bool is_deleted_ = false;
};

size_t DtypeByteSize(int dtype);

// Bytes to allocate for a zero-element tensor. Metal rejects zero-length buffers and
// MPSNDArray binds a tensor as if every zero dimension were 1 ([0,4] f32 needs 16 bytes),
// so a 4-byte placeholder aborts the process. Capped at 64 MiB (4 bytes beyond that).
size_t ZeroElementPlaceholderBytes(int dtype, const std::vector<int64_t>& dims);

// Compute row-major byte strides for a given shape and element size.
std::vector<int64_t> ComputeRowMajorByteStrides(const std::vector<int64_t>& dims, size_t elem_size);

} // namespace jax_metallib
