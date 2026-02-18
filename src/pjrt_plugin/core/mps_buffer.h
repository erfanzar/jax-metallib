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
#include <string>
#include <vector>

namespace jax_metallib {

class MpsDevice;

class MpsBuffer {
public:
    using BufferReleaser = std::function<void(void* metal_buffer)>;

    MpsBuffer(MpsDevice* device, void* metal_buffer, int dtype, const std::vector<int64_t>& dims,
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

    bool ToHostBuffer(void* dst, std::function<void()> on_done, std::string* error = nullptr);

    bool CopyRawToHost(void* dst, int64_t offset, int64_t transfer_size, std::string* error = nullptr);

    std::unique_ptr<MpsBuffer> Clone(std::string* error = nullptr) const;

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
    BufferReleaser releaser_;
    CompletionEvent producer_completion_;
    bool is_deleted_ = false;
};

size_t DtypeByteSize(int dtype);

} // namespace jax_metallib
