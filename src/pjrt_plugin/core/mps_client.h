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

#include <memory>
#include <string>
#include <vector>

#ifdef __OBJC__
@class MTLDevice;
@class MPSGraph;
#else
typedef void* MTLDevice;
typedef void* MPSGraph;
#endif

namespace mps {
struct ParsedModule;
}

namespace jax_metallib {

class MpsDevice;
class MpsBuffer;
class MpsExecutable;

class MpsClient {
public:
    static std::unique_ptr<MpsClient> Create();
    ~MpsClient();

    const std::string& platform_name() const {
        return platform_name_;
    }
    const std::string& platform_version() const {
        return platform_version_;
    }
    int process_index() const {
        return 0;
    }

    int device_count() const;
    int addressable_device_count() const;
    MpsDevice* device(int index);
    MpsDevice* addressable_device(int index);
    MpsDevice* LookupDevice(int device_id);

    std::unique_ptr<MpsBuffer> BufferFromHostBuffer(const void* data, int dtype,
                                                    const std::vector<int64_t>& dims,
                                                    const std::vector<int64_t>& byte_strides,
                                                    MpsDevice* device);

    std::unique_ptr<MpsBuffer> CreateUninitializedBuffer(int dtype, const std::vector<int64_t>& dims,
                                                         MpsDevice* device);

    std::unique_ptr<MpsExecutable> CompileStableHLO(mps::ParsedModule module, MpsDevice* device);

    struct MemoryStats {
        int64_t bytes_in_use = 0;
        int64_t bytes_limit = 0;
        bool bytes_limit_is_set = false;
    };

    MemoryStats GetMemoryStats() const;

    void* metal_device() const {
        return metal_device_;
    }
    void* command_queue() const {
        return command_queue_;
    }

private:
    MpsClient();
    bool Initialize();

    std::string platform_name_;
    std::string platform_version_;
    void* metal_device_;
    void* command_queue_;
    std::vector<std::unique_ptr<MpsDevice>> devices_;
};

} // namespace jax_metallib
