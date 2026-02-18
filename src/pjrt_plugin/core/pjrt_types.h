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
#include "pjrt_plugin/core/mps_buffer.h"
#include "pjrt_plugin/core/mps_client.h"
#include "pjrt_plugin/core/mps_device.h"
#include "pjrt_plugin/core/mps_executable.h"

#include <memory>
#include <string>
#include <vector>
#include <xla/pjrt/c/pjrt_c_api.h>

struct PJRT_TopologyDescription;
struct PJRT_DeviceDescription;
struct PJRT_Memory;

struct PJRT_Client {
    std::unique_ptr<jax_metallib::MpsClient> client;
    std::vector<PJRT_Device*> devices;
    std::vector<PJRT_Memory*> memories;
    PJRT_TopologyDescription* topology;
};

struct PJRT_Device {
    jax_metallib::MpsDevice* device;
    PJRT_Client* client;
    PJRT_DeviceDescription* description;
    PJRT_Memory* default_memory;
};

struct PJRT_DeviceDescription {
    PJRT_Device* device;
};

struct PJRT_Memory {
    PJRT_Device* device;
    PJRT_Client* client;
    int id;
};

struct PJRT_TopologyDescription {
    PJRT_Client* client;
};

struct PJRT_Buffer {
    std::unique_ptr<jax_metallib::MpsBuffer> buffer;
    PJRT_Client* client;
};

struct PJRT_Executable {
    std::unique_ptr<jax_metallib::MpsExecutable> executable;
    PJRT_Client* client;

    mutable std::string optimized_program_cache;

    mutable std::vector<const char*> output_memory_kinds;
    mutable std::vector<size_t> output_memory_kind_sizes;
    mutable std::vector<PJRT_Buffer_Type> output_types;
    mutable std::vector<int64_t> output_dims;
    mutable std::vector<size_t> output_dim_sizes;
    mutable bool output_metadata_initialized = false;

    void initOutputMetadata() const {
        if (output_metadata_initialized)
            return;
        size_t num_outputs = executable ? executable->num_outputs() : 1;
        output_memory_kinds.resize(num_outputs, "device");
        output_memory_kind_sizes.resize(num_outputs, 6);
        output_types.resize(num_outputs, PJRT_Buffer_Type_F32);
        output_dims.resize(num_outputs * 8, 0);
        output_dim_sizes.resize(num_outputs, 0);
        output_metadata_initialized = true;
    }
};

struct PJRT_LoadedExecutable {
    PJRT_Executable* executable;
    PJRT_Client* client;
    std::vector<PJRT_Device*> addressable_devices;
};

struct PJRT_Event {
    jax_metallib::CompletionEvent completion;
};

struct PJRT_Error {
    std::string message;
    PJRT_Error_Code code;
};

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL);

PJRT_Client* GetOrCreateDefaultClient();
PJRT_Client* GetClient(PJRT_Client* client);

extern const char* const kPlatformName;
extern const char* const kPlatformVersion;
