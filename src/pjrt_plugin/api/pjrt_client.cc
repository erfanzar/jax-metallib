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

#include "pjrt_plugin/core/issue_url.h"
#include "pjrt_plugin/core/logging.h"
#include "pjrt_plugin/core/pjrt_types.h"
#include "pjrt_plugin/core/stablehlo_parser.h"

#include <cstring>

void MPS_Error_Destroy(PJRT_Error_Destroy_Args* args) {
    delete args->error;
}

void MPS_Error_Message(PJRT_Error_Message_Args* args) {
    if (args->error) {
        args->message = args->error->message.c_str();
        args->message_size = args->error->message.size();
    }
}

PJRT_Error* MPS_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    args->code = args->error ? args->error->code : PJRT_Error_Code_OK;
    return nullptr;
}

PJRT_Error* MPS_Plugin_Initialize(PJRT_Plugin_Initialize_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Plugin_Initialize\n");
    return nullptr;
}

PJRT_Error* MPS_Plugin_Attributes(PJRT_Plugin_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

PJRT_Error* MPS_Client_Create(PJRT_Client_Create_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_Create called, args=%p\n", (void*)args);

    PJRT_Client* client = GetOrCreateDefaultClient();
    if (!client) {
        return MakeError("Failed to create MPS client");
    }

    MPS_LOG_DEBUG(" PJRT_Client_Create setting client=%p\n", (void*)client);
    args->client = client;
    MPS_LOG_DEBUG(" PJRT_Client_Create returning nullptr (success)\n");
    return nullptr;
}

PJRT_Error* MPS_Client_Destroy(PJRT_Client_Destroy_Args* args) {

    return nullptr;
}

PJRT_Error* MPS_Client_PlatformName(PJRT_Client_PlatformName_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_PlatformName called\n");
    args->platform_name = kPlatformName;
    args->platform_name_size = strlen(kPlatformName);
    return nullptr;
}

PJRT_Error* MPS_Client_ProcessIndex(PJRT_Client_ProcessIndex_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_ProcessIndex called\n");
    args->process_index = 0;
    return nullptr;
}

PJRT_Error* MPS_Client_PlatformVersion(PJRT_Client_PlatformVersion_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_PlatformVersion called\n");
    args->platform_version = kPlatformVersion;
    args->platform_version_size = strlen(kPlatformVersion);
    return nullptr;
}

PJRT_Error* MPS_Client_Devices(PJRT_Client_Devices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_Devices called, client=%p\n", (void*)args->client);
    PJRT_Client* client = GetClient(args->client);
    if (!client) {
        args->devices = nullptr;
        args->num_devices = 0;
        MPS_LOG_DEBUG(" PJRT_Client_Devices: no client, returning 0\n");
        return nullptr;
    }
    MPS_LOG_DEBUG(" PJRT_Client_Devices: %zu devices, data=%p\n", client->devices.size(),
                  (void*)client->devices.data());
    for (size_t i = 0; i < client->devices.size(); i++) {
        MPS_LOG_DEBUG(" PJRT_Client_Devices: device[%zu]=%p\n", i, (void*)client->devices[i]);
    }
    args->devices = client->devices.data();
    args->num_devices = client->devices.size();
    MPS_LOG_DEBUG(" PJRT_Client_Devices returning\n");
    return nullptr;
}

PJRT_Error* MPS_Client_AddressableDevices(PJRT_Client_AddressableDevices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_AddressableDevices called\n");
    PJRT_Client* client = GetClient(args->client);
    if (!client) {
        args->addressable_devices = nullptr;
        args->num_addressable_devices = 0;
        return nullptr;
    }
    args->addressable_devices = client->devices.data();
    args->num_addressable_devices = client->devices.size();
    MPS_LOG_DEBUG(" PJRT_Client_AddressableDevices returning %zu\n", client->devices.size());
    return nullptr;
}

PJRT_Error* MPS_Client_LookupDevice(PJRT_Client_LookupDevice_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_LookupDevice called, id=%d\n", (int)args->id);
    PJRT_Client* client = GetClient(args->client);
    if (client && (size_t)args->id < client->devices.size()) {
        args->device = client->devices[args->id];
        MPS_LOG_DEBUG(" Returning device %p\n", (void*)args->device);
    } else {
        args->device = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_LookupAddressableDevice(PJRT_Client_LookupAddressableDevice_Args* args) {
    PJRT_Client* client = GetClient(args->client);
    if (client && (size_t)args->local_hardware_id < client->devices.size()) {
        args->addressable_device = client->devices[args->local_hardware_id];
    } else {
        args->addressable_device = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_AddressableMemories(PJRT_Client_AddressableMemories_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_AddressableMemories called\n");
    PJRT_Client* client = GetClient(args->client);
    if (client && !client->memories.empty()) {
        args->addressable_memories = client->memories.data();
        args->num_addressable_memories = client->memories.size();
    } else {
        args->addressable_memories = nullptr;
        args->num_addressable_memories = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_Compile(PJRT_Client_Compile_Args* args) {
    MPS_LOG_INFO("Compiling StableHLO program\n");

    PJRT_Client* client = GetClient(args->client);
    if (!client || !client->client) {
        return MakeError("No MPS client available for compilation. Metal GPU may not be present.");
    }
    if (!client->client->metal_device()) {
        return MakeError("No Metal GPU device available. MPS backend requires Apple Silicon or AMD GPU.");
    }

    std::string format_str(args->program->format, args->program->format_size);
    MPS_LOG_DEBUG(" Program format: %s (size=%zu)\n", format_str.c_str(), args->program->format_size);
    MPS_LOG_DEBUG(" Program code size: %zu\n", args->program->code_size);

    mps::ParsedModule parsed_module;

    if (format_str == "mlir") {

        parsed_module = mps::parseStableHLOBytecode(args->program->code, args->program->code_size);
    } else if (format_str == "hlo" || format_str == "hlo_with_config") {

        std::string program_str(args->program->code, args->program->code_size);
        parsed_module = mps::parseStableHLOText(program_str);
    } else {
        return MakeError("Unknown program format: " + format_str);
    }

    if (!parsed_module.ok()) {
        return MakeError("Failed to parse StableHLO program. "
                         "The program may be malformed or use an unsupported format.");
    }

    if (!parsed_module.unsupported_ops.empty()) {
        MPS_LOG_DEBUG(" Found %zu unsupported operations:\n", parsed_module.unsupported_ops.size());
        for (size_t i = 0; i < parsed_module.unsupported_ops.size(); i++) {
            MPS_LOG_DEBUG("   - %s\n", parsed_module.unsupported_ops[i].c_str());
        }
    }

    if (!parsed_module.unsupported_ops.empty()) {
        return MakeError(jax_metallib::UnsupportedOpsMessage(parsed_module.unsupported_ops));
    }

    auto mps_exec = client->client->CompileStableHLO(std::move(parsed_module), nullptr);

    if (!mps_exec) {
        return MakeError("Failed to compile StableHLO to MPS");
    }

    if (!mps_exec->IsValid()) {
        return MakeError("MPS compilation failed: " + mps_exec->error());
    }

    auto* executable = new PJRT_Executable();
    executable->executable = std::move(mps_exec);
    executable->client = client;

    auto* loaded_executable = new PJRT_LoadedExecutable();
    loaded_executable->executable = executable;
    loaded_executable->client = client;
    loaded_executable->addressable_devices = client->devices;

    args->executable = loaded_executable;
    MPS_LOG_INFO("Compilation successful\n");
    return nullptr;
}

PJRT_Error* MPS_Client_DefaultDeviceAssignment(PJRT_Client_DefaultDeviceAssignment_Args* args) {

    if (args->default_assignment && args->default_assignment_size > 0) {
        args->default_assignment[0] = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Client_BufferFromHostBuffer(PJRT_Client_BufferFromHostBuffer_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Client_BufferFromHostBuffer\n");

    PJRT_Client* client = GetClient(args->client);
    if (!client || !client->client) {
        return MakeError("No MPS client available. Metal GPU may not be present.");
    }
    if (!client->client->metal_device()) {
        return MakeError("No Metal GPU device available. MPS backend requires Apple Silicon or AMD GPU.");
    }
    if (!args->data) {
        return MakeError("Cannot create buffer: null data pointer provided");
    }

    std::vector<int64_t> dims(args->dims, args->dims + args->num_dims);
    std::vector<int64_t> byte_strides;
    if (args->byte_strides && args->num_byte_strides > 0) {
        byte_strides.assign(args->byte_strides, args->byte_strides + args->num_byte_strides);
    }

    auto mps_buffer =
        client->client->BufferFromHostBuffer(args->data, static_cast<int>(args->type), dims, byte_strides,
                                             args->device ? args->device->device : nullptr);

    if (!mps_buffer) {
        return MakeError("Failed to create Metal buffer. GPU memory may be exhausted.");
    }

    auto* buffer = new PJRT_Buffer();
    buffer->buffer = std::move(mps_buffer);
    buffer->client = client;

    args->buffer = buffer;

    auto* event = new PJRT_Event();
    event->completion = jax_metallib::MakeReadyCompletionEvent();
    args->done_with_host_buffer = event;

    return nullptr;
}

PJRT_Error* MPS_Client_CreateViewOfDeviceBuffer(PJRT_Client_CreateViewOfDeviceBuffer_Args* args) {
    PJRT_Client* client = GetClient(args->client);
    if (!client || !client->client) {
        return MakeError("CreateViewOfDeviceBuffer: no client available");
    }
    if (!args->device_buffer_ptr) {
        return MakeError("CreateViewOfDeviceBuffer: null device_buffer_ptr",
                         PJRT_Error_Code_INVALID_ARGUMENT);
    }

    std::vector<int64_t> dims(args->dims, args->dims + args->num_dims);
    int dtype = static_cast<int>(args->element_type);

    void* buffer_ptr = args->device_buffer_ptr;
    auto on_delete = args->on_delete_callback;
    void* on_delete_arg = args->on_delete_callback_arg;

    jax_metallib::MpsBuffer::BufferReleaser releaser;
    if (on_delete) {
        releaser = [buffer_ptr, on_delete, on_delete_arg](void*) { on_delete(buffer_ptr, on_delete_arg); };
    } else {

        releaser = [](void*) {};
    }

    jax_metallib::MpsDevice* device =
        args->device && args->device->device ? args->device->device : client->client->device(0);

    auto mps_buffer = std::make_unique<jax_metallib::MpsBuffer>(device, args->device_buffer_ptr, dtype, dims,
                                                                std::move(releaser));

    auto* buffer = new PJRT_Buffer();
    buffer->buffer = std::move(mps_buffer);
    buffer->client = client;
    args->buffer = buffer;

    return nullptr;
}

PJRT_Error* MPS_Client_CreateUninitializedBuffer(PJRT_Client_CreateUninitializedBuffer_Args* args) {
    PJRT_Client* client = GetClient(args->client);
    if (!client || !client->client) {
        return MakeError("CreateUninitializedBuffer: no client available");
    }
    if (!client->client->metal_device()) {
        return MakeError("CreateUninitializedBuffer: no Metal device");
    }

    std::vector<int64_t> dims(args->shape_dims, args->shape_dims + args->shape_num_dims);
    int dtype = static_cast<int>(args->shape_element_type);

    jax_metallib::MpsDevice* device =
        args->device && args->device->device ? args->device->device : client->client->device(0);

    auto mps_buffer = client->client->CreateUninitializedBuffer(dtype, dims, device);
    if (!mps_buffer) {
        return MakeError("CreateUninitializedBuffer: failed to allocate Metal buffer");
    }

    auto* buffer = new PJRT_Buffer();
    buffer->buffer = std::move(mps_buffer);
    buffer->client = client;
    args->buffer = buffer;

    return nullptr;
}

PJRT_Error* MPS_Client_CreateErrorBuffer(PJRT_Client_CreateErrorBuffer_Args* args) {
    PJRT_Client* client = GetClient(args->client);
    if (!client) {
        return MakeError("CreateErrorBuffer: no client available");
    }

    std::vector<int64_t> dims(args->shape_dims, args->shape_dims + args->shape_num_dims);
    int dtype = static_cast<int>(args->shape_element_type);

    auto error_completion = jax_metallib::MakePendingCompletionEvent();
    std::string error_msg(args->error_message, args->error_message_size);
    jax_metallib::MarkCompletionEventError(error_completion, std::move(error_msg));

    jax_metallib::MpsDevice* device = client->client ? client->client->device(0) : nullptr;

    auto mps_buffer = std::make_unique<jax_metallib::MpsBuffer>(
        device, nullptr, dtype, dims, jax_metallib::MpsBuffer::BufferReleaser{}, std::move(error_completion));

    auto* buffer = new PJRT_Buffer();
    buffer->buffer = std::move(mps_buffer);
    buffer->client = client;
    args->buffer = buffer;

    return nullptr;
}

PJRT_Error* MPS_Compile(PJRT_Compile_Args* args) {
    return MakeError("PJRT_Compile not implemented", PJRT_Error_Code_UNIMPLEMENTED);
}
