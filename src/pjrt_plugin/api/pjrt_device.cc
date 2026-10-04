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

#include "pjrt_plugin/core/logging.h"
#include "pjrt_plugin/core/pjrt_types.h"

PJRT_Error* MPS_DeviceDescription_Id(PJRT_DeviceDescription_Id_Args* args) {
    MPS_LOG_DEBUG(" PJRT_DeviceDescription_Id called\n");
    if (args->device_description && args->device_description->device) {
        args->id =
            args->device_description->device->device ? args->device_description->device->device->id() : 0;
    } else {
        args->id = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_ProcessIndex(PJRT_DeviceDescription_ProcessIndex_Args* args) {
    args->process_index = 0;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_Attributes(PJRT_DeviceDescription_Attributes_Args* args) {
    args->num_attributes = 0;
    args->attributes = nullptr;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_Kind(PJRT_DeviceDescription_Kind_Args* args) {
    static const char* kind = "gpu";
    args->device_kind = kind;
    args->device_kind_size = 3;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_DebugString(PJRT_DeviceDescription_DebugString_Args* args) {
    static const char* str = "MPS:0";
    args->debug_string = str;
    args->debug_string_size = 5;
    return nullptr;
}

PJRT_Error* MPS_DeviceDescription_ToString(PJRT_DeviceDescription_ToString_Args* args) {
    static const char* str = "MpsDevice(id=0)";
    args->to_string = str;
    args->to_string_size = 15;
    return nullptr;
}

PJRT_Error* MPS_Device_GetDescription(PJRT_Device_GetDescription_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Device_GetDescription called, device=%p\n", (void*)args->device);
    if (args->device) {
        args->device_description = args->device->description;
        MPS_LOG_DEBUG(" Returning description=%p\n", (void*)args->device_description);
    } else {
        args->device_description = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_IsAddressable(PJRT_Device_IsAddressable_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Device_IsAddressable called, device=%p\n", (void*)args->device);
    args->is_addressable = true;
    MPS_LOG_DEBUG(" PJRT_Device_IsAddressable returning\n");
    return nullptr;
}

PJRT_Error* MPS_Device_LocalHardwareId(PJRT_Device_LocalHardwareId_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Device_LocalHardwareId called, device=%p\n", (void*)args->device);
    args->local_hardware_id =
        args->device && args->device->device ? args->device->device->local_hardware_id() : 0;
    MPS_LOG_DEBUG(" PJRT_Device_LocalHardwareId returning %d\n", args->local_hardware_id);
    return nullptr;
}

PJRT_Error* MPS_Device_AddressableMemories(PJRT_Device_AddressableMemories_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Device_AddressableMemories called\n");
    if (args->device && args->device->default_memory) {
        args->memories = &args->device->default_memory;
        args->num_memories = 1;
    } else {
        args->memories = nullptr;
        args->num_memories = 0;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_DefaultMemory(PJRT_Device_DefaultMemory_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Device_DefaultMemory called, device=%p\n", (void*)args->device);
    if (args->device) {
        args->memory = args->device->default_memory;
        MPS_LOG_DEBUG(" Returning memory=%p\n", (void*)args->memory);
    } else {
        args->memory = nullptr;
    }
    return nullptr;
}

PJRT_Error* MPS_Device_MemoryStats(PJRT_Device_MemoryStats_Args* args) {
    if (!args->device || !args->device->client || !args->device->client->client) {
        return MakeError("MemoryStats: no client available", PJRT_Error_Code_INTERNAL);
    }

    auto stats = args->device->client->client->GetMemoryStats();

    args->bytes_in_use = stats.bytes_in_use;

    args->peak_bytes_in_use_is_set = false;
    args->num_allocs_is_set = false;
    args->largest_alloc_size_is_set = false;

    args->bytes_limit = stats.bytes_limit;
    args->bytes_limit_is_set = stats.bytes_limit_is_set;

    args->bytes_reserved_is_set = false;
    args->peak_bytes_reserved_is_set = false;
    args->bytes_reservable_limit_is_set = false;
    args->largest_free_block_bytes_is_set = false;
    args->pool_bytes_is_set = false;
    args->peak_pool_bytes_is_set = false;
#if PJRT_API_MINOR >= 114
    // Older callers advertise only the prefix through peak_pool_bytes_is_set.
    if (args->struct_size >= PJRT_STRUCT_SIZE(PJRT_Device_MemoryStats_Args, peak_allocated_bytes_is_set)) {
        args->peak_allocated_bytes_is_set = false;
    }
#endif

    return nullptr;
}
