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

PJRT_Error* MPS_Memory_Id(PJRT_Memory_Id_Args* args) {
    args->id = 0;
    return nullptr;
}

PJRT_Error* MPS_Memory_Kind(PJRT_Memory_Kind_Args* args) {
    static const char* kind_str = "device";
    args->kind = kind_str;
    args->kind_size = 6;
    return nullptr;
}

PJRT_Error* MPS_Memory_DebugString(PJRT_Memory_DebugString_Args* args) {
    static const char* str = "MPS Memory";
    args->debug_string = str;
    args->debug_string_size = 10;
    return nullptr;
}

PJRT_Error* MPS_Memory_ToString(PJRT_Memory_ToString_Args* args) {
    static const char* str = "MpsMemory()";
    args->to_string = str;
    args->to_string_size = 11;
    return nullptr;
}

PJRT_Error* MPS_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args) {
    args->kind_id = args->memory ? args->memory->id : 0;
    return nullptr;
}

PJRT_Error* MPS_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Memory_AddressableByDevices called\n");
    if (args->memory && args->memory->device) {
        static PJRT_Device* dev_array[1];
        dev_array[0] = args->memory->device;
        args->devices = dev_array;
        args->num_devices = 1;
    } else {
        args->devices = nullptr;
        args->num_devices = 0;
    }
    return nullptr;
}
