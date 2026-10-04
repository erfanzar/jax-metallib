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

#include "pjrt_plugin/api/pjrt_handles.h"
#include "pjrt_plugin/core/logging.h"

#include <utility>

// Keep owned host handles here, in an existing API translation unit, so their
// lifetime/ABI can be tested without linking MLIR or the Metal runtime.
MpsPjrtError::MpsPjrtError(const std::string& message, PJRT_Error_Code code) : message(message), code(code) {
#if PJRT_API_MINOR >= 107
    static constexpr PJRT_Error_FunctionTable table = {PJRT_Error_FunctionTable_STRUCT_SIZE,
                                                       PJRT_Error_STRUCT_SIZE,
                                                       nullptr,
                                                       Destroy,
                                                       Message,
                                                       GetCode,
                                                       ForEachPayload};
    vtable = &table;
#endif
}

void MpsPjrtError::Destroy(PJRT_Error* error) {
    delete static_cast<MpsPjrtError*>(error);
}

void MpsPjrtError::Message(const PJRT_Error* error, const char** message, size_t* message_size) {
    const auto* impl = static_cast<const MpsPjrtError*>(error);
    *message = impl->message.c_str();
    *message_size = impl->message.size();
}

PJRT_Error_Code MpsPjrtError::GetCode(const PJRT_Error* error) {
    return static_cast<const MpsPjrtError*>(error)->code;
}

#if PJRT_API_MINOR >= 99
void MpsPjrtError::ForEachPayload(const PJRT_Error*, PJRT_Error_PayloadVisitor, void*) {
    // MakeError stores no payloads; an empty iteration is still a real callback.
}
#endif

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code) {
    return new MpsPjrtError(msg, code);
}

void MPS_Error_Destroy(PJRT_Error_Destroy_Args* args) {
    if (!args->error)
        return;
#if PJRT_API_MINOR >= 107
    args->error->vtable->destroy(args->error);
#else
    MpsPjrtError::Destroy(args->error);
#endif
}

void MPS_Error_Message(PJRT_Error_Message_Args* args) {
    args->message = nullptr;
    args->message_size = 0;
    if (!args->error)
        return;
#if PJRT_API_MINOR >= 107
    args->error->vtable->message(args->error, &args->message, &args->message_size);
#else
    MpsPjrtError::Message(args->error, &args->message, &args->message_size);
#endif
}

PJRT_Error* MPS_Error_GetCode(PJRT_Error_GetCode_Args* args) {
    args->code = PJRT_Error_Code_OK;
    if (args->error) {
#if PJRT_API_MINOR >= 107
        args->code = args->error->vtable->get_code(args->error);
#else
        args->code = MpsPjrtError::GetCode(args->error);
#endif
    }
    return nullptr;
}

#if PJRT_API_MINOR >= 99
PJRT_Error* MPS_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args) {
    if (args->error) {
#if PJRT_API_MINOR >= 107
        args->error->vtable->for_each_payload(args->error, args->visitor, args->user_arg);
#else
        MpsPjrtError::ForEachPayload(args->error, args->visitor, args->user_arg);
#endif
    }
    return nullptr;
}
#endif

MpsPjrtMemory::MpsPjrtMemory(PJRT_Device* device, PJRT_Client* client, int id)
    : device(device), client(client), id(id) {
#if PJRT_API_MINOR >= 105
    static constexpr PJRT_Memory_FunctionTable table = {PJRT_Memory_FunctionTable_STRUCT_SIZE, nullptr,
                                                        PJRT_Memory_STRUCT_SIZE, GetUserData, SetUserData};
    vtable = &table;
#endif
}

MpsPjrtMemory::~MpsPjrtMemory() {
#if PJRT_API_MINOR >= 105
    // Detach before invoking caller destructors: they may re-enter get_user_data.
    decltype(user_data_) detached;
    {
        std::lock_guard<std::mutex> lock(user_data_mutex_);
        detached.swap(user_data_);
    }
#endif
}

#if PJRT_API_MINOR >= 105
void* MpsPjrtMemory::GetUserData(PJRT_Memory* memory, const void* key) {
    auto* impl = static_cast<MpsPjrtMemory*>(memory);
    std::lock_guard<std::mutex> lock(impl->user_data_mutex_);
    auto it = impl->user_data_.find(key);
    // This is a borrowed pointer; callers must synchronize its use with removal.
    return it == impl->user_data_.end() ? nullptr : it->second.get();
}

void MpsPjrtMemory::SetUserData(PJRT_Memory* memory, const void* key, void* data, void (*dtor)(void*)) {
    auto* impl = static_cast<MpsPjrtMemory*>(memory);
    OwnedUserData replacement(data, UserDataDeleter{dtor});
    {
        std::lock_guard<std::mutex> lock(impl->user_data_mutex_);
        auto it = impl->user_data_.find(key);
        if (it == impl->user_data_.end()) {
            if (data)
                impl->user_data_.emplace(key, std::move(replacement));
        } else {
            // Retire the previous ownership outside the mutex. A null data
            // pointer removes the key, and a null destructor is permitted.
            replacement.swap(it->second);
            if (!data)
                impl->user_data_.erase(it);
        }
    }
}
#endif

PJRT_Error* MPS_Memory_Id(PJRT_Memory_Id_Args* args) {
    args->id = args->memory ? static_cast<MpsPjrtMemory*>(args->memory)->id : 0;
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
    args->kind_id = args->memory ? static_cast<MpsPjrtMemory*>(args->memory)->id : 0;
    return nullptr;
}

PJRT_Error* MPS_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args) {
    MPS_LOG_DEBUG(" PJRT_Memory_AddressableByDevices called\n");
    auto* memory = static_cast<MpsPjrtMemory*>(args->memory);
    if (memory && memory->device) {
        // The immutable device slot belongs to this handle, not a shared array.
        args->devices = &memory->device;
        args->num_devices = 1;
    } else {
        args->devices = nullptr;
        args->num_devices = 0;
    }
    return nullptr;
}
