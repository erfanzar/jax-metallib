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
#include <mutex>
#include <string>
#include <unordered_map>
#include <xla/pjrt/c/pjrt_c_api.h>

// These handles were opaque until API105 (memory) / API107 (error). Complete
// only the older opaque bases; newer headers own their public vtable layout.
#if PJRT_API_MINOR < 105
struct PJRT_Memory {};
#endif
#if PJRT_API_MINOR < 107
struct PJRT_Error {};
#endif

// No C++ virtual functions: the public PJRT base must remain the ABI prefix.
// Allocate/delete these implementations, never their non-virtual public bases.
struct MpsPjrtError : PJRT_Error {
    MpsPjrtError(const std::string& message, PJRT_Error_Code code);

    std::string message;
    PJRT_Error_Code code;

    static void Destroy(PJRT_Error* error);
    static void Message(const PJRT_Error* error, const char** message, size_t* message_size);
    static PJRT_Error_Code GetCode(const PJRT_Error* error);
#if PJRT_API_MINOR >= 99
    static void ForEachPayload(const PJRT_Error* error, PJRT_Error_PayloadVisitor visitor, void* user_arg);
#endif
};

struct MpsPjrtMemory : PJRT_Memory {
    MpsPjrtMemory(PJRT_Device* device, PJRT_Client* client, int id);
    ~MpsPjrtMemory();

    PJRT_Device* device;
    PJRT_Client* client;
    int id;

private:
#if PJRT_API_MINOR >= 105
    struct UserDataDeleter {
        void (*dtor)(void*);
        void operator()(void* data) const {
            if (data && dtor)
                dtor(data);
        }
    };
    using OwnedUserData = std::unique_ptr<void, UserDataDeleter>;

    static void* GetUserData(PJRT_Memory* memory, const void* key);
    static void SetUserData(PJRT_Memory* memory, const void* key, void* data, void (*dtor)(void*));

    std::mutex user_data_mutex_;
    std::unordered_map<const void*, OwnedUserData> user_data_;
#endif
};

PJRT_Error* MakeError(const std::string& msg, PJRT_Error_Code code = PJRT_Error_Code_INTERNAL);
