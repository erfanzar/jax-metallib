# Copyright 2026 Erfan Zare Chavoshi (@erfanzar)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Compile and exercise the actual host handle implementation, without JAX/Metal.

Native dependency headers must already be installed. To cover both old and new
PJRT headers, set JAX_METALLIB_TEST_PJRT_PREFIXES to colon-separated prefixes:

    JAX_METALLIB_TEST_PJRT_PREFIXES="$HOME/.local/jax-metallib-deps:/tmp/jm-deps15" \
        pytest -q tests/test_pjrt_handle_abi.py

The executable links only pjrt_memory.cc, not the plugin or any GPU runtime.
Address/undefined-behavior sanitizers also check derived-handle destruction.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPS_PREFIX = Path(os.environ.get("JAX_METALLIB_DEPS_PREFIX", str(Path.home() / ".local/jax-metallib-deps")))
HEADER_PREFIXES = [
    Path(path) for path in os.environ.get("JAX_METALLIB_TEST_PJRT_PREFIXES", str(DEPS_PREFIX)).split(os.pathsep)
]

HOST_TEST = r"""
#include "pjrt_plugin/api/pjrt_handles.h"

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

void MPS_Error_Destroy(PJRT_Error_Destroy_Args* args);
void MPS_Error_Message(PJRT_Error_Message_Args* args);
PJRT_Error* MPS_Error_GetCode(PJRT_Error_GetCode_Args* args);
#if PJRT_API_MINOR >= 99
PJRT_Error* MPS_Error_ForEachPayload(PJRT_Error_ForEachPayload_Args* args);
#endif
PJRT_Error* MPS_Memory_Id(PJRT_Memory_Id_Args* args);
PJRT_Error* MPS_Memory_Kind(PJRT_Memory_Kind_Args* args);
PJRT_Error* MPS_Memory_DebugString(PJRT_Memory_DebugString_Args* args);
PJRT_Error* MPS_Memory_ToString(PJRT_Memory_ToString_Args* args);
PJRT_Error* MPS_Memory_Kind_Id(PJRT_Memory_Kind_Id_Args* args);
PJRT_Error* MPS_Memory_AddressableByDevices(PJRT_Memory_AddressableByDevices_Args* args);

static void Check(bool condition, const char* description) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::abort();
    }
}

#if PJRT_API_MINOR >= 99
static void CountPayload(const char*, size_t, const char*, size_t, void* count) {
    ++*static_cast<int*>(count);
}
#endif

static void TestErrors() {
    static_assert(std::is_base_of_v<PJRT_Error, MpsPjrtError>);
    static_assert(!std::is_polymorphic_v<MpsPjrtError>);
    for (int value = PJRT_Error_Code_CANCELLED; value <= PJRT_Error_Code_UNAUTHENTICATED; ++value) {
        const auto code = static_cast<PJRT_Error_Code>(value);
        const char bytes[] = "host\0error with an owned message longer than the small-string buffer";
        const std::string expected(bytes, sizeof(bytes) - 1);
        std::string source = expected;
        PJRT_Error* error = MakeError(source, code);
        source.assign("changed after MakeError");
        PJRT_Error_Message_Args message{PJRT_Error_Message_Args_STRUCT_SIZE, nullptr, error, nullptr, 0};
        MPS_Error_Message(&message);
        Check(std::string(message.message, message.message_size) == expected, "owned, binary-safe error message");
        PJRT_Error_GetCode_Args get_code{PJRT_Error_GetCode_Args_STRUCT_SIZE, nullptr, error, PJRT_Error_Code_OK};
        Check(MPS_Error_GetCode(&get_code) == nullptr && get_code.code == code, "global error code");
#if PJRT_API_MINOR >= 99
        int payloads = 0;
        PJRT_Error_ForEachPayload_Args payload{PJRT_Error_ForEachPayload_Args_STRUCT_SIZE, nullptr,
                                             error, CountPayload, &payloads};
        Check(MPS_Error_ForEachPayload(&payload) == nullptr && payloads == 0, "global empty payload iteration");
#endif
#if PJRT_API_MINOR >= 107
        Check(error->vtable != nullptr, "initialized error vtable");
        Check(error->vtable->struct_size == PJRT_Error_FunctionTable_STRUCT_SIZE, "error table size");
        Check(error->vtable->instance_size == PJRT_Error_STRUCT_SIZE, "public error instance size");
        Check(error->vtable->extension_start == nullptr, "error table extension");
        Check(error->vtable->destroy && error->vtable->message && error->vtable->get_code &&
                  error->vtable->for_each_payload, "all error callbacks present");
        const char* direct_message = nullptr;
        size_t direct_size = 0;
        error->vtable->message(error, &direct_message, &direct_size);
        Check(direct_message == message.message && direct_size == message.message_size,
              "global/instance message agreement and lifetime");
        Check(error->vtable->get_code(error) == code, "global/instance code agreement");
        error->vtable->for_each_payload(error, CountPayload, &payloads);
        Check(payloads == 0, "instance empty payload iteration");
        PJRT_Error* second = MakeError(expected, code);
        Check(second->vtable == error->vtable, "shared static-lifetime error table");
        second->vtable->destroy(second);
#endif
        PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
        MPS_Error_Destroy(&destroy);
    }
    PJRT_Error_Destroy_Args destroy_null{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, nullptr};
    MPS_Error_Destroy(&destroy_null);
    PJRT_Error_GetCode_Args get_null{PJRT_Error_GetCode_Args_STRUCT_SIZE, nullptr, nullptr, PJRT_Error_Code_INTERNAL};
    Check(MPS_Error_GetCode(&get_null) == nullptr && get_null.code == PJRT_Error_Code_OK, "null error code");
}

#if PJRT_API_MINOR >= 107
struct ForeignError : PJRT_Error {
    bool* destroyed;
    static void Destroy(PJRT_Error* error) {
        auto* impl = static_cast<ForeignError*>(error);
        *impl->destroyed = true;
        delete impl;
    }
    static void Message(const PJRT_Error*, const char** message, size_t* size) {
        *message = "foreign";
        *size = 7;
    }
    static PJRT_Error_Code Code(const PJRT_Error*) { return PJRT_Error_Code_NOT_FOUND; }
    static void Payload(const PJRT_Error*, PJRT_Error_PayloadVisitor visitor, void* arg) {
        visitor("key", 3, "value", 5, arg);
    }
};

static void TestForeignErrorDispatch() {
    static const PJRT_Error_FunctionTable table{PJRT_Error_FunctionTable_STRUCT_SIZE, PJRT_Error_STRUCT_SIZE,
                                               nullptr, ForeignError::Destroy, ForeignError::Message,
                                               ForeignError::Code, ForeignError::Payload};
    bool destroyed = false;
    auto* error = new ForeignError{{&table}, &destroyed};
    PJRT_Error_Message_Args message{PJRT_Error_Message_Args_STRUCT_SIZE, nullptr, error, nullptr, 0};
    MPS_Error_Message(&message);
    Check(std::string(message.message, message.message_size) == "foreign", "global dispatch to foreign message");
    PJRT_Error_GetCode_Args code{PJRT_Error_GetCode_Args_STRUCT_SIZE, nullptr, error, PJRT_Error_Code_OK};
    Check(MPS_Error_GetCode(&code) == nullptr && code.code == PJRT_Error_Code_NOT_FOUND,
          "global dispatch to foreign code");
    int visited = 0;
    PJRT_Error_ForEachPayload_Args payload{PJRT_Error_ForEachPayload_Args_STRUCT_SIZE, nullptr,
                                         error, CountPayload, &visited};
    Check(MPS_Error_ForEachPayload(&payload) == nullptr && visited == 1, "global dispatch to foreign payload");
    PJRT_Error_Destroy_Args destroy{PJRT_Error_Destroy_Args_STRUCT_SIZE, nullptr, error};
    MPS_Error_Destroy(&destroy);
    Check(destroyed, "global dispatch to foreign destruction");
}
#endif

static void TestMemoryMetadata() {
    static_assert(std::is_base_of_v<PJRT_Memory, MpsPjrtMemory>);
    static_assert(!std::is_polymorphic_v<MpsPjrtMemory>);
    // Opaque device tokens; these are never dereferenced or backed by Metal.
    char first_token, second_token;
    auto* first_device = reinterpret_cast<PJRT_Device*>(&first_token);
    auto* second_device = reinterpret_cast<PJRT_Device*>(&second_token);
    MpsPjrtMemory first(first_device, nullptr, 7);
    MpsPjrtMemory second(second_device, nullptr, 12);
    PJRT_Memory_AddressableByDevices_Args first_devices{
        PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE, nullptr, &first, nullptr, 0};
    PJRT_Memory_AddressableByDevices_Args second_devices{
        PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE, nullptr, &second, nullptr, 0};
    Check(MPS_Memory_AddressableByDevices(&first_devices) == nullptr, "first memory devices");
    Check(MPS_Memory_AddressableByDevices(&second_devices) == nullptr, "second memory devices");
    Check(first_devices.num_devices == 1 && first_devices.devices[0] == first_device,
          "borrowed device array remains tied to first memory");
    Check(second_devices.num_devices == 1 && second_devices.devices[0] == second_device,
          "borrowed device array remains tied to second memory");
    PJRT_Memory_Id_Args id{PJRT_Memory_Id_Args_STRUCT_SIZE, nullptr, &second, -1};
    Check(MPS_Memory_Id(&id) == nullptr && id.id == 12, "implementation memory id");
    PJRT_Memory_Kind_Id_Args kind_id{PJRT_Memory_Kind_Id_Args_STRUCT_SIZE, nullptr, &second, -1};
    Check(MPS_Memory_Kind_Id(&kind_id) == nullptr && kind_id.kind_id == 12, "existing memory kind id");
    PJRT_Memory_Kind_Args kind{PJRT_Memory_Kind_Args_STRUCT_SIZE, nullptr, &first, nullptr, 0};
    Check(MPS_Memory_Kind(&kind) == nullptr && std::string(kind.kind, kind.kind_size) == "device", "memory kind");
    PJRT_Memory_DebugString_Args debug{PJRT_Memory_DebugString_Args_STRUCT_SIZE, nullptr, &first, nullptr, 0};
    Check(MPS_Memory_DebugString(&debug) == nullptr &&
              std::string(debug.debug_string, debug.debug_string_size) == "MPS Memory", "memory debug string");
    PJRT_Memory_ToString_Args text{PJRT_Memory_ToString_Args_STRUCT_SIZE, nullptr, &first, nullptr, 0};
    Check(MPS_Memory_ToString(&text) == nullptr &&
              std::string(text.to_string, text.to_string_size) == "MpsMemory()", "memory string");
    MpsPjrtMemory empty(nullptr, nullptr, 0);
    PJRT_Memory_AddressableByDevices_Args no_devices{
        PJRT_Memory_AddressableByDevices_Args_STRUCT_SIZE, nullptr, &empty, nullptr, 99};
    Check(MPS_Memory_AddressableByDevices(&no_devices) == nullptr && no_devices.num_devices == 0 &&
              no_devices.devices == nullptr, "empty memory devices");
}

#if PJRT_API_MINOR >= 105
struct OwnedData {
    std::atomic<int>* destructions;
    PJRT_Memory* reenter = nullptr;
    const void* key = nullptr;
};

static void DestroyData(void* pointer) {
    auto* data = static_cast<OwnedData*>(pointer);
    if (data->reenter) {
        data->reenter->vtable->get_user_data(data->reenter, data->key);
    }
    ++*data->destructions;
    delete data;
}

static void TestMemoryUserData() {
    std::atomic<int> destructions{0};
    static char key_a, key_b, key_c;
    {
        MpsPjrtMemory memory(nullptr, nullptr, 0);
        MpsPjrtMemory other(nullptr, nullptr, 1);
        PJRT_Memory* handle = &memory;
        const auto* table = handle->vtable;
        Check(table && table == other.vtable, "shared static-lifetime memory table");
        Check(table->struct_size == PJRT_Memory_FunctionTable_STRUCT_SIZE, "memory table size");
        Check(table->instance_struct_size == PJRT_Memory_STRUCT_SIZE, "public memory instance size");
        Check(table->extension_start == nullptr && table->get_user_data && table->set_user_data,
              "complete memory table");
        Check(table->get_user_data(handle, &key_a) == nullptr, "absent user-data key");
        auto* first = new OwnedData{&destructions, handle, &key_a};
        auto* second = new OwnedData{&destructions};
        table->set_user_data(handle, &key_a, first, DestroyData);
        Check(table->get_user_data(handle, &key_a) == first, "get attached user data");
        Check(other.vtable->get_user_data(&other, &key_a) == nullptr, "per-memory user data");
        table->set_user_data(handle, &key_a, second, DestroyData);
        Check(destructions == 1 && table->get_user_data(handle, &key_a) == second,
              "replacement destroys old data outside the lock");
        table->set_user_data(handle, &key_a, nullptr, DestroyData);
        Check(destructions == 2 && table->get_user_data(handle, &key_a) == nullptr,
              "null data removes and destroys old entry");
        table->set_user_data(handle, &key_a, nullptr, DestroyData);
        Check(destructions == 2, "null data does not call destructor");
        int borrowed = 42;
        table->set_user_data(handle, &key_b, &borrowed, nullptr);
        Check(table->get_user_data(handle, &key_b) == &borrowed, "nullable user-data destructor");
        table->set_user_data(handle, &key_b, nullptr, nullptr);
        table->set_user_data(handle, &key_c, new OwnedData{&destructions, handle, &key_c}, DestroyData);
    }
    Check(destructions == 3, "memory destruction releases attached data exactly once");

    MpsPjrtMemory concurrent(nullptr, nullptr, 0);
    const auto* table = concurrent.vtable;
    static char shared_key;
    constexpr int thread_count = 4;
    constexpr int iterations = 64;
    std::vector<std::thread> threads;
    for (int i = 0; i < thread_count; ++i) {
        threads.emplace_back([&] {
            char local_key;
            for (int j = 0; j < iterations; ++j) {
                table->set_user_data(&concurrent, &local_key, &local_key, nullptr);
                Check(table->get_user_data(&concurrent, &local_key) == &local_key, "concurrent independent keys");
                table->set_user_data(&concurrent, &shared_key, new OwnedData{&destructions}, DestroyData);
            }
            table->set_user_data(&concurrent, &local_key, nullptr, nullptr);
        });
    }
    for (auto& thread : threads) thread.join();
    table->set_user_data(&concurrent, &shared_key, nullptr, nullptr);
    Check(destructions == 3 + thread_count * iterations, "concurrent replacement owns every value once");
}
#endif

int main() {
    TestErrors();
#if PJRT_API_MINOR >= 107
    TestForeignErrorDispatch();
#endif
    TestMemoryMetadata();
#if PJRT_API_MINOR >= 105
    TestMemoryUserData();
#endif
    std::printf("PJRT API%d host handle ABI checks passed\n", PJRT_API_MINOR);
}
"""


@pytest.mark.parametrize("header_prefix", HEADER_PREFIXES, ids=lambda path: path.name)
def test_pjrt_host_handle_abi(header_prefix, tmp_path):
    compiler = shutil.which(os.environ.get("CXX", "clang++"))
    if compiler is None:
        pytest.skip("C++ compiler required for the host ABI regression")
    header = header_prefix / "include/xla/pjrt/c/pjrt_c_api.h"
    if not header.is_file():
        if "JAX_METALLIB_TEST_PJRT_PREFIXES" in os.environ:
            pytest.fail(f"Requested PJRT header is missing: {header}")
        pytest.skip(f"Native dependency headers not installed: {header}")
    source = tmp_path / "pjrt_handle_abi.cc"
    executable = tmp_path / "pjrt_handle_abi"
    source.write_text(HOST_TEST)
    command = [
        compiler,
        "-std=c++17",
        "-pthread",
        "-O1",
        "-g",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-I" + str(header_prefix / "include"),
        "-I" + str(DEPS_PREFIX / "include"),
        "-I" + str(ROOT / "src"),
        str(source),
        str(ROOT / "src/pjrt_plugin/api/pjrt_memory.cc"),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, text=True, capture_output=True, timeout=120, check=False)
    assert compiled.returncode == 0, f"Host ABI compilation failed:\n{compiled.stdout}\n{compiled.stderr}"
    executed = subprocess.run([str(executable)], text=True, capture_output=True, timeout=30, check=False)
    assert executed.returncode == 0, f"Host ABI regression failed:\n{executed.stdout}\n{executed.stderr}"
    assert "host handle ABI checks passed" in executed.stdout
    print(executed.stdout, end="")
