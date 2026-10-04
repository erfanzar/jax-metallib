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

"""Real C-API client ownership and MemoryStats regressions, without GPU work.

To execute, both JAX_METALLIB_TEST_PJRT_LIBRARY (candidate dylib) and
JAX_METALLIB_TEST_PJRT_HEADER_PREFIX (installed API114 header prefix) are required.
There is deliberately no installed-plugin, JAX, or default-header fallback.
Only the public header is compiled; the executable dlopens the exact candidate.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CLIENT_TEST = r"""
#include <xla/pjrt/c/pjrt_c_api.h>

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static_assert(PJRT_API_MINOR == 114, "Use the pinned API114 header for this regression");

static void Check(bool condition, const char* description) {
    if (!condition) {
        std::fprintf(stderr, "FAIL: %s\n", description);
        std::exit(1);
    }
}

static const PJRT_Api* api;

static void Success(PJRT_Error* error, const char* operation) {
    if (!error) return;
    PJRT_Error_Message_Args message{};
    message.struct_size = PJRT_Error_Message_Args_STRUCT_SIZE;
    message.error = error;
    api->PJRT_Error_Message(&message);
    std::fprintf(stderr, "FAIL: %s: %.*s\n", operation,
                 static_cast<int>(message.message_size), message.message ? message.message : "");
    PJRT_Error_Destroy_Args destroy{};
    destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
    destroy.error = error;
    api->PJRT_Error_Destroy(&destroy);
    std::exit(1);
}

static PJRT_Client* CreateClient() {
    PJRT_Client_Create_Args args{};
    args.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
    Success(api->PJRT_Client_Create(&args), "Client_Create");
    Check(args.client != nullptr, "Client_Create returned a client");
    return args.client;
}

static void DestroyClient(PJRT_Client* client) {
    PJRT_Client_Destroy_Args args{};
    args.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
    args.client = client;
    Success(api->PJRT_Client_Destroy(&args), "Client_Destroy");
}

static PJRT_Device* FirstDevice(PJRT_Client* client) {
    PJRT_Client_Devices_Args args{};
    args.struct_size = PJRT_Client_Devices_Args_STRUCT_SIZE;
    args.client = client;
    Success(api->PJRT_Client_Devices(&args), "Client_Devices");
    Check(args.num_devices > 0 && args.devices && args.devices[0], "Client_Devices returned a device");
    return args.devices[0];
}

static PJRT_Memory* FirstMemory(PJRT_Client* client) {
    PJRT_Client_AddressableMemories_Args args{};
    args.struct_size = PJRT_Client_AddressableMemories_Args_STRUCT_SIZE;
    args.client = client;
    Success(api->PJRT_Client_AddressableMemories(&args), "Client_AddressableMemories");
    Check(args.num_addressable_memories > 0 && args.addressable_memories && args.addressable_memories[0],
          "Client_AddressableMemories returned a memory");
    auto* memory = args.addressable_memories[0];
    PJRT_Device_DefaultMemory_Args device_memory{};
    device_memory.struct_size = PJRT_Device_DefaultMemory_Args_STRUCT_SIZE;
    device_memory.device = FirstDevice(client);
    Success(api->PJRT_Device_DefaultMemory(&device_memory), "Device_DefaultMemory");
    Check(device_memory.memory == memory, "default memory belongs to its client");
    Check(memory->vtable && memory->vtable->struct_size >= PJRT_Memory_FunctionTable_STRUCT_SIZE &&
              memory->vtable->get_user_data && memory->vtable->set_user_data,
          "memory userdata callbacks are present");
    return memory;
}

struct OwnedData {
    int* destructions;
};

static void DestroyData(void* pointer) {
    auto* data = static_cast<OwnedData*>(pointer);
    ++*data->destructions;
    delete data;
}

static void TestOwnership() {
    auto* first = CreateClient();
    auto* second = CreateClient();
    Check(first != second, "two Client_Create calls must return distinct owned clients");
    auto* first_memory = FirstMemory(first);
    auto* second_memory = FirstMemory(second);
    Check(first_memory != second_memory, "clients must not share default-memory handles");
    Check(FirstDevice(first) != FirstDevice(second), "clients must not share public device handles");
    PJRT_Device_AddressableMemories_Args first_addressable{}, second_addressable{};
    first_addressable.struct_size = second_addressable.struct_size = PJRT_Device_AddressableMemories_Args_STRUCT_SIZE;
    first_addressable.device = FirstDevice(first);
    second_addressable.device = FirstDevice(second);
    Success(api->PJRT_Device_AddressableMemories(&first_addressable), "first Device_AddressableMemories");
    Success(api->PJRT_Device_AddressableMemories(&second_addressable), "second Device_AddressableMemories");
    Check(first_addressable.num_memories == 1 && second_addressable.num_memories == 1 &&
              first_addressable.memories && second_addressable.memories &&
              first_addressable.memories[0] == first_memory && second_addressable.memories[0] == second_memory,
          "device-owned memory arrays remain isolated across client calls");
    static char key;
    int replaced = 0, first_destroyed = 0, second_destroyed = 0;
    auto* old_data = new OwnedData{&replaced};
    auto* first_data = new OwnedData{&first_destroyed};
    auto* second_data = new OwnedData{&second_destroyed};
    first_memory->vtable->set_user_data(first_memory, &key, old_data, DestroyData);
    second_memory->vtable->set_user_data(second_memory, &key, second_data, DestroyData);
    Check(first_memory->vtable->get_user_data(first_memory, &key) == old_data &&
              second_memory->vtable->get_user_data(second_memory, &key) == second_data,
          "userdata with the same key is isolated between clients");
    first_memory->vtable->set_user_data(first_memory, &key, first_data, DestroyData);
    Check(replaced == 1 && first_destroyed == 0 && second_destroyed == 0,
          "replacement invokes only the replaced userdata destructor once");
    DestroyClient(first);
    Check(replaced == 1 && first_destroyed == 1 && second_destroyed == 0,
          "owning client destruction invokes its memory userdata destructor exactly once");
    Check(FirstMemory(second) == second_memory, "other client remains usable after first client destruction");
    Check(second_memory->vtable->get_user_data(second_memory, &key) == second_data,
          "other client's userdata remains alive");
    DestroyClient(second);
    Check(replaced == 1 && first_destroyed == 1 && second_destroyed == 1,
          "each client destroys only its own userdata exactly once");
}

static void TestStats(bool legacy) {
    auto* client = CreateClient();
    PJRT_Device_MemoryStats_Args args{};
    // API90 ends at peak_pool_bytes_is_set; poison every byte outside that
    // advertised prefix, including padding, to detect writes past struct_size.
    constexpr size_t legacy_size = offsetof(PJRT_Device_MemoryStats_Args, peak_pool_bytes_is_set) +
                                   sizeof(args.peak_pool_bytes_is_set);
    args.struct_size = legacy ? legacy_size : PJRT_Device_MemoryStats_Args_STRUCT_SIZE;
    args.device = FirstDevice(client);
    args.peak_bytes_in_use_is_set = true;
    args.num_allocs_is_set = true;
    args.largest_alloc_size_is_set = true;
    args.bytes_reserved_is_set = true;
    args.peak_bytes_reserved_is_set = true;
    args.bytes_reservable_limit_is_set = true;
    args.largest_free_block_bytes_is_set = true;
    args.pool_bytes_is_set = true;
    args.peak_pool_bytes_is_set = true;
    args.peak_allocated_bytes_is_set = true;
    unsigned char canary[sizeof(args)];
    if (legacy) {
        std::memset(reinterpret_cast<unsigned char*>(&args) + legacy_size, 0xa5, sizeof(args) - legacy_size);
        std::memcpy(canary, &args, sizeof(args));
    }
    Success(api->PJRT_Device_MemoryStats(&args), "Device_MemoryStats");
    Check(!args.peak_bytes_in_use_is_set && !args.num_allocs_is_set && !args.largest_alloc_size_is_set &&
              !args.bytes_reserved_is_set && !args.peak_bytes_reserved_is_set &&
              !args.bytes_reservable_limit_is_set && !args.largest_free_block_bytes_is_set &&
              !args.pool_bytes_is_set && !args.peak_pool_bytes_is_set,
          "MemoryStats clears all unavailable prefix fields");
    if (legacy) {
        Check(std::memcmp(reinterpret_cast<unsigned char*>(&args) + legacy_size,
                          canary + legacy_size, sizeof(args) - legacy_size) == 0,
              "MemoryStats must leave bytes beyond legacy struct_size untouched");
    } else {
        Check(!args.peak_allocated_bytes_is_set, "MemoryStats clears the API114 optional tail flag");
    }
    DestroyClient(client);
}

int main(int argc, char** argv) {
    Check(argc == 3, "expected candidate library and regression case");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!library) {
        std::fprintf(stderr, "FAIL: dlopen candidate: %s\n", dlerror());
        return 1;
    }
    using GetApi = const PJRT_Api* (*)();
    auto get_api = reinterpret_cast<GetApi>(dlsym(library, "GetPjrtApi"));
    Check(get_api != nullptr, "candidate exports GetPjrtApi");
    api = get_api();
    Check(api && api->struct_size >= PJRT_Api_STRUCT_SIZE, "pinned API114 table size is present");
    Check(api->pjrt_api_version.major_version == PJRT_API_MAJOR &&
              api->pjrt_api_version.minor_version == PJRT_API_MINOR, "candidate reports pinned API114");
    Check(api->PJRT_Error_Message && api->PJRT_Error_Destroy && api->PJRT_Client_Create &&
              api->PJRT_Client_Destroy && api->PJRT_Client_Devices && api->PJRT_Client_AddressableMemories &&
              api->PJRT_Device_DefaultMemory && api->PJRT_Device_AddressableMemories && api->PJRT_Device_MemoryStats,
          "all required C-API callbacks are present");
    if (api->PJRT_Plugin_Initialize) {
        PJRT_Plugin_Initialize_Args initialize{};
        initialize.struct_size = PJRT_Plugin_Initialize_Args_STRUCT_SIZE;
        Success(api->PJRT_Plugin_Initialize(&initialize), "Plugin_Initialize");
    }
    if (std::strcmp(argv[2], "ownership") == 0) {
        TestOwnership();
    } else if (std::strcmp(argv[2], "stats-full") == 0) {
        TestStats(false);
    } else if (std::strcmp(argv[2], "stats-legacy") == 0) {
        TestStats(true);
    } else {
        Check(false, "unknown regression case");
    }
    std::printf("PJRT API114 %s checks passed\n", argv[2]);
    // No dlclose: no destructor callbacks run after unloading the plugin.
    return 0;
}
"""


@pytest.fixture(scope="module")
def candidate_harness(tmp_path_factory):
    library_env = "JAX_METALLIB_TEST_PJRT_LIBRARY"
    header_env = "JAX_METALLIB_TEST_PJRT_HEADER_PREFIX"
    if library_env not in os.environ and header_env not in os.environ:
        pytest.skip("optional release-native C-API harness requires an explicit candidate library and API114 headers")
    if not os.environ.get(library_env) or not os.environ.get(header_env):
        pytest.fail(f"Set both {library_env} and {header_env}; this regression never falls back")
    library = Path(os.environ[library_env]).resolve(strict=True)
    header_prefix = Path(os.environ[header_env]).resolve(strict=True)
    header = header_prefix / "include/xla/pjrt/c/pjrt_c_api.h"
    assert library.is_file(), f"Requested candidate library is not a file: {library}"
    assert header.is_file(), f"Requested pinned header is missing: {header}"
    compiler = shutil.which(os.environ.get("CXX", "clang++"))
    assert compiler is not None, "C++ compiler required for real C-API regression"
    directory = tmp_path_factory.mktemp("pjrt_client_ownership")
    source = directory / "pjrt_client_ownership.cc"
    executable = directory / "pjrt_client_ownership"
    source.write_text(CLIENT_TEST)
    command = [
        compiler,
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-I" + str(header_prefix / "include"),
        str(source),
        "-o",
        str(executable),
    ]
    compiled = subprocess.run(command, text=True, capture_output=True, timeout=60, check=False)
    assert compiled.returncode == 0, f"C-API harness compilation failed:\n{compiled.stdout}\n{compiled.stderr}"
    return executable, library


@pytest.mark.parametrize("case", ["ownership", "stats-full", "stats-legacy"])
def test_pjrt_client_ownership(candidate_harness, case):
    executable, library = candidate_harness
    executed = subprocess.run(
        [str(executable), str(library), case], text=True, capture_output=True, timeout=30, check=False
    )
    assert executed.returncode == 0, f"C-API {case} regression failed:\n{executed.stdout}\n{executed.stderr}"
    assert f"PJRT API114 {case} checks passed" in executed.stdout
    print(executed.stdout, end="")
