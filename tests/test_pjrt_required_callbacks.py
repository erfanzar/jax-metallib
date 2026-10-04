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

"""Exercise version-required callbacks through the actual exported PJRT table.

Release gates set JAX_METALLIB_LIBRARY_PATH and JAX_METALLIB_TEST_PJRT_PREFIX
explicitly. No library/header fallback is allowed once either is requested.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

CPP = r"""
#include <xla/pjrt/c/pjrt_c_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

static void Check(bool ok, const char* message) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", message);
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    Check(argc == 3, "library and case required");
    void* library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    Check(library != nullptr, "load explicit candidate");
    auto get_api = reinterpret_cast<const PJRT_Api* (*)()>(dlsym(library, "GetPjrtApi"));
    Check(get_api != nullptr, "exported GetPjrtApi");
    const PJRT_Api* api = get_api();
    Check(api && api->pjrt_api_version.major_version == PJRT_API_MAJOR &&
          api->pjrt_api_version.minor_version == PJRT_API_MINOR,
          "candidate/header versions match exactly");
#if PJRT_API_MINOR >= 101
    if (std::strcmp(argv[2], "fingerprint") == 0) {
        Check(api->PJRT_TopologyDescription_Fingerprint != nullptr,
              "API101+ topology Fingerprint callback present");
        PJRT_Client_Create_Args create{};
        create.struct_size = PJRT_Client_Create_Args_STRUCT_SIZE;
        Check(api->PJRT_Client_Create(&create) == nullptr && create.client,
              "client creation");
        PJRT_Client_TopologyDescription_Args topology{};
        topology.struct_size = PJRT_Client_TopologyDescription_Args_STRUCT_SIZE;
        topology.client = create.client;
        Check(api->PJRT_Client_TopologyDescription(&topology) == nullptr && topology.topology,
              "client topology");
        PJRT_TopologyDescription_Fingerprint_Args fp{};
        fp.struct_size = PJRT_TopologyDescription_Fingerprint_Args_STRUCT_SIZE;
        fp.topology = topology.topology;
        Check(api->PJRT_TopologyDescription_Fingerprint(&fp) == nullptr,
              "topology fingerprint succeeds");
        uint64_t first = fp.fingerprint;
        Check(api->PJRT_TopologyDescription_Fingerprint(&fp) == nullptr && fp.fingerprint == first,
              "fingerprint repeatability");
        std::printf("FINGERPRINT %llu\n", static_cast<unsigned long long>(first));
        fp.topology = nullptr;
        PJRT_Error* error = api->PJRT_TopologyDescription_Fingerprint(&fp);
        Check(error != nullptr, "null topology gives structured error");
        PJRT_Error_GetCode_Args code{};
        code.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
        code.error = error;
        Check(api->PJRT_Error_GetCode(&code) == nullptr && code.code == PJRT_Error_Code_INVALID_ARGUMENT,
              "invalid topology error code");
        PJRT_Error_Destroy_Args destroy_error{};
        destroy_error.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroy_error.error = error;
        api->PJRT_Error_Destroy(&destroy_error);
        PJRT_Client_Destroy_Args destroy{};
        destroy.struct_size = PJRT_Client_Destroy_Args_STRUCT_SIZE;
        destroy.client = create.client;
        Check(api->PJRT_Client_Destroy(&destroy) == nullptr, "client destruction");
        return 0;
    }
#endif
#if PJRT_API_MINOR >= 102
    if (std::strcmp(argv[2], "parameter_memory_kinds") == 0) {
        Check(api->PJRT_Executable_ParameterMemoryKinds != nullptr,
              "API102+ ParameterMemoryKinds callback present");
        PJRT_Executable_ParameterMemoryKinds_Args args{};
        args.struct_size = PJRT_Executable_ParameterMemoryKinds_Args_STRUCT_SIZE;
        PJRT_Error* error = api->PJRT_Executable_ParameterMemoryKinds(&args);
        Check(error != nullptr, "null executable gives structured error");
        PJRT_Error_GetCode_Args code{};
        code.struct_size = PJRT_Error_GetCode_Args_STRUCT_SIZE;
        code.error = error;
        Check(api->PJRT_Error_GetCode(&code) == nullptr && code.code == PJRT_Error_Code_INVALID_ARGUMENT,
              "invalid executable error code");
        PJRT_Error_Destroy_Args destroy{};
        destroy.struct_size = PJRT_Error_Destroy_Args_STRUCT_SIZE;
        destroy.error = error;
        api->PJRT_Error_Destroy(&destroy);
        return 0;
    }
#endif
    Check(false, "case unavailable in selected header");
}
"""


@pytest.fixture(scope="module")
def callback_probe(tmp_path_factory):
    library_setting = os.environ.get("JAX_METALLIB_LIBRARY_PATH")
    prefix_setting = os.environ.get("JAX_METALLIB_TEST_PJRT_PREFIX")
    if library_setting is None and prefix_setting is None:
        pytest.skip("release callback checks require an explicit native library and matching PJRT headers")
    assert library_setting and prefix_setting, "both explicit release library and header prefix are required"
    library = Path(library_setting).resolve()
    header = Path(prefix_setting).resolve() / "include/xla/pjrt/c/pjrt_c_api.h"
    assert library.is_file(), f"refusing candidate fallback: {library}"
    assert header.is_file(), f"refusing header fallback: {header}"
    compiler = shutil.which("clang++")
    assert compiler, "clang++ is required for the release callback gate"
    work = tmp_path_factory.mktemp("pjrt-required-callbacks")
    source = work / "probe.cc"
    source.write_text(CPP)
    executable = work / "probe"
    compiled = subprocess.run(
        [compiler, "-std=c++17", "-I", str(header.parents[3]), str(source), "-o", str(executable)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return executable, library


def _run(probe, case):
    executable, library = probe
    result = subprocess.run(
        [str(executable), str(library), case], capture_output=True, text=True, timeout=60, check=False
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_topology_fingerprint_is_stable_across_processes(callback_probe):
    assert _run(callback_probe, "fingerprint") == _run(callback_probe, "fingerprint")


def test_parameter_memory_kinds_reports_invalid_executable(callback_probe):
    _run(callback_probe, "parameter_memory_kinds")
