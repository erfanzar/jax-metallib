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

#import "pjrt_plugin/core/mps_device.h"

#import "pjrt_plugin/core/mps_client.h"

#import <Metal/Metal.h>

namespace jax_metallib {

MpsDevice::MpsDevice(MpsClient* client, int id, const std::string& name)
    : client_(client), id_(id), device_kind_("GPU") {
    debug_string_ = "MPS:" + std::to_string(id) + " " + name;
}

MpsDevice::~MpsDevice() = default;

} // namespace jax_metallib
