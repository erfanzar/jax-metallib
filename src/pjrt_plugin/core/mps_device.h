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

#include <string>

namespace jax_metallib {

class MpsClient;

class MpsDevice {
public:
    MpsDevice(MpsClient* client, int id, const std::string& name);
    ~MpsDevice();

    int id() const {
        return id_;
    }
    int local_hardware_id() const {
        return id_;
    }
    const std::string& device_kind() const {
        return device_kind_;
    }
    const std::string& debug_string() const {
        return debug_string_;
    }
    const std::string& to_string() const {
        return debug_string_;
    }

    MpsClient* client() const {
        return client_;
    }

    bool IsAddressable() const {
        return true;
    }

private:
    MpsClient* client_;
    int id_;
    std::string device_kind_;
    std::string debug_string_;
};

} // namespace jax_metallib
