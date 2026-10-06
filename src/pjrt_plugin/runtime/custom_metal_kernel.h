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

#ifndef JAX_METALLIB_PJRT_PLUGIN_RUNTIME_CUSTOM_METAL_KERNEL_H_
#define JAX_METALLIB_PJRT_PLUGIN_RUNTIME_CUSTOM_METAL_KERNEL_H_

#import <Metal/Metal.h>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace jax_metallib {

// Buffers bind inputs first, then outputs. The caller retains this helper and
// its buffers until GPU completion; Encode only records commands, never waits.
class CustomMetalKernel {
public:
    static std::shared_ptr<CustomMetalKernel> Compile(id<MTLDevice> device, const std::string& backend_config,
                                                      size_t input_count, size_t output_count,
                                                      std::string* error);
    bool Encode(id<MTLComputeCommandEncoder> encoder, const std::vector<id<MTLBuffer>>& buffers,
                std::string* error) const;
    ~CustomMetalKernel();

    CustomMetalKernel(const CustomMetalKernel&) = delete;
    CustomMetalKernel& operator=(const CustomMetalKernel&) = delete;

private:
    struct Impl;
    explicit CustomMetalKernel(std::unique_ptr<Impl> impl);
    std::unique_ptr<Impl> impl_;
};

} // namespace jax_metallib

#endif // JAX_METALLIB_PJRT_PLUGIN_RUNTIME_CUSTOM_METAL_KERNEL_H_
