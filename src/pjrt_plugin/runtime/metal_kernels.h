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

#ifndef JAX_METALLIB_PJRT_PLUGIN_RUNTIME_METAL_KERNELS_H_
#define JAX_METALLIB_PJRT_PLUGIN_RUNTIME_METAL_KERNELS_H_

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mutex>
#include <string>
#include <unordered_map>

namespace jax_metallib {

struct FastNativeKernelCacheEntry {
    id<MTLLibrary> library = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
};

NSString* FastNativeKernelsSource();

// MPP (MetalPerformancePrimitives) GEMM kernel source — requires Metal 3.1+ with tensor support.
// Returns nil if not supported (pre-M3 hardware).
NSString* MppGemmKernelSource();

} // namespace jax_metallib

#endif
