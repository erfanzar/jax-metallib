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

#import "pjrt_plugin/runtime/metal_kernels.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>

namespace jax_metallib {

static std::mutex g_fused_cache_mu;
static std::unordered_map<uintptr_t, std::unordered_map<std::string, id<MTLComputePipelineState>>>
    g_fused_cache;

NSString* FastNativeKernelsSource() {
    static const char* kSource = R"METAL(
#include <metal_stdlib>
using namespace metal;

kernel void unary_neg_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = -x[gid];
}
kernel void unary_neg_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = -x[gid];
}
kernel void unary_floor_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = floor(x[gid]);
}
kernel void unary_floor_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = floor(x[gid]);
}
kernel void unary_rint_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = rint(x[gid]);
}
kernel void unary_rint_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = rint(x[gid]);
}
kernel void unary_rsqrt_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = rsqrt(x[gid]);
}
kernel void unary_rsqrt_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                            constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = rsqrt(x[gid]);
}
kernel void unary_sign_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    half v = x[gid];
    y[gid] = v > (half)0.0 ? (half)1.0 : (v < (half)0.0 ? (half)-1.0 : (half)0.0);
  }
}
kernel void unary_sign_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    float v = x[gid];
    y[gid] = v > 0.0f ? 1.0f : (v < 0.0f ? -1.0f : 0.0f);
  }
}
kernel void unary_square_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                             constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    half v = x[gid];
    y[gid] = v * v;
  }
}
kernel void unary_square_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                             constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) {
    float v = x[gid];
    y[gid] = v * v;
  }
}
kernel void unary_abs_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = abs(x[gid]);
}
kernel void unary_abs_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = abs(x[gid]);
}
kernel void unary_exp_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = exp(x[gid]);
}
kernel void unary_exp_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = exp(x[gid]);
}
kernel void unary_log_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = log(x[gid]);
}
kernel void unary_log_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = log(x[gid]);
}
kernel void unary_tanh_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = tanh(x[gid]);
}
kernel void unary_tanh_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = tanh(x[gid]);
}
kernel void unary_ceil_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = ceil(x[gid]);
}
kernel void unary_ceil_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = ceil(x[gid]);
}
kernel void unary_sin_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = sin(x[gid]);
}
kernel void unary_sin_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = sin(x[gid]);
}
kernel void unary_cos_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = cos(x[gid]);
}
kernel void unary_cos_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                          constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = cos(x[gid]);
}
kernel void unary_sqrt_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = sqrt(x[gid]);
}
kernel void unary_sqrt_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                           constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = sqrt(x[gid]);
}
kernel void unary_sigmoid_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) { half v = x[gid]; y[gid] = 1.0h / (1.0h + exp(-v)); }
}
kernel void unary_sigmoid_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                              constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) { float v = x[gid]; y[gid] = 1.0f / (1.0f + exp(-v)); }
}
kernel void binary_add_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] + b[gid];
}
kernel void binary_add_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] + b[gid];
}
kernel void binary_sub_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] - b[gid];
}
kernel void binary_sub_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] - b[gid];
}
kernel void binary_mul_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] * b[gid];
}
kernel void binary_mul_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] * b[gid];
}
kernel void binary_div_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] / b[gid];
}
kernel void binary_div_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = a[gid] / b[gid];
}
kernel void binary_max_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = max(a[gid], b[gid]);
}
kernel void binary_max_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = max(a[gid], b[gid]);
}
kernel void binary_min_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = min(a[gid], b[gid]);
}
kernel void binary_min_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = min(a[gid], b[gid]);
}
kernel void ternary_select_f16(device const uchar* p [[buffer(0)]], device const half* a [[buffer(1)]],
                               device const half* b [[buffer(2)]], device half* y [[buffer(3)]],
                               constant uint& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = p[gid] ? a[gid] : b[gid];
}
kernel void ternary_select_f32(device const uchar* p [[buffer(0)]], device const float* a [[buffer(1)]],
                               device const float* b [[buffer(2)]], device float* y [[buffer(3)]],
                               constant uint& n [[buffer(4)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = p[gid] ? a[gid] : b[gid];
}
kernel void unary_square_vec4_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                  constant uint& n4 [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid < n4) {
    float4 v = x[gid];
    y[gid] = v * v;
  }
}
kernel void unary_square_vec4_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                  constant uint& n4 [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid < n4) {
    half4 v = x[gid];
    y[gid] = v * v;
  }
}
kernel void binary_mul_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] * b[gid];
}
kernel void binary_mul_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] * b[gid];
}
kernel void binary_max_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = max(a[gid], b[gid]);
}
kernel void binary_max_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = max(a[gid], b[gid]);
}
kernel void binary_min_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = min(a[gid], b[gid]);
}
kernel void binary_min_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = min(a[gid], b[gid]);
}
kernel void ternary_select_vec4_f32(device const uchar4* p [[buffer(0)]],
                                    device const float4* a [[buffer(1)]],
                                    device const float4* b [[buffer(2)]], device float4* y [[buffer(3)]],
                                    constant uint& n4 [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < n4) {
    uchar4 m = p[gid];
    float4 av = a[gid];
    float4 bv = b[gid];
    y[gid] = float4(m[0] ? av[0] : bv[0], m[1] ? av[1] : bv[1], m[2] ? av[2] : bv[2],
                    m[3] ? av[3] : bv[3]);
  }
}
kernel void ternary_select_vec4_f16(device const uchar4* p [[buffer(0)]],
                                    device const half4* a [[buffer(1)]],
                                    device const half4* b [[buffer(2)]], device half4* y [[buffer(3)]],
                                    constant uint& n4 [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < n4) {
    uchar4 m = p[gid];
    half4 av = a[gid];
    half4 bv = b[gid];
    y[gid] = half4(m[0] ? av[0] : bv[0], m[1] ? av[1] : bv[1], m[2] ? av[2] : bv[2],
                   m[3] ? av[3] : bv[3]);
  }
}
kernel void binary_add_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] + b[gid];
}
kernel void binary_add_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] + b[gid];
}
kernel void binary_sub_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] - b[gid];
}
kernel void binary_sub_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] - b[gid];
}
kernel void binary_div_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] / b[gid];
}
kernel void binary_div_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = a[gid] / b[gid];
}
kernel void unary_neg_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = -x[i]; y[i+1] = -x[i+1]; }
}
kernel void unary_exp_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = exp(x[i]); y[i+1] = exp(x[i+1]); }
}
kernel void unary_log_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = log(x[i]); y[i+1] = log(x[i+1]); }
}
kernel void unary_tanh_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = tanh(x[i]); y[i+1] = tanh(x[i+1]); }
}
kernel void unary_sin_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = sin(x[i]); y[i+1] = sin(x[i+1]); }
}
kernel void unary_cos_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = cos(x[i]); y[i+1] = cos(x[i+1]); }
}
kernel void unary_sqrt_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = sqrt(x[i]); y[i+1] = sqrt(x[i+1]); }
}
kernel void unary_abs_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = abs(x[i]); y[i+1] = abs(x[i+1]); }
}
kernel void unary_rsqrt_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                 constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = rsqrt(x[i]); y[i+1] = rsqrt(x[i+1]); }
}
kernel void unary_ceil_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = ceil(x[i]); y[i+1] = ceil(x[i+1]); }
}
kernel void unary_floor_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                 constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = floor(x[i]); y[i+1] = floor(x[i+1]); }
}
kernel void unary_rint_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = rint(x[i]); y[i+1] = rint(x[i+1]); }
}
kernel void unary_sigmoid_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                   constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = 1.0f / (1.0f + exp(-x[i]));
    y[i+1] = 1.0f / (1.0f + exp(-x[i+1]));
  }
}
kernel void unary_square_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                  constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; float4 v0 = x[i]; float4 v1 = x[i+1]; y[i] = v0*v0; y[i+1] = v1*v1; }
}
kernel void unary_sign_vec8_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    float4 v0 = x[i]; float4 v1 = x[i+1];
    y[i] = float4(v0[0]>0?1:(v0[0]<0?-1:0), v0[1]>0?1:(v0[1]<0?-1:0),
                   v0[2]>0?1:(v0[2]<0?-1:0), v0[3]>0?1:(v0[3]<0?-1:0));
    y[i+1] = float4(v1[0]>0?1:(v1[0]<0?-1:0), v1[1]>0?1:(v1[1]<0?-1:0),
                     v1[2]>0?1:(v1[2]<0?-1:0), v1[3]>0?1:(v1[3]<0?-1:0));
  }
}
kernel void binary_add_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = a[i] + b[i]; y[i+1] = a[i+1] + b[i+1]; }
}
kernel void binary_sub_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = a[i] - b[i]; y[i+1] = a[i+1] - b[i+1]; }
}
kernel void binary_mul_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = a[i] * b[i]; y[i+1] = a[i+1] * b[i+1]; }
}
kernel void binary_div_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = a[i] / b[i]; y[i+1] = a[i+1] / b[i+1]; }
}
kernel void binary_max_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = max(a[i], b[i]); y[i+1] = max(a[i+1], b[i+1]); }
}
kernel void binary_min_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = min(a[i], b[i]); y[i+1] = min(a[i+1], b[i+1]); }
}
// ── vec8 fp16 kernels: each thread processes 8 halfs (two half4 loads) ──
// Doubles work-per-thread for fp16, matching MLX's "8 bytes per thread" strategy.
// Only fp16 benefits; fp32 vec4 already does 16 bytes/thread.
kernel void unary_square_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                  constant uint& n8 [[buffer(2)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    half4 v0 = x[i], v1 = x[i + 1];
    y[i] = v0 * v0;
    y[i + 1] = v1 * v1;
  }
}
kernel void unary_neg_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = -x[i]; y[i+1] = -x[i+1]; }
}
kernel void unary_exp_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = exp(x[i]); y[i+1] = exp(x[i+1]); }
}
kernel void unary_log_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = log(x[i]); y[i+1] = log(x[i+1]); }
}
kernel void unary_tanh_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = tanh(x[i]); y[i+1] = tanh(x[i+1]); }
}
kernel void unary_sin_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = sin(x[i]); y[i+1] = sin(x[i+1]); }
}
kernel void unary_cos_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = cos(x[i]); y[i+1] = cos(x[i+1]); }
}
kernel void unary_sqrt_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = sqrt(x[i]); y[i+1] = sqrt(x[i+1]); }
}
kernel void unary_abs_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                               constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = abs(x[i]); y[i+1] = abs(x[i+1]); }
}
kernel void unary_rsqrt_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                 constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = rsqrt(x[i]); y[i+1] = rsqrt(x[i+1]); }
}
kernel void unary_ceil_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = ceil(x[i]); y[i+1] = ceil(x[i+1]); }
}
kernel void unary_floor_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                 constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = floor(x[i]); y[i+1] = floor(x[i+1]); }
}
kernel void unary_rint_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = rint(x[i]); y[i+1] = rint(x[i+1]); }
}
kernel void unary_sigmoid_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                   constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = 1.0h / (1.0h + exp(-x[i]));
    y[i+1] = 1.0h / (1.0h + exp(-x[i+1]));
  }
}
kernel void unary_sign_vec8_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                constant uint& n8 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    half4 v0 = x[i]; half4 v1 = x[i+1];
    y[i] = half4(v0[0]>0?1:(v0[0]<0?-1:0), v0[1]>0?1:(v0[1]<0?-1:0),
                  v0[2]>0?1:(v0[2]<0?-1:0), v0[3]>0?1:(v0[3]<0?-1:0));
    y[i+1] = half4(v1[0]>0?1:(v1[0]<0?-1:0), v1[1]>0?1:(v1[1]<0?-1:0),
                    v1[2]>0?1:(v1[2]<0?-1:0), v1[3]>0?1:(v1[3]<0?-1:0));
  }
}
kernel void binary_add_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = a[i] + b[i];
    y[i + 1] = a[i + 1] + b[i + 1];
  }
}
kernel void binary_sub_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = a[i] - b[i];
    y[i + 1] = a[i + 1] - b[i + 1];
  }
}
kernel void binary_mul_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = a[i] * b[i];
    y[i + 1] = a[i + 1] * b[i + 1];
  }
}
kernel void binary_div_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = a[i] / b[i];
    y[i + 1] = a[i + 1] / b[i + 1];
  }
}
kernel void binary_max_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = max(a[i], b[i]);
    y[i + 1] = max(a[i + 1], b[i + 1]);
  }
}
kernel void binary_min_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = min(a[i], b[i]);
    y[i + 1] = min(a[i + 1], b[i + 1]);
  }
}
kernel void ternary_select_vec8_f16(device const uchar4* p [[buffer(0)]],
                                    device const half4* a [[buffer(1)]],
                                    device const half4* b [[buffer(2)]], device half4* y [[buffer(3)]],
                                    constant uint& n8 [[buffer(4)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    uchar4 m0 = p[i];
    half4 a0 = a[i], b0 = b[i];
    y[i] = half4(m0[0] ? a0[0] : b0[0], m0[1] ? a0[1] : b0[1], m0[2] ? a0[2] : b0[2], m0[3] ? a0[3] : b0[3]);
    uchar4 m1 = p[i + 1];
    half4 a1 = a[i + 1], b1 = b[i + 1];
    y[i + 1] = half4(m1[0] ? a1[0] : b1[0], m1[1] ? a1[1] : b1[1], m1[2] ? a1[2] : b1[2], m1[3] ? a1[3] : b1[3]);
  }
}
// ── Grid-stride loop kernels: fewer threads, each processing many float4 elements in a loop.
// This gives the GPU memory controller many more outstanding requests per thread,
// dramatically improving bandwidth utilization (from ~93 GB/s to ~140+ GB/s on M5).

// === Unary stride f32 ===
#define UNARY_STRIDE_F32(name, op) \
kernel void unary_##name##_stride_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]], \
                                       constant uint& n4 [[buffer(2)]], \
                                       uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) { \
    for (uint i = gid; i < n4; i += gs) { y[i] = op; } \
}
UNARY_STRIDE_F32(neg, -x[i])
UNARY_STRIDE_F32(exp, exp(x[i]))
UNARY_STRIDE_F32(log, log(x[i]))
UNARY_STRIDE_F32(tanh, tanh(x[i]))
UNARY_STRIDE_F32(sin, sin(x[i]))
UNARY_STRIDE_F32(cos, cos(x[i]))
UNARY_STRIDE_F32(sqrt, sqrt(x[i]))
UNARY_STRIDE_F32(abs, abs(x[i]))
UNARY_STRIDE_F32(rsqrt, rsqrt(x[i]))
UNARY_STRIDE_F32(ceil, ceil(x[i]))
UNARY_STRIDE_F32(floor, floor(x[i]))
UNARY_STRIDE_F32(rint, rint(x[i]))
kernel void unary_sigmoid_stride_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                      constant uint& n4 [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) { y[i] = 1.0f / (1.0f + exp(-x[i])); }
}
kernel void unary_square_stride_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                     constant uint& n4 [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) { float4 v = x[i]; y[i] = v * v; }
}
kernel void unary_sign_stride_f32(device const float4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                   constant uint& n4 [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) {
        float4 v = x[i];
        y[i] = float4(v[0]>0?1:(v[0]<0?-1:0), v[1]>0?1:(v[1]<0?-1:0),
                       v[2]>0?1:(v[2]<0?-1:0), v[3]>0?1:(v[3]<0?-1:0));
    }
}
#undef UNARY_STRIDE_F32

// === Binary stride f32 ===
#define BINARY_STRIDE_F32(name, op) \
kernel void binary_##name##_stride_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]], \
                                        device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]], \
                                        uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) { \
    for (uint i = gid; i < n4; i += gs) { y[i] = op; } \
}
BINARY_STRIDE_F32(add, a[i] + b[i])
BINARY_STRIDE_F32(sub, a[i] - b[i])
BINARY_STRIDE_F32(mul, a[i] * b[i])
BINARY_STRIDE_F32(div, a[i] / b[i])
BINARY_STRIDE_F32(max, max(a[i], b[i]))
BINARY_STRIDE_F32(min, min(a[i], b[i]))
#undef BINARY_STRIDE_F32

// === Ternary select stride f32 ===
kernel void ternary_select_stride_f32(device const uchar4* p [[buffer(0)]],
                                       device const float4* a [[buffer(1)]],
                                       device const float4* b [[buffer(2)]], device float4* y [[buffer(3)]],
                                       constant uint& n4 [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) {
        uchar4 m = p[i]; float4 av = a[i], bv = b[i];
        y[i] = float4(m[0]?av[0]:bv[0], m[1]?av[1]:bv[1], m[2]?av[2]:bv[2], m[3]?av[3]:bv[3]);
    }
}

// === Unary stride f16 ===
#define UNARY_STRIDE_F16(name, op) \
kernel void unary_##name##_stride_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]], \
                                       constant uint& n4 [[buffer(2)]], \
                                       uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) { \
    for (uint i = gid; i < n4; i += gs) { y[i] = op; } \
}
UNARY_STRIDE_F16(neg, -x[i])
UNARY_STRIDE_F16(exp, exp(x[i]))
UNARY_STRIDE_F16(log, log(x[i]))
UNARY_STRIDE_F16(tanh, tanh(x[i]))
UNARY_STRIDE_F16(sin, sin(x[i]))
UNARY_STRIDE_F16(cos, cos(x[i]))
UNARY_STRIDE_F16(sqrt, sqrt(x[i]))
UNARY_STRIDE_F16(abs, abs(x[i]))
UNARY_STRIDE_F16(rsqrt, rsqrt(x[i]))
UNARY_STRIDE_F16(ceil, ceil(x[i]))
UNARY_STRIDE_F16(floor, floor(x[i]))
UNARY_STRIDE_F16(rint, rint(x[i]))
kernel void unary_sigmoid_stride_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                      constant uint& n4 [[buffer(2)]],
                                      uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) { y[i] = half4(1.0h) / (half4(1.0h) + exp(-x[i])); }
}
kernel void unary_square_stride_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                     constant uint& n4 [[buffer(2)]],
                                     uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) { half4 v = x[i]; y[i] = v * v; }
}
kernel void unary_sign_stride_f16(device const half4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                   constant uint& n4 [[buffer(2)]],
                                   uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) {
        half4 v = x[i];
        y[i] = half4(v[0]>0?1:(v[0]<0?-1:0), v[1]>0?1:(v[1]<0?-1:0),
                      v[2]>0?1:(v[2]<0?-1:0), v[3]>0?1:(v[3]<0?-1:0));
    }
}
#undef UNARY_STRIDE_F16

// === Binary stride f16 ===
#define BINARY_STRIDE_F16(name, op) \
kernel void binary_##name##_stride_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]], \
                                        device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]], \
                                        uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) { \
    for (uint i = gid; i < n4; i += gs) { y[i] = op; } \
}
BINARY_STRIDE_F16(add, a[i] + b[i])
BINARY_STRIDE_F16(sub, a[i] - b[i])
BINARY_STRIDE_F16(mul, a[i] * b[i])
BINARY_STRIDE_F16(div, a[i] / b[i])
BINARY_STRIDE_F16(max, max(a[i], b[i]))
BINARY_STRIDE_F16(min, min(a[i], b[i]))
#undef BINARY_STRIDE_F16

// === Ternary select stride f16 ===
kernel void ternary_select_stride_f16(device const uchar4* p [[buffer(0)]],
                                       device const half4* a [[buffer(1)]],
                                       device const half4* b [[buffer(2)]], device half4* y [[buffer(3)]],
                                       constant uint& n4 [[buffer(4)]],
                                       uint gid [[thread_position_in_grid]], uint gs [[threads_per_grid]]) {
    for (uint i = gid; i < n4; i += gs) {
        uchar4 m = p[i]; half4 av = a[i], bv = b[i];
        y[i] = half4(m[0]?av[0]:bv[0], m[1]?av[1]:bv[1], m[2]?av[2]:bv[2], m[3]?av[3]:bv[3]);
    }
}

kernel void reduce_sum_last_f16_tg(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = 0.0f;
  // vec8 (two half4) reads for fp16 to maximize memory bandwidth.
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
    half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
    acc += (float)v0[0] + (float)v0[1] + (float)v0[2] + (float)v0[3]
         + (float)v1[0] + (float)v1[1] + (float)v1[2] + (float)v1[3];
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc += (float)x[base + i];
  float simd_acc = simd_sum(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : 0.0f;
    float total = simd_sum(block);
    if (lane == 0) y[row] = (half)total;
  }
}
kernel void reduce_sum_last_f32_tg(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = 0.0f;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
    float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
    acc += v0[0] + v0[1] + v0[2] + v0[3]
         + v1[0] + v1[1] + v1[2] + v1[3];
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc += x[base + i];
  float simd_acc = simd_sum(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : 0.0f;
    float total = simd_sum(block);
    if (lane == 0) y[row] = total;
  }
}
kernel void reduce_max_last_f16_tg(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = -INFINITY;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
    half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
    acc = max(acc, max(max(max((float)v0[0], (float)v0[1]), max((float)v0[2], (float)v0[3])),
                       max(max((float)v1[0], (float)v1[1]), max((float)v1[2], (float)v1[3]))));
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc = max(acc, (float)x[base + i]);
  float simd_acc = simd_max(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : -INFINITY;
    float total = simd_max(block);
    if (lane == 0) y[row] = (half)total;
  }
}
kernel void reduce_max_last_f32_tg(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = -INFINITY;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
    float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
    acc = max(acc, max(max(max(v0[0], v0[1]), max(v0[2], v0[3])),
                       max(max(v1[0], v1[1]), max(v1[2], v1[3]))));
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc = max(acc, x[base + i]);
  float simd_acc = simd_max(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : -INFINITY;
    float total = simd_max(block);
    if (lane == 0) y[row] = total;
  }
}
kernel void reduce_min_last_f16_tg(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = INFINITY;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
    half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
    acc = min(acc, min(min(min((float)v0[0], (float)v0[1]), min((float)v0[2], (float)v0[3])),
                       min(min((float)v1[0], (float)v1[1]), min((float)v1[2], (float)v1[3]))));
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc = min(acc, (float)x[base + i]);
  float simd_acc = simd_min(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : INFINITY;
    float total = simd_min(block);
    if (lane == 0) y[row] = (half)total;
  }
}
kernel void reduce_min_last_f32_tg(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                   constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                   uint tid [[thread_index_in_threadgroup]],
                                   uint row [[threadgroup_position_in_grid]],
                                   uint lane [[thread_index_in_simdgroup]],
                                   uint simd_group [[simdgroup_index_in_threadgroup]],
                                   uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = INFINITY;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
    float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
    acc = min(acc, min(min(min(v0[0], v0[1]), min(v0[2], v0[3])),
                       min(min(v1[0], v1[1]), min(v1[2], v1[3]))));
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc = min(acc, x[base + i]);
  float simd_acc = simd_min(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : INFINITY;
    float total = simd_min(block);
    if (lane == 0) y[row] = total;
  }
}
kernel void reduce_sum_last_f16_lin(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = 0.0f;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
      half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
      acc += (float)v0[0] + (float)v0[1] + (float)v0[2] + (float)v0[3]
           + (float)v1[0] + (float)v1[1] + (float)v1[2] + (float)v1[3];
    }
    for (uint i = vec8_cols; i < cols; ++i) acc += (float)x[base + i];
    y[gid] = (half)acc;
  }
}
kernel void reduce_sum_last_f32_lin(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = 0.0f;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
      float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
      acc += v0[0] + v0[1] + v0[2] + v0[3]
           + v1[0] + v1[1] + v1[2] + v1[3];
    }
    for (uint i = vec8_cols; i < cols; ++i) acc += x[base + i];
    y[gid] = acc;
  }
}
kernel void reduce_max_last_f16_lin(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = -INFINITY;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
      half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
      acc = max(acc, max(max(max((float)v0[0], (float)v0[1]), max((float)v0[2], (float)v0[3])),
                         max(max((float)v1[0], (float)v1[1]), max((float)v1[2], (float)v1[3]))));
    }
    for (uint i = vec8_cols; i < cols; ++i) acc = max(acc, (float)x[base + i]);
    y[gid] = (half)acc;
  }
}
kernel void reduce_max_last_f32_lin(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = -INFINITY;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
      float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
      acc = max(acc, max(max(max(v0[0], v0[1]), max(v0[2], v0[3])),
                         max(max(v1[0], v1[1]), max(v1[2], v1[3]))));
    }
    for (uint i = vec8_cols; i < cols; ++i) acc = max(acc, x[base + i]);
    y[gid] = acc;
  }
}
kernel void reduce_min_last_f16_lin(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = INFINITY;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
      half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
      acc = min(acc, min(min(min((float)v0[0], (float)v0[1]), min((float)v0[2], (float)v0[3])),
                         min(min((float)v1[0], (float)v1[1]), min((float)v1[2], (float)v1[3]))));
    }
    for (uint i = vec8_cols; i < cols; ++i) acc = min(acc, (float)x[base + i]);
    y[gid] = (half)acc;
  }
}
kernel void reduce_min_last_f32_lin(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = INFINITY;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
      float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
      acc = min(acc, min(min(min(v0[0], v0[1]), min(v0[2], v0[3])),
                         min(min(v1[0], v1[1]), min(v1[2], v1[3]))));
    }
    for (uint i = vec8_cols; i < cols; ++i) acc = min(acc, x[base + i]);
    y[gid] = acc;
  }
}
kernel void reduce_mean_last_f16_tg(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = 0.0f;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
    half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
    acc += (float)v0[0] + (float)v0[1] + (float)v0[2] + (float)v0[3]
         + (float)v1[0] + (float)v1[1] + (float)v1[2] + (float)v1[3];
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc += (float)x[base + i];
  float simd_acc = simd_sum(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : 0.0f;
    float total = simd_sum(block);
    if (lane == 0) y[row] = (half)(total / (float)cols);
  }
}
kernel void reduce_mean_last_f32_tg(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float acc = 0.0f;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
    float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
    acc += v0[0] + v0[1] + v0[2] + v0[3]
         + v1[0] + v1[1] + v1[2] + v1[3];
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) acc += x[base + i];
  float simd_acc = simd_sum(acc);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial[kMaxSimdGroups];
  if (lane == 0) partial[simd_group] = simd_acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float block = (lane < num_simd) ? partial[lane] : 0.0f;
    float total = simd_sum(block);
    if (lane == 0) y[row] = total / (float)cols;
  }
}
kernel void reduce_mean_last_f16_lin(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                     constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = 0.0f;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
      half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
      acc += (float)v0[0] + (float)v0[1] + (float)v0[2] + (float)v0[3]
           + (float)v1[0] + (float)v1[1] + (float)v1[2] + (float)v1[3];
    }
    for (uint i = vec8_cols; i < cols; ++i) acc += (float)x[base + i];
    y[gid] = (half)(acc / (float)cols);
  }
}
kernel void reduce_mean_last_f32_lin(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                     constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                     uint gid [[thread_position_in_grid]]) {
  if (gid < rows) {
    uint base = gid * cols;
    float acc = 0.0f;
    uint vec8_cols = cols & ~7u;
    for (uint i = 0; i < vec8_cols; i += 8) {
      float4 v0 = *reinterpret_cast<device const float4*>(x + base + i);
      float4 v1 = *reinterpret_cast<device const float4*>(x + base + i + 4);
      acc += v0[0] + v0[1] + v0[2] + v0[3]
           + v1[0] + v1[1] + v1[2] + v1[3];
    }
    for (uint i = vec8_cols; i < cols; ++i) acc += x[base + i];
    y[gid] = acc / (float)cols;
  }
}
kernel void softmax_pass1_f32_tg(device const float* x [[buffer(0)]],
                                 device float* maxsum [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) {
    float v = x[base + i];
    if (v > local_max) {
      local_sum = local_sum * exp(local_max - v) + 1.0f;
      local_max = v;
    } else {
      local_sum += exp(v - local_max);
    }
  }
  for (ushort offset = 16; offset >= 1; offset >>= 1) {
    float other_max = simd_shuffle_down(local_max, offset);
    float other_sum = simd_shuffle_down(local_sum, offset);
    float new_max = max(local_max, other_max);
    if (new_max == -INFINITY) {
      local_sum = 0.0f;
    } else {
      local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
    }
    local_max = new_max;
  }
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_max[kMaxSimdGroups];
  threadgroup float partial_sum[kMaxSimdGroups];
  if (lane == 0) {
    partial_max[simd_group] = local_max;
    partial_sum[simd_group] = local_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float m = (lane < num_simd) ? partial_max[lane] : -INFINITY;
    float s = (lane < num_simd) ? partial_sum[lane] : 0.0f;
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
      float om = simd_shuffle_down(m, offset);
      float os = simd_shuffle_down(s, offset);
      float nm = max(m, om);
      if (nm == -INFINITY) {
        s = 0.0f;
      } else {
        s = s * exp(m - nm) + os * exp(om - nm);
      }
      m = nm;
    }
    if (lane == 0) {
      maxsum[row * 2] = m;
      maxsum[row * 2 + 1] = s;
    }
  }
}
kernel void softmax_pass1_f16_tg(device const half* x [[buffer(0)]],
                                 device float* maxsum [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) {
    float v = (float)x[base + i];
    if (v > local_max) {
      local_sum = local_sum * exp(local_max - v) + 1.0f;
      local_max = v;
    } else {
      local_sum += exp(v - local_max);
    }
  }
  for (ushort offset = 16; offset >= 1; offset >>= 1) {
    float other_max = simd_shuffle_down(local_max, offset);
    float other_sum = simd_shuffle_down(local_sum, offset);
    float new_max = max(local_max, other_max);
    if (new_max == -INFINITY) {
      local_sum = 0.0f;
    } else {
      local_sum = local_sum * exp(local_max - new_max) + other_sum * exp(other_max - new_max);
    }
    local_max = new_max;
  }
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_max[kMaxSimdGroups];
  threadgroup float partial_sum[kMaxSimdGroups];
  if (lane == 0) {
    partial_max[simd_group] = local_max;
    partial_sum[simd_group] = local_sum;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float m = (lane < num_simd) ? partial_max[lane] : -INFINITY;
    float s = (lane < num_simd) ? partial_sum[lane] : 0.0f;
    for (ushort offset = 16; offset >= 1; offset >>= 1) {
      float om = simd_shuffle_down(m, offset);
      float os = simd_shuffle_down(s, offset);
      float nm = max(m, om);
      if (nm == -INFINITY) {
        s = 0.0f;
      } else {
        s = s * exp(m - nm) + os * exp(om - nm);
      }
      m = nm;
    }
    if (lane == 0) {
      maxsum[row * 2] = m;
      maxsum[row * 2 + 1] = s;
    }
  }
}
// Pass 2: normalize.  Each thread computes one element: exp(x - max) / sumexp.
kernel void softmax_pass2_f32(device const float* x [[buffer(0)]],
                              device const float* maxsum [[buffer(1)]],
                              device float* y [[buffer(2)]],
                              constant uint& cols [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  uint row = gid / cols;
  uint col = gid % cols;
  float m = maxsum[row * 2];
  float s = maxsum[row * 2 + 1];
  y[gid] = exp(x[gid] - m) / s;
}
kernel void softmax_pass2_f16(device const half* x [[buffer(0)]],
                              device const float* maxsum [[buffer(1)]],
                              device half* y [[buffer(2)]],
                              constant uint& cols [[buffer(3)]],
                              uint gid [[thread_position_in_grid]]) {
  uint row = gid / cols;
  uint col = gid % cols;
  float m = maxsum[row * 2];
  float s = maxsum[row * 2 + 1];
  y[gid] = (half)(exp((float)x[gid] - m) / s);
}

// --- LayerNorm 2-pass fused kernels ---
// Pass 1: compute mean and variance per row using threadgroup reduction.
// Outputs [rows × 2] floats: [mean, inv_std] pairs.
kernel void layernorm_pass1_f32_tg(device const float* x [[buffer(0)]],
                                    device float* stats [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]],
                                    constant uint& cols [[buffer(3)]],
                                    constant float& eps [[buffer(4)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float sum_val = 0.0f;
  float sum_sq = 0.0f;
  uint vec4_cols = cols & ~3u;
  for (uint i = tid * 4; i < vec4_cols; i += tg_size * 4) {
    float4 v = *reinterpret_cast<device const float4*>(x + base + i);
    sum_val += v[0] + v[1] + v[2] + v[3];
    sum_sq += v[0]*v[0] + v[1]*v[1] + v[2]*v[2] + v[3]*v[3];
  }
  for (uint i = vec4_cols + tid; i < cols; i += tg_size) {
    float v = x[base + i];
    sum_val += v;
    sum_sq += v * v;
  }
  float s_val = simd_sum(sum_val);
  float s_sq  = simd_sum(sum_sq);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_sum[kMaxSimdGroups];
  threadgroup float partial_sq[kMaxSimdGroups];
  if (lane == 0) {
    partial_sum[simd_group] = s_val;
    partial_sq[simd_group]  = s_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float ps = (lane < num_simd) ? partial_sum[lane] : 0.0f;
    float pq = (lane < num_simd) ? partial_sq[lane]  : 0.0f;
    ps = simd_sum(ps);
    pq = simd_sum(pq);
    if (lane == 0) {
      float mean = ps / (float)cols;
      float var  = pq / (float)cols - mean * mean;
      float inv_std = rsqrt(var + eps);
      stats[row * 2]     = mean;
      stats[row * 2 + 1] = inv_std;
    }
  }
}
kernel void layernorm_pass1_f16_tg(device const half* x [[buffer(0)]],
                                    device float* stats [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]],
                                    constant uint& cols [[buffer(3)]],
                                    constant float& eps [[buffer(4)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  float sum_val = 0.0f;
  float sum_sq = 0.0f;
  uint vec8_cols = cols & ~7u;
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) {
    half4 v0 = *reinterpret_cast<device const half4*>(x + base + i);
    half4 v1 = *reinterpret_cast<device const half4*>(x + base + i + 4);
    float4 f0 = float4(v0);
    float4 f1 = float4(v1);
    sum_val += f0[0] + f0[1] + f0[2] + f0[3] + f1[0] + f1[1] + f1[2] + f1[3];
    sum_sq  += f0[0]*f0[0] + f0[1]*f0[1] + f0[2]*f0[2] + f0[3]*f0[3]
             + f1[0]*f1[0] + f1[1]*f1[1] + f1[2]*f1[2] + f1[3]*f1[3];
  }
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) {
    float v = (float)x[base + i];
    sum_val += v;
    sum_sq += v * v;
  }
  float s_val = simd_sum(sum_val);
  float s_sq  = simd_sum(sum_sq);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_sum[kMaxSimdGroups];
  threadgroup float partial_sq[kMaxSimdGroups];
  if (lane == 0) {
    partial_sum[simd_group] = s_val;
    partial_sq[simd_group]  = s_sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float ps = (lane < num_simd) ? partial_sum[lane] : 0.0f;
    float pq = (lane < num_simd) ? partial_sq[lane]  : 0.0f;
    ps = simd_sum(ps);
    pq = simd_sum(pq);
    if (lane == 0) {
      float mean = ps / (float)cols;
      float var  = pq / (float)cols - mean * mean;
      float inv_std = rsqrt(var + eps);
      stats[row * 2]     = mean;
      stats[row * 2 + 1] = inv_std;
    }
  }
}
// Pass 2: normalize each element: y = (x - mean) * inv_std
kernel void layernorm_pass2_f32(device const float* x [[buffer(0)]],
                                device const float* stats [[buffer(1)]],
                                device float* y [[buffer(2)]],
                                constant uint& cols [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  uint row = gid / cols;
  float mean    = stats[row * 2];
  float inv_std = stats[row * 2 + 1];
  y[gid] = (x[gid] - mean) * inv_std;
}
kernel void layernorm_pass2_f16(device const half* x [[buffer(0)]],
                                device const float* stats [[buffer(1)]],
                                device half* y [[buffer(2)]],
                                constant uint& cols [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  uint row = gid / cols;
  float mean    = stats[row * 2];
  float inv_std = stats[row * 2 + 1];
  y[gid] = (half)((float)x[gid] - mean) * (half)inv_std;
}
// --- Native gather (embedding lookup) kernel ---
// Gathers rows from an operand table using int32 indices.
// operand: [table_rows, cols], indices: [num_indices], output: [num_indices, cols]
kernel void gather_rows_f32(device const float* operand [[buffer(0)]],
                            device const int* indices [[buffer(1)]],
                            device float* output [[buffer(2)]],
                            constant uint& num_indices [[buffer(3)]],
                            constant uint& cols [[buffer(4)]],
                            constant uint& table_rows [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  uint total = num_indices * cols;
  if (gid >= total) return;
  uint idx_r = gid / cols;
  uint c = gid - idx_r * cols;
  int row = indices[idx_r];
  if (row >= 0 && (uint)row < table_rows) {
    output[gid] = operand[(uint)row * cols + c];
  } else {
    output[gid] = 0.0f;
  }
}
kernel void gather_rows_f16(device const half* operand [[buffer(0)]],
                            device const int* indices [[buffer(1)]],
                            device half* output [[buffer(2)]],
                            constant uint& num_indices [[buffer(3)]],
                            constant uint& cols [[buffer(4)]],
                            constant uint& table_rows [[buffer(5)]],
                            uint gid [[thread_position_in_grid]]) {
  uint total = num_indices * cols;
  if (gid >= total) return;
  uint idx_r = gid / cols;
  uint c = gid - idx_r * cols;
  int row = indices[idx_r];
  if (row >= 0 && (uint)row < table_rows) {
    output[gid] = operand[(uint)row * cols + c];
  } else {
    output[gid] = (half)0.0f;
  }
}
// --- Scatter variants (mul, max, min) ---
kernel void scatter_mul_rows_f32(device float* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const float* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = out[dst] * updates[gid];
  }
}
kernel void scatter_mul_rows_f16(device half* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const half* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = out[dst] * updates[gid];
  }
}
kernel void scatter_max_rows_f32(device float* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const float* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = max(out[dst], updates[gid]);
  }
}
kernel void scatter_max_rows_f16(device half* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const half* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = max(out[dst], updates[gid]);
  }
}
kernel void scatter_min_rows_f32(device float* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const float* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = min(out[dst], updates[gid]);
  }
}
kernel void scatter_min_rows_f16(device half* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const half* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    uint dst = (uint)row * cols + c;
    out[dst] = min(out[dst], updates[gid]);
  }
}

kernel void iota_i32(device int* y [[buffer(0)]], constant uint& n [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = (int)gid;
}
kernel void iota_f32(device float* y [[buffer(0)]], constant uint& n [[buffer(1)]],
                     uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = (float)gid;
}
// --- Fill kernels (branch-free, used by pad fill+copy path) ---
kernel void fill_f16_vec4(device half4* y [[buffer(0)]], constant float& val [[buffer(1)]],
                          constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = half4((half)val);
}
kernel void fill_f32_vec4(device float4* y [[buffer(0)]], constant float& val [[buffer(1)]],
                          constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = float4(val);
}
// --- Strided 2D copy (branch-free, used by pad after fill) ---
kernel void copy2d_f16(device const half* src [[buffer(0)]], device half* dst [[buffer(1)]],
                       constant uint& src_cols [[buffer(2)]], constant uint& dst_cols [[buffer(3)]],
                       constant uint& dst_row_off [[buffer(4)]], constant uint& dst_col_off [[buffer(5)]],
                       constant uint& src_total [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= src_total) return;
  uint r = gid / src_cols;
  uint c = gid - r * src_cols;
  dst[(r + dst_row_off) * dst_cols + c + dst_col_off] = src[gid];
}
kernel void copy2d_f32(device const float* src [[buffer(0)]], device float* dst [[buffer(1)]],
                       constant uint& src_cols [[buffer(2)]], constant uint& dst_cols [[buffer(3)]],
                       constant uint& dst_row_off [[buffer(4)]], constant uint& dst_col_off [[buffer(5)]],
                       constant uint& src_total [[buffer(6)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= src_total) return;
  uint r = gid / src_cols;
  uint c = gid - r * src_cols;
  dst[(r + dst_row_off) * dst_cols + c + dst_col_off] = src[gid];
}
// --- Legacy pad kernels (kept for fallback if fill+copy path disabled) ---
kernel void pad2d_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                      constant uint& in_h [[buffer(2)]], constant uint& in_w [[buffer(3)]],
                      constant uint& out_w [[buffer(4)]], constant uint& low_h [[buffer(5)]],
                      constant uint& low_w [[buffer(6)]], constant float& pad_val [[buffer(7)]],
                      constant uint& n [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  uint oh = gid / out_w;
  uint ow = gid - oh * out_w;
  if (oh >= low_h && oh < low_h + in_h && ow >= low_w && ow < low_w + in_w) {
    uint ih = oh - low_h;
    uint iw = ow - low_w;
    y[gid] = x[ih * in_w + iw];
  } else {
    y[gid] = (half)pad_val;
  }
}
kernel void pad2d_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                      constant uint& in_h [[buffer(2)]], constant uint& in_w [[buffer(3)]],
                      constant uint& out_w [[buffer(4)]], constant uint& low_h [[buffer(5)]],
                      constant uint& low_w [[buffer(6)]], constant float& pad_val [[buffer(7)]],
                      constant uint& n [[buffer(8)]], uint gid [[thread_position_in_grid]]) {
  if (gid >= n) return;
  uint oh = gid / out_w;
  uint ow = gid - oh * out_w;
  if (oh >= low_h && oh < low_h + in_h && ow >= low_w && ow < low_w + in_w) {
    uint ih = oh - low_h;
    uint iw = ow - low_w;
    y[gid] = x[ih * in_w + iw];
  } else {
    y[gid] = pad_val;
  }
}
kernel void scatter_add_rows_f32(device float* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const float* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    // Non-atomic: duplicate-index rows may lose updates (rare in practice).
    uint dst = (uint)row * cols + c;
    out[dst] = out[dst] + updates[gid];
  }
}
kernel void scatter_add_rows_f16(device half* out [[buffer(0)]],
                                 device const int* indices [[buffer(1)]],
                                 device const half* updates [[buffer(2)]],
                                 constant uint& k [[buffer(3)]],
                                 constant uint& cols [[buffer(4)]],
                                 constant uint& out_rows [[buffer(5)]],
                                 constant uint& total [[buffer(6)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= total) return;
  uint r = gid / cols;
  if (r >= k) return;
  uint c = gid - r * cols;
  int row = indices[r];
  if (row >= 0 && (uint)row < out_rows) {
    // Metal has no atomic_half.  Use non-atomic read-add-write.
    // Duplicate-index rows may lose updates (same as MPSGraph path).
    uint dst = (uint)row * cols + c;
    out[dst] = out[dst] + updates[gid];
  }
}
kernel void convert_f16_to_f32(device const half* x [[buffer(0)]], device float* y [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = (float)x[gid];
}
kernel void convert_f32_to_f16(device const float* x [[buffer(0)]], device half* y [[buffer(1)]],
                               constant uint& n [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = (half)x[gid];
}
kernel void convert_f16_to_f32_vec4(device const half4* x [[buffer(0)]], device float4* y [[buffer(1)]],
                                    constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = float4(x[gid]);
}
kernel void convert_f32_to_f16_vec4(device const float4* x [[buffer(0)]], device half4* y [[buffer(1)]],
                                    constant uint& n4 [[buffer(2)]], uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = half4(x[gid]);
}

)METAL";
    // --- GEMM kernel code generator (MLX Steel-inspired) ---
    // Key optimizations vs naive:
    //   1. Vectorized loads (float4/half8) — 16 bytes per thread per load
    //   2. Threadgroup memory padding (+4 float / +8 half per row) — eliminates bank conflicts
    //   3. Block swizzling — remaps threadgroup IDs for L2 cache locality
    //   4. Serpentine MMA iteration — reverses N on odd M rows for register reuse
    //   5. Fixed thread layout (no div/mod per element) — compile-time address mapping

    struct GemmConfig {
        int bm, bn, bk;        // tile sizes
        int wm, wn;            // simdgroup grid (wm x wn simdgroups)
        int acc_rows, acc_cols; // 8x8 accumulators per simdgroup
    };
    static const GemmConfig configs[] = {
        // Variant 0: BM=32, BN=32, BK=16, 2x2 SGs (128 thr), 2x2 acc/SG
        {32, 32, 16, 2, 2, 2, 2},
        // Variant 1: BM=64, BN=64, BK=16, 2x2 SGs (128 thr), 4x4 acc/SG (MLX default)
        {64, 64, 16, 2, 2, 4, 4},
        // Variant 2: BM=64, BN=64, BK=32, 2x2 SGs (128 thr), 4x4 acc/SG
        {64, 64, 32, 2, 2, 4, 4},
        // Variant 3: BM=64, BN=32, BK=32, 2x2 SGs (128 thr), 4x2 acc/SG
        {64, 32, 32, 2, 2, 4, 2},
    };

    auto generateGemmKernel = [](const GemmConfig& cfg, const char* dtype,
                                  const char* sg_type, const char* zero_val) -> std::string {
        char name[128];
        snprintf(name, sizeof(name), "gemm_nn_%s_bm%d_bn%d_bk%d",
                 dtype, cfg.bm, cfg.bn, cfg.bk);
        int num_sg = cfg.wm * cfg.wn;
        int num_threads = num_sg * 32;
        bool is_f16 = (strcmp(dtype, "f16") == 0);
        const char* elem_type = is_f16 ? "half" : "float";
        int elem_bytes = is_f16 ? 2 : 4;
        int tg_pad = 16 / elem_bytes;  // 4 for float, 8 for half

        // Padded leading dimensions for threadgroup memory
        int a_ld = cfg.bk + tg_pad;  // A is BM x BK, stored row-major
        int b_ld = cfg.bn + tg_pad;  // B is BK x BN, stored row-major
        int a_tg_size = cfg.bm * a_ld;
        int b_tg_size = cfg.bk * b_ld;

        // Vectorized load parameters for A tile (BM x BK)
        // Each thread loads vec_a elements contiguously along K dimension
        int a_total = cfg.bm * cfg.bk;
        int a_vec = a_total / num_threads;  // elements per thread
        // Clamp vec to valid Metal vector sizes (1,2,4)
        if (a_vec > 4) a_vec = 4;
        int a_tcols = cfg.bk / a_vec;       // threads across K
        int a_trows = num_threads / a_tcols; // threads across M
        int a_nrows = (cfg.bm + a_trows - 1) / a_trows; // iterations per thread

        // Vectorized load parameters for B tile (BK x BN)
        int b_total = cfg.bk * cfg.bn;
        int b_vec = b_total / num_threads;
        if (b_vec > 4) b_vec = 4;
        int b_tcols = cfg.bn / b_vec;
        int b_trows = num_threads / b_tcols;
        int b_nrows = (cfg.bk + b_trows - 1) / b_trows;

        // Vector type names
        char a_vec_type[32], b_vec_type[32];
        if (a_vec == 1) snprintf(a_vec_type, sizeof(a_vec_type), "%s", elem_type);
        else snprintf(a_vec_type, sizeof(a_vec_type), "%s%d", elem_type, a_vec);
        if (b_vec == 1) snprintf(b_vec_type, sizeof(b_vec_type), "%s", elem_type);
        else snprintf(b_vec_type, sizeof(b_vec_type), "%s%d", elem_type, b_vec);

        std::string s;
        s.reserve(8192);
        char buf[512];

        // Kernel signature
        snprintf(buf, sizeof(buf),
            "kernel void %s(device const %s* A [[buffer(0)]],\n"
            "    device const %s* B [[buffer(1)]],\n"
            "    device %s* C [[buffer(2)]],\n"
            "    constant uint& M [[buffer(3)]],\n"
            "    constant uint& N [[buffer(4)]],\n"
            "    constant uint& K [[buffer(5)]],\n"
            "    uint3 tgid [[threadgroup_position_in_grid]],\n"
            "    uint sgid  [[simdgroup_index_in_threadgroup]],\n"
            "    uint tid   [[thread_index_in_threadgroup]]) {\n",
            name, elem_type, elem_type, elem_type);
        s += buf;

        // Threadgroup tile position (no swizzle — tested, doesn't help on M5)
        snprintf(buf, sizeof(buf),
            "  uint bm = tgid.x * %du;\n"
            "  uint bn = tgid.y * %du;\n",
            cfg.bm, cfg.bn);
        s += buf;

        // Simdgroup layout (WM x WN grid)
        snprintf(buf, sizeof(buf),
            "  uint sg_row = sgid / %du;\n"
            "  uint sg_col = sgid %% %du;\n",
            cfg.wn, cfg.wn);
        s += buf;

        // Accumulators
        snprintf(buf, sizeof(buf),
            "  %s acc[%d][%d];\n"
            "  for (uint i = 0; i < %du; i++)\n"
            "    for (uint j = 0; j < %du; j++)\n"
            "      acc[i][j] = %s;\n",
            sg_type, cfg.acc_rows, cfg.acc_cols,
            cfg.acc_rows, cfg.acc_cols, zero_val);
        s += buf;

        // Threadgroup memory with padding
        snprintf(buf, sizeof(buf),
            "  threadgroup %s As[%d];\n"
            "  threadgroup %s Bs[%d];\n",
            elem_type, a_tg_size,
            elem_type, b_tg_size);
        s += buf;

        // Precompute loader positions (fixed per thread, no div/mod in loop)
        snprintf(buf, sizeof(buf),
            "  uint a_row = tid / %du;\n"
            "  uint a_col = (tid %% %du) * %du;\n"
            "  uint b_row = tid / %du;\n"
            "  uint b_col = (tid %% %du) * %du;\n",
            a_tcols, a_tcols, a_vec,
            b_tcols, b_tcols, b_vec);
        s += buf;

        // Device memory pointers for A and B (will be advanced each K iteration)
        s += "  device const ";
        s += elem_type;
        s += "* Aptr = A + bm * K;\n";
        s += "  device const ";
        s += elem_type;
        s += "* Bptr = B + bn;\n";

        // K-loop
        snprintf(buf, sizeof(buf),
            "  for (uint kb = 0; kb < K; kb += %du) {\n", cfg.bk);
        s += buf;

        // Load A tile: BM x BK -> As[BM x (BK+pad)]
        // Each thread loads a_vec elements per row, iterating over a_nrows rows
        s += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        if (a_nrows == 1) {
            if (a_vec == 1) {
                snprintf(buf, sizeof(buf),
                    "    As[a_row * %d + a_col] = Aptr[a_row * K + kb + a_col];\n",
                    a_ld);
            } else {
                snprintf(buf, sizeof(buf),
                    "    *(threadgroup %s*)(As + a_row * %d + a_col) = *(device const %s*)(Aptr + a_row * K + kb + a_col);\n",
                    a_vec_type, a_ld, a_vec_type);
            }
            s += buf;
        } else {
            snprintf(buf, sizeof(buf),
                "    for (uint ar = 0; ar < %du; ar++) {\n"
                "      uint row = a_row + ar * %du;\n",
                a_nrows, a_trows);
            s += buf;
            if (a_vec == 1) {
                snprintf(buf, sizeof(buf),
                    "      As[row * %d + a_col] = Aptr[row * K + kb + a_col];\n",
                    a_ld);
            } else {
                snprintf(buf, sizeof(buf),
                    "      *(threadgroup %s*)(As + row * %d + a_col) = *(device const %s*)(Aptr + row * K + kb + a_col);\n",
                    a_vec_type, a_ld, a_vec_type);
            }
            s += buf;
            s += "    }\n";
        }

        // Load B tile: BK x BN -> Bs[BK x (BN+pad)]
        if (b_nrows == 1) {
            if (b_vec == 1) {
                snprintf(buf, sizeof(buf),
                    "    Bs[b_row * %d + b_col] = Bptr[(kb + b_row) * N + b_col];\n",
                    b_ld);
            } else {
                snprintf(buf, sizeof(buf),
                    "    *(threadgroup %s*)(Bs + b_row * %d + b_col) = *(device const %s*)(Bptr + (kb + b_row) * N + b_col);\n",
                    b_vec_type, b_ld, b_vec_type);
            }
            s += buf;
        } else {
            snprintf(buf, sizeof(buf),
                "    for (uint br = 0; br < %du; br++) {\n"
                "      uint row = b_row + br * %du;\n",
                b_nrows, b_trows);
            s += buf;
            if (b_vec == 1) {
                snprintf(buf, sizeof(buf),
                    "      Bs[row * %d + b_col] = Bptr[(kb + row) * N + b_col];\n",
                    b_ld);
            } else {
                snprintf(buf, sizeof(buf),
                    "      *(threadgroup %s*)(Bs + row * %d + b_col) = *(device const %s*)(Bptr + (kb + row) * N + b_col);\n",
                    b_vec_type, b_ld, b_vec_type);
            }
            s += buf;
            s += "    }\n";
        }

        s += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";

        // Per-simdgroup base in threadgroup memory
        snprintf(buf, sizeof(buf),
            "    uint base_m = sg_row * %du;\n"
            "    uint base_n = sg_col * %du;\n",
            cfg.acc_rows * 8, cfg.acc_cols * 8);
        s += buf;

        // Inner k-loop over 8x8 tiles with serpentine MMA
        snprintf(buf, sizeof(buf),
            "    for (uint kk = 0; kk < %du; kk += 8) {\n", cfg.bk);
        s += buf;

        // Load A sub-tiles from threadgroup (padded stride)
        for (int ai = 0; ai < cfg.acc_rows; ai++) {
            snprintf(buf, sizeof(buf),
                "      %s a%d;\n"
                "      simdgroup_load(a%d, As + (base_m + %du) * %d + kk, %d);\n",
                sg_type, ai, ai, ai * 8, a_ld, a_ld);
            s += buf;
        }

        // Load B sub-tiles from threadgroup (padded stride)
        for (int bj = 0; bj < cfg.acc_cols; bj++) {
            snprintf(buf, sizeof(buf),
                "      %s b%d;\n"
                "      simdgroup_load(b%d, Bs + kk * %d + base_n + %du, %d);\n",
                sg_type, bj, bj, b_ld, bj * 8, b_ld);
            s += buf;
        }

        // MMA calls with serpentine iteration
        for (int ai = 0; ai < cfg.acc_rows; ai++) {
            for (int bj_raw = 0; bj_raw < cfg.acc_cols; bj_raw++) {
                // Serpentine: reverse N direction on odd M rows
                int bj = (ai % 2 == 0) ? bj_raw : (cfg.acc_cols - 1 - bj_raw);
                snprintf(buf, sizeof(buf),
                    "      simdgroup_multiply_accumulate(acc[%d][%d], a%d, b%d, acc[%d][%d]);\n",
                    ai, bj, ai, bj, ai, bj);
                s += buf;
            }
        }

        s += "    }\n";  // end kk loop
        s += "  }\n";  // end kb loop

        // Store results to device memory
        for (int ai = 0; ai < cfg.acc_rows; ai++) {
            for (int bj = 0; bj < cfg.acc_cols; bj++) {
                snprintf(buf, sizeof(buf),
                    "  simdgroup_store(acc[%d][%d], C + (bm + sg_row * %du + %du) * N + bn + sg_col * %du + %du, N);\n",
                    ai, bj, cfg.acc_rows * 8, ai * 8, cfg.acc_cols * 8, bj * 8);
                s += buf;
            }
        }

        s += "}\n\n";
        return s;
    };

    // Build combined source: static non-GEMM kernels + generated GEMM variants
    static NSString* source = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
        std::string combined(kSource);
        // Generate all GEMM variants for both f32 and f16
        for (const auto& cfg : configs) {
            combined += generateGemmKernel(cfg, "f32",
                "simdgroup_float8x8",
                "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)");
            combined += generateGemmKernel(cfg, "f16",
                "simdgroup_half8x8",
                "make_filled_simdgroup_matrix<half, 8, 8>((half)0.0h)");
        }
        source = [[NSString alloc] initWithUTF8String:combined.c_str()];
    });
    return source;
}

NSString* MppGemmKernelSource() {
    // MPP GEMM kernel using MetalPerformancePrimitives matmul2d.
    // Takes raw device pointers + dimensions and constructs inline tensors
    // using tensor(ptr, extents<>()) constructor syntax.
    // 64x64 tile with 4 simdgroups (128 threads), dynamic K.
    static const char* kMppSource = R"METAL(
#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void gemm_mpp_f32(device float* A_ptr [[buffer(0)]],
                          device float* B_ptr [[buffer(1)]],
                          device float* C_ptr [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tgid [[threadgroup_position_in_grid]]) {
    // Construct inline 2D tensors from raw device pointers.
    // Row-major data: A[M×K], B[K×N], C[M×N].
    // Metal tensor convention: extents are (dim0=inner/col, dim1=outer/row)
    // for row-major, dim0 = num_cols (stride=1), dim1 = num_rows (stride=cols)
    // Indexing: tensor[col, row] for row-major data.
    auto A = tensor(A_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(K, M));
    auto B = tensor(B_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(N, K));
    auto C_t = tensor(C_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(N, M));

    constexpr auto desc = matmul2d_descriptor(64, 64,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    // slice(dim0_offset, dim1_offset) with col-major convention:
    // dim0=col offset, dim1=row offset
    // For C[M×N]: row tile = tgid.y*64, col tile = tgid.x*64
    auto mA = A.slice(0, tgid.y * 64);            // A: shift rows by tgid.y*64
    auto mB = B.slice(tgid.x * 64, 0);            // B: shift cols by tgid.x*64
    auto mC = C_t.slice(tgid.x * 64, tgid.y * 64); // C: shift cols+rows
    matmulOp.run(mA, mB, mC);
}

kernel void gemm_mpp_f16(device half* A_ptr [[buffer(0)]],
                          device half* B_ptr [[buffer(1)]],
                          device half* C_ptr [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint2 tgid [[threadgroup_position_in_grid]]) {
    auto A = tensor(A_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(K, M));
    auto B = tensor(B_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(N, K));
    auto C_t = tensor(C_ptr, extents<int32_t, dynamic_extent, dynamic_extent>(N, M));

    constexpr auto desc = matmul2d_descriptor(64, 64,
        static_cast<int>(dynamic_extent), false, false, true);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 64, 0);
    auto mC = C_t.slice(tgid.x * 64, tgid.y * 64);
    matmulOp.run(mA, mB, mC);
}
)METAL";
    return [NSString stringWithUTF8String:kMppSource];
}

} // namespace jax_metallib
