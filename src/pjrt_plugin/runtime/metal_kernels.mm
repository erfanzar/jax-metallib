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

// StableHLO maximum/minimum propagate NaN; Metal max/min return the non-NaN
// operand. NaN is tested on the bit pattern (the library is built with default
// fast math, where isnan() may be folded) and the NaN operand is returned.
inline float nb_max(float a, float b) {
  float m = max(a, b);
  m = (as_type<uint>(b) & 0x7fffffffu) > 0x7f800000u ? b : m;
  return (as_type<uint>(a) & 0x7fffffffu) > 0x7f800000u ? a : m;
}
inline float nb_min(float a, float b) {
  float m = min(a, b);
  m = (as_type<uint>(b) & 0x7fffffffu) > 0x7f800000u ? b : m;
  return (as_type<uint>(a) & 0x7fffffffu) > 0x7f800000u ? a : m;
}
inline half nb_max(half a, half b) {
  half m = max(a, b);
  m = (as_type<ushort>(b) & 0x7fffu) > 0x7c00u ? b : m;
  return (as_type<ushort>(a) & 0x7fffu) > 0x7c00u ? a : m;
}
inline half nb_min(half a, half b) {
  half m = min(a, b);
  m = (as_type<ushort>(b) & 0x7fffu) > 0x7c00u ? b : m;
  return (as_type<ushort>(a) & 0x7fffu) > 0x7c00u ? a : m;
}
inline float4 nb_max(float4 a, float4 b) {
  float4 m = select(max(a, b), b, (as_type<uint4>(b) & uint4(0x7fffffffu)) > uint4(0x7f800000u));
  return select(m, a, (as_type<uint4>(a) & uint4(0x7fffffffu)) > uint4(0x7f800000u));
}
inline float4 nb_min(float4 a, float4 b) {
  float4 m = select(min(a, b), b, (as_type<uint4>(b) & uint4(0x7fffffffu)) > uint4(0x7f800000u));
  return select(m, a, (as_type<uint4>(a) & uint4(0x7fffffffu)) > uint4(0x7f800000u));
}
inline half4 nb_max(half4 a, half4 b) {
  half4 m = select(max(a, b), b, (as_type<ushort4>(b) & ushort4(0x7fffu)) > ushort4(0x7c00u));
  return select(m, a, (as_type<ushort4>(a) & ushort4(0x7fffu)) > ushort4(0x7c00u));
}
inline half4 nb_min(half4 a, half4 b) {
  half4 m = select(min(a, b), b, (as_type<ushort4>(b) & ushort4(0x7fffu)) > ushort4(0x7c00u));
  return select(m, a, (as_type<ushort4>(a) & ushort4(0x7fffu)) > ushort4(0x7c00u));
}

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
  if (gid < n) y[gid] = nb_max(a[gid], b[gid]);
}
kernel void binary_max_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = nb_max(a[gid], b[gid]);
}
kernel void binary_min_f16(device const half* a [[buffer(0)]], device const half* b [[buffer(1)]],
                           device half* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = nb_min(a[gid], b[gid]);
}
kernel void binary_min_f32(device const float* a [[buffer(0)]], device const float* b [[buffer(1)]],
                           device float* y [[buffer(2)]], constant uint& n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
  if (gid < n) y[gid] = nb_min(a[gid], b[gid]);
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
  if (gid < n4) y[gid] = nb_max(a[gid], b[gid]);
}
kernel void binary_max_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = nb_max(a[gid], b[gid]);
}
kernel void binary_min_vec4_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = nb_min(a[gid], b[gid]);
}
kernel void binary_min_vec4_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n4 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n4) y[gid] = nb_min(a[gid], b[gid]);
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
  if (gid < n8) { uint i = gid * 2; y[i] = nb_max(a[i], b[i]); y[i+1] = nb_max(a[i+1], b[i+1]); }
}
kernel void binary_min_vec8_f32(device const float4* a [[buffer(0)]], device const float4* b [[buffer(1)]],
                                device float4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) { uint i = gid * 2; y[i] = nb_min(a[i], b[i]); y[i+1] = nb_min(a[i+1], b[i+1]); }
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
    y[i] = nb_max(a[i], b[i]);
    y[i + 1] = nb_max(a[i + 1], b[i + 1]);
  }
}
kernel void binary_min_vec8_f16(device const half4* a [[buffer(0)]], device const half4* b [[buffer(1)]],
                                device half4* y [[buffer(2)]], constant uint& n8 [[buffer(3)]],
                                uint gid [[thread_position_in_grid]]) {
  if (gid < n8) {
    uint i = gid * 2;
    y[i] = nb_min(a[i], b[i]);
    y[i + 1] = nb_min(a[i + 1], b[i + 1]);
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
BINARY_STRIDE_F32(max, nb_max(a[i], b[i]))
BINARY_STRIDE_F32(min, nb_min(a[i], b[i]))
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
BINARY_STRIDE_F16(max, nb_max(a[i], b[i]))
BINARY_STRIDE_F16(min, nb_min(a[i], b[i]))
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
// Max/min reductions must propagate NaN (StableHLO maximum/minimum do), but
// Metal max/min and simd_max/simd_min return the non-NaN operand, and this
// library is compiled with the default (fast) math mode, where isnan() may be
// folded away. The max/min kernels below therefore carry an integer NaN flag
// built from the bit pattern and store a quiet NaN when any element was NaN
// (before this, jnp.max/min over >= 4096 elements silently dropped NaN).
inline uint rd_nan4(float4 v) { return any((as_type<uint4>(v) & uint4(0x7fffffffu)) > uint4(0x7f800000u)) ? 1u : 0u; }
inline uint rd_nan4(half4 v) { return any((as_type<ushort4>(v) & ushort4(0x7fffu)) > ushort4(0x7c00u)) ? 1u : 0u; }
inline uint rd_nan(float v) { return (as_type<uint>(v) & 0x7fffffffu) > 0x7f800000u ? 1u : 0u; }
inline uint rd_nan(half v) { return (uint(as_type<ushort>(v)) & 0x7fffu) > 0x7c00u ? 1u : 0u; }
inline float rd_qnan_like(float) { return as_type<float>(0x7fc00000u); }
inline half rd_qnan_like(half) { return as_type<half>((ushort)0x7e00u); }

// Order keys for exact max/min. f32 subnormals flush to a zero of the same sign
// (XLA:CPU does this), and -0 orders below +0. Comparing the keys as signed
// integers gives the XLA order; NaN is tracked separately by the callers.
// Half inputs widen to normal floats, so their subnormals are kept.
inline int rd_key(float f) {
  int b = as_type<int>(f);
  if ((b & 0x7f800000) == 0) b &= int(0x80000000u);
  return b < 0 ? (b ^ 0x7fffffff) : b;
}
inline int4 rd_key4(float4 f) {
  int4 b = as_type<int4>(f);
  b = select(b, b & int4(int(0x80000000u)), (b & int4(0x7f800000)) == int4(0));
  return select(b, b ^ int4(0x7fffffff), b < int4(0));
}
inline float rd_unkey(int k) { return as_type<float>(k < 0 ? (k ^ 0x7fffffff) : k); }

#define REDUCE_MINMAX_LAST_TG(NAME, T, V4, OP, INIT) \
kernel void NAME(device const T* x [[buffer(0)]], device T* y [[buffer(1)]], \
                 constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], \
                 uint tid [[thread_index_in_threadgroup]], \
                 uint row [[threadgroup_position_in_grid]], \
                 uint lane [[thread_index_in_simdgroup]], \
                 uint simd_group [[simdgroup_index_in_threadgroup]], \
                 uint tg_size [[threads_per_threadgroup]]) { \
  if (row >= rows) return; \
  uint base = row * cols; \
  int acc = rd_key(INIT); \
  uint nan_seen = 0u; \
  uint vec8_cols = cols & ~7u; \
  for (uint i = tid * 8; i < vec8_cols; i += tg_size * 8) { \
    V4 v0 = *reinterpret_cast<device const V4*>(x + base + i); \
    V4 v1 = *reinterpret_cast<device const V4*>(x + base + i + 4); \
    nan_seen |= rd_nan4(v0) | rd_nan4(v1); \
    int4 k = OP(rd_key4(float4(v0)), rd_key4(float4(v1))); \
    acc = OP(acc, OP(OP(k[0], k[1]), OP(k[2], k[3]))); \
  } \
  for (uint i = vec8_cols + tid; i < cols; i += tg_size) { \
    T v = x[base + i]; \
    nan_seen |= rd_nan(v); \
    acc = OP(acc, rd_key(float(v))); \
  } \
  int simd_acc = simd_##OP(acc); \
  uint simd_nan = simd_max(nan_seen); \
  constexpr uint kMaxSimdGroups = 32; \
  threadgroup int partial[kMaxSimdGroups]; \
  threadgroup uint partial_nan[kMaxSimdGroups]; \
  if (lane == 0) { \
    partial[simd_group] = simd_acc; \
    partial_nan[simd_group] = simd_nan; \
  } \
  threadgroup_barrier(mem_flags::mem_threadgroup); \
  if (simd_group == 0) { \
    uint num_simd = (tg_size + 31) >> 5; \
    int block = (lane < num_simd) ? partial[lane] : rd_key(INIT); \
    uint block_nan = (lane < num_simd) ? partial_nan[lane] : 0u; \
    int total = simd_##OP(block); \
    uint any_nan = simd_max(block_nan); \
    if (lane == 0) y[row] = any_nan ? rd_qnan_like(T(0)) : T(rd_unkey(total)); \
  } \
}
REDUCE_MINMAX_LAST_TG(reduce_max_last_f16_tg, half, half4, max, -INFINITY)
REDUCE_MINMAX_LAST_TG(reduce_max_last_f32_tg, float, float4, max, -INFINITY)
REDUCE_MINMAX_LAST_TG(reduce_min_last_f16_tg, half, half4, min, INFINITY)
REDUCE_MINMAX_LAST_TG(reduce_min_last_f32_tg, float, float4, min, INFINITY)
#undef REDUCE_MINMAX_LAST_TG
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
// NaN-propagating per-thread row max/min (see rd_nan above).
#define REDUCE_MINMAX_LAST_LIN(NAME, T, V4, OP, INIT) \
kernel void NAME(device const T* x [[buffer(0)]], device T* y [[buffer(1)]], \
                 constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], \
                 uint gid [[thread_position_in_grid]]) { \
  if (gid < rows) { \
    uint base = gid * cols; \
    int acc = rd_key(INIT); \
    uint nan_seen = 0u; \
    uint vec8_cols = cols & ~7u; \
    for (uint i = 0; i < vec8_cols; i += 8) { \
      V4 v0 = *reinterpret_cast<device const V4*>(x + base + i); \
      V4 v1 = *reinterpret_cast<device const V4*>(x + base + i + 4); \
      nan_seen |= rd_nan4(v0) | rd_nan4(v1); \
      int4 k = OP(rd_key4(float4(v0)), rd_key4(float4(v1))); \
      acc = OP(acc, OP(OP(k[0], k[1]), OP(k[2], k[3]))); \
    } \
    for (uint i = vec8_cols; i < cols; ++i) { \
      T v = x[base + i]; \
      nan_seen |= rd_nan(v); \
      acc = OP(acc, rd_key(float(v))); \
    } \
    y[gid] = nan_seen ? rd_qnan_like(T(0)) : T(rd_unkey(acc)); \
  } \
}
REDUCE_MINMAX_LAST_LIN(reduce_max_last_f16_lin, half, half4, max, -INFINITY)
REDUCE_MINMAX_LAST_LIN(reduce_max_last_f32_lin, float, float4, max, -INFINITY)
REDUCE_MINMAX_LAST_LIN(reduce_min_last_f16_lin, half, half4, min, INFINITY)
REDUCE_MINMAX_LAST_LIN(reduce_min_last_f32_lin, float, float4, min, INFINITY)
#undef REDUCE_MINMAX_LAST_LIN
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
// --- Axis-0 (column-wise) reduce kernels ---
// For input [rows, cols], reduce across rows for each column → output [cols].
// Each thread handles one column, looping over all rows (strided reads).
kernel void reduce_sum_axis0_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= cols) return;
  float acc = 0.0f;
  for (uint r = 0; r < rows; r++) acc += x[r * cols + gid];
  y[gid] = acc;
}
kernel void reduce_sum_axis0_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= cols) return;
  float acc = 0.0f;
  for (uint r = 0; r < rows; r++) acc += (float)x[r * cols + gid];
  y[gid] = (half)acc;
}
kernel void reduce_mean_axis0_f32(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                                  constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= cols) return;
  float acc = 0.0f;
  for (uint r = 0; r < rows; r++) acc += x[r * cols + gid];
  y[gid] = acc / (float)rows;
}
kernel void reduce_mean_axis0_f16(device const half* x [[buffer(0)]], device half* y [[buffer(1)]],
                                  constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]],
                                  uint gid [[thread_position_in_grid]]) {
  if (gid >= cols) return;
  float acc = 0.0f;
  for (uint r = 0; r < rows; r++) acc += (float)x[r * cols + gid];
  y[gid] = (half)(acc / (float)rows);
}
// NaN-propagating column max/min (see rd_nan above).
#define REDUCE_MINMAX_AXIS0(NAME, T, OP, INIT) \
kernel void NAME(device const T* x [[buffer(0)]], device T* y [[buffer(1)]], \
                 constant uint& rows [[buffer(2)]], constant uint& cols [[buffer(3)]], \
                 uint gid [[thread_position_in_grid]]) { \
  if (gid >= cols) return; \
  float acc = INIT; \
  uint nan_seen = 0u; \
  for (uint r = 0; r < rows; r++) { \
    T v = x[r * cols + gid]; \
    nan_seen |= rd_nan(v); \
    acc = OP(acc, float(v)); \
  } \
  y[gid] = nan_seen ? rd_qnan_like(T(0)) : T(acc); \
}
REDUCE_MINMAX_AXIS0(reduce_max_axis0_f32, float, max, -INFINITY)
REDUCE_MINMAX_AXIS0(reduce_max_axis0_f16, half, max, -INFINITY)
REDUCE_MINMAX_AXIS0(reduce_min_axis0_f32, float, min, INFINITY)
REDUCE_MINMAX_AXIS0(reduce_min_axis0_f16, half, min, INFINITY)
#undef REDUCE_MINMAX_AXIS0
// XLA float sort order: every NaN sorts after +inf, and -0 == +0. A plain `>`
// never moves NaN, and +INF padding would sort ahead of real NaNs.
inline bool xla_sort_gt(float a, float b) {
  bool an = isnan(a), bn = isnan(b);
  if (an || bn) return an && !bn;
  return a > b;
}
// Order-preserving key for XLA's float sort comparator: -0, +0 and every
// subnormal share one key (XLA:CPU compares with denormals-as-zero, so they
// tie and keep input order), every NaN gets one key above +inf, and
// 0xFFFFFFFF is reserved for padding. No finite value or inf maps to
// 0xFFFFFFFE or 0xFFFFFFFF.
inline uint xla_sort_key(float v) {
  uint b = as_type<uint>(v);
  uint m = b & 0x7fffffffu;
  if (m > 0x7f800000u) return 0xfffffffeu;
  if (m < 0x00800000u) return 0x80000000u;
  return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}
// Stable bitonic sort over (key, original index) pairs, keys in the first
// padded_n uints of threadgroup memory and ushort indices right after them
// (padded_n * 6 bytes, so padded_n <= 4096). Ties resolve by index, which is
// XLA's stable order, and the store gathers raw input bits by index, so NaN
// payloads and the sign of zero are preserved.
inline void bitonic_stable_network(threadgroup uint* keys, threadgroup ushort* idx, uint padded_n, uint tid,
                                   uint tg_size) {
  uint half_n = padded_n >> 1;
  for (uint size = 2; size <= padded_n; size <<= 1) {
    for (uint stride = size >> 1; stride >= 1; stride >>= 1) {
      uint stride_mask = stride - 1;
      for (uint i = tid; i < half_n; i += tg_size) {
        uint left = ((i & ~stride_mask) << 1) | (i & stride_mask);
        uint right = left + stride;
        bool ascending = (left & size) == 0;
        uint kl = keys[left], kr = keys[right];
        ushort il = idx[left], ir = idx[right];
        bool gt = (kl > kr) || (kl == kr && il > ir);
        if (ascending == gt) {
          keys[left] = kr;
          keys[right] = kl;
          idx[left] = ir;
          idx[right] = il;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
}
// --- Bitonic sort kernel (single threadgroup, in-place) ---
// Sorts up to padded_n elements in threadgroup memory using bitonic merge.
// Input: n actual elements, padded to next power-of-2 with NaN (largest).
// Output: sorted n elements written back to device memory.
kernel void bitonic_sort_asc_f32(device const float* input [[buffer(0)]],
                                 device float* output [[buffer(1)]],
                                 constant uint& n [[buffer(2)]],
                                 constant uint& padded_n [[buffer(3)]],
                                 threadgroup float* shared [[threadgroup(0)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
  threadgroup uint* keys = (threadgroup uint*)shared;
  threadgroup ushort* idx = (threadgroup ushort*)(keys + padded_n);
  for (uint i = tid; i < padded_n; i += tg_size) {
    keys[i] = (i < n) ? xla_sort_key(input[i]) : 0xffffffffu;
    idx[i] = (ushort)i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_stable_network(keys, idx, padded_n, tid, tg_size);
  for (uint i = tid; i < n; i += tg_size) {
    output[i] = input[idx[i]];
  }
}
// --- Bitonic sort kernel for 2D row-wise sort ---
// Each threadgroup sorts one row. row index = threadgroup_position_in_grid.x
// row_stride = number of columns (elements per row in memory)
// Uses bitwise operations instead of integer division (stride/size are power-of-2).
kernel void bitonic_sort_rows_asc_f32(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& n [[buffer(2)]],
                                       constant uint& padded_n [[buffer(3)]],
                                       constant uint& row_stride [[buffer(4)]],
                                       threadgroup float* shared [[threadgroup(0)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint tg_size [[threads_per_threadgroup]],
                                       uint row [[threadgroup_position_in_grid]]) {
  uint base = row * row_stride;
  threadgroup uint* keys = (threadgroup uint*)shared;
  threadgroup ushort* idx = (threadgroup ushort*)(keys + padded_n);
  for (uint i = tid; i < padded_n; i += tg_size) {
    keys[i] = (i < n) ? xla_sort_key(input[base + i]) : 0xffffffffu;
    idx[i] = (ushort)i;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  bitonic_stable_network(keys, idx, padded_n, tid, tg_size);
  for (uint i = tid; i < n; i += tg_size) {
    output[base + i] = input[base + idx[i]];
  }
}

// --- Radix sort for 2D row-wise sort (n ≤ 1024, ascending, f32) ---
// Uses 4-bit radix (16 bins), 8 passes over 32-bit float keys.
// Fewer total operations than bitonic sort for n >= 64.
// tg_size should be >= n (1 element per thread for scatter).
kernel void radix_sort_rows_asc_f32(device const float* input [[buffer(0)]],
                                     device float* output [[buffer(1)]],
                                     constant uint& n [[buffer(2)]],
                                     constant uint& padded_n_unused [[buffer(3)]],
                                     constant uint& row_stride [[buffer(4)]],
                                     threadgroup uint* scratch [[threadgroup(0)]],
                                     uint tid [[thread_index_in_threadgroup]],
                                     uint tg_size [[threads_per_threadgroup]],
                                     uint row [[threadgroup_position_in_grid]]) {
  // scratch layout: [0..padded-1] = keys, [padded..2*padded-1] = alt_keys, [2*padded..+15] = histogram
  uint padded = tg_size; // tg_size is set to next power of 2 >= n
  threadgroup uint* keys_a = scratch;
  threadgroup uint* keys_b = scratch + padded;
  threadgroup uint* hist = scratch + 2 * padded;

  uint base = row * row_stride;

  // Load and transform float bits for radix sort ordering.
  // IEEE 754 trick: flip sign bit for positives, flip all bits for negatives.
  // This gives correct ascending sort order on the uint representation.
  uint bits = 0xFFFFFFFF; // +INF representation after transform (padding)
  if (tid < n) {
    uint raw = as_type<uint>(input[base + tid]);
    uint mask = -int(raw >> 31) | 0x80000000u;
    bits = raw ^ mask;
  }
  keys_a[tid] = bits;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  threadgroup uint* src = keys_a;
  threadgroup uint* dst = keys_b;

  // 8 passes of 4-bit radix (processes all 32 bits)
  for (uint pass = 0; pass < 8; pass++) {
    uint shift = pass * 4;
    uint digit = (src[tid] >> shift) & 0xF;

    // Clear histogram
    if (tid < 16) hist[tid] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Count digits using atomic add
    // For small n, use sequential counting by thread 0 for simplicity
    // For larger n, each thread atomically increments its digit count
    threadgroup atomic_uint* ahist = (threadgroup atomic_uint*)hist;
    if (tid < n) {
      atomic_fetch_add_explicit(&ahist[digit], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Exclusive prefix sum over histogram (thread 0 does it, only 16 bins)
    if (tid == 0) {
      uint total = 0;
      for (uint i = 0; i < 16; i++) {
        uint count = hist[i];
        hist[i] = total;
        total += count;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Scatter: each thread finds its output position.
    // To get a stable sort, threads with the same digit need deterministic ordering.
    // Use a simple approach: each thread with digit d atomically claims a position.
    if (tid < n) {
      uint pos = atomic_fetch_add_explicit(&ahist[digit], 1u, memory_order_relaxed);
      dst[pos] = src[tid];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Swap src/dst
    threadgroup uint* tmp = src;
    src = dst;
    dst = tmp;
  }

  // Write back (transform bits back to float)
  if (tid < n) {
    uint sorted_bits = src[tid];
    // Reverse the IEEE 754 transform
    uint mask = ((sorted_bits >> 31) - 1) | 0x80000000u;
    sorted_bits ^= mask;
    output[base + tid] = as_type<float>(sorted_bits);
  }
}

// --- Register-based bitonic sort for 2D row-wise sort (n ≤ 256, tg=32) ---
// Each thread holds up to 8 elements in registers. Uses simd_shuffle_xor for
// cross-thread comparisons (strides < 32) and local register swaps for
// intra-thread comparisons (strides ≥ 32). No threadgroup memory needed.
// This eliminates all shared memory latency from the sort hot path.
kernel void bitonic_sort_rows_reg_f32(device const float* input [[buffer(0)]],
                                       device float* output [[buffer(1)]],
                                       constant uint& n [[buffer(2)]],
                                       constant uint& padded_n [[buffer(3)]],
                                       constant uint& row_stride [[buffer(4)]],
                                       uint tid [[thread_index_in_threadgroup]],
                                       uint row [[threadgroup_position_in_grid]]) {
  // Thread t owns elements at positions: t, t+32, t+64, ..., t+224
  // reg[k] = element at global position (row * row_stride + tid + k*32)
  uint base = row * row_stride;
  uint elems = padded_n >> 5; // elements per thread = padded_n / 32
  float reg[8]; // max 8 elements per thread (256 / 32)
  // Load with padding
  for (uint k = 0; k < elems; ++k) {
    uint pos = tid + k * 32;
    reg[k] = (pos < n) ? input[base + pos] : NAN;
  }

  // Bitonic sort network — all passes done in registers + simd_shuffle
  for (uint size = 2; size <= padded_n; size <<= 1) {
    for (uint stride = size >> 1; stride >= 1; stride >>= 1) {
      if (stride >= 32) {
        // Intra-thread: compare elements within this thread's register array.
        // stride/32 = distance between register indices.
        uint reg_stride = stride >> 5;
        for (uint k = 0; k < elems; ++k) {
          uint global_pos = tid + k * 32;
          // Only process "left" elements (where the stride-th bit is 0)
          if ((global_pos & stride) == 0) {
            uint partner_k = k + reg_stride;
            if (partner_k < elems) {
              bool ascending = (global_pos & size) == 0;
              float lv = reg[k], rv = reg[partner_k];
              if (ascending ? xla_sort_gt(lv, rv) : xla_sort_gt(rv, lv)) {
                reg[k] = rv;
                reg[partner_k] = lv;
              }
            }
          }
        }
      } else {
        // Cross-thread: use simd_shuffle_xor to exchange with partner lane.
        // For stride s < 32, the partner thread is tid ^ s.
        for (uint k = 0; k < elems; ++k) {
          uint global_pos = tid + k * 32;
          float my_val = reg[k];
          float partner_val = simd_shuffle_xor(my_val, (ushort)stride);
          // Determine if this thread holds the "left" or "right" of the pair
          bool is_left = (tid & stride) == 0;
          bool ascending = (global_pos & size) == 0;
          // For left: keep min if ascending, keep max if descending
          // For right: keep max if ascending, keep min if descending
          bool should_swap = is_left ? (ascending ? (my_val > partner_val) : (my_val < partner_val))
                                     : (ascending ? (my_val < partner_val) : (my_val > partner_val));
          if (should_swap) {
            reg[k] = partner_val;
          }
        }
      }
    }
  }

  // Write back sorted results (only first n elements)
  for (uint k = 0; k < elems; ++k) {
    uint pos = tid + k * 32;
    if (pos < n) {
      output[base + pos] = reg[k];
    }
  }
}

// --- Tiled 2D transpose kernel ---
// Transposes the last two dims of a batched tensor: [..., rows, cols] → [..., cols, rows]
// Uses 32×32 threadgroup tiles with +1 padding to avoid bank conflicts.
// Grid: (ceil(cols/32), ceil(rows/32), batch), TG: (32, 8, 1) = 256 threads
kernel void transpose2d_f32(device const float* input [[buffer(0)]],
                            device float* output [[buffer(1)]],
                            constant uint& rows [[buffer(2)]],
                            constant uint& cols [[buffer(3)]],
                            threadgroup float* tile [[threadgroup(0)]],
                            uint3 tgid [[threadgroup_position_in_grid]],
                            uint3 tid3 [[thread_position_in_threadgroup]]) {
  constexpr uint TILE = 32;
  constexpr uint TILE_PAD = 33; // +1 to avoid bank conflicts
  uint batch_offset_in = tgid.z * rows * cols;
  uint batch_offset_out = tgid.z * cols * rows;
  uint tile_col = tgid.x * TILE; // column start in input
  uint tile_row = tgid.y * TILE; // row start in input
  // Load tile: each of 256 threads (32×8) loads 4 rows
  for (uint i = 0; i < TILE; i += 8) {
    uint r = tile_row + tid3.y + i;
    uint c = tile_col + tid3.x;
    if (r < rows && c < cols) {
      tile[(tid3.y + i) * TILE_PAD + tid3.x] = input[batch_offset_in + r * cols + c];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Write transposed: output row = input col, output col = input row
  uint out_tile_col = tgid.y * TILE; // swapped: input row → output col
  uint out_tile_row = tgid.x * TILE; // swapped: input col → output row
  for (uint i = 0; i < TILE; i += 8) {
    uint r = out_tile_row + tid3.y + i;
    uint c = out_tile_col + tid3.x;
    if (r < cols && c < rows) {
      output[batch_offset_out + r * rows + c] = tile[tid3.x * TILE_PAD + tid3.y + i];
    }
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
      // -inf contributes exp(-inf) = 0; without this guard a thread whose
      // first element is -inf computed exp(-inf - -inf) = NaN for the row.
      if (v != -INFINITY) local_sum += exp(v - local_max);
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
      // -inf contributes exp(-inf) = 0; without this guard a thread whose
      // first element is -inf computed exp(-inf - -inf) = NaN for the row.
      if (v != -INFINITY) local_sum += exp(v - local_max);
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

// --- Single-pass full softmax: max + sum_exp + normalize in one kernel ---
// Reads each row once into threadgroup memory, computes max and sum_exp,
// then normalizes and writes output. Memory traffic = 1R + 1W
// instead of 2R + 1W for the 2-pass approach.
// Requires threadgroup memory >= cols * sizeof(float) (max 8K cols).
kernel void softmax_full_f32_tg(device const float* x [[buffer(0)]],
                                 device float* y [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 threadgroup float* shared [[threadgroup(0)]],
                                 uint tid [[thread_index_in_threadgroup]],
                                 uint row [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]],
                                 uint simd_group [[simdgroup_index_in_threadgroup]],
                                 uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;

  // Phase 1: Load row into threadgroup memory AND find local max + sum_exp.
  float local_max = -INFINITY;
  float local_sum = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) {
    float v = x[base + i];
    shared[i] = v;
    if (v > local_max) {
      local_sum = local_sum * exp(local_max - v) + 1.0f;
      local_max = v;
    } else {
      // -inf contributes exp(-inf) = 0; without this guard a thread whose
      // first element is -inf computed exp(-inf - -inf) = NaN for the row.
      if (v != -INFINITY) local_sum += exp(v - local_max);
    }
  }

  // Reduce max + sum across SIMD groups (online softmax reduction).
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

  float row_max, row_sum;
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
      partial_max[0] = m;
      partial_sum[0] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_max = partial_max[0];
  row_sum = partial_sum[0];

  // Phase 2: Read from threadgroup memory (free!) and write normalized output.
  float inv_sum = 1.0f / row_sum;
  for (uint i = tid; i < cols; i += tg_size) {
    y[base + i] = exp(shared[i] - row_max) * inv_sum;
  }
}

kernel void softmax_full_f16_tg(device const half* x [[buffer(0)]],
                                 device half* y [[buffer(1)]],
                                 constant uint& rows [[buffer(2)]],
                                 constant uint& cols [[buffer(3)]],
                                 threadgroup float* shared [[threadgroup(0)]],
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
    shared[i] = v;
    if (v > local_max) {
      local_sum = local_sum * exp(local_max - v) + 1.0f;
      local_max = v;
    } else {
      // -inf contributes exp(-inf) = 0; without this guard a thread whose
      // first element is -inf computed exp(-inf - -inf) = NaN for the row.
      if (v != -INFINITY) local_sum += exp(v - local_max);
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

  float row_max, row_sum;
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
      partial_max[0] = m;
      partial_sum[0] = s;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_max = partial_max[0];
  row_sum = partial_sum[0];

  float inv_sum = 1.0f / row_sum;
  for (uint i = tid; i < cols; i += tg_size) {
    y[base + i] = (half)(exp(shared[i] - row_max) * inv_sum);
  }
}

// --- f16 softmax with XLA:CPU rounding points ---
// JAX lowers f16 softmax as max -> sub (f16) -> exp (f16) -> convert f32 ->
// sum -> convert f16 -> divide, and XLA:CPU evaluates the divide as a multiply
// by an f16 reciprocal. Bit-exact emulation: d = half(x - m), e = half(exp(d)),
// s = half(sum_f32(e)), r = half(1 / s), y = half(e * r). This library compiles
// with fast math, so exp/divide use precise:: and NaN/inf are tested by bits.
// A row whose max is NaN, +inf or -inf (all -inf) is all NaN, as in XLA.
// Requires cols <= 8192 (threadgroup memory holds the row as float).
kernel void softmax_full_f16_xla_tg(device const half* x [[buffer(0)]],
                                    device half* y [[buffer(1)]],
                                    constant uint& rows [[buffer(2)]],
                                    constant uint& cols [[buffer(3)]],
                                    threadgroup float* shared [[threadgroup(0)]],
                                    uint tid [[thread_index_in_threadgroup]],
                                    uint row [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint simd_group [[simdgroup_index_in_threadgroup]],
                                    uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float pmax[kMaxSimdGroups];
  threadgroup uint pnan[kMaxSimdGroups];
  threadgroup float psum[kMaxSimdGroups];
  uint num_simd = (tg_size + 31) >> 5;

  float m = -INFINITY;
  uint nan_seen = 0;
  for (uint i = tid; i < cols; i += tg_size) {
    half h = x[base + i];
    ushort b = as_type<ushort>(h);
    if ((b & 0x7fffu) > 0x7c00u) nan_seen = 1u;
    float v = float(h);
    shared[i] = v;
    m = max(m, v);
  }
  float sm = simd_max(m);
  uint sn = simd_max(nan_seen);
  if (lane == 0) {
    pmax[simd_group] = sm;
    pnan[simd_group] = sn;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float rm = -INFINITY;
  uint rn = 0u;
  for (uint g = 0; g < num_simd; ++g) {
    rm = max(rm, pmax[g]);
    rn |= pnan[g];
  }
  // Uniform across the threadgroup: every thread takes the same branch.
  if (rn != 0u || (as_type<uint>(rm) & 0x7fffffffu) == 0x7f800000u) {
    for (uint i = tid; i < cols; i += tg_size) {
      y[base + i] = as_type<half>((ushort)0x7e00u);
    }
    return;
  }

  float lsum = 0.0f;
  for (uint i = tid; i < cols; i += tg_size) {
    float v = shared[i];
    float e = 0.0f;
    if (as_type<uint>(v) != 0xff800000u) {  // -inf contributes exp(-inf) = 0
      half d = half(v - rm);
      e = float(half(precise::exp(float(d))));
    }
    shared[i] = e;
    lsum += e;
  }
  float ss = simd_sum(lsum);
  if (lane == 0) psum[simd_group] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float s = 0.0f;
  for (uint g = 0; g < num_simd; ++g) s += psum[g];
  float r = float(half(precise::divide(1.0f, float(half(s)))));
  for (uint i = tid; i < cols; i += tg_size) {
    y[base + i] = half(shared[i] * r);
  }
}

// --- Fused softmax numerator: exp(x - max(x, axis=-1, keepdims=True)) ---
// Single kernel: reads each row once into threadgroup memory, finds max,
// then computes exp(x - max) and writes output. Memory traffic = 1R + 1W
// instead of 2R + 1W for separate reduce_max + exp(x-max) passes.
// Requires threadgroup memory >= cols * sizeof(float).
kernel void softmax_numerator_f32_tg(device const float* x [[buffer(0)]],
                                      device float* y [[buffer(1)]],
                                      constant uint& rows [[buffer(2)]],
                                      constant uint& cols [[buffer(3)]],
                                      threadgroup float* shared [[threadgroup(0)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;

  // Phase 1: Load row into threadgroup memory AND find partial max.
  // max/simd_max drop NaN, so a bit-pattern NaN flag is carried alongside:
  // XLA's x - max(x) is NaN for the whole row when the row holds a NaN.
  float local_max = -INFINITY;
  uint nan_seen = 0u;
  for (uint i = tid; i < cols; i += tg_size) {
    float v = x[base + i];
    shared[i] = v;
    nan_seen |= rd_nan(v);
    local_max = max(local_max, v);
  }

  // Reduce max across SIMD group
  float simd_max_val = simd_max(local_max);
  uint simd_nan = simd_max(nan_seen);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_max[kMaxSimdGroups];
  threadgroup uint partial_nan[kMaxSimdGroups];
  if (lane == 0) {
    partial_max[simd_group] = simd_max_val;
    partial_nan[simd_group] = simd_nan;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Final reduce (first SIMD group)
  float row_max;
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float m = (lane < num_simd) ? partial_max[lane] : -INFINITY;
    uint nn = (lane < num_simd) ? partial_nan[lane] : 0u;
    m = simd_max(m);
    nn = simd_max(nn);
    if (lane == 0) {
      partial_max[0] = m; // broadcast via shared memory
      partial_nan[0] = nn;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_max = partial_max[0];
  // Uniform across the threadgroup: every thread takes the same branch.
  if (partial_nan[0] != 0u) {
    for (uint i = tid; i < cols; i += tg_size) y[base + i] = rd_qnan_like(0.0f);
    return;
  }

  // Phase 2: Read from threadgroup memory (free!) and write exp(x - max).
  for (uint i = tid; i < cols; i += tg_size) {
    y[base + i] = exp(shared[i] - row_max);
  }
}

kernel void softmax_numerator_f16_tg(device const half* x [[buffer(0)]],
                                      device half* y [[buffer(1)]],
                                      constant uint& rows [[buffer(2)]],
                                      constant uint& cols [[buffer(3)]],
                                      threadgroup float* shared [[threadgroup(0)]],
                                      uint tid [[thread_index_in_threadgroup]],
                                      uint row [[threadgroup_position_in_grid]],
                                      uint lane [[thread_index_in_simdgroup]],
                                      uint simd_group [[simdgroup_index_in_threadgroup]],
                                      uint tg_size [[threads_per_threadgroup]]) {
  if (row >= rows) return;
  uint base = row * cols;

  // NaN flag as in softmax_numerator_f32_tg.
  float local_max = -INFINITY;
  uint nan_seen = 0u;
  for (uint i = tid; i < cols; i += tg_size) {
    half h = x[base + i];
    nan_seen |= rd_nan(h);
    float v = (float)h;
    shared[i] = v;
    local_max = max(local_max, v);
  }

  float simd_max_val = simd_max(local_max);
  uint simd_nan = simd_max(nan_seen);
  constexpr uint kMaxSimdGroups = 32;
  threadgroup float partial_max[kMaxSimdGroups];
  threadgroup uint partial_nan[kMaxSimdGroups];
  if (lane == 0) {
    partial_max[simd_group] = simd_max_val;
    partial_nan[simd_group] = simd_nan;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float row_max;
  if (simd_group == 0) {
    uint num_simd = (tg_size + 31) >> 5;
    float m = (lane < num_simd) ? partial_max[lane] : -INFINITY;
    uint nn = (lane < num_simd) ? partial_nan[lane] : 0u;
    m = simd_max(m);
    nn = simd_max(nn);
    if (lane == 0) {
      partial_max[0] = m;
      partial_nan[0] = nn;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  row_max = partial_max[0];
  if (partial_nan[0] != 0u) {
    for (uint i = tid; i < cols; i += tg_size) y[base + i] = rd_qnan_like(half(0));
    return;
  }

  for (uint i = tid; i < cols; i += tg_size) {
    y[base + i] = (half)exp(shared[i] - row_max);
  }
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

// --- Coalesced gather: float4 reads, shared indices, 2D dispatch ---
// Each threadgroup handles ROWS_PER_TG rows. Thread.y iterates over float4 columns.
// Eliminates integer division and redundant index reads.
kernel void gather_rows_coalesced_f32(device const float4* operand [[buffer(0)]],
                                       device const int* indices [[buffer(1)]],
                                       device float4* output [[buffer(2)]],
                                       constant uint& num_indices [[buffer(3)]],
                                       constant uint& cols4 [[buffer(4)]],
                                       constant uint& table_rows [[buffer(5)]],
                                       uint2 tgid [[threadgroup_position_in_grid]],
                                       uint2 tid [[thread_position_in_threadgroup]],
                                       uint2 tg_size [[threads_per_threadgroup]]) {
    uint row = tgid.x * tg_size.x + tid.x;
    if (row >= num_indices) return;
    int src_row = indices[row];
    if (src_row < 0 || (uint)src_row >= table_rows) {
        for (uint c = tid.y; c < cols4; c += tg_size.y) {
            output[row * cols4 + c] = float4(0.0f);
        }
        return;
    }
    uint src_base = (uint)src_row * cols4;
    uint dst_base = row * cols4;
    for (uint c = tid.y; c < cols4; c += tg_size.y) {
        output[dst_base + c] = operand[src_base + c];
    }
}
kernel void gather_rows_coalesced_f16(device const half4* operand [[buffer(0)]],
                                       device const int* indices [[buffer(1)]],
                                       device half4* output [[buffer(2)]],
                                       constant uint& num_indices [[buffer(3)]],
                                       constant uint& cols4 [[buffer(4)]],
                                       constant uint& table_rows [[buffer(5)]],
                                       uint2 tgid [[threadgroup_position_in_grid]],
                                       uint2 tid [[thread_position_in_threadgroup]],
                                       uint2 tg_size [[threads_per_threadgroup]]) {
    uint row = tgid.x * tg_size.x + tid.x;
    if (row >= num_indices) return;
    int src_row = indices[row];
    if (src_row < 0 || (uint)src_row >= table_rows) {
        for (uint c = tid.y; c < cols4; c += tg_size.y) {
            output[row * cols4 + c] = half4(0.0h);
        }
        return;
    }
    uint src_base = (uint)src_row * cols4;
    uint dst_base = row * cols4;
    for (uint c = tid.y; c < cols4; c += tg_size.y) {
        output[dst_base + c] = operand[src_base + c];
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
// --- Naive GEMM for small/unaligned matrices ---
// Each thread computes one output element: C[b][r][c] = sum_k(A[b][r][k] * B[b][k][c])
// gid.z = batch index (1 for unbatched)
kernel void gemm_naive_f32(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]],
                           device float* C [[buffer(2)]],
                           constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
                           constant uint& K [[buffer(5)]],
                           uint3 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint r = gid.y;
  if (r >= M || c >= N) return;
  uint batch_off_a = gid.z * M * K;
  uint batch_off_b = gid.z * K * N;
  uint batch_off_c = gid.z * M * N;
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) {
    acc += A[batch_off_a + r * K + k] * B[batch_off_b + k * N + c];
  }
  C[batch_off_c + r * N + c] = acc;
}
kernel void gemm_naive_f16(device const half* A [[buffer(0)]], device const half* B [[buffer(1)]],
                           device half* C [[buffer(2)]],
                           constant uint& M [[buffer(3)]], constant uint& N [[buffer(4)]],
                           constant uint& K [[buffer(5)]],
                           uint3 gid [[thread_position_in_grid]]) {
  uint c = gid.x;
  uint r = gid.y;
  if (r >= M || c >= N) return;
  uint batch_off_a = gid.z * M * K;
  uint batch_off_b = gid.z * K * N;
  uint batch_off_c = gid.z * M * N;
  float acc = 0.0f;
  for (uint k = 0; k < K; k++) {
    acc += (float)A[batch_off_a + r * K + k] * (float)B[batch_off_b + k * N + c];
  }
  C[batch_off_c + r * N + c] = (half)acc;
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
        int bm, bn, bk;         // tile sizes
        int wm, wn;             // simdgroup grid (wm x wn simdgroups)
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

    auto generateGemmKernel = [](const GemmConfig& cfg, const char* dtype, const char* sg_type,
                                 const char* zero_val) -> std::string {
        char name[128];
        snprintf(name, sizeof(name), "gemm_nn_%s_bm%d_bn%d_bk%d", dtype, cfg.bm, cfg.bn, cfg.bk);
        int num_sg = cfg.wm * cfg.wn;
        int num_threads = num_sg * 32;
        bool is_f16 = (strcmp(dtype, "f16") == 0);
        const char* elem_type = is_f16 ? "half" : "float";
        int elem_bytes = is_f16 ? 2 : 4;
        int tg_pad = 16 / elem_bytes; // 4 for float, 8 for half

        // Padded leading dimensions for threadgroup memory
        int a_ld = cfg.bk + tg_pad; // A is BM x BK, stored row-major
        int b_ld = cfg.bn + tg_pad; // B is BK x BN, stored row-major
        int a_tg_size = cfg.bm * a_ld;
        int b_tg_size = cfg.bk * b_ld;

        // Vectorized load parameters for A tile (BM x BK)
        // Each thread loads vec_a elements contiguously along K dimension
        int a_total = cfg.bm * cfg.bk;
        int a_vec = a_total / num_threads; // elements per thread
        // Clamp vec to valid Metal vector sizes (1,2,4)
        if (a_vec > 4)
            a_vec = 4;
        int a_tcols = cfg.bk / a_vec;                   // threads across K
        int a_trows = num_threads / a_tcols;            // threads across M
        int a_nrows = (cfg.bm + a_trows - 1) / a_trows; // iterations per thread

        // Vectorized load parameters for B tile (BK x BN)
        int b_total = cfg.bk * cfg.bn;
        int b_vec = b_total / num_threads;
        if (b_vec > 4)
            b_vec = 4;
        int b_tcols = cfg.bn / b_vec;
        int b_trows = num_threads / b_tcols;
        int b_nrows = (cfg.bk + b_trows - 1) / b_trows;

        // Vector type names
        char a_vec_type[32], b_vec_type[32];
        if (a_vec == 1)
            snprintf(a_vec_type, sizeof(a_vec_type), "%s", elem_type);
        else
            snprintf(a_vec_type, sizeof(a_vec_type), "%s%d", elem_type, a_vec);
        if (b_vec == 1)
            snprintf(b_vec_type, sizeof(b_vec_type), "%s", elem_type);
        else
            snprintf(b_vec_type, sizeof(b_vec_type), "%s%d", elem_type, b_vec);

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

        // Batch offset via tgid.z (1 for unbatched)
        s += "  uint batch_z = tgid.z;\n";

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
                 sg_type, cfg.acc_rows, cfg.acc_cols, cfg.acc_rows, cfg.acc_cols, zero_val);
        s += buf;

        // Threadgroup memory with padding
        snprintf(buf, sizeof(buf),
                 "  threadgroup %s As[%d];\n"
                 "  threadgroup %s Bs[%d];\n",
                 elem_type, a_tg_size, elem_type, b_tg_size);
        s += buf;

        // Precompute loader positions (fixed per thread, no div/mod in loop)
        snprintf(buf, sizeof(buf),
                 "  uint a_row = tid / %du;\n"
                 "  uint a_col = (tid %% %du) * %du;\n"
                 "  uint b_row = tid / %du;\n"
                 "  uint b_col = (tid %% %du) * %du;\n",
                 a_tcols, a_tcols, a_vec, b_tcols, b_tcols, b_vec);
        s += buf;

        // Device memory pointers for A and B (with batch offset, advanced each K iteration)
        s += "  device const ";
        s += elem_type;
        s += "* Aptr = A + batch_z * M * K + bm * K;\n";
        s += "  device const ";
        s += elem_type;
        s += "* Bptr = B + batch_z * K * N + bn;\n";

        // K-loop
        snprintf(buf, sizeof(buf), "  for (uint kb = 0; kb < K; kb += %du) {\n", cfg.bk);
        s += buf;

        // Load A tile: BM x BK -> As[BM x (BK+pad)]
        // Each thread loads a_vec elements per row, iterating over a_nrows rows
        s += "    threadgroup_barrier(mem_flags::mem_threadgroup);\n";
        if (a_nrows == 1) {
            if (a_vec == 1) {
                snprintf(buf, sizeof(buf), "    As[a_row * %d + a_col] = Aptr[a_row * K + kb + a_col];\n",
                         a_ld);
            } else {
                snprintf(buf, sizeof(buf),
                         "    *(threadgroup %s*)(As + a_row * %d + a_col) = *(device const %s*)(Aptr + a_row "
                         "* K + kb + a_col);\n",
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
                snprintf(buf, sizeof(buf), "      As[row * %d + a_col] = Aptr[row * K + kb + a_col];\n",
                         a_ld);
            } else {
                snprintf(buf, sizeof(buf),
                         "      *(threadgroup %s*)(As + row * %d + a_col) = *(device const %s*)(Aptr + row * "
                         "K + kb + a_col);\n",
                         a_vec_type, a_ld, a_vec_type);
            }
            s += buf;
            s += "    }\n";
        }

        // Load B tile: BK x BN -> Bs[BK x (BN+pad)]
        if (b_nrows == 1) {
            if (b_vec == 1) {
                snprintf(buf, sizeof(buf), "    Bs[b_row * %d + b_col] = Bptr[(kb + b_row) * N + b_col];\n",
                         b_ld);
            } else {
                snprintf(buf, sizeof(buf),
                         "    *(threadgroup %s*)(Bs + b_row * %d + b_col) = *(device const %s*)(Bptr + (kb + "
                         "b_row) * N + b_col);\n",
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
                snprintf(buf, sizeof(buf), "      Bs[row * %d + b_col] = Bptr[(kb + row) * N + b_col];\n",
                         b_ld);
            } else {
                snprintf(buf, sizeof(buf),
                         "      *(threadgroup %s*)(Bs + row * %d + b_col) = *(device const %s*)(Bptr + (kb + "
                         "row) * N + b_col);\n",
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
        snprintf(buf, sizeof(buf), "    for (uint kk = 0; kk < %du; kk += 8) {\n", cfg.bk);
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
                         "      simdgroup_multiply_accumulate(acc[%d][%d], a%d, b%d, acc[%d][%d]);\n", ai, bj,
                         ai, bj, ai, bj);
                s += buf;
            }
        }

        s += "    }\n"; // end kk loop
        s += "  }\n";   // end kb loop

        // Store results to device memory (with batch offset)
        for (int ai = 0; ai < cfg.acc_rows; ai++) {
            for (int bj = 0; bj < cfg.acc_cols; bj++) {
                snprintf(buf, sizeof(buf),
                         "  simdgroup_store(acc[%d][%d], C + batch_z * M * N + (bm + sg_row * %du + %du) * N "
                         "+ bn + sg_col * "
                         "%du + %du, N);\n",
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
          combined += generateGemmKernel(cfg, "f32", "simdgroup_float8x8",
                                         "make_filled_simdgroup_matrix<float, 8, 8>(0.0f)");
          combined += generateGemmKernel(cfg, "f16", "simdgroup_half8x8",
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
                          uint3 tgid [[threadgroup_position_in_grid]]) {
    // Batch offset via tgid.z
    device float* A_batch = A_ptr + tgid.z * M * K;
    device float* B_batch = B_ptr + tgid.z * K * N;
    device float* C_batch = C_ptr + tgid.z * M * N;

    auto A = tensor(A_batch, extents<int32_t, dynamic_extent, dynamic_extent>(K, M));
    auto B = tensor(B_batch, extents<int32_t, dynamic_extent, dynamic_extent>(N, K));
    auto C_t = tensor(C_batch, extents<int32_t, dynamic_extent, dynamic_extent>(N, M));

    constexpr auto desc = matmul2d_descriptor(64, 64,
        static_cast<int>(dynamic_extent), false, false, false);
    matmul2d<desc, execution_simdgroups<4>> matmulOp;

    auto mA = A.slice(0, tgid.y * 64);
    auto mB = B.slice(tgid.x * 64, 0);
    auto mC = C_t.slice(tgid.x * 64, tgid.y * 64);
    matmulOp.run(mA, mB, mC);
}

kernel void gemm_mpp_f16(device half* A_ptr [[buffer(0)]],
                          device half* B_ptr [[buffer(1)]],
                          device half* C_ptr [[buffer(2)]],
                          constant uint& M [[buffer(3)]],
                          constant uint& N [[buffer(4)]],
                          constant uint& K [[buffer(5)]],
                          uint3 tgid [[threadgroup_position_in_grid]]) {
    // Batch offset via tgid.z
    device half* A_batch = A_ptr + tgid.z * M * K;
    device half* B_batch = B_ptr + tgid.z * K * N;
    device half* C_batch = C_ptr + tgid.z * M * N;

    auto A = tensor(A_batch, extents<int32_t, dynamic_extent, dynamic_extent>(K, M));
    auto B = tensor(B_batch, extents<int32_t, dynamic_extent, dynamic_extent>(N, K));
    auto C_t = tensor(C_batch, extents<int32_t, dynamic_extent, dynamic_extent>(N, M));

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

NSString* MxQuantizedKernelSource() {
    static const char* kSource = R"METAL(
#include <metal_stdlib>
#include <metal_simdgroup>
using namespace metal;

// ═══════════════════════════════════════════════════════════════════
// MX Microscaling Format Types (OCP Standard + NVIDIA Extension)
// ═══════════════════════════════════════════════════════════════════

// FP4 E2M1: 4-bit float with 1 sign, 2 exponent, 1 mantissa bits.
// Representable values: ±{0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0}
struct fp4_e2m1 {
    uint8_t bits;

    // Quantize float → 4-bit E2M1 with nearest-even rounding
    fp4_e2m1() thread = default;
    fp4_e2m1(float f) {
        if (isnan(f)) { bits = 0x7; return; }
        uint8_t sign = signbit(f) ? 0x8 : 0x0;
        float a = abs(f);
        uint8_t mag;
        // Nearest-even rounding boundaries
        if      (a >= 5.0f)  mag = 0x7; // 6.0
        else if (a >= 3.5f)  mag = 0x6; // 4.0
        else if (a >= 2.5f)  mag = 0x5; // 3.0
        else if (a >= 1.75f) mag = 0x4; // 2.0
        else if (a >= 1.25f) mag = 0x3; // 1.5
        else if (a >= 0.75f) mag = 0x2; // 1.0
        else if (a >= 0.25f) mag = 0x1; // 0.5
        else                 mag = 0x0; // 0.0
        bits = sign | mag;
    }

    // Dequantize 4-bit E2M1 → half → float
    operator half() const {
        // Bit trick: shift mantissa to half format position and scale
        uint16_t v = (uint16_t)(bits & 0x7) << 9;
        half converted = as_type<half>(v);
        converted *= 16384.0h; // 2^14 to adjust exponent
        return (bits & 0x8) ? -converted : converted;
    }
    operator float() const { return (float)(half)(*this); }
};

// FP8 E4M3: 8-bit float, max representable = 448.
struct fp8_e4m3 {
    uint8_t bits;

    fp8_e4m3() thread = default;
    fp8_e4m3(float f) {
        // From PyTorch E4M3FN conversion
        uint32_t fp8_max = 543u << 21;
        uint32_t denorm_mask = 141u << 23;
        uint32_t f_bits = as_type<uint32_t>(f);
        uint32_t sign = f_bits & 0x80000000u;
        f_bits ^= sign;
        if (f_bits >= fp8_max) {
            bits = 0x7E; // saturate
        } else if (f_bits < (121u << 23)) {
            f_bits = as_type<uint32_t>(as_type<float>(f_bits) + as_type<float>(denorm_mask));
            bits = (uint8_t)(f_bits - denorm_mask);
        } else {
            uint8_t mant_odd = (f_bits >> 20) & 1;
            f_bits += ((uint32_t)(7 - 127) << 23) + 0x7FFFFu;
            f_bits += mant_odd;
            bits = (uint8_t)(f_bits >> 20);
        }
        bits |= (uint8_t)(sign >> 24);
    }

    operator half() const {
        uint16_t v = (uint16_t)(bits & 127) << 7;
        half converted = as_type<half>(v);
        converted *= 256.0h;
        return (bits & 128) ? -converted : converted;
    }
    operator float() const { return (float)(half)(*this); }
};

// FP8 E8M0: pure power-of-two scale (8 exponent bits, no mantissa).
// Represents 2^(bits - 127). Range: 2^-127 to 2^127.
struct fp8_e8m0 {
    uint8_t bits;

    fp8_e8m0() thread = default;
    fp8_e8m0(float x) {
        if (!isfinite(x)) { bits = 0xFF; return; }
        if (x <= 0.0f) { bits = 0x00; return; }
        float le = log2(x);
        int n = (int)rint(le);
        n = clamp(n, -127, 127);
        bits = (uint8_t)(n + 127);
    }

    operator float() const {
        // Reconstruct: place exponent byte directly into IEEE 754 exponent field
        uint32_t out = (bits == 0) ? 0x00400000u : ((uint32_t)bits << 23);
        return as_type<float>(out);
    }
};

// ═══════════════════════════════════════════════════════════════════
// Dequantize helpers
// ═══════════════════════════════════════════════════════════════════

// Dequantize scale byte based on group size (compile-time selection).
// group_size==16 → NVFP4 (E4M3 scale), group_size==32 → MX (E8M0 scale)
template <int group_size>
inline float dequantize_scale(uint8_t s) {
    if (group_size == 16) {
        fp8_e4m3 sc; sc.bits = s;
        return (float)sc;
    } else {
        fp8_e8m0 sc; sc.bits = s;
        return (float)sc;
    }
}

// Dequantize a single 4-bit element
inline float dequant_fp4(uint8_t nibble) {
    fp4_e2m1 v; v.bits = nibble & 0xF;
    return (float)v;
}

// Dequantize a single 8-bit E4M3 element
inline float dequant_fp8(uint8_t byte) {
    fp8_e4m3 v; v.bits = byte;
    return (float)v;
}

// ═══════════════════════════════════════════════════════════════════
// Quantize kernels
// ═══════════════════════════════════════════════════════════════════

// Generic MX quantize: f32 → packed uint8 + scale bytes.
// Each SIMD group (32 threads) processes one group for MX modes,
// or two groups (16 threads each) for NV mode.
// Launch as 1D grid: one thread per element.

kernel void mx_quantize_f32(device const float* w       [[buffer(0)]],
                            device uint8_t* out          [[buffer(1)]],
                            device uint8_t* scales       [[buffer(2)]],
                            constant uint& numel         [[buffer(3)]],
                            constant uint& group_size    [[buffer(4)]],
                            constant uint& bits          [[buffer(5)]],
                            uint gid [[thread_position_in_grid]],
                            uint simd_lid [[thread_index_in_simdgroup]]) {
    if (gid >= numel) return;

    float val = w[gid];
    float aval = abs(val);

    // Compute per-group max using SIMD reduction
    float group_max;
    if (group_size == 32) {
        group_max = simd_max(aval);
    } else {
        // NV mode: group_size=16, split SIMD into two halves
        float left_max  = simd_max(simd_lid < 16 ? aval : 0.0f);
        float right_max = simd_max(simd_lid >= 16 ? aval : 0.0f);
        group_max = (simd_lid < 16) ? left_max : right_max;
    }

    // Normalize by max representable value of element type
    float max_repr = (bits == 4) ? 6.0f : 448.0f;
    float raw_scale = group_max / max_repr;

    // Quantize scale
    uint8_t q_scale;
    float scale;
    if (group_size == 32) {
        // MX mode: E8M0 scale
        fp8_e8m0 s(raw_scale);
        q_scale = s.bits;
        scale = (float)s;
    } else {
        // NV mode: E4M3 scale
        fp8_e4m3 s(raw_scale);
        q_scale = s.bits;
        scale = (float)s;
    }

    // Write scale (first thread in group writes)
    uint group_idx = gid / group_size;
    if (gid % group_size == 0) {
        scales[group_idx] = q_scale;
    }

    // Quantize element
    float normalized = (scale == 0.0f) ? 0.0f : val / scale;
    uint8_t quantized;
    if (bits == 4) {
        fp4_e2m1 q(normalized);
        quantized = q.bits;
    } else {
        fp8_e4m3 q(normalized);
        quantized = q.bits;
    }

    // Pack FP4: two elements per byte
    if (bits == 4) {
        uint8_t neighbor = simd_shuffle_down(quantized, 1);
        quantized |= (neighbor << 4);
        if (gid % 2 == 0) {
            out[gid / 2] = quantized;
        }
    } else {
        out[gid] = quantized;
    }
}

// ═══════════════════════════════════════════════════════════════════
// Dequantize kernels
// ═══════════════════════════════════════════════════════════════════

kernel void mx_dequantize_f32(device const uint8_t* packed  [[buffer(0)]],
                              device const uint8_t* scales   [[buffer(1)]],
                              device float* out              [[buffer(2)]],
                              constant uint& numel           [[buffer(3)]],
                              constant uint& group_size      [[buffer(4)]],
                              constant uint& bits            [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= numel) return;

    uint group_idx = gid / group_size;

    // Dequantize scale
    float scale;
    if (group_size == 32) {
        fp8_e8m0 s; s.bits = scales[group_idx];
        scale = (float)s;
    } else {
        fp8_e4m3 s; s.bits = scales[group_idx];
        scale = (float)s;
    }

    // Dequantize element
    float val;
    if (bits == 4) {
        uint8_t byte = packed[gid / 2];
        uint8_t nibble = (gid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        val = dequant_fp4(nibble);
    } else {
        val = dequant_fp8(packed[gid]);
    }

    out[gid] = scale * val;
}

kernel void mx_dequantize_f16(device const uint8_t* packed  [[buffer(0)]],
                              device const uint8_t* scales   [[buffer(1)]],
                              device half* out               [[buffer(2)]],
                              constant uint& numel           [[buffer(3)]],
                              constant uint& group_size      [[buffer(4)]],
                              constant uint& bits            [[buffer(5)]],
                              uint gid [[thread_position_in_grid]]) {
    if (gid >= numel) return;

    uint group_idx = gid / group_size;
    float scale;
    if (group_size == 32) {
        fp8_e8m0 s; s.bits = scales[group_idx];
        scale = (float)s;
    } else {
        fp8_e4m3 s; s.bits = scales[group_idx];
        scale = (float)s;
    }

    float val;
    if (bits == 4) {
        uint8_t byte = packed[gid / 2];
        uint8_t nibble = (gid % 2 == 0) ? (byte & 0x0F) : (byte >> 4);
        val = dequant_fp4(nibble);
    } else {
        val = dequant_fp8(packed[gid]);
    }

    out[gid] = (half)(scale * val);
}

// ═══════════════════════════════════════════════════════════════════
// Quantized GEMM: C[M,N] = A[M,K] @ dequant(B_packed[N,K/pack], B_scales[N,K/gs])^T
// Weights B are stored transposed and packed. Each row of B_packed is one output neuron.
// This is the standard inference layout: activations × weight^T.
//
// Strategy: each thread computes one output element by iterating over K
// with on-the-fly dequantization. Simple but effective for moderate sizes.
// ═══════════════════════════════════════════════════════════════════

// --- Quantized matrix-vector multiply (batch of rows) ---
// Each SIMD group handles one output row. Threads cooperatively reduce over K.

kernel void mx_qgemm_mxfp4_f32(
    device const float* A          [[buffer(0)]],  // [M, K]
    device const uint8_t* B_packed [[buffer(1)]],  // [N, K/2] packed FP4
    device const uint8_t* B_scales [[buffer(2)]],  // [N, K/32] E8M0 scales
    device float* C                [[buffer(3)]],  // [M, N]
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    // Each threadgroup handles a tile of output: TILE_M rows × TILE_N cols
    // With 128 threads = 4 SIMD groups, each SIMD handles one (row, col) pair
    const uint TILE_M = 4;  // rows per threadgroup
    const uint TILE_N = 1;  // cols per threadgroup (1 SIMD per output)
    const uint GROUP_SIZE = 32;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    // Pointers
    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);   // K/2 bytes for FP4
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;

    // Process K in groups of 32 (one scale per group)
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        // Dequantize scale for this group
        fp8_e8m0 sc; sc.bits = s_row[g];
        float scale = (float)sc;

        // Each lane processes 1 element within the group (32 lanes = 32 elements)
        // But we need all 32 elements, so each lane handles its own element
        uint k_base = g * GROUP_SIZE;
        uint k = k_base + lane;

        // Load activation
        float a_val = a_row[k];

        // Load and dequantize weight (FP4 packed: 2 per byte)
        uint byte_idx = k / 2;
        uint8_t packed_byte = b_row[byte_idx];
        uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        float w_val = scale * dequant_fp4(nibble);

        // Multiply and reduce across SIMD
        float prod = a_val * w_val;
        acc += simd_sum(prod);
    }

    // Write output (only lane 0 writes)
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_mxfp4_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint TILE_M = 4;
    const uint GROUP_SIZE = 32;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        fp8_e8m0 sc; sc.bits = s_row[g];
        float scale = (float)sc;
        uint k = g * GROUP_SIZE + lane;
        float a_val = (float)a_row[k];
        uint8_t packed_byte = b_row[k / 2];
        uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
        float w_val = scale * dequant_fp4(nibble);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

kernel void mx_qgemm_mxfp8_f32(
    device const float* A          [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],  // [N, K] raw E4M3 bytes
    device const uint8_t* B_scales [[buffer(2)]],  // [N, K/32] E8M0 scales
    device float* C                [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint TILE_M = 4;
    const uint GROUP_SIZE = 32;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        fp8_e8m0 sc; sc.bits = s_row[g];
        float scale = (float)sc;
        uint k = g * GROUP_SIZE + lane;
        float a_val = a_row[k];
        float w_val = scale * dequant_fp8(b_row[k]);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_mxfp8_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint TILE_M = 4;
    const uint GROUP_SIZE = 32;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * K;
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        fp8_e8m0 sc; sc.bits = s_row[g];
        float scale = (float)sc;
        uint k = g * GROUP_SIZE + lane;
        float a_val = (float)a_row[k];
        float w_val = scale * dequant_fp8(b_row[k]);
        acc += simd_sum(a_val * w_val);
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

kernel void mx_qgemm_nvfp4_f32(
    device const float* A          [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],  // [N, K/2] packed FP4
    device const uint8_t* B_scales [[buffer(2)]],  // [N, K/16] E4M3 scales
    device float* C                [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint TILE_M = 4;
    const uint GROUP_SIZE = 16;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const float* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        fp8_e4m3 sc; sc.bits = s_row[g];
        float scale = (float)sc;
        // NV mode: group_size=16, only first 16 lanes active per group
        uint k = g * GROUP_SIZE + (lane % 16);
        if (lane < 16) {
            float a_val = a_row[k];
            uint8_t packed_byte = b_row[k / 2];
            uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            float w_val = scale * dequant_fp4(nibble);
            acc += simd_sum(a_val * w_val);
        } else {
            // Upper 16 lanes contribute 0 to the simd_sum
            acc += simd_sum(0.0f);
        }
    }
    if (lane == 0) {
        C[row * N + col] = acc;
    }
}

kernel void mx_qgemm_nvfp4_f16(
    device const half* A           [[buffer(0)]],
    device const uint8_t* B_packed [[buffer(1)]],
    device const uint8_t* B_scales [[buffer(2)]],
    device half* C                 [[buffer(3)]],
    constant uint& M               [[buffer(4)]],
    constant uint& N               [[buffer(5)]],
    constant uint& K               [[buffer(6)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint simd_idx [[simdgroup_index_in_threadgroup]]) {

    const uint TILE_M = 4;
    const uint GROUP_SIZE = 16;

    uint row = tgid.y * TILE_M + simd_idx;
    uint col = tgid.x;
    if (row >= M || col >= N) return;

    device const half* a_row = A + row * K;
    device const uint8_t* b_row = B_packed + col * (K / 2);
    device const uint8_t* s_row = B_scales + col * (K / GROUP_SIZE);

    float acc = 0.0f;
    uint num_groups = K / GROUP_SIZE;
    for (uint g = 0; g < num_groups; g++) {
        fp8_e4m3 sc; sc.bits = s_row[g];
        float scale = (float)sc;
        uint k = g * GROUP_SIZE + (lane % 16);
        if (lane < 16) {
            float a_val = (float)a_row[k];
            uint8_t packed_byte = b_row[k / 2];
            uint8_t nibble = (k % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);
            float w_val = scale * dequant_fp4(nibble);
            acc += simd_sum(a_val * w_val);
        } else {
            acc += simd_sum(0.0f);
        }
    }
    if (lane == 0) {
        C[row * N + col] = (half)acc;
    }
}

)METAL";
    return [NSString stringWithUTF8String:kSource];
}

} // namespace jax_metallib
