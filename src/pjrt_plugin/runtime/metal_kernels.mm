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
  uint vec_cols = cols & ~3u;
  for (uint i = tid * 4; i < vec_cols; i += tg_size * 4) {
    float4 v = *reinterpret_cast<device const float4*>(x + base + i);
    acc += v[0] + v[1] + v[2] + v[3];
  }
  for (uint i = vec_cols + tid; i < cols; i += tg_size) acc += x[base + i];
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
  uint vec_cols = cols & ~3u;
  for (uint i = tid * 4; i < vec_cols; i += tg_size * 4) {
    float4 v = *reinterpret_cast<device const float4*>(x + base + i);
    acc = max(acc, max(max(v[0], v[1]), max(v[2], v[3])));
  }
  for (uint i = vec_cols + tid; i < cols; i += tg_size) acc = max(acc, x[base + i]);
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
  uint vec_cols = cols & ~3u;
  for (uint i = tid * 4; i < vec_cols; i += tg_size * 4) {
    float4 v = *reinterpret_cast<device const float4*>(x + base + i);
    acc = min(acc, min(min(v[0], v[1]), min(v[2], v[3])));
  }
  for (uint i = vec_cols + tid; i < cols; i += tg_size) acc = min(acc, x[base + i]);
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
    uint vec_cols = cols & ~3u;
    for (uint i = 0; i < vec_cols; i += 4) {
      float4 v = *reinterpret_cast<device const float4*>(x + base + i);
      acc += v[0] + v[1] + v[2] + v[3];
    }
    for (uint i = vec_cols; i < cols; ++i) acc += x[base + i];
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
    uint vec_cols = cols & ~3u;
    for (uint i = 0; i < vec_cols; i += 4) {
      float4 v = *reinterpret_cast<device const float4*>(x + base + i);
      acc = max(acc, max(max(v[0], v[1]), max(v[2], v[3])));
    }
    for (uint i = vec_cols; i < cols; ++i) acc = max(acc, x[base + i]);
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
    uint vec_cols = cols & ~3u;
    for (uint i = 0; i < vec_cols; i += 4) {
      float4 v = *reinterpret_cast<device const float4*>(x + base + i);
      acc = min(acc, min(min(v[0], v[1]), min(v[2], v[3])));
    }
    for (uint i = vec_cols; i < cols; ++i) acc = min(acc, x[base + i]);
    y[gid] = acc;
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
    static NSString* source = nil;
    static dispatch_once_t onceToken;
    dispatch_once(&onceToken, ^{
      source = [[NSString alloc] initWithUTF8String:kSource];
    });
    return source;
}

} // namespace jax_metallib
