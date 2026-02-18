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

#import "pjrt_plugin/ops/registry.h"

namespace jax_metallib {

static ProcessResult HandleFft(HandlerContext& ctx) {
    auto fftOp = mlir::dyn_cast<mlir::stablehlo::FftOp>(ctx.op);
    if (!fftOp) {
        return ProcessResult::Error("fft: expected FftOp");
    }

    MPSGraphTensor* input = GetInputTensor(ctx, 0);
    if (!input)
        return ProcessResult::Error("fft: missing input tensor");

    MPSGraph* g = ctx.graph;

    NSArray<NSNumber*>* inputShape = input.shape;
    NSUInteger rank = inputShape.count;
    auto fftLength = fftOp.getFftLength();
    NSUInteger nAxes = fftLength.size();

    if (nAxes == 0 || nAxes > rank) {
        return ProcessResult::Error("fft: invalid fft_length rank");
    }

    NSMutableArray<NSNumber*>* axes = [NSMutableArray arrayWithCapacity:nAxes];
    NSUInteger startAxis = rank - nAxes;
    for (NSUInteger i = 0; i < nAxes; ++i) {
        [axes addObject:@(startAxis + i)];
    }

    MPSGraphFFTDescriptor* desc = [MPSGraphFFTDescriptor descriptor];
    desc.roundToOddHermitean = NO;

    MPSGraphTensor* result = nil;
    auto fftType = fftOp.getFftType();
    switch (fftType) {
    case mlir::stablehlo::FftType::FFT:
        desc.inverse = NO;
        desc.scalingMode = MPSGraphFFTScalingModeNone;
        result = [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
        break;
    case mlir::stablehlo::FftType::IFFT:
        desc.inverse = YES;
        desc.scalingMode = MPSGraphFFTScalingModeSize;
        result = [g fastFourierTransformWithTensor:input axes:axes descriptor:desc name:nil];
        break;
    case mlir::stablehlo::FftType::RFFT:
        desc.inverse = NO;
        desc.scalingMode = MPSGraphFFTScalingModeNone;
        result = [g realToHermiteanFFTWithTensor:input axes:axes descriptor:desc name:nil];
        break;
    case mlir::stablehlo::FftType::IRFFT:
        desc.inverse = YES;
        desc.scalingMode = MPSGraphFFTScalingModeSize;

        desc.roundToOddHermitean = (fftLength.back() % 2 == 1) ? YES : NO;
        result = [g HermiteanToRealFFTWithTensor:input axes:axes descriptor:desc name:nil];
        break;
    default:
        return ProcessResult::Error("fft: unsupported fft type");
    }

    return Result(ctx, result, "fft");
}
REGISTER_MPS_OP("stablehlo.fft", HandleFft);

} // namespace jax_metallib
