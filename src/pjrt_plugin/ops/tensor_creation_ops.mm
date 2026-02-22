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

static ProcessResult HandleConstant(HandlerContext& ctx) {
    auto constantOp = mlir::dyn_cast<mlir::stablehlo::ConstantOp>(ctx.op);
    if (!constantOp) {
        return ProcessResult::Error("constant: expected ConstantOp");
    }

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("constant: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op);
    auto value = constantOp.getValue();

    bool isEmpty = false;
    for (NSNumber* dim in shape) {
        if ([dim integerValue] == 0) {
            isEmpty = true;
            break;
        }
    }
    if (isEmpty) {

        MPSGraphTensor* result = [ctx.graph constantWithScalar:0 shape:@[@1] dataType:dtype];
        SetOutputTensor(ctx.values, ctx.op, result);
        return ProcessResult{};
    }

    MPSGraphTensor* result = nil;
    if (auto denseAttr = mlir::dyn_cast<mlir::DenseElementsAttr>(value)) {

        if (denseAttr.isSplat()) {
            auto elemType = denseAttr.getElementType();
            double scalarValue = 0.0;

            if (auto complexType = mlir::dyn_cast<mlir::ComplexType>(elemType)) {
                auto complexVal = denseAttr.getSplatValue<std::complex<float>>();
                double realPart = complexVal.real();
                double imagPart = complexVal.imag();
                if (shape.count == 0) {
                    result = [ctx.graph constantWithRealPart:realPart imaginaryPart:imagPart dataType:dtype];
                } else {
                    result = [ctx.graph constantWithRealPart:realPart
                                               imaginaryPart:imagPart
                                                       shape:shape
                                                    dataType:dtype];
                }
                return Result(ctx, result, "constant");
            }

            if (elemType.isF32()) {
                scalarValue = denseAttr.getSplatValue<float>();
            } else if (elemType.isF64()) {
                scalarValue = denseAttr.getSplatValue<double>();
            } else if (elemType.isF16() || elemType.isBF16()) {
                auto apVal = denseAttr.getSplatValue<llvm::APFloat>();
                scalarValue = apVal.convertToFloat();
            } else if (auto intType = mlir::dyn_cast<mlir::IntegerType>(elemType)) {
                auto apInt = denseAttr.getSplatValue<llvm::APInt>();

                if (intType.getWidth() == 64 && (dtype == MPSDataTypeInt64 || dtype == MPSDataTypeUInt64)) {
                    uint64_t bits = apInt.getZExtValue();
                    NSData* data = [NSData dataWithBytes:&bits length:sizeof(bits)];

                    MPSGraphTensor* one = [ctx.graph constantWithData:data shape:@[@1] dataType:dtype];
                    MPSGraphTensor* scalar = [ctx.graph reshapeTensor:one withShape:@[] name:nil];
                    if (shape.count == 0) {
                        result = scalar;
                    } else {
                        result = [ctx.graph broadcastTensor:scalar toShape:shape name:nil];
                    }
                    return Result(ctx, result, "constant");
                }

                if (intType.isUnsigned()) {
                    scalarValue = static_cast<double>(apInt.getZExtValue());
                } else {

                    scalarValue = static_cast<double>(apInt.getSExtValue());
                }
            }

            if (shape.count == 0) {

                result = [ctx.graph constantWithScalar:scalarValue dataType:dtype];
            } else {

                result = [ctx.graph constantWithScalar:scalarValue shape:shape dataType:dtype];
            }
        } else {

            auto rawData = denseAttr.getRawData();
            NSData* data = [NSData dataWithBytes:rawData.data() length:rawData.size()];
            result = [ctx.graph constantWithData:data shape:shape dataType:dtype];
        }
    } else {
        std::string kind = value.getAbstractAttribute().getName().str();
        return ProcessResult::Error("constant: unsupported attribute kind: " + kind +
                                    " (expected DenseElementsAttr)");
    }

    if (!result)
        return ProcessResult::Error("constant: unsupported value type");
    return Result(ctx, result, "constant");
}
REGISTER_MPS_OP("stablehlo.constant", HandleConstant);

static ProcessResult HandleIota(HandlerContext& ctx) {
    auto iotaOp = mlir::dyn_cast<mlir::stablehlo::IotaOp>(ctx.op);
    if (!iotaOp) {
        return ProcessResult::Error("iota: expected IotaOp");
    }

    MPSDataType dtype = GetResultMpsType(ctx.op);
    if (dtype == MPSDataTypeInvalid) {
        return ProcessResult::Error("iota: invalid dtype");
    }

    NSArray<NSNumber*>* shape = GetOutputShape(ctx.op);
    int64_t iotaDim = static_cast<int64_t>(iotaOp.getIotaDimension());

    MPSGraphTensor* result = [ctx.graph coordinateAlongAxis:(NSInteger)iotaDim withShape:shape name:nil];

    if (result.dataType != dtype) {
        result = [ctx.graph castTensor:result toType:dtype name:nil];
    }

    return Result(ctx, result, "iota");
}
REGISTER_MPS_OP("stablehlo.iota", HandleIota);

} // namespace jax_metallib
