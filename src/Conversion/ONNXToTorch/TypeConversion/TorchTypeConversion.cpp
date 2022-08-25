//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

void onnx_mlir::getTorchTypeConversionDependentDialects(
    DialectRegistry &registry) {
  //registry.insert<TorchConversionDialect>();
}

//===----------------------------------------------------------------------===//
// Type conversion setup.
//===----------------------------------------------------------------------===//

static torch::Torch::NonValueTensorType getNonValueTensorFromBuiltinTensor(TensorType type) {
  auto context = type.getContext();
  if (type.isa<RankedTensorType>()) {
    return torch::Torch::NonValueTensorType::get(context, type.getShape(), type.getElementType());
  }
  return torch::Torch::NonValueTensorType::get(context, None, type.getElementType());
}

static void
setupTensorToNonValueTensorConversion(ConversionTarget &target,
                                          TypeConverter &typeConverter) {
  target.addLegalOp<UnrealizedConversionCastOp>();
  typeConverter.addConversion(
      [](TensorType type) -> Optional<Type> {
        return getNonValueTensorFromBuiltinTensor(type);
      });
  typeConverter.addTargetMaterialization([](OpBuilder &builder,
                                            Torch::NonValueTensorType type,
                                            ValueRange inputs,
                                            Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<TensorType>());
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]).getResult(0);
  });
  auto sourceMaterialization = [](OpBuilder &builder, TensorType type,
                                  ValueRange inputs, Location loc) -> Value {
    assert(inputs.size() == 1);
    assert(inputs[0].getType().isa<Torch::BaseTensorType>());
    return builder.create<UnrealizedConversionCastOp>(loc, type, inputs[0]).getResult(0);
  };
  typeConverter.addSourceMaterialization(sourceMaterialization);
  typeConverter.addArgumentMaterialization(sourceMaterialization);
}

void onnx_mlir::setupTorchTypeConversion(
    ConversionTarget &target, TypeConverter &typeConverter) {
  setupTensorToNonValueTensorConversion(target, typeConverter);
}
