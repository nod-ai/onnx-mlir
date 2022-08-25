//===------------------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#ifndef ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H
#define ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H

#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace onnx_mlir {

/// Get the dependent dialects which might be involved in a backend type
/// conversion.
void getTorchTypeConversionDependentDialects(DialectRegistry &registry);

void setupTorchTypeConversion(ConversionTarget &target,
                                TypeConverter &typeConverter);
} // namespace onnx_mlir

#endif // ONNXMLIR_DIALECT_TORCHTYPECONVERSION_H
