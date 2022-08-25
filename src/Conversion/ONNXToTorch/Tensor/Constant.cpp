/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;

namespace onnx_mlir {

namespace {
class ConvertONNXConstantOp : public OpConversionPattern<ONNXConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ONNXConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.value().has_value())
      return rewriter.notifyMatchFailure(op, "unimplemented: non-dense values are unsupported");
    ElementsAttr value = adaptor.valueAttr();
    auto newResultType = getTypeConverter()->convertType(op.getResult().getType());
    rewriter.replaceOpWithNewOp<Torch::NonValueTensorLiteralOp>(op, newResultType, value);
    return success();
  }
};
} // namespace


void populateLoweringONNXConstantOpToTorchPattern(
    TypeConverter &typeConverter, RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ConvertONNXConstantOp>(typeConverter, ctx);
}

} // namespace onnx_mlir
