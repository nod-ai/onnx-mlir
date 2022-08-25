/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ONNXToMhloCommon.hpp - ONNX dialects to Mhlo lowering --------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains common code shared by the functions performing the
// lowering to the Mhlo dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/TypeSwitch.h"

#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"

#include "src/Dialect/Mlir/IndexExpr.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Transform/ONNX/ConstPropHelper.hpp"
#include "src/Conversion/ONNXToTorch/TypeConversion/TorchTypeConversion.hpp"

//===----------------------------------------------------------------------===//
// Functions to add lowering patterns for frontend operations.
//===----------------------------------------------------------------------===//

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Common functions used when lowering the ONNX frontend dialect to torch-mlir.
//===----------------------------------------------------------------------===//

// `Math` directory methods:
void populateLoweringONNXElementwiseOpToTorchPattern(
    TypeConverter &, RewritePatternSet &, MLIRContext *);

// `Tensor` directory methods:
void populateLoweringONNXConstantOpToTorchPattern(
    TypeConverter &, RewritePatternSet &, MLIRContext *);
} // namespace onnx_mlir

