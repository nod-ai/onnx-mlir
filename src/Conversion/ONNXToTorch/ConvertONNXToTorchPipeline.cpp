/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertONNXToTorch.cpp - ONNX dialects to Torch lowering -------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file registers the pipeline for converting ONNX to Torch Backend IR
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;

namespace onnx_mlir {

void registerONNXFrontendToTorchBackendPasses() {
  //mlir::registerPasses();
  PassPipelineRegistration<>(
    "onnx-to-torch-pipeline",
    "Pipeline converting ONNX to Torch dialect.",
    onnx_mlir::createONNXFrontendToTorchBackendPasses
  );
}

void createONNXFrontendToTorchBackendPasses(
    OpPassManager &pm) {
  pm.addPass(createLowerToTorchPass());
  pm.addPass(createFuncTorchTypeConversionPass());
  pm.addPass(createFinalizingTorchTypeConversionPass());
  pm.addPass(createEraseONNXEntryPointPass());
  pm.addPass(createRefineFuncValueSemanticsPass());
}

} // namespace onnx_mlir
