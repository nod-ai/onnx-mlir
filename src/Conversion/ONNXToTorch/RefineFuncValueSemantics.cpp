//===- RefineFuncValueSemantics.cpp ------------------------------*- C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/Conversion/ONNXToTorch/ONNXToTorchCommon.hpp"

using namespace mlir;
using namespace mlir::torch;
using namespace onnx_mlir;

namespace onnx_mlir {

struct RefineFuncValueSemanticsPass
    : public PassWrapper<RefineFuncValueSemanticsPass, OperationPass<ModuleOp>> {

  StringRef getArgument() const override { return "refine-torch-func-value-semantics"; }

  StringRef getDescription() const override {
    return "Refine the value semantics for functions without callers.";
  }

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  RefineFuncValueSemanticsPass() = default;
  RefineFuncValueSemanticsPass(const RefineFuncValueSemanticsPass &pass)
      : PassWrapper<RefineFuncValueSemanticsPass, OperationPass<ModuleOp>>() {}

  void runOnOperation() override {
    auto module = getOperation();
    module.walk([&](func::FuncOp func) {
      if (func.getVisibility() != SymbolTable::Visibility::Public)
        return;
      if (func.isExternal())
        return;
      auto uses = SymbolTable::getSymbolUses(func, module);
      if (!uses || uses->begin() != uses->end()) {
        func.emitError() << "unimplemented: cannot refine func value semantics for "
                         << "public function with uses";
        return signalPassFailure();
      }
      rewriteSignature(func);
    });
  }

  void rewriteSignature(func::FuncOp func) {
    SmallVector<Type> newTypes;
    SmallVector<Value> newOperands;
    OpBuilder builder(func);
    auto funcType = func.getFunctionType();
    auto b = func->getAttrDictionary();
    b.dump();
    for (auto type : llvm::enumerate(funcType.getInputs())) {
      Type newType = type.value();
      if (auto tensorType = newType.dyn_cast<Torch::BaseTensorType>()) {
        auto valueTensorType = tensorType.getWithValueSemantics();
        newTypes.push_back(valueTensorType);
        auto nonValueTensor = func.getBody().getArgument(type.index());
        nonValueTensor.setType(valueTensorType);
      } else {
        newTypes.push_back(newType);
      }
    }
    // Update the function type.
    func.setType(FunctionType::get(funcType.getContext(), newTypes, funcType.getResults()));
  }
};

} // namespace onnx_mlir

//std::unique_ptr<OperationPass<ModuleOp>>
std::unique_ptr<mlir::Pass>
onnx_mlir::createRefineFuncValueSemanticsPass() {
  return std::make_unique<RefineFuncValueSemanticsPass>();
}
