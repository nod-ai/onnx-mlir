/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ ConvertKrnlToLLVM.cpp - Krnl Dialect Lowering  ---------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file implements the lowering of Krnl operations to a combination of
// other dialects (affine, std, LLVM).
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithmeticToLLVM/ArithmeticToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/ShapeToStandard/ShapeToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Endian.h"

#include "onnx/onnx_pb.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Conversion/KrnlToLLVM/ConvertKrnlToLLVM.hpp"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Conversion/KrnlToLLVM/RuntimeAPI.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
#include "src/Support/Common.hpp"

using namespace mlir;

#define DEBUG_TYPE "krnl_to_llvm"

namespace onnx_mlir {
namespace krnl {

uint64_t KRNL_ENTRY_POINT_ID = 0;

void determineOwnershipForOutputOMTensors(
    ModuleOp &module, SmallVectorImpl<bool> &outputOMTensorOwnerships) {
  Operation *entryPointOp;
  auto walkResult = module->walk([&](mlir::Operation *op) -> WalkResult {
    if (llvm::dyn_cast<KrnlEntryPointOp>(op)) {
      entryPointOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Do nothing if there is no EntryPoint.
  if (!walkResult.wasInterrupted())
    return;

  // Get entry function name.
  StringRef entryPointFuncName =
      entryPointOp
          ->getAttrOfType<SymbolRefAttr>(
              KrnlEntryPointOp::getEntryPointFuncAttrName())
          .getLeafReference()
          .getValue();

  // Get entry function op.
  Operation *entryFunc;
  module->walk([&](func::FuncOp op) -> WalkResult {
    if (SymbolRefAttr::get(op).getValue() == entryPointFuncName) {
      entryFunc = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  assert(entryFunc && "Entry function not found");

  // Get ReturnOp of the entry function op.
  Operation *returnOp;
  entryFunc->walk([&](Operation *op) -> WalkResult {
    if (llvm::dyn_cast<func::ReturnOp>(op)) {
      returnOp = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Check, for each output, if it was transitively produced by a constant or
  // a block argument.
  for (Value v : returnOp->getOperands()) {
    bool shouldOwn = true;
    Operation *definingOp = v.getDefiningOp();
    if (!definingOp)
      // Block argument, do not own this since it is an input that can be owned
      // by an input OMTensor.
      shouldOwn = false;
    else {
      // If output is just a view, trace back to find which op was producing the
      // source memref.
      while (auto viewOp = llvm::dyn_cast<ViewLikeOpInterface>(definingOp)) {
        Value source = viewOp.getViewSource();
        definingOp = source.getDefiningOp();
        // Block argument, stop.
        if (!definingOp)
          break;
      }
      if (!definingOp)
        // Block argument, do not own this since it is an input that can be
        // owned by an input OMTensor.
        shouldOwn = false;
      else if (llvm::dyn_cast<KrnlGlobalOp>(definingOp))
        // Do not own a constant that is defined by KrnlGlobalOp.
        shouldOwn = false;
    }
    outputOMTensorOwnerships.emplace_back(shouldOwn);
    LLVM_DEBUG(llvm::dbgs()
               << "Should the OMTensor own the entry function output? "
               << shouldOwn << "\n");
  }
}

void populateAffineAndKrnlToLLVMConversion(RewritePatternSet &patterns,
    LLVMTypeConverter &typeConverter, MLIRContext *ctx,
    ArrayRef<bool> constantOutputs, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors) {
  // TODO: look at what is done in
  // mlir/lib/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.cpp in function
  // LowerVectorToLLVMPass::runOnOperation() and see what we should do about it.
  // They run it in two steps, and add additional lowerings.

  vector::populateVectorToVectorCanonicalizationPatterns(patterns);
  vector::populateVectorBroadcastLoweringPatterns(patterns);
  vector::populateVectorContractLoweringPatterns(patterns);
  vector::populateVectorTransposeLoweringPatterns(patterns);

  populateAffineToStdConversionPatterns(patterns);
  populateSCFToControlFlowConversionPatterns(patterns);

  populateShapeToStandardConversionPatterns(patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMConversionPatterns(typeConverter, patterns);
  populateVectorToLLVMMatrixConversionPatterns(typeConverter, patterns);
  memref::populateExpandOpsPatterns(patterns);
  // Use polynomial approximation for math.{tanh, sin, cos and exp} for better
  // performance.
  populateMathPolynomialApproximationPatterns(patterns);
  arith::populateArithmeticExpandOpsPatterns(patterns);
  populateMathToLLVMConversionPatterns(typeConverter, patterns);
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithmeticToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  populateReconcileUnrealizedCastsPatterns(patterns);
  krnl::populateKrnlToLLVMConversion(typeConverter, patterns, ctx,
      constantOutputs, singleEntryPoint, entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps, verifyInputTensors);
}

bool hasSingleEntryPoint(ModuleOp &module) {
  uint64_t i = 0;
  module->walk([&](KrnlEntryPointOp entryOp) -> WalkResult {
    if (++i >= 2)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return (i == 1);
}

/// This function emits three functions: omQueryEntryPoints, omInputSignature
/// and omOutputSignature.
/// - omQueryEntryPoints has type of `**i8 (*i64)` to query an array of entry
/// point names.
/// - omInputSignature and omOutputSignature have type of type `*i8 (*i8)` to
/// return input and output signatures of the given entry point.
void genSignatureFunction(ModuleOp &module,
    const SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    const SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    const SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps) {
  MLIRContext *context = module.getContext();
  Location loc = module.getLoc();
  OpBuilder b(context);

  // Common information.
  Type i8Type = IntegerType::get(context, 8);
  Type i32Type = IntegerType::get(context, 32);
  Type i64Type = IntegerType::get(context, 64);
  Type i64PtrTy = LLVM::LLVMPointerType::get(i64Type);
  Type i8PtrTy = LLVM::LLVMPointerType::get(i8Type);
  Type i8PtrPtrTy = LLVM::LLVMPointerType::get(i8PtrTy);
  IntegerAttr zeroI32Attr = b.getI32IntegerAttr(0);
  IntegerAttr zeroI64Attr = b.getI64IntegerAttr(0);
  IntegerAttr oneI64Attr = b.getI64IntegerAttr(1);

  uint64_t numOfEntryPoints = entryGlobalOps.size();

  // A helper function to get a pointer to the first element in an array.
  auto getGlobalOpGEP = [&loc, &b, &i8PtrTy, &i64Type, &zeroI64Attr](
                            LLVM::GlobalOp op) {
    Value zeroI64 = b.create<LLVM::ConstantOp>(loc, i64Type, zeroI64Attr);
    Value address = b.create<LLVM::AddressOfOp>(loc, op);
    LLVM::GEPOp gepOp = b.create<LLVM::GEPOp>(
        loc, i8PtrTy, address, ArrayRef<Value>({zeroI64, zeroI64}));
    return gepOp;
  };

  // Emit a global constant to store an array of pointers pointing to each entry
  // point constants. The array ends with NULL.
  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  auto arrayType = LLVM::LLVMArrayType::get(i8PtrTy, entryGlobalOps.size() + 1);
  auto entryArrayOp = b.create<LLVM::GlobalOp>(loc, arrayType,
      /*isConstant=*/true, LLVM::Linkage::Internal, "_entry_point_arrays",
      Attribute());
  { // Fill the initializer with pointers to entry point constants.
    Region &region = entryArrayOp.getInitializerRegion();
    Block *block = b.createBlock(&region);

    // Initialize an array with the addresses of the global strings.
    b.setInsertionPointToStart(block);
    Value array = b.create<LLVM::UndefOp>(loc, arrayType);

    uint32_t index = 0;
    Value lastValue = array;
    for (const LLVM::GlobalOp &globalOp : entryGlobalOps) {
      LLVM::GEPOp strAddr = getGlobalOpGEP(globalOp);
      lastValue = b.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
          strAddr, b.getArrayAttr({b.getIndexAttr(index++)}));
    }

    // The last element of the array is NULL.
    Value nullPtr = b.create<LLVM::NullOp>(loc, i8PtrTy);
    lastValue = b.create<LLVM::InsertValueOp>(loc, arrayType, lastValue,
        nullPtr, b.getArrayAttr({b.getIndexAttr(index++)}));
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({lastValue}));
  }

  // Emit a function, omQueryEntryPoints, of type `**i8 (*i64)` to query an
  // array of entry point names.
  {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // Emit the function type.
    Type llvmFnType =
        LLVM::LLVMFunctionType::get(i8PtrPtrTy, {i64PtrTy}, false);
    LLVM::LLVMFuncOp funcOp =
        b.create<LLVM::LLVMFuncOp>(loc, "omQueryEntryPoints", llvmFnType);
    // Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);
    Value numOfEntryPoints = entryBlock->getArgument(0);
    // If the argument is not NULL, update its value to return the number of
    // entry points.
    Block *condBlock = b.getInsertionBlock();
    Block *trueBlock = condBlock->splitBlock(b.getInsertionPoint());
    Block *falseBlock = b.createBlock(
        trueBlock->getParent(), std::next(Region::iterator(trueBlock)));
    Block *endBlock = b.createBlock(
        falseBlock->getParent(), std::next(Region::iterator(falseBlock)));
    // Emit code for the condition block: test NULL.
    b.setInsertionPointToEnd(condBlock);
    Value nullPtr = b.create<LLVM::NullOp>(loc, i64PtrTy);
    Value found = b.create<LLVM::ICmpOp>(
        loc, LLVM::ICmpPredicate::ne, numOfEntryPoints, nullPtr);
    // Branch the block into the true and false blocks.
    b.create<LLVM::CondBrOp>(
        loc, found, trueBlock, ValueRange(), falseBlock, ValueRange());

    // Emit code for the true block: update the value.
    b.setInsertionPointToStart(trueBlock);
    Value zero = b.create<LLVM::ConstantOp>(loc, i64Type, zeroI64Attr);
    Value numOfEntryPointsPtr = b.create<LLVM::GEPOp>(
        loc, i64PtrTy, numOfEntryPoints, ArrayRef<Value>({zero}));
    Value noep = b.create<LLVM::ConstantOp>(
        loc, i64Type, b.getI64IntegerAttr(entryGlobalOps.size()));
    b.create<LLVM::StoreOp>(loc, noep, numOfEntryPointsPtr);
    b.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

    // Emit code for the false block: do nothing.
    b.setInsertionPointToStart(falseBlock);
    b.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

    // Emit code for the end block to return the entry point array.
    b.setInsertionPointToStart(endBlock);
    Value entryAddr = b.create<LLVM::AddressOfOp>(loc, entryArrayOp);
    Value entryI8Ptr = b.create<LLVM::BitcastOp>(loc, i8PtrPtrTy, entryAddr);
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({entryI8Ptr}));
  }

  // Emit two signature functions, omInputSignature and omOutputSignature, of
  // type `*i8 (*i8)` at the end of the module.
  SmallVector<std::string, 2> funcNames = {
      "omInputSignature", "omOutputSignature"};
  for (uint64_t i = 0; i < funcNames.size(); ++i) {
    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPointToEnd(module.getBody());
    // 1. Emit the function type.
    Type llvmFnType = LLVM::LLVMFunctionType::get(i8PtrTy, {i8PtrTy}, false);
    LLVM::LLVMFuncOp funcOp =
        b.create<LLVM::LLVMFuncOp>(loc, funcNames[i], llvmFnType);

    // 2. Emit the body of the function.
    Block *entryBlock = funcOp.addEntryBlock();
    OpBuilder::InsertionGuard bodyGuard(b);
    b.setInsertionPointToStart(entryBlock);

    Value zeroI32 = b.create<LLVM::ConstantOp>(loc, i32Type, zeroI32Attr);
    Value oneI64 = b.create<LLVM::ConstantOp>(loc, i64Type, oneI64Attr);

    // 2.1 A buffer to keep a pointer pointing to the return signature string.
    Value ptrToReturnSig = b.create<LLVM::AllocaOp>(loc, i8PtrPtrTy, oneI64,
        /*alignment=*/0);

    // 2.2 The name of the entry point that we want to return its signature.
    Value input = entryBlock->getArgument(0);

    // 2.3 Emit code to find the signature of the given entry point.
    // Iterate over the list of the entry points and check string equality.

    // Split the current block into condition, true, false, and end blocks.
    // - If the user's entry point name is found, go to the true block, then the
    // end block.
    // - Otherwise, recursively split the false block.
    Block *condBlock, *trueBlock, *falseBlock, *endBlock;
    condBlock = b.getInsertionBlock();
    trueBlock = condBlock->splitBlock(b.getInsertionPoint());
    falseBlock = b.createBlock(
        trueBlock->getParent(), std::next(Region::iterator(trueBlock)));
    endBlock = b.createBlock(
        falseBlock->getParent(), std::next(Region::iterator(falseBlock)));

    // Emit code for the end block.
    b.setInsertionPointToStart(endBlock);
    Value res = b.create<LLVM::LoadOp>(loc, i8PtrTy, ptrToReturnSig);
    b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({res}));

    // Emit code for the condition, true and false blocks.
    for (uint64_t j = 0; j < numOfEntryPoints; ++j) {
      LLVM::GlobalOp globalEntryPoint = entryGlobalOps[j];
      LLVM::GlobalOp globalSignature =
          (i == 0) ? inSigGlobalOps[j] : outSigGlobalOps[j];
      assert(globalEntryPoint.getValueAttr().isa<StringAttr>() &&
             "Entry point value is not StringAttr");
      StringAttr entryPointValueAttr =
          globalEntryPoint.getValueAttr().cast<StringAttr>();
      // Emit code for the condition block.
      b.setInsertionPointToEnd(condBlock);
      // Read an entry point name.
      Value entryI8Ptr = getGlobalOpGEP(globalEntryPoint).getResult();
      // Compare it with the user's entry point name.
      FlatSymbolRefAttr StrncmpRef = krnl::getOrInsertStrncmp(b, module);
      Value length = b.create<LLVM::ConstantOp>(loc, i64Type,
          b.getI64IntegerAttr(entryPointValueAttr.getValue().size()));
      Value strncmpResult = b.create<LLVM::CallOp>(loc, i32Type, StrncmpRef,
                                 ArrayRef<Value>({input, entryI8Ptr, length}))
                                .getResult(0);
      // Equal if strncmp returns `0`.
      Value found = b.create<LLVM::ICmpOp>(
          loc, LLVM::ICmpPredicate::eq, strncmpResult, zeroI32);
      // Branch the block into the true and false blocks.
      b.create<LLVM::CondBrOp>(
          loc, found, trueBlock, ValueRange(), falseBlock, ValueRange());

      // Emit code for the true block.
      b.setInsertionPointToStart(trueBlock);
      Value sigAddr = b.create<LLVM::AddressOfOp>(loc, globalSignature);
      Value sigI8Ptr = b.create<LLVM::BitcastOp>(loc, i8PtrTy, sigAddr);
      b.create<LLVM::StoreOp>(loc, sigI8Ptr, ptrToReturnSig);
      b.create<LLVM::BrOp>(loc, ValueRange(), endBlock);

      // Emit code for the false block.
      b.setInsertionPointToStart(falseBlock);
      if (j == numOfEntryPoints - 1) {
        // Return NULL if the entry point name is not found.
        Value nullPtr = b.create<LLVM::NullOp>(loc, i8PtrTy);
        b.create<LLVM::ReturnOp>(loc, ArrayRef<Value>({nullPtr}));
      } else {
        // Recursively do with the other entry point names.
        condBlock = b.getInsertionBlock();
        trueBlock = condBlock->splitBlock(b.getInsertionPoint());
        falseBlock = b.createBlock(
            trueBlock->getParent(), std::next(Region::iterator(trueBlock)));
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Krnl Dialect lowering pass
//===----------------------------------------------------------------------===//

struct ConvertKrnlToLLVMPass
    : public PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertKrnlToLLVMPass)

  // Make sure that we have a valid default constructor and copy
  // constructor to make sure that the options are initialized properly.
  ConvertKrnlToLLVMPass() = default;
  ConvertKrnlToLLVMPass(const ConvertKrnlToLLVMPass &pass)
      : PassWrapper<ConvertKrnlToLLVMPass, OperationPass<ModuleOp>>() {}
  ConvertKrnlToLLVMPass(bool verifyInputTensors) {
    this->verifyInputTensors = verifyInputTensors;
  }

  StringRef getArgument() const override { return "convert-krnl-to-llvm"; }

  StringRef getDescription() const override {
    return "Lower the Krnl Affine and Std dialects to LLVM.";
  }

  void runOnOperation() final;

  Option<bool> verifyInputTensors{*this, "verify-input-tensors",
      llvm::cl::desc(
          "Verify input tensors whenever the entry point function is called.\n"
          "Data type and shape are verified. Enable this may introduce "
          "overhead in inferencing."),
      llvm::cl::init(false)};
};

void ConvertKrnlToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = &getContext();
  const auto &dataLayoutAnalysis = getAnalysis<DataLayoutAnalysis>();
  LowerToLLVMOptions options(ctx, dataLayoutAnalysis.getAtOrAbove(module));
  options.emitCWrappers = true;
  KRNL_ENTRY_POINT_ID = 0;

  // Record entry point names and their input/output signatures.
  // This info is used to generate global signature functions.
  SmallVector<LLVM::GlobalOp, 1> entryGlobalOps, inSigGlobalOps,
      outSigGlobalOps;

  // Determine the module has a single entry point or not.
  bool singleEntryPoint = hasSingleEntryPoint(module);

  // Determine whether an output OMTensor should own the underlying buffer or
  // not.
  SmallVector<bool, 4> outputOMTensorOwnerships;
  determineOwnershipForOutputOMTensors(module, outputOMTensorOwnerships);

  // Define the target for this lowering i.e. the LLVM dialect.
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  // Conversion target for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->conversionTargetKrnlToLLVM(target);

  // Convert types to legal types for the LLVM dialect.
  LLVMTypeConverter typeConverter(ctx, options);
  customizeTypeConverter(typeConverter);

#if 0
  typeConverter.addConversion([&](MemRefType type) -> llvm::Optional<Type> {
    Type elementType = type.getElementType();
    if (!elementType.isa<StringType>())
      return llvm::None;

    elementType = elementType.cast<StringType>().getLLVMType(type.getContext());
    return typeConverter.convertType(
        MemRefType::get(type.getShape(), elementType));
  });

  typeConverter.addConversion([&](StringType type) -> Type {
    return typeConverter.convertType(type.getLLVMType(type.getContext()));
  });
#endif

  // We have a combination of `krnl`, `affine`, `vector`, and `std` operations.
  // We lower in stages until all the code is in the LLVM dialect.
  RewritePatternSet patterns(ctx);

  populateAffineAndKrnlToLLVMConversion(patterns, typeConverter, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, verifyInputTensors);

  // Rewrite patterns for accelerators.
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators())
    accel->rewritePatternKrnlToLLVM(patterns, typeConverter, ctx);

  // We want to completely lower to LLVM, so we use a `FullConversion`. This
  // ensures that only legal operations will remain after the conversion.
  if (failed(
          applyFullConversion(getOperation(), target, std::move(patterns)))) {
    signalPassFailure();
  }

  // Generate signature functions.
  if (entryGlobalOps.size() >= 1)
    genSignatureFunction(
        module, entryGlobalOps, inSigGlobalOps, outSigGlobalOps);
}

/// Create the pass for lowering `Krnl`, `Affine` and `Std` dialects to LLVM.
std::unique_ptr<Pass> createConvertKrnlToLLVMPass() {
  return std::make_unique<ConvertKrnlToLLVMPass>();
}
std::unique_ptr<Pass> createConvertKrnlToLLVMPass(bool verifyInputTensors) {
  return std::make_unique<ConvertKrnlToLLVMPass>(verifyInputTensors);
}

void populateKrnlToLLVMConversion(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx,
    ArrayRef<bool> outputOMTensorOwnerships, bool singleEntryPoint,
    SmallVectorImpl<LLVM::GlobalOp> &entryGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &inSigGlobalOps,
    SmallVectorImpl<LLVM::GlobalOp> &outSigGlobalOps, bool verifyInputTensors) {
  krnl::populateLoweringKrnlEntryPointOpPattern(typeConverter, patterns, ctx,
      outputOMTensorOwnerships, singleEntryPoint, entryGlobalOps,
      inSigGlobalOps, outSigGlobalOps, verifyInputTensors);
  krnl::populateLoweringKrnlCallOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlFindIndexOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGlobalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlGetRefOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlInstrumentOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlMemcpyOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlPrintOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlPrintTensorOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlVectorTypeCastOpPattern(
      typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlRandomNormalOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrlenOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlUnaryMathOpPattern(typeConverter, patterns, ctx);
  krnl::populateLoweringKrnlStrncmpOpPattern(typeConverter, patterns, ctx);
}

} // namespace krnl
} // namespace onnx_mlir