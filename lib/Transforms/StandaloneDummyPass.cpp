//===- StandaloneDummyPattern.cpp --- A dummy pass
//------------------------------*-===//

#include "Transforms/Passes.h"
#include "mlir/Dialect/Affine/Passes.h"

#include "mlir/Dialect/Affine/Analysis/AffineAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/AffineStructures.h"
#include "mlir/Dialect/Affine/Analysis/LoopAnalysis.h"
#include "mlir/Dialect/Affine/Analysis/Utils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Affine/IR/AffineValueMap.h"
#include "mlir/Dialect/Affine/LoopUtils.h"
#include "mlir/Dialect/Affine/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include <optional>

using namespace mlir;

#define DEBUG_TYPE "standalone-dummy-pass"

namespace mlir {
namespace {

/// =========================================================
/// Dummy Pattern
/// =========================================================

/// A pass to perform loop tiling on all suitable loop nests of a Function.
struct StandaloneDummyPattern
    : public standalone::StandaloneDummyPassBase<StandaloneDummyPattern> {
  explicit StandaloneDummyPattern() {
    // TODO: add your pattern init code
    // this->data-field-1 = kDefaultTileSize / 1024;
  }

  void runOnOperation() override;
  // TODO: add your pattern data fields and methods
  // void getTileSizes(ArrayRef<AffineForOp> band,
  //                   SmallVectorImpl<unsigned> *tileSizes);
  // constexpr static unsigned kDefaultTileSize = 4;
};

void StandaloneDummyPattern::runOnOperation() {
  // TODO: add your customised pattern implementation
}

} // namespace

// bypass the implementation
std::unique_ptr<OperationPass<mlir::func::FuncOp>>
standalone::createStandaloneDummyPass() {
  return std::make_unique<StandaloneDummyPattern>();
}

} // namespace mlir
