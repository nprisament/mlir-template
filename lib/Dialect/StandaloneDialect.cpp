//===- StandaloneDialect.cpp - Standalone dialect ---------------*- C++ -*-===//

#include "Dialect/StandaloneDialect.h"
#include "Dialect/StandaloneOps.h"

using namespace mlir;
using namespace mlir::standalone;

//===----------------------------------------------------------------------===//
// Standalone dialect.
//===----------------------------------------------------------------------===//

void StandaloneDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "Dialect/StandaloneOps.cpp.inc"
      >();
}
