#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

namespace mlir {
namespace standalone {

// Bootstrap utility for entire system
void bootstrapStandaloneCompiler(mlir::DialectRegistry &registry);

} // namespace standalone
} // namespace mlir
