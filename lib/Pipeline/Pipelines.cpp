#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <mlir/Conversion/Passes.h>
#include <mlir/Dialect/Affine/Passes.h>
#include <mlir/Dialect/Linalg/Passes.h>

#include "Pipeline/Pipelines.h"
#include "Dialect/StandaloneDialect.h"
#include "Transforms/Passes.h"

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// CodeGen-related Pass Pipeline Helpers
//===----------------------------------------------------------------------===//
//
// helper pass-pipeline to convert linalg -> linalg.generic_op
void buildStandaloneDummyPipeline(mlir::OpPassManager &pm) {
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createCSEPass());
}

void registerStandalonePipelines() {
  mlir::PassPipelineRegistration<>("standalone-dummy-pipeline",
                                   "fractal dummy pass pipeline",
                                   buildStandaloneDummyPipeline);
}

/// Add all the MLIR dialects to the provided registry.
/// TODO: AirDialect has issue, cannot find getTypeID impl, fix when needed
inline void registerStandaloneDialects(mlir::DialectRegistry &registry) {
  // clang-format off
  // registry.insert<mlir::standalone::StandaloneDialect>();
  // clang-format on
}

/// Append all the MLIR dialects to the registry contained in the given context.
// inline void registerStandaloneDialects(mlir::MLIRContext &context) {
//   mlir::DialectRegistry registry;
//   registerStandaloneDialects(registry);
//   context.appendDialectRegistry(registry);
// }
} // namespace

namespace mlir {
namespace standalone {

void bootstrapStandaloneCompiler(mlir::DialectRegistry &registry) {

  // register dialect for ChopperRT target
  registerStandaloneDialects(registry);

  // prepare codegen
  // mlir::RTI::registerFractalRTIPasses();
  // mlir::registerFractalCodegenPasses();
  // mlir::registerFractalOptimisationPasses();
  // mlir::registerFractalConversionPasses();

  // prepare pipelines
  registerStandalonePipelines();
}

} // namespace standalone
} // namespace mlir
