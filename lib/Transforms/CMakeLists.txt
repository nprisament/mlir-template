add_mlir_dialect_library(StandaloneTransforms
  StandaloneDummyPass.cpp

  LINK_COMPONENTS
  Core
  Support

  DEPENDS
  StandaloneTransformsPassesIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRStandalone
  )

