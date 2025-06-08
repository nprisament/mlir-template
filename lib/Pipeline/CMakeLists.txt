add_mlir_conversion_library(StandalonePipeline
  Pipelines.cpp

  LINK_COMPONENTS
  Core

  LINK_LIBS PUBLIC
  MLIRIR
)
