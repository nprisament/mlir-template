add_subdirectory(Dialect)
add_subdirectory(Transforms)
add_subdirectory(Pipeline)

# Create a combined library that includes all components
add_library(MLIRStandaloneCombined INTERFACE)
target_link_libraries(MLIRStandaloneCombined
  INTERFACE
  
  MLIRStandalone
  StandalonePipeline
  StandaloneTransforms
)
