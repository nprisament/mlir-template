file(GLOB_RECURSE BENCHMARK_SOURCES "b*.cpp")

add_executable(benchmarks ${BENCHMARK_SOURCES})
target_link_libraries(benchmarks
  PRIVATE
  benchmark::benchmark_main
  MLIRStandaloneCombined
)

# Set output directory to bin
set_target_properties(benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

file(GLOB_RECURSE ASYMPTOTIC_BENCHMARK_SOURCES "a*.cpp")

add_executable(asymptotic-benchmarks ${ASYMPTOTIC_BENCHMARK_SOURCES})
target_link_libraries(asymptotic-benchmarks
  PRIVATE
  benchmark::benchmark_main
  MLIRStandaloneCombined
)

# Set output directory to bin
set_target_properties(asymptotic-benchmarks PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
