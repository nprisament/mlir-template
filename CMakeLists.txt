cmake_minimum_required(VERSION 3.13.4)

project(standalone-dialect LANGUAGES CXX C)

find_program(LLD_PATH NAMES lld)

if (LLD_PATH)
  message(STATUS "Found LLD linker: ${LLD_PATH}. Adding -fuse-ld=lld to link options.")
  add_link_options("-fuse-ld=lld")
else()
  message(STATUS "LLD linker not found. Not adding -fuse-ld=lld to link options.")
endif()

find_package(MLIR REQUIRED CONFIG)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ standard to conform to")

include(TableGen)
include(AddMLIR)
include(AddLLVM)
include(HandleLLVMOptions)

set(CMAKE_CXX_STANDARD 20 CACHE STRING "C++ standard to conform to")

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)
link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

set(LLVM_LIT_ARGS "-sv" CACHE STRING "lit default options")
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake/modules")
include(sanitizers)

# -----------------------------------------------------------------------------
# Hybrid Googletest Inclusion
# -----------------------------------------------------------------------------

enable_testing()

find_package(GTest CONFIG QUIET)

if(NOT GTest_FOUND)
  message(STATUS "GTest library not found, attempting to download with FetchContent...")

  include(FetchContent)

  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        main
  )
  set(gtest_force_shared_crt ON CACHE BOOL "For Windows: Prevent overriding the parent project's compiler/linker settings" FORCE)
  set(BUILD_GMOCK OFF CACHE BOOL "Disable building gmock" FORCE)
  set(BUILD_GTEST_TESTS OFF CACHE BOOL "Disable building gtest tests" FORCE)
  # set(BUILD_GMOCK_TESTS OFF CACHE BOOL "" FORCE)
  set(BUILD_GTEST_EXAMPLES OFF CACHE BOOL "Disable building gtest examples" FORCE)
  # set(BUILD_GMOCK_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(_original_cxx_flags "${CMAKE_CXX_FLAGS}")
  set(_original_c_flags "${CMAKE_C_FLAGS}")
  # Add warning supression to gtest due to clang non-compliance
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-covered-switch-default")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-covered-switch-default")
  FetchContent_MakeAvailable(googletest)
  # Restore original flags to avoid affecting other targets
  set(CMAKE_CXX_FLAGS "${_original_cxx_flags}")
  set(CMAKE_C_FLAGS "${_original_c_flags}")

  message(STATUS "GTest downloaded and configured via FetchContent.")
else()
  message(STATUS "GTest library found at ${GTest_DIR}")
endif()

# Check if we have GTest targets available (either from find_package or FetchContent)
if(TARGET GTest::gtest_main)
  message(STATUS "GTest targets are available")
else()
  message(FATAL_ERROR "Failed to find or download Googletest. Cannot proceed.")
endif()

# -----------------------------------------------------------------------------
# Hybrid Google Benchmark Inclusion
# -----------------------------------------------------------------------------

find_package(benchmark CONFIG QUIET)

if(NOT benchmark_FOUND)
  message(STATUS "Google Benchmark library not found, attempting to download with FetchContent...")

  include(FetchContent)

  FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        main
  )
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark testing" FORCE)
  set(BENCHMARK_ENABLE_EXCEPTIONS OFF CACHE BOOL "Disable benchmark exceptions" FORCE)
  set(BENCHMARK_ENABLE_LTO ON CACHE BOOL "Enable benchmark LTO" FORCE)
  set(BENCHMARK_USE_BUNDLED_GTEST OFF CACHE BOOL "Use the previously fecthed gtest for google benchmark" FORCE)
  FetchContent_MakeAvailable(benchmark)

  message(STATUS "Google Benchmark downloaded and configured via FetchContent.")
else()
  message(STATUS "Google Benchmark library found at ${benchmark_DIR}")
endif()

# Check if we have benchmark targets available (either from find_package or FetchContent)
if(TARGET benchmark::benchmark_main)
  message(STATUS "Google Benchmark targets are available")
else()
  message(FATAL_ERROR "Failed to find or download Google Benchmark. Cannot proceed.")
endif()

# -----------------------------------------------------------------------------
# Subdirectories
# -----------------------------------------------------------------------------

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(test)
add_subdirectory(tools)
add_subdirectory(unittest)
add_subdirectory(benchmark)

# -----------------------------------------------------------------------------
# Custom Run Targets
# -----------------------------------------------------------------------------

add_custom_target(run_unit_tests
  COMMAND $<TARGET_FILE:unittests>
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
  COMMENT "Running Google Tests..."
  DEPENDS unittests
)

add_custom_target(run_lit_tests
  COMMAND ${CMAKE_COMMAND} --build . --target check-standalone-opt
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running LIT Tests..."
)

add_custom_target(run_all_tests
  COMMAND ${CMAKE_COMMAND} --build . --target run_unit_tests run_lit_tests
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(run_basic_benchmarks
  COMMAND $<TARGET_FILE:benchmarks>
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Basic Google Benchmarks..."
  DEPENDS benchmark
)

add_custom_target(run_asymptotic_benchmarks
  COMMAND $<TARGET_FILE:asymptotic-benchmarks>
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Asymptotic Google Benchmarks..."
  DEPENDS asymptotic-benchmarks
)

add_custom_target(run_all_benchmarks
  COMMAND ${CMAKE_COMMAND} --build . --target run_basic_benchmarks run_asymptotic_benchmarks
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

add_custom_target(run_all
  COMMAND ${CMAKE_COMMAND} --build . --target run_all_tests run_all_benchmarks
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
