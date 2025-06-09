#!/bin/bash

# Source shared variables
source "$(dirname "$0")/vars.sh"

# Default to debug LLVM build for sanitizer builds
set_debug_build

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            set_debug_build
            shift
            ;;
        --release)
            set_release_build
            shift
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_help
            exit 1
            ;;
    esac
done

# Check if LLVM/MLIR is built
check_directory "$LLVM_BUILD_DIR"
check_directory "$MLIR_DIR"
check_directory "$LLVM_DIR"

print_section "Building Project with Sanitizers: ($CMAKE_BUILD_TYPE)"

# Create sanitizer build directory
mkdir -p "$SOURCE_DIR/build-sanitizer"
cd "$SOURCE_DIR/build-sanitizer"

# Configure with sanitizers
cmake "$SOURCE_DIR" \
    -G "$CMAKE_GENERATOR" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DUSE_SANITIZER='Address;Undefined' \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_LIT" \
    -DCMAKE_INSTALL_PREFIX="$SOURCE_DIR/build-sanitizer" \
    -DCMAKE_LINKER=lld \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Build with sanitizers
cmake --build . --target check-standalone-opt run_unit_tests -- -j"$NUM_JOBS"

print_section "Sanitizer Build Complete"
echo "Project has been built in: $(pwd)"
echo "Build type: Debug (with sanitizers)"
echo "LLVM build type: $CMAKE_BUILD_TYPE" 
