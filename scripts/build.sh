#!/bin/bash

# Source shared variables
source "$(dirname "$0")/vars.sh"

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

print_section "Building Project: ($CMAKE_BUILD_TYPE)"

# Create and enter build directory
mkdir -p "$PROJECT_BUILD_DIR"
cd "$PROJECT_BUILD_DIR"

# Configure the project
cmake "$SOURCE_DIR" \
    -G "$CMAKE_GENERATOR" \
    -DMLIR_DIR="$MLIR_DIR" \
    -DLLVM_DIR="$LLVM_DIR" \
    -DLLVM_EXTERNAL_LIT="$LLVM_LIT" \
    -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
    -DCMAKE_INSTALL_PREFIX="$CMAKE_INSTALL_PREFIX" \
    -DLLVM_ENABLE_LLD=ON \
    -DCMAKE_LINKER=lld

# Build the project
cmake --build . -j "$NUM_JOBS"

print_section "Build Complete"
echo "Project has been built in: $PROJECT_BUILD_DIR"
echo "Build type: $CMAKE_BUILD_TYPE"
