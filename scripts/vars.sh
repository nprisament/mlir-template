#!/bin/bash

# Script directory and source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(realpath "$SCRIPT_DIR/..")"

# LLVM/MLIR related paths
LLVM_SOURCE_DIR="$HOME/llvm-project"
LLVM_BUILD_DIR="$LLVM_SOURCE_DIR/build-release"
LLVM_INSTALL_DIR="$LLVM_BUILD_DIR"  # We install to build directory by default
MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
LLVM_LIT="$LLVM_INSTALL_DIR/bin/llvm-lit"

# Project build directory
PROJECT_BUILD_DIR="$SOURCE_DIR/build"

# Build configuration
CMAKE_GENERATOR="Ninja"
CMAKE_BUILD_TYPE="Release"
CMAKE_INSTALL_PREFIX="$PROJECT_BUILD_DIR"

# Number of parallel jobs for building
NUM_JOBS=$(nproc)

# Function to check if a directory exists
check_directory() {
    if [ ! -d "$1" ]; then
        echo "Error: Directory $1 does not exist."
        exit 1
    fi
}

# Function to print section headers
print_section() {
    echo "=========================================="
    echo "$1"
    echo "=========================================="
} 