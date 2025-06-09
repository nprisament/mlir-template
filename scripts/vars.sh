#!/bin/bash

# Script directory and source directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_DIR="$(realpath "$SCRIPT_DIR/..")"

# LLVM/MLIR related paths
LLVM_SOURCE_DIR="$HOME/llvm-project"
LLVM_BUILD_DIR_RELEASE="$LLVM_SOURCE_DIR/build-release"
LLVM_BUILD_DIR_DEBUG="$LLVM_SOURCE_DIR/build-debug"
LLVM_INSTALL_DIR_RELEASE="$LLVM_BUILD_DIR_RELEASE"  # We install to build directory by default
LLVM_INSTALL_DIR_DEBUG="$LLVM_BUILD_DIR_DEBUG"  # We install to build directory by default

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

# Function to check if a build directory exists
check_build_exists() {
    if [ -d "$1" ]; then
        return 0  # exists
    else
        return 1  # doesn't exist
    fi
}

# Function to print section headers
print_section() {
    echo "=========================================="
    echo "$1"
    echo "=========================================="
}

# Shared function to set build configuration with fallback
set_build_config() {
    local primary_build_dir="$1"
    local primary_install_dir="$2"
    local primary_type="$3"
    local fallback_build_dir="$4"
    local fallback_install_dir="$5"
    local fallback_type="$6"
    local warning_message="$7"
    
    if check_build_exists "$primary_build_dir"; then
        LLVM_BUILD_DIR="$primary_build_dir"
        LLVM_INSTALL_DIR="$primary_install_dir"
        MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
        LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
        LLVM_LIT="$LLVM_INSTALL_DIR/bin/llvm-lit"
        CMAKE_BUILD_TYPE="$primary_type"
    elif check_build_exists "$fallback_build_dir"; then
        echo "$warning_message"
        LLVM_BUILD_DIR="$fallback_build_dir"
        LLVM_INSTALL_DIR="$fallback_install_dir"
        MLIR_DIR="$LLVM_INSTALL_DIR/lib/cmake/mlir"
        LLVM_DIR="$LLVM_INSTALL_DIR/lib/cmake/llvm"
        LLVM_LIT="$LLVM_INSTALL_DIR/bin/llvm-lit"
        CMAKE_BUILD_TYPE="$fallback_type"
    else
        echo "Error: No LLVM build found. Please run ./scripts/install.sh first."
        exit 1
    fi
}

# Function to initialize default build directory with fallback
init_default_build() {
    set_build_config \
        "$LLVM_BUILD_DIR_RELEASE" "$LLVM_INSTALL_DIR_RELEASE" "Release" \
        "$LLVM_BUILD_DIR_DEBUG" "$LLVM_INSTALL_DIR_DEBUG" "Debug" \
        "Warning: Release LLVM build not found, using debug build"
}

# Function to set debug build paths
set_debug_build() {
    set_build_config \
        "$LLVM_BUILD_DIR_DEBUG" "$LLVM_INSTALL_DIR_DEBUG" "Debug" \
        "$LLVM_BUILD_DIR_RELEASE" "$LLVM_INSTALL_DIR_RELEASE" "Release" \
        "Warning: Debug LLVM build not found at $LLVM_BUILD_DIR_DEBUG, falling back to release build at $LLVM_BUILD_DIR_RELEASE"
}

# Function to set release build paths
set_release_build() {
    set_build_config \
        "$LLVM_BUILD_DIR_RELEASE" "$LLVM_INSTALL_DIR_RELEASE" "Release" \
        "$LLVM_BUILD_DIR_DEBUG" "$LLVM_INSTALL_DIR_DEBUG" "Debug" \
        "Warning: Release LLVM build not found at $LLVM_BUILD_DIR_RELEASE, falling back to debug build at $LLVM_BUILD_DIR_DEBUG"
}

# Function to print help
print_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --debug     Use debug LLVM build (allows for sanitizer builds)"
    echo "  --release   Use release LLVM build (default)"
    echo "  --help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0              # Use release build"
    echo "  $0 --debug      # Use debug build"
    echo "  $0 --release    # Use release build"
}

# Initialize default build directory
init_default_build
