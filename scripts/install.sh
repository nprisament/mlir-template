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

# Function to install a command if it's missing
install_if_missing() {
    local cmd=$1
    local package=$2  # Package name might be different from command name
    
    if ! command -v "$cmd" &> /dev/null; then
        print_section "Installing $cmd"
        if command -v apt-get &> /dev/null; then
            sudo apt-get update
            sudo apt-get install -y "$package"
        elif command -v dnf &> /dev/null; then
            sudo dnf install -y "$package"
        elif command -v yum &> /dev/null; then
            sudo yum install -y "$package"
        else
            echo "Error: Could not find a supported package manager (apt-get, dnf, or yum)"
            exit 1
        fi
    else
        echo "$cmd is already installed"
    fi
}

# Function to install system dependencies
install_system_deps() {
    print_section "Installing System Dependencies"
    
    # Check if we're on a supported system
    if ! command -v apt-get &> /dev/null && ! command -v dnf &> /dev/null && ! command -v yum &> /dev/null; then
        echo "Error: This script currently only supports systems with apt-get, dnf, or yum"
        exit 1
    fi

    # Install essential build tools
    install_if_missing "git" "git"
    install_if_missing "cmake" "cmake"
    install_if_missing "ninja" "ninja-build"
    install_if_missing "python3" "python3"
    install_if_missing "pip3" "python3-pip"
    install_if_missing "clang" "clang"
    install_if_missing "clang-format" "clang-format"
    install_if_missing "lld" "lld"

    # Install additional dependencies with different package names per manager
    if command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            build-essential \
            python3-venv \
            libz-dev \
            libncurses-dev \
            libxml2-dev \
            libedit-dev \
            libsqlite3-dev \
            libgtest-dev
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y \
            gcc-c++ \
            python3-devel \
            zlib-devel \
            ncurses-devel \
            libxml2-devel \
            libedit-devel \
            sqlite-devel \
            gtest-devel
    elif command -v yum &> /dev/null; then
        sudo yum install -y \
            gcc-c++ \
            python3-devel \
            zlib-devel \
            ncurses-devel \
            libxml2-devel \
            libedit-devel \
            sqlite-devel \
            gtest-devel
    fi
}

# Function to clone and update LLVM
setup_llvm() {
    print_section "Setting up LLVM/MLIR"
    
    if [ ! -d "$LLVM_SOURCE_DIR" ]; then
        echo "Cloning LLVM repository..."
        git clone https://github.com/llvm/llvm-project.git "$LLVM_SOURCE_DIR"
    else
        echo "Updating existing LLVM repository..."
        cd "$LLVM_SOURCE_DIR"
        git fetch
        git checkout main
        git pull
    fi
}

# Function to build LLVM/MLIR
build_llvm() {
    print_section "Building LLVM/MLIR ($CMAKE_BUILD_TYPE)"
    
    mkdir -p "$LLVM_BUILD_DIR"
    cd "$LLVM_BUILD_DIR"
    
    # Add sanitizer flags for debug builds
    local SANITIZER_FLAGS=""
    if [ "$CMAKE_BUILD_TYPE" = "Debug" ]; then
        SANITIZER_FLAGS="-DLLVM_USE_SANITIZER='Address;Undefined'"
    fi
    
    cmake "$LLVM_SOURCE_DIR/llvm" \
        -G "$CMAKE_GENERATOR" \
        -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
        -DLLVM_ENABLE_PROJECTS="mlir" \
        -DLLVM_TARGETS_TO_BUILD="host" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_C_COMPILER=clang \
        -DCMAKE_CXX_COMPILER=clang++ \
        -DLLVM_USE_LINKER=lld \
        -DLLVM_INSTALL_UTILS=ON \
        -DLLVM_BUILD_UTILS=ON \
        -DLLVM_INSTALL_TOOLCHAIN_ONLY=OFF \
        $SANITIZER_FLAGS \
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    
    cmake --build . -j "$NUM_JOBS"
}

# Main installation process
main() {
    print_section "Starting Installation: ($CMAKE_BUILD_TYPE)"
    
    # Install system dependencies (including required commands)
    install_system_deps
    
    # Setup and build LLVM/MLIR
    setup_llvm
    build_llvm
    
    print_section "Installation Complete"
    echo "LLVM/MLIR has been built in: $LLVM_BUILD_DIR"
    echo "Build type: $CMAKE_BUILD_TYPE"
    echo "You can now run ./scripts/build.sh to build your project"
}

# Run main function
main 
