# An out-of-tree template for MLIR

This repository contains a template for an out-of-tree [MLIR](https://mlir.llvm.org/) dialect as well as a standalone `opt`-like tool to operate on that dialect. This template provides a starting point for developing custom MLIR dialects, Transforms and Pass Pipelines outside of the LLVM tree.

## Prerequisites

Before building this project, you need to have:

1. **LLVM and MLIR**: A recent version of LLVM/MLIR (built from source)
   - Required components: MLIR, LLVM, and LLVM utilities
   - Build LLVM with: `-DLLVM_INSTALL_UTILS=ON` to get required tools like `FileCheck`
   - Recommended: Build with `-DLLVM_ENABLE_ASSERTIONS=ON` for development

2. **Build Tools**:
   - CMake (version 3.13.4 or higher)
   - Ninja (recommended) or Make
   - C++ compiler with C++20 support
   - Python 3 (for running tests)

3. **Optional but Recommended**:
   - LLD linker (for faster linking)
   - clang-format and clang-tidy (for code formatting and linting)

## Utility Scripts

The project includes several utility scripts to help with setup, building, and customization:

### Installation Script (`scripts/install.sh`)

This script automates the installation of all required dependencies and builds LLVM/MLIR:

```sh
# Build LLVM in release mode (default)
./scripts/install.sh

# Build LLVM in debug mode (includes sanitizers)
./scripts/install.sh --debug

# Build LLVM in release mode (explicit)
./scripts/install.sh --release

# Show help
./scripts/install.sh --help
```

The script will:
1. Install system dependencies that aren't already installed (including git, cmake, ninja, python3, clang, etc.)
2. Clone/update the LLVM repository
3. Build LLVM/MLIR with the required components
4. Set up the environment for building the project

> **Note**: This script requires sudo access to install system packages. It currently supports systems with apt-get, dnf, or yum package managers.

### Build Script (`scripts/build.sh`)

This script provides a convenient way to build the project with the correct configuration:

```sh
# Build project using release LLVM (default)
./scripts/build.sh

# Build project using debug LLVM
./scripts/build.sh --debug

# Build project using release LLVM (explicit)
./scripts/build.sh --release

# Show help
./scripts/build.sh --help
```

The script will:
1. Check that LLVM/MLIR is properly built
2. Create and configure the build directory
3. Build the project with the correct settings
4. Use the optimal number of parallel jobs for your system

> **Note**: This script uses variables defined in `scripts/vars.sh` if you are using this script independant of `scripts/install.sh`, make sure to update the variables in the former script appropriately

### Test Runner Script (`scripts/run-tests.sh`)

This script provides a convenient way to run different types of tests and benchmarks. By default, it runs all tests and benchmarks using the normal build:

```sh
# Run all tests and benchmarks using normal build (default)
./scripts/run-tests.sh

# Run all tests and benchmarks using sanitizer build
./scripts/run-tests.sh -b sanitizer

# Run all tests and benchmarks using normal build (explicit)
./scripts/run-tests.sh -b normal

# Run specific types of tests
./scripts/run-tests.sh -t all          # Run all tests AND benchmarks
./scripts/run-tests.sh -t tests        # Run all tests (unit AND LIT tests)
./scripts/run-tests.sh -t unit         # Run only unit tests
./scripts/run-tests.sh -t lit          # Run only LIT tests
./scripts/run-tests.sh -t benchmark    # Run all benchmarks
./scripts/run-tests.sh -t basic-bench  # Run only basic benchmarks
./scripts/run-tests.sh -t asymptotic-bench  # Run only asymptotic benchmarks

# Combine build type and test type
./scripts/run-tests.sh -b sanitizer -t unit    # Run unit tests with sanitizer build
./scripts/run-tests.sh -b normal -t benchmark  # Run benchmarks with normal build

# Run with specific number of parallel jobs
./scripts/run-tests.sh -j 4            # Use 4 parallel jobs

# Run with verbose output
./scripts/run-tests.sh -v              # Show detailed build output

# Use custom build directory
./scripts/run-tests.sh -b /path/to/custom/build -t all

# Show help message
./scripts/run-tests.sh -h
```

The script will:
1. Check that the specified build directory exists
2. Run the selected test type(s) in the specified build
3. Show clear progress and results
4. Exit with appropriate status code on failure

> **Note**: This script uses variables defined in `scripts/vars.sh` if you are using this script independant of `scripts/install.sh`, make sure to update the variables in the former script appropriately

### Project Renaming Script (`scripts/rename-project.sh`)

This script helps you create a new project based on this template:

```sh
# Create a new project with a different name
./scripts/rename-project.sh <old_name> <new_name> <destination_path>

# Example: Create a new project called "MyDialect"
./scripts/rename-project.sh Standalone MyDialect /path/to/new/project
```

The script will:
1. Copy the entire project to the new location
2. Rename all occurrences of the old name to the new name
3. Update file contents, paths, and namespaces
4. Remove the .git directory from the copy
5. Provide next steps for setting up the new project

> **Note**: The script requires Python 3 and will prompt for confirmation before making changes.

### Environment Variables (`scripts/vars.sh`)

This script defines common environment variables used by other scripts, namely the installation and build scripts. You can source it to use these variables in your shell:

```sh
# Source the variables
source scripts/vars.sh

# Now you can use variables like:
echo $LLVM_BUILD_DIR
echo $MLIR_DIR

# Function to switch variables to debug build
set_debug_build

# Switch to release build
set_release_build
```

Key variables include:
- `LLVM_BUILD_DIR`: Path to LLVM build directory
- `MLIR_DIR`: Path to MLIR CMake files
- `LLVM_DIR`: Path to LLVM CMake files
- `PROJECT_BUILD_DIR`: Path to project build directory

### Sanitizer Build Script (`scripts/build-sanitizer.sh`)

For building and testing with sanitizers:

```sh
# Build with sanitizers (uses debug LLVM by default)
./scripts/build-sanitizer.sh

# Build with sanitizers using debug LLVM (explicit)
./scripts/build-sanitizer.sh --debug

# Build with sanitizers using release LLVM
./scripts/build-sanitizer.sh --release

# Show help
./scripts/build-sanitizer.sh --help
```

**Sanitizer Build Features:**
- Automatically uses debug LLVM build (recommended for sanitizers)
- Enables AddressSanitizer and UndefinedBehaviorSanitizer for your project
- Runs both unit tests and LIT tests with sanitizers
- Isolated build directory (`build-sanitizer`) to avoid conflicts
- Catches memory issues and undefined behavior during development

After Building with the sanatizer, use standard testing commands but with the `build-sanitizer` directory instead of `build`.

#### Build Directory Structure

The scripts use separate build directories for different configurations:

```
$HOME/llvm-project/
├── build-release/          # Release LLVM build
└── build-debug/            # Debug LLVM build (with sanitizers)

mlir-template/
├── build/                  # Main project build
└── build-sanitizer/        # Sanitizer-enabled build
```

## Building the Project

> **Note**: All commands in this guide should be run from the project root directory unless otherwise specified.

### 1. Set up Environment Variables

First, set up your environment variables. You'll need paths to your LLVM/MLIR build:

```sh
# If you built LLVM/MLIR from source:
export LLVM_BUILD_DIR=/path/to/llvm-project/build # export LLVM_BUILD_DIR=$HOME/llvm-project/build
export MLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
export LLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"
export LLVM_EXTERNAL_LIT="${LLVM_BUILD_DIR}/bin/llvm-lit"

# If you installed LLVM/MLIR to a prefix:
export PREFIX=/path/to/llvm/install
export MLIR_DIR="${PREFIX}/lib/cmake/mlir"
export LLVM_EXTERNAL_LIT="${PREFIX}/bin/llvm-lit"
```

### 2. Configure and Build

```sh
# Create and enter build directory
mkdir build && cd build

# Configure with CMake
cmake -G Ninja .. \
    -DMLIR_DIR="${MLIR_DIR}" \
    -DLLVM_DIR="${LLVM_DIR}" \
    -DLLVM_EXTERNAL_LIT="${LLVM_EXTERNAL_LIT}"

# Build the project
cmake --build .

# Leave the build directory
cd ..
```

Alternative Workflow Examples:

```sh
# Configure with CMake Without Using Variables
cmake -S . -B build \
    -DLLVM_EXTERNAL_LIT="$HOME/llvm-project/build/bin/llvm-lit" \
    -DLLVM_DIR="$HOME/llvm-project/build-release/lib/cmake/llvm" \
    -DMLIR_DIR="$HOME/llvm-project/build-release/lib/cmake/mlir"

# Build only a specific subset of targets
cmake --build build --target <standalone-opt|unittests|benchmarks|asymptotic-benchmarks>
```

### 3. Running Tests and Benchmarks

The project includes both unit tests and MLIR LIT tests. You can run them using:

```sh
# Run all tests
cmake --build build --target run_all_tests

# Run only gtest unit tests
cmake --build build --target run_unit_tests

# Run only LIT tests
cmake --build build --target run_lit_tests
```

The project includes benchmarks to measure performance:

```sh
# Run all benchmarks
cmake --build build --target run_all_benchmarks

# Run basic benchmarks only
cmake --build build --target run_basic_benchmarks

# Run asymptotic benchmarks only
cmake --build build --target run_asymptotic_benchmarks
```

To run all of both the tests and benchmarks:

```sh
cmake --build build --target run_all
```

### 4. Building Documentation

To generate documentation from the TableGen description of the dialect operations:

```sh
cmake --build build --target mlir-doc
```

## Project Structure

- `include/`: Public headers
- `lib/`: Implementation files
- `test/`: MLIR LIT tests
- `unittest/`: C++ unit tests
- `benchmark/`: Performance benchmarks
- `tools/`: Command-line executables built by this project
- `scripts/`: Build and utility scripts
- `cmake/`: CMake modules and configuration

## Development

### Code Style

The project uses clang-format and clang-tidy for code formatting and linting. Configuration files are provided:
- `.clang-format`: Code formatting rules
- `.clang-tidy`: Linting rules

You can format your code using the provided `scripts/format_code.sh` script. This script will:
- Format all C++ source files using clang-format
- Run clang-tidy checks on all C++ source files
- Apply suggested fixes automatically where possible
- Skip formatting of generated files and build directories
- Show a summary of changes made

To use the script, simply run:

```sh
./scripts/format_code.sh
```

### Adding New Passes and Operations

1. Define the pass or operation in TableGen (`.td` files)
2. Implement the pass or operation in C++
3. Add unit tests and system tests
4. Add benchmarks if performance is critical

## Excluded Features and Alternatives

This template does not include:
1. C-API Bindings
2. Python Bindings

Other templates/examples and alternatives include:
1. [jmgorius/mlir-standalone-template](https://github.com/jmgorius/mlir-standalone-template/tree/main): The template that this one is based on!
2. [codes1gn/mlir-boilerplate](https://github.com/codes1gn/mlir-boilerplate/tree/master#): A template that this project took major examples from.
2. [makslevental/mmlir](https://github.com/makslevental/mmlir/tree/main): A minimal (really) out-of-tree MLIR example [includes c-api and python]
3. [llvm-project/mlir/examples/standalone](https://github.com/llvm/llvm-project/tree/main/mlir/examples/standalone): An out-of-tree MLIR dialect [includes c-api and python]
4. [llvm-project/mlir/examples/toy/Ch7](https://github.com/llvm/llvm-project/tree/main/mlir/examples/toy/Ch7): Example MLIR project that includes lowering to LLVM and running with the Execution Engine
5. [j2kun/mlir-tutorial](https://github.com/j2kun/mlir-tutorial/tree/main): MLIR Tutorial that also uses `bazel` for building

## Troubleshooting

Common issues and solutions:

1. **CMake can't find MLIR/LLVM**:
   - Verify your `MLIR_DIR` and `LLVM_DIR` paths
   - Ensure LLVM was built with MLIR enabled
   - Check that the CMake files exist in the specified directories

2. **Missing `FileCheck`**:
   - Rebuild LLVM with `-DLLVM_INSTALL_UTILS=ON`
   - Ensure `FileCheck` is in your PATH

3. **Warning that Google Benchmark was built in `Debug` mode**:
   - This likely means that you are using an install of `libbenchmark-dev`. This library is only built in debug mode for this package. I recommend uninstalling it and this build system will automatically install a new gbench dependancy from github
   - To test this and quickly resolve the issue, comment out the line `find_package(benchmark CONFIG QUIET)` in the top-level `CMakeLists.txt`

3. **Build errors**:
   - Try performing a clean re-build:
   
   ```sh
   rm -rf build
   ./scripts/build.sh
   ```

   - If you are using manually build commands, make sure you supply the correct build directory path from the working directory to the `--build` option

4. **Linker errors**:
   - Try using LLD linker for better performance
   - Check that all required LLVM/MLIR libraries are found
   - If you have added new MLIR includes to your files, make sure they are added to the relevant `CMakeLists.txt` file for linking
   - Consider uncommenting `# -DLLVM_PARALLEL_LINK_JOBS=1` in the install script

## Licensing and Attributions

### License Structure

This project uses a dual licensing approach:

- **New contributions and modifications**: Licensed under the MIT License (`LICENSE.txt`)
- **Original project base**: Licensed under the Apache License v2.0 with LLVM Exceptions (`LICENSE-APACHE20-LLVM.txt`)

### Project Attribution

This project is a fork of the MLIR Standalone Template originally created by Jean-Michel Gorius.

**Original Project**: [jmgorius/mlir-standalone-template](https://github.com/jmgorius/mlir-standalone-template)  
**Original Author**: Jean-Michel Gorius  
**Original License**: Apache License v2.0 with LLVM Exceptions  
**Original Copyright**: 2020-2023 Jean-Michel Gorius

For complete attribution information and license compliance requirements, please refer to the `NOTICES.txt` file in the project root.

### License Compliance

When distributing this software, you must:
1. Include a copy of the MIT License (`LICENSE.txt`)
2. Include a copy of the Apache License v2.0 with LLVM Exceptions (`LICENSE-APACHE20-LLVM.txt`)
3. Include a copy of the NOTICES file (`NOTICES.txt`)
