#!/bin/bash

# Source shared variables
source "$(dirname "$0")/vars.sh"

# Function to print usage
print_usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --type TYPE     Type of tests to run (all, tests, unit, lit, benchmark, basic-bench, asymptotic-bench)"
    echo "                      Default: all (runs all tests and benchmarks)"
    echo "  -b, --build-dir DIR Build directory to use (normal, sanitizer, or custom path)"
    echo "                      Default: normal (uses $PROJECT_BUILD_DIR)"
    echo "                      Options: normal, sanitizer, or full path"
    echo "  -j, --jobs NUM      Number of parallel jobs (default: number of CPU cores)"
    echo "  -v, --verbose       Enable verbose output"
    echo "  -h, --help          Show this help message"
    echo
    echo "Examples:"
    echo "  $0                  # Run all tests and benchmarks using normal build (default)"
    echo "  $0 -b sanitizer     # Run all tests using sanitizer build"
    echo "  $0 -b normal        # Run all tests using normal build (explicit)"
    echo "  $0 -b /path/to/build # Run all tests using custom build directory"
    echo "  $0 -t tests         # Run all tests (unit AND LIT tests)"
    echo "  $0 -t unit          # Run only unit tests"
    echo "  $0 -t lit           # Run only LIT tests"
    echo "  $0 -t benchmark     # Run all benchmarks"
    echo "  $0 -t all -j 4      # Run all tests and benchmarks with 4 parallel jobs"
    echo "  $0 -b sanitizer -t unit # Run unit tests with sanitizer build"
}

# Default values
TEST_TYPE="all"
BUILD_DIR="$PROJECT_BUILD_DIR"
NUM_JOBS=$(nproc)
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -b|--build-dir)
            BUILD_DIR_SPEC="$2"
            shift 2
            ;;
        -j|--jobs)
            NUM_JOBS="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            print_usage
            exit 1
            ;;
    esac
done

# Process build directory specification
if [ -n "$BUILD_DIR_SPEC" ]; then
    case $BUILD_DIR_SPEC in
        normal)
            BUILD_DIR="$PROJECT_BUILD_DIR"
            ;;
        sanitizer)
            BUILD_DIR="$SOURCE_DIR/sanitizer-build"
            ;;
        *)
            # Assume it's a custom path
            BUILD_DIR="$BUILD_DIR_SPEC"
            ;;
    esac
fi

# Validate test type
case $TEST_TYPE in
    all|tests|unit|lit|benchmark|basic-bench|asymptotic-bench)
        ;;
    *)
        echo "Error: Invalid test type '$TEST_TYPE'"
        print_usage
        exit 1
        ;;
esac

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "Error: Build directory not found at $BUILD_DIR"
    if [ "$BUILD_DIR" = "$PROJECT_BUILD_DIR" ]; then
        echo "Please run ./scripts/build.sh first"
    elif [ "$BUILD_DIR" = "$SOURCE_DIR/sanitizer-build" ]; then
        echo "Please run ./scripts/build-sanitizer.sh first"
    else
        echo "Please ensure the build directory exists and is properly configured"
    fi
    exit 1
fi

# Function to run tests with proper formatting
run_test_target() {
    local target=$1
    local description=$2
    
    print_section "Running $description"
    echo "Using build directory: $BUILD_DIR"
    if ! cmake --build "$BUILD_DIR" --target "$target" -j "$NUM_JOBS" $VERBOSE; then
        echo "Error: $description failed"
        exit 1
    fi
}

# Run the selected test type
case $TEST_TYPE in
    all)
        run_test_target "run_all" "all tests and benchmarks"
        ;;
    tests)
        run_test_target "run_all_tests" "all tests"
        ;;
    unit)
        run_test_target "run_unit_tests" "unit tests"
        ;;
    lit)
        run_test_target "run_lit_tests" "LIT tests"
        ;;
    benchmark)
        run_test_target "run_all_benchmarks" "all benchmarks"
        ;;
    basic-bench)
        run_test_target "run_basic_benchmarks" "basic benchmarks"
        ;;
    asymptotic-bench)
        run_test_target "run_asymptotic_benchmarks" "asymptotic benchmarks"
        ;;
esac

print_section "Test Run Complete"
echo "All selected tests completed successfully"
