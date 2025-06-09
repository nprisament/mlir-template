#!/bin/bash

# Exit on error
set -e

# Find clang-format
if ! command -v clang-format &> /dev/null; then
    echo "Error: clang-format not found. Please install it first."
    exit 1
fi

# Find all C++ source files
echo "Finding C++ source files..."
FILES=$(find . -type f \( -name "*.cpp" -o -name "*.h" -o -name "*.hpp" -o -name "*.cc" -o -name "*.c" \) \
    -not -path "./build/*" \
    -not -path "./.git/*")

# Count files
NUM_FILES=$(echo "$FILES" | wc -l)
echo "Found $NUM_FILES files to format"

# Format each file
echo "Formatting files..."
for file in $FILES; do
    echo "Formatting $file"
    clang-format -i "$file"
done

echo "Formatting complete!" 
