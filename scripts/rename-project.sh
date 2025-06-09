#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/rename_project.py"

# Validate arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <old_name> <new_name> <destination_path>"
    echo "Example: $0 Standalone MyDialect /path/to/new/project"
    exit 1
fi

OLD_NAME="$1"
NEW_NAME="$2"
DEST_PATH="$3"

# Check prerequisites
[ -f "$PYTHON_SCRIPT" ] || { echo "Error: rename_project.py not found"; exit 1; }
command -v python3 &> /dev/null || { echo "Error: Python 3 required"; exit 1; }
[ -e "$DEST_PATH" ] && { echo "Error: Destination '$DEST_PATH' already exists"; exit 1; }

# Create destination and copy project
echo "Creating project at: $DEST_PATH"
mkdir -p "$DEST_PATH" || { echo "Error: Failed to create destination"; exit 1; }

# Determine source directory (project root)
SRC_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"

# Copy project files
echo "Copying project files..."
cp -rp "$SRC_DIR"/* "$DEST_PATH/" || { echo "Error: Failed to copy project"; exit 1; }

# Remove .git from copy if present
[ -d "$DEST_PATH/.git" ] && rm -rf "$DEST_PATH/.git"

# Remove build directories from copy if present
[ -d "$DEST_PATH/build" ] && rm -rf "$DEST_PATH/build"
[ -d "$DEST_PATH/build-sanitizer" ] && rm -rf "$DEST_PATH/build-sanitizer"

# Run rename script
echo "Renaming project..."
cd "$DEST_PATH" && python3 "$PYTHON_SCRIPT" "$OLD_NAME" "$NEW_NAME" --yes

if [ $? -eq 0 ]; then
    echo -e "\nProject successfully copied and renamed!"
    echo "New project location: $DEST_PATH"
    echo -e "\nNext steps:"
    echo "1. cd $DEST_PATH"
    echo "2. git init (if you want to start a new repository)"
    echo "3. Update the README.md with your project's specific information"
    echo "4. Run CMake to generate new build files"
    echo "5. Build the project to verify everything works"
else
    echo "Error: Project renaming failed"
    exit 1
fi
