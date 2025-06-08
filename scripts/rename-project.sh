#!/bin/bash

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PYTHON_SCRIPT="$SCRIPT_DIR/rename_project.py"

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Could not find rename_project.py in $SCRIPT_DIR"
    exit 1
fi

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is required but not found"
    exit 1
fi

# Check if script is executable
if [ ! -x "$PYTHON_SCRIPT" ]; then
    echo "Making script executable..."
    chmod +x "$PYTHON_SCRIPT"
fi

# Check if we have the correct number of arguments
if [ $# -ne 3 ]; then
    echo "Usage: $0 <old_name> <new_name> <destination_path>"
    echo "Example: $0 Standalone MyDialect /path/to/new/project"
    exit 1
fi

OLD_NAME="$1"
NEW_NAME="$2"
DEST_PATH="$3"

# Check if destination path exists
if [ -e "$DEST_PATH" ]; then
    echo "Error: Destination path '$DEST_PATH' already exists"
    echo "Please choose a different destination path or remove the existing one"
    exit 1
fi

# Create destination directory
echo "Creating destination directory: $DEST_PATH"
if ! mkdir -p "$DEST_PATH"; then
    echo "Error: Failed to create destination directory"
    exit 1
fi

# Determine source directory
# If we're in a test environment (temp directory), use the current directory
# Otherwise, use the project root
if [[ "$(pwd)" == /tmp/rename_project_test_* ]]; then
    SRC_DIR="$(pwd)"
else
    SRC_DIR="$( cd "$SCRIPT_DIR/.." &> /dev/null && pwd )"
fi

# Copy the project to the new location
echo "Copying project to: $DEST_PATH"
if ! cp -rp "$SRC_DIR"/* "$DEST_PATH/"; then
    echo "Error: Failed to copy project to destination"
    exit 1
fi

# Remove the .git directory from the copy if it exists
if [ -d "$DEST_PATH/.git" ]; then
    echo "Removing .git directory from the copy"
    rm -rf "$DEST_PATH/.git"
fi

# Change to the new directory and run the Python script
echo "Changing to new project directory"
cd "$DEST_PATH" || {
    echo "Error: Failed to change to new project directory"
    exit 1
}

# Run the Python script in the new location with --yes flag
echo "Running rename script in new location..."
python3 "$PYTHON_SCRIPT" "$OLD_NAME" "$NEW_NAME" --yes

# Check if the Python script was successful
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