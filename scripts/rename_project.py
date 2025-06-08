#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
from pathlib import Path
import re
import stat

def rename_files_and_dirs(directory, old_name, new_name):
    """Rename files and directories containing the old name."""
    renamed_count = 0
    old_name_lower = old_name.lower()
    
    # First pass: collect all paths that need to be renamed
    paths_to_rename = []
    for root, dirs, files in os.walk(directory, topdown=True):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not any(skip in d for skip in {'.git', 'build', '_deps', 'cmake-build-*'})]
        
        # Process directories first (topdown=True ensures we process parents before children)
        for name in dirs:
            # Use case-insensitive pattern matching for directory names
            if re.search(r'(?i)' + re.escape(old_name), name):
                old_path = os.path.join(root, name)
                # For directories, preserve the case of the new name
                new_name_dir = re.sub(r'(?i)' + re.escape(old_name), new_name, name, flags=re.IGNORECASE)
                new_path = os.path.join(root, new_name_dir)
                if old_path != new_path:
                    paths_to_rename.append((old_path, new_path))
        
        # Then process files
        for name in files:
            # Skip certain file types
            if any(name.endswith(ext) for ext in {'.o', '.a', '.so', '.dylib', '.dll', '.exe', '.pyc', '.pyo'}):
                continue
                
            # Use case-insensitive pattern matching for file names
            if re.search(r'(?i)' + re.escape(old_name), name):
                old_path = os.path.join(root, name)
                # For files, handle mixed case and special characters
                if '_mixed' in name.lower():
                    # For mixed-case files, use lowercase for the new name
                    new_name_file = re.sub(r'(?i)' + re.escape(old_name), new_name.lower(), name, flags=re.IGNORECASE)
                else:
                    # For regular files, preserve the case of the new name
                    new_name_file = re.sub(r'(?i)' + re.escape(old_name), new_name, name, flags=re.IGNORECASE)
                new_path = os.path.join(root, new_name_file)
                if old_path != new_path:
                    paths_to_rename.append((old_path, new_path))
    
    # Sort paths by length in descending order to ensure we process deeper paths first
    paths_to_rename.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Second pass: perform the renames
    for old_path, new_path in paths_to_rename:
        try:
            # Skip if source doesn't exist (might have been renamed already)
            if not os.path.exists(old_path):
                continue
                
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(new_path), exist_ok=True)
            
            # Preserve file permissions
            st = os.stat(old_path)
            
            # If destination exists, remove it first
            if os.path.exists(new_path):
                if os.path.isdir(new_path):
                    shutil.rmtree(new_path)
                else:
                    os.remove(new_path)
            
            # Move the file/directory
            shutil.move(old_path, new_path)
            
            # Restore permissions
            os.chmod(new_path, st.st_mode)
            
            print(f"Renaming: {old_path} -> {new_path}")
            renamed_count += 1
        except Exception as e:
            print(f"Error renaming {old_path}: {e}")
            continue
    
    return renamed_count

def replace_in_files(directory, old_name, new_name):
    """Replace occurrences of old_name with new_name in file contents."""
    modified_count = 0
    old_name_lower = old_name.lower()
    
    # Skip certain directories and file types
    skip_dirs = {'.git', 'build', '_deps', 'cmake-build-*'}
    skip_extensions = {'.o', '.a', '.so', '.dylib', '.dll', '.exe', '.pyc', '.pyo'}
    
    # Patterns to match in files
    patterns = [
        # Case-insensitive replacement of the name, preserving case of new name
        (re.compile(r'(?i)' + re.escape(old_name)), new_name),
        
        # Project name in CMakeLists.txt
        (re.compile(r'project\s*\(\s*MLIR' + re.escape(old_name) + r'\s*\)', re.IGNORECASE), 
         f'project(MLIR{new_name})'),
        
        # Class names
        (re.compile(r'class\s+' + re.escape(old_name) + r'Dialect', re.IGNORECASE), 
         f'class {new_name}Dialect'),
        (re.compile(r'class\s+' + re.escape(old_name) + r'Ops', re.IGNORECASE), 
         f'class {new_name}Ops'),
        (re.compile(r'void\s+' + re.escape(old_name) + r'Ops::initialize\(\)', re.IGNORECASE), 
         f'void {new_name}Ops::initialize()'),
        
        # Include paths
        (re.compile(r'#include\s+"' + re.escape(old_name) + r'/', re.IGNORECASE), 
         f'#include "{new_name}/'),
        (re.compile(r'#include\s+"' + re.escape(old_name) + r'Ops\.h"', re.IGNORECASE), 
         f'#include "{new_name}Ops.h"'),
        (re.compile(r'#include\s+"' + re.escape(old_name) + r'Dialect\.h"', re.IGNORECASE), 
         f'#include "{new_name}Dialect.h"'),
        
        # Special case for mixed case files
        (re.compile(r'class\s+' + old_name_lower + r'_mixed', re.IGNORECASE), 
         f'class {new_name.lower()}_mixed'),
        
        # Special case for files with special characters
        (re.compile(r'#include\s+"' + re.escape(old_name) + r'/' + re.escape(old_name) + r'Ops\.h"', re.IGNORECASE), 
         f'#include "{new_name}/{new_name}Ops.h"'),
         
        # Special case for files with special characters in the name
        (re.compile(r'#include\s+"' + re.escape(old_name) + r'/' + re.escape(old_name) + r'-' + re.escape(old_name) + r'Ops\.h"', re.IGNORECASE),
         f'#include "{new_name}/{new_name}-{new_name}Ops.h"'),
    ]
    
    for root, dirs, files in os.walk(directory):
        # Skip certain directories
        dirs[:] = [d for d in dirs if not any(skip in d for skip in skip_dirs)]
        
        for name in files:
            # Skip certain file types
            if any(name.endswith(ext) for ext in skip_extensions):
                continue
                
            file_path = os.path.join(root, name)
            try:
                # Skip binary files
                if not is_text_file(file_path):
                    continue
                    
                # Read file content
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Check if file needs modification
                original_content = content
                for pattern, replacement in patterns:
                    content = pattern.sub(replacement, content)
                
                # Write back if modified
                if content != original_content:
                    # Preserve file permissions
                    st = os.stat(file_path)
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    os.chmod(file_path, st.st_mode)
                    print(f"Modified: {file_path}")
                    modified_count += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return modified_count

def is_text_file(file_path):
    """Check if a file is a text file."""
    try:
        with open(file_path, 'rb') as f:
            chunk = f.read(1024)
            return not bool(b'\x00' in chunk)
    except:
        return False

def process_directory(directory, old_name, new_name, yes=False):
    """Process the directory to rename the project."""
    print(f"Renaming project from '{old_name}' to '{new_name}'")
    print(f"Working directory: {directory}")
    
    if not yes:
        print("\nThis will modify all files in the current directory and subdirectories.")
        response = input("Do you want to continue? [y/N] ").lower()
        if response != 'y':
            print("Operation cancelled.")
            return False
    
    print("\nRenaming files and directories...")
    renamed_count = rename_files_and_dirs(directory, old_name, new_name)
    if renamed_count == 0:
        print("No files or directories were renamed. Aborting.")
        return False
    print(f"Renamed {renamed_count} files/directories\n")
    
    print("Replacing content in files...")
    modified_count = replace_in_files(directory, old_name, new_name)
    if modified_count == 0:
        print("No file contents were modified. Aborting.")
        return False
    print(f"Modified {modified_count} files\n")
    
    print("Project renaming completed successfully!")
    return True

def main():
    parser = argparse.ArgumentParser(description='Rename MLIR project files and content.')
    parser.add_argument('old_name', help='The old project name to replace')
    parser.add_argument('new_name', help='The new project name')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically answer yes to prompts')
    args = parser.parse_args()
    
    # Get the current directory
    current_dir = os.getcwd()
    
    # Process the directory
    success = process_directory(current_dir, args.old_name, args.new_name, args.yes)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main() 