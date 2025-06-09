#!/usr/bin/env python3

import os
import sys
import shutil
import argparse
import re
from pathlib import Path

def is_text_file(file_path):
    """Check if a file is a text file."""
    try:
        with open(file_path, 'rb') as f:
            return not bool(b'\x00' in f.read(1024))
    except:
        return False

def should_skip_path(path):
    """Check if path should be skipped during processing."""
    skip_dirs = {'.git', 'build', 'build-sanitizer', 'build-debug', 'build-release', '_deps', 'cmake-build-*'}
    skip_extensions = {'.o', '.a', '.so', '.dylib', '.dll', '.exe', '.pyc', '.pyo'}

    return (any(skip in path for skip in skip_dirs) or 
            any(path.endswith(ext) for ext in skip_extensions))

def rename_project(directory, old_name, new_name):
    """Rename files, directories, and content in the project."""
    renamed_count = 0
    modified_count = 0
    
    # Collect all paths that need processing
    paths_to_process = []
    for root, dirs, files in os.walk(directory, topdown=True):
        # Filter directories to skip
        dirs[:] = [d for d in dirs if not should_skip_path(d)]
        
        # Add directories and files to process
        for name in dirs + files:
            path = os.path.join(root, name)
            if re.search(r'(?i)' + re.escape(old_name), name):
                paths_to_process.append(path)
    
    # Sort by path length (deepest first) to avoid conflicts
    paths_to_process.sort(key=len, reverse=True)
    
    # Rename files and directories
    for old_path in paths_to_process:
        if not os.path.exists(old_path):
            continue
            
        name = os.path.basename(old_path)
        new_name_file = re.sub(r'(?i)' + re.escape(old_name), new_name, name, flags=re.IGNORECASE)
        new_path = os.path.join(os.path.dirname(old_path), new_name_file)
        
        if old_path != new_path:
            try:
                shutil.move(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")
                renamed_count += 1
            except Exception as e:
                print(f"Error renaming {old_path}: {e}")
    
    # Replace content in files
    for root, dirs, files in os.walk(directory):
        dirs[:] = [d for d in dirs if not should_skip_path(d)]
        
        for name in files:
            if should_skip_path(name):
                continue
                
            file_path = os.path.join(root, name)
            if not is_text_file(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                original_content = content
                
                # Apply replacements
                content = re.sub(r'(?i)' + re.escape(old_name), new_name, content)
                content = re.sub(r'project\s*\(\s*MLIR' + re.escape(old_name) + r'\s*\)', 
                               f'project(MLIR{new_name})', content, flags=re.IGNORECASE)
                content = re.sub(r'class\s+' + re.escape(old_name) + r'Dialect', 
                               f'class {new_name}Dialect', content, flags=re.IGNORECASE)
                content = re.sub(r'class\s+' + re.escape(old_name) + r'Ops', 
                               f'class {new_name}Ops', content, flags=re.IGNORECASE)
                content = re.sub(r'#include\s+"' + re.escape(old_name) + r'/', 
                               f'#include "{new_name}/', content, flags=re.IGNORECASE)
                
                if content != original_content:
                    with open(file_path, 'w', encoding='utf-8') as f:
                        f.write(content)
                    print(f"Modified: {file_path}")
                    modified_count += 1
                    
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    return renamed_count, modified_count

def main():
    parser = argparse.ArgumentParser(description='Rename MLIR project files and content.')
    parser.add_argument('old_name', help='The old project name to replace')
    parser.add_argument('new_name', help='The new project name')
    parser.add_argument('--yes', '-y', action='store_true', help='Automatically answer yes to prompts')
    args = parser.parse_args()
    
    current_dir = os.getcwd()
    
    if not args.yes:
        print(f"Renaming project from '{args.old_name}' to '{args.new_name}'")
        response = input("Continue? [y/N] ").lower()
        if response != 'y':
            print("Operation cancelled.")
            sys.exit(1)
    
    print(f"Renaming project from '{args.old_name}' to '{args.new_name}'...")
    
    renamed_count, modified_count = rename_project(current_dir, args.old_name, args.new_name)
    
    if renamed_count == 0 and modified_count == 0:
        print("No changes made. Aborting.")
        sys.exit(1)
    
    print(f"\n✅ Project renamed successfully!")
    print(f"📁 Renamed {renamed_count} files/directories")
    print(f"📝 Modified {modified_count} files")
    
    sys.exit(0)

if __name__ == '__main__':
    main()
