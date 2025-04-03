import os
import shutil
import argparse
import hashlib
from pathlib import Path

def calculate_file_hash(file_path: Path, chunk_size: int = 8192) -> str:
    """Calculate MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def move_files_to_root(root_dir, force=False, keep_structure=False):
    # Convert path to Path object
    root = Path(root_dir).resolve()
    
    # Validate directory exists
    if not root.is_dir():
        print(f"Error: Directory '{root}' does not exist")
        return
    
    # Ask for confirmation if not in force mode
    if not force:
        print(f"\nWill move all files from subdirectories under {root} to root directory.")
        confirm = input("Continue? (y/n): ").strip().lower()
        if confirm != 'y':
            print("Operation cancelled")
            return

    # Get all files and directories
    all_files = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip root directory
        if dirpath == str(root):
            continue
        
        # Collect full paths of all files
        for file in filenames:
            src = Path(dirpath) / file
            all_files.append(src)
    
    # Initialize counters
    moved_count = 0
    failed_count = 0
    same_hash_count = 0
    different_hash_count = 0
    
    for src in all_files:
        try:
            if keep_structure:
                rel_path = src.relative_to(root)
                path_prefix = str(rel_path.parent).replace(os.sep, '_')
                new_name = f"{path_prefix}_{src.name}" if path_prefix != '.' else src.name
                dst = root / new_name
            else:
                dst = root / src.name
            
            # Handle filename conflicts
            if dst.exists():
                print(f"Identical files found: {src.name} - removing duplicate")
                os.remove(src)
                same_hash_count += 1
                continue
            
            # Move file
            shutil.move(str(src), str(dst))
            print(f"Moved: {src.name} -> {dst.name}")
            moved_count += 1
            
        except Exception as e:
            print(f"Failed to move {src}: {str(e)}")
            failed_count += 1
    
    # Remove empty directories
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        if dirpath == str(root):
            continue
        try:
            os.rmdir(dirpath)
            print(f"Removed empty directory: {dirpath}")
        except OSError:
            # Skip if directory is not empty
            pass
    
    # Print statistics
    print("\nOperation Complete!")
    print("Statistics:")
    print(f"Successfully moved: {moved_count} files")
    print(f"Failed: {failed_count} files")
    print(f"Identical files skipped: {same_hash_count} files")
    print(f"Name collisions resolved: {different_hash_count} files")

def main():
    parser = argparse.ArgumentParser(
        description='Recursively move all files from subdirectories to root directory.',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        'directory', 
        help='Directory path to process'
    )
    
    parser.add_argument(
        '-f', '--force',
        action='store_true',
        help='Execute without confirmation'
    )
    
    parser.add_argument(
        '--keep-structure',
        action='store_true',
        help='Preserve folder structure information in filenames'
    )

    args = parser.parse_args()
    move_files_to_root(args.directory, args.force, args.keep_structure)

if __name__ == "__main__":
    main()
