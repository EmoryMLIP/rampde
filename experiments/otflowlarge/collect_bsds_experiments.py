#!/usr/bin/env python3
import os
import shutil
import tempfile
import glob
from pathlib import Path

def main():
    # Pattern to search for
    pattern = "bsds"
    
    # Base directory to search in (current directory)
    base_dir = Path.cwd()
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix=f"experiments_{pattern}_")
    print(f"Created temp directory: {temp_dir}")
    
    # Find all directories containing the pattern
    found_dirs = []
    for root, dirs, files in os.walk(base_dir):
        for dir_name in dirs:
            if pattern in dir_name.lower():
                full_path = os.path.join(root, dir_name)
                # Skip the temp directory itself
                if not full_path.startswith(temp_dir):
                    found_dirs.append(full_path)
    
    print(f"Found {len(found_dirs)} directories with pattern '{pattern}':")
    for d in found_dirs:
        print(f"  - {d}")
    
    if not found_dirs:
        print("No directories found. Exiting.")
        shutil.rmtree(temp_dir)
        return
    
    # Copy directories to temp location
    for src_dir in found_dirs:
        # Create relative path structure in temp dir
        rel_path = os.path.relpath(src_dir, base_dir)
        dst_dir = os.path.join(temp_dir, rel_path)
        
        print(f"Copying {src_dir} -> {dst_dir}")
        shutil.copytree(src_dir, dst_dir)
    
    # Remove saved models from temp directory
    model_patterns = [
        "*.pth",
        "*.pt",
        "*.pkl",
        "*.ckpt",
        "checkpoint*",
        "*.h5",
        "*.hdf5",
        "*.safetensors",
        "*.npy",  # Often used for saved arrays in otflow
        "*.npz"   # Compressed numpy arrays
    ]
    
    removed_count = 0
    for pattern_glob in model_patterns:
        for model_file in glob.glob(os.path.join(temp_dir, "**", pattern_glob), recursive=True):
            print(f"Removing model file: {model_file}")
            os.remove(model_file)
            removed_count += 1
    
    print(f"Removed {removed_count} model files")
    
    # Create zip file
    zip_name = f"experiments_{pattern}"
    zip_path = os.path.join(base_dir, f"{zip_name}.zip")
    
    print(f"Creating zip file: {zip_path}")
    shutil.make_archive(zip_name, 'zip', temp_dir)
    
    # Clean up temp directory
    shutil.rmtree(temp_dir)
    print(f"Cleaned up temp directory")
    
    # Print final stats
    total_size = os.path.getsize(zip_path) / (1024 * 1024)  # Convert to MB
    print(f"\nDone! Created {zip_path}")
    print(f"Zip file size: {total_size:.2f} MB")
    print(f"Contains {len(found_dirs)} experiment directories with '{pattern}' in their names")

if __name__ == "__main__":
    main()