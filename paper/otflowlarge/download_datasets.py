#!/usr/bin/env python3
"""
Download and extract BSDS300 dataset from Zenodo for OTFlow experiments.

From Papamakarios et al.'s MAF paper:
https://zenodo.org/record/1161203
"""

import sys
import urllib.request
import tarfile
from pathlib import Path

# Zenodo download URL for all datasets (tar.gz archive)
ZENODO_URL = "https://zenodo.org/records/1161203/files/data.tar.gz?download=1"

def download_file(url, destination):
    """Download a file with progress indication."""
    print(f"Downloading {url}...")

    def progress_hook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}%")
        sys.stdout.flush()

    urllib.request.urlretrieve(url, destination, progress_hook)
    print()  # New line after progress

def extract_bsds300_from_tar(tar_path, data_dir):
    """Extract only BSDS300 from the tar.gz archive."""
    print(f"Extracting BSDS300 from {tar_path}...")
    with tarfile.open(tar_path, 'r:gz') as tar:
        # Find and extract only BSDS300 files
        members_to_extract = [m for m in tar.getmembers() if 'BSDS300' in m.name]
        if not members_to_extract:
            print("  Warning: No BSDS300 files found in archive")
            return False

        print(f"  Extracting {len(members_to_extract)} BSDS300 files...")
        for member in members_to_extract:
            # Strip leading 'data/' from archive paths to avoid nested directories
            if member.name.startswith('data/'):
                member.name = member.name[5:]  # Remove 'data/' prefix
            tar.extract(member, data_dir)

    print(f"  Extracted to {data_dir}")
    return True

def download_bsds300(data_dir):
    """Download and extract BSDS300 dataset."""
    tar_path = data_dir / "data.tar.gz"

    # Download
    try:
        download_file(ZENODO_URL, tar_path)
    except Exception as e:
        print(f"Error downloading BSDS300: {e}")
        return False

    # Extract only BSDS300
    try:
        if not extract_bsds300_from_tar(tar_path, data_dir):
            return False
    except Exception as e:
        print(f"Error extracting BSDS300: {e}")
        return False
    finally:
        # Clean up tar file
        if tar_path.exists():
            tar_path.unlink()
            print("  Cleaned up archive file")

    return True

def check_bsds300_exists(data_dir):
    """Check if BSDS300 dataset is already downloaded."""
    return (data_dir / 'BSDS300' / 'BSDS300.hdf5').exists()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Download BSDS300 dataset')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    # Get the data directory relative to this script
    script_dir = Path(__file__).parent
    data_dir = script_dir / 'data'
    data_dir.mkdir(exist_ok=True)

    print("BSDS300 Dataset Downloader")
    print("=" * 50)
    print(f"Data directory: {data_dir.absolute()}")
    print()

    # Check if BSDS300 is already downloaded
    if check_bsds300_exists(data_dir):
        print("✓ BSDS300 is already downloaded")
        return 0

    print("✗ BSDS300 dataset not found")
    print()

    # Ask user if they want to download (unless --yes flag is set)
    if not args.yes:
        response = input("Download datasets archive (~857MB, only BSDS300 will be extracted)? [y/N]: ").strip().lower()
        if response not in ['y', 'yes']:
            print("Download cancelled.")
            return 1

    print()

    # Download BSDS300
    if download_bsds300(data_dir):
        print("✓ BSDS300 downloaded successfully")
        return 0
    else:
        print("✗ Failed to download BSDS300")
        return 1

if __name__ == '__main__':
    sys.exit(main())
