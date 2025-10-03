import os
import sys
from pathlib import Path

root = 'data/'

# Check if data directory exists and offer to download datasets if missing
_data_dir = Path(__file__).parent.parent / root
if not _data_dir.exists() or not any(_data_dir.iterdir()):
    print(f"\nWarning: Data directory '{_data_dir}' is missing or empty.")
    print("Run the following to download datasets:")
    print(f"  python {Path(__file__).parent.parent / 'download_datasets.py'}")
    print()

from .bsds300 import BSDS300

