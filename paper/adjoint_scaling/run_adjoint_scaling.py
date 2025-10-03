#!/usr/bin/env python3
"""
Wrapper script for adjoint scaling experiment (paper results).

This script runs the adjoint scaling test from tests/core/test_adjoint_scaling.py
and saves outputs to paper/outputs/adjoint_scaling/ for inclusion in the paper.
"""
import sys
import os
import pathlib
import shutil
import unittest

# Add project root to path
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import the test module
from tests.core import test_adjoint_scaling

# Set up paper output directory (same as script directory)
PAPER_OUTPUT_DIR = SCRIPT_DIR
PAPER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    """Run the adjoint scaling test and copy results to paper output directory."""

    print(f"Running adjoint scaling experiment...")
    print(f"Test output will be saved to: {test_adjoint_scaling.OUT_DIR}")
    print(f"Paper output will be copied to: {PAPER_OUTPUT_DIR}")
    print()

    # Run the test
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(test_adjoint_scaling)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Copy results to paper output directory
    if result.wasSuccessful():
        print(f"\n{'='*60}")
        print("Test completed successfully. Copying results to paper directory...")

        # Copy all output files
        for item in test_adjoint_scaling.OUT_DIR.iterdir():
            dest = PAPER_OUTPUT_DIR / item.name
            if item.is_file():
                shutil.copy2(item, dest)
                print(f"  Copied: {item.name}")

        print(f"\nPaper outputs saved to: {PAPER_OUTPUT_DIR}")
        print(f"{'='*60}")
        return 0
    else:
        print(f"\n{'='*60}")
        print("Test failed. Results may be incomplete.")
        print(f"{'='*60}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
