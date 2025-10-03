#!/usr/bin/env python3
"""
Master script to process all experiment results for the Mixed Precision Neural ODE paper.

This script orchestrates the processing of all experiments by calling their respective
processing scripts. Each experiment handles its own figure and table generation.

Experiments processed:
- CNF: Figure 2 (CNF overview) + Table 2 (CNF results)
- STL10: Figure 4 (training loss) + Table (summary statistics)
- OTFlowLarge: Table 3 (OTFlow results)
- Roundoff: Figure 3 (roundoff error plots)

Usage:
    python process_all_results.py [options]

Options:
    --experiments EXPS    Comma-separated list of experiments to process
                          (cnf,stl10,otflowlarge,roundoff). Default: all
    --skip-tables         Skip table generation for applicable experiments

Environment:
    This script assumes the correct Python environment is already activated.

Outputs:
    Each experiment saves outputs to its own outputs/ directory:
    - paper/cnf/outputs/
    - paper/stl10/outputs/
    - paper/otflowlarge/outputs/
    - paper/roundoff/outputs/
"""

import subprocess
import sys
import os
from pathlib import Path
import argparse
from typing import List, Optional, Dict


def run_experiment_script(experiment: str, script_path: Path, extra_args: List[str] = None) -> bool:
    """Run an experiment's processing script."""
    print("\n" + "="*70)
    print(f"📊 Processing {experiment.upper()} Experiment")
    print("="*70)

    cmd = [str(script_path)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"Running: {' '.join(cmd)}")
    print(f"Working directory: {script_path.parent}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=script_path.parent,
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        print(f"\n✅ {experiment.upper()} processing completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {experiment.upper()} processing failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Processing script not found: {script_path}")
        return False
    except Exception as e:
        print(f"\n❌ {experiment.upper()} processing failed: {e}")
        return False


def check_prerequisites() -> bool:
    """Check that we're in the right directory."""
    if not Path("cnf").exists() or not Path("stl10").exists():
        print("❌ Error: Must run from the paper/ directory")
        print("   Current directory:", Path.cwd())
        return False
    return True


def print_summary(results: Dict[str, bool]):
    """Print a summary of processing results."""
    print("\n" + "="*70)
    print("📋 PROCESSING SUMMARY")
    print("="*70)

    total = len(results)
    successful = sum(results.values())
    failed = total - successful

    print(f"\nTotal experiments processed: {total}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}\n")

    for experiment, success in results.items():
        status = "✅" if success else "❌"
        print(f"{status} {experiment.upper()}")

    if failed == 0:
        print("\n🎉 All experiments processed successfully!")
        print("\nOutputs are available in each experiment's outputs/ directory:")
        print("  - paper/cnf/outputs/")
        print("  - paper/stl10/outputs/")
        print("  - paper/otflowlarge/outputs/")
        print("  - paper/roundoff/outputs/")
    else:
        print(f"\n⚠️  {failed} experiment(s) failed. Check the output above for details.")


def main():
    parser = argparse.ArgumentParser(
        description='Process all experiment results for the paper',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--experiments',
        type=str,
        default='cnf,stl10,otflowlarge,roundoff',
        help='Comma-separated list of experiments to process (default: all)'
    )
    parser.add_argument(
        '--skip-tables',
        action='store_true',
        help='Skip table generation for applicable experiments'
    )
    args = parser.parse_args()

    # Parse experiment list
    experiments_to_run = [e.strip() for e in args.experiments.split(',')]

    print("🚀 Mixed Precision Neural ODE - Results Processing")
    print("="*70)
    print(f"Experiments to process: {', '.join(experiments_to_run)}")
    if args.skip_tables:
        print("Tables: SKIPPED")
    print()

    # Check prerequisites
    if not check_prerequisites():
        return 1

    # Track results
    results = {}

    # Process each experiment
    for experiment in experiments_to_run:
        script_path = Path(f"{experiment}/process_results.sh")

        if not script_path.exists():
            print(f"\n⚠️  Skipping {experiment}: processing script not found at {script_path}")
            results[experiment] = False
            continue

        # Build extra arguments
        extra_args = []
        if args.skip_tables and experiment in ['stl10']:
            extra_args.append('--skip-table')
        if args.skip_tables and experiment == 'roundoff':
            extra_args.append('--combined-only')

        # Run the experiment's processing script
        results[experiment] = run_experiment_script(experiment, script_path, extra_args)

    # Print summary
    print_summary(results)

    # Return exit code
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
