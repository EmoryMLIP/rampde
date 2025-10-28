#!/usr/bin/env python3
"""
Evaluate trained STL10 model checkpoints on the test set.

This script loads checkpoints from experiment directories and computes
final test accuracy and loss, saving results to test_loss.txt files.
"""

import os
import sys
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch.amp import autocast

# Add parent directory to path for common imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment_runtime import setup_environment, get_precision_dtype

# Import from ode_stl10
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ode_stl10 import MPNODE_STL10, get_stl10_loaders, accuracy


def evaluate_checkpoint(ckpt_path, device='cuda:0'):
    """Load a checkpoint and evaluate on test set."""

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found: {ckpt_path}")
        return None, None

    try:
        # Load checkpoint (weights_only=False needed for argparse.Namespace in checkpoint)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        args = checkpoint['args']

        # Get base directory
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))

        # Setup environment
        odeint_func, DynamicScaler = setup_environment(args.odeint, base_dir)

        # Get precision settings
        precision = get_precision_dtype(args.precision)

        # Determine scaler settings from args
        grad_scaler_enabled = not args.no_grad_scaler
        dynamic_scaler_enabled = not args.no_dynamic_scaler

        # Create model
        model = MPNODE_STL10(
            args.width, args, precision, odeint_func, DynamicScaler,
            dynamic_scaler_enabled, grad_scaler_enabled
        ).to(device)

        # Load state dict
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        # Get test loader
        _, test_loader, _ = get_stl10_loaders(
            batch_size=args.batch_size,
            test_batch_size=args.test_batch_size,
            seed=args.seed
        )

        # Evaluate
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=precision):
                test_acc, test_loss = accuracy(model, test_loader, device)

        return test_acc, test_loss

    except Exception as e:
        print(f"Error evaluating {ckpt_path}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def process_experiment_directory(exp_dir, device='cuda:0'):
    """Process a single experiment directory."""
    exp_path = Path(exp_dir)

    if not exp_path.is_dir():
        print(f"Not a directory: {exp_dir}")
        return False

    # Check if test_loss.txt already exists
    test_loss_file = exp_path / 'test_loss.txt'
    ckpt_file = exp_path / 'ckpt.pth'

    if test_loss_file.exists():
        print(f"✓ Test metrics already exist: {exp_path.name}")
        return True

    if not ckpt_file.exists():
        print(f"✗ No checkpoint found: {exp_path.name}")
        return False

    print(f"Evaluating: {exp_path.name}")

    # Evaluate checkpoint
    test_acc, test_loss = evaluate_checkpoint(str(ckpt_file), device=device)

    if test_acc is None or test_loss is None:
        print(f"✗ Evaluation failed: {exp_path.name}")
        return False

    # Save results
    with open(test_loss_file, 'w') as f:
        f.write(f"Test Loss: {test_loss:.6f}\n")
        f.write(f"Test Accuracy: {test_acc:.6f}\n")

    print(f"✓ Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
    return True


def main():
    parser = argparse.ArgumentParser(description='Evaluate STL10 checkpoints')
    parser.add_argument('--results-dir', default='./raw_data/ode_stl10',
                       help='Directory containing experiment subdirectories')
    parser.add_argument('--device', default='cuda:0',
                       help='Device to use for evaluation')
    parser.add_argument('--exp-dir', default=None,
                       help='Single experiment directory to evaluate (optional)')
    args = parser.parse_args()

    if args.exp_dir:
        # Evaluate single experiment
        success = process_experiment_directory(args.exp_dir, args.device)
        return 0 if success else 1

    # Process all experiments in results directory
    results_path = Path(args.results_dir)

    if not results_path.exists():
        print(f"Error: Results directory not found: {results_path}")
        return 1

    # Find all experiment directories (those with ckpt.pth files)
    exp_dirs = []
    for item in sorted(results_path.iterdir()):
        if item.is_dir() and (item / 'ckpt.pth').exists():
            exp_dirs.append(item)

    if not exp_dirs:
        print(f"No experiment directories with checkpoints found in {results_path}")
        return 1

    print(f"Found {len(exp_dirs)} experiment directories with checkpoints\n")

    # Process each experiment
    successes = 0
    failures = 0
    skipped = 0

    for exp_dir in exp_dirs:
        if (exp_dir / 'test_loss.txt').exists():
            skipped += 1
            continue

        success = process_experiment_directory(exp_dir, args.device)
        if success:
            successes += 1
        else:
            failures += 1
        print()  # Blank line between experiments

    # Summary
    print("="*60)
    print(f"Evaluation complete:")
    print(f"  Successful: {successes}")
    print(f"  Failed: {failures}")
    print(f"  Skipped (already evaluated): {skipped}")
    print(f"  Total: {len(exp_dirs)}")

    return 0 if failures == 0 else 1


if __name__ == '__main__':
    exit(main())
