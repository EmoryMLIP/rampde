#!/usr/bin/env python3
"""
Batch evaluation script for all OTFlowLarge experiment results.
Automatically parses experiment parameters and runs precision-aware evaluation.
"""

import os
import sys
import argparse
import pandas as pd
import json
import subprocess
import glob
import time
from pathlib import Path

def create_parser():
    """Create argument parser for batch evaluation."""
    parser = argparse.ArgumentParser(description='Batch evaluate all OTFlowLarge experiment results')
    
    parser.add_argument('--results_dir', type=str, 
                        default='./results/otflowlarge',
                        help='Path to otflowlarge results directory')
    parser.add_argument('--eval_script', type=str,
                        default='./evaluate_otflowlarge.py',
                        help='Path to evaluation script')
    parser.add_argument('--batch_size', type=int, default=5000,
                        help='Evaluation batch size')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device to use')
    parser.add_argument('--force', action='store_true',
                        help='Re-evaluate even if outputs exist')
    parser.add_argument('--filter', type=str, default=None,
                        help='Filter experiments by pattern (e.g., "bsds300", "float16")')
    parser.add_argument('--dry_run', action='store_true',
                        help='Show what would be evaluated without running')
    parser.add_argument('--parallel', type=int, default=1,
                        help='Number of parallel evaluations (default: 1)')
    parser.add_argument('--nt', type=int, default=18,
                        help='Number of integration time steps for evaluation')
    
    return parser

def find_experiment_directories(results_dir):
    """Find all experiment directories in the results directory."""
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    experiment_dirs = []
    for item in results_path.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            experiment_dirs.append(item)
    
    return sorted(experiment_dirs)

def parse_experiment_parameters(exp_dir):
    """Parse experiment parameters from args.csv file."""
    args_file = exp_dir / 'args.csv'
    
    if not args_file.exists():
        print(f"Warning: No args.csv found in {exp_dir}")
        return None
    
    try:
        # Read the CSV file
        df = pd.read_csv(args_file)
        if len(df) == 0:
            print(f"Warning: Empty args.csv in {exp_dir}")
            return None
        
        # Convert first row to dictionary
        params = df.iloc[0].to_dict()
        
        # Parse alpha if it's a string
        if 'alpha' in params and isinstance(params['alpha'], str):
            # Remove brackets and quotes, split by comma
            alpha_str = params['alpha'].strip('[]"\'')
            params['alpha'] = alpha_str
        
        return params
    
    except Exception as e:
        print(f"Error parsing args.csv in {exp_dir}: {e}")
        return None

def find_best_checkpoint(exp_dir):
    """Find the best checkpoint file in the experiment directory."""
    # Priority order: ckpt.pth (best validation), ckpt_final.pth (final state)
    checkpoint_files = ['ckpt.pth', 'ckpt_final.pth']
    
    for ckpt_file in checkpoint_files:
        ckpt_path = exp_dir / ckpt_file
        if ckpt_path.exists():
            return str(ckpt_path)
    
    # Look for any .pth file
    pth_files = list(exp_dir.glob('*.pth'))
    if pth_files:
        # Filter out emergency/gradient_nan checkpoints if better ones exist
        good_checkpoints = [f for f in pth_files 
                           if 'emergency' not in f.name and 'gradient_nan' not in f.name]
        if good_checkpoints:
            return str(good_checkpoints[0])
        else:
            return str(pth_files[0])
    
    return None

def check_evaluation_completed(exp_dir):
    """Check if evaluation has already been completed."""
    eval_dir = exp_dir / 'evaluation'
    if not eval_dir.exists():
        return False
    
    # Check for evaluation results file
    results_file = eval_dir / 'evaluation_results.json'
    return results_file.exists()

def construct_evaluation_command(exp_dir, params, checkpoint_path, args):
    """Construct the evaluation command with proper parameters."""
    cmd = [
        'python', args.eval_script,
        '--data', params['data'],
        '--resume', checkpoint_path,
        '--precision', params['precision'],
        '--odeint', params['odeint'],
        '--method', params['method'],
        '--batch_size', str(args.batch_size),
        '--gpu', str(args.gpu),
        '--nt', str(args.nt),
        '--output_dir', str(exp_dir),
    ]
    
    # Add alpha if available
    if 'alpha' in params and params['alpha']:
        cmd.extend(['--alpha', str(params['alpha'])])
    
    # Add scaler flags based on experiment parameters
    if params.get('grad_scaler', False):
        cmd.append('--grad_scaler')
    
    if params.get('dynamic_scaler', False):
        cmd.append('--dynamic_scaler')
    
    # Add seed if available
    if 'seed' in params and pd.notna(params['seed']):
        cmd.extend(['--seed', str(int(params['seed']))])
    
    return cmd

def run_evaluation(cmd, exp_dir, dry_run=False):
    """Run the evaluation command."""
    if dry_run:
        print(f"Would run: {' '.join(cmd)}")
        return True, "Dry run - not executed"
    
    print(f"Running evaluation for: {exp_dir.name}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Set environment variables to fix MKL threading issues
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        env['MKL_SERVICE_FORCE_INTEL'] = '1'
        
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)  # 1 hour timeout
        end_time = time.time()
        
        if result.returncode == 0:
            print(f"✓ Evaluation completed successfully in {end_time - start_time:.1f}s")
            return True, "Success"
        else:
            error_msg = result.stderr
            print(f"✗ Evaluation failed with return code {result.returncode}")
            
            # Check for specific MKL errors
            if "MKL_THREADING_LAYER" in error_msg or "libgomp" in error_msg:
                print(f"  MKL threading error detected - environment fix may need adjustment")
                return False, f"MKL threading error: {error_msg}"
            else:
                print(f"STDERR: {error_msg}")
                return False, f"Failed with code {result.returncode}: {error_msg}"
    
    except subprocess.TimeoutExpired:
        print(f"✗ Evaluation timed out after 1 hour")
        return False, "Timeout after 1 hour"
    
    except Exception as e:
        print(f"✗ Evaluation failed with exception: {e}")
        return False, f"Exception: {e}"

def filter_experiments(experiment_dirs, filter_pattern):
    """Filter experiments based on pattern matching."""
    if not filter_pattern:
        return experiment_dirs
    
    filtered = []
    for exp_dir in experiment_dirs:
        if filter_pattern.lower() in exp_dir.name.lower():
            filtered.append(exp_dir)
    
    return filtered

def main():
    parser = create_parser()
    args = parser.parse_args()
    
    print("OTFlowLarge Batch Evaluation")
    print(f"Results directory: {args.results_dir}")
    print(f"Evaluation script: {args.eval_script}")
    print(f"Filter pattern: {args.filter}")
    print(f"Dry run: {args.dry_run}")
    print(f"Force re-evaluation: {args.force}")
    print()
    
    # Check if evaluation script exists
    if not os.path.exists(args.eval_script):
        print(f"Error: Evaluation script not found: {args.eval_script}")
        return 1
    
    # Find all experiment directories
    try:
        experiment_dirs = find_experiment_directories(args.results_dir)
        print(f"Found {len(experiment_dirs)} experiment directories")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    # Apply filter if specified
    if args.filter:
        experiment_dirs = filter_experiments(experiment_dirs, args.filter)
        print(f"After filtering by '{args.filter}': {len(experiment_dirs)} experiments")
    
    if len(experiment_dirs) == 0:
        print("No experiments found to evaluate")
        return 0
    
    # Process each experiment
    results_summary = []
    skipped_count = 0
    success_count = 0
    failed_count = 0
    
    for i, exp_dir in enumerate(experiment_dirs, 1):
        print(f"\n[{i}/{len(experiment_dirs)}] Processing: {exp_dir.name}")
        
        # Parse experiment parameters
        params = parse_experiment_parameters(exp_dir)
        if params is None:
            print(f"Skipping {exp_dir.name}: Could not parse parameters")
            failed_count += 1
            continue
        
        # Find checkpoint
        checkpoint_path = find_best_checkpoint(exp_dir)
        if checkpoint_path is None:
            print(f"Skipping {exp_dir.name}: No checkpoint found")
            failed_count += 1
            continue
        
        print(f"  Parameters: data={params.get('data')}, precision={params.get('precision')}, "
              f"odeint={params.get('odeint')}, method={params.get('method')}")
        print(f"  Checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Check if already evaluated
        if not args.force and check_evaluation_completed(exp_dir):
            print(f"  Skipping: Evaluation already completed (use --force to re-evaluate)")
            skipped_count += 1
            continue
        
        # Construct and run evaluation command
        cmd = construct_evaluation_command(exp_dir, params, checkpoint_path, args)
        success, message = run_evaluation(cmd, exp_dir, args.dry_run)
        
        # Record result
        result = {
            'experiment': exp_dir.name,
            'success': success,
            'message': message,
            'checkpoint': os.path.basename(checkpoint_path),
            'data': params.get('data'),
            'precision': params.get('precision'),
            'odeint': params.get('odeint'),
            'method': params.get('method'),
        }
        results_summary.append(result)
        
        if success:
            success_count += 1
        else:
            failed_count += 1
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    print(f"Total experiments: {len(experiment_dirs)}")
    print(f"Successful evaluations: {success_count}")
    print(f"Failed evaluations: {failed_count}")
    print(f"Skipped (already completed): {skipped_count}")
    
    if failed_count > 0:
        print(f"\nFailed experiments:")
        for result in results_summary:
            if not result['success']:
                print(f"  {result['experiment']}: {result['message']}")
    
    # Save summary to file
    if not args.dry_run:
        summary_file = os.path.join(args.results_dir, 'evaluation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump({
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'args': vars(args),
                'summary': {
                    'total': len(experiment_dirs),
                    'success': success_count,
                    'failed': failed_count,
                    'skipped': skipped_count,
                },
                'results': results_summary
            }, f, indent=2)
        print(f"\nDetailed summary saved to: {summary_file}")
    
    return 0 if failed_count == 0 else 1

if __name__ == '__main__':
    sys.exit(main())