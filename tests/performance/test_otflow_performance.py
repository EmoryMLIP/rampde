#!/usr/bin/env python3
"""
OTFlow performance test for torchmpnode three-variant architecture.

This test specifically focuses on complex ODE models like OTFlow to ensure
performance improvements are maintained for realistic use cases.

Based on the original test_otflowlarge_performance.py from the ablation study.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path

# Add paths for imports
current_dir = Path(__file__).parent
base_dir = current_dir.parent.parent
sys.path.insert(0, str(base_dir))

from torchmpnode import odeint
from torchmpnode.odeint import _select_ode_solver
from torchmpnode.loss_scalers import DynamicScaler

from utils.test_models import create_otflow_model, create_test_data, get_model_info
from utils.timing_utils import run_timing_test, verify_solver_selection
from utils.comparison_utils import compare_against_torchdiffeq, format_comparison_results


def load_baselines():
    """Load OTFlow performance baselines."""
    baseline_file = current_dir / "baselines" / "otflow_baseline.json"
    
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No baseline file found at {baseline_file}")
        return {}


def save_baselines(results):
    """Save OTFlow performance baselines."""
    baseline_file = current_dir / "baselines" / "otflow_baseline.json"
    
    # Create baselines directory if it doesn't exist
    baseline_file.parent.mkdir(exist_ok=True)
    
    baseline_data = {}
    for result in results:
        if result['Success']:
            baseline_data[result['Configuration']] = {
                'mean_time': result['Mean Time (s)'],
                'std_time': result['Std Time (s)'],
                'solver': result['Solver']
            }
    
    with open(baseline_file, 'w') as f:
        json.dump(baseline_data, f, indent=2)


def test_otflow_performance():
    """Test OTFlow performance across all configurations."""
    print("Testing OTFlow performance...")
    
    device = torch.device('cuda:0')
    
    # Create OTFlow model (smaller for testing)
    d = 256  # data dimension
    m = 128  # hidden dimension
    nt = 8   # number of timesteps
    
    model = create_otflow_model(d=d, m=m, nt=nt)
    x, t = create_test_data('otflow', device)
    
    print(f"Model: d={d}, m={m}, nt={nt}")
    print(f"Batch size: {x.shape[0]}")
    print(f"Device: {device}")
    print()
    
    # Test configurations
    configs = [
        ("float32", torch.float32, None),
        ("bfloat16", torch.bfloat16, None),
        ("float16", torch.float16, None),
        ("float16_Dynamic", torch.float16, DynamicScaler(torch.float16)),
    ]
    
    results = []
    
    for name, precision, scaler in configs:
        print(f"Testing {name}...")
        
        # Verify solver selection
        expected_solver, actual_solver, matches = verify_solver_selection(scaler, precision)
        
        if matches:
            print(f"  ✓ Solver selection: {actual_solver}")
        else:
            print(f"  ✗ Solver selection: Expected {expected_solver}, got {actual_solver}")
        
        # Run performance test
        mean_time, std_time, success = run_timing_test(
            model, x, t, precision, scaler, method='rk4', num_runs=10, warmup=3
        )
        
        if success:
            print(f"  ✓ Time: {mean_time:.4f} ± {std_time:.4f} s")
            
            results.append({
                'Configuration': name,
                'Precision': str(precision).split('.')[-1],
                'Scaler': type(scaler).__name__ if scaler else 'None',
                'Solver': actual_solver.replace('FixedGridODESolver', ''),
                'Mean Time (s)': mean_time,
                'Std Time (s)': std_time,
                'Success': True
            })
        else:
            print(f"  ✗ Failed to run")
            results.append({
                'Configuration': name,
                'Precision': str(precision).split('.')[-1],
                'Scaler': type(scaler).__name__ if scaler else 'None',
                'Solver': expected_solver.replace('FixedGridODESolver', ''),
                'Mean Time (s)': None,
                'Std Time (s)': None,
                'Success': False
            })
        
        print()
    
    return results


def test_otflow_vs_torchdiffeq():
    """Test OTFlow performance against torchdiffeq."""
    print("Testing OTFlow vs torchdiffeq...")
    
    device = torch.device('cuda:0')
    model = create_otflow_model(d=128, m=64, nt=8)  # Smaller for comparison
    x, t = create_test_data('otflow', device)
    
    # Select key configurations for comparison
    configs = [
        ("torchmpnode_unscaled", torch.float32, None),
        ("torchmpnode_bfloat16", torch.bfloat16, None),
        ("torchmpnode_float16", torch.float16, None),
    ]
    
    comparison_results = compare_against_torchdiffeq(model, x, t, configs)
    
    print("Comparison Results:")
    print(format_comparison_results(comparison_results))
    
    return comparison_results


def check_otflow_regression(results, baselines, threshold=0.1):
    """Check for OTFlow performance regression."""
    print("Checking for performance regression...")
    
    regressions = []
    
    for result in results:
        if not result['Success']:
            continue
            
        config = result['Configuration']
        current_time = result['Mean Time (s)']
        
        if config in baselines:
            baseline_time = baselines[config]['mean_time']
            regression = (current_time - baseline_time) / baseline_time
            
            if regression > threshold:
                regressions.append({
                    'config': config,
                    'baseline': baseline_time,
                    'current': current_time,
                    'regression': regression
                })
                print(f"  ⚠️  {config}: {regression:.1%} slower than baseline")
            else:
                print(f"  ✓ {config}: Within {threshold:.1%} of baseline")
        else:
            print(f"  ⚠️  {config}: No baseline available")
    
    return regressions


def analyze_solver_performance(results):
    """Analyze performance by solver variant."""
    print("Analyzing solver performance...")
    
    successful_results = [r for r in results if r['Success']]
    if not successful_results:
        print("No successful results to analyze")
        return
    
    df = pd.DataFrame(successful_results)
    
    # Group by solver
    solver_stats = df.groupby('Solver').agg({
        'Mean Time (s)': ['mean', 'std', 'count', 'min', 'max']
    }).round(4)
    
    print("Solver Performance Analysis:")
    print(solver_stats)
    
    # Find best and worst performers
    best_idx = df['Mean Time (s)'].idxmin()
    worst_idx = df['Mean Time (s)'].idxmax()
    
    best_config = df.loc[best_idx]
    worst_config = df.loc[worst_idx]
    
    improvement = worst_config['Mean Time (s)'] / best_config['Mean Time (s)']
    
    print(f"\nBest Performance: {best_config['Configuration']} ({best_config['Solver']})")
    print(f"  - Time: {best_config['Mean Time (s)']:.4f}s")
    print(f"  - Precision: {best_config['Precision']}")
    print(f"  - Scaler: {best_config['Scaler']}")
    
    print(f"\nWorst Performance: {worst_config['Configuration']} ({worst_config['Solver']})")
    print(f"  - Time: {worst_config['Mean Time (s)']:.4f}s")
    print(f"  - Precision: {worst_config['Precision']}")
    print(f"  - Scaler: {worst_config['Scaler']}")
    
    print(f"\nPerformance Range: {improvement:.2f}x difference")


def main():
    """Run OTFlow performance tests."""
    print("=" * 80)
    print("OTFlow Performance Test Suite")
    print("=" * 80)
    
    # Test 1: OTFlow Performance
    otflow_results = test_otflow_performance()
    
    # Test 2: Comparison against torchdiffeq
    print("=" * 80)
    comparison_results = test_otflow_vs_torchdiffeq()
    print()
    
    # Load baselines and check for regressions
    baselines = load_baselines()
    regressions = check_otflow_regression(otflow_results, baselines)
    print()
    
    # Analyze solver performance
    analyze_solver_performance(otflow_results)
    print()
    
    # Create summary table
    successful_results = [r for r in otflow_results if r['Success']]
    if successful_results:
        df = pd.DataFrame(successful_results)
        
        # Calculate relative performance
        baseline_time = df['Mean Time (s)'].max()
        df['Relative Performance'] = baseline_time / df['Mean Time (s)']
        df['Speedup'] = df['Relative Performance'].apply(lambda x: f"{x:.2f}x")
        
        print("=" * 80)
        print("OTFLOW PERFORMANCE SUMMARY")
        print("=" * 80)
        
        display_cols = ['Configuration', 'Precision', 'Scaler', 'Solver', 'Mean Time (s)', 'Speedup']
        display_df = df[display_cols].copy()
        display_df['Mean Time (s)'] = display_df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        # Save results
        results_file = current_dir / "otflow_performance_results.csv"
        df.to_csv(results_file, index=False)
        print(f"\nResults saved to: {results_file}")
        
        # Save new baselines if no regressions
        if not regressions:
            save_baselines(successful_results)
    
    # Final summary
    print("\n" + "=" * 80)
    print("OTFLOW TEST SUMMARY")
    print("=" * 80)
    
    success_rate = len(successful_results) / len(otflow_results) if otflow_results else 0
    print(f"Success Rate: {success_rate:.1%} ({len(successful_results)}/{len(otflow_results)})")
    print(f"Performance Regressions: {len(regressions)}")
    
    if regressions:
        print("\nRegressions Detected:")
        for reg in regressions:
            print(f"  - {reg['config']}: {reg['regression']:.1%} slower")
    
    # Check torchdiffeq comparison
    torchdiffeq_available = 'torchdiffeq' in comparison_results and comparison_results['torchdiffeq']['success']
    if torchdiffeq_available:
        print("\ntorchdiffeq comparison: Available")
    else:
        print("\ntorchdiffeq comparison: Not available")
    
    all_passed = (success_rate > 0.8 and not regressions)
    print(f"\nOverall Result: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)