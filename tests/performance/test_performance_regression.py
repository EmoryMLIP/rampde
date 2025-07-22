#!/usr/bin/env python3
"""
Main performance regression test for torchmpnode three-variant architecture.

This test verifies that:
1. The correct solver is selected for each configuration
2. Performance meets established baselines
3. All variants produce correct results
4. Performance comparison against torchdiffeq is maintained

Based on the original test_new_architecture.py from the ablation study.
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

from utils.test_models import create_simple_ode_model, create_test_data, get_model_info
from utils.timing_utils import run_timing_test, verify_solver_selection, format_timing_results
from utils.comparison_utils import compare_against_torchdiffeq, format_comparison_results


def load_baselines(test_type):
    """Load performance baselines for comparison."""
    baseline_file = current_dir / "baselines" / f"{test_type}_baseline.json"
    
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Warning: No baseline file found at {baseline_file}")
        return {}


def save_baselines(test_type, results):
    """Save performance baselines for future comparison."""
    baseline_file = current_dir / "baselines" / f"{test_type}_baseline.json"
    
    # Create baselines directory if it doesn't exist
    baseline_file.parent.mkdir(exist_ok=True)
    
    with open(baseline_file, 'w') as f:
        json.dump(results, f, indent=2)


def test_solver_selection():
    """Test that the correct solver is selected for each configuration."""
    print("Testing solver selection logic...")
    
    test_cases = [
        (None, torch.float32, "FixedGridODESolverUnscaled"),
        (None, torch.bfloat16, "FixedGridODESolverUnscaled"),
        (None, torch.float16, "FixedGridODESolverUnscaledSafe"),
        (DynamicScaler(torch.float16), torch.float16, "FixedGridODESolverDynamic"),
    ]
    
    all_passed = True
    
    for scaler, precision, expected_name in test_cases:
        expected_solver, actual_solver, matches = verify_solver_selection(scaler, precision)
        
        if matches:
            scaler_name = type(scaler).__name__ if scaler else 'None'
            print(f"  ✓ {precision} + {scaler_name}: {actual_solver}")
        else:
            scaler_name = type(scaler).__name__ if scaler else 'None'
            print(f"  ✗ {precision} + {scaler_name}: Expected {expected_name}, got {actual_solver}")
            all_passed = False
    
    return all_passed


def test_performance_regression():
    """Test performance against established baselines."""
    print("Testing performance regression...")
    
    device = torch.device('cuda:0')
    model = create_simple_ode_model(dim=32)  # Match the test data dim
    x, t = create_test_data('simple', device)
    
    # Test configurations
    configs = [
        ("float32", torch.float32, None),
        ("bfloat16", torch.bfloat16, None),
        ("float16", torch.float16, None),
        ("float16_Dynamic", torch.float16, DynamicScaler(torch.float16)),
    ]
    
    results = []
    
    for name, precision, scaler in configs:
        print(f"  Testing {name}...")
        
        # Verify solver selection
        expected_solver, actual_solver, matches = verify_solver_selection(scaler, precision)
        
        if not matches:
            print(f"    ✗ Wrong solver selected: {actual_solver}")
            continue
        
        # Run performance test
        mean_time, std_time, success = run_timing_test(
            model, x, t, precision, scaler, method='rk4', num_runs=10, warmup=3
        )
        
        if success:
            print(f"    ✓ Time: {mean_time:.4f} ± {std_time:.4f} s")
            
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
            print(f"    ✗ Failed to run")
            results.append({
                'Configuration': name,
                'Precision': str(precision).split('.')[-1],
                'Scaler': type(scaler).__name__ if scaler else 'None',
                'Solver': expected_solver.replace('FixedGridODESolver', ''),
                'Mean Time (s)': None,
                'Std Time (s)': None,
                'Success': False
            })
    
    return results


def test_functionality():
    """Test that all variants produce correct results."""
    print("Testing functionality across all variants...")
    
    device = torch.device('cuda:0')
    model = create_simple_ode_model(dim=32)  # Match the test data dim
    model.to(device)  # Move model to device
    x, t = create_test_data('simple', device)
    
    # Test configurations
    configs = [
        ("float32", torch.float32, None),
        ("bfloat16", torch.bfloat16, None),
        ("float16", torch.float16, None),
        ("float16_dynamic", torch.float16, DynamicScaler(torch.float16)),
    ]
    
    results = {}
    
    for name, precision, scaler in configs:
        try:
            from torch.amp import autocast
            with autocast(device_type='cuda', dtype=precision):
                solution = odeint(model, x, t, method='rk4', loss_scaler=scaler)
                results[name] = solution.cpu()
                print(f"  ✓ {name}: Shape {solution.shape}, dtype {solution.dtype}")
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
            return False
    
    # Check that solutions are similar (allowing for precision differences)
    if len(results) > 1:
        ref_key = list(results.keys())[0]
        ref_solution = results[ref_key]
        
        for name, solution in results.items():
            if name != ref_key:
                diff = torch.norm(solution - ref_solution).item()
                print(f"    Difference vs {ref_key}: {diff:.2e}")
                
                # Allow for reasonable precision differences
                if diff > 1e-2:
                    print(f"    ✗ Large difference detected: {diff:.2e}")
                    return False
    
    return True


def test_backward_pass():
    """Test that backward pass works correctly."""
    print("Testing backward pass...")
    
    device = torch.device('cuda:0')
    model = create_simple_ode_model(dim=32)  # Match the test data dim
    x, t = create_test_data('simple', device)
    
    # Test configurations
    configs = [
        ("float32", torch.float32, None),
        ("bfloat16", torch.bfloat16, None),
        ("float16", torch.float16, None),
    ]
    
    for name, precision, scaler in configs:
        try:
            # Create fresh model for each test
            test_model = create_simple_ode_model(dim=32)  # Match the test data dim
            test_model.to(device)
            optimizer = torch.optim.Adam(test_model.parameters(), lr=1e-3)
            
            optimizer.zero_grad()
            
            from torch.amp import autocast
            with autocast(device_type='cuda', dtype=precision):
                solution = odeint(test_model, x, t, method='rk4', loss_scaler=scaler)
                loss = torch.norm(solution[-1])**2
            
            loss.backward()
            
            # Check that gradients exist and are finite
            grad_norm = 0
            for param in test_model.parameters():
                if param.grad is not None:
                    grad_norm += torch.norm(param.grad).item()**2
            grad_norm = grad_norm**0.5
            
            if grad_norm > 0 and torch.isfinite(torch.tensor(grad_norm)):
                print(f"  ✓ {name}: Gradient norm {grad_norm:.4f}")
            else:
                print(f"  ✗ {name}: Invalid gradient norm {grad_norm}")
                return False
            
        except Exception as e:
            print(f"  ✗ {name}: Error - {e}")
            return False
    
    return True


def check_performance_regression(results, baselines, threshold=0.1):
    """Check for performance regression against baselines."""
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


def main():
    """Run all performance regression tests."""
    print("=" * 80)
    print("torchmpnode Performance Regression Test Suite")
    print("=" * 80)
    
    # Test 1: Solver Selection
    solver_selection_passed = test_solver_selection()
    print()
    
    # Test 2: Functionality
    functionality_passed = test_functionality()
    print()
    
    # Test 3: Backward Pass
    backward_pass_passed = test_backward_pass()
    print()
    
    # Test 4: Performance Regression
    performance_results = test_performance_regression()
    print()
    
    # Load baselines and check for regressions
    baselines = load_baselines('simple_ode')
    regressions = check_performance_regression(performance_results, baselines)
    
    # Create summary table
    successful_results = [r for r in performance_results if r['Success']]
    if successful_results:
        df = pd.DataFrame(successful_results)
        
        # Calculate relative performance
        baseline_time = df['Mean Time (s)'].max()
        df['Relative Performance'] = baseline_time / df['Mean Time (s)']
        df['Speedup'] = df['Relative Performance'].apply(lambda x: f"{x:.2f}x")
        
        print("=" * 80)
        print("PERFORMANCE SUMMARY")
        print("=" * 80)
        
        display_cols = ['Configuration', 'Precision', 'Scaler', 'Solver', 'Mean Time (s)', 'Speedup']
        display_df = df[display_cols].copy()
        display_df['Mean Time (s)'] = display_df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
        
        print(display_df.to_string(index=False))
        
        # Save new baselines if no regressions
        if not regressions:
            baseline_data = {}
            for result in successful_results:
                baseline_data[result['Configuration']] = {
                    'mean_time': result['Mean Time (s)'],
                    'std_time': result['Std Time (s)'],
                    'solver': result['Solver']
                }
            save_baselines('simple_ode', baseline_data)
    
    # Final summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    print(f"Solver Selection: {'PASS' if solver_selection_passed else 'FAIL'}")
    print(f"Functionality: {'PASS' if functionality_passed else 'FAIL'}")
    print(f"Backward Pass: {'PASS' if backward_pass_passed else 'FAIL'}")
    print(f"Performance Regression: {'PASS' if not regressions else 'FAIL'}")
    
    if regressions:
        print(f"\nPerformance Regressions Detected: {len(regressions)}")
        for reg in regressions:
            print(f"  - {reg['config']}: {reg['regression']:.1%} slower")
    
    all_passed = (solver_selection_passed and functionality_passed and 
                  backward_pass_passed and not regressions)
    
    print(f"\nOverall Result: {'PASS' if all_passed else 'FAIL'}")
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)