"""
Timing and measurement utilities for performance tests.
"""

import time
import torch
import numpy as np
from torch.amp import autocast

def run_timing_test(model, x, t, precision, scaler, method='rk4', num_runs=10, warmup=3):
    """
    Run timing test with specified configuration.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        precision: Precision dtype
        scaler: Loss scaler
        method: ODE method
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    
    Returns:
        tuple: (mean_time, std_time, success)
    """
    from torchmpnode import odeint
    
    device = x.device
    model.to(device)
    model.train()
    
    # Create optimizer for gradient computation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    try:
        # Warmup runs
        for _ in range(warmup):
            optimizer.zero_grad()
            with autocast(device_type='cuda', dtype=precision):
                y = odeint(model, x, t, method=method, loss_scaler=scaler)
                loss = torch.norm(y[-1])**2
            loss.backward()
            optimizer.step()
        
        # Timing runs
        torch.cuda.synchronize()
        times = []
        
        for _ in range(num_runs):
            optimizer.zero_grad()
            
            start = time.perf_counter()
            with autocast(device_type='cuda', dtype=precision):
                y = odeint(model, x, t, method=method, loss_scaler=scaler)
                loss = torch.norm(y[-1])**2
            loss.backward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        return np.mean(times), np.std(times), True
        
    except Exception as e:
        print(f"Error in timing test: {e}")
        return None, None, False


def measure_memory_usage(model, x, t, precision, scaler, method='rk4'):
    """
    Measure memory usage for a configuration.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        precision: Precision dtype
        scaler: Loss scaler
        method: ODE method
    
    Returns:
        dict: Memory usage statistics
    """
    from torchmpnode import odeint
    
    device = x.device
    model.to(device)
    model.train()
    
    # Clear cache and measure baseline
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    baseline_memory = torch.cuda.memory_allocated()
    
    try:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        optimizer.zero_grad()
        
        with autocast(device_type='cuda', dtype=precision):
            y = odeint(model, x, t, method=method, loss_scaler=scaler)
            loss = torch.norm(y[-1])**2
        
        forward_memory = torch.cuda.memory_allocated()
        
        loss.backward()
        
        backward_memory = torch.cuda.memory_allocated()
        peak_memory = torch.cuda.max_memory_allocated()
        
        return {
            'baseline_mb': baseline_memory / 1024**2,
            'forward_mb': forward_memory / 1024**2,
            'backward_mb': backward_memory / 1024**2,
            'peak_mb': peak_memory / 1024**2,
            'forward_increase_mb': (forward_memory - baseline_memory) / 1024**2,
            'backward_increase_mb': (backward_memory - forward_memory) / 1024**2,
            'total_increase_mb': (backward_memory - baseline_memory) / 1024**2,
        }
        
    except Exception as e:
        print(f"Error in memory measurement: {e}")
        return None


def verify_solver_selection(scaler, precision):
    """
    Verify that the correct solver is selected for the given configuration.
    
    Args:
        scaler: Loss scaler
        precision: Precision dtype
    
    Returns:
        tuple: (expected_solver, actual_solver, matches)
    """
    from torchmpnode.odeint import _select_ode_solver
    from torchmpnode.loss_scalers import DynamicScaler
    
    # Determine expected solver
    if isinstance(scaler, DynamicScaler):
        expected_solver = "FixedGridODESolverDynamic"
    elif scaler is None and precision in [torch.float32, torch.bfloat16]:
        expected_solver = "FixedGridODESolverUnscaled"
    else:
        expected_solver = "FixedGridODESolverUnscaledSafe"
    
    # Get actual solver
    actual_solver_class, actual_scaler = _select_ode_solver(scaler, precision)
    actual_solver = actual_solver_class.__name__
    
    return expected_solver, actual_solver, expected_solver == actual_solver


def check_numerical_accuracy(model, x, t, config1, config2, tolerance=1e-4):
    """
    Check numerical accuracy between two configurations.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        config1: First configuration (precision, scaler, method)
        config2: Second configuration (precision, scaler, method)
        tolerance: Tolerance for accuracy check
    
    Returns:
        tuple: (max_diff, relative_error, accurate)
    """
    from torchmpnode import odeint
    
    device = x.device
    model.to(device)
    model.eval()
    
    try:
        # Run first configuration
        precision1, scaler1, method1 = config1
        with autocast(device_type='cuda', dtype=precision1):
            y1 = odeint(model, x, t, method=method1, loss_scaler=scaler1)
        
        # Run second configuration
        precision2, scaler2, method2 = config2
        with autocast(device_type='cuda', dtype=precision2):
            y2 = odeint(model, x, t, method=method2, loss_scaler=scaler2)
        
        # Convert to same precision for comparison
        y1_fp32 = y1.float()
        y2_fp32 = y2.float()
        
        # Calculate differences
        diff = torch.abs(y1_fp32 - y2_fp32)
        max_diff = torch.max(diff).item()
        relative_error = (torch.norm(diff) / torch.norm(y1_fp32)).item()
        
        accurate = max_diff < tolerance
        
        return max_diff, relative_error, accurate
        
    except Exception as e:
        print(f"Error in accuracy check: {e}")
        return None, None, False


def format_timing_results(results, baseline_time=None):
    """
    Format timing results for display.
    
    Args:
        results: List of result dictionaries
        baseline_time: Baseline time for speedup calculation
    
    Returns:
        str: Formatted results table
    """
    import pandas as pd
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate speedup if baseline provided
    if baseline_time:
        df['Speedup'] = baseline_time / df['Mean Time (s)']
        df['Speedup'] = df['Speedup'].apply(lambda x: f"{x:.2f}x")
    
    # Format time columns
    df['Mean Time (s)'] = df['Mean Time (s)'].apply(lambda x: f"{x:.4f}")
    
    return df.to_string(index=False)