"""
Utilities for comparing rampde performance against torchdiffeq.
"""

import time
import torch
import numpy as np
from torch.amp import autocast


def run_torchdiffeq_baseline(model, x, t, method='rk4', num_runs=10, warmup=3):
    """
    Run torchdiffeq baseline for comparison.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        method: ODE method
        num_runs: Number of timing runs
        warmup: Number of warmup runs
    
    Returns:
        tuple: (mean_time, std_time, success)
    """
    try:
        from torchdiffeq import odeint
    except ImportError:
        print("torchdiffeq not available for comparison")
        return None, None, False
    
    device = x.device
    model.to(device)
    model.train()
    
    # Create optimizer for gradient computation
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    try:
        # Warmup runs
        for _ in range(warmup):
            optimizer.zero_grad()
            y = odeint(model, x, t, method=method)
            loss = torch.norm(y[-1])**2
            loss.backward()
            optimizer.step()
        
        # Timing runs
        torch.cuda.synchronize()
        times = []
        
        for _ in range(num_runs):
            optimizer.zero_grad()
            
            start = time.perf_counter()
            y = odeint(model, x, t, method=method)
            loss = torch.norm(y[-1])**2
            loss.backward()
            torch.cuda.synchronize()
            end = time.perf_counter()
            
            times.append(end - start)
        
        return np.mean(times), np.std(times), True
        
    except Exception as e:
        print(f"Error in torchdiffeq baseline: {e}")
        return None, None, False


def compare_against_torchdiffeq(model, x, t, configs, method='rk4'):
    """
    Compare rampde configurations against torchdiffeq baseline.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        configs: List of (name, precision, scaler) configurations
        method: ODE method
    
    Returns:
        dict: Comparison results
    """
    from rampde import odeint as rampde_odeint
    
    results = {}
    
    # Run torchdiffeq baseline
    print("Running torchdiffeq baseline...")
    torchdiffeq_mean, torchdiffeq_std, torchdiffeq_success = run_torchdiffeq_baseline(
        model, x, t, method=method
    )
    
    if torchdiffeq_success:
        results['torchdiffeq'] = {
            'mean_time': torchdiffeq_mean,
            'std_time': torchdiffeq_std,
            'success': True
        }
    else:
        results['torchdiffeq'] = {
            'mean_time': None,
            'std_time': None,
            'success': False
        }
    
    # Run rampde configurations
    for name, precision, scaler in configs:
        print(f"Running {name}...")
        
        device = x.device
        model.to(device)
        model.train()
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        try:
            # Warmup
            for _ in range(3):
                optimizer.zero_grad()
                with autocast(device_type='cuda', dtype=precision):
                    y = rampde_odeint(model, x, t, method=method, loss_scaler=scaler)
                    loss = torch.norm(y[-1])**2
                loss.backward()
                optimizer.step()
            
            # Timing
            torch.cuda.synchronize()
            times = []
            
            for _ in range(10):
                optimizer.zero_grad()
                
                start = time.perf_counter()
                with autocast(device_type='cuda', dtype=precision):
                    y = rampde_odeint(model, x, t, method=method, loss_scaler=scaler)
                    loss = torch.norm(y[-1])**2
                loss.backward()
                torch.cuda.synchronize()
                end = time.perf_counter()
                
                times.append(end - start)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            
            results[name] = {
                'mean_time': mean_time,
                'std_time': std_time,
                'success': True
            }
            
        except Exception as e:
            print(f"Error in {name}: {e}")
            results[name] = {
                'mean_time': None,
                'std_time': None,
                'success': False
            }
    
    return results


def calculate_performance_ratio(rampde_time, torchdiffeq_time):
    """
    Calculate performance ratio between rampde and torchdiffeq.
    
    Args:
        rampde_time: rampde execution time
        torchdiffeq_time: torchdiffeq execution time
    
    Returns:
        float: Performance ratio (>1 means rampde is slower)
    """
    if torchdiffeq_time is None or torchdiffeq_time == 0:
        return None
    
    return rampde_time / torchdiffeq_time


def format_comparison_results(results):
    """
    Format comparison results for display.
    
    Args:
        results: Results dictionary from compare_against_torchdiffeq
    
    Returns:
        str: Formatted comparison table
    """
    import pandas as pd
    
    comparison_data = []
    
    torchdiffeq_time = results.get('torchdiffeq', {}).get('mean_time')
    
    for name, result in results.items():
        if result['success']:
            ratio = calculate_performance_ratio(result['mean_time'], torchdiffeq_time)
            
            comparison_data.append({
                'Implementation': name,
                'Mean Time (s)': f"{result['mean_time']:.4f}",
                'Std Time (s)': f"{result['std_time']:.4f}",
                'vs torchdiffeq': f"{ratio:.2f}x" if ratio else "N/A"
            })
        else:
            comparison_data.append({
                'Implementation': name,
                'Mean Time (s)': "Failed",
                'Std Time (s)': "Failed",
                'vs torchdiffeq': "N/A"
            })
    
    df = pd.DataFrame(comparison_data)
    return df.to_string(index=False)


def check_accuracy_against_torchdiffeq(model, x, t, precision, scaler, method='rk4', tolerance=1e-3):
    """
    Check numerical accuracy against torchdiffeq.
    
    Args:
        model: ODE model
        x: Input tensor
        t: Time points
        precision: Precision dtype
        scaler: Loss scaler
        method: ODE method
        tolerance: Tolerance for accuracy check
    
    Returns:
        tuple: (max_diff, relative_error, accurate)
    """
    try:
        from torchdiffeq import odeint as torchdiffeq_odeint
    except ImportError:
        print("torchdiffeq not available for accuracy check")
        return None, None, False
    
    from rampde import odeint as rampde_odeint
    
    device = x.device
    model.to(device)
    model.eval()
    
    try:
        # Run torchdiffeq
        with torch.no_grad():
            y_torchdiffeq = torchdiffeq_odeint(model, x, t, method=method)
        
        # Run rampde
        with torch.no_grad():
            with autocast(device_type='cuda', dtype=precision):
                y_rampde = rampde_odeint(model, x, t, method=method, loss_scaler=scaler)
        
        # Convert to same precision for comparison
        y_torchdiffeq_fp32 = y_torchdiffeq.float()
        y_rampde_fp32 = y_rampde.float()
        
        # Calculate differences
        diff = torch.abs(y_torchdiffeq_fp32 - y_rampde_fp32)
        max_diff = torch.max(diff).item()
        relative_error = (torch.norm(diff) / torch.norm(y_torchdiffeq_fp32)).item()
        
        accurate = max_diff < tolerance
        
        return max_diff, relative_error, accurate
        
    except Exception as e:
        print(f"Error in torchdiffeq accuracy check: {e}")
        return None, None, False