#!/usr/bin/env python
"""
Generates a figure showing speedup of FP16 vs FP32 for varying problem sizes.
This figure can be used in a paper to demonstrate the performance benefits
of mixed-precision computation with the rampde package.

The script:
1. Measures forward and backward pass times for different problem sizes
2. Computes speedup ratios (FP32 time / FP16 time)
3. Generates a publication-quality figure showing these speedups
4. Saves the raw data to a CSV for future reference

Usage:
    python generate_speedup_figure.py
"""

import time
import csv
import os
import sys
import numpy as np
from contextlib import nullcontext
import matplotlib.pyplot as plt
import matplotlib as mpl

import torch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rampde import odeint

# Disable TF32 to get cleaner FP32 vs FP16 comparison
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Set figure style for publication quality
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = ['Times New Roman']
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.linewidth'] = 0.5
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['lines.linewidth'] = 1.5
mpl.rcParams['lines.markersize'] = 6
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.pad_inches'] = 0.05

# Output directory
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'results_speedup')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Experiment Parameters
# ------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device != "cuda":
    print("CUDA not available, results will not be meaningful")

# Problem dimensions to test
# Testing both dimension and batch size scaling
# 1. Fixed batch size, varying dimension
dimensions = [64, 128, 256, 512, 1024, 2048]
batch_fixed = 256

# 2. Fixed dimension, varying batch size
batch_sizes = [32, 64, 128, 256, 512, 1024, 2048]
dim_fixed = 256

# Common parameters
t_final = 1.0
Nsteps = 5
method = "rk4"
dtype_low = torch.float16
dtype_hi = torch.float32

# Number of repeats for timing stability
NUM_REPEATS = 3

# ------------------------------------------------------------
# ODE Setup: Linear ODE dy/dt = -y (solution: y(t) = e^(-t)*y0)
# ------------------------------------------------------------
class LinearODE(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        A = -torch.eye(dim, device=device, dtype=dtype_hi)
        self.layer = torch.nn.Linear(dim, dim, bias=False, device=device)
        self.layer.weight.data = A

    def forward(self, t, y):
        return self.layer(y)

def analytic_solution(y0, t):
    """y(t) = e^{-t} * y0"""
    t = torch.as_tensor(t, device=y0.device, dtype=y0.dtype)
    return torch.exp(-t) * y0

# ------------------------------------------------------------
# Benchmark Function
# ------------------------------------------------------------
def benchmark_problem(dim, batch_size, precision):
    """
    Benchmark ODE solution for a given dimension, batch size, and precision.
    Returns forward time, backward time, and accuracy.
    """
    mixed = (precision == dtype_low)
    
    # Setup problem
    rhs = LinearODE(dim).to(device)
    y0 = torch.randn(batch_size, dim, device=device, requires_grad=True)
    t_grid = torch.linspace(0.0, t_final, Nsteps + 1, device=device, dtype=dtype_hi)
    
    # Context for mixed precision
    ac_ctx = torch.autocast(device_type="cuda", dtype=precision) if mixed else nullcontext()
    
    # Warmup
    with ac_ctx:
        y_all = odeint(rhs, y0, t_grid, method=method)
        yN = y_all[-1]
        loss = yN.to(dtype_hi).sum()
        loss.backward()
    y0.grad = None
    rhs.zero_grad()
    torch.cuda.synchronize()
    
    # Forward timing
    forward_times = []
    for _ in range(NUM_REPEATS):
        torch.cuda.synchronize()
        with ac_ctx:
            start_time = time.time()
            y_all = odeint(rhs, y0, t_grid, method=method)
            yN = y_all[-1]
            torch.cuda.synchronize()
            end_time = time.time()
        forward_times.append(end_time - start_time)
    
    # Backward timing
    backward_times = []
    for _ in range(NUM_REPEATS):
        loss = yN.to(dtype_hi).sum()
        torch.cuda.synchronize()
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.time()
        backward_times.append(end_time - start_time)
        y0.grad = None
        rhs.zero_grad()
    
    # Accuracy check
    y_true = analytic_solution(y0.to(dtype_hi), t_final)
    y_error = (yN.to(dtype_hi) - y_true).abs().max().item()
    
    # Memory usage in MB
    memory_usage = torch.cuda.max_memory_allocated() / 1e6
    torch.cuda.reset_peak_memory_stats()
    
    return {
        'forward_time': min(forward_times),  # Use minimum time as most reliable
        'backward_time': min(backward_times),
        'total_time': min(forward_times) + min(backward_times),
        'y_error': y_error,
        'memory': memory_usage
    }

# ------------------------------------------------------------
# Run Benchmarks
# ------------------------------------------------------------
def run_benchmarks():
    """Run all benchmarks and return results."""
    results_dim = []
    results_batch = []
    
    # 1. Fixed batch, varying dimension
    print("\nRunning fixed batch, varying dimension benchmarks...")
    for dim in dimensions:
        print(f"  Dimension: {dim}")
        result_fp32 = benchmark_problem(dim, batch_fixed, dtype_hi)
        result_fp16 = benchmark_problem(dim, batch_fixed, dtype_low)
        
        # Calculate speedups
        fwd_speedup = result_fp32['forward_time'] / result_fp16['forward_time']
        bwd_speedup = result_fp32['backward_time'] / result_fp16['backward_time']
        total_speedup = result_fp32['total_time'] / result_fp16['total_time']
        mem_ratio = result_fp16['memory'] / result_fp32['memory']
        
        results_dim.append({
            'dim': dim,
            'batch': batch_fixed,
            'fp32_fwd': result_fp32['forward_time'],
            'fp16_fwd': result_fp16['forward_time'],
            'fp32_bwd': result_fp32['backward_time'],
            'fp16_bwd': result_fp16['backward_time'],
            'fp32_total': result_fp32['total_time'],
            'fp16_total': result_fp16['total_time'],
            'fp32_error': result_fp32['y_error'],
            'fp16_error': result_fp16['y_error'],
            'fp32_memory': result_fp32['memory'],
            'fp16_memory': result_fp16['memory'],
            'fwd_speedup': fwd_speedup,
            'bwd_speedup': bwd_speedup,
            'total_speedup': total_speedup,
            'mem_ratio': mem_ratio
        })
        print(f"    FP16 vs FP32: {fwd_speedup:.2f}x forward, {bwd_speedup:.2f}x backward")
    
    # 2. Fixed dimension, varying batch
    print("\nRunning fixed dimension, varying batch benchmarks...")
    for batch in batch_sizes:
        print(f"  Batch size: {batch}")
        result_fp32 = benchmark_problem(dim_fixed, batch, dtype_hi)
        result_fp16 = benchmark_problem(dim_fixed, batch, dtype_low)
        
        # Calculate speedups
        fwd_speedup = result_fp32['forward_time'] / result_fp16['forward_time']
        bwd_speedup = result_fp32['backward_time'] / result_fp16['backward_time']
        total_speedup = result_fp32['total_time'] / result_fp16['total_time']
        mem_ratio = result_fp16['memory'] / result_fp32['memory']
        
        results_batch.append({
            'dim': dim_fixed,
            'batch': batch,
            'fp32_fwd': result_fp32['forward_time'],
            'fp16_fwd': result_fp16['forward_time'],
            'fp32_bwd': result_fp32['backward_time'],
            'fp16_bwd': result_fp16['backward_time'],
            'fp32_total': result_fp32['total_time'],
            'fp16_total': result_fp16['total_time'],
            'fp32_error': result_fp32['y_error'],
            'fp16_error': result_fp16['y_error'],
            'fp32_memory': result_fp32['memory'],
            'fp16_memory': result_fp16['memory'],
            'fwd_speedup': fwd_speedup,
            'bwd_speedup': bwd_speedup,
            'total_speedup': total_speedup,
            'mem_ratio': mem_ratio
        })
        print(f"    FP16 vs FP32: {fwd_speedup:.2f}x forward, {bwd_speedup:.2f}x backward")
    
    return results_dim, results_batch

# ------------------------------------------------------------
# Generate Figures
# ------------------------------------------------------------
def generate_figures(results_dim, results_batch):
    """Generate publication-quality figures from benchmark results."""
    # Create a 2x2 subplot figure
    fig, axs = plt.subplots(2, 2, figsize=(10, 9))
    
    # Extract data for plotting
    dim_values = [r['dim'] for r in results_dim]
    dim_fwd_speedup = [r['fwd_speedup'] for r in results_dim]
    dim_bwd_speedup = [r['bwd_speedup'] for r in results_dim]
    dim_total_speedup = [r['total_speedup'] for r in results_dim]
    dim_mem_ratio = [r['mem_ratio'] for r in results_dim]
    
    batch_values = [r['batch'] for r in results_batch]
    batch_fwd_speedup = [r['fwd_speedup'] for r in results_batch]
    batch_bwd_speedup = [r['bwd_speedup'] for r in results_batch]
    batch_total_speedup = [r['total_speedup'] for r in results_batch]
    batch_mem_ratio = [r['mem_ratio'] for r in results_batch]
    
    # Plot 1: Dimension scaling - Speedup
    ax = axs[0, 0]
    ax.plot(dim_values, dim_fwd_speedup, 'o-', label='Forward')
    ax.plot(dim_values, dim_bwd_speedup, 's-', label='Backward')
    ax.plot(dim_values, dim_total_speedup, '^-', label='Total')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Feature Dimension (batch size = {})'.format(batch_fixed))
    ax.set_ylabel('Speedup (FP32 time / FP16 time)')
    ax.set_title('(a) Speedup vs. Feature Dimension')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot 2: Batch scaling - Speedup
    ax = axs[0, 1]
    ax.plot(batch_values, batch_fwd_speedup, 'o-', label='Forward')
    ax.plot(batch_values, batch_bwd_speedup, 's-', label='Backward')
    ax.plot(batch_values, batch_total_speedup, '^-', label='Total')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch Size (dimension = {})'.format(dim_fixed))
    ax.set_ylabel('Speedup (FP32 time / FP16 time)')
    ax.set_title('(b) Speedup vs. Batch Size')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot 3: Dimension scaling - Memory
    ax = axs[1, 0]
    ax.plot(dim_values, dim_mem_ratio, 'o-', color='green')
    ax.axhline(y=0.5, linestyle='--', color='red', alpha=0.7, label='Ideal (0.5)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Feature Dimension (batch size = {})'.format(batch_fixed))
    ax.set_ylabel('Memory Ratio (FP16 / FP32)')
    ax.set_title('(c) Memory Usage vs. Feature Dimension')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Plot 4: Batch scaling - Memory
    ax = axs[1, 1]
    ax.plot(batch_values, batch_mem_ratio, 'o-', color='green')
    ax.axhline(y=0.5, linestyle='--', color='red', alpha=0.7, label='Ideal (0.5)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('Batch Size (dimension = {})'.format(dim_fixed))
    ax.set_ylabel('Memory Ratio (FP16 / FP32)')
    ax.set_title('(d) Memory Usage vs. Batch Size')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend()
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    figure_path = os.path.join(OUTPUT_DIR, 'speedup_figure.png')
    plt.savefig(figure_path, dpi=300)
    print(f"\nFigure saved to: {figure_path}")
    
    # Also save PDF for publication
    pdf_path = os.path.join(OUTPUT_DIR, 'speedup_figure.pdf')
    plt.savefig(pdf_path)
    print(f"PDF saved to: {pdf_path}")
    
    # Save raw data as CSV
    dim_csv_path = os.path.join(OUTPUT_DIR, 'dimension_scaling_results.csv')
    batch_csv_path = os.path.join(OUTPUT_DIR, 'batch_scaling_results.csv')
    
    with open(dim_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_dim[0].keys())
        writer.writeheader()
        writer.writerows(results_dim)
    
    with open(batch_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results_batch[0].keys())
        writer.writeheader()
        writer.writerows(results_batch)
    
    print(f"Raw data saved to: {dim_csv_path} and {batch_csv_path}")
    
    return fig

# ------------------------------------------------------------
# Main Function
# ------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 80)
    print("Generating speedup benchmarks for rampde (FP16 vs FP32)")
    print("=" * 80)
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("Warning: CUDA not available, results will not be meaningful")
    
    # Print CUDA device info
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name()}")
        print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
    
    # Run benchmarks
    results_dim, results_batch = run_benchmarks()
    
    # Generate figures
    fig = generate_figures(results_dim, results_batch)
    
    print("\nBenchmark complete!")
    print("=" * 80)
