#!/usr/bin/env python
"""
Performance test for torchmpnode package, suitable for CI pipelines.

This test verifies that:
1. FP16 computation is faster than FP32 for representative problem sizes
2. Accuracy is not degraded below acceptable thresholds

CI environments can run this test to ensure performance benefits persist
as the codebase evolves.

Note: This test requires CUDA. It will be skipped if CUDA is not available.
"""

import os
import sys
import unittest
import time
from contextlib import nullcontext

import torch
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint, NoScaler

# Control test output verbosity
QUIET = os.environ.get('TORCHMPNODE_TEST_QUIET', '0') == '1'

class SpeedTest(unittest.TestCase):
    """Test case for performance benchmarking of torchmpnode."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures once for all tests."""
        # Skip if CUDA is not available
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA not available, skipping performance tests")
        
        # Disable TF32 to get cleaner FP32 vs FP16 comparison
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        
        # Store device
        cls.device = "cuda"
        
        # Precision types
        cls.dtype_hi = torch.float32
        cls.dtype_low = torch.float16
        
        # Set up test parameters - smaller than paper figure for CI speed
        cls.dimensions = [ 1024, 2048, 4096]  # Problem dimensions to test
        cls.batch_size = 1024
        cls.t_final = 1.0
        cls.Nsteps = 8
        cls.method = "rk4"
        
        # Acceptable thresholds
        cls.min_speedup = 1.2  # FP16 should be at least 20% faster
        cls.max_error_ratio = 10.0  # FP16 error should be at most 10x FP32 error

    def setUp(self):
        """Set up test fixtures before each test."""
        if not QUIET:
            print(f"\nRunning performance test with dimensions: {self.dimensions}")
    
    def create_ode_problem(self, dim):
        """Create a linear ODE problem with the given dimension."""
        # Set up A matrix for linear ODE dy/dt = Ay with A = -I
        A = -0.1*torch.eye(dim, device=self.device, dtype=self.dtype_hi)
        
        # Create ODE model
        rhs = LinearODE(A).to(self.device)
        
        # Create initial state
        y0 = torch.randn(self.batch_size, dim, device=self.device, requires_grad=True)
        
        # Create time grid
        t_grid = torch.linspace(0.0, self.t_final, self.Nsteps + 1, 
                                device=self.device, dtype=self.dtype_hi)
        
        return rhs, y0, t_grid
    
    def benchmark_precision(self, dim, precision):
        """Benchmark ODE solution for a given dimension and precision."""
        mixed = (precision == self.dtype_low)
        
        # Set up ODE problem
        rhs, y0, t_grid = self.create_ode_problem(dim)
        
        # Context for mixed precision
        ac_ctx = torch.autocast(device_type="cuda", dtype=precision) if mixed else nullcontext()
        
        # Warmup
        with ac_ctx:
            y_all = odeint(rhs, y0, t_grid, method=self.method)
            yN = y_all[-1]
            loss = yN.to(self.dtype_hi).sum()
            loss.backward()
        y0.grad = None
        rhs.zero_grad()
        torch.cuda.synchronize()
        
        # Forward timing
        torch.cuda.synchronize()
        with ac_ctx:
            start_time = time.time()
            y_all = odeint(rhs, y0, t_grid, method=self.method)
            yN = y_all[-1]
            torch.cuda.synchronize()
            end_time = time.time()
        forward_time = end_time - start_time
        
        # Backward timing
        loss = yN.to(self.dtype_hi).sum()
        torch.cuda.synchronize()
        start_time = time.time()
        loss.backward()
        torch.cuda.synchronize()
        end_time = time.time()
        backward_time = end_time - start_time
        
        # Accuracy check
        y_true = analytic_solution(y0.to(self.dtype_hi), self.t_final)
        y_error = (yN.to(self.dtype_hi) - y_true).abs().max().item()
        
        return {
            'forward_time': forward_time,
            'backward_time': backward_time,
            'total_time': forward_time + backward_time,
            'y_error': y_error
        }
    
    def test_performance_scaling(self):
        """Test that FP16 is faster than FP32 for all problem sizes."""
        results = []
        
        for dim in self.dimensions:
            # Run benchmarks for both precisions
            result_fp32 = self.benchmark_precision(dim, self.dtype_hi)
            result_fp16 = self.benchmark_precision(dim, self.dtype_low)
            
            # Calculate speedups
            fwd_speedup = result_fp32['forward_time'] / result_fp16['forward_time']
            bwd_speedup = result_fp32['backward_time'] / result_fp16['backward_time']
            total_speedup = result_fp32['total_time'] / result_fp16['total_time']
            
            # Calculate error ratio
            error_ratio = result_fp16['y_error'] / max(result_fp32['y_error'], 1e-10)
            
            results.append({
                'dim': dim,
                'fp32_time': result_fp32['total_time'],
                'fp16_time': result_fp16['total_time'],
                'speedup': total_speedup,
                'fp32_error': result_fp32['y_error'],
                'fp16_error': result_fp16['y_error'],
                'error_ratio': error_ratio
            })
            
            if not QUIET:
                print(f"Dimension {dim}: {total_speedup:.2f}x speedup, " 
                      f"error ratio: {error_ratio:.2f}")
            
            # Assert that FP16 is faster
            self.assertGreater(
                total_speedup, self.min_speedup,
                f"FP16 should be at least {self.min_speedup}x faster than FP32 "
                f"(got {total_speedup:.2f}x)"
            )
            
            # Assert that error is acceptable
            self.assertLess(
                error_ratio, self.max_error_ratio,
                f"FP16 error should be at most {self.max_error_ratio}x FP32 error "
                f"(got {error_ratio:.2f}x)"
            )
        
        # Print summary table
        if not QUIET:
            print("\nPerformance Summary:")
            print("-" * 70)
            print(f"{'Dimension':<10} {'FP32 Time (s)':<14} {'FP16 Time (s)':<14} "
                  f"{'Speedup':<10} {'Error Ratio':<12}")
            print("-" * 70)
            for r in results:
                print(f"{r['dim']:<10} {r['fp32_time']:<14.6f} {r['fp16_time']:<14.6f} "
                      f"{r['speedup']:<10.2f} {r['error_ratio']:<12.2f}")
            print("-" * 70)

# ------------------------------------------------------------
# Helper Classes & Functions
# ------------------------------------------------------------
class LinearODE(torch.nn.Module):
    """Linear ODE model: dy/dt = Ay"""
    def __init__(self, A):
        super().__init__()
        d = A.shape[0]
        self.layer = torch.nn.Linear(d, d, bias=False, device=A.device)
        self.layer.weight.data = A

    def forward(self, t, y):
        return self.layer(y)

def analytic_solution(y0, t):
    """Analytic solution: y(t) = e^{-t} * y0"""
    t = torch.as_tensor(t, device=y0.device, dtype=y0.dtype)
    return torch.exp(-t) * y0

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    unittest.main()
