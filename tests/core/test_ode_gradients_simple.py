#!/usr/bin/env python3
"""
Simplified gradient test for ODE STL10 components.
This test creates minimal versions of the ODE components to test gradient flow.
"""

import torch
import torch.nn as nn
import numpy as np
import random
import math


class SimpleODEFunc(nn.Module):
    """Simplified version of ODEFunc for gradient testing."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)
        self.norm = nn.InstanceNorm2d(channels, affine=True)
        self.act = nn.ReLU()
        
    def forward(self, t, x):
        # Simple time-dependent function
        scale = 1.0 + 0.1 * t  # Mild time dependence
        x = self.conv(x) * scale
        x = self.norm(x)
        x = self.act(x)
        return x


def gradient_check(func, inputs, names=None, eps=1e-6, rtol=1e-6, atol=1e-8):
    """
    Perform numerical gradient checking for a function.
    
    Args:
        func: Function to test (should return a scalar)
        inputs: List of tensors to check gradients for
        names: Optional list of names for the inputs
        eps: Finite difference step size
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        bool: True if gradients are accurate within tolerance
    """
    if names is None:
        names = [f"input_{i}" for i in range(len(inputs))]
        
    # Ensure inputs require grad
    for inp in inputs:
        inp.requires_grad_(True)
        if inp.grad is not None:
            inp.grad.zero_()
    
    # Compute analytical gradients
    output = func()
    output.backward()
    
    analytical_grads = []
    for inp in inputs:
        if inp.grad is not None:
            analytical_grads.append(inp.grad.clone())
        else:
            analytical_grads.append(torch.zeros_like(inp))
    
    # Check each input tensor
    all_passed = True
    for i, (inp, analytical_grad, name) in enumerate(zip(inputs, analytical_grads, names)):
        print(f"Checking gradients for {name}...")
        
        # Flatten for easier iteration
        inp_flat = inp.view(-1)
        grad_flat = analytical_grad.view(-1)
        
        # Sample a subset of elements for large tensors
        n_samples = min(20, inp_flat.numel())
        # Use a generator with fixed seed for deterministic sampling
        generator = torch.Generator()
        generator.manual_seed(42)
        indices = torch.randperm(inp_flat.numel(), generator=generator)[:n_samples]
        
        max_error = 0.0
        failed_count = 0
        
        for idx in indices:
            # Clear gradients
            for inp_clear in inputs:
                if inp_clear.grad is not None:
                    inp_clear.grad.zero_()
            
            # Positive perturbation
            inp_flat[idx] += eps
            f_plus = func()
            
            # Negative perturbation
            inp_flat[idx] -= 2 * eps
            f_minus = func()
            
            # Restore original value
            inp_flat[idx] += eps
            
            # Numerical gradient
            numerical_grad = (f_plus - f_minus) / (2 * eps)
            analytical_val = grad_flat[idx]
            
            # Compute relative and absolute error
            error = abs(numerical_grad - analytical_val)
            if abs(analytical_val) > 1e-10:
                rel_error = error / abs(analytical_val)
            else:
                rel_error = error
            
            max_error = max(max_error, error)
            
            # Check if within tolerance
            if error > atol and rel_error > rtol:
                if failed_count < 3:  # Only print first few failures
                    print(f"  Element {idx}: Numerical={numerical_grad:.8e}, "
                          f"Analytical={analytical_val:.8e}, Error={error:.8e}")
                failed_count += 1
                all_passed = False
        
        print(f"  {name}: Max error = {max_error:.8e}, Failed elements = {failed_count}/{n_samples}")
        
    return all_passed


def test_simple_ode_func():
    """Test gradients for SimpleODEFunc."""
    print("=== Testing SimpleODEFunc Gradients ===")

    # Set deterministic seeds for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64  # Use double precision for accurate gradients
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Create model
    channels = 4
    model = SimpleODEFunc(channels).to(device).to(dtype)
    
    # Create inputs
    batch_size = 1
    img_size = 8
    x = torch.randn(batch_size, channels, img_size, img_size, 
                   device=device, dtype=dtype, requires_grad=True)
    t = torch.tensor(0.5, device=device, dtype=dtype, requires_grad=True)
    
    # Test function
    def func():
        output = model(t, x)
        return torch.sum(output)
    
    # Test gradients for inputs and first few parameters
    params = list(model.parameters())[:3]  # Just first few parameters for speed
    param_names = ["conv.weight", "conv.bias", "norm.weight"]
    
    inputs = [x, t] + params
    names = ["input_x", "time_t"] + param_names
    
    # Perform gradient check
    passed = gradient_check(func, inputs, names)
    
    if passed:
        print("✓ SimpleODEFunc gradient check PASSED")
    else:
        print("✗ SimpleODEFunc gradient check FAILED")
    
    return passed


def test_precision_robustness():
    """Test how gradient accuracy changes with precision."""
    print("\n=== Testing Precision Robustness ===")

    # Set deterministic seeds for reproducible tests
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)

    # Set deterministic algorithms for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precisions = [torch.float64, torch.float32, torch.float16]
    
    results = {}
    
    for dtype in precisions:
        print(f"\nTesting with {dtype}...")
        
        # Skip float16 if not supported
        if dtype == torch.float16 and device.type == "cpu":
            print("  Skipping float16 on CPU")
            continue
            
        try:
            # Create simple model
            model = SimpleODEFunc(2).to(device).to(dtype)
            
            # Create inputs
            x = torch.randn(1, 2, 4, 4, device=device, dtype=dtype, requires_grad=True)
            t = torch.tensor(0.3, device=device, dtype=dtype, requires_grad=True)
            
            # Test function
            def func():
                output = model(t, x)
                return torch.sum(output)
            
            # Test just input gradients for speed
            inputs = [x, t]
            names = ["input_x", "time_t"]
            
            # Adjust tolerances based on precision
            if dtype == torch.float64:
                rtol, atol = 1e-6, 1e-8
            elif dtype == torch.float32:
                rtol, atol = 1e-4, 1e-6
            else:  # float16
                rtol, atol = 1e-2, 1e-4
            
            passed = gradient_check(func, inputs, names, rtol=rtol, atol=atol)
            results[dtype] = passed
            
            print(f"  {dtype}: {'PASSED' if passed else 'FAILED'}")
            
        except Exception as e:
            print(f"  {dtype}: ERROR - {e}")
            results[dtype] = False
    
    return results


def main():
    """Run all gradient tests."""
    print("Running ODE STL10 Gradient Tests")
    print("=" * 50)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Test 1: Simple ODE function
    test1_passed = test_simple_ode_func()
    
    # Test 2: Precision robustness
    test2_results = test_precision_robustness()
    
    # Summary
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Simple ODE Function: {'PASSED' if test1_passed else 'FAILED'}")
    
    print("Precision tests:")
    for dtype, passed in test2_results.items():
        print(f"  {dtype}: {'PASSED' if passed else 'FAILED'}")
    
    # Overall result
    overall_passed = test1_passed and all(test2_results.values())
    print(f"\nOverall: {'PASSED' if overall_passed else 'FAILED'}")
    
    return overall_passed


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
