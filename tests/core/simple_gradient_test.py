#!/usr/bin/env python3
"""
Simple test to verify the gradient checking functionality works.
This is a standalone script that doesn't depend on the full STL10 model.
"""

import torch
import math

def gradient_check_simple():
    """Simple gradient check for a basic function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Simple quadratic function: f(x) = sum(x^2)
    x = torch.randn(5, device=device, dtype=dtype, requires_grad=True)
    
    def func():
        return torch.sum(x ** 2)
    
    # Analytical gradient
    output = func()
    output.backward()
    analytical_grad = x.grad.clone() if x.grad is not None else torch.zeros_like(x)
    
    # Numerical gradient
    eps = 1e-8
    if x.grad is not None:
        x.grad.zero_()
    numerical_grad = torch.zeros_like(x)
    
    for i in range(x.numel()):
        # Positive perturbation
        x.data[i] += eps
        f_plus = func()
        
        # Negative perturbation  
        x.data[i] -= 2 * eps
        f_minus = func()
        
        # Restore
        x.data[i] += eps
        
        # Compute numerical gradient
        numerical_grad[i] = (f_plus - f_minus) / (2 * eps)
    
    # Compare
    error = torch.max(torch.abs(analytical_grad - numerical_grad))
    rel_error = error / torch.max(torch.abs(analytical_grad))
    
    print(f"Analytical gradient: {analytical_grad}")
    print(f"Numerical gradient:  {numerical_grad}")
    print(f"Max absolute error:  {error:.2e}")
    print(f"Max relative error:  {rel_error:.2e}")
    
    # Check if gradient is correct (should be 2*x)
    expected_grad = 2 * x.data
    expected_error = torch.max(torch.abs(analytical_grad - expected_grad))
    print(f"Expected gradient:   {expected_grad}")
    print(f"Error vs expected:   {expected_error:.2e}")
    
    if error < 1e-6:
        print("✓ Gradient check PASSED")
        return True
    else:
        print("✗ Gradient check FAILED")
        return False

if __name__ == "__main__":
    gradient_check_simple()
