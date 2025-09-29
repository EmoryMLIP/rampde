"""
Test dtype preservation without autocast.

This test verifies that rampde correctly preserves data types (float32, float64, bfloat16)
throughout ODE computations when autocast is not used. It ensures that all intermediates
maintain the expected precision.
"""

import unittest
import torch
import torch.nn as nn
from rampde import odeint


class SimpleDtypeCheckingODE(nn.Module):
    """ODE that checks dtype of all intermediates."""
    def __init__(self, dim=10, target_dtype=torch.float32):
        super().__init__()
        self.target_dtype = target_dtype
        self.dim = dim
        
        # Initialize weights in target dtype
        self.W1 = nn.Parameter(torch.randn(64, dim, dtype=target_dtype) * 0.1)
        self.b1 = nn.Parameter(torch.zeros(64, dtype=target_dtype))
        self.W2 = nn.Parameter(torch.randn(dim, 64, dtype=target_dtype) * 0.1)
        self.b2 = nn.Parameter(torch.zeros(dim, dtype=target_dtype))
        
    def forward(self, t, y):
        # Check input dtypes
        assert t.dtype == self.target_dtype, f"t has dtype {t.dtype}, expected {self.target_dtype}"
        assert y.dtype == self.target_dtype, f"y has dtype {y.dtype}, expected {self.target_dtype}"
        
        # First layer
        h = torch.matmul(y, self.W1.t()) + self.b1
        assert h.dtype == self.target_dtype, f"Hidden layer has dtype {h.dtype}, expected {self.target_dtype}"
        
        # ReLU
        h = torch.relu(h)
        assert h.dtype == self.target_dtype, f"After ReLU has dtype {h.dtype}, expected {self.target_dtype}"
        
        # Second layer
        out = torch.matmul(h, self.W2.t()) + self.b2
        assert out.dtype == self.target_dtype, f"Output has dtype {out.dtype}, expected {self.target_dtype}"
        
        return out


class TestDtypePreservation(unittest.TestCase):
    
    def _test_dtype_preservation(self, dtype, device='cpu'):
        """Helper to test dtype preservation for a specific dtype."""
        # Skip if dtype not supported on device
        if device == 'cuda' and not torch.cuda.is_available():
            self.skipTest("CUDA not available")
        if dtype == torch.bfloat16 and device == 'cuda' and not torch.cuda.is_bf16_supported():
            self.skipTest("bfloat16 not supported on this GPU")
            
        # Create ODE with target dtype
        func = SimpleDtypeCheckingODE(dim=10, target_dtype=dtype).to(device)
        
        # Create inputs with target dtype
        y0 = torch.randn(10, dtype=dtype, device=device)
        t = torch.linspace(0, 1, 10, dtype=dtype, device=device)
        
        # Run ODE solver - should maintain dtype throughout
        sol = odeint(func, y0, t, method='rk4', loss_scaler=False)
        
        # Check output dtype
        self.assertEqual(sol.dtype, dtype, f"Output dtype {sol.dtype} != {dtype}")
        
    def test_float32_cpu(self):
        """Test float32 preservation on CPU."""
        self._test_dtype_preservation(torch.float32, 'cpu')
        
    def test_float64_cpu(self):
        """Test float64 preservation on CPU."""
        self._test_dtype_preservation(torch.float64, 'cpu')
        
    def test_float32_cuda(self):
        """Test float32 preservation on CUDA."""
        self._test_dtype_preservation(torch.float32, 'cuda')
        
    def test_float64_cuda(self):
        """Test float64 preservation on CUDA."""
        self._test_dtype_preservation(torch.float64, 'cuda')
        
    def test_bfloat16_cuda(self):
        """Test bfloat16 preservation on CUDA."""
        self._test_dtype_preservation(torch.bfloat16, 'cuda')
        
    def test_float16_cuda(self):
        """Test float16 preservation on CUDA."""
        self._test_dtype_preservation(torch.float16, 'cuda')
        
    def test_float64_gradients(self):
        """Test that gradients preserve float64."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        func = SimpleDtypeCheckingODE(dim=10, target_dtype=torch.float64).cuda()
        
        y0 = torch.randn(10, dtype=torch.float64, device='cuda', requires_grad=True)
        t = torch.linspace(0, 1, 10, dtype=torch.float64, device='cuda')
        
        sol = odeint(func, y0, t, method='rk4', loss_scaler=False)
        loss = sol[-1].sum()
        loss.backward()
        
        # Check gradient dtypes
        self.assertEqual(y0.grad.dtype, torch.float64, "y0 gradient should be float64")
        self.assertEqual(func.W1.grad.dtype, torch.float64, "W1 gradient should be float64")
        self.assertEqual(func.W2.grad.dtype, torch.float64, "W2 gradient should be float64")


    def test_float16_with_dynamic_scaler(self):
        """Test float16 with DynamicScaler (uses FixedGridODESolverDynamic)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        from rampde.loss_scalers import DynamicScaler
        
        # Create ODE with float16
        func = SimpleDtypeCheckingODE(dim=10, target_dtype=torch.float16).cuda()
        
        # Create float16 inputs with requires_grad for gradient computation
        y0 = torch.randn(10, dtype=torch.float16, device='cuda', requires_grad=True)
        t = torch.linspace(0, 1, 10, dtype=torch.float16, device='cuda')
        
        # Create dynamic scaler explicitly
        scaler = DynamicScaler(dtype_low=torch.float16)
        
        # Run with dynamic scaler - should use FixedGridODESolverDynamic
        sol = odeint(func, y0, t, method='rk4', loss_scaler=scaler)
        
        # Check output dtype
        self.assertEqual(sol.dtype, torch.float16, f"Output dtype {sol.dtype} != float16")
        
        # Compute gradients to trigger scaler usage
        loss = sol[-1].sum()
        loss.backward()
        
        # Check that gradients were computed (scaler functionality was used)
        self.assertIsNotNone(y0.grad, "Gradients should be computed")
        self.assertEqual(y0.grad.dtype, torch.float16, "y0 gradient should be float16")
        self.assertIsNotNone(func.W1.grad, "W1 gradients should be computed")
        self.assertEqual(func.W1.grad.dtype, torch.float16, "W1 gradient should be float16")
        
    def test_float16_no_scaler(self):
        """Test float16 without scaler (uses FixedGridODESolverUnscaledSafe)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        # Create ODE with float16
        func = SimpleDtypeCheckingODE(dim=10, target_dtype=torch.float16).cuda()
        
        # Create float16 inputs with requires_grad for gradient computation
        y0 = torch.randn(10, dtype=torch.float16, device='cuda', requires_grad=True)
        t = torch.linspace(0, 1, 10, dtype=torch.float16, device='cuda')
        
        # Run without scaler - should use FixedGridODESolverUnscaledSafe for float16
        sol = odeint(func, y0, t, method='rk4', loss_scaler=False)
        
        # Check output dtype
        self.assertEqual(sol.dtype, torch.float16, f"Output dtype {sol.dtype} != float16")
        
        # Compute gradients to test unscaled safe solver behavior
        loss = sol[-1].sum()
        loss.backward()
        
        # Check that gradients were computed (no scaler functionality was used)
        self.assertIsNotNone(y0.grad, "Gradients should be computed")
        self.assertEqual(y0.grad.dtype, torch.float16, "y0 gradient should be float16")
        self.assertIsNotNone(func.W1.grad, "W1 gradients should be computed")
        self.assertEqual(func.W1.grad.dtype, torch.float16, "W1 gradient should be float16")
        
    def test_solver_selection(self):
        """Test that correct solver variants are selected based on dtype and scaler."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")
            
        from rampde.odeint import _select_ode_solver
        from rampde.loss_scalers import DynamicScaler
        from rampde.fixed_grid_unscaled import FixedGridODESolverUnscaled
        from rampde.fixed_grid_dynamic import FixedGridODESolverDynamic
        from rampde.fixed_grid_unscaled_safe import FixedGridODESolverUnscaledSafe
        
        # Test float64 - should use unscaled
        solver, _ = _select_ode_solver(None, torch.float64)
        self.assertEqual(solver, FixedGridODESolverUnscaled, "float64 should use unscaled solver")
        
        # Test float32 - should use unscaled
        solver, _ = _select_ode_solver(None, torch.float32)
        self.assertEqual(solver, FixedGridODESolverUnscaled, "float32 should use unscaled solver")
        
        # Test bfloat16 - should use unscaled
        solver, _ = _select_ode_solver(None, torch.bfloat16)
        self.assertEqual(solver, FixedGridODESolverUnscaled, "bfloat16 should use unscaled solver")
        
        # Test float16 with None - should create DynamicScaler and use dynamic solver
        solver, scaler = _select_ode_solver(None, torch.float16)
        self.assertEqual(solver, FixedGridODESolverDynamic, "float16 with None should use dynamic solver")
        self.assertIsInstance(scaler, DynamicScaler, "float16 with None should create DynamicScaler")
        
        # Test float16 with explicit False - should use unscaled safe
        solver, scaler = _select_ode_solver(False, torch.float16)
        self.assertEqual(solver, FixedGridODESolverUnscaledSafe, "float16 with False should use unscaled safe")
        self.assertIsNone(scaler, "float16 with False should have no scaler")
        
        # Test with DynamicScaler - should use dynamic solver
        scaler = DynamicScaler(dtype_low=torch.float16)
        solver, _ = _select_ode_solver(scaler, torch.float16)
        self.assertEqual(solver, FixedGridODESolverDynamic, "DynamicScaler should use dynamic solver")


if __name__ == '__main__':
    unittest.main()