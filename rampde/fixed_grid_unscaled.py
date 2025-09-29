"""
Unscaled fixed grid ODE solver - optimal performance variant.

This variant provides the fastest performance by eliminating all scaling
infrastructure. It should be used as the default for float32 and bfloat16
precision where overflow is not a concern.

Performance: Optimal performance baseline - significantly faster than variants
with scaling or exception handling overhead.
"""

from typing import Any, Optional, Tuple
import torch
from torch.amp import autocast
from .fixed_grid_base import FixedGridODESolverBase

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverUnscaled(FixedGridODESolverBase):
    """
    Unscaled fixed grid ODE solver for optimal performance.
    
    This variant eliminates all scaling infrastructure to provide the fastest
    possible performance. It performs simple gradient computation without:
    - Scaling loops
    - Parameter dtype conversion
    - Overflow checking
    - Exception handling
    
    Use this variant when:
    - Precision is float32 or bfloat16
    - No overflow concerns
    - Maximum performance is needed
    """

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Unscaled backward pass - optimal performance.
        
        This implementation provides the fastest backward pass by eliminating
        all scaling infrastructure. It performs direct gradient computation
        without any overflow protection or scaling loops.
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients: (None, None, grad_y0, grad_t, None, *grad_params)
        """
        # Retrieve saved tensors and context
        yt, *params = ctx.saved_tensors
        increment_func = ctx.increment_func
        ode_func = ctx.ode_func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        
        # Determine precision
        dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Backward pass loop - no scaling, no exceptions
        with torch.no_grad():
            for i in reversed(range(N - 1)):
                dti = t[i + 1] - t[i]
                
                # Prepare current state - directly from saved tensor
                y = yt[i].detach().requires_grad_(True)
                
                # Prepare time variables - no unnecessary cloning
                ti = t[i].detach()
                dti_local = dti.detach()
                if t.requires_grad:
                    ti.requires_grad_(True)
                    dti_local.requires_grad_(True)
                
                # Rebuild computational graph
                with torch.enable_grad():
                    dy = increment_func(ode_func, y, ti, dti_local)
                
                # Compute gradients - optimized for different cases
                if t.requires_grad and any_param_requires_grad:
                    # Full gradient computation
                    grads = torch.autograd.grad(
                        dy, (y, ti, dti_local, *params), a,
                        create_graph=False, allow_unused=True
                    )
                    da, gti, gdti, *dparams = grads
                    
                    # Handle None gradients
                    gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                    gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti)
                    gdti2 = torch.sum(a * dy, dim=-1)
                elif t.requires_grad:
                    # Only time gradients needed
                    grads = torch.autograd.grad(
                        dy, (y, ti, dti_local), a,
                        create_graph=False, allow_unused=True
                    )
                    da, gti, gdti = grads
                    dparams = [torch.zeros_like(p) for p in params]
                    
                    # Handle None gradients
                    gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                    gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti)
                    gdti2 = torch.sum(a * dy, dim=-1)
                elif any_param_requires_grad:
                    # Only parameter gradients needed
                    grads = torch.autograd.grad(
                        dy, (y, *params), a,
                        create_graph=False, allow_unused=True
                    )
                    da, *dparams = grads
                    gti = gdti = gdti2 = None
                    
                    # Handle None gradients for parameters
                    dparams = [d if d is not None else torch.zeros_like(p) 
                              for d, p in zip(dparams, params)]
                else:
                    # Only adjoint gradient needed
                    da = torch.autograd.grad(dy, y, a, create_graph=False)[0]
                    dparams = [torch.zeros_like(p) for p in params]
                    gti = gdti = gdti2 = None
                
                # Update gradients - optimized with in-place operations
                # Convert da once and reuse
                da_hi = da.to(dtype_hi)
                a.add_(dti * da_hi).add_(at[i].to(dtype_hi))
                
                if any_param_requires_grad:
                    # Use in-place operations for parameter gradient accumulation
                    for g, d in zip(grad_theta, dparams):
                        if d is not None:
                            g.add_(dti * d.to(g.dtype))
                
                if grad_t is not None:
                    gdti2_hi = gdti2.to(dtype_hi)
                    grad_t[i].add_(dti * (gti - gdti)).sub_(gdti2_hi)
                    grad_t[i + 1].add_(dti * gdti).add_(gdti2_hi)
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, y0, t, loss_scaler, *params)
        return (None, None, a, grad_t, None, *grad_theta)