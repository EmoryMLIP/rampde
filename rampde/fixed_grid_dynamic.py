"""
Dynamic scaling fixed grid ODE solver.

This variant includes dynamic scaling infrastructure to handle mixed precision
training with DynamicScaler. It includes scaling loops, parameter dtype conversion,
and overflow checking but no exception handling.

Performance: Moderate overhead compared to unscaled variant due to scaling loops
and overflow checking. Required when using DynamicScaler for mixed precision.
"""

from typing import Any, Optional, Tuple
import torch
from torch.amp import autocast
from .fixed_grid_base import FixedGridODESolverBase
from .utils import _is_any_infinite

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverDynamic(FixedGridODESolverBase):
    """
    Dynamic scaling fixed grid ODE solver.
    
    This variant includes dynamic scaling infrastructure to handle mixed precision
    training with DynamicScaler. It includes:
    - Scaling loops for overflow handling
    - Parameter dtype conversion
    - Overflow checking and scaler updates
    - No exception handling (uses RuntimeError on failure)
    
    Use this variant when:
    - DynamicScaler is being used
    - Mixed precision with float16
    - Dynamic scaling is required
    """

    @staticmethod
    @custom_bwd
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Dynamic scaling backward pass.
        
        This implementation includes dynamic scaling infrastructure to handle
        mixed precision training with DynamicScaler. It performs gradient
        computation with scaling loops and overflow checking.
        
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
        scaler = ctx.loss_scaler
        
        # Determine precision
        dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else dtype_hi
        
        # Initialize gradients
        N = t.shape[0]
        params = tuple(params)
        
        # Initialize the dynamic scaler
        if scaler.S is None:
            scaler.init_scaling(at[-1])
        
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        # Parameter dtype conversion for scaling
        old_params = {name: param.data.clone() for name, param in ode_func.named_parameters()}
        for name, param in ode_func.named_parameters():
            param.data = param.data.to(dtype_low)
        
        # Fast path check - skip parameter gradients if not needed
        any_param_requires_grad = any(p.requires_grad for p in params) if params else False
        
        # Backward pass loop with dynamic scaling
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
                
                # Dynamic scaling loop
                attempts = 0
                while attempts < scaler.max_attempts:
                    # Check for overflow in scaled gradients
                    if _is_any_infinite((scaler.S * a,)):
                        scaler.update_on_overflow()
                        attempts += 1
                        continue
                    
                    # Rebuild computational graph (moved inside loop for recomputation on scale change)
                    with torch.enable_grad():
                        dy = increment_func(ode_func, y, ti, dti_local)
                    
                    # Compute gradients with scaling - optimized for different cases
                    if t.requires_grad and any_param_requires_grad:
                        # Full gradient computation
                        grads = torch.autograd.grad(
                            dy, (y, ti, dti_local, *params), scaler.S * a,
                            create_graph=False, allow_unused=True
                        )
                        da, gti, gdti, *dparams = grads
                        
                        # Handle None gradients
                        gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                        gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti)
                        gdti2 = torch.sum(scaler.S * a * dy, dim=-1)
                    elif t.requires_grad:
                        # Only time gradients needed
                        grads = torch.autograd.grad(
                            dy, (y, ti, dti_local), scaler.S * a,
                            create_graph=False, allow_unused=True
                        )
                        da, gti, gdti = grads
                        dparams = [torch.zeros_like(p) for p in params]
                        
                        # Handle None gradients
                        gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                        gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti)
                        gdti2 = torch.sum(scaler.S * a * dy, dim=-1)
                    elif any_param_requires_grad:
                        # Only parameter gradients needed
                        grads = torch.autograd.grad(
                            dy, (y, *params), scaler.S * a,
                            create_graph=False, allow_unused=True
                        )
                        da, *dparams = grads
                        gti = gdti = gdti2 = None
                        
                        # Handle None gradients for parameters
                        dparams = [d if d is not None else torch.zeros_like(p) 
                                  for d, p in zip(dparams, params)]
                    else:
                        # Only adjoint gradient needed
                        da = torch.autograd.grad(dy, y, scaler.S * a, create_graph=False)[0]
                        dparams = [torch.zeros_like(p) for p in params]
                        gti = gdti = gdti2 = None
                    
                    # Check for overflow in computed gradients
                    if _is_any_infinite((da, gti, gdti, dparams)):
                        scaler.update_on_overflow()
                        attempts += 1
                        continue
                    else:
                        break
                
                # Check if we exceeded maximum attempts
                if attempts >= scaler.max_attempts:
                    raise RuntimeError(
                        f"Reached maximum number of {scaler.max_attempts} attempts "
                        f"in backward pass at time step i={i}"
                    )
                
                # Update gradients with descaling - optimized with in-place operations
                # Convert da once and reuse, compute scale factor once
                da_hi = da.to(dtype_hi)
                scale_factor = dti / scaler.S
                a.add_(scale_factor * da_hi).add_(at[i].to(dtype_hi))
                
                if any_param_requires_grad:
                    # Use in-place operations for parameter gradient accumulation
                    for g, d in zip(grad_theta, dparams):
                        if d is not None:
                            g.add_(scale_factor * d.to(g.dtype))
                
                if grad_t is not None:
                    gdti2_hi = gdti2.to(dtype_hi) / scaler.S
                    grad_t[i].add_(scale_factor * (gti - gdti)).sub_(gdti2_hi)
                    grad_t[i + 1].add_(scale_factor * gdti).add_(gdti2_hi)
                
                # Check for overflow in accumulated gradients
                if _is_any_infinite((a, grad_t, grad_theta)):
                    # print isfinite for a, grad_t, grad_theta
                    if not a.isfinite().all():
                        print(f"Gradient a is not finite at time step i={i}: {a}")
                    if grad_t is not None and not grad_t[i].isfinite().all():
                        print(f"Gradient grad_t[{i}] is not finite: {grad_t[i]}")
                    if grad_t is not None and not grad_t[i + 1].isfinite().all():
                        print(f"Gradient grad_t[{i + 1}] is not finite: {grad_t[i + 1]}")
                    if any(not g.isfinite().all() for g in grad_theta):
                        print(f"Gradient grad_theta is not finite at time step i={i}: {[g for g in grad_theta if not g.isfinite().all()]}")
                    raise RuntimeError(
                        f"Gradients are not representable at time step i={i}"
                    )
                
                # Adjust upward scaling if the norm is too small
                if attempts == 0 and scaler.check_for_increase(a):
                    scaler.update_on_small_grad()
        
        # Restore original parameter dtypes
        for name, param in ode_func.named_parameters():
            param.data = old_params[name].data
        
        # Return gradients for all inputs to forward pass
        # (increment_func, ode_func, y0, t, loss_scaler, *params)
        return (None, None, a, grad_t, None, *grad_theta)