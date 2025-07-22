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
    @custom_bwd(device_type='cuda')
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
        dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        
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
        
        # Preallocate buffer for efficiency
        y_buffer = torch.zeros_like(yt[0])
        
        # Backward pass loop with dynamic scaling
        with torch.no_grad():
            for i in reversed(range(N - 1)):
                dti = t[i + 1] - t[i]
                
                # Prepare current state
                y_buffer.data.copy_(yt[i])
                y = y_buffer.detach().requires_grad_(True)
                
                # Prepare time variables
                ti = t[i].clone().detach()
                dti_local = dti.clone().detach()
                if t.requires_grad:
                    ti.requires_grad_(True)
                    dti_local.requires_grad_(True)
                
                # Rebuild computational graph
                with torch.enable_grad():
                    dy = increment_func(ode_func, y, ti, dti_local)
                
                # Dynamic scaling loop
                attempts = 0
                while attempts < scaler.max_attempts:
                    # Check for overflow in scaled gradients
                    if scaler._is_any_infinite((scaler.S * a)):
                        scaler.update_on_overflow()
                        continue
                    
                    # Compute gradients with scaling
                    if t.requires_grad:
                        grads = torch.autograd.grad(
                            dy, (y, ti, dti_local, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, gti, gdti, *dparams = grads
                        
                        # Handle None gradients
                        gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                        gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti)
                        gdti2 = torch.sum(scaler.S * a * dy, dim=-1)
                    else:
                        grads = torch.autograd.grad(
                            dy, (y, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, *dparams = grads
                        gti = gdti = gdti2 = None
                        
                        # Handle None gradients for parameters
                        dparams = [d if d is not None else torch.zeros_like(p) 
                                  for d, p in zip(dparams, params)]
                    
                    # Check for overflow in computed gradients
                    if scaler._is_any_infinite((da, gti, gdti, dparams)):
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
                
                # Update gradients with descaling
                a = a + (dti / scaler.S) * da.to(dtype_hi) + at[i].to(dtype_hi)
                grad_theta = [g + (dti / scaler.S) * d.to(g.dtype) 
                             for g, d in zip(grad_theta, dparams)]
                
                if grad_t is not None:
                    grad_t[i] = grad_t[i] + (dti / scaler.S) * (gti - gdti) - (gdti2.to(dtype_hi)) / scaler.S
                    grad_t[i + 1] = grad_t[i + 1] + (dti / scaler.S) * gdti + gdti2.to(dtype_hi) / scaler.S
                
                # Check for overflow in accumulated gradients
                if scaler._is_any_infinite((a, grad_t, grad_theta)):
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