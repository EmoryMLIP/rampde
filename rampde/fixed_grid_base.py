"""
Base class for fixed grid ODE solvers.

This module provides the shared forward pass implementation that is identical
across all fixed grid solver variants. Only the backward pass differs between
variants to handle different scaling and exception handling strategies.
"""

from typing import Any, Optional, Tuple, Union
import torch
from torch.amp import autocast

# Import custom_fwd and custom_bwd from torch.cuda.amp
try:
    from torch.amp import custom_fwd, custom_bwd
except ImportError:
    from torch.cuda.amp import custom_fwd, custom_bwd


class FixedGridODESolverBase(torch.autograd.Function):
    """
    Base class for fixed grid ODE solvers with shared forward pass.
    
    This class implements the forward pass that is identical across all variants:
    - Unscaled (optimal performance)
    - Dynamic (with scaling loop)
    - Unscaled Safe (with exception handling)
    
    Subclasses only need to implement the backward pass according to their
    specific scaling and exception handling strategy.
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float16)
    def forward(
        ctx: Any, 
        increment_func: torch.nn.Module, 
        ode_func: torch.nn.Module, 
        y0: torch.Tensor, 
        t: torch.Tensor, 
        loss_scaler: Any, 
        *params: torch.Tensor
    ) -> torch.Tensor:
        """
        Shared forward pass implementation.
        
        This method is identical across all solver variants and implements
        the fixed grid forward integration using the specified increment function.
        
        Args:
            ctx: PyTorch autograd context for saving information for backward pass
            increment_func: Increment function (Euler, RK4, etc.)
            ode_func: ODE function f(t, y)
            y0: Initial condition tensor
            t: Time points tensor
            loss_scaler: Loss scaler for mixed precision (DynamicScaler or NoScaler)
            *params: Parameters of the ODE function
            
        Returns:
            yt: Solution tensor at all time points
        """
        with torch.no_grad():
            # Determine precision levels
            dtype_hi = y0.dtype
            dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else dtype_hi
            
            # Initialize solution storage
            N = t.shape[0]
            y = y0
            yt = torch.zeros(N, *y.shape, dtype=dtype_low, device=y.device)
            yt[0] = y0.to(dtype_low)
            
            # Forward integration loop
            for i in range(N - 1):
                dt = t[i + 1] - t[i]
                
                # Compute increment in low precision
                with autocast(device_type='cuda', dtype=dtype_low):
                    dy = increment_func(ode_func, y, t[i], dt)
                
                # Update solution in high precision, then convert to low precision
                with autocast(device_type='cuda', enabled=False):
                    y = y + dt * dy
                
                yt[i + 1] = y.to(dtype_low)
        
        # Save information for backward pass
        ctx.save_for_backward(yt, *params)
        ctx.increment_func = increment_func
        ctx.ode_func = ode_func
        ctx.t = t
        ctx.dtype_hi = dtype_hi
        ctx.loss_scaler = loss_scaler
        
        return yt
    
    @staticmethod
    def backward(ctx: Any, at: torch.Tensor) -> Tuple[Optional[torch.Tensor], ...]:
        """
        Abstract backward method - must be implemented by subclasses.
        
        Each subclass implements this method according to its specific
        scaling and exception handling strategy:
        - Unscaled: Simple, fast backward pass
        - Dynamic: Backward pass with scaling loop
        - Unscaled Safe: Backward pass with exception handling
        
        Args:
            ctx: PyTorch autograd context with saved tensors and attributes
            at: Gradient tensor from subsequent operations
            
        Returns:
            Tuple of gradients for all inputs to forward pass
        """
        raise NotImplementedError(
            "Subclasses must implement the backward method according to their "
            "specific scaling and exception handling strategy."
        )