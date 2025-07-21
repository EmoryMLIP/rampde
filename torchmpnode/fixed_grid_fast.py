"""
Fixed Grid ODE Solver - Fast Version (High Memory Usage)

This version stores the dy values with full computational graph during forward pass
to avoid recomputation during backward pass. This trades memory for speed.
"""

from torch.amp import autocast, custom_fwd, custom_bwd
import torch
from .loss_scalers import DynamicScaler


class Euler(torch.nn.Module):
    order = 1
    name = 'euler'

    def forward(self, func, y, t, dt):
        return func(t, y)


_one_sixth = 1/6


class RK4(torch.nn.Module):
    order = 4
    name = 'rk4'

    def forward(self, func, y, t, dt):
        half_dt = dt * 0.5
        k1 = func(t, y)
        k2 = func(t + half_dt, y + k1*half_dt)
        k3 = func(t + half_dt, y + k2*half_dt)
        k4 = func(t + dt, y + k3*dt)
        return (k1 + 2*(k2 + k3) + k4)*_one_sixth


class FixedGridODESolverFast(torch.autograd.Function):
    """
    Fast version that stores dy values during forward pass to avoid recomputation.
    Uses more memory but is computationally faster during backward pass.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, step, func, y0, t, loss_scaler, *params):
        dtype_low = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else torch.float32
        dtype_hi = y0.dtype
        N = t.shape[0]
        
        # Initialize trajectory storage
        yt = torch.zeros(N, *y0.shape, dtype=dtype_low, device=y0.device)
        yt[0] = y0.to(dtype_low)
        
        # Storage for dy values with gradients - this is the key difference
        dy_stored = []
        y_for_grad = []  # Store y values that were used to compute dy
        t_used = []      # Store t values used
        dt_used = []     # Store dt values used
        
        # Start with y0 but detach for trajectory computation
        y_traj = y0.detach()  # For trajectory computation (no gradients)
        
        for i in range(N - 1):
            dt = t[i + 1] - t[i]
            ti = t[i]
            
            # Create a version of y for gradient computation
            # This y will have gradients and will be used to compute dy
            y_grad = yt[i].to(dtype_hi).detach().requires_grad_(True)
            
            # Compute dy with full gradients enabled
            with autocast(device_type='cuda', dtype=dtype_low):
                dy = step(func, y_grad, ti, dt)
            
            # Store everything we need for backward pass
            dy_stored.append(dy)  # dy with full computational graph
            y_for_grad.append(y_grad)  # y that was used to compute dy
            t_used.append(ti)
            dt_used.append(dt)
            
            # Update trajectory using detached dy to avoid storing unnecessary gradients
            with autocast(device_type='cuda', enabled=False):
                y_traj = y_traj + dt * dy.detach()
            yt[i + 1] = y_traj.to(dtype_low)

        # Save everything for backward pass
        ctx.save_for_backward(yt, t, *params, *dy_stored, *y_for_grad, *t_used, *dt_used)
        ctx.step = step
        ctx.func = func
        ctx.dtype_hi = dtype_hi
        ctx.dtype_low = dtype_low
        ctx.loss_scaler = loss_scaler
        ctx.N = N

        return yt
    
    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_outputs):
        """
        Backward pass using stored dy values - no recomputation needed!
        """
        at = grad_outputs  # Standard naming for autograd function
        N = ctx.N
        step = ctx.step
        func = ctx.func
        dtype_hi = ctx.dtype_hi
        dtype_low = ctx.dtype_low
        
        # Unpack saved tensors
        # Order: yt, t, *params, *dy_stored, *y_for_grad, *t_used, *dt_used
        saved_tensors = ctx.saved_tensors
        yt = saved_tensors[0]
        t = saved_tensors[1]
        
        # Count parameters
        n_params = len([p for p in func.parameters()])
        
        # Extract sections
        params = saved_tensors[2:2+n_params]
        dy_stored = saved_tensors[2+n_params:2+n_params+(N-1)]
        y_for_grad = saved_tensors[2+n_params+(N-1):2+n_params+2*(N-1)]
        t_used = saved_tensors[2+n_params+2*(N-1):2+n_params+3*(N-1)]
        dt_used = saved_tensors[2+n_params+3*(N-1):2+n_params+4*(N-1)]

        # Initialize the dynamic scaler
        scaler = ctx.loss_scaler
        if scaler.S is None:
            scaler.init_scaling(at[-1])
        
        # Initialize gradients
        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)

        # Temporarily convert parameters to low precision for consistency
        old_params = {}
        for name, param in func.named_parameters():
            old_params[name] = param.data.clone()
            param.data = param.data.to(dtype_low)
            
        with torch.no_grad():
            # Backward pass through time - using stored dy values
            for i in reversed(range(N - 1)):
                # Get stored values for this time step
                dy = dy_stored[i]  # This has the full computational graph!
                y_grad = y_for_grad[i]
                ti = t_used[i]
                dti = dt_used[i]
                
                # Initialize variables for this time step
                da = None
                dparams = []
                gti = None
                gdti = None
                
                attempts = 0
                while attempts < scaler.max_attempts:
                    if scaler._is_any_infinite((scaler.S*a)):
                        scaler.update_on_overflow()
                        continue

                    # The key improvement: use stored dy instead of recomputing!
                    # Compute gradients directly from stored dy
                    if t.requires_grad:
                        # Need gradients w.r.t. y, t, and parameters
                        grads = torch.autograd.grad(
                            dy, (y_grad, ti, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, gti, *dparams = grads
                        
                        # Handle time gradients
                        gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                        gdti = torch.sum(scaler.S * a * dy, dim=-1)  # Gradient w.r.t. dt
                        
                    else:
                        # Only need gradients w.r.t. y and parameters
                        grads = torch.autograd.grad(
                            dy, (y_grad, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, *dparams = grads
                        
                    # Handle None gradients
                    dparams = [d if d is not None else torch.zeros_like(p) for d, p in zip(dparams, params)]
                    da = da if da is not None else torch.zeros_like(y_grad)
                        
                    # Check for infinite gradients
                    if scaler._is_any_infinite((da, gti, gdti, dparams)):
                        scaler.update_on_overflow()
                        attempts += 1
                        continue
                    else:
                        break
                
                if attempts >= scaler.max_attempts:
                    raise RuntimeError(f"Reached maximum number of {scaler.max_attempts} attempts in backward pass at time step i={i}")

                # Update adjoint state and gradients
                da_term = (dti/scaler.S) * da.to(dtype_hi) if da is not None else 0
                a = a + da_term + at[i].to(dtype_hi)
                grad_theta = [g + (dti/scaler.S) * d.to(g.dtype) for g, d in zip(grad_theta, dparams)]
                
                # Update time gradients if needed
                if grad_t is not None:
                    if gti is not None:
                        grad_t[i] = grad_t[i] + (dti/scaler.S) * gti
                    if gdti is not None:
                        grad_t[i + 1] = grad_t[i + 1] + gdti.to(dtype_hi)/scaler.S
                
                # Check for infinite gradients
                if scaler._is_any_infinite((a, grad_t, grad_theta)):
                    raise RuntimeError(f"Gradients are not representable at time step i={i}.")

                # Check for scaling updates
                if attempts == 0 and scaler.check_for_increase(a):
                    scaler.update_on_small_grad()

        # Restore original parameter precision
        for name, param in func.named_parameters():
            param.data = old_params[name]

        return (None, None, a, grad_t, None, *grad_theta)


def odeint_fixed_grid_fast(func, y0, t, method='euler', loss_scaler=None, **kwargs):
    """
    Fast fixed grid ODE solver that trades memory for speed.
    
    This version stores dy values during forward pass to avoid recomputation
    during backward pass. Uses more memory but is computationally faster.
    
    Args:
        func: ODE function dy/dt = func(t, y)
        y0: Initial condition
        t: Time points where solution is desired
        step: Integration method ('euler' or 'rk4')
        loss_scaler: Dynamic loss scaler for mixed precision
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Tensor of shape (len(t), *y0.shape) containing the solution
    """
    
    if loss_scaler is None:
        loss_scaler = DynamicScaler(dtype_low=torch.float16)  # Default to float16 for mixed precision
    
    # Select integration method
    if method == 'euler':
        stepper = Euler()
    elif method == 'rk4':
        stepper = RK4()
    else:
        raise ValueError(f"Unknown step method: {method}")

    # Get parameters for gradient computation
    params = tuple(func.parameters())
    
    # Solve ODE using fast method
    return FixedGridODESolverFast.apply(stepper, func, y0, t, loss_scaler, *params)


# Alias for compatibility
odeint = odeint_fixed_grid_fast
