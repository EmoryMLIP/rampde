from torch.amp import custom_fwd, custom_bwd, autocast
import torch
from .loss_scalers import DynamicScaler

class Euler(torch.nn.Module):
    order = 1
    name = 'euler'

    def forward(self, func, y, t, dt):
        dy =  func(t, y)
        return dy

_one_sixth= 1/6
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


class FixedGridODESolver(torch.autograd.Function):
    """Mixed-precision adjoint on a fixed time grid, with optional predictive scaling for faster backward pass.

    Parameters
    ----------
    strategy : {"default", "predictive"}
        * ``"default"`` – check for over/underflow and scale the output
        * ``"predictive"`` – k-periodic scaling using a Lipschitz constant estimate
    k : int, default 3
        Period (in time‐steps) for the predictive check.
    """

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, step, func, y0, t, loss_scaler,
                *params, strategy: str = "default", k: int = 3):
        if strategy not in {"default", "predictive"}:
            raise ValueError("strategy must be 'default' or 'predictive'.")

        with torch.no_grad():
            dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
            dtype_hi = y0.dtype
            N = t.shape[0]
            y = y0
            yt = torch.zeros(N, *y.shape, dtype=dtype_low, device=y.device)
            yt[0] = y0.to(dtype_low)

            for i in range(N - 1):
                dt = t[i + 1] - t[i]
                with autocast(device_type='cuda', dtype=dtype_low):
                    dy = step(func, y, t[i], dt)
                with autocast(device_type='cuda', enabled=False):
                    y = y + dt* dy
                yt[i + 1] = y.to(dtype_low)

        ctx.save_for_backward(yt, *params)
        ctx.step = step
        ctx.func = func
        ctx.t = t
        ctx.dtype_hi = dtype_hi
        ctx.loss_scaler = loss_scaler
        ctx.strategy = strategy
        ctx.k_period = k
        return yt

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, at):
        yt, *params = ctx.saved_tensors
        step = ctx.step
        func = ctx.func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
        dtype_t = t.dtype
        strategy = ctx.strategy
        k_period = ctx.k_period

        N = t.shape[0]
        params = tuple(params)

        # Initialize the dynamic scaler and scale the output adjoints
        scaler = ctx.loss_scaler
        if scaler.S is None:
            scaler.init_scaling(at[-1])

        a = at[-1].to(dtype_hi)
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)

        old_params = {name: param.data.clone() for name, param in func.named_parameters()}
        for name, param in func.named_parameters():
            param.data = param.data.to(dtype_low)

        # Constants for predictive scaling
        Mmax = 1.0 / torch.finfo(torch.float16).eps
        Mmin = torch.finfo(torch.float16).eps
        counter = 0

        y_buffer = torch.zeros_like(yt[0])

        with torch.no_grad():
            for i in reversed(range(N - 1)):
                dti = t[i + 1] - t[i]
                y_buffer.data.copy_(yt[i])
                y = y_buffer.detach().requires_grad_(True)

                ti = t[i].clone().detach()
                dti_local = dti.clone().detach()
                if t.requires_grad:
                    ti.requires_grad_(True)
                    dti_local.requires_grad_(True)

                with torch.enable_grad():
                    # rebuild computational graph for the current time step
                    dy = step(func, y, ti, dti_local)


                attempts = 0
                while attempts < scaler.max_attempts:

                    if strategy == "predictive":
                        counter += 1

                        # Only perform the check every k_period steps
                        if counter % k_period == 0:
                            # Generate a random unit vector v in the tangent space of y
                            v = torch.randn_like(y)
                            v = v / v.norm(p=2).clamp(min=Mmin)

                            # Compute Jv = ∂(dy)/∂y @ v
                            # We already have dy as a function of y, so backpropagate from dy w.r.t. y
                            Jv = torch.autograd.grad(dy, y, v, retain_graph=True, allow_unused=True)[0]
                            # Estimate Lipschitz constant L_j ≈ ||Jv||₂
                            Lj = Jv.norm(p=2)

                            # Predict the growth of the adjoint over the next k_period – 1 steps
                            pred = a.norm() * ((1 + Lj * dti) ** (k_period - 1))

                            if pred > Mmax:
                                # Overflow predicted: down‐scale
                                scaler.update_on_overflow()
                                # Since S decreased, rescale existing parameter gradients
                                # grad_theta = [g * scaler.decrease_factor for g in grad_theta]
                                # if grad_t is not None:
                                #     grad_t *= scaler.decrease_factor

                            elif pred < Mmin:
                                # Underflow predicted: up‐scale
                                scaler.update_on_small_grad()
                            #     grad_theta = [g * scaler.increase_factor for g in grad_theta]
                            #     if grad_t is not None:
                            #         grad_t *= scaler.increase_factor
                    # else:
                        # For the default strategy, we only check for overflow/underflow
                    if scaler._is_any_infinite((scaler.S * a)):
                        scaler.update_on_overflow()
                        attempts += 1
                        continue

                    if t.requires_grad:
                        grads = torch.autograd.grad(
                            dy, (y, ti, dti_local, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, gti, gdti, *dparams = grads
                        gti = gti.to(dtype_hi) if gti is not None else torch.zeros_like(ti)
                        gdti = gdti.to(dtype_hi) if gdti is not None else torch.zeros_like(dti_local.clone())
                        gdti2 = torch.sum(scaler.S * a * dy, dim=-1)
                    else:
                        grads = torch.autograd.grad(
                            dy, (y, *params), scaler.S * a,
                            create_graph=True, allow_unused=True
                        )
                        da, *dparams = grads
                        gti = gdti = gdti2 = None

                    dparams = [d if d is not None else torch.zeros_like(p) for d, p in zip(dparams, params)]

                    if scaler._is_any_infinite((da, gti, gdti, dparams)):
                        scaler.update_on_overflow()
                        attempts += 1
                        continue
                    else:
                        break

                if attempts >= scaler.max_attempts:
                    raise RuntimeError(f"Reached maximum number of attempts in backward pass at time step i={i}")

                a = a + (dti/scaler.S) * da.to(dtype_hi) + at[i].to(dtype_hi)
                grad_theta = [g + (dti/scaler.S) * d.to(g.dtype) for g, d in zip(grad_theta, dparams)]

                if grad_t is not None:
                    grad_t[i] = grad_t[i] + (dti/scaler.S) * (gti - gdti) - (gdti2.to(dtype_hi))/scaler.S
                    grad_t[i + 1] = grad_t[i + 1] + (dti/scaler.S) * gdti + gdti2.to(dtype_hi)/scaler.S

                if scaler._is_any_infinite((a, grad_t, grad_theta)):
                    raise RuntimeError(f"Gradients are not representable at time step i={i}. ")

                # Adjust upward scaling if the norm is too small
                if attempts == 0 and scaler.check_for_increase(a):
                    if strategy == "default":
                        scaler.update_on_small_grad()

        for name, param in func.named_parameters():
            param.data = old_params[name].data

        return (None, None, a, grad_t, None, *grad_theta)
