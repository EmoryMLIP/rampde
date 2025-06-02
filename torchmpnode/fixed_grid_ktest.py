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
        # with autocast(device_type='cuda', enabled=False):
        return (k1 + 2*(k2 + k3) + k4)*_one_sixth


class FixedGridODESolver(torch.autograd.Function):
    """Mixed‑precision adjoint on a fixed time grid.

    Parameters
    ----------
    strategy : {"default", "predictive"}
        * ``"default"`` – check for over/underflow and scale the output
        * ``"predictive"`` – k‑periodic predictive scaling
    k : int, default 3
        Period for the predictive predictor.
    """

    @staticmethod
    @custom_fwd(device_type="cuda")
    def forward(ctx, step, func, y0, t, loss_scaler,
                *params, strategy: str = "predictive", k: int = 3):
        if strategy not in {"default", "predictive"}:
            raise ValueError("strategy must be 'default' or 'predictive'.")

        with torch.no_grad():
            dtype_low = (torch.get_autocast_dtype("cuda")
                         if torch.is_autocast_enabled() else torch.float32)
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
    @custom_bwd(device_type="cuda")
    def backward(ctx, at):

        yt, *params = ctx.saved_tensors

        step        = ctx.step
        func        = ctx.func
        t           = ctx.t
        dtype_hi    = ctx.dtype_hi
        dtype_low   = (torch.get_autocast_dtype("cuda")
                       if torch.is_autocast_enabled() else torch.float32)
        dtype_t     = t.dtype
        strategy    = ctx.strategy
        k_period    = ctx.k_period

        N = t.shape[0]
        params = tuple(params)

        # Initialize the dynamic scaler and scale the output adjoints
        scaler = ctx.loss_scaler
        if scaler.is_initialized==False:
            scaler.init_scaling(at[-1])
        
        at = scaler.scale(at)
        a = at[-1].to(dtype_hi)
        
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)

        old_params = {name: param.data.clone() for name, param in func.named_parameters()}
        for name, param in func.named_parameters():
            param.data = param.data.to(dtype_low)

        Mmax   = 1.0 / torch.finfo(torch.float16).eps
        Mmin   = torch.finfo(torch.float16).eps

        y_buffer  = torch.zeros_like(yt[0])
        counter = 0

        # --------------- main reversed loop ---------
        with torch.no_grad():
            for i in reversed(range(N - 1)):
                dti = t[i + 1] - t[i]
                y_buffer.data.copy_(yt[i])
                y = y_buffer.detach().requires_grad_(True)
                # y = yt[i].clone().detach().requires_grad_(True)
                attempts = 0
                # Attempt gradient computation until all values are finite
                while attempts < scaler.max_attempts:
                    ti = t[i].clone().detach()
                    dti_local = dti.clone().detach()
                    if t.requires_grad:
                        ti.requires_grad_(True)
                        dti_local.requires_grad_(True)
                        with torch.enable_grad():
                            dy = step(func, y, ti, dti_local)
                    
                            grads = torch.autograd.grad(
                                dy, (y, ti, dti_local, *params), a,
                                create_graph=True, allow_unused=True
                            )
                            da, gti, gdti, *dparams = grads
                            gdti2 = torch.sum(a * dy, dim=-1)
                    else: 
                        with torch.enable_grad():
                            dy = step(func, y, ti, dti_local)
                            grads = torch.autograd.grad(
                                dy, (y, *params), a,
                                create_graph=True, allow_unused=True
                            )
                            da, *dparams = grads
                            gti = gdti = gdti2 = None

                    dparams = [d if d is not None else torch.zeros_like(p)
                               for d, p in zip(dparams, params)]
                    if scaler._is_any_infinite((da, gti, gdti, dparams)):
                        scaler.update_on_overflow(a, at, grad_theta, grad_t,
                                                  gti, in_place=True)
                        attempts += 1
                        continue
                    break
                else:
                    raise RuntimeError(f"Max attempts exceeded at step {i}")

                update_attempts = 0
                while update_attempts < scaler.max_attempts:
                # ---- adjoint and gradients update ----
                    a = a + dti * da.to(dtype_hi) + at[i].to(dtype_hi)
                    grad_theta = [g + dti * d.to(g.dtype) for g, d in zip(grad_theta, dparams)]
                    # print(f'time steps: {i}, a: {a}, grad_theta: {grad_theta}')
                    if grad_t is not None:
                        grad_t = grad_t.clone()
                        gti = gti.clone() if gti is not None else torch.zeros_like(t[i])
                        gdti = gdti.clone() if gdti is not None else torch.zeros_like(t[i])
                        gdti2 = gdti2.clone() if gdti2 is not None else torch.zeros_like(t[i])
                        grad_t[i] = grad_t[i] + dti * (gti - gdti) - gdti2
                        grad_t[i + 1] = grad_t[i + 1] + dti * gdti + gdti2
                    else:
                        grad_t = None

                    if scaler.max_attempts > 1:
                        # ---- periodic scaling ----
                        counter += 1
                        if strategy == "predictive" and counter % k_period == 0:
                            # ||Jv|| / ||v||
                            v = torch.randn_like(y)
                            v = v / v.norm(p=2).clamp(min=Mmin)
                            Jv = torch.autograd.grad(dy, y, v,
                                                    retain_graph=True, allow_unused=True)[0]
                            Lj = Jv.norm(p=2)              # since ||v|| = 1
                            pred = a.norm() * ((1 + Lj * dti) ** (k_period - 1))
                            if pred > Mmax:
                                print(f"Overflow detected at step {i}, scaling down in pred.")

                                a, da = scaler.update_on_overflow(a, da)
                                grad_theta = [g * scaler.decrease_factor for g in grad_theta]
                                if grad_t is not None:
                                    grad_t *= scaler.decrease_factor
                            elif pred < Mmin:
                                print(f"Underflow detected at step {i}, scaling up in pred.")
                                a, da = scaler.update_on_small_grad(a, da)
                                grad_theta = [g * scaler.increase_factor for g in grad_theta]
                                if grad_t is not None:
                                    grad_t *= scaler.increase_factor


                    if scaler._is_any_infinite((a, grad_theta, grad_t)):
                        scaler.update_on_overflow(a, at, grad_theta, grad_t,
                                                in_place=True)
                        update_attempts += 1
                        if update_attempts >= scaler.max_attempts:
                            raise RuntimeError("Update overflow retry limit hit.")
                    else:
                        break

        for name, param in func.named_parameters():
            param.data = old_params[name].data

        # Unscale the gradients before returning
        scaler.unscale((a, grad_theta,grad_t),in_place=True)
        if t.requires_grad:
            grad_t = grad_t.to(dtype_t)

        return (None, None, a, grad_t, None, *grad_theta)
