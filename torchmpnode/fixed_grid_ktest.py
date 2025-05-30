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

    @staticmethod
    @custom_fwd(device_type='cuda')
    def forward(ctx, step, func, y0, t, loss_scaler, *params):
        with torch.no_grad():
            dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
            dtype_hi = y0.dtype
            N = t.shape[0]
            y = y0

            yt = torch.zeros(N, *y.shape, dtype=dtype_low, device=y.device)
            Ls = torch.zeros(N-1, dtype=torch.float32, device=y.device)
            yt[0] = y0.to(dtype_low)

            for i in range(N - 1):
                dt = t[i + 1] - t[i]
                with autocast(device_type='cuda', dtype=dtype_low):
                    dy = step(func, y, t[i], dt)

                num = dy.norm(p=2)
                den = y.norm(p=2).clamp(min=1e-12)
                Ls[i] = (num / den).to(torch.float32)

                with autocast(device_type='cuda', enabled=False):
                    y = y + dt* dy

                yt[i + 1] = y.to(dtype_low)

        ctx.save_for_backward(yt, Ls, *params)
        ctx.step = step
        ctx.func = func
        ctx.t = t
        ctx.dtype_hi = dtype_hi
        ctx.loss_scaler = loss_scaler

        return yt

    @staticmethod
    @custom_bwd(device_type='cuda')
    def backward(ctx, at):
        import os
        import csv

        yt, Ls, *params = ctx.saved_tensors
        step = ctx.step
        func = ctx.func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
        dtype_t = t.dtype

        N = t.shape[0]
        params = tuple(params)

        # --- set up logging ---
        log_file = 'backward_log_news.csv'
        if os.path.exists(log_file):
            with open(log_file, newline='') as f:
                reader = csv.reader(f)
                rows = list(reader)
            epoch = int(rows[-1][0]) + 1 if rows and rows[-1] else 0
            file_exists = True
        else:
            with open(log_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Step', 'da_mean', 'dparams_mean', 'grad_theta', 'initial_S'])
            epoch = 0
            file_exists = True

        scaler = ctx.loss_scaler
        if scaler.is_initialized==False:
            scaler.init_scaling(at[-1])
        at = scaler.scale(at)
        a = at[-1].to(dtype_hi)

        dt_vec = (t[1:] - t[:-1]).to(torch.float32)
        k = scaler.k  # fewer periodic rescale frequency

        if k > 1:
            with torch.no_grad():
                factors = 1.0 + Ls * dt_vec
                cumprod = torch.cumprod(factors, dim=0)
                F = float(cumprod.max().item())
                uN = float(a.abs().max().to(torch.float32).item())
                bmax = uN * F
                bmin = uN / F

            old_S = scaler.S
            scaler.tighten_scale_with_bound(bmax, bmin)
            k_scale = scaler.S / old_S
            if k_scale != 1.0:
                at = scaler._apply_scaling(at, k_scale)
                a = at[-1].to(dtype_hi)

        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)

        old_params = {name: param.data.clone() for name, param in func.named_parameters()}
        for name, param in func.named_parameters():
            param.data = param.data.to(dtype_low)

        y_buffer = torch.zeros_like(yt[0])
        Mmax = 1.0 / torch.finfo(torch.float16).eps
        Mmin = torch.finfo(torch.float16).eps #2.0**-14
        counter = 0
        with torch.no_grad():
            for i in reversed(range(N - 1)):
                dti = dt_vec[i]
                y_buffer.data.copy_(yt[i])
                y = y_buffer.detach().requires_grad_(True)
                attempts = 0

                # --- compute gradients with retry on overflow ---
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
                                create_graph=False, allow_unused=True
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

                    try:
                        da_mean = float(da.mean().item())
                    except:
                        da_mean = float('nan')
                    if dparams:
                        valid = [d for d in dparams if d is not None]
                        if all(torch.isfinite(d).all() for d in valid):
                            dparams_mean = float(
                                sum(torch.mean(d) for d in valid).item() / len(valid)
                            )
                        else:
                            dparams_mean = float('nan')
                    else:
                        dparams_mean = float('nan')

                    gtheta = grad_theta[0].mean().item() if grad_theta else float('nan')

                    with open(log_file, 'a', newline='') as f:
                        writer = csv.writer(f)
                        if not file_exists:
                            writer.writerow(['Step', 'da_mean', 'dparams_mean', 'grad_theta', 'initial_S'])
                            file_exists = True
                        writer.writerow([i, da_mean, dparams_mean, gtheta, scaler.S])

                    dparams = [
                        d if d is not None else torch.zeros_like(p)
                        for d, p in zip(dparams, params)
                    ]

                    if scaler._is_any_infinite((da, gti, gdti, dparams)):
                        scaler.update_on_overflow(a, at, grad_theta, grad_t, gti, in_place=True)
                        attempts += 1
                        continue
                    else:
                        break
                if attempts >= scaler.max_attempts:
                    raise RuntimeError(f"Reached maximum number of attempts in backward pass at time step i={i}")

                # --- adjoint & param updates ---
                a = a + dti * da.to(dtype_hi) + at[i].to(dtype_hi)
                for idx, d in enumerate(dparams):
                    grad_theta[idx] = grad_theta[idx] + (d * dti).to(grad_theta[idx].dtype)
                if grad_t is not None and gti is not None:
                    grad_t = grad_t.clone()
                    grad_t[i]   += dti * (gti   - gdti)
                    grad_t[i+1] += dti * gdti2

                # --- periodic scaling ---
                counter += 1
                if counter % k == 0:
                    L_j = da.norm(p=2) / a.norm(p=2).clamp(min=Mmin)
                    pred = a.norm() * ((1 + L_j * dti)**(k-1))
                    if pred  > Mmax: #* scaler.S
                        #down‐scale so that a_{i-k} does not overflow
                        a, da = scaler.update_on_overflow(a, da)
                        grad_theta = [g * scaler.decrease_factor for g in grad_theta]
                        if grad_t is not None:
                            grad_t = grad_t * scaler.decrease_factor
                    elif pred < Mmin: #* scaler.S 
                        #up‐scale so that a_{i-k} is not too small
                        a, da = scaler.update_on_small_grad(a, da)
                        grad_theta = [g * scaler.increase_factor for g in grad_theta]
                        if grad_t is not None:
                            grad_t = grad_t * scaler.increase_factor

        # restore parameters and unscale
        for name, param in func.named_parameters():
            param.data = old_params[name].data

        # Unscale the gradients 
        scaler.unscale((a, grad_theta,grad_t),in_place=True)
        if t.requires_grad:
            grad_t = grad_t.to(dtype_t)

        return (None, None, a, grad_t, None, *grad_theta)
