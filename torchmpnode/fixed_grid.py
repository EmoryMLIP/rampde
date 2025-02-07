from torch.amp import custom_fwd, custom_bwd, autocast
import torch

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
    def forward(ctx, step, func, y0, t, *params):
        dtype_low = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
        dtype_hi = y0.dtype
        N = t.shape[0]
        y = y0
        yt = torch.zeros(N, *y.shape, dtype=dtype_low, device=y.device)
        yt[0] = y0.to(dtype_low)
        # print(f"yt: {yt.dtype}, y0: {y0.dtype}, t: {t.dtype}, dtype_hi: {dtype_hi}, dtype_low: {dtype_low}")
        with torch.no_grad():
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
        # ctx.dtype_low = dtype_low

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

        # print(f"yt: {yt.dtype}, at: {at.dtype}, t: {t.dtype}, dtype_hi: {dtype_hi}, dtype_low: {dtype_low}, dtype_t: {dtype_t}")
        
        N = t.shape[0]
        a = at[-1].to(dtype_hi)
        params = tuple(params) 
        
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        with autocast(device_type='cuda', enabled=False):
            old_params = dict(func.named_parameters())
            for name, param in func.named_parameters():
                param.data = param.data.to(dtype_low)
                # print(f"Parameter name: {name}, dtype: {param.dtype}")

            
            for i in reversed(range(N-1)):
                with torch.enable_grad():
                    dt = t[i+1] - t[i]
                    y = yt[i].clone().detach().requires_grad_(True)

                    if t.requires_grad:
                        ti = t[i].clone().detach().requires_grad_(True)
                        dti = dt.clone().detach().requires_grad_(True)
                        dy = step(func, y, ti, dti)
                        da, gti, gdti, *dparams = torch.autograd.grad(dy, (y, ti, dti, *params), a,create_graph=True,allow_unused=True)
                    else:
                        ti = t[i]
                        dti = dt
                        dy = step(func, y, ti, dti)
                        da, *dparams = torch.autograd.grad(dy, (y, *params), a, create_graph=True)
                        
                a = a + dt*(da.to(dtype_hi) + at[i].to(dtype_hi))
                for j, dparam in enumerate(dparams):
                    grad_theta[j] = grad_theta[j] + dt*dparam.to(grad_theta[j].dtype).detach()
                if  grad_t is not None:
                    gti = gti if gti is not None else torch.zeros_like(t[i])
                    gdti = gdti if gdti is not None else torch.zeros_like(t[i])
                    grad_t[i] = grad_t[i] + dt*(gti - gdti)
                    grad_t[i+1] = grad_t[i+1] + dt*gdti
            
            for name, param in func.named_parameters():
                param.data = old_params[name].data
            
        grad_t = grad_t.to(dtype_t) if grad_t is not None else None
        return (None, None, a, grad_t,  *grad_theta)