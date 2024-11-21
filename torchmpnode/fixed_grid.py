import torch

class Euler(torch.nn.Module):
    order = 1
    name = 'euler'

    def forward(self, func, y, t, dt, dtype=None):
        dy = dt * func(t, y)
        return dy if dtype is None else dy.to(dtype)

_one_sixth= 1/6
class RK4(torch.nn.Module):
    order = 4
    name = 'rk4'

    def forward(self, func, y, t, dt,dtype=None):
        half_dt = dt * 0.5
        k1 = func(t, y)
        k2 = func(t + half_dt, y + k1*half_dt)
        k3 = func(t + half_dt, y + k2*half_dt)
        k4 = func(t + dt, y + k3*dt)
        if dtype is None:
            return (k1 + 2*(k2 + k3) + k4) * (dt * _one_sixth)
        else:
            return (k1.to(dtype) + 2*( k2.to(dtype) + k3.to(dtype))  + k4.to(dtype)) * (dt * _one_sixth)


class FixedGridODESolver(torch.autograd.Function):

    @staticmethod
    def forward(ctx, step, func, y0, t, dtype_hi,  *params):
        dtype_y = y0.dtype
        y = y0.to(dtype_hi)
        t = t.to(dtype_hi)
        N = t.shape[0]
        
        yt = torch.zeros(N, *y.shape, dtype=dtype_y, device=y.device)
        yt[0] = y0

        with torch.no_grad():
            for i in range(N - 1):
                dt = t[i + 1] - t[i]
                dy = step(func, y.to(dtype_y), t[i].to(dtype_y), dt.to(dtype_y), dtype_y)
                y = y + dy
                yt[i + 1] = y.to(dtype_y)
        
        
        ctx.save_for_backward(yt, *params)
        ctx.step = step
        ctx.func = func
        ctx.t = t
        ctx.dtype_hi = dtype_hi
        ctx.dtype_y = dtype_y

        return yt
    
    @staticmethod
    def backward(ctx, at):
        yt, *params = ctx.saved_tensors
        step = ctx.step
        func = ctx.func
        t = ctx.t
        dtype_hi = ctx.dtype_hi
        dtype_y = ctx.dtype_y
        dtype_t = t.dtype

        t = t.to(dtype_hi)
        
        N = t.shape[0]
        a = at[-1].to(dtype_hi)
        params = tuple(params)        

        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        for i in reversed(range(N-1)):
            with torch.enable_grad():
                dt = t[i+1] - t[i]
                y = yt[i].clone().detach().requires_grad_(True)
                if t.requires_grad:
                    ti = t[i].clone().detach().requires_grad_(True)
                    dti = dt.clone().detach().requires_grad_(True)
                    dy = step(func, y.to(dtype_y), ti.to(dtype_y), dti.to(dtype_y), dtype_y)
                    da, gti, gdti, *dparams = torch.autograd.grad(dy, (y, ti, dti, *params), a,allow_unused=True)

                else:
                    ti = t[i]
                    dti = dt
                    dy = step(func, y.to(dtype_y), ti.to(dtype_y), dti.to(dtype_y), dtype_y)
                    da,  *dparams = torch.autograd.grad(dy, (y,  *params), a)

            a = a + da + at[i]
            for j, dparam in enumerate(dparams):
                grad_theta[j] = grad_theta[j] + dparam.detach()
            if  grad_t is not None:
                gti = gti if gti is not None else torch.zeros_like(t[i])
                grad_t[i] = grad_t[i] + gti - gdti
                grad_t[i+1] = grad_t[i+1] + gdti
            
        grad_t = grad_t.to(dtype_t) if grad_t is not None else None
        return (None, None, a.to(dtype_y), grad_t, None, *grad_theta)