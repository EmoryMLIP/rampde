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

class DynamicScaler:
    def __init__(self, dtype_low, target_factor=None, increase_factor=2.0, decrease_factor=0.5,
                 small_grad_threshold=1.0, max_attempts=10, delta=0):
        self.dtype_low = dtype_low
        # Set a target norm if not provided: 1/epsilon for low precision.
        self.eps = torch.finfo(dtype_low).eps
        self.target = target_factor if target_factor is not None else 1.0 / self.eps
        self.increase_factor = increase_factor
        self.decrease_factor = decrease_factor
        self.small_grad_threshold = small_grad_threshold
        self.max_attempts = max_attempts
        self.delta = delta
        self.S = None  # This will be initialized later

    def init_scaling(self, a):
        # Initialize S such that S * ||a|| ~ target.
        self.S = self.target / (a.norm() + self.delta)
        self.S = 2**torch.round(torch.log2(self.S))
        # make sure S is a power of 2
        anew = self.S * a
        while not(anew.isfinite().all()):
            self.S *= 0.5
            anew = self.S * a
        
    def scale(self, tensor):
        return self.S * tensor

    def unscale(self, tensor):
        return tensor / self.S

    def update_on_overflow(self):
        self.S *= self.decrease_factor

    def update_on_small_grad(self):
        self.S *= self.increase_factor

    

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

        N = t.shape[0]
        params = tuple(params) 

        # Initialize dynamic scaler.
        scaler = DynamicScaler(dtype_low=dtype_low)
        scaler.init_scaling(at[-1])
        
        at = scaler.scale(at)
        a = at[-1].to(dtype_hi)
        
        grad_theta = [torch.zeros_like(param) for param in params]
        grad_t = None if not t.requires_grad else torch.zeros_like(t)
        
        with autocast(device_type='cuda', enabled=False):
            old_params = {name: param.data.clone() for name, param in func.named_parameters()}
            for name, param in func.named_parameters():
                param.data = (param.data).to(dtype_low)
                
            for i in reversed(range(N-1)):
                with torch.enable_grad():
                    dti = t[i+1] - t[i]
                    y = yt[i].clone().detach().requires_grad_(True)

                    attempts = 0
                    gti = None
                    gdti = None
                    while attempts < scaler.max_attempts:
                        if t.requires_grad:
                            ti = t[i].clone().detach().requires_grad_(True)
                            dti = dti.clone().detach().requires_grad_(True)
                            
                            dy = step(func, y, ti, dti)
                            da, gti, gdti, *dparams = torch.autograd.grad(dy, (y, ti, dti, *params), a,create_graph=True,allow_unused=True)
                            gdti2 = torch.sum(a*dy,dim=-1)   
                        else:
                            ti = t[i]
                            dy = step(func, y, ti, dti)
                            da, *dparams = torch.autograd.grad(dy, (y, *params), a, create_graph=True)
                    
                           
                        if da.isfinite().all() and (gti is None or gti.isfinite().all()) and (gdti is None or gdti.isfinite().all()) and torch.all(torch.stack([dparam.isfinite().all() for dparam in dparams])):
                            #
                            # print("i=", i, "accepted at attempt ", attempts)
                            break
                        else:
                            # print("i=", i,"reduce scaling because of infs")
                            scaler.update_on_overflow()
                            # we reduced the scaling, so we need to rescale the gradients we computed so far and remaining adjoint signals
                            a = scaler.decrease_factor*a
                            at = scaler.decrease_factor*at
                            grad_theta = [scaler.decrease_factor*grad for grad in grad_theta]                            
                            if grad_t is not None:
                                grad_t = scaler.decrease_factor*grad_t
                            if gti is not None:
                                gti = scaler.decrease_factor*gti
                            attempts += 1
                            if attempts == scaler.max_attempts:
                                # throw an error if we have reached the maximum number of attempts
                                raise RuntimeError("Reached maximum number of attempts in backward pass at time step i={}".format(i))


                # print("norm(a): ", a.norm().item(), "norm(at[i]): ", at[i].norm().item(), "norm(scale(a)): ",scaler.scale(a).norm().item())
                update_attempts = 0
                while update_attempts < scaler.max_attempts:
                    a_new = a + dti*(da.to(dtype_hi)) + at[i].to(dtype_hi)
                    grad_theta_new = [grad + dti*dparam.to(grad.dtype) for grad, dparam in zip(grad_theta, dparams)]        
                    # print norm of gradients
                    # print("grad_theta_new", [torch.norm(grad) for grad in dparams])
                    if grad_t is not None:
                        grad_t_new = grad_t.clone() 
                    
                        gti_new = gti.clone() if gti is not None else torch.zeros_like(t[i])
                        gdti_new = gdti.clone() if gdti is not None else torch.zeros_like(t[i])
                        gdti2_new = gdti2.clone() if gdti is not None else torch.zeros_like(t[i])
                        grad_t_new[i] = grad_t[i] + dti*(gti_new - gdti_new) - gdti2_new
                        grad_t_new[i+1] = grad_t[i+1] + dti*gdti_new + gdti2_new
                    else:
                        grad_t_new = None
                    
                    if a_new.isfinite().all() and (grad_t_new is None or grad_t_new.isfinite().all()) and torch.all(torch.stack([grad_new.isfinite().all() for grad_new in grad_theta_new])):
                        break
                    else:
                        # print("i", i, "reduce scaling because of infs in updates")
                        scaler.update_on_overflow()
                        # we reduced the scaling, so we need to rescale the gradients we computed so far and remaining adjoint signals
                        a = scaler.decrease_factor*a
                        at = scaler.decrease_factor*at
                        da = scaler.decrease_factor*da
                        grad_theta = [scaler.decrease_factor*grad for grad in grad_theta]
                        dparams = [scaler.decrease_factor*dparam for dparam in dparams]
                        
                        if grad_t is not None:
                            gti = scaler.decrease_factor*gti
                            gdti = scaler.decrease_factor*gdti
                            gdti2 = scaler.decrease_factor*gdti2
                            grad_t = scaler.decrease_factor*grad_t
                        at = scaler.decrease_factor*at                        
                        update_attempts += 1
                        if update_attempts == scaler.max_attempts:
                            # throw an error if we have reached the maximum number of attempts
                            raise RuntimeError("Reached maximum number of attempts in updating at time step i={}".format(i))

                # if not(torch.all(torch.stack([(grad).isfinite().all() for grad in grad_theta_new]))):
                #     print("i", i, "grad_theta is infinite after updating")
                #     assert 1==0

                
                if attempts == 0 and update_attempts==0 and torch.norm(a_new)/scaler.target < 0.5:
                    if (scaler.increase_factor*a_new).isfinite().all() and (scaler.increase_factor*at).isfinite().all() and (gti is None or (scaler.increase_factor*grad_t_new).isfinite().all()) and (gdti is None or (scaler.increase_factor*gdti).isfinite().all()) and torch.all(torch.stack([(scaler.increase_factor*grad).isfinite().all() for grad in grad_theta_new])):
                        # print("i", i, "increase scaling because of small adjoint")
                        scaler.update_on_small_grad()
                        # we increased the scaling, so we need to rescale the gradients we computed so far and remaining adjoint signals
                        a_new = scaler.increase_factor*a_new
                        at = scaler.increase_factor*at
                        grad_theta_new = [scaler.increase_factor*grad for grad in grad_theta_new]
                        if grad_t is not None:
                            grad_t_new = scaler.increase_factor*grad_t_new
                        if gti is not None:
                            gti_new = scaler.increase_factor*gti_new
                # if not(torch.all(torch.stack([(grad).isfinite().all() for grad in grad_theta_new]))):
                #     print("i", i, "grad_theta is infinite after scaling")
                #     assert 1==0

                elif torch.norm(a_new)/scaler.target > 2:
                    # print("i", i, "reduce scaling because of large adjoint")
                    scaler.update_on_overflow()
                    # we reduced the scaling, so we need to rescale the gradients we computed so far and remaining adjoint signals
                    a_new = scaler.decrease_factor*a_new
                    at = scaler.decrease_factor*at
                    grad_theta_new = [scaler.decrease_factor*grad for grad in grad_theta_new]
                    if grad_t is not None:
                        grad_t_new = scaler.decrease_factor*grad_t_new
                    if gti is not None:
                        gti_new = scaler.decrease_factor*gti_new
                                
                a = a_new
                # if not(torch.all(torch.stack([(grad).isfinite().all() for grad in grad_theta_new]))):
                    # print("i", i, "grad_theta_new is infinite before overwriting grad_theta")

                grad_theta = grad_theta_new
                if grad_t is not None:
                    grad_t = grad_t_new
                   

                        
            for name, param in func.named_parameters():
                param.data = old_params[name].data
        
        # Unscale the gradients
        a = scaler.unscale(a)
        grad_theta = [scaler.unscale(grad) for grad in grad_theta]
        # if not(torch.all(torch.stack([(grad).isfinite().all() for grad in grad_theta]))):
        #             print("i", i, "grad_theta is infinite after unscaling")
        #             assert 1==0


        if grad_t is not None:
            grad_t = scaler.unscale(grad_t).to(dtype_t)
            
        return (None, None, a, grad_t,  *grad_theta)