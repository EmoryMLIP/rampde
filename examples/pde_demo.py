import os
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import RunningAverageMeter, RunningMaximumMeter

from torch.amp import autocast


parser = argparse.ArgumentParser('PDE demo')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16','bfloat16'], default='float32')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchdiffeq')


args = parser.parse_args()
from torchdiffeq import odeint as odeint_fwd

if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    assert args.method == 'rk4' 
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from torchmpnode import odeint
else:    
    print("using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

class RegularGrid2D:
            
        def __init__(self, omega,m, device='cpu', dtype=torch.float32):
            self.omega = omega
            self.m = m
            self.h = [(omega[1] - omega[0]) / self.m[0], (omega[3] - omega[2]) / self.m[1]]
            self.device = device
            self.dtype = dtype
            
        def xc(self, i):
            return torch.linspace(self.omega[2*i] + self.h[i]/2, self.omega[2*i+1] - self.h[i]/2, self.m[i], device=self.device)
        
        def xn(self, i):
            return torch.linspace(self.omega[2*i], self.omega[2*i+1], self.m[i]+1, device=self.device)

        def avg1(self):
            kernel = torch.tensor([[[[1.], [1.]]]], device=self.device)
            A = lambda u: 0.5*F.conv2d(u,kernel)
            At = lambda u: 0.5*F.conv_transpose2d(u,kernel)
            return A, At
        
        def avg2(self):
            kernel = torch.tensor([[[[1.0,1.0]]]],device=self.device)
            A = lambda u: 0.5*F.conv2d(u,kernel)
            At = lambda u: 0.5*F.conv_transpose2d(u,kernel)
            return A, At
        def diff1(self):
            kernel = torch.tensor([[[[-1.], [1.]]]], device=self.device)
            hinv = 1/self.h[0]
            A = lambda u: hinv*F.conv2d(u,kernel)
            At = lambda u: hinv*F.conv_transpose2d(u,kernel)
            return A, At
        
        def diff2(self):
            kernel = torch.tensor([[[[-1.0,1.0]]]],device=self.device)
            hinv = 1/self.h[1]
            A = lambda u: hinv*F.conv2d(u,kernel)
            At = lambda u: hinv*F.conv_transpose2d(u,kernel)
            return A, At               


class ODEFunc(nn.Module):

    def __init__(self,domain):
        super(ODEFunc, self).__init__()
        self.domain = domain
        self.A1, self.A1t = domain.avg1()
        self.A2, self.A2t = domain.avg2()
        self.D1, self.D1t = domain.diff1()
        self.D2, self.D2t = domain.diff2()
        self.sigma = torch.nn.Parameter(torch.ones(domain.m,device = domain.device)).unsqueeze(0).unsqueeze(0)
        

    def forward(self, t, u):
        u1 = self.D1t(self.A1(self.sigma)*self.D1(u))
        u2 = self.D2t(self.A2(self.sigma)*self.D2(u))
        return -u1-u2

if __name__ == '__main__':
    ii = 0

    m = tuple([128, 128])
    omega = torch.tensor([0., 1., 0., 1.])
    domain = RegularGrid2D(omega, m, device=device)
    
    x0 = domain.xc(0)
    x1 = domain.xc(1)
    u0 = (torch.cos(2*np.pi*x0).unsqueeze(1)*torch.cos(2*np.pi*x1).unsqueeze(0)).unsqueeze(0).unsqueeze(0)
    t_span = torch.linspace(0., 0.1, args.data_size).to(device)

    true_func = ODEFunc(domain).to(device)
    box = [0.25, 0.75, 0.25, 0.75]
    ind_box = ((x0<=box[1])*(x0>=box[0])).unsqueeze(1)*((x1<=box[3])*(x1>=box[2])).unsqueeze(0)
    true_func.sigma.data = (0.01*torch.ones(m,device = device) + 0.1*ind_box ) .unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        true_y = odeint(true_func, u0, t_span, method='dopri5', rtol=1e-3, atol=1e-3)

    print(true_func(0.0,u0))
    print(torch.norm(true_y.view(50,-1),dim=1))
    
    func = ODEFunc(domain).to(device)
    # set optimizer to BFGS
    optimizer = optim.LBFGS([func.parameters()], max_iter=2, max_eval=25, history_size=100, line_search_fn='strong_wolfe')
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    mem_meter = RunningMaximumMeter()
    loss_meter = RunningAverageMeter(0.97)

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        torch.cuda.reset_peak_memory_stats(device)
        def closure():
            with autocast(device_type='cuda', dtype=args.precision):
                pred_y = odeint(func, u0, t_span, method=args.method).to(device)
                loss = torch.norm(pred_y - true_y)**2
                loss.backward()
            return loss
        optimizer.step(closure)
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB
        mem_meter.update(peak_memory)

        if itr % args.test_freq == 1:
            with torch.no_grad():
                pred_y = odeint(func, u0 , t_span, method=args.method)
                loss = torch.mean(torch.abs(pred_y - true_y))
                print('Iter {:04d} | Total Loss {:.6f} | Time {:.4f}s | Max Memory {:.1f}MB'.format(itr, loss.item(), time_meter.avg, mem_meter.max))
                ii += 1

        end = time.time()