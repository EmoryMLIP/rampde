import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import time
import math
from torch.nn.functional import pad
from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', default=True, action='store_true')
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default="./results/cnf")
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16')
# This argument is now only used as a default; we will loop over both options.
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'rampde'], default='rampde')
parser.add_argument('--results_dir', type=str, default="./results/cnf")
parser.add_argument('--scaler', type=str, choices=['noscaler', 'dynamicscaler'], default='noscaler')

args = parser.parse_args()



precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]


def hyper_trace(W, B, U, x, target_dtype):

    W = W.to(target_dtype)  # [w, d, 1]
    B = B #.to(target_dtype)  # [w, 1, 1]
    U = U.to(target_dtype)  # [w, 1, d]
    x = x.to(target_dtype)  # [n, d]

    w, d, _ = W.shape
    n = x.shape[0]
    x_exp = x.unsqueeze(0).expand(w, -1, -1)  # [w, n, d]

    # s_j = x @ w_j + b_j
    s = torch.bmm(x_exp, W).squeeze(-1)       # [w, n]
    s = s.to(torch.float32) + B.to(torch.float32).squeeze(-1)                     # [w, n]
    deriv = 1 - torch.tanh(s.to(target_dtype))**2              # [w, n]

    # u_j ⋅ w_j per neuron
    uw_dot = torch.bmm(U, W).squeeze(-1).squeeze(-1)  # [w]
    uw_dot = uw_dot.view(w, 1)                        # [w, 1]

    trace_all = deriv.to(target_dtype) * uw_dot       # [w, n]
    trace_sum = trace_all.to(torch.float32).sum(dim=0) 
    trace_est = trace_sum / w        
    trace_est = trace_est.to(target_dtype)  
    # print('tracedtype:', trace_est.dtype)
    return trace_est.view(n, 1)   

class CNF(nn.Module):
    """Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.hyper_net = HyperNetwork(in_out_dim, hidden_dim, width)  

    def forward(self, t, states):
        z = states[0]
        logp_z = states[1]
        batchsize = z.shape[0]
        W, B, U = self.hyper_net(t) 

        z = z.to(W.dtype)
        Z = z.unsqueeze(0).repeat(self.width, 1, 1)
        h = torch.tanh(torch.matmul(Z, W) + B)
        f = torch.matmul(h, U).mean(0) 

        target_dtype = torch.get_autocast_dtype('cuda') if torch.is_autocast_enabled() else torch.float32
        trace_est = hyper_trace(W, B, U, z, target_dtype)  
        # dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1) # this stays in f32
        dlogp_z_dt = -trace_est
        dz_dt = f

        # print("dz_dt dtype:", dz_dt.dtype, "trace_est dtype:", trace_est.dtype)
        return (dz_dt, dlogp_z_dt)

def trace_df_dz(f, z):
    """Calculates the trace of the Jacobian df/dz.
    Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
    """

    sum_diag = 0.
    for i in range(z.shape[1]):
        sum_diag += torch.autograd.grad(f[:, i].sum(), z, create_graph=True)[0].contiguous()[:, i].contiguous()
    return sum_diag.contiguous()
    
class HyperNetwork(nn.Module):
    """Hyper-network allowing f(z(t), t) to change with time.

    Adapted from the NumPy implementation at:
    https://gist.github.com/rtqichen/91924063aa4cc95e7ef30b3a5491cc52
    """
    def __init__(self, in_out_dim, hidden_dim, width):
        super().__init__()
        blocksize = width * in_out_dim
        self.fc1 = nn.Linear(1, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 3 * blocksize + width)
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.width = width
        self.blocksize = blocksize

    def forward(self, t):

        params = t.reshape(1, 1)
        params = params.to(self.fc1.weight.dtype)
        params = torch.tanh(self.fc1(params))
        params = torch.tanh(self.fc2(params))
        params = self.fc3(params)

        params = params.reshape(-1)
        W = params[:self.blocksize].reshape(self.width, self.in_out_dim, 1)
        U = params[self.blocksize:2 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        G = params[2 * self.blocksize:3 * self.blocksize].reshape(self.width, 1, self.in_out_dim)
        U = U * torch.sigmoid(G)
        B = params[3 * self.blocksize:].reshape(self.width, 1, 1)
        return [W, B, U]

def gaussian_logpdf(x, mu, sigma):
    d = x.shape[1]
    return -0.5 * d * np.log(2 * np.pi) - d * np.log(sigma) - ((x - mu)**2).sum(dim=1) / (2 * sigma**2)

def mixture_logpdf(x, sigma = 0.1, R = 2.0, N = 8):
    logpdfs = []
    for k in range(N):
        theta = 2 * math.pi * k / N
        mu_k = torch.tensor([R * math.cos(theta), R * math.sin(theta)],
                            device=x.device, dtype=x.dtype)
        logpdfs.append(gaussian_logpdf(x, mu_k, sigma))
    logpdfs = torch.stack(logpdfs, dim=0)
    max_logpdf, _ = torch.max(logpdfs, dim=0)
    log_sum_exp = max_logpdf + torch.log(torch.exp(logpdfs - max_logpdf).sum(dim=0))
    return log_sum_exp - math.log(N)

def sample_mixture(num_samples, device, sigma = 0.1, R = 2.0, N = 8):
    comps = torch.randint(0, N, (num_samples,), device=device)
    means = []
    for k in range(N):
        theta = 2 * math.pi * k / N
        mu_k = torch.tensor([R * math.cos(theta), R * math.sin(theta)], 
                            device=device, dtype=torch.float32)
        means.append(mu_k)
    means = torch.stack(means, dim=0)
    chosen_means = means[comps]
    noise = torch.randn(num_samples, 2, device=device) * sigma
    return chosen_means + noise

def get_batch(num_samples, device):
    x = sample_mixture(num_samples, device)
    logp_diff_t1 = torch.zeros(num_samples, 1, device=device, dtype=torch.float32)
    return x, logp_diff_t1

comparison_logp_diff = {}
viz_samples = 30000
viz_timesteps = 41
t0 = 0
t1 = 1
combined_t = np.linspace(t0, t1, viz_timesteps)

p_z0 = torch.distributions.MultivariateNormal(
    loc=torch.tensor([0.0, 0.0], device=device),
    covariance_matrix=torch.tensor([[1, 0.0], [0.0, 1]], device=device)
)

combined_dir = args.results_dir
if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)



if args.viz:
    with torch.no_grad():

        z0_batch      = p_z0.sample([viz_samples]).to(device)
        logp0_batch   = torch.zeros(viz_samples, 1, device=device, dtype=torch.float32)
        ts_path       = torch.linspace(t0, t1, viz_timesteps).to(device)

        comparison_logp_diff = {}
        learned_logp_grids   = {}

        x_lin       = np.linspace(-4, 4, 100)
        y_lin       = np.linspace(-4, 4, 100)
        X, Y        = np.meshgrid(x_lin, y_lin)
        grid_points = np.vstack([X.ravel(), Y.ravel()]).T
        grid_tensor = torch.tensor(grid_points, dtype=torch.float32, device=device)

        for odeint_option in ['torchdiffeq', 'rampde']:
            func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)

            train_dir_opt = args.train_dir #+ '_' + odeint_option
            ckpt_path     = os.path.join(train_dir_opt, f'{odeint_option}_2000.pth')
            checkpoint    = torch.load(ckpt_path, map_location=device)
            func.load_state_dict(checkpoint['func_state_dict'])
            func.to(device).eval()
            print(f"Loaded checkpoint for {odeint_option} from {ckpt_path}")

            if odeint_option == 'rampde':
                import sys
                sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
                from rampde import odeint as odeint_fn
                from rampde import NoScaler, DynamicScaler
                scaler_map = {
                    'noscaler':     NoScaler(dtype_low=args.precision),
                    'dynamicscaler': DynamicScaler(dtype_low=args.precision)
                }
                solver_kwargs = {'loss_scaler': scaler_map[args.scaler]}
                method = 'rk4'
            else:
                from torchdiffeq import odeint_adjoint, odeint
                odeint_fn     = odeint_adjoint if args.adjoint else odeint
                solver_kwargs = {}
                method         = args.method

            z_path, logp_path = odeint_fn(
                func,
                (z0_batch, logp0_batch),
                ts_path,
                atol=1e-5, rtol=1e-5,
                method=method,
                **solver_kwargs
            )
            mad_list = []
            for i in range(viz_timesteps):
                learned_lp    = p_z0.log_prob(z_path[i]) - logp_path[i].view(-1)
                analytical_lp = mixture_logpdf(z_path[i])
                mad_list.append((analytical_lp - learned_lp).abs().mean().item())
            comparison_logp_diff[odeint_option] = mad_list

            ts_density = torch.linspace(t1, t0, viz_timesteps).to(device)
            z_den, logp_den = odeint_fn(
                func,
                (grid_tensor, torch.zeros(grid_tensor.shape[0], 1, device=device)),
                ts_density,
                atol=1e-5, rtol=1e-5,
                method='rk4',
                **solver_kwargs
            )
            z_final      = z_den[-1]
            logp_final   = logp_den[-1].view(-1)
            learned_den  = torch.exp(p_z0.log_prob(z_final) - logp_final)
            learned_logp_grids[odeint_option] = learned_den.cpu().numpy()


        analytical_den = torch.exp(mixture_logpdf(grid_tensor)).cpu().numpy()


        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)

        # Target density
        cs0 = axes[0].tricontourf(
            grid_points[:, 0], grid_points[:, 1],
            analytical_den, levels=50
        )
        axes[0].set_title("Target density")
        axes[0].set_xticks([]); axes[0].set_yticks([])
        fig.colorbar(cs0, ax=axes[0])

        # torchdiffeq result
        cs1 = axes[1].tricontourf(
            grid_points[:, 0], grid_points[:, 1],
            learned_logp_grids['torchdiffeq'], levels=50
        )
        axes[1].set_title("Learned density\n(torchdiffeq)")
        axes[1].set_xticks([]); axes[1].set_yticks([])
        fig.colorbar(cs1, ax=axes[1])

        # rampde result
        cs2 = axes[2].tricontourf(
            grid_points[:, 0], grid_points[:, 1],
            learned_logp_grids['rampde'], levels=50
        )
        axes[2].set_title("Learned density\n(rampde)")
        axes[2].set_xticks([]); axes[2].set_yticks([])
        fig.colorbar(cs2, ax=axes[2])

        out_path = os.path.join(combined_dir, "density_comparison.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved combined density comparison at {out_path}")


plt.figure(figsize=(6, 4))

plot_styles = {
    'torchdiffeq':  {'color': 'blue', 'linestyle': '-', 'marker': 'o'},
    'rampde':  {'color': 'red',  'linestyle': '-', 'marker': 'o'},
}

for method, diffs in comparison_logp_diff.items():
    style = plot_styles.get(method, {})
    plt.plot(combined_t, diffs, label=method, markersize=2, **style)

plt.xlabel("Time t")
plt.ylabel("Log density difference")
plt.legend()
combined_path = os.path.join(combined_dir, "logp_path_diff.png")
plt.savefig(combined_path)
plt.close()
print("Saved combined logp path difference plot at", combined_path)

