#!/usr/bin/env python3
import os
import argparse
import glob
import csv
import shutil
import time
import math
import datetime
import sys

from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast

from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', default=True, action='store_true')
parser.add_argument('--niters', type=int, default=2000)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--num_samples_val', type=int, default=1024)
parser.add_argument('--num_timesteps', type=int, default=128)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default="./results/cnf")
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16', 'bfloat16'], default='float32')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
parser.add_argument('--results_dir', type=str, default="./results/cnf")
parser.add_argument('--scaler', type=str, choices=['noscaler', 'dynamicscaler'], default='dynamicscaler')
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--seed', type=int, default=0)

args = parser.parse_args()

# Map precision string to torch dtype
precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


os.makedirs(args.results_dir, exist_ok=True)
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{args.precision}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
# Save a copy of this script in the results directory.
script_path = os.path.abspath(__file__)
shutil.copy(script_path, os.path.join(args.results_dir, os.path.basename(script_path)))

# Redirect stdout and stderr to a log file.
log_path = os.path.join(args.results_dir, folder_name + ".txt")
log_file = open(log_path, "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

print("Experiment started at", datetime.datetime.now())
print("Arguments:", vars(args))
print("Results will be saved in:", args.results_dir)


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

    # u_j * w_j 
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
precisions = ['float32', 'float16'] #
viz_timesteps = 41
t0, t1 = 0.0, 1.0
combined_t = np.linspace(t0, t1, viz_timesteps)

combined_dir = f"{args.results_dir}_combined"
os.makedirs(combined_dir, exist_ok=True)
combined_csv = os.path.join(combined_dir, "combined_logp_diff.csv")
with open(combined_csv, "w", newline="") as cf:
    writer = csv.writer(cf)
    header = ["time"] + [f"{p}_{m}" for p in precisions for m in ['torchmpnode','torchdiffeq']]
    writer.writerow(header)

for precision_str in precisions:
    args.precision = precision_map[precision_str]
    for odeint_option in ['torchmpnode','torchdiffeq']: #
        results_dir_exp = f"{args.results_dir}_{odeint_option}_{precision_str}"
        os.makedirs(results_dir_exp, exist_ok=True)
        train_dir_option = f"{args.train_dir}_{odeint_option}_{precision_str}" if args.train_dir else None
        if train_dir_option:
            os.makedirs(train_dir_option, exist_ok=True)

        csv_path = os.path.join(train_dir_option, "metrics.csv")


        if odeint_option == 'torchmpnode':
            print("Using torchmpnode")
            # For torchmpnode, method must be 'rk4'
            assert args.method == 'rk4'
            import sys
            sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
            from torchmpnode import odeint as odeint_fn
            from torchmpnode import NoScaler, DynamicScaler
            scaler_map = {
                'noscaler': NoScaler(dtype_low=args.precision),
                'dynamicscaler': DynamicScaler(dtype_low=args.precision)
            }
            scaler = scaler_map[args.scaler]
            solver_kwargs = {'loss_scaler': scaler}
        else:
            print("Using torchdiffeq")
            if args.adjoint:
                from torchdiffeq import odeint_adjoint as odeint_fn
            else:
                from torchdiffeq import odeint as odeint_fn
            solver_kwargs = {}

        func = CNF(in_out_dim=2, hidden_dim=args.hidden_dim, width=args.width).to(device)
        optimizer = optim.Adam(func.parameters(), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

        p_z0 = torch.distributions.MultivariateNormal(
            loc=torch.tensor([0.0, 0.0], device=device),
            covariance_matrix=torch.tensor([[1, 0.0], [0.0, 1]], device=device)
        )

        loss_meter = RunningAverageMeter()
        time_meter = RunningAverageMeter()
        mem_meter = RunningMaximumMeter()

        checkpoint_files = glob.glob(os.path.join(train_dir_option, f'{odeint_option}_*.pth'))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            cp = torch.load(latest_checkpoint, map_location=device)
            func.load_state_dict(cp['func_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            print(f"Loaded checkpoint from {latest_checkpoint}")
        else:
            with open(csv_path, "w", newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow([
                    "iter","lr",
                    "running_loss","val_loss","val_loss_mp",
                    "running_NLL","val_NLL","val_NLL_mp",
                    "time","max_memory"
                ])
                for itr in range(1, args.niters + 1):
                    optimizer.zero_grad()
                    x, logp_diff_t1 = get_batch(args.num_samples, device)
                    start_time = time.perf_counter()
                    torch.cuda.reset_peak_memory_stats(device)

                    with autocast(device_type='cuda', dtype=args.precision):
                        ts = torch.linspace(t1, t0, args.num_timesteps, device=device)
                        z_t, logp_diff_t = odeint_fn(
                            func,
                            (x, logp_diff_t1),
                            ts,
                            atol=1e-5,
                            rtol=1e-5,
                            method=args.method,
                            **solver_kwargs
                        )
                        z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
                        logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
                        loss = -logp_x.mean(0)

                        loss.backward()
                        optimizer.step()
                        scheduler.step()

                    elapsed_time = time.perf_counter() - start_time
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    time_meter.update(elapsed_time)
                    mem_meter.update(peak_memory)
                    loss_meter.update(loss.item())

                    if itr % args.test_freq == 0:

                        # compute validation losses
                        with torch.no_grad():
                            x_val, lpv1 = get_batch(args.num_samples_val, device)
                            ts_val = torch.linspace(t1, t0, args.num_timesteps, device=device)
                            z_v, lp_v = odeint_fn(func, (x_val, lpv1), ts_val,
                                                atol=1e-5, rtol=1e-5,
                                                method=args.method,
                                                **solver_kwargs)
                            z0_v, lp0_v = z_v[-1], lp_v[-1]
                            logp_val = p_z0.log_prob(z0_v) - lp0_v.view(-1)
                            loss_val = -logp_val.mean()

                            with autocast(device_type='cuda', dtype=args.precision):
                                lpv1_mp = torch.zeros_like(lpv1)
                                z_v_mp, lp_v_mp = odeint_fn(func, (x_val, lpv1_mp), ts_val,
                                                            atol=1e-5, rtol=1e-5,
                                                            method=args.method,
                                                            **solver_kwargs)
                                z0_mp, lp0_mp = z_v_mp[-1], lp_v_mp[-1]
                                logp_val_mp = p_z0.log_prob(z0_mp) - lp0_mp.view(-1)
                                loss_val_mp = -logp_val_mp.mean()

                        print(
                            f"Iter {itr:4d} | "
                            f"Train loss: {loss_meter.avg:.4f} | "
                            f"Val loss: {loss_val.item():.4f} | "
                            f"Val loss (mp): {loss_val_mp.item():.4f} | "
                            f"Time avg: {time_meter.avg:.4f}s | "
                            f"Mem peak: {mem_meter.val:.2f}MB"
                        )

                        csv_writer.writerow([
                            itr,
                            optimizer.param_groups[0]['lr'],
                            loss_meter.avg,
                            loss_val.item(),
                            loss_val_mp.item(),
                            loss_meter.avg,                 # running_NLL
                            (-logp_val.mean()).item(),      # val_NLL
                            (-logp_val_mp.mean()).item(),   # val_NLL_mp
                            time_meter.avg,
                            mem_meter.val
                        ])
                        csv_file.flush()


                torch.save({
                    'func_state_dict': func.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(train_dir_option, f"{odeint_option}_{args.niters}.pth"))
                
                # csv_file.close()

            # ------------------------------
            # Create optimization stats plots
            # ------------------------------
            # for odeint_option in ['torchmpnode','torchdiffeq']:
            # results_dir_exp = f"{args.results_dir}_{odeint_option}_{precision_str}"
            # os.makedirs(results_dir_exp, exist_ok=True)
            # train_dir_option = f"{args.train_dir}_{odeint_option}_{precision_str}" if args.train_dir else None
            # if train_dir_option:
            #     os.makedirs(train_dir_option, exist_ok=True)

            # csv_path = os.path.join(train_dir_option, "metrics.csv")
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader)
                data = np.array(list(reader)).astype(np.float32)

            iters       = data[:, 0]
            lr_vals     = data[:, 1]
            running_loss= data[:, 2]
            val_loss    = data[:, 3]
            val_loss_mp = data[:, 4]
            running_NLL = data[:, 5]
            val_NLL     = data[:, 6]
            val_NLL_mp  = data[:, 7]
            time_vals   = data[:, 8]
            max_memory  = data[:, 9]

            fig, axs = plt.subplots(2, 2, figsize=(12, 8))

            # 1) Loss Function subplot
            axs[0, 0].plot(iters, running_loss,    label="running loss")
            axs[0, 0].plot(iters, val_loss,        label="val loss")
            axs[0, 0].plot(iters, val_loss_mp, "--",label="val loss (mp)")
            axs[0, 0].set_title("Loss Function")
            axs[0, 0].set_xlabel("Iteration")
            axs[0, 0].set_ylabel("Loss")
            axs[0, 0].legend()

            # 2) NLL subplot
            axs[0, 1].plot(iters, running_NLL,    label="running NLL")
            axs[0, 1].plot(iters, val_NLL,        label="val NLL")
            axs[0, 1].plot(iters, val_NLL_mp, "--",label="val NLL (mp)")
            axs[0, 1].set_title("Negative Log-Likelihood")
            axs[0, 1].set_xlabel("Iteration")
            axs[0, 1].set_ylabel("NLL")
            axs[0, 1].legend()

            # 3) Learning Rate subplot
            axs[1, 0].semilogy(iters, lr_vals, label="learning rate")
            axs[1, 0].set_title("Learning Rate")
            axs[1, 0].set_xlabel("Iteration")
            axs[1, 0].set_ylabel("LR")
            axs[1, 0].legend()

            # 4) Max Memory subplot
            axs[1, 1].plot(iters, max_memory, label="max memory (MB)")
            axs[1, 1].set_title("Max Memory")
            axs[1, 1].set_xlabel("Iteration")
            axs[1, 1].set_ylabel("Memory (MB)")
            axs[1, 1].legend()

            plt.tight_layout()
            stats_fig_path = os.path.join(results_dir_exp, "optimization_stats.png")
            plt.savefig(stats_fig_path, bbox_inches='tight')
            plt.close()
            print(f"Saved optimization stats plot at {stats_fig_path}")



        if args.viz:
            with torch.no_grad():
                
                z_t0_sample = p_z0.sample([viz_samples]).to(device)
                logp_diff_t0_sample = torch.zeros(viz_samples, 1, device=device, dtype=torch.float32)
                ts_samples = torch.linspace(t0, t1, viz_timesteps).to(device)
                z_t_samples, logp_diff_samples = odeint_fn(
                    func,
                    (z_t0_sample, logp_diff_t0_sample),
                    ts_samples,
                    atol=1e-5,
                    rtol=1e-5,
                    method=args.method,
                )


                logp_differences = []
                for i, tval in enumerate(ts_samples):
                    learned_logp = p_z0.log_prob(z_t_samples[i]) - logp_diff_samples[i].view(-1)
                    analytical_logp = mixture_logpdf(z_t_samples[i])
                    diff = analytical_logp - learned_logp
                    mad = diff.abs().mean().item()
                    logp_differences.append(mad)
                    print(f"t={tval:.2f}, logp_diff={mad:.4f}")


                key = f"{precision_str}_{odeint_option}"
                comparison_logp_diff[key] = logp_differences


                x_lin = np.linspace(-4, 4, 100)
                y_lin = np.linspace(-4, 4, 100)
                X, Y = np.meshgrid(x_lin, y_lin)
                grid_points = np.vstack([X.ravel(), Y.ravel()]).T
                grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
                logp_diff_grid = torch.zeros(grid_tensor.shape[0], 1, device=device, dtype=torch.float32)
                ts_density = torch.linspace(t1, t0, viz_timesteps).to(device)
                # Compute the learned density evolution on the grid
                z_t_density, logp_diff_density = odeint_fn(
                    func,
                    (grid_tensor, logp_diff_grid),
                    ts_density,
                    atol=1e-5,
                    rtol=1e-5,
                    method='rk4', 
                    **solver_kwargs
                )
                z_final = z_t_density[-1]
                logp_final = logp_diff_density[-1]
                learned_logp_grid = torch.exp(p_z0.log_prob(z_final) - logp_final.view(-1))
                analytical_logp_grid = torch.exp(mixture_logpdf(grid_tensor)) 

                fig, axs = plt.subplots(1, 2, figsize=(12, 5))
                cs1 = axs[0].tricontourf(grid_points[:, 0], grid_points[:, 1],
                                        learned_logp_grid.detach().cpu().numpy(), levels=50)
                axs[0].set_title("Learned log density")
                fig.colorbar(cs1, ax=axs[0])
                cs2 = axs[1].tricontourf(grid_points[:, 0], grid_points[:, 1],
                                        analytical_logp_grid.detach().cpu().numpy(), levels=50)
                axs[1].set_title("Target log density")
                fig.colorbar(cs2, ax=axs[1])
                plt.savefig(os.path.join(results_dir_exp, "density_comparison.png"))
                plt.close()

                for (t, z_sample, z_density, logp_diff) in zip(
                        np.linspace(t0, t1, viz_timesteps),
                        z_t_samples, z_t_density, logp_diff_density
                ):
                    fig = plt.figure(figsize=(12, 4), dpi=200)
                    plt.tight_layout()
                    plt.axis('off')
                    plt.margins(0, 0)
                    fig.suptitle(f'{t:.2f}s')
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax1.set_title('Target')
                    ax1.get_xaxis().set_ticks([])
                    ax1.get_yaxis().set_ticks([])
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax2.set_title('Samples')
                    ax2.get_xaxis().set_ticks([])
                    ax2.get_yaxis().set_ticks([])
                    ax3 = fig.add_subplot(1, 3, 3)
                    ax3.set_title('Log Probability')
                    ax3.get_xaxis().set_ticks([])
                    ax3.get_yaxis().set_ticks([])

                    target_sample, _ = get_batch(viz_samples, device)
                    ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                            range=[[-4, 4], [-4, 4]])
                    ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                            range=[[-4, 4], [-4, 4]])
                    logp_model = p_z0.log_prob(z_density) - logp_diff.view(-1)
                    ax3.tricontourf(*grid_points.T,
                                    np.exp(logp_model.detach().cpu().numpy()), 200)
                    plt.savefig(os.path.join(results_dir_exp, f"cnf-viz-{int(t*1000):05d}.jpg"),
                            pad_inches=0.2, bbox_inches='tight')
                    plt.close()

                imgs = sorted(glob.glob(os.path.join(results_dir_exp, f"cnf-viz-*.jpg")))
                if len(imgs) > 0:
                    img, *rest_imgs = [Image.open(f) for f in imgs]
                    img.save(fp=os.path.join(results_dir_exp, "cnf-viz.gif"), format='GIF', append_images=rest_imgs,
                            save_all=True, duration=250, loop=0)
                print('Saved visualizations for', odeint_option)

combined_dir = f"{args.results_dir}_combined"
os.makedirs(combined_dir, exist_ok=True)

plt.figure(figsize=(6,4))
styles = {
    'torchdiffeq':  {'linestyle':'-','marker':'o', 'markersize': 4},
    'torchmpnode':  {'linestyle':'--','marker':'s', 'markersize': 4},
}

for key, diffs in comparison_logp_diff.items():
    precision_str, method = key.split('_', 1)
    style = styles.get(method, {})
    label = f"{precision_str} / {method}"
    plt.plot(combined_t, diffs, label=label, **style)

plt.xlabel("Time t")
plt.ylabel("Mean absolute log-pdf difference")
plt.legend()
plt.tight_layout()

out_path = os.path.join(combined_dir, "logp_path_diff.png")
plt.savefig(out_path, bbox_inches='tight')
plt.close()
print(f"Saved combined log-pdf difference at {out_path}")

