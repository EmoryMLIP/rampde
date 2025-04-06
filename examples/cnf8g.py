import os
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', default=True, action='store_true')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--num_samples', type=int, default=1024)
parser.add_argument('--width', type=int, default=128)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16')
# This argument is now only used as a default; we will loop over both options.
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
parser.add_argument('--results_dir', type=str, default="./results/cnf")
args = parser.parse_args()

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]

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
        with torch.set_grad_enabled(True):
            z.requires_grad_(True)
            W, B, U = self.hyper_net(t)
            Z = torch.unsqueeze(z, 0).repeat(self.width, 1, 1)
            h = torch.tanh(torch.matmul(Z, W) + B)
            dz_dt = torch.matmul(h, U).mean(0)
            dlogp_z_dt = -trace_df_dz(dz_dt, z).view(batchsize, 1)
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
        
        # print('dtypes:', params.dtype, self.fc1.weight.dtype, self.fc1.bias.dtype, self.fc2.weight.dtype, self.fc2.bias.dtype)
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
viz_timesteps = 256
t0 = 0
t1 = 1

for odeint_option in ['torchmpnode', 'torchdiffeq']:
    
    results_dir_exp = args.results_dir + '_' + odeint_option
    if not os.path.exists(results_dir_exp):
        os.makedirs(results_dir_exp)
    
    if args.train_dir is not None:
        train_dir_option = args.train_dir + '_' + odeint_option
        if not os.path.exists(train_dir_option):
            os.makedirs(train_dir_option)
    else:
        train_dir_option = None

    if odeint_option == 'torchmpnode':
        print("Using torchmpnode")
        # For torchmpnode, method must be 'rk4'
        assert args.method == 'rk4'
        import sys
        sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
        from torchmpnode import odeint as odeint_fn
    else:
        print("Using torchdiffeq")
        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint_fn
        else:
            from torchdiffeq import odeint as odeint_fn

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

    if train_dir_option is not None:
        ckpt_path = os.path.join(train_dir_option, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            optimizer.zero_grad()
            x, logp_diff_t1 = get_batch(args.num_samples, device)
            start_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

            with autocast(device_type='cuda', dtype=args.precision):
                # Integrate from t1 to t0
                ts = torch.linspace(t1, t0, 256).to(device)
                z_t, logp_diff_t = odeint_fn(
                    func,
                    (x, logp_diff_t1),
                    ts,
                    atol=1e-5,
                    rtol=1e-5,
                    method=args.method,
                )
                z_t0, logp_diff_t0 = z_t[-1], logp_diff_t[-1]
                logp_x = p_z0.log_prob(z_t0) - logp_diff_t0.view(-1)
                nll = -logp_x.mean(0)

                transport_cost = 0.0
                for i in range(len(ts) - 1):
                    dt = (ts[i+1] - ts[i]).abs()
                    v, _ = func(ts[i], (z_t[i], logp_diff_t[i]))
                    transport_cost += 0.5 * v.pow(2).sum(dim=1).mean() * dt

                loss = nll #+ transport_cost

                loss.backward()
                optimizer.step()
                scheduler.step()

            elapsed_time = time.perf_counter() - start_time
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            time_meter.update(elapsed_time)
            mem_meter.update(peak_memory)
            loss_meter.update(loss.item())

            if itr % 100 == 0:
                print('Iter: {}, running avg loss: {:.4f}'.format(itr, loss_meter.avg), 
                      'time: {:.4f} sec'.format(time_meter.avg),
                      'peak mem: {:.2f} MB'.format(mem_meter.max),
                      'lr: {:.5f}'.format(optimizer.param_groups[0]['lr']))
    except KeyboardInterrupt:
        if train_dir_option is not None:
            ckpt_path = os.path.join(train_dir_option, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored checkpoint at {}'.format(ckpt_path))
    print('Training complete after {} iters for {}'.format(itr, odeint_option))

    if args.viz:
        with torch.no_grad():
            # Evolve a large sample forward for visualization (if needed)
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
            comparison_logp_diff[odeint_option] = logp_differences

            # Create a grid for density visualization
            x_lin = np.linspace(-4, 4, 100)
            y_lin = np.linspace(-4, 4, 100)
            X, Y = np.meshgrid(x_lin, y_lin)
            grid_points = np.vstack([X.ravel(), Y.ravel()]).T
            grid_tensor = torch.tensor(grid_points, dtype=torch.float32).to(device)
            logp_diff_grid = torch.zeros(grid_tensor.shape[0], 1, device=device, dtype=torch.float32)
            
            # Compute the learned density evolution on the grid
            ts_density = torch.linspace(t1, t0, viz_timesteps).to(device)
            z_t_density, logp_diff_density = odeint_fn(
                func,
                (grid_tensor, logp_diff_grid),
                ts_density,
                atol=1e-5,
                rtol=1e-5,
                method='rk4',
            )
            

            z_final = z_t_density[-1]
            logp_final = logp_diff_density[-1]
            learned_logp_grid = p_z0.log_prob(z_final) - logp_final.view(-1)
            analytical_logp_grid = mixture_logpdf(grid_tensor)
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            cs1 = axs[0].tricontourf(grid_points[:, 0], grid_points[:, 1],
                                     learned_logp_grid.detach().cpu().numpy(), levels=50)
            axs[0].set_title("Learned log density (t=0)")
            fig.colorbar(cs1, ax=axs[0])
            z_vals = analytical_logp_grid.detach().cpu().numpy()
            z_vals = np.nan_to_num(z_vals, nan=-20, posinf=-20, neginf=-20)
            cs2 = axs[1].tricontourf(grid_points[:, 0], grid_points[:, 1],
                                     z_vals, levels=50)
            axs[1].set_title("Target log density (t=0)")
            fig.colorbar(cs2, ax=axs[1])
            plt.savefig(os.path.join(results_dir_exp, "density_comparison.png"))
            plt.close()

            density_gaussian = torch.exp(p_z0.log_prob(grid_tensor))
            density_mixture = torch.exp(mixture_logpdf(grid_tensor))


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
    print('Experiment with {} completed.'.format(odeint_option))


combined_dir = args.results_dir + '_combined'
if not os.path.exists(combined_dir):
    os.makedirs(combined_dir)


plt.figure(figsize=(6, 4))
for method, diffs in comparison_logp_diff.items():
    plt.plot(np.linspace(t0, t1, viz_timesteps), diffs, marker='o', label=method)
plt.xlabel("Time t")
plt.ylabel("Log density difference")
plt.legend()
combined_path = os.path.join(combined_dir, "logp_path_diff.png")
plt.savefig(combined_path)
plt.close()
print("Saved combined logp path difference plot at", combined_path)
