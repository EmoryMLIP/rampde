#adapted from https://github.com/EmoryMLIP/OT-Flow/blob/master/trainLargeOTflow.py
import os
import argparse
import glob
import csv
import shutil
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
import datetime

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.mmd import mmd
from torch.nn.functional import pad

import datasets
from Phi import Phi
from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true', default=True,
                    help="2D‐slice visualization on high-D data")
parser.add_argument('--niters', type=int, default=15000)
parser.add_argument('--num_timesteps', type=int, default=8)
parser.add_argument('--test_freq', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay', type=float, default=.5)
parser.add_argument('--lr_decay_steps', type=int, default=20)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--num_samples_val', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str,
                    choices=['float32', 'float16','bfloat16'], default='float16')
parser.add_argument('--odeint', type=str,
                    choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data', type=str, default='miniboone',
                    choices=['miniboone', 'bsds300', 'power', 'gas', 'hepmass'],
                    help="Dataset to use")
parser.add_argument('--results_dir', type=str, default="./results/otf16-mpnodet")
args = parser.parse_args()

if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

# choose ODE solver
if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from torchmpnode import odeint
else:
    print("Using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

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


#check if CSV exists
csv_path = os.path.join(args.results_dir, "metrics.csv")
if not os.path.exists(csv_path):
    print(f"Writing metrics to {csv_path}")
    csv_file = open(csv_path, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "iter", "lr",
        "running_loss", "val_loss",
        "running_L",   "val_L",
        "running_NLL", "val_NLL",
        "running_HJB", "val_HJB",
        "time", "max_memory"
    ])
    csv_file.flush()
else:
    print(f"Metrics CSV already exists at {csv_path}")
    csv_file = open(csv_path, "a", newline='')
    csv_writer = csv.writer(csv_file)

# precision mapping
precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
func_dtype = precision_map[args.precision]

def get_minibatch(X, num_samples):
    idx = torch.randint(0, X.size(0), (num_samples,), device=X.device)
    x = X[idx]
    B = x.size(0)
    z = torch.zeros(B, 1, dtype=torch.float32, device=X.device)
    return x, z.clone(), z.clone(), z.clone()

class OTFlow(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, alpha=[1.0]*2):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.Phi = Phi(2, hidden_dim, in_out_dim, alph=alpha)

    def forward(self, t, states):
        x = states[0]
        z = pad(x, (0,1,0,0), value=t)
        gradPhi, trH = self.Phi.trHess(z)
        dPhi_dx = gradPhi[:, :self.in_out_dim]
        dPhi_dt = gradPhi[:, self.in_out_dim].view(-1,1)

        dz_dt       = -(1.0/self.alpha[0]) * dPhi_dx
        dlogp_dt    = -(1.0/self.alpha[0]) * trH.view(-1,1)
        cost_L_dt   = 0.5 * torch.norm(dPhi_dx, dim=1, keepdim=True)**2
        cost_HJB_dt = torch.abs(-dPhi_dt + self.alpha[0]*cost_L_dt)
        return dz_dt, dlogp_dt, cost_L_dt, cost_HJB_dt

def load_data(name):

    if name == 'bsds300':
        return datasets.BSDS300()

    elif name == 'power':
        return datasets.POWER()

    elif name == 'gas':
        return datasets.GAS()

    elif name == 'hepmass':
        return datasets.HEPMASS()

    elif name == 'miniboone':
        return datasets.MINIBOONE()

    else:
        raise ValueError('Unknown dataset')
    
if __name__ == '__main__':
    device = torch.device(f'cuda:{args.gpu}' 
                          if torch.cuda.is_available() else 'cpu')




    # load data
    # data    = datasets.MINIBOONE()
    data = load_data(args.data)
    train_x = torch.from_numpy(data.trn.x).float().to(device)
    val_x   = torch.from_numpy(data.val.x).float().to(device)
    d       = train_x.size(1)
    print(f"Loaded Miniboone: train={train_x.shape}, val={val_x.shape}")

    # setup model, optimizer, meters
    t0, t1 = 0.0, 1.0
    alpha  = [1.0, 15.0, 100.0]
    func   = OTFlow(in_out_dim=d, hidden_dim=args.hidden_dim, alpha=alpha).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cov = torch.eye(d, device=device) * 0.1
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(d, device=device),
        covariance_matrix=cov
    )

    loss_meter     = RunningAverageMeter()
    NLL_meter      = RunningAverageMeter()
    cost_L_meter   = RunningAverageMeter()
    cost_HJB_meter = RunningAverageMeter()
    time_meter     = RunningAverageMeter()
    mem_meter      = RunningMaximumMeter()

    try:
        # Check if a saved model exists and load it
        ckpts = glob.glob(os.path.join(args.results_dir, "model_iter_*.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            cp = torch.load(latest_ckpt, map_location=device)
            func.load_state_dict(cp['model_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            print(f"Loaded checkpoint {latest_ckpt}")
        else:
            clampMax, clampMin = 5, -5
            for itr in range(1, args.niters+1):
                optimizer.zero_grad()
                z0, logp0, cL0, cH0 = get_minibatch(train_x, args.num_samples)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()

                # clamp & grad clip
                for p in func.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)
                torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)

                with autocast(device_type='cuda', dtype=func_dtype):
                    ts = torch.linspace(t0, t1, args.num_timesteps, device=device)
                    z_t, logp_t, cL_t, cH_t = odeint(
                        func,
                        (z0, logp0, cL0, cH0),
                        ts,
                        method=args.method
                    )
                    z1, logp1, cL1, cH1 = z_t[-1], logp_t[-1], cL_t[-1], cH_t[-1]
                    logp_x = p_z0.log_prob(z1).view(-1,1) + logp1
                    loss   = (-alpha[2]*logp_x.mean()
                              + alpha[0]*cL1.mean()
                              + alpha[1]*cH1.mean())
                    loss.backward()

                optimizer.step()
                elapsed = time.perf_counter() - start
                peak_m  = torch.cuda.max_memory_allocated(device) / (1024**2)

                # update meters
                time_meter.update(elapsed)
                mem_meter.update(peak_m)
                loss_meter.update(loss.item())
                NLL_meter.update((-logp_x.mean()).item())
                cost_L_meter.update(cL1.mean().item())
                cost_HJB_meter.update(cH1.mean().item())

                # validation & CSV logging
                if itr % args.test_freq == 0:
                    with torch.no_grad():
                        vz0, vpl0, vL0, vH0 = get_minibatch(val_x, args.num_samples_val)
                        vz_t, vpl_t, vL_t, vH_t = odeint(
                            func,
                            (vz0, vpl0, vL0, vH0),
                            torch.linspace(t0, t1, args.num_timesteps, device=device),
                            method=args.method
                        )
                        vz1, vpl1, vL1, vH1 = vz_t[-1], vpl_t[-1], vL_t[-1], vH_t[-1]
                        logp_val = p_z0.log_prob(vz1).view(-1,1) + vpl1
                        loss_val = (-alpha[2]*logp_val.mean()
                                    + alpha[0]*vL1.mean()
                                    + alpha[1]*vH1.mean())

                    print(f"[Iter {itr:5d}] train loss {loss_meter.avg:.4f}, "
                          f"val loss {loss_val:.4f}, time {time_meter.avg:.3f}s, "
                          f"mem {mem_meter.max:.0f}MB")

                    # write one row to CSV
                    csv_writer.writerow([
                        itr,
                        optimizer.param_groups[0]['lr'],
                        loss_meter.avg,
                        loss_val.item(),
                        cost_L_meter.avg,
                        vL1.mean().item(),
                        NLL_meter.avg,
                        (-logp_val.mean()).item(),
                        cost_HJB_meter.avg,
                        vH1.mean().item(),
                        time_meter.avg,
                        mem_meter.max
                    ])
                    csv_file.flush()

                    # save checkpoint
                    ckpt_path = os.path.join(args.results_dir, f"model_iter_{itr}.pt")
                    torch.save({
                        'iteration': itr,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_meter.avg,
                    }, ckpt_path)
                    print(f"Model checkpoint saved at {ckpt_path}")

                if itr == args.lr_decay_steps:
                    for g in optimizer.param_groups:
                        g['lr'] *= args.lr_decay
                    print(f"Decayed LR to {optimizer.param_groups[0]['lr']}")

    except KeyboardInterrupt:
        print("Interrupted. Exiting training loop.")

    csv_file.close()

    # ------------------------------
    # Create optimization stats plots
    # ------------------------------
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        next(reader)
        data = np.array(list(reader), dtype=np.float32)

    iters       = data[:, 0]
    lr_vals     = data[:, 1]
    run_loss    = data[:, 2]
    val_loss    = data[:, 3]
    run_L       = data[:, 4]
    val_L       = data[:, 5]
    run_NLL     = data[:, 6]
    val_NLL     = data[:, 7]
    run_HJB     = data[:, 8]
    val_HJB     = data[:, 9]
    max_mem     = data[:, 11]

    fig, axs = plt.subplots(2, 3, figsize=(15, 8))

    # 1) Loss
    axs[0, 0].plot(iters, run_loss,    label="running loss")
    axs[0, 0].plot(iters, val_loss,    label="val loss")
    axs[0, 0].set_title("Loss Function")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].legend()

    # 2) Transport cost L
    axs[0, 1].plot(iters, run_L,    label="running L")
    axs[0, 1].plot(iters, val_L,    label="val L")
    axs[0, 1].set_title("Transport Cost L")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].legend()

    # 3) NLL
    axs[0, 2].plot(iters, run_NLL,  label="running NLL")
    axs[0, 2].plot(iters, val_NLL,  label="val NLL")
    axs[0, 2].set_title("Negative Log‐Likelihood")
    axs[0, 2].set_xlabel("Iteration")
    axs[0, 2].legend()

    # 4) HJB penalty
    axs[1, 0].plot(iters, run_HJB,  label="running HJB")
    axs[1, 0].plot(iters, val_HJB,  label="val HJB")
    axs[1, 0].set_title("HJB Penalty")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].legend()

    # 5) Learning Rate
    axs[1, 1].semilogy(iters, lr_vals, label="learning rate")
    axs[1, 1].set_title("Learning Rate")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].legend()

    # 6) Max memory
    axs[1, 2].plot(iters, max_mem, label="max memory (MB)")
    axs[1, 2].set_title("Max Memory")
    axs[1, 2].set_xlabel("Iteration")
    axs[1, 2].legend()

    plt.tight_layout()
    stats_fig = os.path.join(args.results_dir, "optimization_stats.png")
    plt.savefig(stats_fig, bbox_inches='tight')
    plt.close()
    print(f"Saved optimization stats plot at {stats_fig}")

    if args.viz:
        
        print("Generating 2D‐slice visualizations…")

        val_np = val_x.cpu().numpy()
        N      = min(val_np.shape[0], args.num_samples_val)
        testData = val_np[:N]

        #forward map f(x)
        with torch.no_grad():
            # x_batch = torch.from_numpy(testData).to(device)
            vz0, vpl0, vL0, vH0 = get_minibatch(val_x, args.num_samples_val)
            t_grid = torch.linspace(t0, t1, args.num_timesteps, device=device)
            z_t, logp_t, cL_t, cH_t   = odeint(
                func,
                (vz0, vpl0, vL0, vH0),
                t_grid,
                method=args.method
            )
        z_fwd = z_t[-1]
        modelFx = z_fwd[:, :d].cpu().numpy()

        #Gaussian samples & inverse map f⁻¹(y)
        y = p_z0.sample([N]).to(device)

        logp0 = torch.zeros(N, 1, device=device)
        cL0   = torch.zeros_like(logp0)
        cH0   = torch.zeros_like(logp0)
        # for backward, reverse the time grid
        t_grid_inv = torch.linspace(t1, t0, args.num_timesteps, device=device)
        with torch.no_grad():
            z_inv_t, _, _, _ = odeint(
                func,
                (y, logp0, cL0, cH0),
                t_grid_inv,
                method=args.method
            )
        z_inv = z_inv_t[-1]
        modelGen    = z_inv[:, :d].cpu().numpy()
        normSamples = y.cpu().numpy()

        nSamples = min(testData.shape[0], modelGen.shape[0])
        testSamps  = testData[:nSamples, :]
        modelSamps = modelGen[:nSamples, :]

        mmd_val = mmd(modelSamps, testSamps)
        print(f"MMD( ourGen , ρ0 )  num(ourGen)={modelSamps.shape[0]}, "
              f"num(ρ0)={testSamps.shape[0]} : {mmd_val:.5e}")

        nBins = 33
        LOW, HIGH = -4, 4
        if hasattr(data, 'gas') and args.results_dir.lower().find('gas')>=0:
            LOW, HIGH = -2, 2
        LOWrho0, HIGHrho0 = LOW, HIGH

        bounds    = [[LOW, HIGH], [LOW, HIGH]]
        boundsR0  = [[LOWrho0, HIGHrho0], [LOWrho0, HIGHrho0]]

        for d1 in range(0, d-1, 2):
            d2 = d1 + 1
            fig, axs = plt.subplots(1, 2, figsize=(10, 5))
            # fig.suptitle(f"Miniboone slices: dims {d1} vs {d2}", y=0.98, fontsize=18)

            im1 = axs[0].hist2d(testData[:,d1], testData[:,d2],
                 bins=nBins, range=boundsR0)[3]
            axs[0].set_title(r"$x\sim\rho_0(x)$", fontsize=16)

            # im2 = axs[0,1].hist2d(modelFx[:,d1], modelFx[:,d2],
            #      bins=nBins, range=bounds)[3]
            # axs[0,1].set_title(r"$f(x)$", fontsize=16)

            # im3 = axs[1,0].hist2d(normSamples[:,d1], normSamples[:,d2],
            #      bins=nBins, range=bounds)[3]
            # axs[1,0].set_title(r"$y\sim\rho_1(y)$", fontsize=16)

            im2 = axs[1].hist2d(modelGen[:,d1], modelGen[:,d2],
                 bins=nBins, range=boundsR0)[3]
            axs[1].set_title(r"$f^{-1}(y)$", fontsize=16)

            for ax, im in zip(axs.flatten(), [im1, im2]): #im1,im2,im3,
                fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
                ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect('equal')

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out_file = os.path.join(args.results_dir,
                f"slice_{d1}v{d2}.pdf")
            plt.savefig(out_file, dpi=400)
            plt.close(fig)

        print(f"Saved visualizations in {args.results_dir}")
    else:
        print("Visualization skipped (use --viz).")

