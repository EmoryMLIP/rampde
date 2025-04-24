#adapted from https://github.com/EmoryMLIP/OT-Flow/blob/master/trainLargeOTflow.py
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
from mmd import mmd
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
parser.add_argument('--test_freq', type=int, default=500)
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
parser.add_argument('--results_dir', type=str, default="./results/otminiboone256")
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
    print("using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

os.makedirs(args.results_dir, exist_ok=True)
# precision mapping
precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
dtype = precision_map[args.precision]

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
    
    

if __name__ == '__main__':
    device = torch.device(f'cuda:{args.gpu}' 
                          if torch.cuda.is_available() else 'cpu')

    data    = datasets.MINIBOONE()
    train_x = torch.from_numpy(data.trn.x).float().to(device)
    val_x   = torch.from_numpy(data.val.x).float().to(device)
    d       = train_x.size(1)
    print(f"Loaded Miniboone: train={train_x.shape}, val={val_x.shape}")

    t0, t1 = 0.0, 1.0
    alpha  = [1.0, 15.0, 100.0]
    func   = OTFlow(in_out_dim=d, hidden_dim=args.hidden_dim, alpha=alpha).to(device)

    optimizer = optim.Adam(func.parameters(), lr=args.lr)
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
    func_dtype     = dtype

    try:
        # Check if a saved model exists and load it
        checkpoint_files = glob.glob(os.path.join(args.results_dir, "model_iter_*.pt"))
        if checkpoint_files:
            latest_checkpoint = max(checkpoint_files, key=os.path.getctime)
            checkpoint = torch.load(latest_checkpoint, map_location=device)
            func.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        else:

            clampMax = 5
            clampMin = -5
            for itr in range(1, args.niters+1):
                optimizer.zero_grad()
                z0, logp0, cL0, cH0 = get_minibatch(train_x, args.num_samples)
                torch.cuda.reset_peak_memory_stats(device)
                start = time.perf_counter()
                # clip parameters
                for p in func.parameters():
                    p.data = torch.clamp(p.data, clampMin, clampMax)

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(func.parameters(), max_norm=2.0)

                with autocast(device_type='cuda', dtype=func_dtype):
                    z_t, logp_t, cL_t, cH_t = odeint(
                        func,
                        (z0, logp0, cL0, cH0),
                        torch.linspace(t0, t1, args.num_timesteps, device=device),
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
                peak_m  = torch.cuda.max_memory_allocated(device)/(1024**2)
                time_meter.update(elapsed)
                mem_meter.update(peak_m)
                loss_meter.update(loss.item())
                NLL_meter.update(-logp_x.mean().item())
                cost_L_meter.update(cL1.mean().item())
                cost_HJB_meter.update(cH1.mean().item())

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
                    
                    # Save the model checkpoint
                    checkpoint_path = os.path.join(args.results_dir, f"model_iter_{itr}.pt")
                    torch.save({
                        'iteration': itr,
                        'model_state_dict': func.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss_meter.avg,
                    }, checkpoint_path)
                    print(f"Model checkpoint saved at {checkpoint_path}")

                if itr == args.lr_decay_steps:
                    for g in optimizer.param_groups:
                        g['lr'] *= args.lr_decay
                    print("Decayed LR to", optimizer.param_groups[0]['lr'])

    except KeyboardInterrupt:
        print("Interrupted. No checkpointing implemented.")


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
            fig, axs = plt.subplots(2, 2, figsize=(10, 10))
            fig.suptitle(f"Miniboone slices: dims {d1} vs {d2}", y=0.98, fontsize=18)

            im1 = axs[0,0].hist2d(testData[:,d1], testData[:,d2],
                 bins=nBins, range=boundsR0)[3]
            axs[0,0].set_title(r"$x\sim\rho_0(x)$", fontsize=16)

            im2 = axs[0,1].hist2d(modelFx[:,d1], modelFx[:,d2],
                 bins=nBins, range=bounds)[3]
            axs[0,1].set_title(r"$f(x)$", fontsize=16)

            im3 = axs[1,0].hist2d(normSamples[:,d1], normSamples[:,d2],
                 bins=nBins, range=bounds)[3]
            axs[1,0].set_title(r"$y\sim\rho_1(y)$", fontsize=16)

            im4 = axs[1,1].hist2d(modelGen[:,d1], modelGen[:,d2],
                 bins=nBins, range=boundsR0)[3]
            axs[1,1].set_title(r"$f^{-1}(y)$", fontsize=16)

            for ax, im in zip(axs.flatten(), [im1,im2,im3,im4]):
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


