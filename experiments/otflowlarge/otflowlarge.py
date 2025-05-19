#adapted from https://github.com/EmoryMLIP/OT-Flow/blob/master/trainLargeOTflow.py
import os
job_id = os.environ.get("SLURM_JOB_ID", "")
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


from torch.nn.functional import pad
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
print("Base directory:", base_dir)
sys.path.insert(0, base_dir)

from src.mmd import mmd
sys.path.insert(0, base_dir)
import datasets


parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true', default=True,
                    help="2D‐slice visualization on high-D data")
parser.add_argument('--niters', type=int, default=8000)
parser.add_argument('--num_timesteps', type=int, default=6)
parser.add_argument('--num_timesteps_val', type=int, default=10)
parser.add_argument('--val_freq', type=int, default=100)
parser.add_argument('--lr', type=float, default=5e-3)
parser.add_argument('--num_samples', type=int, default=2000)
parser.add_argument('--num_samples_val', type=int, default=5000)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str,
                    choices=['float32', 'float16','bfloat16'], default='float16')
parser.add_argument('--odeint', type=str,
                    choices=['torchdiffeq', 'torchmpnode'], default='torchdiffeq')
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--data', type=str, default='miniboone',
                    choices=['miniboone', 'bsds300', 'power', 'gas', 'hepmass'],
                    help="Dataset to use")
parser.add_argument('--results_dir', type=str, default="./results/otflowlarge") #results/test
parser.add_argument('--early_stopping', type=int, default=20,
                    help="# of val checks w/o improvement before dropping LR")
parser.add_argument('--lr_drop', type=float, default=10.0,
                    help="Factor to divide LR by on plateau")
parser.add_argument('--drop_freq', type=int, default=0,
                    help="Drop LR every drop_freq iterations")
parser.add_argument('--alpha', type=str, default='1.0,100.0,15.0',
                    help="alpha values for L, NLL, HJB respectively")

args = parser.parse_args()
args.alpha = [float(a) for a in args.alpha.split(',')]


if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

ndecs = 0
n_vals_wo_improve = 0
best_val_loss = float('inf')

def update_lr(optimizer, n_vals_without_improvement):
    global ndecs
    if ndecs == 0 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / args.lr_drop
        ndecs = 1
    elif ndecs == 1 and n_vals_without_improvement > args.early_stopping:
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / (args.lr_drop ** 2)
        ndecs = 2
    else:
        ndecs += 1
        for param_group in optimizer.param_groups:
            param_group["lr"] = args.lr / (args.lr_drop ** ndecs)


# os.makedirs(args.results_dir, exist_ok=True)
# seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
# timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# folder_name = f"{args.precision}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
# # Save a copy of this script in the results directory.
# script_path = os.path.abspath(__file__)
# shutil.copy(script_path, os.path.join(args.results_dir, os.path.basename(script_path)))

# Use provided seed in folder name; otherwise, use 'noseed'
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{args.data}_{args.precision}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join(base_dir, "results", "otflowlarge", folder_name) 
os.makedirs(result_dir, exist_ok=True)

result_file = f"result_dir_{job_id}.txt" if job_id else "result_dir.txt"
with open(result_file, "w") as f:
    f.write(result_dir)
if args.viz:
    png_dir = os.path.join(result_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
else:
    png_dir = None

script_path = os.path.abspath(__file__)
shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))

# Redirect stdout and stderr to a log file.
log_path = os.path.join(result_dir, "log.txt")
if os.path.exists(log_path):
    log_path = os.path.join(result_dir, "newlog.txt")
log_file = open(log_path, "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file


device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)


# Print environment and hardware info for reproducibility and debugging
print("Environment Info:")
print(f"  Python version: {sys.version}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")
print(f"  GPU Device Name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")
print(f"  Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

print("Experiment started at", datetime.datetime.now())
print("Arguments:", vars(args))
print("Results will be saved in:", result_dir)
print("SLURM job id",job_id )
# print("Model checkpoint path:", ckpt_path)



#check if CSV exists
csv_path = os.path.join(result_dir, folder_name + ".csv")
if not os.path.exists(csv_path):
    print(f"Writing metrics to {csv_path}")
    csv_file = open(csv_path, "w", newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["iteration", "learning rate","running loss","val loss", "val loss (mp)","running L", "val L", "val L(mp)", "running NLL", "val NLL", "val NLL (mp)", "running HJB", "val HJB", "val HJB (mp)",  "time fwd","time bwd", "max memory (MB)","nfe fwd", "nfe bwd", "mmd"])

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

def batch_iter(X, batch_size, shuffle=True):
    if shuffle:
        idxs = torch.randperm(X.shape[0], device=X.device)
    else:
        idxs = torch.arange(X.shape[0], device=X.device)
    for batch_idxs in idxs.split(batch_size):
        yield X[batch_idxs]


sys.path.insert(0, os.path.join(base_dir, "examples"))
from utils import RunningAverageMeter, RunningMaximumMeter

# choose ODE solver
if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    from torchmpnode import odeint
else:
    print("Using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint
from Phi import Phi

class OTFlow(nn.Module):
    def __init__(self, in_out_dim, hidden_dim, alpha=[1.0]*2):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.Phi = Phi(2, hidden_dim, in_out_dim, alph=alpha)
        self.nfe = 0

    def forward(self, t, states):
        self.nfe += 1
        x = states[0]
        z = pad(x, (0,1,0,0), value=t)
        gradPhi, trH = self.Phi.trHess(z)
        dPhi_dx = gradPhi[:, :self.in_out_dim]
        dPhi_dt = gradPhi[:, self.in_out_dim].view(-1,1)

        dz_dt       = -(1.0/self.alpha[0]) * dPhi_dx
        dlogp_dt    = -(1.0/self.alpha[0]) * trH.view(-1,1)
        cost_L_dt   = 0.5 * torch.sum(torch.pow(dz_dt, 2) , 1 ,keepdims=True) #0.5 * torch.norm(dPhi_dx, dim=1, keepdim=True)**2
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
    print(f"Loaded Data: train={train_x.shape}, val={val_x.shape}")

    # setup model, optimizer, meters
    t0, t1 = 0.0, 1.0

    alpha = args.alpha
    func   = OTFlow(in_out_dim=d, hidden_dim=args.hidden_dim, alpha=alpha).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cov = torch.eye(d, device=device)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.zeros(d, device=device),
        covariance_matrix=cov
    )

    loss_meter     = RunningAverageMeter()
    NLL_meter      = RunningAverageMeter()
    cost_L_meter   = RunningAverageMeter()
    cost_HJB_meter = RunningAverageMeter()
    # time_meter     = RunningAverageMeter()
    time_fwd = RunningAverageMeter()
    time_bwd = RunningAverageMeter()
    mem_meter      = RunningMaximumMeter()
    nfe_fwd = 0
    nfe_bwd = 0

    try:
        # Check if a saved model exists and load it
        ckpts = glob.glob(os.path.join(result_dir, "model_best.pt"))
        if ckpts:
            latest_ckpt = max(ckpts, key=os.path.getctime)
            cp = torch.load(latest_ckpt, map_location=device)
            func.load_state_dict(cp['model_state_dict'])
            optimizer.load_state_dict(cp['optimizer_state_dict'])
            print(f"Loaded checkpoint {latest_ckpt}")
        else:
            itr = 1
            clampMax, clampMin = 1.5, -1.5
            stop_training = False
            while itr <= args.niters:
                for x_batch in batch_iter(train_x, args.num_samples, shuffle=True):
                    optimizer.zero_grad()
                    z0 = x_batch
                    B = z0.size(0)
                    logp0 = torch.zeros(B, 1, dtype=torch.float32, device=device)
                    cL0 = logp0.clone()
                    cH0 = logp0.clone()

                    torch.cuda.reset_peak_memory_stats(device)
                    start = time.perf_counter()

                    for p in func.parameters():
                        p.data = torch.clamp(p.data, clampMin, clampMax)

                    with autocast(device_type='cuda', dtype=func_dtype):
                        # print('==============================training==============================')
                        start_time = time.perf_counter()
                        func.nfe = 0
                        ts = torch.linspace(t0, t1, args.num_timesteps+1, device=device) 
                        z_t, logp_t, cL_t, cH_t = odeint(
                            func,
                            (z0, logp0, cL0, cH0),
                            ts,
                            method=args.method
                        )
                        z1, logp1, cL1, cH1 = z_t[-1], logp_t[-1], cL_t[-1], cH_t[-1]
                        logp_x = p_z0.log_prob(z1).view(-1,1) + logp1
                        loss   = (-alpha[1]*logp_x.mean()
                                + alpha[0]*cL1.mean()
                                + alpha[2]*cH1.mean())
                        time_fwd.update(time.perf_counter() - start_time)
                        nfe_fwd += func.nfe
                        func.nfe = 0

                        start_time = time.perf_counter()
                        loss.backward()
                        time_bwd.update(time.perf_counter() - start_time)
                        nfe_bwd += func.nfe

                    optimizer.step()

                    # elapsed = time.perf_counter() - start
                    peak_m  = torch.cuda.max_memory_allocated(device) / (1024**2)

                    # update meters
                    # time_meter.update(elapsed)
                    mem_meter.update(peak_m)
                    loss_meter.update(loss.item())
                    NLL_meter.update((-logp_x.mean()).item())
                    cost_L_meter.update(cL1.mean().item())
                    cost_HJB_meter.update(cH1.mean().item())

                    # validation & CSV logging
                    if itr % args.val_freq == 0 or itr == args.niters:
                        with torch.no_grad():
                            # print('==============================validation==32============================')
                            vz0 = next(batch_iter(val_x, args.num_samples, shuffle=False))
                            B = vz0.size(0)
                            vpl0 = torch.zeros(B, 1, dtype=torch.float32, device=device)
                            vL0 = vpl0.clone()
                            vH0 = vpl0.clone()
                            vz_t, vpl_t, vL_t, vH_t = odeint(
                                func,
                                (vz0, vpl0, vL0, vH0),
                                torch.linspace(t0, t1, args.num_timesteps_val+1, device=device),
                                method=args.method
                            )
                            vz1, vpl1, vL1, vH1 = vz_t[-1], vpl_t[-1], vL_t[-1], vH_t[-1]
                            logp_val = p_z0.log_prob(vz1).view(-1,1) + vpl1
                            loss_val = (-alpha[1]*logp_val.mean()
                                        + alpha[0]*vL1.mean()
                                        + alpha[2]*vH1.mean())
                            
                            with autocast(device_type='cuda', dtype=func_dtype):
                                # print('==============================validation mp==============================')
                                logp_diff_val_mp_t0 = torch.zeros_like(vpl0)
                                cost_L_val_mp_t0 = torch.zeros_like(vL0)
                                cost_HJB_val_mp_t0 = torch.zeros_like(vH0)
                                
                                z_val_mp_t, logp_diff_val_mp_t, cost_L_val_mp_t, cost_HJB_val_mp_t = odeint(
                                    func,
                                    (vz0, logp_diff_val_mp_t0, cost_L_val_mp_t0, cost_HJB_val_mp_t0),
                                    torch.linspace(t0, t1, args.num_timesteps_val+1).to(device),
                                    method=args.method
                                )
                                z_val_mp_t1, logp_diff_val_mp_t1, cost_L_val_mp_t1, cost_HJB_val_mp_t1 = z_val_mp_t[-1], logp_diff_val_mp_t[-1], cost_L_val_mp_t[-1], cost_HJB_val_mp_t[-1]
                                logp_x_val_mp = p_z0.log_prob(z_val_mp_t1).to(device) + logp_diff_val_mp_t1.view(-1)
                                loss_val_mp = alpha[0]* cost_L_val_mp_t1.mean(0) - alpha[1]*logp_x_val_mp.mean(0) +  alpha[2] * cost_HJB_val_mp_t1.mean(0)
                            B=10000  
                            y = p_z0.sample([B]).to(device)
                            logp0 = torch.zeros(B, 1, dtype=torch.float32, device=device)
                            cL0   = logp0.clone()
                            cH0   = logp0.clone()
                            t_grid_inv = torch.linspace(t1, t0, args.num_timesteps_val+1, device=device)
                            z_inv_t, _, _, _ = odeint(
                                func,
                                (y, logp0, cL0, cH0),
                                t_grid_inv,
                                method=args.method
                            )
                            z_inv1    = z_inv_t[-1]
                            modelGen  = z_inv1[:, :d].cpu().numpy()
                            val_batch = vz0.cpu().numpy()
                            # compute MMD between generated samples and val batch
                            mmd_val = mmd(modelGen, val_batch)


                        print('Iter: {}, lr {:.3e} running loss: {:.3e}, val loss {:.3e}, val loss (mp) {:.3e}'.format(itr, optimizer.param_groups[0]['lr'], loss_meter.avg, loss_val.item(), loss_val_mp.item()),
                            'running L: {:.3e}, val L: {:.3e}, val L (mp): {:.3e}'.format(cost_L_meter.avg, vL1.mean().item(), cost_L_val_mp_t1.mean(0).item()), 
                            'running NLL: {:.3e}, val NLL: {:.3e}, val NLL (mp): {:.3e}'.format(NLL_meter.avg,(-logp_val.mean()).item(), -logp_x_val_mp.mean(0).item()), 
                            'running HJB: {:.3e}, val HJB: {:.3e}, val HJB (mp): {:.3e}'.format(cost_HJB_meter.avg, vH1.mean().item(), cost_HJB_val_mp_t1.mean(0).item()), 
                            'time (fwd): {:.4f}s'.format(time_fwd.avg),'time (bwd): {:.4f}s'.format(time_bwd.avg), 'max memory: {:.0f}MB'.format(peak_m),
                            'nfe (fwd) : {}, nfe (bwd): {}'.format(nfe_fwd, nfe_bwd), 'mmd: {:.5e}'.format(mmd_val))

                        csv_writer.writerow([itr, optimizer.param_groups[0]['lr'], loss_meter.avg, loss_val.item(), loss_val_mp.item(), cost_L_meter.avg, vL1.mean().item(), cost_L_val_mp_t1.mean(0).item(), NLL_meter.avg, (-logp_val.mean()).item(), -logp_x_val_mp.mean(0).item(), cost_HJB_meter.avg, vH1.mean().item(), cost_HJB_val_mp_t1.mean(0).item(), time_fwd.avg, time_bwd.avg, peak_m,nfe_fwd,nfe_bwd, mmd_val])

                        csv_file.flush()
                        nfe_fwd = nfe_bwd = 0

                        if loss_val.item() < best_val_loss:
                            best_val_loss = loss_val.item()
                            n_vals_wo_improve = 0
                            ckpt_path = os.path.join(result_dir, f"model_best.pt")
                            torch.save({
                                'iteration': itr,
                                'model_state_dict': func.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'loss': loss_meter.avg,
                            }, ckpt_path)
                            print(f"Model checkpoint saved at {ckpt_path}")
                        else:
                            n_vals_wo_improve += 1


                    if args.drop_freq == 0:
                        if n_vals_wo_improve > args.early_stopping:
                            if ndecs > 2:
                                print("early stopping engaged")
                                ckpt_path = os.path.join(result_dir, f"model_final.pt")
                                torch.save({
                                    'iteration': itr,
                                    'model_state_dict': func.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                    'loss': loss_meter.avg,
                                }, ckpt_path)
                                print(f"Model checkpoint saved at {ckpt_path}")
                                stop_training = True
                                break
                            else:
                                update_lr(optimizer, n_vals_wo_improve)
                                n_vals_wo_improve = 0
                    else:
                        if itr % args.drop_freq == 0:
                            for pg in optimizer.param_groups:
                                pg['lr'] /= args.lr_drop
                            print(f"Decayed LR to {optimizer.param_groups[0]['lr']}")

                    itr += 1

                # if itr > args.niters:
                #     break
                if stop_training:
                    break
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
    val_loss_mp = data[:, 4]
    run_L       = data[:, 5]
    val_L       = data[:, 6]
    val_L_mp    = data[:, 7]
    run_NLL     = data[:, 8]
    val_NLL     = data[:, 9]
    val_NLL_mp  = data[:, 10]
    run_HJB     = data[:, 11]
    val_HJB     = data[:, 12]
    val_HJB_mp  = data[:, 13]
    time_fwd    = data[:, 14]
    time_bwd    = data[:, 15]
    max_mem     = data[:, 16]
    nfe_fwd     = data[:, 17]
    nfe_bwd     = data[:, 18]
    mmd_vals   = data[:, 19]



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

    # 5) NFE
    axs[1, 1].plot(iters, nfe_fwd, label="forward")
    axs[1, 1].plot(iters, nfe_bwd, label="backward")
    axs[1, 1].set_title("NFE")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].legend()

    # 6) MMD
    axs[1, 2].plot(iters, mmd_vals, label="MMD")
    axs[1, 2].set_title("MMD")
    axs[1, 2].set_xlabel("Iteration")
    axs[1, 2].legend()


    plt.tight_layout()
    stats_fig = os.path.join(result_dir, "optimization_stats.png")
    plt.savefig(stats_fig, bbox_inches='tight')
    plt.close()
    print(f"Saved optimization stats plot at {stats_fig}")

    if args.viz:
        
        print("Generating 2D‐slice visualizations…")

        val_np = val_x.cpu().numpy()
        N      = min(val_np.shape[0], args.num_samples_val)
        testData = val_np[:N]
        # Load the best model checkpoint
        best_ckpt_path = os.path.join(result_dir, "model_best.pt")
        if os.path.exists(best_ckpt_path):
            print(f"Loading best model checkpoint from {best_ckpt_path}")
            cp = torch.load(best_ckpt_path, map_location=device)
            func.load_state_dict(cp['model_state_dict'])
        else:
            print(f"Best model checkpoint not found at {best_ckpt_path}. Using current model.")

        #forward map f(x)
        with torch.no_grad():
            # x_batch = torch.from_numpy(testData).to(device)
            # vz0, vpl0, vL0, vH0 = get_minibatch(val_x, args.num_samples_val)
            vz0 = next(batch_iter(val_x, args.num_samples_val, shuffle=False))
            B = vz0.size(0)
            vpl0 = torch.zeros(B, 1, dtype=torch.float32, device=device)
            vL0 = vpl0.clone()
            vH0 = vpl0.clone()
      
            t_grid = torch.linspace(t0, t1, args.num_timesteps_val+1, device=device)
            z_t, logp_t, cL_t, cH_t   = odeint(
                func,
                (vz0, vpl0, vL0, vH0),
                t_grid,
                method=args.method
            )
        z_fwd = z_t[-1]
        modelFx = z_fwd[:, :d].cpu().numpy()

        #Gaussian samples & inverse map
        y = p_z0.sample([N]).to(device)

        logp0 = torch.zeros(N, 1, device=device)
        cL0   = torch.zeros_like(logp0)
        cH0   = torch.zeros_like(logp0)
        # for backward, reverse the time grid
        t_grid_inv = torch.linspace(t1, t0, args.num_timesteps_val+1, device=device)
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

        nSamples = min(testData.shape[0], modelGen.shape[0], 3000)
        testSamps  = testData[:nSamples, :]
        modelSamps = modelGen[:nSamples, :]

        mmd_val = mmd(modelSamps, testSamps)
        print(f"MMD( ourGen , ρ0 )  num(ourGen)={modelSamps.shape[0]}, "
              f"num(ρ0)={testSamps.shape[0]} : {mmd_val:.5e}")

        # Inverse-error evaluation 

        nData = testData.shape[0]
        # allocate arrays of shape (nData, d)
        modelFx_inv   = np.zeros(testData.shape)
        modelFinvfx   = np.zeros(testData.shape)

        idx = 0
        batch_size = args.num_samples_val
        for start in range(0, nData, batch_size):
            end = min(start + batch_size, nData)
            # grab numpy slice and convert
            x0_np = testData[start:end]
            x0 = torch.from_numpy(x0_np).float().to(device)

            with torch.no_grad():
                # forward map 
                logp0 = torch.zeros(x0.size(0), 1, device=device)
                cL0   = torch.zeros_like(logp0)
                cH0   = torch.zeros_like(logp0)
                ts    = torch.linspace(t0, t1, args.num_timesteps_val+1, device=device)
                z_t, _, _, _  = odeint(func,
                                       (x0, logp0, cL0, cH0),
                                       ts,
                                       method=args.method)
                fx     = z_t[-1][:, :d]

                # inverse map 
                ts_inv   = torch.linspace(t1, t0, args.num_timesteps_val+1, device=device)
                z_inv_t, _, _, _ = odeint(func,
                                          (fx, logp0, cL0, cH0),
                                          ts_inv,
                                          method=args.method)
                finvfx = z_inv_t[-1][:, :d]

            sz = end - start
            modelFx_inv[start:end, :] = fx.cpu().numpy()
            modelFinvfx[start:end, :] = finvfx.cpu().numpy()
            idx += sz

        # compute per-sample L2 error and then the mean
        inv_errors   = np.linalg.norm(modelFinvfx - testData, axis=1)
        mean_inv_err = inv_errors.mean()
        print(f"[Inverse Error] Mean L2 ‖f⁻¹(f(x)) – x‖: {mean_inv_err:.5e}")



        nBins = 33
        LOW, HIGH = -4, 4
        if hasattr(data, 'gas') and result_dir.lower().find('gas')>=0:
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
            out_file = os.path.join(result_dir,
                f"slice_{d1}v{d2}.pdf")
            plt.savefig(out_file, dpi=400)
            plt.close(fig)

        print(f"Saved visualizations in {result_dir}")
    else:
        print("Visualization skipped (use --viz).")
