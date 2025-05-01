import os
job_id = os.environ.get("SLURM_JOB_ID", "")
import sys
import glob
import argparse
import time
import datetime
import math
import csv
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast

from torchdiffeq import odeint as odeint_diffeq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as odeint_mp

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(base_dir, "examples"))
from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size',     type=int, default=30000)
parser.add_argument('--batch_time',    type=int, default=100)
parser.add_argument('--batch_size',    type=int, default=20)
parser.add_argument('--niters',        type=int, default=2000)
parser.add_argument('--test_freq',     type=int, default=10)
parser.add_argument('--viz',           action='store_true', default=True)
parser.add_argument('--gpu',           type=int, default=0)
parser.add_argument('--adjoint',       action='store_true')
parser.add_argument('--method',        type=str, choices=['rk4','dopri5','euler'], default='rk4')
parser.add_argument('--precision',     type=str, choices=['float32','float16','bfloat16'], default='float16')
parser.add_argument('--odeint',        type=str, choices=['torchdiffeq','torchmpnode'], default='torchmpnode')
parser.add_argument('--results_dir',   type=str, default='./results/png_rmsproptest')
parser.add_argument('--hidden_dim',    type=int, default=128)
parser.add_argument('--lr',            type=float, default=1e-4)
parser.add_argument('--seed',         type=int, default=0)
args = parser.parse_args()

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]
device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

true_y0 = torch.tensor([[2., 0.]], device=device)
t       = torch.linspace(0., 30., args.data_size, device=device)
true_A  = torch.tensor([[-0.1, 6.0], [-2.0, -0.1]], device=device)

torch.manual_seed(0)
np.random.seed(0)
with torch.no_grad():
    true_y = odeint_diffeq(lambda tt, yy: (yy**3) @ true_A, true_y0, t, method='dopri5')

def get_batch():
    idx = np.random.choice(np.arange(args.data_size - args.batch_time),
                           args.batch_size, replace=False)
    y0 = true_y[idx]
    bt = t[:args.batch_time]
    y  = torch.stack([true_y[idx + i] for i in range(args.batch_time)], dim=0)
    return y0.to(device), bt.to(device), y.to(device)

def makedirs(d):
    os.makedirs(d, exist_ok=True)

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

seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{args.precision}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join(args.results_dir, folder_name)
os.makedirs(result_dir, exist_ok=True)
# Write result directory to a Slurm-job-specific file
result_file = f"result_dir_{job_id}.txt" if job_id else "result_dir.txt"
with open(result_file, "w") as f:
    f.write(result_dir)
if args.viz:
    png_dir = os.path.join(result_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
else:
    png_dir = None

# Redirect stdout and stderr to a log file.
log_path = os.path.join(result_dir, "log.txt")
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
# print("SLURM job id",job_id )
# print("Model checkpoint path:", ckpt_path)

if args.viz:
    
    fig      = plt.figure(figsize=(12,4), facecolor='white')
    ax_traj1 = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_phase = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize_compare(true_y, func_d, itr, result_dir):
    with torch.no_grad():
        pred_d = odeint_diffeq(func_d, true_y0, t, method=args.method)

        ax_traj1.cla()
        ax_traj1.set_title('y₁')
        ax_traj1.plot(t.cpu(), true_y[:,0,0].cpu(), 'g-', label='True')
        ax_traj1.plot(t.cpu(), pred_d[:,0,0].cpu(), 'b--', label='Estimated')
        ax_traj1.legend()

        ax_traj2.cla()
        ax_traj2.set_title('y₂')
        ax_traj2.plot(t.cpu(), true_y[:,0,1].cpu(), 'g-', label='True')
        ax_traj2.plot(t.cpu(), pred_d[:,0,1].cpu(), 'b--', label='Estimated')

        ax_phase.cla()
        ax_phase.set_title('Phase')
        ax_phase.plot(true_y[:,0,0].cpu(), true_y[:,0,1].cpu(), 'g-', alpha=0.6)
        ax_phase.plot(pred_d[:,0,0].cpu(), pred_d[:,0,1].cpu(), 'b--')


        fig.tight_layout()
       
        plt.savefig(os.path.join(result_dir, f'compare_{itr:03d}.png'))
        plt.draw(); plt.pause(0.001)

class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, args.hidden_dim),
            nn.Tanh(),
            nn.Linear(args.hidden_dim, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, t, y):
        return self.net(y**3)

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    func = ODEFunc().to(device)
    optimizer  = optim.RMSprop(func.parameters(), lr=args.lr)


    ckpts = glob.glob(os.path.join(result_dir, "final.pt"))
    if ckpts:
        latest_ckpt = max(ckpts, key=os.path.getctime)
        cp = torch.load(latest_ckpt, map_location=device)
        func.load_state_dict(cp['model_state_dict'])
        optimizer.load_state_dict(cp['optimizer_state_dict'])
        print(f"Loaded checkpoint {latest_ckpt}")
    else:
        csv_path = os.path.join(result_dir, f"{folder_name}metrics.csv")
        print(f"[INFO] Logging metrics to {csv_path}")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'iter','train_loss','val_loss','time_avg_s','mem_peak_MB'
        ])
        time_meter = RunningAverageMeter(0.97)
        mem_meter  = RunningMaximumMeter()
        loss_meter = RunningAverageMeter(0.97)

        last_time = time.time()
        for itr in range(1, args.niters+1):
            optimizer.zero_grad()
            y0, bt, y = get_batch()
            torch.cuda.reset_peak_memory_stats(device)

            with autocast(device_type='cuda', dtype=args.precision):
                pred = odeint(func, y0, bt, method=args.method)
                loss = torch.mean(torch.abs(pred - y))
                loss.backward()
            optimizer.step()

            now = time.time()
            time_meter.update(now - last_time)
            loss_meter.update(loss.item())
            peak = torch.cuda.max_memory_allocated(device) / (1024*1024)
            mem_meter.update(peak)
            last_time = now

            if itr % args.test_freq == 0:
                
                y0_val, bt_val, y_val = get_batch()
                with torch.no_grad():
                    pred_val = odeint(func, y0_val, bt_val,
                                            method=args.method)
                    val_loss = float(torch.mean(torch.abs(pred_val - y_val)))

                print(f"Iter {itr:4d} | "
                        f"train {loss_meter.avg:.6f} | "
                        f"val {val_loss:.6f} | "
                        f"time {time_meter.avg:.4f}s | "
                        f"mem {mem_meter.max:.1f}MB")

                csv_writer.writerow([
                    itr,
                    loss_meter.avg,
                    val_loss,
                    time_meter.avg,
                    mem_meter.max
                ])
                csv_file.flush()

        final_path = os.path.join(result_dir,
                                    'final.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        print(f"[INFO] Saved final model to {final_path}")

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)                       # skip header
            rows = list(reader) 

        data = np.array(rows, dtype=np.float32)

        iters      = data[:, 0]
        train_loss = data[:, 1]
        val_loss   = data[:, 2]
        time_vals  = data[:, 3]
        mem_vals   = data[:, 4]

        fig, axs = plt.subplots(1, 2, figsize=(12, 8))

        # 1) Loss subplot
        axs[0].plot(iters, train_loss, label="train loss")
        axs[0].plot(iters,   val_loss, label="val loss")
        axs[0].set_title("Loss")
        axs[0].set_xlabel("Iteration")
        axs[0].set_ylabel("Loss")
        axs[0].legend()

        # 2) Memory subplot
        axs[1].plot(iters, mem_vals, label="peak mem (MB)")
        axs[1].set_title("Memory Usage")
        axs[1].set_xlabel("Iteration")
        axs[1].set_ylabel("Memory (MB)")
        axs[1].legend()

        plt.tight_layout()
        stats_fig = os.path.join(result_dir, "optimization_stats.png")
        plt.savefig(stats_fig, bbox_inches='tight')
        plt.close()
        print(f"Saved optimization stats at {stats_fig}")


    if args.viz:
        visualize_compare(true_y, func, itr, result_dir)


