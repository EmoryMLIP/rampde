import os
import sys
import argparse
import time
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

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size',     type=int, default=5000)
parser.add_argument('--batch_time',    type=int, default=100)
parser.add_argument('--batch_size',    type=int, default=20)
parser.add_argument('--niters',        type=int, default=1000)
parser.add_argument('--test_freq',     type=int, default=10)
parser.add_argument('--viz',           action='store_true', default=True)
parser.add_argument('--gpu',           type=int, default=0)
parser.add_argument('--adjoint',       action='store_true')
parser.add_argument('--method',        type=str, choices=['rk4','dopri5','euler'], default='rk4')
parser.add_argument('--precision',     type=str, choices=['float32','float16','bfloat16'], default='bfloat16')
parser.add_argument('--odeint',        type=str, choices=['torchdiffeq','torchmpnode'], default='torchmpnode')
parser.add_argument('--results_dir',   type=str, default='png_rmsprop_b16')
parser.add_argument('--hidden_dim',    type=int, default=128)
parser.add_argument('--lr',            type=float, default=1e-4)
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
true_A  = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]], device=device)

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


if args.viz:
    
    fig      = plt.figure(figsize=(12,4), facecolor='white')
    ax_traj1 = fig.add_subplot(131, frameon=False)
    ax_traj2 = fig.add_subplot(132, frameon=False)
    ax_phase = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize_compare(true_y, func_d, func_m, odeint_option, itr):
    with torch.no_grad():
        pred_d = odeint_diffeq(func_d, true_y0, t, method=args.method)
        pred_m = odeint_mp  (func_m, true_y0, t, method=args.method)

        ax_traj1.cla()
        ax_traj1.set_title('y₁')
        ax_traj1.plot(t.cpu(), true_y[:,0,0].cpu(), 'g-', label='True')
        ax_traj1.plot(t.cpu(), pred_d[:,0,0].cpu(), 'b--', label='Torchdiffeq')
        ax_traj1.plot(t.cpu(), pred_m[:,0,0].cpu(), 'r--', label='Torchmpode')
        ax_traj1.legend()

        ax_traj2.cla()
        ax_traj2.set_title('y₂')
        ax_traj2.plot(t.cpu(), true_y[:,0,1].cpu(), 'g-')
        ax_traj2.plot(t.cpu(), pred_d[:,0,1].cpu(), 'b--')
        ax_traj2.plot(t.cpu(), pred_m[:,0,1].cpu(), 'r--')

        ax_phase.cla()
        ax_phase.set_title('Phase')
        ax_phase.plot(true_y[:,0,0].cpu(), true_y[:,0,1].cpu(), 'g-', alpha=0.6)
        ax_phase.plot(pred_d[:,0,0].cpu(), pred_d[:,0,1].cpu(), 'b--')
        ax_phase.plot(pred_m[:,0,0].cpu(), pred_m[:,0,1].cpu(), 'r--')

        fig.tight_layout()
        out_dir = f"{args.results_dir}"
        plt.savefig(os.path.join(out_dir, f'compare_{itr:03d}.png'))
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

    func_d = ODEFunc().to(device)
    func_m = ODEFunc().to(device)
    opt_d  = optim.RMSprop(func_d.parameters(), lr=args.lr)
    opt_m  = optim.RMSprop(func_m.parameters(), lr=args.lr)



    for odeint_option, func, optimizer, odeint_fn in [
        ('torchdiffeq', func_d, opt_d, odeint_diffeq),
        ('torchmpnode', func_m, opt_m, odeint_mp)
    ]:
        results_dir_exp = f"{args.results_dir}_{odeint_option}"
        makedirs(results_dir_exp)

        csv_path = os.path.join(results_dir_exp, f"metrics.csv")
        print(f"[INFO] Logging metrics to {csv_path}")
        csv_file = open(csv_path, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'odeint_option','iter',
            'train_loss','val_loss','time_avg_s','mem_peak_MB'
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
                pred = odeint_fn(func, y0, bt, method=args.method)
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
                    pred_val = odeint_fn(func, y0_val, bt_val,
                                         method=args.method)
                    val_loss = float(torch.mean(torch.abs(pred_val - y_val)))

                print(f"Iter {itr:4d} | {odeint_option} | "
                      f"train {loss_meter.avg:.6f} | "
                      f"val {val_loss:.6f} | "
                      f"time {time_meter.avg:.4f}s | "
                      f"mem {mem_meter.max:.1f}MB")

                csv_writer.writerow([
                    odeint_option,
                    itr,
                    loss_meter.avg,
                    val_loss,
                    time_meter.avg,
                    mem_meter.max
                ])
                csv_file.flush()

        final_path = os.path.join(results_dir_exp,
                                  f'{odeint_option}_final.pth')
        torch.save({
            'func_state_dict': func.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        print(f"[INFO] Saved final model to {final_path}")

        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)                      # skip header
            rows = [row[1:] for row in reader]  # drop the odeint_option string column
        data = np.array(rows, dtype=np.float32)

        iters      = data[:, 0]
        train_loss = data[:, 1]
        val_loss   = data[:, 2]
        time_vals  = data[:, 3]
        mem_vals   = data[:, 4]

        fig, axs = plt.subplots(2, 2, figsize=(12, 8))

        # 1) Loss subplot
        axs[0, 0].plot(iters, train_loss, label="train loss")
        axs[0, 0].plot(iters,   val_loss, label="val loss")
        axs[0, 0].set_title("Loss")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()

        # 2) Memory subplot
        axs[1, 0].plot(iters, mem_vals, label="peak mem (MB)")
        axs[1, 0].set_title("Memory Usage")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("Memory (MB)")
        axs[1, 0].legend()

        plt.tight_layout()
        stats_fig = os.path.join(results_dir_exp, "optimization_stats.png")
        plt.savefig(stats_fig, bbox_inches='tight')
        plt.close()
        print(f"Saved optimization stats at {stats_fig}")


    if args.viz:
        visualize_compare(true_y, func_d, func_m, odeint_option, itr)


