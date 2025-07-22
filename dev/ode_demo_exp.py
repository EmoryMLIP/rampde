import os
import sys
import argparse
import time
import numpy as np
import datetime
import shutil
import csv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast

from utils import RunningAverageMeter, RunningMaximumMeter

# ------------------------------
# Parse arguments
# ------------------------------
parser = argparse.ArgumentParser('ODE demo')
parser.add_argument('--data_size', type=int, default=1000)
parser.add_argument('--batch_time', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--test_freq', type=int, default=20)
parser.add_argument('--viz', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--adjoint', action='store_true')
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16', 'bfloat16'], default='float16')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')
parser.add_argument('--seed', type=int, default=None, help="Random seed; if not provided, no seeding will occur")
args = parser.parse_args()

# ------------------------------
# Set up the output directory
# ------------------------------
precision_str = args.precision
precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[precision_str]

# Use provided seed in folder name; otherwise, use 'noseed'
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{precision_str}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join("results", "ode_demo", folder_name)
os.makedirs(result_dir, exist_ok=True)
with open("result_dir.txt", "w") as f:
    f.write(result_dir)
if args.viz:
    png_dir = os.path.join(result_dir, "png")
    os.makedirs(png_dir, exist_ok=True)
else:
    png_dir = None

# Save a copy of this script in the results directory.
script_path = os.path.abspath(__file__)
shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))

# Redirect stdout and stderr to a log file.
log_path = os.path.join(result_dir, folder_name + ".txt")
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

print("Experiment started at", datetime.datetime.now())
print("Arguments:", vars(args))
print("Results will be saved in:", result_dir)

# Set up CSV file to log numerical data.
csv_path = os.path.join(result_dir, folder_name + ".csv")
csv_file = open(csv_path, "w", newline="")
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iteration", "running loss","test loss","test loss (mp)", "peak_memory_MB", "elapsed_time_s"])

# ------------------------------
# Set device and seeds
# ------------------------------
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
else:
    print("No seed provided; using random initialization.")

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)

# ------------------------------
# Import ODE solver
# ------------------------------
from torchdiffeq import odeint as odeint_fwd

if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from torchmpnode import odeint
else:
    print("Using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

# ------------------------------
# Set up ODE problem
# ------------------------------
true_y0 = torch.tensor([[2., 0.]]).to(device)
t = torch.linspace(0., 25., args.data_size).to(device)
true_A = torch.tensor([[-0.1, 2.0], [-2.0, -0.1]]).to(device)

class Lambda(nn.Module):
    def forward(self, t, y):
        return torch.mm(y**3, true_A)

with torch.no_grad():
    true_y = odeint_fwd(Lambda(), true_y0, t, method='dopri5')

def get_batch():
    s = torch.from_numpy(np.random.choice(np.arange(args.data_size - args.batch_time, dtype=np.int64),
                                            args.batch_size, replace=False))
    batch_y0 = true_y[s]  # (M, D)
    batch_t = t[:args.batch_time]  # (T)
    batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)  # (T, M, D)
    return batch_y0.to(device), batch_t.to(device), batch_y.to(device)

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

# ------------------------------
# Visualization setup
# ------------------------------
if args.viz:
    print("Visualization enabled, creating figure.")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    ax_traj = fig.add_subplot(131, frameon=False)
    ax_phase = fig.add_subplot(132, frameon=False)
    ax_vecfield = fig.add_subplot(133, frameon=False)
    plt.show(block=False)

def visualize(true_y, pred_y, odefunc, itr):
    if args.viz:
        ax_traj.cla()
        ax_traj.set_title('Trajectories')
        ax_traj.set_xlabel('t')
        ax_traj.set_ylabel('x,y')
        ax_traj.plot(t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 0],
                     t.cpu().numpy(), true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_traj.plot(t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 0], '--',
                     t.cpu().numpy(), pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_traj.set_xlim(t.cpu().min(), t.cpu().max())
        ax_traj.set_ylim(-2, 2)
        ax_traj.legend()

        ax_phase.cla()
        ax_phase.set_title('Phase Portrait')
        ax_phase.set_xlabel('x')
        ax_phase.set_ylabel('y')
        ax_phase.plot(true_y.cpu().numpy()[:, 0, 0], true_y.cpu().numpy()[:, 0, 1], 'g-')
        ax_phase.plot(pred_y.cpu().numpy()[:, 0, 0], pred_y.cpu().numpy()[:, 0, 1], 'b--')
        ax_phase.set_xlim(-2, 2)
        ax_phase.set_ylim(-2, 2)

        ax_vecfield.cla()
        ax_vecfield.set_title('Learned Vector Field')
        ax_vecfield.set_xlabel('x')
        ax_vecfield.set_ylabel('y')
        y_grid, x_grid = np.mgrid[-2:2:21j, -2:2:21j]
        grid = np.stack([x_grid, y_grid], -1).reshape(21 * 21, 2)
        dydt = odefunc(0, torch.Tensor(grid).to(device)).cpu().detach().numpy()
        mag = np.sqrt(dydt[:, 0]**2 + dydt[:, 1]**2).reshape(-1, 1)
        dydt = (dydt / mag)
        dydt = dydt.reshape(21, 21, 2)
        ax_vecfield.streamplot(x_grid, y_grid, dydt[:, :, 0], dydt[:, :, 1], color="black")
        ax_vecfield.set_xlim(-2, 2)
        ax_vecfield.set_ylim(-2, 2)

        fig.tight_layout()
        plt.savefig(os.path.join(png_dir, '{:03d}.png'.format(itr)))
        plt.draw()
        plt.pause(0.001)

class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y**3)

# ------------------------------
# Main experiment
# ------------------------------
if __name__ == '__main__':
    ii = 0
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
    else:
        print("No seed provided; using random initialization.")

    func = ODEFunc().to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=1e-3)
    end = time.time()

    time_meter = RunningAverageMeter(0.97)
    mem_meter = RunningMaximumMeter()
    loss_meter = RunningAverageMeter(0.97)
    loss_history = []

    for itr in range(1, args.niters + 1):
        optimizer.zero_grad()
        batch_y0, batch_t, batch_y = get_batch()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        with autocast(device_type='cuda', dtype=args.precision):
            pred_y = odeint(func, batch_y0, batch_t, method=args.method).to(device)
            if not torch.all(torch.isfinite(pred_y)):
                raise ValueError("pred_y is not finite")
            loss = torch.mean(torch.abs(pred_y - batch_y))
            loss.backward()
        optimizer.step()

        time_meter.update(time.time() - end)
        loss_meter.update(loss.item())
        loss_history.append(loss.item())
        if device.type == 'cuda':
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # in MB
        else:
            peak_memory = 0
        mem_meter.update(peak_memory)

        
        if itr==0 or itr % args.test_freq == 0:
            with torch.no_grad():
                pred_y_test = odeint(func, true_y0, t, method=args.method)
                test_loss = torch.mean(torch.abs(pred_y_test - true_y))
                with autocast(device_type='cuda', dtype=args.precision):
                    pred_y_mp = odeint(func, true_y0, t, method=args.method)
                    test_loss_mp = torch.mean(torch.abs(pred_y_mp - true_y))
                print('Iter {:04d} | Total Loss (fp32) {:.6f} | Total Loss (mp) {:.6f} | Time {:.4f}s | Max Memory {:.1f}MB'.format(
                    itr, test_loss.item(), test_loss_mp.item(), time_meter.avg, mem_meter.max))
                visualize(true_y, pred_y_test, func, ii)
                ii += 1
                csv_writer.writerow([itr, loss_meter.avg, test_loss.item(), test_loss_mp.item(), peak_memory, time_meter.avg])
        
        end = time.time()

    # Save model checkpoint, arguments, and loss history.
    model_save_path = os.path.join(result_dir, folder_name + ".pth")
    torch.save({
        "model_state_dict": func.state_dict(),
        "args": vars(args),
        "loss_history": loss_history,
    }, model_save_path)
    print("Saved model and run info to", model_save_path)

    # Clean up file handles.
    csv_file.close()
    log_file.close()