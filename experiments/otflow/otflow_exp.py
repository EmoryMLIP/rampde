import os, sys
 # Get Slurm job ID for unique result_dir file
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import datetime
import time
import shutil
import csv
import numpy as np
import glob
from PIL import Image
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
from torch.nn.functional import pad
import toy_data

# Compute project root directory (two levels up)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', choices=['swissroll', '8gaussians', 'pinwheel', 'circles', 'moons', '2spirals', 'checkerboard', 'rings'],
    type=str, default='8gaussians')

parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=5000)
parser.add_argument('--sample_freq', type=int, default=25)
parser.add_argument('--num_timesteps', type=int, default=8)
parser.add_argument('--num_timesteps_val', type=int, default=12)
parser.add_argument('--test_freq', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--lr_decay', type=float, default=.5)
parser.add_argument('--lr_decay_steps', type=int, default=500)
parser.add_argument('--alpha'  , type=str, default='1.0,30.0,1.0')
parser.add_argument('--num_samples', type=int, default=4096)
parser.add_argument('--num_samples_val', type=int, default=4096)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
parser.add_argument('--seed', type=int, default=None, help="Random seed; if not provided, no seeding will occur")
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32','tfloat32', 'float16','bfloat16'], default='float32')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchdiffeq')

args = parser.parse_args()
args.alpha = [float(a) for a in args.alpha.split(',')]

# ------------------------------
# Set up the output directory
# ------------------------------
precision_str = args.precision
precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'tfloat32': torch.float32
}
args.precision = precision_map[precision_str]

# Use provided seed in folder name; otherwise, use 'noseed'
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{args.data}_{precision_str}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join(base_dir, "results", "otflow", folder_name)
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

# Save a copy of this script in the results directory.
script_path = os.path.abspath(__file__)
shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))

# Redirect stdout and stderr to a log file.
log_path = os.path.join(result_dir, folder_name + ".txt")
log_file = open(log_path, "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file
ckpt_path = os.path.join(result_dir, 'ckpt.pth')

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')
print("Running on device:", device)

if precision_str == 'float32':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Using strict float32 precision")
elif precision_str == 'tfloat32':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Using TF32 precision")


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
print("Model checkpoint path:", ckpt_path)

# Set up CSV file to log numerical data.
csv_path = os.path.join(result_dir, folder_name + ".csv")
csv_file = open(csv_path, "w", newline="", buffering=1)
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["iter", "lr","running_loss","val_loss", "running_L", "val_L", "running_NLL", "val_NLL", "running_HJB", "val_HJB", "time_fwd","time_bwd", "max_memory_mb"])

# ------------------------------
# Set device and seeds
# ------------------------------
if args.seed is not None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
else:
    print("No seed provided; using random initialization.")


sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples"))
from utils import RunningAverageMeter, RunningMaximumMeter
if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    from torchmpnode import odeint
else:    
    print("using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint


from Phi import Phi
class OTFlow(nn.Module):
    def __init__(self, in_out_dim, hidden_dim,alpha=[1.0] * 2):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha= alpha
        self.Phi = Phi(2, hidden_dim, in_out_dim, r=2, alph=alpha)
        self.nfe = 0

    def forward(self, t, states):
        self.nfe += 1
        x = states[0]

        z = pad(x, (0,1,0,0), value=t)

        gradPhi, trH = self.Phi.trHess(z)

        dPhi_dx = gradPhi[:,0:self.in_out_dim]
        dPhi_dt = gradPhi[:,self.in_out_dim].view(-1,1)
        dz_dt = -(1.0/self.alpha[0]) * dPhi_dx
        dlogp_z_dt = -(1.0/self.alpha[0]) * trH.view(-1,1)
        dcost_L_dt = 0.5 * torch.norm(dPhi_dx, dim=1,keepdim=True)**2
        dcost_HJB_dt = torch.abs(-dPhi_dt + self.alpha[0] * dcost_L_dt)
        return (dz_dt, dlogp_z_dt, dcost_L_dt, dcost_HJB_dt)


def get_batch(num_samples):
    # points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    points = toy_data.inf_train_gen(args.data, batch_size=num_samples)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(x.shape[0], 1).type(torch.float32).to(device)
    cost_L = torch.zeros_like(logp_diff_t1)
    cost_HJB = torch.zeros_like(logp_diff_t1)

    return(x, logp_diff_t1, cost_L, cost_HJB)


if __name__ == '__main__':
    t0 = 0.0
    t1 = 1.0
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # model
    alpha = args.alpha
    lr = args.lr
    func = OTFlow(in_out_dim=2, hidden_dim=args.hidden_dim, alpha=alpha).to(device)
    optimizer = optim.Adam(func.parameters(), lr=lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
    )
    loss_meter = RunningAverageMeter()
    NLL_meter = RunningAverageMeter()
    cost_L_meter = RunningAverageMeter()
    cost_HJB_meter = RunningAverageMeter()
    fwd_time_meter = RunningAverageMeter()
    bwd_time_meter = RunningAverageMeter()
    mem_meter = RunningMaximumMeter()
    z_val_t0, logp_diff_val_t0, cost_L_val_t0, cost_HJB_val_t0 = get_batch(args.num_samples_val)
                    
    ckpt_path = os.path.join(result_dir, 'ckpt.pth')
    try:
        training_start = time.perf_counter()
        for itr in range(1, args.niters + 1):
            
            optimizer.zero_grad()
            if itr==1 or itr % args.sample_freq == 0:
                # get a batch of samples
                z_t0, logp_diff_t0,cost_L_t0, cost_HJB_t0 = get_batch(args.num_samples)
            
            torch.cuda.reset_peak_memory_stats(device)
            
            # Time forward pass
            torch.cuda.synchronize()
            fwd_start = time.perf_counter()

            with autocast(device_type='cuda', dtype=args.precision):
                z_t, logp_diff_t, cost_L_t, cost_HJB_t = odeint(
                    func,
                    (z_t0, logp_diff_t0, cost_L_t0, cost_HJB_t0),
                    torch.linspace(t0, t1, args.num_timesteps).to(device),
                    method=args.method
                )
                z_t1, logp_diff_t1, cost_L_t1, cost_HJB_t1 = z_t[-1], logp_diff_t[-1], cost_L_t[-1], cost_HJB_t[-1]

                logp_x = p_z0.log_prob(z_t1).view(-1,1) + logp_diff_t1
                loss =  alpha[0]* cost_L_t1.mean(0) - alpha[1]* logp_x.mean(0) + alpha[2] * cost_HJB_t1.mean(0)
            
            torch.cuda.synchronize()
            fwd_time = time.perf_counter() - fwd_start
            
            # Time backward pass
            torch.cuda.synchronize()
            bwd_start = time.perf_counter()
            
            loss.backward()
            
            torch.cuda.synchronize()
            bwd_time = time.perf_counter() - bwd_start
                
            optimizer.step()

            
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB

            fwd_time_meter.update(fwd_time)
            bwd_time_meter.update(bwd_time)
            mem_meter.update(peak_memory)
            loss_meter.update(loss.item())
            NLL_meter.update(-logp_x.mean(0).item())
            cost_L_meter.update(cost_L_t1.mean(0).item())
            cost_HJB_meter.update(cost_HJB_t1.mean(0).item())

            if itr % args.test_freq == 0:
                with torch.no_grad():
                    z_val_t, logp_diff_val_t, cost_L_val_t, cost_HJB_val_t = odeint(
                        func,
                        (z_val_t0, logp_diff_val_t0, cost_L_val_t0, cost_HJB_val_t0),
                        torch.linspace(t0, t1, args.num_timesteps_val).to(device),
                        method=args.method
                    )
                    z_val_t1, logp_diff_val_t1, cost_L_val_t1, cost_HJB_val_t1 = z_val_t[-1], logp_diff_val_t[-1], cost_L_val_t[-1], cost_HJB_val_t[-1]
                    logp_x_val = p_z0.log_prob(z_val_t1).to(device) + logp_diff_val_t1.view(-1)
                    loss_val =  alpha[0]* cost_L_val_t1.mean(0) - alpha[1]*logp_x_val.mean(0) + alpha[2] * cost_HJB_val_t1.mean(0)


                print('Iter: {}, lr {:.3e} | running loss: {:.3e}, val loss {:.3e}'.format(itr, lr, loss_meter.avg, loss_val.item()),
                      '| running L: {:.3e}, val L: {:.3e}'.format(cost_L_meter.avg, cost_L_val_t1.mean(0).item()), 
                      '| running NLL: {:.3e}, val NLL: {:.3e}'.format(NLL_meter.avg, -logp_x_val.mean(0).item()), 
                      '| running HJB: {:.3e}, val HJB: {:.3e}'.format(cost_HJB_meter.avg, cost_HJB_val_t1.mean(0).item()), 
                      '| fwd: {:.4f}s, bwd: {:.4f}s'.format(fwd_time_meter.avg, bwd_time_meter.avg), 
                      '| max memory: {:.0f}MB'.format(peak_memory))
                sys.stdout.flush()
                csv_writer.writerow([itr, lr, loss_meter.avg, loss_val.item(), cost_L_meter.avg, cost_L_val_t1.mean(0).item(), NLL_meter.avg, -logp_x_val.mean(0).item(), cost_HJB_meter.avg, cost_HJB_val_t1.mean(0).item(), fwd_time_meter.avg, bwd_time_meter.avg, peak_memory])
                csv_file.flush()
                
            
            if itr % args.lr_decay_steps == 0:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
                    lr = param_group['lr']
                    print("new learning rate: {}".format(lr))
                    sys.stdout.flush()
        # save the final model
        torch.save({
            'theta': func.state_dict(),
            'args': args                
        }, ckpt_path)
        print('Stored final model at {}'.format(ckpt_path))

        total_training_time = time.perf_counter() - training_start
        print('Training complete after {} iters and {:.2f} seconds.'.format(itr, total_training_time))
    except KeyboardInterrupt:
        if args.train_dir is not None:
            torch.save({
                'theta': func.state_dict(),
                'args': args                
            }, ckpt_path)
            print('Stored final model at {}'.format(ckpt_path))
    
    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _, _, _ = get_batch(viz_samples)

        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)
            cost_L_t0 = torch.zeros_like(logp_diff_t0)
            cost_HJB_t0 = torch.zeros_like(logp_diff_t0)
            z_t_samples, _, _, _ = odeint(
                func,
                (z_t0, logp_diff_t0, cost_L_t0, cost_HJB_t0),
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                method=args.method
            )

            # Generate evolution of density
            x_min, x_max = z_t0[:,0].min().item(), z_t0[:,0].max().item()
            y_min, y_max = z_t0[:,1].min().item(), z_t0[:,1].max().item()

            x = np.linspace(x_min, x_max, 100)
            y = np.linspace(x_min, x_max, 100)
            points = np.vstack(np.meshgrid(x, y)).reshape([2, -1]).T

            z_t1 = torch.tensor(points).type(torch.float32).to(device)
            logp_diff_t1 = torch.zeros(z_t1.shape[0], 1).type(torch.float32).to(device)
            cost_L_t1 = torch.zeros_like(logp_diff_t1)
            cost_HJB_t1 = torch.zeros_like(logp_diff_t1)
            z_t_density, logp_diff_t,_,_ = odeint(
                func,
                (z_t1, logp_diff_t1, cost_L_t1, cost_HJB_t1),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
                method=args.method
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
            ):
                fig = plt.figure(figsize=(12, 4), dpi=200)
                plt.tight_layout()
                # plt.axis('off')
                plt.margins(0, 0)
                fig.suptitle(f'{t:.2f}s')

                ax1 = fig.add_subplot(1, 3, 1)
                ax1.set_title('Target')
                # ax1.get_xaxis().set_ticks([])
                # ax1.get_yaxis().set_ticks([])
                ax2 = fig.add_subplot(1, 3, 2)
                ax2.set_title('Samples')
                # ax2.get_xaxis().set_ticks([])
                # ax2.get_yaxis().set_ticks([])
                ax3 = fig.add_subplot(1, 3, 3)
                ax3.set_title('Log Probability')
                # ax3.get_xaxis().set_ticks([])
                # ax3.get_yaxis().set_ticks([])

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[x_min,x_max], [y_min, y_max]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[x_min, x_max], [y_min, y_max]])

                logp = p_z0.log_prob(z_density) + logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(png_dir, f"otflow-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(png_dir, f"otflow-viz-*.jpg")))]
            img.save(fp=os.path.join(png_dir, "otflow-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)
        
        # ------------------------------
        # Create optimization stats plots
        # ------------------------------
        
        # Read data from CSV file
        with open(csv_path, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = np.array(list(reader)).astype(np.float32)

        iters = data[:, 0]
        lr_vals = data[:, 1]
        running_loss = data[:, 2]
        val_loss = data[:, 3]
        running_L = data[:, 4]
        val_L = data[:, 5]
        running_NLL = data[:, 6]
        val_NLL = data[:, 7]
        running_HJB = data[:, 8]
        val_HJB = data[:, 9]
        fwd_time = data[:, 10]
        bwd_time = data[:, 11]
        max_memory = data[:, 12]

        fig, axs = plt.subplots(2, 3, figsize=(15, 8))

        # 1) Loss function subplot
        axs[0, 0].plot(iters, running_loss, label="running loss")
        axs[0, 0].plot(iters, val_loss, label="val loss")
        axs[0, 0].set_title("Loss Function")
        axs[0, 0].set_xlabel("Iteration")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].legend()

        # 2) Transport costs subplot
        axs[0, 1].plot(iters, running_L, label="running L")
        axs[0, 1].plot(iters, val_L, label="val L")
        axs[0, 1].set_title("Transport Costs")
        axs[0, 1].set_xlabel("Iteration")
        axs[0, 1].set_ylabel("Cost")
        axs[0, 1].legend()

        # 3) NLL subplot
        axs[0, 2].plot(iters, running_NLL, label="running NLL")
        axs[0, 2].plot(iters, val_NLL, label="val NLL")
        axs[0, 2].set_title("NLL")
        axs[0, 2].set_xlabel("Iteration")
        axs[0, 2].set_ylabel("NLL")
        axs[0, 2].legend()

        # 4) HJB Penalty subplot
        axs[1, 0].plot(iters, running_HJB, label="running HJB")
        axs[1, 0].plot(iters, val_HJB, label="val HJB")
        axs[1, 0].set_title("HJB Penalty")
        axs[1, 0].set_xlabel("Iteration")
        axs[1, 0].set_ylabel("HJB")
        axs[1, 0].legend()

        # 5) Learning rate subplot (semilogy)
        axs[1, 1].semilogy(iters, lr_vals, label="learning rate")
        axs[1, 1].set_title("Learning Rate")
        axs[1, 1].set_xlabel("Iteration")
        axs[1, 1].set_ylabel("Learning Rate")
        axs[1, 1].legend()

        # 6) Timing subplot
        axs[1, 2].plot(iters, fwd_time, label="forward time")
        axs[1, 2].plot(iters, bwd_time, label="backward time")
        axs[1, 2].set_title("Forward/Backward Pass Time")
        axs[1, 2].set_xlabel("Iteration")
        axs[1, 2].set_ylabel("Time (s)")
        axs[1, 2].legend()

        plt.tight_layout()
        stats_fig_path = os.path.join(png_dir, "optimization_stats.png")
        plt.savefig(stats_fig_path, bbox_inches='tight')
        plt.close()

        print('Saved optimization stats plot at {}'.format(stats_fig_path))

        print('Saved visualization animation at {}'.format(os.path.join(png_dir, "otflow-viz.gif")))

# Close log and CSV after all output is done
csv_file.close()
log_file.close()
