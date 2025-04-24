import os, sys
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

# Compute project root directory (two levels up)
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))


parser = argparse.ArgumentParser()
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
parser.add_argument('--precision', type=str, choices=['float32', 'float16','bfloat16'], default='float32')
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
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[precision_str]

# Use provided seed in folder name; otherwise, use 'noseed'
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{precision_str}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join(base_dir, "results", "otflow", folder_name)
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
csv_writer.writerow(["iteration","running loss","val loss", "val loss (mp)", "running NLL", "val NLL", "val NLL (mp)", "running HJB", "val HJB", "val HJB (mp)",  "elapsed time (s)", "max memory (MB)"])

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

sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "examples"))
if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    from utils import RunningAverageMeter, RunningMaximumMeter

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

    def forward(self, t, states):
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
    points, _ = make_circles(n_samples=num_samples, noise=0.06, factor=0.5)
    x = torch.tensor(points).type(torch.float32).to(device)
    logp_diff_t1 = torch.zeros(num_samples, 1).type(torch.float32).to(device)
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
    func = OTFlow(in_out_dim=2, hidden_dim=args.hidden_dim, alpha=alpha).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[1.0, 0.0], [0.0, 1.0]]).to(device)
    )
    loss_meter = RunningAverageMeter()
    NLL_meter = RunningAverageMeter()
    cost_L_meter = RunningAverageMeter()
    cost_HJB_meter = RunningAverageMeter()
    time_meter = RunningAverageMeter()
    mem_meter = RunningMaximumMeter()

    ckpt_path = os.path.join(result_dir, 'ckpt.pth')
    try:
        for itr in range(1, args.niters + 1):
            
            optimizer.zero_grad()
            if itr==1 or itr % args.sample_freq == 0:
                # get a batch of samples
                z_t0, logp_diff_t0,cost_L_t0, cost_HJB_t0 = get_batch(args.num_samples)
            
            start_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

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

                loss.backward()
                # print name and norm of every parameter 
                # for name, param in func.named_parameters():
                    # print(name, param.grad.norm())
                
            optimizer.step()

            elapsed_time = time.perf_counter() - start_time
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB

            time_meter.update(elapsed_time)
            mem_meter.update(peak_memory)
            loss_meter.update(loss.item())
            NLL_meter.update(-logp_x.mean(0).item())
            cost_L_meter.update(cost_L_t1.mean(0).item())
            cost_HJB_meter.update(cost_HJB_t1.mean(0).item())

            if itr % args.test_freq == 0:
                with torch.no_grad():

                    z_val_t0, logp_diff_val_t0, cost_L_val_t0, cost_HJB_val_t0 = get_batch(args.num_samples_val)
                    z_val_t, logp_diff_val_t, cost_L_val_t, cost_HJB_val_t = odeint(
                        func,
                        (z_val_t0, logp_diff_val_t0, cost_L_val_t0, cost_HJB_val_t0),
                        torch.linspace(t0, t1, args.num_timesteps_val).to(device),
                        method=args.method
                    )
                    z_val_t1, logp_diff_val_t1, cost_L_val_t1, cost_HJB_val_t1 = z_val_t[-1], logp_diff_val_t[-1], cost_L_val_t[-1], cost_HJB_val_t[-1]
                    logp_x_val = p_z0.log_prob(z_val_t1).to(device) + logp_diff_val_t1.view(-1)
                    loss_val =  alpha[0]* cost_L_val_t1.mean(0) - alpha[1]*logp_x_val.mean(0) + alpha[2] * cost_HJB_val_t1.mean(0)

                    with autocast(device_type='cuda', dtype=args.precision):
                        logp_diff_val_mp_t0 = torch.zeros_like(logp_diff_val_t0)
                        cost_L_val_mp_t0 = torch.zeros_like(cost_L_val_t0)
                        cost_HJB_val_mp_t0 = torch.zeros_like(cost_HJB_val_t0)
                        
                        z_val_mp_t, logp_diff_val_mp_t, cost_L_val_mp_t, cost_HJB_val_mp_t = odeint(
                            func,
                            (z_val_t0, logp_diff_val_mp_t0, cost_L_val_mp_t0, cost_HJB_val_mp_t0),
                            torch.linspace(t0, t1, args.num_timesteps_val).to(device),
                            method=args.method
                        )
                        z_val_mp_t1, logp_diff_val_mp_t1, cost_L_val_mp_t1, cost_HJB_val_mp_t1 = z_val_mp_t[-1], logp_diff_val_mp_t[-1], cost_L_val_mp_t[-1], cost_HJB_val_mp_t[-1]
                        logp_x_val_mp = p_z0.log_prob(z_val_mp_t1).to(device) + logp_diff_val_mp_t1.view(-1)
                        loss_val_mp = alpha[0]* cost_L_val_mp_t1.mean(0) - alpha[1]*logp_x_val_mp.mean(0) +  alpha[2] * cost_HJB_val_mp_t1.mean(0)


                print('Iter: {}, running loss: {:.4f}, val loss {:.4f}, val loss (mp) {:.4f}'.format(itr, loss_meter.avg, loss_val.item(), loss_val_mp.item()),
                  'running NLL: {:.4f}, val NLL: {:.4f}, val NLL (mp): {:.4f}'.format(NLL_meter.avg, logp_x_val.mean(0).item(), logp_x_val_mp.mean(0).item()), 
                  'running HJB: {:.4f}, val HJB: {:.4f}, val HJB (mp): {:.4f}'.format(cost_HJB_meter.avg, cost_HJB_val_t1.mean(0).item(), cost_HJB_val_mp_t1.mean(0).item()), 
                   'time: {:.4f}s'.format(time_meter.avg), 'max memory: {:.0f}MB'.format(peak_memory))
                csv_writer.writerow([itr, loss_meter.avg, loss_val.item(), loss_val_mp.item(), NLL_meter.avg, logp_x_val.mean(0).item(), logp_x_val_mp.mean(0).item(), cost_HJB_meter.avg, cost_HJB_val_t1.mean(0).item(), cost_HJB_val_t1.mean(0).item(), time_meter.avg, peak_memory])

            
            # decay learning rate
            if itr == args.lr_decay_steps:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= args.lr_decay
                    print("new learning rate: {}".format(param_group['lr']))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    finally:
        csv_file.close()
        log_file.close()
    print('Training complete after {} iters.'.format(itr))

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
            x = np.linspace(-1.5, 1.5, 100)
            y = np.linspace(-1.5, 1.5, 100)
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
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) + logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(png_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(png_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(png_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(png_dir, "cnf-viz.gif")))
