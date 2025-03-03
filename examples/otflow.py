import os
import argparse
import glob
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast
import time
from torch.nn.functional import pad

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--adjoint', action='store_true')
parser.add_argument('--viz', action='store_true')
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--num_timesteps', type=int, default=64)
parser.add_argument('--test_freq', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--num_samples', type=int, default=512)
parser.add_argument('--num_samples_val', type=int, default=1024)
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train_dir', type=str, default=None)
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16','bfloat16'], default='float32')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchdiffeq')
parser.add_argument('--results_dir', type=str, default="./results")
args = parser.parse_args()

if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    assert args.method == 'rk4' 
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from torchmpnode import odeint
else:    
    print("using torchdiffeq")
    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]

from Phi import Phi
class OTFlow(nn.Module):
    def __init__(self, in_out_dim, hidden_dim,alpha=[1.0] * 2):
        super().__init__()
        self.in_out_dim = in_out_dim
        self.hidden_dim = hidden_dim
        self.alpha= alpha
        self.Phi = Phi(2, hidden_dim, in_out_dim, r=10, alph=alpha)

    def forward(self, t, states):
        x = states[0]

        z = pad(x, (0,1,0,0), value=t)

        gradPhi, trH = self.net.trHess(z)

        dPhi_dx = gradPhi[:,0:self.in_out_dim]
        dPhi_dt = gradPhi[:,self.in_out_dim]

        dz_dt = -(1.0/self.alpha[0]) * dPhi_dx
        dlogp_z_dt = -(1.0/self.alpha[0]) * trH
        dcost_L_dt = 0.5 * torch.norm(dPhi_dx, dim=1)**2
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
    t0 = 0
    t1 = 10
    device = torch.device('cuda:' + str(args.gpu)
                          if torch.cuda.is_available() else 'cpu')

    # model
    alpha = [1.0, 5.0] 
    func = OTFlow(in_out_dim=2, hidden_dim=args.hidden_dim, alpha=alpha).to(device)
    optimizer = optim.Adam(func.parameters(), lr=args.lr)
    p_z0 = torch.distributions.MultivariateNormal(
        loc=torch.tensor([0.0, 0.0]).to(device),
        covariance_matrix=torch.tensor([[0.1, 0.0], [0.0, 0.1]]).to(device)
    )
    loss_meter = RunningAverageMeter()
    NLL_meter = RunningAverageMeter()
    cost_L_meter = RunningAverageMeter()
    cost_HJB_meter = RunningAverageMeter()
    time_meter = RunningAverageMeter()
    mem_meter = RunningMaximumMeter()

    if args.train_dir is not None:
        if not os.path.exists(args.train_dir):
            os.makedirs(args.train_dir)
        ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
        if os.path.exists(ckpt_path):
            checkpoint = torch.load(ckpt_path)
            func.load_state_dict(checkpoint['func_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Loaded ckpt from {}'.format(ckpt_path))

    try:
        for itr in range(1, args.niters + 1):
            
            optimizer.zero_grad()

            x, logp_diff_t1 = get_batch(args.num_samples)
            
            start_time = time.perf_counter()
            torch.cuda.reset_peak_memory_stats(device)

            with autocast(device_type='cuda', dtype=args.precision):
                z_t, logp_diff_t, cost_L, cost_HJB = odeint(
                    func,
                    (x, logp_diff_t1, cost_L, cost_HJB),
                    torch.linspace(t0, t1, args.num_timesteps).to(device),
                    method=args.method
                )

                z_t0, logp_diff_t0, cost_L, cost_HJB = z_t[-1], logp_diff_t[-1], cost_L[-1], cost_HJB[-1]

                logp_x = p_z0.log_prob(z_t0).to(device) - logp_diff_t0.view(-1)
                loss = -logp_x.mean(0) + alpha[0]* cost_L.mean(0) + alpha[1] * cost_HJB.mean(0)

                loss.backward()
            optimizer.step()

            elapsed_time = time.perf_counter() - start_time
            peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)  # Convert to MB

            time_meter.update(elapsed_time)
            mem_meter.update(peak_memory)
            loss_meter.update(loss.item())
            NLL_meter.update(-logp_x.mean(0).item())
            cost_L_meter.update(cost_L.mean(0).item())
            cost_HJB_meter.update(cost_HJB.mean(0).item())

            if itr % args.test_freq == 0:
                with torch.no_grad():

                    x_val, logp_diff_t_val, cost_L_val, cost_HJB_val = get_batch(args.num_samples_val)
                    z_val, logp_diff_t_val, cost_L_val, cost_HJB_val = odeint(
                        func,
                        (x_val, logp_diff_t_val, cost_L, cost_HJB),
                        torch.linspace(t0, t1, args.num_timesteps).to(device),
                        method=args.method
                    )
                    z_val, logp_diff_t_val, cost_L_val, cost_HJB_val = z_val[-1], logp_diff_t_val[-1], cost_L[-1], cost_HJB[-1]
                    logp_x_val = p_z0.log_prob(z_val).to(device) - logp_diff_t_val.view(-1)
                    loss_val = -logp_x_val.mean(0) + alpha[0]* cost_L_val.mean(0) + alpha[1] * cost_HJB_val.mean(0)

                    with autocast(device_type='cuda', dtype=args.precision):
                        logp_diff_t_val_mp = torch.zeros_like(logp_diff_t_val)
                        cost_L_val_mp = torch.zeros_like(cost_L_val)
                        cost_HJB_val_mp = torch.zeros_like(cost_HJB_val)
                        
                        z_val_mp, logp_diff_t_val_mp, cost_L_val_mp, cost_HJB_val_mp = odeint(
                            func,
                            (x_val, logp_diff_t_val_mp, cost_L_val_mp, cost_HJB_val_mp),
                            torch.linspace(t0, t1, args.num_timesteps).to(device),
                            method=args.method
                        )
                        z_val_mp, logp_diff_t_val_mp, cost_L_val_mp, cost_HJB_val_mp = z_val_mp[-1], logp_diff_t_val_mp[-1], cost_L_val_mp[-1], cost_HJB_val_mp[-1]
                        logp_x_val_mp = p_z0.log_prob(z_val_mp).to(device) - logp_diff_t_val_mp.view(-1)
                        loss_val_mp = -logp_x_val_mp.mean(0) + alpha[0]* cost_L_val_mp.mean(0) + alpha[1] * cost_HJB_val_mp.mean(0)

            print('Iter: {}, running loss: {:.4f}, val loss {:.4f}, val loss (mp) :.4f}'.format(itr, loss_meter.avg, NLL_meter.avg, loss_val.item()), loss_val_mp.item(),
                  'running NLL: {:.4f}, val NLL: {:.4f}, val NLL (mp): {:.4f}'.format(NLL_meter.avg, logp_x_val.mean(0).item(), logp_x_val_mp.mean(0).item()), 
                   'time: {:.4f}s'.format(time_meter.avg), 'max memory: {:.0f}MB'.format(mem_meter.max))

    except KeyboardInterrupt:
        if args.train_dir is not None:
            ckpt_path = os.path.join(args.train_dir, 'ckpt.pth')
            torch.save({
                'func_state_dict': func.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, ckpt_path)
            print('Stored ckpt at {}'.format(ckpt_path))
    print('Training complete after {} iters.'.format(itr))

    if args.viz:
        viz_samples = 30000
        viz_timesteps = 41
        target_sample, _, _, _ = get_batch(viz_samples)

        if not os.path.exists(args.results_dir):
            os.makedirs(args.results_dir)
        with torch.no_grad():
            # Generate evolution of samples
            z_t0 = p_z0.sample([viz_samples]).to(device)
            logp_diff_t0 = torch.zeros(viz_samples, 1).type(torch.float32).to(device)
            cost_L_t0 = torch.zeros_like(logp_diff_t0)
            cost_HJB_t0 = torch.zeros_like(logp_diff_t0)
            z_t_samples, _, _, _ = odeint(
                func,
                (z_t0, logp_diff_t0, cost_L_t0, cost_HJB_t0),
                torch.tensor(np.linspace(t0, t1, viz_timesteps)).to(device),
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
                torch.tensor(np.linspace(t1, t0, viz_timesteps)).to(device),
                atol=1e-5,
                rtol=1e-5,
                method='dopri5',
            )

            # Create plots for each timestep
            for (t, z_sample, z_density, logp_diff) in zip(
                    np.linspace(t0, t1, viz_timesteps),
                    z_t_samples, z_t_density, logp_diff_t
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

                ax1.hist2d(*target_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                ax2.hist2d(*z_sample.detach().cpu().numpy().T, bins=300, density=True,
                           range=[[-1.5, 1.5], [-1.5, 1.5]])

                logp = p_z0.log_prob(z_density) - logp_diff.view(-1)
                ax3.tricontourf(*z_t1.detach().cpu().numpy().T,
                                np.exp(logp.detach().cpu().numpy()), 200)

                plt.savefig(os.path.join(args.results_dir, f"cnf-viz-{int(t*1000):05d}.jpg"),
                           pad_inches=0.2, bbox_inches='tight')
                plt.close()

            img, *imgs = [Image.open(f) for f in sorted(glob.glob(os.path.join(args.results_dir, f"cnf-viz-*.jpg")))]
            img.save(fp=os.path.join(args.results_dir, "cnf-viz.gif"), format='GIF', append_images=imgs,
                     save_all=True, duration=250, loop=0)

        print('Saved visualization animation at {}'.format(os.path.join(args.results_dir, "cnf-viz.gif")))
