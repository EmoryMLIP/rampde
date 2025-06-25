import os, sys
job_id = os.environ.get("SLURM_JOB_ID", "")
import argparse
import logging
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
# import torchvision.datasets as datasets
from torchvision.datasets import STL10
import torchvision.transforms as transforms
from torch.amp import autocast
import time
import datetime

import csv
import shutil
import sys

import pandas as pd
import matplotlib.pyplot as plt

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, os.path.join(base_dir, "examples"))
from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=100)
parser.add_argument('--weight_decay', type=float, default=1e-4)
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'euler'], default='rk4')
parser.add_argument('--precision', type=str, choices=['tfloat32', 'float32', 'float16','bfloat16'], default='float16')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'torchmpnode'], default='torchmpnode')

parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--test_freq', type=int, default=1,
                    help='evaluate / log every N training steps')
parser.add_argument('--width', type=int, default=64,
                    help='Base channel width (default: 64)')
args = parser.parse_args()


if args.odeint == 'torchmpnode':
    print("Using torchmpnode")
    assert args.method == 'rk4' 
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
    from torchmpnode import odeint, DynamicScaler
else:    
    print("using torchdiffeq")
    from torchdiffeq import odeint
precision_str = args.precision
precision_map = {
    'float32': torch.float32,
    'tfloat32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]


os.makedirs(args.results_dir, exist_ok=True)
seed_str = f"seed{args.seed}" if args.seed is not None else "noseed"
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
folder_name = f"{precision_str}_{args.odeint}_{args.method}_{seed_str}_{timestamp}"
result_dir = os.path.join(base_dir, "results", "ode_stl10", folder_name)
ckpt_path = os.path.join(result_dir, 'ckpt.pth')
os.makedirs(result_dir, exist_ok=True)
with open("result_dir.txt", "w") as f:
    f.write(result_dir)
script_path = os.path.abspath(__file__)
shutil.copy(script_path, os.path.join(result_dir, os.path.basename(script_path)))

# Redirect stdout and stderr to a log file.
log_path = os.path.join(result_dir, folder_name + ".txt")
log_file = open(log_path, "w", buffering=1)
sys.stdout = log_file
sys.stderr = log_file

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


if precision_str == 'float32':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print("Using strict float32 precision")
elif precision_str == 'tfloat32':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("Using TF32 precision")

torch.backends.cudnn.verbose = True
torch.backends.cudnn.benchmark = True

# Print environment and hardware info for reproducibility and debugging
print("Environment Info:")
print(f"  Python version: {sys.version}")
print(f"  PyTorch version: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  CUDA version: {torch.version.cuda}")
print(f"  cuDNN version: {torch.backends.cudnn.version()}")
print("   cuDNN enabled:", torch.backends.cudnn.enabled)
print(f"  GPU Device Name: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")
print(f"  Current Device: {torch.cuda.current_device() if torch.cuda.is_available() else 'N/A'}")

print("Experiment started at", datetime.datetime.now())
print("Arguments:", vars(args))
print("Results will be saved in:", result_dir)
print("SLURM job id",job_id )
print("Model checkpoint path:", ckpt_path)

class ODEFunc(nn.Module):
    def __init__(self, ch, t_grid, act=nn.ReLU(inplace=True)):
        super().__init__()

        n_steps = len(t_grid)-1
        
        # Create a parameter for the weight and bias banks
        init_weight = torch.randn(ch, ch, 3, 3).mul_(0.1)
        init_bias = torch.zeros(ch)
        
        self.weight_bank = nn.Parameter(init_weight.unsqueeze(0).repeat(2*n_steps+1, 1, 1, 1, 1))
        self.bias_bank = nn.Parameter(init_bias.unsqueeze(0).repeat(2*n_steps+1, 1))
        
        # per-step norms remain (cheap, no descriptor)
        self.norms = nn.ModuleList([nn.InstanceNorm2d(ch, affine=True)
                                   for _ in range(2*n_steps+1)])

        # map rounded time → index
        t_grid = torch.linspace(t_grid[0], t_grid[-1], 2*n_steps+1)
        self.lookup = {round(t.item(), 6): i for i, t in enumerate(t_grid)}
        
        self.padding = 1
        self.stride = 1
        self.act = act

    def forward(self, t, y):
        idx = self.lookup[round(t.item(), 6)]
        
        # Use F.conv2d directly with the indexed weight and bias
        # This creates a computational graph that autograd can track
        weight = self.weight_bank[idx]
        bias = self.bias_bank[idx]
        
        # Do forward pass using functional operations
        y = F.conv2d(y, weight, bias, stride=self.stride, padding=self.padding)
        y = self.act(y)
        y = self.norms[idx](y)
        
        # Use transposed convolution with the same weight
        # Note: For convolution transpose, we need to permute the dimensions
        y = F.conv_transpose2d(
            y, 
            weight,  # transpose in/out channels
            bias=None, 
            stride=self.stride, 
            padding=self.padding
        )
        
        return -y
    
class ODEBlock(nn.Module):
    def __init__(self, func,t_grid, solver="rk4", steps=4, loss_scaler=None):
        super().__init__()
        self.func   = func
        self.solver = solver
        self.t_grid = t_grid
        self.loss_scaler = loss_scaler

    def forward(self, x):
        if self.loss_scaler is not None:
            out = odeint(self.func, x, self.t_grid, method=self.solver, loss_scaler=self.loss_scaler)
        else:
            out = odeint(self.func, x, self.t_grid, method=self.solver)
        return out[-1]

class MPNODE_STL10(nn.Module):
    def __init__(self,width):
        super().__init__()
        ch = width
        t_grid = torch.linspace(0, 1.0, 5)
        # 1) stem: 3×96×96 -> 64×96×96
        self.stem = nn.Conv2d(3, ch, 3, padding=1, bias=True)
        self.norm1 = nn.InstanceNorm2d(ch, affine=True)

        # 2) ODE block #1
        if args.odeint == 'torchmpnode':
            S1 = DynamicScaler(args.precision)
        else:
            S1 = None
        self.ode1 = ODEBlock(ODEFunc(ch, t_grid), t_grid, solver="rk4", steps=4, loss_scaler=S1)
        # self.norm2 = nn.InstanceNorm2d(ch)

        # 3) down-sample stride-2 3×3
        self.conn1 = nn.Conv2d(ch, 2*ch, 1,  padding=0, bias=True)
        self.avg1 = nn.AvgPool2d(2, stride=2)
        self.norm3 = nn.InstanceNorm2d(2*ch, affine=True)
        # self.norm3 = nn.InstanceNorm2d(ch)

        # 4) ODE block #2
        if args.odeint == 'torchmpnode':
            S2 = DynamicScaler(args.precision)
        else:
            S2 = None
        self.ode2 = ODEBlock(ODEFunc(2*ch, t_grid), t_grid, solver="rk4", steps=4, loss_scaler=S2)
        self.conn2 = nn.Conv2d(2*ch, 4*ch, 1,  padding=0, bias=True)
        self.avg2 = nn.AvgPool2d(2, stride=2)
        self.norm4 = nn.InstanceNorm2d(4*ch, affine=True)
        
        if args.odeint == 'torchmpnode':
            S3 = DynamicScaler(args.precision)
        else:
            S3 = None
        self.ode3 = ODEBlock(ODEFunc(4*ch, t_grid), t_grid, solver="rk4", steps=4, loss_scaler=S3)
        # self.norm5 = nn.InstanceNorm2d(4*ch)
        
        # self.conn3 = nn.Conv2d(4*ch, 4*ch, 1,  padding=0, bias=True)
        # self.avg3 = nn.AvgPool2d(2, stride=2)
        self.act = nn.ReLU(inplace=True)
        # 5) global avg-pool + FC
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # 128×1×1
            nn.Flatten(),
            nn.Linear(4*ch, 10)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.norm1(x)
        x = self.act(x)
        
        x = self.ode1(x)
        # x = self.norm2(x)
        x = self.conn1(x)
        x = self.norm3(x)
        x = self.act(x)
        x = self.avg1(x)
        
        x = self.ode2(x)
        x = self.conn2(x)
        x = self.norm4(x)
        x = self.act(x)
        x = self.avg2(x)
        
        x = self.ode3(x)
        # x = self.conn3(x)
        # x = self.norm5(x)
        # x = self.act(x)
        # x = self.avg3(x)
        
        return self.head(x)


def get_stl10_loaders(batch_size=128,
                      test_batch_size=1000,
                      perc=1.0):
    """Return train_loader, test_loader, train_eval_loader for STL-10.

    Parameters
    ----------
    data_aug : bool          – if True, use random crop + flip.
    batch_size : int
    test_batch_size : int
    perc : float             – unused (kept for interface compatibility).

    All loaders use drop_last=True so the batch count is deterministic.
    """

    # normalization constants for STL-10 RGB
    mean = (0.4467, 0.4398, 0.4066)
    std  = (0.2241, 0.2210, 0.2239)

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(96, scale=(0.5, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
        # optional regularizers:
        # transforms.RandomErasing(p=0.2, scale=(0.02,0.33), ratio=(0.3,3.3)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    # ----- full 5k train set (load twice with different transforms) -----
    full_train_aug  = STL10(root='.data/stl10', split='train',
                            download=True, transform=transform_train)
    full_train_eval = STL10(root='.data/stl10', split='train',
                            download=True, transform=transform_test)

    # ----- deterministic split -----
    g = torch.Generator().manual_seed(42)
    idx = torch.randperm(len(full_train_aug), generator=g)
    idx_train, idx_val = idx[:int(4000*perc)], idx[4000:]          # 4 k / 1 k

    train_set = Subset(full_train_aug,  idx_train)
    val_set   = Subset(full_train_eval, idx_val)         # no augmentation
    train_eval_set   = Subset(full_train_eval, idx_train)         # no augmentation

    # ----- loaders -----
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, drop_last=True)
    val_loader   = DataLoader(val_set,   batch_size=test_batch_size,
                              shuffle=False, num_workers=2)
    train_eval_loader   = DataLoader(train_eval_set,   batch_size=test_batch_size,
                              shuffle=False, num_workers=2)
    # test_loader  = DataLoader(STL10(root='.data/stl10', split='test',
                                    # download=True, transform=transform_eval),
                            #   batch_size=test_batch_size,
                            #   shuffle=False, num_workers=2)
    return train_loader, val_loader, train_eval_loader


    # train_loader = DataLoader(
    #     STL10(root='.data/stl10', split='train', download=True,
    #           transform=transform_train),
    #     batch_size=batch_size, shuffle=True,
    #     num_workers=2, drop_last=True
    # )

    # train_eval_loader = DataLoader(
    #     STL10(root='.data/stl10', split='train', download=True,
    #           transform=transform_test),
    #     batch_size=test_batch_size, shuffle=False,
    #     num_workers=2, drop_last=True
    # )

    # test_loader = DataLoader(
    #     STL10(root='.data/stl10', split='test', download=True,
    #           transform=transform_test),
    #     batch_size=test_batch_size, shuffle=False,
    #     num_workers=2, drop_last=True
    # )

    # return train_loader, test_loader, train_eval_loader


def inf_generator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, (x, y) in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def learning_rate_with_decay(batch_size, batch_denom, batches_per_epoch, boundary_epochs, decay_rates):
    initial_learning_rate = args.lr * batch_size / batch_denom

    boundaries = [int(batches_per_epoch * epoch) for epoch in boundary_epochs]
    vals = [initial_learning_rate * decay for decay in decay_rates]

    def learning_rate_fn(itr):
        lt = [itr < b for b in boundaries] + [True]
        i = np.argmax(lt)
        return vals[i]

    return learning_rate_fn


def one_hot(x, K):
    return np.array(x[:, None] == np.arange(K)[None, :], dtype=int)


def accuracy(model, dataset_loader):
    loss = 0.0
    total_correct = 0
    N = len(dataset_loader.dataset)
    for x, y in dataset_loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss += F.cross_entropy(logits, y).item() * y.size(0)
        predicted_class = logits.argmax(dim=1)
        total_correct += (predicted_class == y).sum().item()
    return total_correct / N, loss/N


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)



if __name__ == '__main__':

    model = MPNODE_STL10(args.width).to(device)
    print(model)
    print('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_stl10_loaders(
         args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    # optimizer = torch.optim.AdamW(model.parameters(), 3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.nepochs*batches_per_epoch, eta_min=1e-4)
    # scheduler that does nothing

    

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    train_loss_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    time_meter = RunningAverageMeter()
    mem_meter = RunningMaximumMeter()

    csv_path = os.path.join(result_dir, folder_name + ".csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'step', 'epoch',
        'train_acc', 'test_acc','running_loss','train_loss', 'test_loss',
        'f_nfe', 'b_nfe',
        'batch_time', 'step_time', 'max_memory'
    ])

    for itr in range(args.nepochs * batches_per_epoch):
        
        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        torch.cuda.synchronize() 
        torch.cuda.reset_peak_memory_stats(device)
        start_time = time.perf_counter()
        
        with autocast(device_type='cuda', dtype=args.precision):
            logits = model(x)
            loss = criterion(logits.float(), y)
            nfe_forward = -1
            loss.backward()
        optimizer.step()
        scheduler.step()
        
        for param in model.parameters():
            param.data = param.data.clamp_(-1, 1)
        torch.cuda.synchronize() 
        elapsed_time = time.perf_counter() - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)        
        nfe_backward = -1

        batch_time_meter.update(elapsed_time)
        f_nfe_meter.update(nfe_forward)
        b_nfe_meter.update(nfe_backward)
        train_loss_meter.update(loss.item())        
        time_meter.update(elapsed_time)
        mem_meter.update(peak_memory)

        # evaluate / log every test_freq steps
        if itr % batches_per_epoch*args.test_freq == 0:
            epoch = itr // batches_per_epoch

            with torch.no_grad():
                with autocast(device_type='cuda', dtype=args.precision):
                    train_acc, train_loss = accuracy(model, train_eval_loader)
                    val_acc, val_loss = accuracy(model, test_loader)
                    if val_acc > best_acc:
                        torch.save(
                            {'state_dict': model.state_dict(), 'args': args}, ckpt_path)
                        best_acc = val_acc

                print(
                    "Step {:06d} | Epoch {:04d} | Time {:.3f} ({:.3f}) | "
                    "Running Loss {:.4f} | Train Loss {:.4f} | Test Loss {:.4f} | "
                    "NFE-F {:.1f} | NFE-B {:.1f} | Train Acc {:.4f} | "
                    "Test Acc {:.4f} | Max Mem {:.1f}MB".format(
                        itr, epoch, batch_time_meter.val, batch_time_meter.avg,
                        train_loss_meter.val, train_loss, val_loss,
                        f_nfe_meter.avg, b_nfe_meter.avg,
                        train_acc, val_acc, mem_meter.max
                    )
                )

            # write metrics row
            # print(json.dumps({"epoch": epoch,"train_acc": train_acc, "val_acc": val_acc}), flush=True)        
            writer.writerow([
                itr,
                epoch,
                train_acc,
                val_acc,
                f_nfe_meter.avg,
                b_nfe_meter.avg,
                batch_time_meter.avg,
                time_meter.avg,
                mem_meter.max
            ])
            csv_file.flush()


    csv_file.close()


    df = pd.read_csv(csv_path)

    # 1) accuracy plot
    plt.figure(figsize=(6, 4))
    plt.plot(df['epoch'], df['train_acc'], label='Train Acc')
    plt.plot(df['epoch'], df['test_acc'], label='Test Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    acc_plot = os.path.join(result_dir, 'accuracy.png')
    plt.savefig(acc_plot, bbox_inches='tight')
    plt.close()
    print(f"Saved accuracy plot at {acc_plot}")

    # # 2) NFE plot
    # plt.figure(figsize=(6, 4))
    # plt.plot(df['epoch'], df['f_nfe'], label='Forward NFE')
    # plt.plot(df['epoch'], df['b_nfe'], label='Backward NFE')
    # plt.xlabel('Epoch')
    # plt.ylabel('Number of Function Evaluations')
    # plt.legend()
    # plt.tight_layout()
    # nfe_plot = os.path.join(result_dir, 'nfe.png')
    # plt.savefig(nfe_plot, bbox_inches='tight')
    # plt.close()
    # print(f"Saved NFE plot at {nfe_plot}")
