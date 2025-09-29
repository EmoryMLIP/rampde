import os
import argparse
import logging
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.amp import autocast
import time
import datetime

import csv
import shutil
import sys

import pandas as pd
import matplotlib.pyplot as plt

from utils import RunningAverageMeter, RunningMaximumMeter

parser = argparse.ArgumentParser()
parser.add_argument('--network', type=str, choices=['resnet', 'odenet'], default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False, choices=[True, False])
parser.add_argument('--downsampling-method', type=str, default='conv', choices=['conv', 'res'])
parser.add_argument('--nepochs', type=int, default=160)
parser.add_argument('--data_aug', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
# new arguments
parser.add_argument('--method', type=str, choices=['rk4', 'dopri5'], default='rk4')
parser.add_argument('--precision', type=str, choices=['float32', 'float16','bfloat16'], default='float16')
parser.add_argument('--odeint', type=str, choices=['torchdiffeq', 'rampde'], default='rampde')

parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


if args.odeint == 'rampde':
    print("Using rampde")
    assert args.method == 'rk4' 
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from rampde import odeint
else:    
    print("using torchdiffeq")
    try:
        if args.adjoint:
            from torchdiffeq import odeint_adjoint as odeint
        else:
            from torchdiffeq import odeint
    except ImportError:
        print("Error: torchdiffeq not available. Install with 'pip install torchdiffeq' or use --odeint=rampde")
        sys.exit(1)

precision_map = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16
}
args.precision = precision_map[args.precision]


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


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc, method, tol, odeint):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.tol = tol
        self.odeint = odeint
        self.integration_time = torch.tensor(np.linspace(0, 1, 32))

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = self.odeint(self.odefunc, x, self.integration_time, rtol=self.tol, atol=self.tol,method=self.method)
        return out[-1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)



def get_mnist_loaders(data_aug=False, batch_size=128, test_batch_size=1000, perc=1.0):
    if data_aug:
        transform_train = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.ToTensor(),
        ])
    else:
        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_train),
        batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )

    train_eval_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=True, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    test_loader = DataLoader(
        datasets.MNIST(root='.data/mnist', train=False, download=True, transform=transform_test),
        batch_size=test_batch_size, shuffle=False, num_workers=2, drop_last=True
    )

    return train_loader, test_loader, train_eval_loader


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
    total_correct = 0
    for x, y in dataset_loader:
        x = x.to(device)
        y = one_hot(np.array(y.numpy()), 10)
        target_class = np.argmax(y, axis=1)
        predicted_class = np.argmax(model(x).cpu().detach().numpy(), axis=1)
        total_correct += np.sum(predicted_class == target_class)
    return total_correct / len(dataset_loader.dataset)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def get_logger(logpath, filepath, package_files=[], displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode="a")
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)
    with open(filepath, "r") as f:
        logger.info(f.read())

    for f in package_files:
        logger.info(f)
        with open(f, "r") as package_f:
            logger.info(package_f.read())

    return logger


if __name__ == '__main__':

    makedirs(args.results_dir)
    logger = get_logger(
        logpath=os.path.join(args.results_dir, 'logs'),
        filepath=os.path.abspath(__file__),
        debug=args.debug
    )
    logger.info(args)

    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    is_odenet = args.network == 'odenet'

    if args.downsampling_method == 'conv':
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
            norm(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),
        ]
    else:  # res
        downsampling_layers = [
            nn.Conv2d(1, 64, 3, 1),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
            ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ]

    feature_layers = [ODEBlock(ODEfunc(64), args.method, args.tol, odeint)] if is_odenet \
                     else [ResBlock(64, 64) for _ in range(6)]
    fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

    model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

    logger.info(model)
    logger.info('Number of parameters: {}'.format(count_parameters(model)))

    criterion = nn.CrossEntropyLoss().to(device)

    train_loader, test_loader, train_eval_loader = get_mnist_loaders(
        args.data_aug, args.batch_size, args.test_batch_size
    )

    data_gen = inf_generator(train_loader)
    batches_per_epoch = len(train_loader)

    lr_fn = learning_rate_with_decay(
        args.batch_size, batch_denom=128, batches_per_epoch=batches_per_epoch,
        boundary_epochs=[60, 100, 140],
        decay_rates=[1, 0.1, 0.01, 0.001]
    )

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

    best_acc = 0
    batch_time_meter = RunningAverageMeter()
    f_nfe_meter = RunningAverageMeter()
    b_nfe_meter = RunningAverageMeter()
    time_meter = RunningAverageMeter()
    mem_meter = RunningMaximumMeter()

    end = time.time()


    csv_path = os.path.join(args.results_dir, 'metrics.csv')
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    writer.writerow([
        'epoch', 'train_acc', 'test_acc',
        'f_nfe', 'b_nfe',
        'batch_time', 'step_time', 'max_memory'
    ])

    for itr in range(args.nepochs * batches_per_epoch):

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_fn(itr)

        optimizer.zero_grad()
        x, y = data_gen.__next__()
        x = x.to(device)
        y = y.to(device)
        start_time = time.perf_counter()
        torch.cuda.reset_peak_memory_stats(device)

        with autocast(device_type='cuda', dtype=args.precision):
            logits = model(x)
            loss = criterion(logits, y)

            if is_odenet:
                nfe_forward = feature_layers[0].nfe
                feature_layers[0].nfe = 0

            loss.backward()
        optimizer.step()

        if is_odenet:
            nfe_backward = feature_layers[0].nfe
            feature_layers[0].nfe = 0

        batch_time_meter.update(time.time() - end)
        if is_odenet:
            f_nfe_meter.update(nfe_forward)
            b_nfe_meter.update(nfe_backward)
        end = time.time()

        elapsed_time = time.perf_counter() - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

        time_meter.update(elapsed_time)
        mem_meter.update(peak_memory)

        # at the end of each epoch, compute and log
        if itr % batches_per_epoch == 0:
            epoch = itr // batches_per_epoch
            with torch.no_grad():
                train_acc = accuracy(model, train_eval_loader)
                val_acc = accuracy(model, test_loader)
                if val_acc > best_acc:
                    torch.save(
                        {'state_dict': model.state_dict(), 'args': args},
                        os.path.join(args.results_dir, 'model.pth')
                    )
                    best_acc = val_acc

                logger.info(
                    "Epoch {:04d} | Time {:.3f} ({:.3f}) | NFE-F {:.1f} | NFE-B {:.1f} | "
                    "Train Acc {:.4f} | Test Acc {:.4f} | Max Mem {:.1f}MB".format(
                        epoch, batch_time_meter.val, batch_time_meter.avg,
                        f_nfe_meter.avg, b_nfe_meter.avg,
                        train_acc, val_acc, mem_meter.max
                    )
                )

            # write metrics row
            writer.writerow([
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
    acc_plot = os.path.join(args.results_dir, 'accuracy.png')
    plt.savefig(acc_plot, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved accuracy plot at {acc_plot}")

    # # 2) NFE plot
    # plt.figure(figsize=(6, 4))
    # plt.plot(df['epoch'], df['f_nfe'], label='Forward NFE')
    # plt.plot(df['epoch'], df['b_nfe'], label='Backward NFE')
    # plt.xlabel('Epoch')
    # plt.ylabel('Number of Function Evaluations')
    # plt.legend()
    # plt.tight_layout()
    # nfe_plot = os.path.join(args.results_dir, 'nfe.png')
    # plt.savefig(nfe_plot, bbox_inches='tight')
    # plt.close()
    # logger.info(f"Saved NFE plot at {nfe_plot}")
