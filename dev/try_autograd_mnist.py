import torch
from torch.amp import autocast
import torch.nn as nn

from torch.nn.functional import pad

# Set device to GPU if available (mixed precision is most useful on GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a simple linear layer
linear = nn.Linear(3, 2).to(device)

# seed the RNG
torch.manual_seed(0)


# Create an input tensor (requires_grad not needed for x in this example)
x = torch.randn(16, 64,8,8, device=device)
t = torch.tensor(0.3, device=device)

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)

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
        # ttx =pad(x, (0,0,0,0,1,0), value=t)

        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        # self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        # out = self.norm3(out)
        return out

def run_test(net, with_autocast=False):
    # Run forward pass optionally under autocast context
    params = list(net.parameters())
    if with_autocast:
        with autocast(device_type='cuda', dtype=torch.float16):
            with torch.enable_grad():
                xi = x.clone().detach().requires_grad_(True)    
                y = net(t,xi)
                print("y.dtype:", y.dtype)
                grad_out = torch.ones_like(y)
                dx, *grads = torch.autograd.grad(y, (xi, *params), grad_out,
                                    create_graph=True, allow_unused=True)
    else:
        with torch.enable_grad():
            xi = x.clone().detach().requires_grad_(True)    
            y = net(t,xi)
            print("y.dtype:", y.dtype)
            grad_out = torch.ones_like(y)
            dx, *grads = torch.autograd.grad(y, (xi, *params), grad_out,
                                    create_graph=True, allow_unused=True)
    return (dx, *grads)

net = ODEfunc(64).to(device)
# First: run without autocast.
# We enable gradient tracking for proper computation.
dydx, *grads = run_test(net, with_autocast=False)
names = [names for names, _ in net.named_parameters()]
print("Without autocast:")
print(" - Gradient w.r.t. x:", dydx.norm())
# print names an corresponding element in grads
for name, grad in zip(names, grads):
    print(f" - Gradient w.r.t. {name}:", grad.norm().item())
print()

# Second: run with autocast.
dydx, *grads = run_test(net, with_autocast=True)
print("With autocast:")
print(" - Gradient w.r.t. x:", dydx.norm())
for name, grad in zip(names, grads):
    print(f" - Gradient w.r.t. {name}:", grad.norm().item())
