import torch
from torch.amp import autocast
import torch.nn as nn

# Set device to GPU if available (mixed precision is most useful on GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create a simple linear layer
linear = nn.Linear(3, 2).to(device)

# Create an input tensor (requires_grad not needed for x in this example)
x = torch.randn(3, 3, device=device)
t = torch.tensor(0.0, device=device)

class NonlinearODE(torch.nn.Module):
    def __init__(self, dim):
        super(NonlinearODE, self).__init__()
        # Always stored in float32.
        self.A = torch.nn.Linear(dim, dim)
        self.B = torch.nn.Linear(dim, dim,  bias=False)

    def forward(self, t,  x):
        # Cast input to float32 so that the parameters are used in float32.
        return self.B(torch.tanh(self.A( x)))

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

net = NonlinearODE(3).to(device)
# First: run without autocast.
# We enable gradient tracking for proper computation.
dydx, *grads = run_test(net, with_autocast=False)
names = [names for names, _ in net.named_parameters()]
print("Without autocast:")
print(" - Gradient w.r.t. x:", dydx)
# print names an corresponding element in grads
for name, grad in zip(names, grads):
    print(f" - Gradient w.r.t. {name}:", grad)
print()

# Second: run with autocast.
dydx, *grads = run_test(net, with_autocast=True)
print("With autocast:")
print(" - Gradient w.r.t. x:", dydx)
for name, grad in zip(names, grads):
    print(f" - Gradient w.r.t. {name}:", grad)
