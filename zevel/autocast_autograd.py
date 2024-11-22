import torch
from torch.amp import autocast

device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device_str}")
device = torch.device(device_str)

# Define a simple function
def func(x):
    print(f"Dtype of x: {x.dtype}")  # x should be in float32
    return torch.matmul(x,x.T)

# Input tensor
x = torch.randn(1,4, requires_grad=True, device = device,  dtype=torch.float32)

# Use autocast and compute gradient
yt = func(x)
gradt = torch.autograd.grad(yt, x, create_graph=True)[0]
with autocast(device_type=device_str,dtype=torch.float16):
    print(f"Dtype of x: {x.dtype}")  # x should be in float32
    y = func(x)
    print(f"Dtype of y: {y.dtype}")  # y should be in float16

    grad = torch.autograd.grad(y, x, create_graph=True)[0]
    print(f"Dtype of grad: {grad.dtype}")  # grad should be in float32


print("err(y-yt) = ",(y-yt).norm().item())
print("err(grad-gradt) = ",(grad-gradt).norm().item())