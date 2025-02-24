import torch
import torch.nn as nn
import os, sys
import time
from torchdiffeq import odeint, odeint_adjoint
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint
from torch.amp import autocast
import argparse
import numpy as np
import matplotlib.pyplot as plt
## use fixed_gridnoscale.py#####



class TrueODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        # dx/dt = A*x
        # self.A = torch.tensor([[1.0, -2.0],
        #                        [2.0, -1.0]], dtype=torch.float32)
        self.A = torch.tensor([[-0.1, 2.0],
                               [-2.0, -0.1]], dtype=torch.float32)
    def forward(self, t, x):
        A = self.A.to(x.device)
        return x @ A.t()

class PrintLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.printed_backward = False 

    def forward(self, x):
        self.last_input = x 
        return self.linear(x)

    def backward_hook(self, module, grad_input, grad_output):
        if not self.printed_backward:
            print(">> Backward hook in", self.__class__.__name__)
            print("   Saved input dtype:", self.last_input.dtype)
            print("   grad_input dtypes:", [g.dtype for g in grad_input if g is not None])
            self.printed_backward = True
        return None

    def register_hooks(self):
        self.register_full_backward_hook(self.backward_hook)


class ODEFunc(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            PrintLinear(2, 50),
            nn.ReLU(),
            PrintLinear(50, 2)
        )
        for module in self.net:
            if isinstance(module, PrintLinear):
                module.register_hooks()
        self.final_printed = False     #print parameter dtypes once
        self.backward_printed = False  # print adjoint once

    def forward(self, t, x):
        if not self.final_printed and torch.allclose(t, torch.tensor(10.0, device=t.device), atol=1e-6):
            print("Learned network parameter precisions:")
            for name, param in self.net.named_parameters():
                print(f"   {name}: {param.dtype}")
            self.final_printed = True

        if not torch.is_autocast_enabled() and not self.backward_printed:
            print(">> Adjoint backward call: autocast disabled, x dtype =", x.dtype)
            self.backward_printed = True

        if not torch.is_autocast_enabled() and not self.backward_printed:
            print(">> Adjoint backward call: autocast disabled, x dtype =", x.dtype)
            self.backward_printed = True
        

        return self.net(x)

def analytic_solution(A, x, T=10):
    # print("A: ", A.shape, "x: ", x.shape, "T: ", T)
    expA = torch.matrix_exp(A.to(x.device) * T)
    expA = expA.unsqueeze(0).expand(x.size(0), -1, -1)
    return torch.bmm(x.unsqueeze(1), expA).squeeze(1)

def run_true_solution(batchsize, odetime, steps, device, seed=42): #change to f64?
    torch.manual_seed(seed)
    func = TrueODEFunc().to(device)
    #f32
    x0 = torch.randn(batchsize, 2, device=device, requires_grad=False)
    t_span = torch.linspace(0, odetime, steps, device=device)
    with torch.no_grad():
        out = odeint(func, x0, t_span, method='rk4')
    final_output = out[-1].detach().clone()
    return x0, final_output, t_span


def train_learned_solution(method, device, true_x0, true_final_output, t_span, num_iters=100, seed=42):
    """
    method: 'odeint', 'odeint_adjoint', or 'mpodeint'
    """
    torch.manual_seed(seed)
    model = ODEFunc().to(device)
    

    # true_x0 = true_x0#.to(torch.float64)
    # true_final_output = true_final_output#.to(torch.float64) remove
    # A = torch.tensor([[1.0, -2.0],
    #                     [2.0, -1.0]], dtype=torch.float32)
    A = torch.tensor([[-0.1, 2.0],
                               [-2.0, -0.1]], dtype=torch.float32)
    # print('analytic_solution: ', analytic_solution(A, true_x0[0], T=10), 'true_final_output: ', true_final_output[0])
    true_final_output = analytic_solution(A, true_x0, T=10)


    optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-3)

    losses = []

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()
    scaler = torch.cuda.amp.GradScaler()
    for i in range(num_iters):
        iter_start_time = time.perf_counter()
        optimizer.zero_grad()

        with autocast(device_type='cuda', dtype=torch.float16):
            if method == 'odeint':
                out = odeint(model, true_x0, t_span, method='rk4')
            elif method == 'odeint_adjoint':
                out = odeint_adjoint(model, true_x0, t_span, method='rk4')
            elif method == 'mpodeint':
                # print("mpodeint")
                out = mpodeint(model, true_x0, t_span, method='rk4')
            else:
                raise ValueError("Unknown integration method.")
            true_final_output = true_final_output.to(out[-1].dtype)
            loss = torch.mean(abs(out[-1] - true_final_output)) 
            # loss = torch.nn.functional.mse_loss(out[-1], true_final_output)
            # print("Loss: ", loss.item(), 'out[-1]: ', out[-1], 'true_final_output: ', true_final_output)
            # loss.backward()
        # for param in model.parameters():
        #     # if param.grad is not None:
        #     #     param.grad = param.grad.half()
        # # for param in model.parameters():
        #     # if torch.isinf(param.grad).any() or torch.isnan(param.grad).any():
        #     #     print(f'Parameter {name} has inf or nan grad')

        #     print("grad: ",torch.norm(param.grad), "dtype: ", param.grad.dtype)    
                
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

        iter_end_time = time.perf_counter()
        iter_elapsed_time = iter_end_time - iter_start_time

        if i % 50 == 0:
            print(f"Iteration {i:3d} ({method}): loss = {loss.item():.6e}, time = {iter_elapsed_time:.6f} seconds")

    torch.cuda.synchronize(device)
    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    learned_final_output = out[-1].detach().clone()
    output_diff = (true_final_output - learned_final_output).abs().mean().item()


    torch.save(model.state_dict(), f"trained_model_{method}.pth")

    return model, true_x0, learned_final_output, losses, elapsed_time, peak_memory


def rename_state_dict_keys(old_sd):
    new_sd = {}
    for k, v in old_sd.items():
        if "net.0.weight" in k:
            new_k = k.replace("net.0.weight", "net.0.linear.weight")
        elif "net.0.bias" in k:
            new_k = k.replace("net.0.bias", "net.0.linear.bias")
        elif "net.2.weight" in k:
            new_k = k.replace("net.2.weight", "net.2.linear.weight")
        elif "net.2.bias" in k:
            new_k = k.replace("net.2.bias", "net.2.linear.bias")
        else:
            new_k = k
        new_sd[new_k] = v
    return new_sd


def evaluate_on_low_precision(method, batchsize, odetime, steps, device, dtype=torch.float16, seed=1):
    """ Evaluate it on new low-precision data. """
    old_sd = torch.load(f"trained_model_{method}.pth")
    new_sd = rename_state_dict_keys(old_sd)
    
    model = ODEFunc().to(device)
    model.load_state_dict(new_sd)  # load renamed keys
    model.eval()
    torch.manual_seed(seed)

    start_eval_time = time.perf_counter()
    # Generate new low-precision test data
    new_x0 = torch.randn(batchsize, 2, device=device, requires_grad=False)
    new_x0low = torch.randn(batchsize, 2, dtype=dtype, device=device, requires_grad=False).to(torch.float32)
    new_t_span = torch.linspace(0, odetime, steps, device=device) #dtype=dtype, 

    with torch.no_grad():
    #     true_out = odeint(true_func, new_x0, new_t_span, method='rk4')

        # A = torch.tensor([[1.0, -2.0],
        #                     [2.0, -1.0]], dtype=torch.float32)
        A = torch.tensor([[-0.1, 2.0],
                               [-2.0, -0.1]], dtype=torch.float32)
        # print('analytic_solution: ', analytic_solution(A, true_x0[0], T=10), 'true_final_output: ', true_final_output[0])
        true_out = analytic_solution(A, new_x0low, T=10)

    # Model prediction using low-precision input
    with torch.no_grad(): #, autocast(device_type='cuda', dtype=dtype)
        pred_out = odeint(model, new_x0low, new_t_span, method='rk4')
    true_out = true_out.to(pred_out[-1].dtype)
    error = (true_out[-1] - pred_out[-1]).abs().mean().item() #torch.nn.functional.mse_loss
    end_eval_time = time.perf_counter()
    eval_elapsed_time = end_eval_time - start_eval_time
    print(f"Evaluation ({method}) - Data Precision: {dtype}, Mean Abs Error: {error:.6e}, Time: {eval_elapsed_time:.6f}s")
    return error


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        raise RuntimeError("Need a CUDA device for this example.")
    
    batchsize = args.batchsize
    epochs = args.epochs
    odetime = args.odetime
    steps = args.steps

    true_x0, true_final_output, t_span = run_true_solution(batchsize, odetime, steps, device, seed=42)

    integration_methods = ['mpodeint', 'odeint', 'odeint_adjoint']
    results = []

    for method in integration_methods:
        model_path = f"trained_model1_{method}.pth"
        if os.path.exists(model_path):
            print(f"{model_path} already exists.")

        else:   
            print(f"=== Training using {method} under autocast to float16===")
            model, x0, learned_output, losses, elapsed_time, peak_memory = train_learned_solution(
                method, device, true_x0, true_final_output, t_span, num_iters=epochs, seed=42)
            final_loss = losses[-1]
            results.append((method, elapsed_time, peak_memory, final_loss, 
                            (true_final_output - learned_output).abs().mean().item(),
                            learned_output.dtype))
    
    print("\n" + f"{'Method':<15} {'Time (s)':<10} {'Peak Mem (MB)':<15} {'Final Loss':<15} {'Mean Diff':<15} {'Output dtype':<12}")
    print("-" * 85)
    for (method, elapsed_time, peak_memory, final_loss, output_diff, dtype_out) in results:
        print(f"{method:<15} {elapsed_time:<10.4f} {peak_memory:<15.2f} {final_loss:<15.6e} {output_diff:<15.6e} {str(dtype_out):<12}")

    precisions = [torch.float16] #, torch.bfloat16
    results = []
    for method in integration_methods:
        for precision in precisions:
            error = evaluate_on_low_precision(method, batchsize, odetime, steps, device, dtype=precision, seed=1)
            results.append((method, precision, error))

    # Print final test results
    print("=== Test Results ===")
    print("\n" + f"{'Method':<15} {'Precision':<10} {'Mean Abs Error':<15}")
    print("-" * 45)
    for method, precision, error in results:
        print(f"{method:<15} {str(precision):<10} {error:<15.6e}")

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ODE demo')
    parser.add_argument('--batchsize', type=int, default=50, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs for training')
    parser.add_argument('--odetime', type=int, default=10, help='ODE integration time')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps for ODE integration')
    args = parser.parse_args()
    main(args)



# === Training using mpodeint under autocast to float16===
# /media/drive2/yding37/Documents/projects/torchmpnode-main/examples/comparetime.py:118: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
#   scaler = torch.cuda.amp.GradScaler()
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# >> Adjoint backward call: autocast disabled, x dtype = torch.float16
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float16
#    grad_input dtypes: [torch.float16]
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float16
#    grad_input dtypes: [torch.float16]
# Iteration   0 (mpodeint): loss = 9.226562e+00, time = 3.351060 seconds
# Iteration  50 (mpodeint): loss = 2.927246e-01, time = 2.733118 seconds
# Iteration 100 (mpodeint): loss = 1.717529e-01, time = 2.732788 seconds
# Iteration 150 (mpodeint): loss = 1.085205e-01, time = 2.733812 seconds
# Iteration 200 (mpodeint): loss = 1.178589e-01, time = 2.733916 seconds
# === Training using odeint under autocast to float16===
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float16
#    grad_input dtypes: [torch.float16]
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float32
#    grad_input dtypes: [torch.float32]
# Iteration   0 (odeint): loss = 9.224547e+00, time = 2.923415 seconds
# Iteration  50 (odeint): loss = 3.285711e-01, time = 2.572581 seconds
# Iteration 100 (odeint): loss = 1.791956e-01, time = 2.469712 seconds
# Iteration 150 (odeint): loss = 1.243475e-01, time = 2.572266 seconds
# Iteration 200 (odeint): loss = 8.031704e-02, time = 2.480157 seconds
# === Training using odeint_adjoint under autocast to float16===
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# >> Adjoint backward call: autocast disabled, x dtype = torch.float32
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float32
#    grad_input dtypes: [torch.float32]
# >> Backward hook in PrintLinear
#    Saved input dtype: torch.float32
#    grad_input dtypes: [torch.float32]
# Iteration   0 (odeint_adjoint): loss = 9.224547e+00, time = 4.031883 seconds
# Iteration  50 (odeint_adjoint): loss = 2.978170e-01, time = 3.522958 seconds
# Iteration 100 (odeint_adjoint): loss = 1.773964e-01, time = 3.517286 seconds
# Iteration 150 (odeint_adjoint): loss = 1.072661e-01, time = 3.519725 seconds
# Iteration 200 (odeint_adjoint): loss = 9.650232e-02, time = 3.519493 seconds

# Method          Time (s)   Peak Mem (MB)   Final Loss      Mean Diff       Output dtype
# -------------------------------------------------------------------------------------
# mpodeint        683.9808   17.72           9.777832e-02    1.701376e+00    torch.float16
# odeint          622.6677   40.01           1.032805e-01    1.710616e+00    torch.float32
# odeint_adjoint  880.3950   18.10           1.037148e-01    1.707971e+00    torch.float32
# >> Adjoint backward call: autocast disabled, x dtype = torch.float32
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# Evaluation (mpodeint) - Data Precision: torch.float16, Mean Abs Error: 1.961447e+00
# >> Adjoint backward call: autocast disabled, x dtype = torch.float32
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# Evaluation (odeint) - Data Precision: torch.float16, Mean Abs Error: 2.070183e+00
# >> Adjoint backward call: autocast disabled, x dtype = torch.float32
# Learned network parameter precisions:
#    0.linear.weight: torch.float32
#    0.linear.bias: torch.float32
#    2.linear.weight: torch.float32
#    2.linear.bias: torch.float32
# Evaluation (odeint_adjoint) - Data Precision: torch.float16, Mean Abs Error: 2.060956e+00
# === Test Results ===

# Method          Precision  Mean Abs Error
# ---------------------------------------------
# mpodeint        torch.float16 1.961447e+00
# odeint          torch.float16 2.070183e+00
# odeint_adjoint  torch.float16 2.060956e+00
