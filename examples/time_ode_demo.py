import os, sys
import time
import torch
from torch.amp import autocast
from torchdiffeq import odeint

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from torchmpnode import odeint as mpodeint

from ode_demo import get_batch, ODEFunc


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Define the parameter sets we want to test
    precisions = ['float32', 'float16', 'bfloat16']
    methods = ['rk4', 'dopri5']
    odeints = [odeint, mpodeint]

    # Map strings to torch dtypes
    precision_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }

    data_size = 20
    y0 = torch.tensor([[2., 0.]]).to(device)
    t_span = torch.linspace(0., 25., data_size).to(device)
    
    func = ODEFunc().to(device)

    # We'll first test if CNF's behaviour is changed by the precision
    print("Testing ODEFunc with different precisions")
    precision_test_results = []
    for prec_str in precisions:
        chosen_precision = precision_map[prec_str]
        
        # call once to warm up
        with autocast(device_type='cuda', dtype=chosen_precision):
            pred = func(t_span[0], y0)

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        start_time = time.perf_counter()
        
        with autocast(device_type='cuda', dtype=chosen_precision):
            pred = func(t_span[0], y0)
        
        torch.cuda.synchronize(device)
        end_time = time.perf_counter()
        
        elapsed_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        
        precision_test_results.append((prec_str, elapsed_time, peak_memory, pred.dtype))

    print(f"{'Precision':<10} {'Time (s)':<10} {'Peak Mem (MB)':<15} {'pred dtype':<12} ")
    print("-" * 65)
    for (prec_str, elapsed_time, peak_memory, dtype_pred) in precision_test_results:
        print(f"{prec_str:<10} {elapsed_time:<10.4f} {peak_memory:<15.2f} {str(dtype_pred):<12}")

    # We'll store results in a list of tuples
    # Each entry: (method, precision, elapsed_time, peak_memory, z_t dtype, logp_diff_t dtype)
    print("Testing odeint with different precisions")
    results = []
    for odeint in odeints:
        for method in methods:
            for prec_str in precisions:
                if odeint == mpodeint and method == 'dopri5':
                    # Skip this combination, as it is not supported
                    continue

                chosen_precision = precision_map[prec_str]

                
                # Create input batch
                # call once to pre-compile
                pred = odeint(func, y0, t_span, atol=1e-5, rtol=1e-5, method=method)

                
                # Setup for measuring memory and time
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
                start_time = time.perf_counter()

                # Run a single ODE solve under autocast
                with autocast(device_type='cuda', dtype=chosen_precision):
                    pred = odeint(func, y0, t_span, atol=1e-5, rtol=1e-5, method=method)

                torch.cuda.synchronize(device)
                end_time = time.perf_counter()

                elapsed_time = end_time - start_time
                peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

                results.append((method, prec_str, elapsed_time, peak_memory, pred.dtype))

    # Print results as a table
    # We will print a header and then each row
    print(f"{'Method':<10} {'Precision':<10} {'Time (s)':<10} {'Peak Mem (MB)':<15} {'pred dtype':<12} ")
    print("-" * 50)
    for (method, prec_str, elapsed_time, peak_memory, dtype_pred) in results:
        print(f"{method:<10} {prec_str:<10} {elapsed_time:<10.4f} {peak_memory:<15.2f} {str(dtype_pred):<12} ")