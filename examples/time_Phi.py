import os, sys

import torch
from torch.amp import autocast
import torch.nn.functional as F
import time



sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Phi import  Phi


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    # print torch version
    print(f"PyTorch version: {torch.__version__}")
    

    # Define the parameter sets we want to test
    precisions = ['float32', 'float16', 'bfloat16']
    precisions = ['float32','bfloat16']
    
    # Map strings to torch dtypes
    precision_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16
    }
    
    dims = [2,16,128,1024]
    widths = [16,32,64,128]
    nexs = [16,32,64,128]

    print("Testing Phi with different precisions")
    precision_test_results = []
    for dim in dims:
        for width in widths:
            for nex in nexs:
                for prec in precisions:
                    chosen_precision = precision_map[prec]
                    print("chosen prec", chosen_precision)
    
                    net = Phi(nTh=5, m=width, d=dim).to(device)
                    x = torch.randn(nex,dim+1).to(device)
                    
                    with autocast(device_type='cuda', dtype=chosen_precision):
                        g,h = net.trHess(x)
                    
                    torch.cuda.reset_peak_memory_stats(device)
                    torch.cuda.synchronize(device)
                    start_time = time.perf_counter()

                    for k in range(100):
                        with autocast(device_type='cuda', dtype=chosen_precision):
                            g,h = net.trHess(x)
                    torch.cuda.synchronize(device)
                    end_time = time.perf_counter()
            
                    elapsed_time = end_time - start_time
                    peak_memory = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    precision_test_results.append((prec, dim, width, nex, elapsed_time, peak_memory, g.dtype, h.dtype))

    print(f"{'Precision':<10} {'Dim':<10} {'Width':<10} {'Samples':<10} {'Time (s)':<10} {'Peak Mem (MB)':<15} {'g dtype':<12} {'h dtype':<12} ")
    print("-" * 65)
    for (prec_str, dim, width, nex,  elapsed_time, peak_memory, gdtype, hdtype) in precision_test_results:
        print(f"{prec_str:<10} {dim:<10d} {width:<10d} {nex:<10d} {elapsed_time:<10.4f} {peak_memory:<15.2f} {str(gdtype):<12} {str(hdtype):<12}")

    