# Roundoff Experiment Configuration Summary

This document summarizes the configurations tested in the roundoff experiments after the recent revision.

## Configurations Tested

The experiments now test the following configurations as requested:

### BFloat16 Configurations (no scaling needed)
1. **torchdiffeq + bf16**: Standard torchdiffeq with bfloat16 precision
2. **torchmpnode + bf16**: torchmpnode with bfloat16 precision

### Float16 Configurations (with various scaling strategies)
3. **torchdiffeq + fp16 + no_scaling**: torchdiffeq with fp16, no gradient scaling
4. **torchdiffeq + fp16 + grad_scaling**: torchdiffeq with fp16 using PyTorch's GradScaler (repeats grad computation until loss scale is found)
5. **torchmpnode + fp16 + no_scaling**: torchmpnode with fp16, explicitly disabled scaling (loss_scaler=False)
6. **torchmpnode + fp16 + grad_scaling**: torchmpnode with fp16 using default gradient scaling (loss_scaler=None)
7. **torchmpnode + fp16 + dynamic_scaling**: torchmpnode with fp16 using explicit DynamicScaler

## Implementation Details

### Scaling Parameter Mapping
- `scaler_type = None`: Used for bf16 (no scaling needed)
- `scaler_type = 'none'`: Explicitly disables scaling for fp16
- `scaler_type = 'grad'`: Gradient scaling (PyTorch GradScaler for torchdiffeq, default for torchmpnode)
- `scaler_type = 'dynamic'`: Dynamic scaling (torchmpnode only)

### Key Changes
1. **Proper bf16 handling**: BF16 configurations now correctly run without any scaling
2. **Explicit fp16 no-scaling**: Using `loss_scaler=False` for torchmpnode to disable scaling
3. **GradScaler for torchdiffeq**: Proper implementation of PyTorch's GradScaler for fp16 gradient scaling
4. **Configuration filtering**: Invalid combinations (e.g., bf16 with scaling) are automatically skipped

## Experiments

Each configuration is tested across:
- **CNF**: 5 timestep values (32, 128, 256, 512, 1024)
- **OTFlow**: 3 timestep values (2, 8, 64)
- **Methods**: Both Euler and RK4
- **Precisions**: Both float16 and bfloat16

Total configurations per experiment:
- 2 BF16 configs (both odeint types, no scaling)
- 5 FP16 configs (various scaling strategies)
- Total: 7 unique configurations × timesteps × methods