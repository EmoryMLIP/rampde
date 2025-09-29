# rampde Experiments

This directory contains experiments for evaluating **rampde**, a high-performance mixed-precision library for Neural Ordinary Differential Equations (ODEs). The experiments compare performance across different precisions, ODE solvers, and datasets.

## Overview

The experiment system has been designed for systematic evaluation with:
- **Unified CSV output format** across all experiments
- **Separate forward/backward pass timing** for performance analysis
- **Automated result collection and visualization**
- **Consistent naming conventions** for easy comparison

## Experiment Types

### Neural ODE Classification
- **`mnist/`** - MNIST digit classification with Neural ODEs
- **`stl10/`** - STL-10 image classification with Neural ODEs

### Continuous Normalizing Flows
- **`cnf/`** - 2D toy dataset experiments with CNFs
- **`otflow/`** - 2D optimal transport flow experiments
- **`otflowlarge/`** - Large-scale optimal transport on high-dimensional datasets

### Method Comparisons
- **`ode/`** - Basic ODE solver method comparisons

## Standardized Experiment Structure

### CSV Output Format
All experiments now output CSV files with standardized headers:

```csv
iter,epoch,lr,train_loss,val_loss,time_fwd,time_bwd,max_memory_mb,train_acc,val_acc
```

**Core Metrics:**
- `iter` - Iteration number
- `epoch` - Epoch number (when applicable)
- `lr` - Current learning rate
- `train_loss`, `val_loss` - Training and validation losses
- `time_fwd`, `time_bwd` - Forward and backward pass times (seconds)
- `max_memory_mb` - Peak GPU memory usage (MB)
- `train_acc`, `val_acc` - Training and validation accuracies

**Experiment-specific metrics** may include additional columns like `running_L`, `running_NLL`, etc.

### Directory Structure
Each experiment run creates a directory with:
```
results/{experiment_type}/{folder_name}/
├── args.csv              # Experiment arguments
├── {folder_name}.csv     # Training metrics (standardized format)
├── {folder_name}.txt     # Full stdout/stderr log
├── ckpt.pth             # Best model checkpoint
├── accuracy.png         # Training plots
└── {experiment_script}.py # Copy of the experiment script
```

### Folder Naming Convention
```
{dataset}_{precision}_{scaler}_{odeint}_{method}_seed{N}_{timestamp}
```

Example: `mnist_float16_grad_rampde_rk4_seed42_20250121_143022`

## Running Experiments

### Individual Experiments

#### MNIST Neural ODE Classification
```bash
cd experiments/mnist
python ode_mnist.py --precision float16 --odeint rampde --method rk4 --seed 42 --nepochs 10
```

#### STL-10 Image Classification
```bash
cd experiments/stl10
python ode_stl10.py --precision bfloat16 --odeint rampde --method rk4 --seed 42 --nepochs 5
```

#### 2D Continuous Normalizing Flows
```bash
cd experiments/cnf
python cnf.py --data swissroll --precision float32 --odeint rampde --niters 1000
```

#### Large-scale Optimal Transport
```bash
cd experiments/otflowlarge
python otflowlarge.py --data bsds300 --precision float16 --odeint rampde --niters 5000
```

### Batch Experiments with SLURM
Run systematic comparisons across precisions and methods:

```bash
cd experiments/stl10
chmod +x run_stl10.sh
./run_stl10.sh
```

This runs comparisons across:
- **Precisions**: float32, tfloat32, bfloat16, float16
- **Integrators**: torchdiffeq, rampde
- **Scaling strategies**: GradScaler, DynamicScaler, no scaling

## Analysis Tools

### 1. Collect Results
Use `collect_results.py` to gather and analyze experiment data:

```bash
# Collect all results
python collect_results.py --results_dir ./results --output all_experiments.csv

# Filter recent experiments
python collect_results.py --days 7 --output recent_experiments.csv

# Filter by precision and method
python collect_results.py --precision float16 bfloat16 --method rk4 --output mixed_precision.csv

# Generate comparison table
python collect_results.py --output summary.csv --comparison comparison_table.csv
```

**Output includes:**
- Complete experiment metadata and final metrics
- Summary statistics by experiment type, precision, method
- Best performers table
- Completion status tracking

### 2. Visualize Results
Use `visualize_results.py` to create publication-ready plots:

```bash
# Create all visualizations
python visualize_results.py all_experiments.csv --output_dir ./plots

# Create specific plot types
python visualize_results.py summary.csv --plots precision timing --output_dir ./analysis

# Focus on specific experiment
python visualize_results.py summary.csv --experiment mnist --plots convergence
```

**Generated visualizations:**
- **Precision comparison**: Bar charts comparing accuracy/loss across precisions
- **Timing analysis**: Forward vs backward pass time scatter plots
- **Memory heatmaps**: Memory usage across experiment configurations
- **Speedup plots**: Performance gains relative to baseline precision
- **Convergence plots**: Training progress examples
- **Summary dashboard**: Comprehensive overview with key statistics

## Performance Analysis Examples

### Example 1: Mixed Precision Speedup Analysis
```bash
# Run experiments across precisions
cd experiments/mnist
python ode_mnist.py --precision float32 --seed 42 --nepochs 5
python ode_mnist.py --precision float16 --seed 42 --nepochs 5
python ode_mnist.py --precision bfloat16 --seed 42 --nepochs 5

# Collect and analyze results
python ../collect_results.py --experiment mnist --output mnist_precision.csv
python ../visualize_results.py mnist_precision.csv --plots speedup precision
```

### Example 2: Forward/Backward Timing Analysis
```bash
# Collect timing data from multiple experiments
python collect_results.py --days 30 --output timing_analysis.csv

# Create timing visualizations
python visualize_results.py timing_analysis.csv --plots timing --output_dir ./timing_plots
```

### Example 3: Method Comparison
```bash
# Run with different ODE solvers
python experiments/cnf/cnf.py --method rk4 --seed 42
python experiments/cnf/cnf.py --method euler --seed 42

# Compare results
python collect_results.py --experiment cnf --output cnf_methods.csv
python visualize_results.py cnf_methods.csv --experiment cnf --plots precision convergence
```

## Key Features

### 1. **Unified Metrics**
All experiments now report consistent metrics:
- Separate forward/backward timing for bottleneck identification
- Standardized memory reporting in MB
- Consistent loss and accuracy naming

### 2. **Automated Analysis**
- Backward-compatible column mapping for legacy results
- Automatic best performer identification
- Statistical aggregation across multiple runs with same parameters

### 3. **Publication-Ready Plots**
- Professional styling with seaborn
- Comparison plots across methods, precisions, and experiments
- Export-ready figures with high DPI

### 4. **Experiment Tracking**
- Complete argument logging in `args.csv`
- Script versioning (copy of exact script used)
- Completion status tracking
- Emergency stop detection

## Best Practices

### For Running Experiments
1. **Always set seeds** for reproducibility: `--seed 42`
2. **Use descriptive names** in batch scripts
3. **Monitor memory usage** especially with mixed precision
4. **Save intermediate checkpoints** for long runs

### For Analysis
1. **Collect results regularly** to avoid data loss
2. **Filter by date** for recent comparisons
3. **Use comparison tables** for systematic evaluation
4. **Generate plots frequently** to spot trends

### For Reproducibility
1. Check `CLAUDE.md` for environment setup
2. All experiment arguments are logged in `args.csv`
3. Exact script versions are copied to result directories
4. Hardware and software info logged in output files

## Troubleshooting

### Common Issues
- **Memory errors**: Reduce batch size or use gradient checkpointing
- **NaN losses**: Check learning rate, use gradient clipping
- **Slow convergence**: Verify data loading and model architecture
- **Mixed precision issues**: Check GradScaler configuration

### Debug Mode
Run experiments with `--debug` flag for verbose output:
```bash
python ode_mnist.py --debug --nepochs 2
```

### Quick Test Runs
Use minimal epochs/iterations for testing:
```bash
python ode_mnist.py --nepochs 2 --test_freq 10  # Quick MNIST test
python cnf.py --niters 100 --test_freq 10       # Quick CNF test
```

## Contributing

When adding new experiments:
1. Follow the standardized CSV header format
2. Implement separate forward/backward timing
3. Use the `setup_experiment()` function from `common.py`
4. Add proper argument parsing and logging
5. Test with `collect_results.py` and `visualize_results.py`

## Citation

If you use this experiment framework in your research, please cite:

```bibtex
@software{rampde_experiments,
  title={rampde Experiment Management System},
  author={Your Name},
  year={2025},
  url={https://github.com/your-org/rampde}
}
```