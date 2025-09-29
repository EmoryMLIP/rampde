# Test Run System for rampde Experiments

This document describes the test run system designed to quickly validate experiment functionality and generate initial performance data.

## Overview

The test run system provides shortened versions of all rampde experiments to accomplish three key goals:

1. **Functionality Verification**: Ensure all experiments run without errors
2. **Initial Performance Data**: Generate timing and memory usage data for analysis
3. **Pipeline Testing**: Validate evaluation and visualization scripts work correctly

## Quick Start

### Run All Test Experiments

```bash
cd experiments
chmod +x run_test_experiments.sh
./run_test_experiments.sh
```

This master script will:
- Run all individual test experiments
- Track timing for each experiment type
- Provide progress monitoring commands
- Create a unified test results directory

### Run Individual Test Experiments

Each experiment type has its own test script:

```bash
# MNIST Neural ODE Classification (fastest)
cd mnist && chmod +x run_mnist_test.sh && ./run_mnist_test.sh

# Continuous Normalizing Flows (2D toy data)
cd cnf && chmod +x run_cnf_test.sh && ./run_cnf_test.sh

# STL10 Image Classification 
cd stl10 && chmod +x run_stl10_test.sh && ./run_stl10_test.sh

# Large-scale Optimal Transport (most complex)
cd otflowlarge && chmod +x run_otflowlarge_test.sh && ./run_otflowlarge_test.sh
```

## Test Configuration Details

### Reduced Training Parameters

| Experiment | Production | Test | Validation Points |
|------------|------------|------|------------------|
| **MNIST** | 160 epochs | 3 epochs | 3 (every epoch) |
| **CNF** | 2000 iterations | 120 iterations | 3 (every 40 iters) |
| **STL10** | 160 epochs | 3 epochs | 3 (every epoch) |
| **OTFlowLarge** | 8000+ iterations | 200 iterations | 3 (every 60 iters) |

### Representative Configurations

Each test focuses on key combinations for speed:

- **Precisions**: float32, float16 (most important comparison)
- **Backends**: torchdiffeq, rampde  
- **Scaling**: Representative scaling strategies for each backend
- **Datasets**: Single representative dataset per experiment type

### Expected Runtimes

- **MNIST**: ~5-10 minutes per job
- **CNF**: ~3-5 minutes per job  
- **STL10**: ~10-15 minutes per job (includes dataset download)
- **OTFlowLarge**: ~15-25 minutes per job (most complex)
- **Total**: ~30-60 minutes for all test experiments

## Monitoring Progress

### SLURM Queue Status
```bash
# Overall queue status
watch -n 30 'squeue -u $USER'

# Individual experiment types
squeue -u $USER | grep mnist
squeue -u $USER | grep cnf
squeue -u $USER | grep stl10
squeue -u $USER | grep otflow
```

### Job Output Logs
```bash
# Check recent output
tail -f mnist/slurm_logs/ode_mnist_*.out
tail -f cnf/slurm_logs/cnf_*.out
tail -f stl10/slurm_logs/ode_stl10_*.out
tail -f otflowlarge/slurm_logs/otflow_*.out

# List all log files
ls -la */slurm_logs/*.out
```

## Results Analysis

### Collect Test Results

Once test experiments complete, analyze results:

```bash
cd experiments

# Collect all test results
python collect_results.py --results_dir results_test --output test_results.csv

# Generate performance visualizations
python visualize_results.py test_results.csv --output_dir test_plots
```

### Expected Output

The test runs will generate:

1. **CSV Files**: Standardized training metrics with timing data
2. **Performance Table**: Forward/backward pass timings across configurations
3. **Memory Usage**: Peak GPU memory for each experiment type
4. **Accuracy Baselines**: Quick validation of model convergence

### Key Metrics to Examine

- **Forward/Backward Timing**: Identify performance bottlenecks
- **Memory Usage**: Validate mixed precision memory efficiency
- **Convergence**: Ensure models train properly in shortened runs
- **Scaling Effects**: Compare gradient scaling strategies

## Test vs Production Differences

### Modified Parameters

1. **Training Length**: Drastically reduced iterations/epochs
2. **Validation Frequency**: More frequent testing for data collection
3. **Seeds**: Different seeds to distinguish test from production runs
4. **Early Stopping**: Disabled in some experiments for consistent runtime
5. **Dataset Selection**: Single representative dataset where applicable

### Preserved Parameters

- **Model Architecture**: Identical to production
- **Precision Settings**: Exact same mixed precision configurations  
- **ODE Solvers**: Same integration methods and tolerances
- **Scaling Strategies**: Identical gradient scaling approaches
- **Batch Sizes**: Generally preserved for realistic memory usage

## Troubleshooting

### Common Issues

1. **Dataset Download**: STL10 first run may be slow due to dataset download
2. **Memory Errors**: Reduce batch size in individual test scripts if needed
3. **SLURM Issues**: Check cluster status and account permissions
4. **Missing Dependencies**: Ensure torch26 environment is activated

### Debug Mode

Run individual experiments with debug flags:

```bash
# Example: debug MNIST experiment
cd mnist
python ode_mnist.py --debug --nepochs 1 --test_freq 1
```

### Quick Verification

Test core functionality without SLURM:

```bash
# Quick local test (no SLURM)
cd mnist
python ode_mnist.py --nepochs 1 --batch_size 32 --test_batch_size 100
```

## Integration with Production

### Results Compatibility

Test results use the same CSV format as production experiments, enabling:

- **Direct Comparison**: Compare test vs full runs
- **Pipeline Validation**: Verify analysis scripts work
- **Performance Baselines**: Initial timing data for planning

### Workflow Integration  

1. **Development**: Use test runs to validate new features
2. **Debugging**: Quick iteration on experiment configurations
3. **Performance Analysis**: Generate initial timing data
4. **Pipeline Testing**: Validate analysis scripts before long runs

## File Structure

```
experiments/
├── run_test_experiments.sh           # Master test script
├── README_TEST_RUNS.md              # This documentation
├── results_test/                    # Test results directory
├── mnist/
│   └── run_mnist_test.sh
├── cnf/
│   └── run_cnf_test.sh
├── stl10/
│   └── run_stl10_test.sh
└── otflowlarge/
    └── run_otflowlarge_test.sh
```

## Next Steps

After running test experiments:

1. **Validate Results**: Check that all experiments completed successfully
2. **Analyze Performance**: Review timing and memory usage patterns
3. **Test Analysis Pipeline**: Run collect_results.py and visualize_results.py
4. **Plan Production Runs**: Use test data to optimize production configurations

The test system provides a reliable foundation for validating experiment functionality and generating initial performance insights before committing to long-running production experiments.