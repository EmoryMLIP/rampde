# Clean Reproducible OTFlow Roundoff Experiment Workflow

## 🎯 **Current Clean State**

After cleaning up temporary files, here's the streamlined workflow:

### **Core Files (Use These)**
- **`roundoff_otflow.py`** - Main single-configuration script with argparse
- **`job_roundoff_single.sbatch`** - SLURM template for single configurations  
- **`run_otflow_roundoff_experiment.sh`** - Master script to submit all jobs
- **`roundoff_analyzer.py`** - Core analysis class (unchanged)

### **Plotting & Analysis**
- **`plot_roundoff_comparison.py`** - Generate comparison plots
- **`results/otflow_roundoff_results.csv`** - Main results file (incrementally built)

### **Supporting Files**
- **`roundoff_cnf.py`** - CNF experiment (separate, stable)
- **`job_roundoff_cnf.sbatch`** - CNF SLURM job
- **Various PNG plots** - Generated comparison figures

## 🚀 **How to Use the Clean Workflow**

### **Run Complete Experiment**
```bash
bash run_otflow_roundoff_experiment.sh
```

### **Run Single Configuration**
```bash
python roundoff_otflow.py --precision float16 --odeint_type torchmpnode --scaler_type dynamic --method rk4 --n_timesteps 64 --seed 42
```

### **Generate Plots**
```bash
python plot_roundoff_comparison.py --experiment otflow --precision float16
python plot_roundoff_comparison.py --experiment otflow --precision bfloat16
```

## ✨ **Key Benefits of Clean State**

1. **No merging operations needed** - Direct CSV append
2. **No JIT conflicts** - Process isolation via SLURM
3. **Perfect reproducibility** - Fixed seeding + deterministic sampling
4. **Scalable execution** - Parallel SLURM jobs
5. **Robust error handling** - Failed configs logged, don't affect others

## 🗑️ **Files Removed (No Longer Needed)**

- `merge_bfloat16_data.py` - Merging workaround
- `replace_bfloat16_data.py` - Data replacement workaround  
- `collect_torchdiffeq_bfloat16.py` - Temporary collection script
- `job_collect_bfloat16.sbatch` - Temporary SLURM job
- `run_cnf_only.sh` - Old CNF runner
- `results/otflow_roundoff_results_complete.csv` - Merged data file
- `results/otflow_torchdiffeq_bfloat16_results.csv` - Temporary collection
- `results/cnf_roundoff_results_backup.csv` - Backup file

## 📊 **Current Results Status**

The main results file `results/otflow_roundoff_results.csv` contains:
- Historical results from previous experiments
- New reproducible results from the clean workflow
- All configurations will be added incrementally going forward

The new workflow ensures all future experiments are:
- ✅ **Reproducible** (fixed seeds)
- ✅ **Scalable** (parallel SLURM execution)  
- ✅ **Robust** (no JIT conflicts, proper error handling)
- ✅ **Clean** (no complex merging operations)

---
**The OTFlow roundoff experiment is now in production-ready state! 🎉**