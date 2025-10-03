# Paper Experiments Refactoring - Product Requirements Document

## Overview

Refactor the experiments and results generation for the paper into a single, well-organized `paper/` directory that will be the sole resource for reproducing all paper results in the public repository.

## Current State

### experiments/ directory
- Individual experiment subdirs: `cnf/`, `stl10/`, `otflowlarge/`, `mnist/`, `roundoff_experiment/`
- Each contains: training code (e.g., `ode_stl10.py`), run scripts (`run_*.sh`), test scripts (`run_*_test.sh`), SLURM job files
- Test scripts already exist with reduced iterations for quick validation

### results/ directory
- `raw_data/`: experiment outputs organized by experiment type (cnf, ode_stl10, otflowlarge, roundoff)
- `code/`: Python scripts including:
  - Shared utilities: `experiment_utils.py`, `requirements.txt`, `run_all.sh`
  - Experiment-specific processing: `generate_cnf_overview_wide.py`, `plot_stl10_convergence.py`, etc.
- `outputs/`: final LaTeX figures and tables, organized by paper labels (fig_cnf_overview, tab_stl10_results, etc.)

## Target Structure

```
paper/
├── README.md                           # Master overview and instructions
├── experiment_utils.py                 # Shared utilities (from results/code/)
├── requirements.txt                    # Python dependencies (from results/code/)
├── cnf/
│   ├── README.md                       # Experiment-specific instructions
│   ├── run_experiment.sh               # From experiments/cnf/run_cnf.sh
│   ├── run_test.sh                     # From experiments/cnf/run_cnf_test.sh
│   ├── cnf.py                          # From experiments/cnf/
│   ├── toy_data.py                     # From experiments/cnf/
│   ├── job_cnf.sbatch                  # From experiments/cnf/
│   ├── generate_cnf_overview_wide.py   # From results/code/
│   ├── extract_cnf_subplots.py         # From results/code/
│   ├── generate_experiment_table.py    # From results/code/ (for Table 2)
│   ├── raw_data/                       # From results/raw_data/cnf/
│   └── outputs/                        # From results/outputs/
│       ├── fig_cnf_overview/
│       └── tab_cnf_results/
├── stl10/
│   ├── README.md
│   ├── run_experiment.sh               # From experiments/stl10/run_stl10.sh
│   ├── run_test.sh                     # From experiments/stl10/run_stl10_test.sh
│   ├── ode_stl10.py                    # From experiments/stl10/
│   ├── job_ode_stl10.sbatch            # From experiments/stl10/
│   ├── plot_stl10_convergence.py       # From results/code/
│   ├── generate_stl10_table.py         # From results/code/
│   ├── raw_data/                       # From results/raw_data/ode_stl10/
│   └── outputs/                        # From results/outputs/
│       ├── fig_stl10_train_loss/
│       └── tab_stl10_results/
├── otflowlarge/
│   ├── README.md
│   ├── run_experiment.sh               # From experiments/otflowlarge/run_otflowlarge.sh
│   ├── run_test.sh                     # From experiments/otflowlarge/run_otflowlarge_test.sh
│   ├── otflowlarge.py                  # From experiments/otflowlarge/
│   ├── Phi.py                          # Supporting files from experiments/otflowlarge/
│   ├── mmd.py                          # Supporting files from experiments/otflowlarge/
│   ├── [other supporting files]        # All .py files from experiments/otflowlarge/
│   ├── job_otflowlarge.sbatch          # From experiments/otflowlarge/
│   ├── generate_otflowlarge_table.py   # From results/code/
│   ├── raw_data/                       # From results/raw_data/otflowlarge/
│   └── outputs/                        # From results/outputs/
│       └── tab_otflowlarge_results/
└── roundoff/
    ├── README.md
    ├── run_experiment.sh               # From experiments/roundoff_experiment/run_roundoff_cnf.sh
    ├── run_test.sh                     # Create minimal test version
    ├── roundoff_cnf.py                 # From experiments/roundoff_experiment/
    ├── roundoff_analyzer.py            # From experiments/roundoff_experiment/
    ├── plot_cnf_roundoff.py            # From results/code/
    ├── [other plotting scripts]        # From experiments/roundoff_experiment/
    ├── job_roundoff_cnf.sbatch         # From experiments/roundoff_experiment/
    ├── raw_data/                       # From results/raw_data/roundoff/
    └── outputs/                        # From results/outputs/
        └── fig_cnf_roundoff/
```

## Requirements

### Functional Requirements

1. **Self-contained experiments**: Each `paper/[experiment]/` directory must contain:
   - All code needed to run the experiment
   - SLURM job submission files
   - Scripts to process results and generate figures/tables
   - Output directories with paper-ready LaTeX files

2. **Test workflows**: Each experiment must have:
   - `run_test.sh`: Quick validation (2-3 epochs/iterations) to verify training direction
   - Clear expected runtime documented in experiment README
   - Validation that outputs match expected behavior

3. **Shared utilities**: Common code at `paper/` root level:
   - `experiment_utils.py`: Shared functions for loading results, parsing directories, creating labels
   - `requirements.txt`: Python dependencies
   - Master `README.md`: Overall structure and instructions

4. **Reproducible outputs**: Each experiment's processing scripts generate:
   - Standalone LaTeX figures (.tex and .pdf files)
   - Tables with proper formatting
   - Files named matching paper labels (e.g., `fig_cnf_overview`, `tab_stl10_results`)

5. **Clear documentation**: README files must explain:
   - How to run experiments (full and test versions)
   - Expected outputs and where they go
   - Runtime estimates
   - Hardware requirements (GPU, SLURM, etc.)

### Non-Functional Requirements

1. **No code changes**: Preserve existing code functionality - only move files
2. **Minimal import changes**: Update import paths only if absolutely necessary (e.g., for `experiment_utils`)
3. **Clean public repo**: Delete `experiments/` and `results/` directories after migration
4. **Version control friendly**: Appropriate .gitignore for large raw_data files

## Implementation Steps

### Phase 1: Setup Structure
1. Create `paper/` directory and subdirectories: `cnf/`, `stl10/`, `otflowlarge/`, `roundoff/`
2. Create empty `raw_data/` and `outputs/` subdirectories within each experiment

### Phase 2: Move Shared Utilities
3. Copy `results/code/experiment_utils.py` to `paper/experiment_utils.py`
4. Copy `results/code/requirements.txt` to `paper/requirements.txt`

### Phase 3: Migrate CNF Experiment
5. Move experiment code:
   - `experiments/cnf/cnf.py` → `paper/cnf/cnf.py`
   - `experiments/cnf/toy_data.py` → `paper/cnf/toy_data.py`
   - `experiments/cnf/run_cnf.sh` → `paper/cnf/run_experiment.sh`
   - `experiments/cnf/run_cnf_test.sh` → `paper/cnf/run_test.sh`
   - `experiments/cnf/job_cnf.sbatch` → `paper/cnf/job_cnf.sbatch`
6. Move processing scripts:
   - `results/code/generate_cnf_overview_wide.py` → `paper/cnf/`
   - `results/code/extract_cnf_subplots.py` → `paper/cnf/`
   - `results/code/generate_experiment_table.py` → `paper/cnf/`
7. Move data and outputs:
   - `results/raw_data/cnf/` → `paper/cnf/raw_data/`
   - `results/outputs/fig_cnf_overview/` → `paper/cnf/outputs/fig_cnf_overview/`
   - `results/outputs/tab_cnf_results/` → `paper/cnf/outputs/tab_cnf_results/`

### Phase 4: Migrate STL10 Experiment
8. Move experiment code:
   - `experiments/stl10/ode_stl10.py` → `paper/stl10/ode_stl10.py`
   - `experiments/stl10/run_stl10.sh` → `paper/stl10/run_experiment.sh`
   - `experiments/stl10/run_stl10_test.sh` → `paper/stl10/run_test.sh`
   - `experiments/stl10/job_ode_stl10.sbatch` → `paper/stl10/job_ode_stl10.sbatch`
9. Move processing scripts:
   - `results/code/plot_stl10_convergence.py` → `paper/stl10/`
   - `results/code/generate_stl10_table.py` → `paper/stl10/`
10. Move data and outputs:
    - `results/raw_data/ode_stl10/` → `paper/stl10/raw_data/`
    - `results/outputs/fig_stl10_train_loss/` → `paper/stl10/outputs/fig_stl10_train_loss/`
    - `results/outputs/tab_stl10_results/` → `paper/stl10/outputs/tab_stl10_results/`

### Phase 5: Migrate OTFlowLarge Experiment
11. Move experiment code:
    - `experiments/otflowlarge/otflowlarge.py` → `paper/otflowlarge/otflowlarge.py`
    - All supporting .py files from `experiments/otflowlarge/` → `paper/otflowlarge/`
    - `experiments/otflowlarge/run_otflowlarge.sh` → `paper/otflowlarge/run_experiment.sh`
    - `experiments/otflowlarge/run_otflowlarge_test.sh` → `paper/otflowlarge/run_test.sh`
    - `experiments/otflowlarge/job_otflowlarge.sbatch` → `paper/otflowlarge/job_otflowlarge.sbatch`
12. Move processing scripts:
    - `results/code/generate_otflowlarge_table.py` → `paper/otflowlarge/`
13. Move data and outputs:
    - `results/raw_data/otflowlarge/` → `paper/otflowlarge/raw_data/`
    - `results/outputs/tab_otflowlarge_results/` → `paper/otflowlarge/outputs/tab_otflowlarge_results/`

### Phase 6: Migrate Roundoff Experiment
14. Move experiment code:
    - `experiments/roundoff_experiment/roundoff_cnf.py` → `paper/roundoff/roundoff_cnf.py`
    - `experiments/roundoff_experiment/roundoff_analyzer.py` → `paper/roundoff/roundoff_analyzer.py`
    - All plotting scripts from `experiments/roundoff_experiment/` → `paper/roundoff/`
    - `experiments/roundoff_experiment/run_roundoff_cnf.sh` → `paper/roundoff/run_experiment.sh`
    - `experiments/roundoff_experiment/job_roundoff_cnf.sbatch` → `paper/roundoff/job_roundoff_cnf.sbatch`
15. Create `paper/roundoff/run_test.sh` (minimal test version)
16. Move processing scripts:
    - `results/code/plot_cnf_roundoff.py` → `paper/roundoff/`
17. Move data and outputs:
    - `results/raw_data/roundoff/` → `paper/roundoff/raw_data/`
    - `results/outputs/fig_cnf_roundoff/` → `paper/roundoff/outputs/fig_cnf_roundoff/`

### Phase 7: Fix Import Paths
18. Update import statements if needed:
    - Scripts importing `experiment_utils` should use: `from ..experiment_utils import ...` or adjust PYTHONPATH
    - Test all processing scripts to ensure they run correctly

### Phase 8: Documentation
19. Create `paper/README.md` with:
    - Overview of structure
    - Instructions for running all experiments
    - Hardware requirements
    - Links to individual experiment READMEs
20. Create README.md for each experiment with:
    - Experiment description
    - How to run full experiment (`./run_experiment.sh`)
    - How to run test (`./run_test.sh`)
    - Expected outputs and locations
    - Runtime estimates

### Phase 9: Cleanup
21. Verify all tests pass: run each `run_test.sh` script
22. Verify processing scripts work: generate figures/tables
23. Delete `experiments/` directory
24. Delete `results/` directory
25. Update root repository documentation if needed

## Success Criteria

1. All experiment code runs successfully from `paper/[experiment]/` directories
2. Test scripts complete in reasonable time (< 30 minutes per experiment)
3. Processing scripts generate paper-ready LaTeX figures and tables
4. All outputs match existing paper figures/tables
5. Documentation is clear and complete
6. Public repository contains only `paper/` directory for experiments
7. No code functionality is broken (only moved)

## Notes

- Raw data directories may be large - consider .gitignore or separate download instructions
- SLURM account (mathg3) is hardcoded in scripts - document this requirement
- Some experiments have specific conda environment requirements - document in READMEs
- Test scripts use different seeds than production to distinguish test runs