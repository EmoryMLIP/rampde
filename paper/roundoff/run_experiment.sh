#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate torch28

# Add rampde to Python path
export PYTHONPATH=/local/scratch/lruthot/code/rampde:$PYTHONPATH

# Run the CNF roundoff experiment
python roundoff_cnf.py