#!/bin/bash

# Activate conda environment
source ~/.bashrc
conda activate torch26

# Add torchmpnode to Python path
export PYTHONPATH=/local/scratch/lruthot/code/torchmpnode:$PYTHONPATH

# Run the CNF roundoff experiment
python roundoff_cnf.py