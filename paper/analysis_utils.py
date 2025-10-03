#!/usr/bin/env python3
"""
Analysis Utilities for rampde Paper Experiment Results.

This module provides utilities for ANALYZING/PROCESSING experiment results - use this
when processing raw data and generating figures/tables from completed experiments.

Key functionality:
- Parsing experiment directory names to extract configuration info
- Loading experiment results from CSV files
- Finding latest results for dataset-config combinations
- Creating human-readable legend labels for plots
- Standard numerical configuration definitions

For RUNNING experiments, use experiment_runtime.py instead.

Typical usage:
    from analysis_utils import parse_experiment_name, load_experiment_results

    # Parse directory name to get configuration
    config = parse_experiment_name('bsds300_float16_dynamic_rampde_rk4_...')

    # Load experiment results
    df = load_experiment_results('/path/to/experiment/dir')
"""

import os
import glob
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Standard numerical configurations used across experiments
NUMERICAL_CONFIGS = [
    'bfloat16_torchdiffeq',
    'bfloat16_rampde',
    'float16_dynamic_rampde',
    'float16_grad_torchdiffeq',
    'float16_grad_rampde',
    'float16_none_torchdiffeq',
    'float16_none_rampde',
    'float32_torchdiffeq',
    'float32_rampde',
    'tfloat32_torchdiffeq',
    'tfloat32_rampde'
]


def parse_experiment_name(dirname: str, experiment_type: str = 'cnf') -> Dict[str, str]:
    """
    Parse experiment directory name to extract key information.
    
    Expected formats:
    - CNF: {dataset}_{numerical_config}_rk4_{additional_params}_{timestamp}
    - OTFlowLarge: {dataset}_{numerical_config}_rk4_alpha_{alpha}_lr_{lr}_..._{timestamp}
    
    Args:
        dirname: Directory name to parse
        experiment_type: Type of experiment ('cnf', 'otflowlarge', etc.)
        
    Returns:
        Dictionary with parsed information including dataset, numerical_config, etc.
    """
    parts = dirname.split('_')
    
    if len(parts) < 3:
        return {}
    
    # Find the numerical configuration in the parts
    numerical_config = None
    dataset = None
    
    # Look for known numerical configurations
    for i, part in enumerate(parts):
        # Try to match numerical config patterns
        for config in NUMERICAL_CONFIGS:
            config_parts = config.split('_')
            if len(parts) >= i + len(config_parts):
                candidate = '_'.join(parts[i:i+len(config_parts)])
                if candidate == config:
                    numerical_config = config
                    dataset = '_'.join(parts[:i])
                    break
        if numerical_config:
            break
    
    if not numerical_config or not dataset:
        return {}
    
    # Extract precision and solver from numerical config
    config_parts = numerical_config.split('_')
    precision = config_parts[0] if config_parts else 'unknown'
    solver = config_parts[-1] if len(config_parts) > 1 else 'unknown'
    
    # Handle special cases for float16 with scalers
    scaler = None
    if precision == 'float16' and len(config_parts) > 2:
        scaler = config_parts[1]  # 'grad', 'dynamic', or 'none'
    
    # Extract additional parameters
    remaining_parts = parts[len(dataset.split('_')) + len(config_parts):]
    
    result = {
        'dataset': dataset,
        'numerical_config': numerical_config,
        'precision': precision,
        'solver': solver,
        'full_name': dirname
    }
    
    if scaler:
        result['scaler'] = scaler
    
    # Try to extract common parameters based on experiment type
    if experiment_type == 'otflowlarge':
        # Look for alpha values
        for i, part in enumerate(remaining_parts):
            if part == 'alpha' and i + 1 < len(remaining_parts):
                result['alpha'] = remaining_parts[i + 1]
            elif part == 'lr' and i + 1 < len(remaining_parts):
                result['learning_rate'] = remaining_parts[i + 1]
            elif part == 'niters' and i + 1 < len(remaining_parts):
                result['num_iterations'] = remaining_parts[i + 1]
            elif part == 'batch_size' and i + 1 < len(remaining_parts):
                result['batch_size'] = remaining_parts[i + 1]
            elif part.startswith('seed'):
                result['seed'] = part
    else:  # cnf and others
        for i, part in enumerate(remaining_parts):
            if part == 'lr' and i + 1 < len(remaining_parts):
                result['learning_rate'] = remaining_parts[i + 1]
            elif part == 'niters' and i + 1 < len(remaining_parts):
                result['num_iterations'] = remaining_parts[i + 1]
            elif part.startswith('seed'):
                result['seed'] = part
    
    return result


def find_latest_folder(dataset: str, config: str, results_path: str) -> Optional[str]:
    """
    Find the latest result folder for a given dataset-config combination.
    
    Args:
        dataset: Dataset name (e.g., '2spirals', 'bsds300')
        config: Numerical configuration (e.g., 'bfloat16_torchdiffeq')
        results_path: Path to results directory
        
    Returns:
        Path to the latest folder, or None if not found
    """
    pattern = f"{dataset}_{config}_rk4_*"
    folders = glob.glob(os.path.join(results_path, pattern))
    
    if not folders:
        return None
    
    # Sort by folder name (timestamp is in the name) and return the latest
    latest_folder = sorted(folders)[-1]
    return latest_folder


def get_available_datasets(results_path: str, experiment_type: str = 'cnf') -> List[str]:
    """
    Get list of available datasets from the results directory.
    
    Args:
        results_path: Path to results directory
        experiment_type: Type of experiment for parsing
        
    Returns:
        List of unique dataset names
    """
    datasets = set()
    
    for folder in os.listdir(results_path):
        folder_path = os.path.join(results_path, folder)
        if os.path.isdir(folder_path):
            parsed = parse_experiment_name(folder, experiment_type)
            if parsed.get('dataset'):
                datasets.add(parsed['dataset'])
    
    return sorted(list(datasets))


def get_available_configs_for_dataset(dataset: str, results_path: str, experiment_type: str = 'cnf') -> List[str]:
    """
    Get list of available numerical configurations for a specific dataset.
    
    Args:
        dataset: Dataset name
        results_path: Path to results directory
        experiment_type: Type of experiment for parsing
        
    Returns:
        List of available numerical configurations for the dataset
    """
    configs = set()
    
    for folder in os.listdir(results_path):
        folder_path = os.path.join(results_path, folder)
        if os.path.isdir(folder_path):
            parsed = parse_experiment_name(folder, experiment_type)
            if parsed.get('dataset') == dataset and parsed.get('numerical_config'):
                configs.add(parsed['numerical_config'])
    
    return sorted(list(configs))


def load_experiment_results(folder_path: str) -> Optional[pd.DataFrame]:
    """
    Load experimental results CSV from a folder.
    
    Args:
        folder_path: Path to experiment folder
        
    Returns:
        DataFrame with results, or None if not found/loadable
    """
    # Look for CSV file in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    # Filter out args.csv and other non-result files
    result_csv = None
    for csv_file in csv_files:
        filename = os.path.basename(csv_file)
        if filename != "args.csv" and not filename.startswith("summary"):
            result_csv = csv_file
            break
    
    if not result_csv:
        return None
    
    try:
        df = pd.read_csv(result_csv)
        # Add folder information to the dataframe
        folder_name = os.path.basename(folder_path)
        parsed_info = parse_experiment_name(folder_name)
        
        for key, value in parsed_info.items():
            df[f'config_{key}'] = value
            
        return df
    except Exception as e:
        print(f"Error loading {result_csv}: {e}")
        return None


def create_legend_label(config_info: Dict[str, str]) -> str:
    """
    Create a human-readable legend label from configuration information.
    
    Args:
        config_info: Dictionary with configuration information
        
    Returns:
        Formatted legend label
    """
    precision = config_info.get('precision', 'unknown')
    solver = config_info.get('solver', 'unknown')
    
    # Create more readable labels
    precision_map = {
        'float32': 'Float32',
        'tfloat32': 'TensorFloat32',
        'float16': 'Float16',
        'bfloat16': 'BFloat16'
    }
    
    solver_map = {
        'torchdiffeq': 'torchdiffeq',
        'rampde': 'rampde'
    }
    
    precision_label = precision_map.get(precision, precision)
    solver_label = solver_map.get(solver, solver)
    
    # Add scaling information if available
    config = config_info.get('numerical_config', '')
    if 'grad' in config:
        precision_label += ' (grad scaling)'
    elif 'dynamic' in config:
        precision_label += ' (dynamic scaling)'
    elif 'none' in config and 'float16' in config:
        precision_label += ' (no scaling)'
    
    return f"{precision_label} + {solver_label}"