"""
Sampling utilities for A-BB differential debiasing.

This module provides intelligent sampling strategies for managing 
computational costs in dynamic sensitivity estimation.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional
import os


def intelligent_sample(data: Union[pd.DataFrame, list], 
                      target_samples: int = 50,
                      min_threshold_factor: float = 0.25,
                      oversample_threshold_factor: float = 0.25,
                      random_state: int = 42,
                      test_mode_override: Optional[int] = None) -> Union[pd.DataFrame, list]:
    """
    Apply intelligent sampling to limit computational cost while maintaining statistical validity.
    
    Strategy:
    - Large datasets (> target_samples): Random subsample
    - Small datasets (< target_samples * oversample_threshold_factor): Oversample with replacement
    - Medium datasets: Use as-is
    - Test mode: Use override value
    
    Parameters:
    -----------
    data : pd.DataFrame or list
        Input data to sample
    target_samples : int
        Target number of samples for optimal performance (default: 50)
    min_threshold_factor : float
        Factor for minimum threshold below which we don't reduce samples
    oversample_threshold_factor : float  
        Factor below which we oversample small datasets
    random_state : int
        Random seed for reproducible sampling
    test_mode_override : int, optional
        Override sample count for test mode
        
    Returns:
    --------
    pd.DataFrame or list : Sampled data
    """
    # Handle test mode override
    if test_mode_override is not None:
        if isinstance(data, pd.DataFrame):
            return data.head(test_mode_override)
        else:
            return data[:test_mode_override]
    
    # Get data length
    data_len = len(data)
    
    if data_len > target_samples:
        # Large dataset: random subsample
        if isinstance(data, pd.DataFrame):
            sampled_data = data.sample(n=target_samples, random_state=random_state).reset_index(drop=True)
        else:
            np.random.seed(random_state)
            indices = np.random.choice(data_len, size=target_samples, replace=False)
            sampled_data = [data[i] for i in indices]
        
        print(f"📊 Subsampled {target_samples} from {data_len} total samples for dynamic scoring")
        return sampled_data
        
    elif data_len < target_samples * oversample_threshold_factor:
        # Small dataset: oversample with replacement
        oversample_factor = max(2, target_samples // data_len)
        
        if isinstance(data, pd.DataFrame):
            repeated_data = pd.concat([data] * oversample_factor, ignore_index=True)
            sampled_data = repeated_data.sample(n=target_samples, random_state=random_state).reset_index(drop=True)
        else:
            repeated_data = data * oversample_factor
            np.random.seed(random_state)
            indices = np.random.choice(len(repeated_data), size=target_samples, replace=False)
            sampled_data = [repeated_data[i] for i in indices]
            
        print(f"📊 Oversampled {data_len} to {target_samples} samples for dynamic scoring")
        return sampled_data
        
    else:
        # Medium dataset: use as-is
        print(f"📊 Using all {data_len} samples for dynamic scoring")
        return data


def get_sampling_config_from_env() -> dict:
    """
    Get sampling configuration from environment variables.
    
    Returns:
    --------
    dict : Sampling configuration parameters
    """
    config = {
        'target_samples': int(os.getenv('DYNAMIC_SAMPLES_TARGET', 50)),
        'min_threshold_factor': float(os.getenv('DYNAMIC_SAMPLES_MIN_FACTOR', 0.25)),
        'oversample_threshold_factor': float(os.getenv('DYNAMIC_SAMPLES_OVERSAMPLE_FACTOR', 0.25)),
        'random_state': int(os.getenv('DYNAMIC_SAMPLES_SEED', 42))
    }
    
    # Test mode override
    test_mode = os.getenv('TEST_MODE') == 'true'
    if test_mode:
        config['test_mode_override'] = int(os.getenv('TEST_SAMPLES', 5))
    else:
        config['test_mode_override'] = None
    
    return config


def apply_intelligent_sampling(data: Union[pd.DataFrame, list], **kwargs) -> Union[pd.DataFrame, list]:
    """
    Apply intelligent sampling using environment-based configuration.
    
    Parameters:
    -----------
    data : pd.DataFrame or list
        Input data to sample
    **kwargs : dict
        Override parameters for sampling configuration
        
    Returns:
    --------
    pd.DataFrame or list : Sampled data
    """
    config = get_sampling_config_from_env()
    config.update(kwargs)  # Allow parameter overrides
    
    return intelligent_sample(data, **config)